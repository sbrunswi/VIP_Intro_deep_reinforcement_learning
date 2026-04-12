"""
heuristic.py
------------
Slalom waypoint heuristic. Two waypoints per gate: one `mid` at the
gate midpoint and one `bypass` near the shared pylon to ensure the
drone clears it when transitioning between gates.

Competition rules:
    - A "gate" is the line segment between two consecutive pylons.
    - The drone must CROSS each gate line on every lap.
    - Crossings alternate: gate 0 IN (outside -> inside), gate 1 OUT
      (inside -> outside), gate 2 IN, gate 3 OUT, ...
    - Pylons have radius R and need a safety clearance.

For each gate k we place:
    mid_k     — at the gate midpoint, pulled slightly toward the
                centroid by `mid_inset`.
    bypass_k  — next to P_{k+1} (the shared pylon with gate k+1),
                placed at distance R + clearance_margin from the
                pylon centre:
                  * After OUT gates: on the OUTSIDE (away from centroid)
                  * After IN gates:  on the INSIDE (toward centroid)

Layout for a 4-pylon course:

    [ mid_0, byp_P1_in, mid_1, byp_P2_out, mid_2, byp_P3_in, mid_3, byp_P0_out ]

Tunables
--------
clearance_margin : metres of free air beyond the pylon radius R when
                   placing the bypass waypoint.
mid_inset        : optional metres to pull the mid waypoint off the
                   literal gate midpoint toward the course centroid
                   (0.0 = exactly on the gate line). Non-zero values
                   can smooth the turn at mid_k.
"""

import numpy as np


PYLON_RADIUS = 0.25  # metres — must match the course definition


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

def default_directions(n_gates):
    """Alternating IN/OUT starting with IN on gate 0."""
    return [(+1 if i % 2 == 0 else -1) for i in range(n_gates)]


# ---------------------------------------------------------------------------
# Main heuristic
# ---------------------------------------------------------------------------

def heuristic_waypoints(
    pylons_xy,
    directions=None,
    clearance_margin=3.5,
    mid_inset=0.3,
    pylon_radius=PYLON_RADIUS,
    bounds=None,
    offset=None,  # legacy alias, ignored
):
    """
    Build a slalom waypoint list.

    Parameters
    ----------
    pylons_xy : array-like, shape (N, 2)
    directions : list of {+1, -1}, optional
        +1 = IN (outside -> inside), -1 = OUT. Defaults to [IN, OUT, ...].
    clearance_margin : float
        Extra clearance (m) beyond the pylon radius for bypass waypoints.
    mid_inset : float
        Metres to pull mid waypoint inward from the gate midpoint. 0.0
        leaves it exactly on the gate line.
    pylon_radius : float
    bounds : dict with min_x/max_x/min_y/max_y, optional
        Course bounds. Bypass waypoints are clamped inside with margin.
    offset : ignored legacy parameter.

    Returns
    -------
    waypoints : list of [x, y]
        Variable length: one mid per gate + bypass where needed.
    """
    wps, _ = _build_waypoints_and_layout(
        pylons_xy, directions, clearance_margin, mid_inset, pylon_radius,
        bounds)
    return wps


def gate_map(pylons_xy, directions=None, clearance_margin=3.5,
             mid_inset=0.3, pylon_radius=PYLON_RADIUS, bounds=None):
    """
    Return a parallel list of (gate_index, kind) tuples matching the
    output of `heuristic_waypoints(pylons_xy, directions)`.

    kind is "mid" or "bypass".  Used by the Q-learners to map each
    waypoint back to the gate whose approach frame it lives in.
    """
    _, layout = _build_waypoints_and_layout(
        pylons_xy, directions, clearance_margin, mid_inset, pylon_radius,
        bounds)
    return layout


def _build_waypoints_and_layout(pylons_xy, directions, clearance_margin,
                                 mid_inset, pylon_radius, bounds=None):
    """Core logic shared by heuristic_waypoints and gate_map."""
    P = np.array(pylons_xy, dtype=np.float64)
    n = len(P)
    if directions is None:
        directions = default_directions(n)
    assert len(directions) == n, "directions must have one entry per gate"

    centroid = P.mean(axis=0)
    clearance_dist = pylon_radius + clearance_margin

    # --- Step 1: compute all gate midpoints ---
    mids = []
    for k in range(n):
        mid = 0.5 * (P[k] + P[(k + 1) % n])
        if mid_inset != 0.0:
            inward = centroid - mid
            inward /= (np.linalg.norm(inward) + 1e-9)
            mid = mid + mid_inset * inward
        mids.append(mid)

    # --- Step 2: insert bypass waypoints ---
    # OUT gates ALWAYS need a bypass on the outside of the shared
    # pylon so the drone swings around and approaches the next gate
    # from the correct (outside) direction.
    # IN gates only need a bypass when a pylon is too close to the
    # mid→mid segment (clearance check).
    turn_buffer = 2.5  # metres — for drone turn dynamics
    safe_dist = clearance_dist + turn_buffer

    waypoints = []
    layout = []
    for k in range(n):
        mk = mids[k]
        mk1 = mids[(k + 1) % n]
        waypoints.append(mk.tolist())
        layout.append((k, "mid"))

        need_bypass = False
        bypass_pylon_idx = (k + 1) % n  # shared pylon by default

        if directions[k] < 0:
            # OUT gate: always need bypass on outside
            need_bypass = True
        else:
            # IN gate: check if any pylon is too close to mk→mk1
            seg = mk1 - mk
            seg_len = np.linalg.norm(seg)
            if seg_len > 1e-9:
                seg_u = seg / seg_len
                worst_deficit = 0.0
                for j in range(n):
                    pj = P[j]
                    t = np.dot(pj - mk, seg_u)
                    t_clip = np.clip(t, 0.0, seg_len)
                    closest = mk + t_clip * seg_u
                    dist = np.linalg.norm(pj - closest)
                    deficit = safe_dist - dist
                    if deficit > worst_deficit:
                        worst_deficit = deficit
                        bypass_pylon_idx = j
                if worst_deficit > 0:
                    need_bypass = True

        if need_bypass:
            pj = P[bypass_pylon_idx]
            if directions[k] < 0:   # OUT → bypass outside
                radial = pj - centroid
            else:                    # IN  → bypass inside
                radial = centroid - pj
            r_norm = np.linalg.norm(radial)
            if r_norm < 1e-9:
                radial = np.array([1.0, 0.0])
            else:
                radial = radial / r_norm
            bypass = pj + clearance_dist * radial
            # Clamp to bounds (with margin for drone overshoot)
            if bounds is not None:
                margin = 1.0
                bypass[0] = np.clip(bypass[0],
                                    bounds["min_x"] + margin,
                                    bounds["max_x"] - margin)
                bypass[1] = np.clip(bypass[1],
                                    bounds["min_y"] + margin,
                                    bounds["max_y"] - margin)
            waypoints.append(bypass.tolist())
            layout.append((k, "bypass"))

    return waypoints, layout


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
#
# With variable-length layout, callers should use `gate_map` to get the
# per-waypoint (gate_index, kind) mapping instead of computing from
# position alone.


def intermediate_indices(n_waypoints):
    """Every waypoint in the list is adjustable."""
    return list(range(n_waypoints))


def bypass_pylon_index(gate_k, n_pylons):
    """Pylon that a bypass after gate k clears (shared pylon with gate k+1)."""
    return (gate_k + 1) % n_pylons
