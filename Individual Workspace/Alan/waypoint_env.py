"""
waypoint_env.py
---------------
Rules-aware 2D point-mass simulator for pylon-racing waypoint evaluation.

The drone flies at constant speed with rate-limited heading changes,
chasing the next waypoint in a user-supplied list. On top of that, the
env enforces the full pylon-racing ruleset:

    1. Pylon clearance — the drone must stay at least
       (pylon_radius + clearance_margin) metres from every pylon at all
       times. Violation is a hard failure.
    2. Gate crossings — each "gate" is the line segment between two
       consecutive pylons. The drone must physically cross every gate
       once, in order (gate 0 first, then 1, 2, ...), and in the
       correct direction (IN / OUT / IN / ...).
    3. Completion — `completed` is true iff all gates were crossed
       legally AND the drone crosses gate 0 one final time (IN) to
       close the lap. Waypoint capture is only a flight-control
       mechanism, not the success criterion.

The drone starts at `waypoints[-1]` (which, for the current heuristic,
is a bypass waypoint outside the course near gate 0) and targets
`waypoints[0]` first. This guarantees the first step segment enters
gate 0 from outside, producing a clean IN crossing.
"""

import numpy as np

from heuristic import default_directions, PYLON_RADIUS


class PointMassEnv2D:
    def __init__(
        self,
        pylons_xy,
        bounds,
        directions=None,
        speed=7.0,
        max_turn_rate=2.0,
        dt=0.05,
        max_steps=2000,
        capture_radius=0.8,
        clearance_margin=1.0,
        pylon_radius=PYLON_RADIUS,
    ):
        self.pylons_xy = np.asarray(pylons_xy, dtype=np.float64)
        self.bounds = bounds
        self.speed = speed
        self.max_turn_rate = max_turn_rate
        self.dt = dt
        self.max_steps = max_steps
        self.capture_radius = capture_radius
        self.clearance_margin = clearance_margin
        self.pylon_radius = pylon_radius
        self.clearance_dist = pylon_radius + clearance_margin

        n = len(self.pylons_xy)
        if directions is None:
            directions = default_directions(n)
        assert len(directions) == n, "one direction per gate required"
        self.directions = list(directions)

        self.n_gates = n
        self.gates = self._build_gates()

    # ------------------------------------------------------------------
    # Gate geometry (pre-computed once per env)
    # ------------------------------------------------------------------

    def _build_gates(self):
        P = self.pylons_xy
        n = len(P)
        centroid = P.mean(axis=0)
        gates = []
        for k in range(n):
            a = P[k]
            b = P[(k + 1) % n]
            mid = 0.5 * (a + b)
            along = b - a
            length = np.linalg.norm(along)
            along_u = along / (length + 1e-12)
            normal = np.array([-along_u[1], along_u[0]])
            if np.dot(normal, centroid - mid) < 0.0:
                normal = -normal
            gates.append({
                "a": a,
                "b": b,
                "mid": mid,
                "normal": normal,
                "required_dir": self.directions[k],
            })
        return gates

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def simulate(self, waypoints):
        """
        Fly the point-mass through `waypoints` and enforce the rules.

        Parameters
        ----------
        waypoints : list of (x, y)
            Target waypoint list. The drone starts a few metres outside
            gate 0 (computed from the gate geometry, not the waypoints)
            and targets `waypoints[0]` first.

        Returns
        -------
        dict with keys:
            total_time       : float  — sim seconds until termination
            total_distance   : float  — path length in metres
            smoothness       : float  — sum of |heading change|
            completed        : bool   — True iff all gates crossed legally
                                        and gate 0 re-crossed to close lap
            trajectory       : list of (x, y)
            gates_crossed    : int    — number of legal crossings recorded
            gate_log         : list of (step, gate_idx, direction)
            pylon_collision  : bool
            collision_pylon  : int or None
            failure_reason   : str or None  — None iff completed
        """
        if len(waypoints) < 2:
            return self._empty_result()

        wps = [np.asarray(w, dtype=np.float64) for w in waypoints]

        # Start outside the course on the far side of gate 0 from the
        # centroid, 3 m out along the gate's outward normal. This
        # guarantees the first step segment enters gate 0 IN cleanly
        # without starting inside any pylon's exclusion zone.
        gate0 = self.gates[0]
        start_offset = 3.0
        start_pos = gate0["mid"] - gate0["normal"] * start_offset
        x, y = float(start_pos[0]), float(start_pos[1])

        # Append return waypoints: first the start position (outside
        # gate 0) to guide the drone back, then wps[0] (inside gate 0)
        # so the drone physically crosses the gate line to close the lap.
        wps.append(start_pos.copy())
        wps.append(wps[0].copy())

        target = wps[0]
        heading = np.arctan2(target[1] - y, target[0] - x)
        wp_idx = 0  # next target index into `wps`

        trajectory = [(x, y)]
        total_dist = 0.0
        smoothness = 0.0

        crossed = set()
        next_expected = 0
        gate_log = []
        awaiting_final_gate0 = False

        def finish(step, **overrides):
            time_elapsed = (step + 1) * self.dt
            result = {
                "total_time": time_elapsed,
                "total_distance": total_dist,
                "smoothness": smoothness,
                "completed": False,
                "trajectory": trajectory,
                "gates_crossed": len(crossed),
                "gate_log": list(gate_log),
                "pylon_collision": False,
                "collision_pylon": None,
                "failure_reason": None,
            }
            result.update(overrides)
            return result

        for step in range(self.max_steps):
            # Desired heading toward current target
            target = wps[wp_idx]
            desired = np.arctan2(target[1] - y, target[0] - x)

            err = self._angle_wrap(desired - heading)
            max_delta = self.max_turn_rate * self.dt
            delta = float(np.clip(err, -max_delta, max_delta))
            heading += delta
            smoothness += abs(delta)

            # Advance position
            px, py = x, y
            vx = self.speed * np.cos(heading)
            vy = self.speed * np.sin(heading)
            x += vx * self.dt
            y += vy * self.dt
            total_dist += self.speed * self.dt
            trajectory.append((x, y))

            # --- 1. Pylon collision ---
            for i, p in enumerate(self.pylons_xy):
                if np.hypot(x - p[0], y - p[1]) < self.clearance_dist:
                    return finish(
                        step,
                        pylon_collision=True,
                        collision_pylon=i,
                        failure_reason="pylon_collision",
                    )

            # --- 2. Gate crossings ---
            if not awaiting_final_gate0:
                # Normal sequence: cross gates 0..N-1 in order.
                for k in range(self.n_gates):
                    if k in crossed:
                        continue
                    gate = self.gates[k]
                    mid = gate["mid"]
                    n = gate["normal"]
                    d_prev = (px - mid[0]) * n[0] + (py - mid[1]) * n[1]
                    d_curr = (x - mid[0]) * n[0] + (y - mid[1]) * n[1]
                    if d_prev * d_curr >= 0.0:
                        continue

                    if not _segment_segment_intersects(
                        px, py, x, y, gate["a"][0], gate["a"][1],
                        gate["b"][0], gate["b"][1],
                    ):
                        continue

                    observed = +1 if (d_prev < 0 and d_curr > 0) else -1

                    if k != next_expected:
                        return finish(step, failure_reason="wrong_order")
                    if observed != gate["required_dir"]:
                        return finish(step, failure_reason="wrong_direction")

                    crossed.add(k)
                    gate_log.append((step, k, observed))
                    next_expected += 1

                    if next_expected == self.n_gates:
                        awaiting_final_gate0 = True
            else:
                # All gates crossed — check for gate 0 re-crossing (IN)
                # to close the lap.
                gate = self.gates[0]
                mid = gate["mid"]
                n = gate["normal"]
                d_prev = (px - mid[0]) * n[0] + (py - mid[1]) * n[1]
                d_curr = (x - mid[0]) * n[0] + (y - mid[1]) * n[1]
                if d_prev * d_curr < 0.0 and _segment_segment_intersects(
                    px, py, x, y, gate["a"][0], gate["a"][1],
                    gate["b"][0], gate["b"][1],
                ):
                    observed = +1 if (d_prev < 0 and d_curr > 0) else -1
                    if observed == gate["required_dir"]:
                        gate_log.append((step, 0, observed))
                        return finish(step, completed=True,
                                      gates_crossed=self.n_gates + 1)

            # --- 3. Waypoint capture (flight control only) ---
            dist_to_wp = np.hypot(target[0] - x, target[1] - y)
            if dist_to_wp < self.capture_radius:
                wp_idx += 1
                if wp_idx >= len(wps):
                    # Visited every waypoint but didn't close the lap
                    return finish(step, failure_reason="incomplete")

            # --- 4. Bounds check ---
            b = self.bounds
            if (x < b["min_x"] or x > b["max_x"] or
                    y < b["min_y"] or y > b["max_y"]):
                return finish(step, failure_reason="out_of_bounds")

        # Ran out of max_steps
        return finish(self.max_steps - 1, failure_reason="timeout")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _empty_result(self):
        return {
            "total_time": 0.0,
            "total_distance": 0.0,
            "smoothness": 0.0,
            "completed": False,
            "trajectory": [],
            "gates_crossed": 0,
            "gate_log": [],
            "pylon_collision": False,
            "collision_pylon": None,
            "failure_reason": "empty_waypoints",
        }


def _segment_segment_intersects(ax, ay, bx, by, cx, cy, dx, dy):
    """
    Return True iff segment AB properly intersects segment CD.

    Uses the parametric form:
        A + u * (B - A)  =  C + s * (D - C)
    and checks 0 <= u <= 1 and 0 <= s <= 1.
    """
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return False  # parallel / colinear
    u = ((cx - ax) * sy - (cy - ay) * sx) / denom
    s = ((cx - ax) * ry - (cy - ay) * rx) / denom
    return 0.0 <= u <= 1.0 and 0.0 <= s <= 1.0
