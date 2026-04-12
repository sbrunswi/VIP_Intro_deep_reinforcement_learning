"""
planner.py
----------
2D Pre-Path Q-learning planner with a unified small-step action space.

State: integer offsets (feet) of all intermediate waypoints from their
heuristic positions, clamped to +/-5 ft per axis.

Action: an 8-dimensional {-1, 0, +1} step applied simultaneously to all
intermediate waypoints (one integer per axis per waypoint). This gives
3^8 = 6561 discrete actions. Large displacements accumulate over many
steps of the episode.

Reward combines path length, turn-angle penalties, gate-crossing bonuses,
stall-speed penalties, and a terminal bonus/penalty for completing the loop.
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Physical / kinematic constants (from fixedwing_4ch.py and task brief)
# ---------------------------------------------------------------------------

THRUST   = 0.56        # N
MASS     = 0.057       # kg
A_MAX    = THRUST / MASS   # ~9.82 m/s^2
V_CRUISE = 5.0         # m/s simplified cruise speed
V_STALL  = 2.0         # m/s stall speed
MAX_TURN_DEG = 60.0    # max yaw change per vertex
FT_TO_M  = 0.3048      # grid spacing


# ---------------------------------------------------------------------------
# Heuristic waypoint initialization
# ---------------------------------------------------------------------------

def heuristic_waypoints(pylons_xy, offset=2.0):
    """
    Place one intermediate waypoint between each consecutive pylon pair,
    offset toward the course centroid for smoother transitions.

    Returns a list of (x, y) in metres, alternating
    [P0, W0, P1, W1, ..., Pn-1, Wn-1].
    """
    pylons = np.array(pylons_xy, dtype=np.float64)
    center = pylons.mean(axis=0)
    n = len(pylons)
    waypoints = []
    for i in range(n):
        p_curr = pylons[i]
        p_next = pylons[(i + 1) % n]
        mid = (p_curr + p_next) / 2.0
        to_center = center - mid
        norm = np.linalg.norm(to_center)
        if norm > 1e-6:
            to_center /= norm
        wp = mid + offset * to_center
        waypoints.append(p_curr.tolist())
        waypoints.append(wp.tolist())
    return waypoints


# ---------------------------------------------------------------------------
# Action encoding: 3^8 discrete actions as base-3 indices
# ---------------------------------------------------------------------------

N_AXES = 8          # 4 waypoints * 2 axes
N_ACTIONS = 3 ** N_AXES   # = 6561


def action_idx_to_vec(idx):
    """Decode an integer action index 0..N_ACTIONS-1 into an (n_wp, 2) array
    of integers in {-1, 0, +1}. Uses base-3 positional encoding."""
    vec = np.zeros(N_AXES, dtype=np.int8)
    for k in range(N_AXES):
        vec[k] = (idx // (3 ** k)) % 3 - 1
    return vec.reshape(-1, 2)


def vec_to_action_idx(vec):
    """Encode an (n_wp, 2) array of {-1, 0, +1} integers into a single int."""
    flat = np.asarray(vec, dtype=np.int8).flatten() + 1  # shift to {0,1,2}
    idx = 0
    for k, v in enumerate(flat):
        idx += int(v) * (3 ** k)
    return idx


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _turn_angles_deg(path_xy):
    """Heading change (deg) at each interior vertex of a closed path.

    path_xy: (M, 2) array where path_xy[0] == path_xy[-1] is NOT required --
             the path is treated as a closed loop.
    Returns an array of length M (one angle per vertex).
    """
    pts = np.asarray(path_xy, dtype=np.float64)
    n = len(pts)
    angles = np.zeros(n, dtype=np.float64)
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % n]
        v_in  = curr_pt - prev_pt
        v_out = next_pt - curr_pt
        na = np.linalg.norm(v_in)
        nb = np.linalg.norm(v_out)
        if na < 1e-9 or nb < 1e-9:
            angles[i] = 0.0
            continue
        cos_a = np.clip(np.dot(v_in, v_out) / (na * nb), -1.0, 1.0)
        # Turn amount: 0 = straight, pi = full reversal
        angles[i] = np.degrees(np.arccos(cos_a))
    return angles


def _segments_cross(a1, a2, b1, b2):
    """Return True if segment a1-a2 intersects segment b1-b2 (2D)."""
    def _ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return (_ccw(a1, b1, b2) != _ccw(a2, b1, b2) and
            _ccw(a1, a2, b1) != _ccw(a1, a2, b2))


def _count_ordered_gate_crossings(trajectory, pylons_xy, gates):
    """
    Walk the trajectory and count how many gates are crossed in sequence.

    Gates is a list of (i_pylon_a, i_pylon_b) index tuples.  Expected order
    matches the list order. The counter advances only when the next-expected
    gate is crossed.
    Returns (ordered_crossings, all_crossings_any_gate).
    """
    if len(trajectory) < 2:
        return 0, 0
    traj = np.asarray(trajectory)
    n_gates = len(gates)
    next_expected = 0
    ordered = 0
    any_count = 0
    for i in range(len(traj) - 1):
        a1 = traj[i]
        a2 = traj[i + 1]
        for gi, (pa, pb) in enumerate(gates):
            b1 = pylons_xy[pa]
            b2 = pylons_xy[pb]
            if _segments_cross(a1, a2, b1, b2):
                any_count += 1
                if gi == next_expected:
                    ordered += 1
                    next_expected = (next_expected + 1) % n_gates
    return ordered, any_count


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(waypoints, sim_result, pylons_xy, gates):
    """
    Scalar reward for a candidate waypoint set.

    Combines:
      - path length (minimize)
      - turn angle penalty at each vertex (quadratic above 60 deg)
      - gate-crossing bonus in sequential order
      - terminal bonus/penalty for completion
      - stall proxy: any vertex turn > 60 deg triggers a stall penalty
    """
    r = 0.0

    # Path length (minimize)
    r -= 0.05 * sim_result["total_distance"]

    # Turn angle penalty over the closed waypoint loop
    wps_arr = np.asarray(waypoints, dtype=np.float64)
    turn_deg = _turn_angles_deg(wps_arr)
    turn_penalty = 0.0
    stall_penalty = 0.0
    for theta in turn_deg:
        if theta > MAX_TURN_DEG:
            turn_penalty += 0.1 * (theta - MAX_TURN_DEG) ** 2
            stall_penalty += 30.0
        else:
            turn_penalty += 0.02 * theta
    r -= turn_penalty
    r -= stall_penalty

    # Gate crossings in correct order
    ordered, _ = _count_ordered_gate_crossings(
        sim_result["trajectory"], pylons_xy, gates)
    r += 20.0 * ordered

    # Terminal bonus/penalty
    if sim_result["completed"]:
        r += 200.0
    else:
        r -= 100.0

    return r


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class PrePathQLearner:
    def __init__(self, n_waypoints=4, offset_max=5,
                 alpha=0.1, gamma=0.95,
                 epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.995,
                 max_steps_per_episode=60):
        self.n_waypoints = n_waypoints
        self.offset_max = offset_max
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_steps = max_steps_per_episode
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))

    # -- state <-> key --

    @staticmethod
    def _offsets_to_state(offsets):
        """Convert an (n_wp, 2) int array to a hashable tuple of tuples."""
        return tuple(tuple(int(v) for v in row) for row in offsets)

    # -- action selection --

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(N_ACTIONS))
        q = self.Q[state]
        best = np.flatnonzero(q == np.max(q))
        return int(np.random.choice(best))

    # -- training --

    def run_optimization(self, heuristic_wps, pylons_xy, env, gates,
                         n_episodes=500, verbose=True):
        """
        Optimise intermediate waypoint positions via Q-learning with a
        unified small-step action space.

        Parameters
        ----------
        heuristic_wps : list of (x, y)
            Full waypoint list including pylons (length 2*n_pylons).
        pylons_xy : array-like (n_pylons, 2)
        env : PointMassEnv2D
        gates : list of (i_pylon_a, i_pylon_b) tuples
        n_episodes : int

        Returns
        -------
        best_wps : list of (x, y)
        rewards_history : list of float
        """
        base = [list(w) for w in heuristic_wps]
        intermediate_idxs = list(range(1, len(base), 2))
        assert len(intermediate_idxs) == self.n_waypoints, (
            f"Expected {self.n_waypoints} intermediates, "
            f"got {len(intermediate_idxs)}")

        best_reward = -np.inf
        best_wps = [list(w) for w in base]
        rewards_history = []

        for ep in range(n_episodes):
            # Reset offsets to zero (heuristic) at start of each episode
            offsets = np.zeros((self.n_waypoints, 2), dtype=np.int16)
            current_wps = [list(w) for w in base]

            # Initial reward/state baseline so the first TD target is valid
            sim0 = env.simulate(current_wps)
            ep_best_reward = compute_reward(current_wps, sim0, pylons_xy, gates)
            ep_best_wps = [list(w) for w in current_wps]

            for step in range(self.max_steps):
                state = self._offsets_to_state(offsets)
                action_idx = self.get_action(state)
                action_vec = action_idx_to_vec(action_idx)

                # Apply with clamping
                new_offsets = np.clip(
                    offsets + action_vec,
                    -self.offset_max, self.offset_max,
                ).astype(np.int16)

                # Build absolute waypoints in metres
                next_wps = [list(w) for w in base]
                for i, wp_idx in enumerate(intermediate_idxs):
                    dx_m = new_offsets[i][0] * FT_TO_M
                    dy_m = new_offsets[i][1] * FT_TO_M
                    next_wps[wp_idx] = [base[wp_idx][0] + dx_m,
                                        base[wp_idx][1] + dy_m]

                # Simulate and compute reward
                sim_res = env.simulate(next_wps)
                reward = compute_reward(next_wps, sim_res, pylons_xy, gates)

                # TD update
                next_state = self._offsets_to_state(new_offsets)
                td_target = reward + self.gamma * np.max(self.Q[next_state])
                self.Q[state][action_idx] += self.alpha * (
                    td_target - self.Q[state][action_idx])

                offsets = new_offsets
                current_wps = next_wps

                if reward > ep_best_reward:
                    ep_best_reward = reward
                    ep_best_wps = [list(w) for w in current_wps]

                # Early termination if action was all zeros (no change)
                if not np.any(action_vec):
                    break

            rewards_history.append(ep_best_reward)

            if ep_best_reward > best_reward:
                best_reward = ep_best_reward
                best_wps = [list(w) for w in ep_best_wps]

            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            if verbose and ((ep + 1) % 50 == 0 or ep == 0):
                print(f"  ep {ep+1:4d}: reward={ep_best_reward:7.1f}  "
                      f"eps={self.epsilon:.3f}  "
                      f"|Q|={len(self.Q)}")

        return best_wps, rewards_history

    # -- persistence --

    def save_policy(self, path):
        keys = []
        vals = []
        for state, q in self.Q.items():
            flat = []
            for row in state:
                flat.extend(row)
            keys.append(flat)
            vals.append(q)
        keys_arr = np.asarray(keys, dtype=np.int16)
        vals_arr = np.asarray(vals, dtype=np.float64)
        np.savez(path, keys=keys_arr, vals=vals_arr)
        print(f"Policy saved -> {path}  ({len(keys)} states)")

    def load_policy(self, path):
        data = np.load(path)
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))
        for k, v in zip(data["keys"], data["vals"]):
            state = tuple(tuple(int(x) for x in k[i:i+2])
                          for i in range(0, len(k), 2))
            self.Q[state] = v
        print(f"Policy loaded <- {path}  ({len(data['keys'])} states)")
