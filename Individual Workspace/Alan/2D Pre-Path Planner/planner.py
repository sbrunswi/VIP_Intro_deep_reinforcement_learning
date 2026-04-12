"""
planner.py
----------
2D Pre-Path Q-learning planner with a per-waypoint small-step action space.

Each episode adjusts all waypoints sequentially. For each waypoint the
state is (waypoint_index, offset_x_bin, offset_y_bin) and the action is
one of 9 compass directions (+stay). This keeps the action space at 9
regardless of the number of waypoints.

The heuristic waypoint initialization is imported from the shared
`heuristic.py` module in the parent directory.
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FT_TO_M = 0.3048  # grid spacing

# 9 discrete actions: 8 compass directions + stay
ACTIONS = [
    ( 0.0,  0.0),   # stay
    ( 1.0,  0.0),   # right
    (-1.0,  0.0),   # left
    ( 0.0,  1.0),   # up
    ( 0.0, -1.0),   # down
    ( 0.71,  0.71), # up-right
    (-0.71,  0.71), # up-left
    ( 0.71, -0.71), # down-right
    (-0.71, -0.71), # down-left
]
N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(sim_result):
    """
    Scalar reward for a candidate waypoint set.

    The rules-aware env already checks pylon collisions, gate crossings
    in order, correct IN/OUT direction, and lap closure.
    """
    if sim_result["completed"]:
        return 100.0 - sim_result["total_time"] - 0.1 * sim_result["smoothness"]
    else:
        return -50.0


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class PrePathQLearner:
    def __init__(self, n_waypoints, n_offset_bins=5, offset_range=4.0,
                 step_size=0.8, alpha=0.1, gamma=0.95,
                 epsilon=0.4, epsilon_min=0.05, epsilon_decay=0.995,
                 max_steps_per_episode=60):
        """
        Parameters
        ----------
        n_waypoints : int
            Number of adjustable waypoints.
        n_offset_bins : int
            Bins per axis for discretizing waypoint offsets.
        offset_range : float
            Max offset from heuristic position (metres).
        step_size : float
            How far each action shifts a waypoint (metres).
        """
        self.n_waypoints = n_waypoints
        self.n_bins = n_offset_bins
        self.offset_range = offset_range
        self.step_size = step_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_steps = max_steps_per_episode
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))

    # -- discretization helpers --

    def _bin(self, value):
        """Map a continuous offset in [-offset_range, +offset_range] to a bin."""
        t = (np.clip(value, -self.offset_range, self.offset_range)
             + self.offset_range) / (2 * self.offset_range)
        return min(int(t * self.n_bins), self.n_bins - 1)

    def _get_state(self, wp_idx, offsets):
        ox, oy = offsets[wp_idx]
        return (wp_idx, self._bin(ox), self._bin(oy))

    # -- action selection --

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q = self.Q[state]
        best = np.flatnonzero(q == np.max(q))
        return int(np.random.choice(best))

    # -- training --

    def run_optimization(self, heuristic_wps, pylons_xy, env, gates,
                         n_episodes=500, verbose=True):
        """
        Optimise waypoint positions via Q-learning.

        Parameters
        ----------
        heuristic_wps : list of (x, y)
            Waypoint list from the shared heuristic (variable length).
        pylons_xy : array-like (n_pylons, 2)
        env : PointMassEnv2D (rules-aware)
        gates : list of (i_pylon_a, i_pylon_b) tuples
        n_episodes : int

        Returns
        -------
        best_wps : list of (x, y)
        rewards_history : list of float
        """
        base = [list(w) for w in heuristic_wps]
        n_wps = len(base)
        assert n_wps == self.n_waypoints, (
            f"Expected {self.n_waypoints} waypoints, got {n_wps}")

        offsets = np.zeros((n_wps, 2))
        best_reward = -np.inf
        best_offsets = offsets.copy()
        rewards_history = []

        for ep in range(n_episodes):
            episode_pairs = []

            for i in range(n_wps):
                state = self._get_state(i, offsets)
                action_idx = self.get_action(state)
                episode_pairs.append((state, action_idx))

                dx, dy = ACTIONS[action_idx]
                offsets[i][0] = np.clip(
                    offsets[i][0] + dx * self.step_size,
                    -self.offset_range, self.offset_range)
                offsets[i][1] = np.clip(
                    offsets[i][1] + dy * self.step_size,
                    -self.offset_range, self.offset_range)

            # Build waypoints with current offsets
            wps = list(base)
            for i in range(n_wps):
                wps[i] = [base[i][0] + offsets[i][0],
                          base[i][1] + offsets[i][1]]

            sim_res = env.simulate(wps)
            reward = compute_reward(sim_res)
            rewards_history.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_offsets = offsets.copy()

            # Monte Carlo update: same reward for all (state, action) pairs
            for state, action_idx in episode_pairs:
                self.Q[state][action_idx] += self.alpha * (
                    reward - self.Q[state][action_idx])

            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            if verbose and ((ep + 1) % 50 == 0 or ep == 0):
                status = "OK" if sim_res["completed"] else "FAIL"
                print(f"  ep {ep+1:4d}: reward={reward:7.1f}  "
                      f"time={sim_res['total_time']:5.1f}s  [{status}]  "
                      f"eps={self.epsilon:.3f}")

        # Build best waypoints
        best_wps = list(base)
        for i in range(n_wps):
            best_wps[i] = [base[i][0] + best_offsets[i][0],
                           base[i][1] + best_offsets[i][1]]

        return best_wps, rewards_history

    # -- persistence --

    def save_policy(self, path):
        keys = np.array([list(k) for k in self.Q.keys()], dtype=np.int32)
        vals = np.array(list(self.Q.values()), dtype=np.float64)
        np.savez(path, keys=keys, vals=vals)
        print(f"Policy saved -> {path}  ({len(keys)} states)")

    def load_policy(self, path):
        data = np.load(path)
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))
        for k, v in zip(data["keys"], data["vals"]):
            self.Q[tuple(k)] = v
        print(f"Policy loaded <- {path}  ({len(data['keys'])} states)")
