#!/usr/bin/env python3
"""
Q-learning agent for pylon racing (mock or ROS2 env).

How it works:
- Lap order: always fly toward P1, then P2, then P3, then P4, then repeat (P1...).
- State: discretized to (x_region, y_region, heading_toward_target, target_pylon_idx) so the
  state space stays small (~400 states) and Q-learning can learn.
- Pass: when within PASS_THRESHOLD_M (6 m) of the current target pylon, we count a pass and
  advance the target to the next pylon in order.
- Reward: +30 for passing; +0.4 getting closer / -0.25 getting farther (to current target);
  alignment bonus for pointing at target; -0.5 outside course bounds; small altitude terms.
  We do not use the env’s +1 per step; only shaping + crash penalty.
- Actions: 11 discrete (aileron, elevator, throttle, rudder) — mostly level flight with
  turns; elevator=0.0 at throttle=0.5 holds altitude at 7 m/s (Sport Cub S2 cruise).
  Positive elevator = climb in both mock and ROS2.

Run: python no_ros2/agents/example_agent.py [--viz] [--train N] [--no-train] [--ros2] [--train-in-ros2]
"""
import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_auav_pkg = os.path.abspath(os.path.join(_root, 'ros2_ws', 'src', 'auav_pylon_2026'))
sys.path.insert(0, _root)
if os.path.isdir(_auav_pkg):
    sys.path.insert(0, _auav_pkg)

import importlib.util
import numpy as np
from collections import defaultdict

from no_ros2.environments.env_factory import make_pylon_env
from no_ros2.environments.pylon_course import get_course, PYLON_MID_HEIGHT_M
from no_ros2.environments.pylon_wrapper import PylonRacingWrapper


def _make_env_from_file(env_path: str, course: str):
    """Load a MockPylonRacingEnv from a .py file path, wrapped for the agent."""
    env_path = os.path.abspath(env_path)
    if not os.path.isfile(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    spec = importlib.util.spec_from_file_location("_env_module", env_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "MockPylonRacingEnv"):
        raise AttributeError(f"{env_path} must define 'MockPylonRacingEnv'")
    return PylonRacingWrapper(mod.MockPylonRacingEnv(course=course))

# --- Discretization (state = rx, ry, heading_band, target_pylon_idx) ---
N_X_BINS = 5
N_Y_BINS = 5
TARGET_ALT_M = PYLON_MID_HEIGHT_M

# Action: (aileron, elevator, throttle, rudder).
# Elevator sign convention (matches both mock and ROS2 after pylon_env.py fix):
#   positive elevator  → nose up → climb
#   negative elevator  → nose down → descend
# Throttle=0.5 → 7 m/s cruise; elevator=0.0 at throttle=0.5 → level flight.
# Rudder: ±1.0 → ±1.5 rad/s yaw rate (≈ 45° coordinated turn at 7 m/s).
DISCRETE_ACTIONS = [
    (0.0,  0.00, 0.50,  0.00),  # 0: straight, level at cruise
    (0.0,  0.00, 0.50,  0.25),  # 1: turn right (medium), level
    (0.0,  0.00, 0.50, -0.25),  # 2: turn left (medium), level
    (0.0,  0.15, 0.55,  0.00),  # 3: climb (altitude recovery)
    (0.0, -0.15, 0.45,  0.00),  # 4: descend (altitude correction)
    (0.0,  0.00, 0.50,  0.50),  # 5: turn right (hard), level
    (0.0,  0.00, 0.50, -0.50),  # 6: turn left (hard), level
    (0.0,  0.00, 0.52,  0.15),  # 7: slight right, level
    (0.0,  0.00, 0.52, -0.15),  # 8: slight left, level
    (0.0,  0.08, 0.53,  0.35),  # 9: slight climb + turn right
    (0.0,  0.08, 0.53, -0.35),  # 10: slight climb + turn left
]
N_ACTIONS = len(DISCRETE_ACTIONS)

PASS_THRESHOLD_M = 6.0


def _bin(value, low, high, n_bins):
    clipped = np.clip(value, low, high)
    if high <= low:
        return 0
    t = (clipped - low) / (high - low)
    idx = int(t * n_bins)
    return min(idx, n_bins - 1)


def obs_to_state(obs, target_pylon_idx, pylons, bounds_rect):
    x, y = obs[0], obs[1]
    if len(obs) >= 15:
        vx, vy = obs[6], obs[7]
    else:
        vx, vy = obs[3], obs[4]
    g = bounds_rect
    rx = _bin(x, g["min_x"], g["max_x"], N_X_BINS)
    ry = _bin(y, g["min_y"], g["max_y"], N_Y_BINS)
    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist_to_target = np.sqrt(dx * dx + dy * dy) + 1e-6
    dx /= dist_to_target
    dy /= dist_to_target
    speed_xy = np.sqrt(vx * vx + vy * vy) + 1e-6
    vx_n = vx / speed_xy
    vy_n = vy / speed_xy
    dot = dx * vx_n + dy * vy_n
    cross = dx * vy_n - dy * vx_n
    if dot > 0.7:
        heading_band = 2
    elif dot < -0.5:
        heading_band = 3
    elif cross > 0:
        heading_band = 0
    else:
        heading_band = 1
    return (rx, ry, heading_band, target_pylon_idx)


def get_reward_shaping(obs, target_pylon_idx, prev_dist, pylons, bounds_rect):
    x, y, z = obs[0], obs[1], obs[2]
    if len(obs) >= 15:
        vx, vy = obs[6], obs[7]
    else:
        vx, vy = obs[3], obs[4]

    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist = np.sqrt(dx * dx + dy * dy) + 1e-6
    dx /= dist
    dy /= dist

    passed = dist < PASS_THRESHOLD_M
    if passed:
        base = 30.0
    else:
        base = 0.0
        if prev_dist is not None:
            base = 0.4 if dist < prev_dist else -0.25
        speed_xy = np.sqrt(vx * vx + vy * vy) + 1e-6
        align = (dx * vx + dy * vy) / speed_xy
        base += 0.3 * align

    g = bounds_rect
    if x < g["min_x"] or x > g["max_x"] or y < g["min_y"] or y > g["max_y"]:
        base -= 0.5

    if z < 1.0:
        base -= 0.2
    elif abs(z - TARGET_ALT_M) < 1.0:
        base += 0.1

    return base, passed


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2, epsilon_min=0.05, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q = self.Q[state]
        max_q = np.max(q)
        best = np.flatnonzero(q == max_q)
        return int(np.random.choice(best))

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, path: str):
        """Save Q-table to a .npz file."""
        keys = np.array([list(k) for k in self.Q.keys()], dtype=np.int32)
        vals = np.array(list(self.Q.values()), dtype=np.float64)
        np.savez(path, keys=keys, vals=vals)
        print(f"Policy saved to {path}.npz  ({len(keys)} states)")

    def load_policy(self, path: str):
        """Load Q-table from a .npz file."""
        data = np.load(path)
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))
        for k, v in zip(data["keys"], data["vals"]):
            self.Q[tuple(k)] = v
        print(f"Policy loaded from {path}  ({len(data['keys'])} states)")


def run_episode(env, agent, course_data, train=True, max_steps=2000):
    pylons = course_data["pylons"]
    bounds_rect = course_data["bounds_rect"]
    n_pylons = len(pylons)
    obs, _ = env.reset()
    target_idx = 0
    prev_dist = None
    total_reward = 0.0
    pylons_passed = 0
    TIME_PENALTY = 0.02
    for step in range(max_steps):
        state = obs_to_state(obs, target_idx, pylons, bounds_rect)
        action_idx = agent.get_action(state, greedy=not train)
        action = np.array(DISCRETE_ACTIONS[action_idx], dtype=np.float32)
        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        extra, passed = get_reward_shaping(obs, target_idx, prev_dist, pylons, bounds_rect)
        reward = extra - TIME_PENALTY
        if terminated and r_env < 0:
            reward += r_env

        if passed:
            pylons_passed += 1
            target_idx = (target_idx + 1) % n_pylons
            prev_dist = None
        else:
            target = pylons[target_idx]
            prev_dist = np.sqrt((target[0] - obs[0])**2 + (target[1] - obs[1])**2)

        next_state = obs_to_state(obs, target_idx, pylons, bounds_rect)
        if train:
            agent.update(state, action_idx, reward, next_state, done)
        total_reward += reward
        if done:
            break

    if train:
        agent.decay_epsilon()
    return total_reward, pylons_passed, step + 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q-Learning agent around pylons")
    parser.add_argument("--viz", action="store_true", help="Show 3D visualization")
    parser.add_argument("--train", type=int, default=400, metavar="N", help="Training episodes (default 400)")
    parser.add_argument("--no-train", action="store_true", help="Skip training, run with random policy (for testing)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes after training")
    parser.add_argument("--show-details", action="store_true", help="Show full state vector in viz (use with --viz)")
    parser.add_argument("--env",          metavar="FILE", default=None, help="Path to environment .py file (e.g. no_ros2/environments_v2/mock_pylon_env.py)")
    parser.add_argument("--save-policy", metavar="FILE", default=None, help="Save Q-table after training (e.g. policy.npz)")
    parser.add_argument("--load-policy", metavar="FILE", default=None, help="Load Q-table instead of training")
    parser.add_argument("--ros2", action="store_true", help="Connect to ROS2 ONLY for the evaluation flight.")
    parser.add_argument("--train-in-ros2", action="store_true", help="Connect to ROS2 for BOTH training and evaluation (Slow/Real-time).")
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument("--purt", action="store_true", help="Use PURT competition course (4 pylons)")
    course_group.add_argument("--sample", action="store_true", help="Use sample course 1 (6 pylons, default)")
    args = parser.parse_args()

    course_name = "purt" if args.purt else "sample"
    course_data = get_course(course_name)
    n_pylons = len(course_data["pylons"])
    print(f"Course: {course_name} ({n_pylons} pylons)")
    
    # If the user wants to train in ROS 2, they definitely want to evaluate in it too
    if args.train_in_ros2:
        args.ros2 = True

    # 1. Initialize ROS 2 if either flag is active
    if args.ros2:
        import rclpy
        print("Initializing ROS 2...")
        rclpy.init()

    agent = QLearningAgent(alpha=0.15, gamma=0.98, epsilon=0.3, epsilon_min=0.05, epsilon_decay=0.997)

    if args.load_policy:
        agent.load_policy(args.load_policy)
        args.no_train = True  # skip training if a policy is loaded

    # ---------------------------------------------------------
    # 2. TRAINING PHASE 
    # ---------------------------------------------------------
    if not args.no_train:
        print(f"\n--- Phase 1: Training ---")
        if args.train_in_ros2:
            print(f"Training for {args.train} episodes in ROS 2 environment (SLOW, Real-Time)...")
            train_env = make_pylon_env(use_ros2=True, course=course_name)
        elif args.env:
            print(f"Training for {args.train} episodes in {os.path.basename(args.env)}...")
            train_env = _make_env_from_file(args.env, course_name)
        else:
            print(f"Training for {args.train} episodes in MOCK environment (FAST, Headless)...")
            train_env = make_pylon_env(use_ros2=False, course=course_name)

        for ep in range(args.train):
            r, passed, steps = run_episode(train_env, agent, course_data, train=True)
            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"  Episode {ep + 1}: reward={r:.1f}, pylons_passed={passed}, steps={steps}, epsilon={agent.epsilon:.3f}")
        
        print("Training complete.\n")
        train_env.close()
        if args.save_policy:
            agent.save_policy(args.save_policy)

    # ---------------------------------------------------------
    # 3. EVALUATION PHASE
    # ---------------------------------------------------------
    print(f"--- Phase 2: Evaluation ---")
    if args.ros2:
        print("Connecting to ROS 2 for the final Evaluation Flight...")
        eval_env = make_pylon_env(use_ros2=True, course=course_name)
    elif args.env:
        print(f"Using {os.path.basename(args.env)} for the Evaluation Flight...")
        eval_env = _make_env_from_file(args.env, course_name)
    else:
        print("Using MOCK environment for the Evaluation Flight...")
        eval_env = make_pylon_env(use_ros2=False, course=course_name)

    viz = None
    if args.viz:
        from no_ros2.viz_3d import PylonRacingViz3D
        viz = PylonRacingViz3D(
            pylons=course_data["pylons"],
            gates=course_data["gates"],
            pylon_names=course_data["pylon_names"],
            bounds_rect=course_data["bounds_rect"],
            pylon_height_m=course_data["pylon_height_m"],
            pylon_radius_m=course_data["pylon_radius_m"],
            show_state_vector=args.show_details,
        )

    pylons = course_data["pylons"]
    bounds_rect = course_data["bounds_rect"]
    n_show = args.eval_episodes if (args.viz or args.ros2) else 1
    print(f"Running {n_show} episode(s) with Greedy Policy...")

    for ep in range(n_show):
        obs, _ = eval_env.reset()
        target_idx = 0
        prev_dist = None
        total_reward = 0.0
        pylons_passed = 0
        step = 0

        if viz:
            viz.clear_trail()
            viz.update(obs, reward=0.0, step_count=step, laps=0)

        TIME_PENALTY = 0.02
        while step < 2000:
            state = obs_to_state(obs, target_idx, pylons, bounds_rect)

            # Action locked to greedy (no random exploration) during evaluation
            action_idx = agent.get_action(state, greedy=True)
            action = np.array(DISCRETE_ACTIONS[action_idx], dtype=np.float32)

            obs, r_env, terminated, truncated, _ = eval_env.step(action)
            extra, passed = get_reward_shaping(obs, target_idx, prev_dist, pylons, bounds_rect)
            reward = extra - TIME_PENALTY

            if terminated and r_env < 0:
                reward += r_env

            if passed:
                pylons_passed += 1
                target_idx = (target_idx + 1) % n_pylons
                prev_dist = None
            else:
                target = pylons[target_idx]
                prev_dist = np.sqrt((target[0] - obs[0])**2 + (target[1] - obs[1])**2)

            total_reward += reward
            step += 1

            if viz:
                viz.update(obs, reward=reward, step_count=step, laps=pylons_passed // n_pylons)

            if terminated or truncated:
                break
                
        print(f"  Eval episode {ep + 1}: reward={total_reward:.1f}, pylons_passed={pylons_passed}, steps={step}")
        if viz and ep < n_show - 1:
            viz.clear_trail()

    # ---------------------------------------------------------
    # 4. CLEANUP
    # ---------------------------------------------------------
    if viz:
        try:
            input("Press Enter to close visualization...")
        except EOFError:
            pass
        viz.close()
        
    eval_env.close()

    if args.ros2:
        rclpy.shutdown()

if __name__ == "__main__":
    main()