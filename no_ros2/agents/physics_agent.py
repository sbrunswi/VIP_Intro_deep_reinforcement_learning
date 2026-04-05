#!/usr/bin/env python3
"""
physics_agent.py
----------------
Q-learning agent tuned for the physics-matched environments_v2 environment.

Changes from example_agent.py:
  1. MAX_STEPS = 10000  (dt=0.02 → 200s per episode, same as old env at dt=0.1)
  2. Elevator trim (+0.042) in all level-flight actions so the aircraft holds altitude.
  3. Coordinated turns: aileron + rudder together instead of rudder-only.
  4. State includes altitude band (too low / on target / too high) so the agent
     can react to altitude drift from the real physics.
  5. Hyperparameters: more exploration (epsilon=0.5), slower decay (0.999),
     more training episodes by default.

Run:
  python no_ros2/agents/physics_agent.py [--viz] [--train N] [--ros2]
  python no_ros2/agents/physics_agent.py --env no_ros2/environments_v2/mock_pylon_env.py
"""

import os
import sys
import importlib.util

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_auav_pkg = os.path.join(_root, 'ros2_ws', 'src', 'auav_pylon_2026')
sys.path.insert(0, _root)
if os.path.isdir(_auav_pkg):
    sys.path.insert(0, _auav_pkg)

import numpy as np
from collections import defaultdict

from no_ros2.environments.env_factory import make_pylon_env
from no_ros2.environments.pylon_course import get_course, PYLON_MID_HEIGHT_M
from no_ros2.environments.pylon_wrapper import PylonRacingWrapper

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

N_X_BINS   = 5
N_Y_BINS   = 5
TARGET_ALT_M  = PYLON_MID_HEIGHT_M   # 3.5 m
ALT_MARGIN_M  = 2.0                  # ± band around target altitude

# ---------------------------------------------------------------------------
# Action set — tuned for v2 physics
#
# Key changes vs example_agent.py:
#   • elevator = +0.042 on all level actions (trim for 7 m/s cruise)
#   • coordinated turns: aileron and rudder together
#   • separate climb/descend actions keep aileron=0 to avoid coupling
# ---------------------------------------------------------------------------
TRIM_ELEV = 0.042   # elevator needed for level flight at 7 m/s in v2 env

DISCRETE_ACTIONS = [
    # (aileron, elevator, throttle, rudder)
    ( 0.0,  TRIM_ELEV,        0.50,  0.00),   # 0: straight, level cruise
    ( 0.3,  TRIM_ELEV,        0.50,  0.25),   # 1: turn right medium (coordinated)
    (-0.3,  TRIM_ELEV,        0.50, -0.25),   # 2: turn left medium  (coordinated)
    ( 0.0,  TRIM_ELEV + 0.15, 0.55,  0.00),   # 3: climb
    ( 0.0,  TRIM_ELEV - 0.15, 0.45,  0.00),   # 4: descend
    ( 0.6,  TRIM_ELEV,        0.50,  0.50),   # 5: turn right hard   (coordinated)
    (-0.6,  TRIM_ELEV,        0.50, -0.50),   # 6: turn left hard    (coordinated)
    ( 0.15, TRIM_ELEV,        0.52,  0.15),   # 7: slight right
    (-0.15, TRIM_ELEV,        0.52, -0.15),   # 8: slight left
    ( 0.3,  TRIM_ELEV + 0.08, 0.53,  0.35),   # 9: climb + turn right
    (-0.3,  TRIM_ELEV + 0.08, 0.53, -0.35),   # 10: climb + turn left
]
N_ACTIONS = len(DISCRETE_ACTIONS)

PASS_THRESHOLD_M = 6.0
MAX_STEPS        = 10000   # 10000 × 0.02 s = 200 s per episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bin(value, low, high, n_bins):
    clipped = np.clip(value, low, high)
    if high <= low:
        return 0
    t = (clipped - low) / (high - low)
    return min(int(t * n_bins), n_bins - 1)


def obs_to_state(obs, target_pylon_idx, pylons, bounds_rect):
    """
    State = (x_region, y_region, heading_band, target_pylon_idx, alt_band)

    alt_band:
      0 = too low  (z < TARGET_ALT_M - ALT_MARGIN_M)
      1 = on target
      2 = too high (z > TARGET_ALT_M + ALT_MARGIN_M)
    """
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]

    g  = bounds_rect
    rx = _bin(x, g["min_x"], g["max_x"], N_X_BINS)
    ry = _bin(y, g["min_y"], g["max_y"], N_Y_BINS)

    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6
    dx /= dist;  dy /= dist

    speed_xy = np.sqrt(vx*vx + vy*vy) + 1e-6
    vx_n = vx / speed_xy;  vy_n = vy / speed_xy
    dot   =  dx*vx_n + dy*vy_n
    cross =  dx*vy_n - dy*vx_n

    if   dot > 0.7:    heading_band = 2   # on target
    elif dot < -0.5:   heading_band = 3   # flying away
    elif cross > 0:    heading_band = 0   # target to the left
    else:              heading_band = 1   # target to the right

    if   z < TARGET_ALT_M - ALT_MARGIN_M:  alt_band = 0
    elif z > TARGET_ALT_M + ALT_MARGIN_M:  alt_band = 2
    else:                                   alt_band = 1

    return (rx, ry, heading_band, target_pylon_idx, alt_band)


def get_reward_shaping(obs, target_pylon_idx, prev_dist, pylons, bounds_rect):
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]

    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6
    dx /= dist;  dy /= dist

    passed = dist < PASS_THRESHOLD_M
    if passed:
        base = 30.0
    else:
        base = 0.0
        if prev_dist is not None:
            base = 0.4 if dist < prev_dist else -0.25
        speed_xy = np.sqrt(vx*vx + vy*vy) + 1e-6
        align = (dx*vx + dy*vy) / speed_xy
        base += 0.3 * align

    g = bounds_rect
    if x < g["min_x"] or x > g["max_x"] or y < g["min_y"] or y > g["max_y"]:
        base -= 0.5

    # Altitude shaping: reward on target, penalise far from it
    alt_err = abs(z - TARGET_ALT_M)
    if alt_err < ALT_MARGIN_M:
        base += 0.1
    elif alt_err > ALT_MARGIN_M * 2:
        base -= 0.3

    return base, passed


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, alpha=0.15, gamma=0.98,
                 epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.999):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q    = self.Q[state]
        best = np.flatnonzero(q == np.max(q))
        return int(np.random.choice(best))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, path: str):
        keys = np.array([list(k) for k in self.Q.keys()], dtype=np.int32)
        vals = np.array(list(self.Q.values()),             dtype=np.float64)
        np.savez(path, keys=keys, vals=vals)
        print(f"Policy saved → {path}.npz  ({len(keys)} states)")

    def load_policy(self, path: str):
        data = np.load(path)
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))
        for k, v in zip(data["keys"], data["vals"]):
            self.Q[tuple(k)] = v
        print(f"Policy loaded ← {path}  ({len(data['keys'])} states)")


# ---------------------------------------------------------------------------
# Episode runner  (same interface as example_agent.run_episode)
# ---------------------------------------------------------------------------

def run_episode(env, agent, course_data, train=True, max_steps=MAX_STEPS):
    pylons      = course_data["pylons"]
    bounds_rect = course_data["bounds_rect"]
    n_pylons    = len(pylons)

    obs, _    = env.reset()
    target_idx = 0
    prev_dist  = None
    total_reward = 0.0
    pylons_passed = 0
    TIME_PENALTY  = 0.02

    for step in range(max_steps):
        state      = obs_to_state(obs, target_idx, pylons, bounds_rect)
        action_idx = agent.get_action(state, greedy=not train)
        action     = np.array(DISCRETE_ACTIONS[action_idx], dtype=np.float32)

        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        extra, passed = get_reward_shaping(obs, target_idx, prev_dist, pylons, bounds_rect)
        reward = extra - TIME_PENALTY
        if terminated and r_env < 0:
            reward += r_env

        if passed:
            pylons_passed += 1
            target_idx = (target_idx + 1) % n_pylons
            prev_dist  = None
        else:
            tgt       = pylons[target_idx]
            prev_dist = np.sqrt((tgt[0]-obs[0])**2 + (tgt[1]-obs[1])**2)

        next_state = obs_to_state(obs, target_idx, pylons, bounds_rect)
        if train:
            agent.update(state, action_idx, reward, next_state, done)
        total_reward += reward
        if done:
            break

    if train:
        agent.decay_epsilon()
    return total_reward, pylons_passed, step + 1


# ---------------------------------------------------------------------------
# Dynamic env loader (same helper as example_agent)
# ---------------------------------------------------------------------------

def _make_env_from_file(env_path: str, course: str):
    env_path = os.path.abspath(env_path)
    if not os.path.isfile(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    spec = importlib.util.spec_from_file_location("_env_module", env_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "MockPylonRacingEnv"):
        raise AttributeError(f"{env_path} must define 'MockPylonRacingEnv'")
    return PylonRacingWrapper(mod.MockPylonRacingEnv(course=course))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Physics-tuned Q-learning agent")
    parser.add_argument("--viz",           action="store_true")
    parser.add_argument("--train",         type=int, default=1000,  help="Training episodes (default 1000)")
    parser.add_argument("--no-train",      action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--show-details",  action="store_true")
    parser.add_argument("--env",           metavar="FILE", default=None, help="Path to environment .py file")
    parser.add_argument("--ros2",          action="store_true")
    parser.add_argument("--train-in-ros2", action="store_true")
    parser.add_argument("--save-policy",   metavar="FILE", default=None)
    parser.add_argument("--load-policy",   metavar="FILE", default=None)
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument("--purt",   action="store_true")
    course_group.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    course_name = "purt" if args.purt else "sample"
    course_data = get_course(course_name)
    print(f"Course: {course_name} ({len(course_data['pylons'])} pylons)")

    if args.train_in_ros2:
        args.ros2 = True
    if args.ros2:
        import rclpy
        rclpy.init()

    agent = QLearningAgent()

    if args.load_policy:
        agent.load_policy(args.load_policy)

    # --- Training ---
    if not args.no_train and not args.load_policy:
        print(f"\n--- Training: {args.train} episodes ---")
        if args.train_in_ros2:
            train_env = make_pylon_env(use_ros2=True,  course=course_name)
        elif args.env:
            print(f"  Env: {os.path.basename(args.env)}")
            train_env = _make_env_from_file(args.env, course_name)
        else:
            train_env = make_pylon_env(use_ros2=False, course=course_name)

        for ep in range(args.train):
            r, passed, steps = run_episode(train_env, agent, course_data, train=True)
            print(f"  ep {ep+1:4d}: reward={r:7.1f}  pylons={passed}  "
                  f"steps={steps:5d}  eps={agent.epsilon:.3f}")
        train_env.close()
        print("Training complete.\n")
        if args.save_policy:
            agent.save_policy(args.save_policy)

    # --- Evaluation ---
    print("--- Evaluation ---")
    if args.ros2:
        eval_env = make_pylon_env(use_ros2=True, course=course_name)
    elif args.env:
        eval_env = _make_env_from_file(args.env, course_name)
    else:
        eval_env = make_pylon_env(use_ros2=False, course=course_name)

    viz = None
    if args.viz:
        from no_ros2.viz_3d import PylonRacingViz3D
        viz = PylonRacingViz3D(
            pylons=course_data["pylons"], gates=course_data["gates"],
            pylon_names=course_data["pylon_names"], bounds_rect=course_data["bounds_rect"],
            pylon_height_m=course_data["pylon_height_m"],
            pylon_radius_m=course_data["pylon_radius_m"],
            show_state_vector=args.show_details,
        )

    pylons      = course_data["pylons"]
    bounds_rect = course_data["bounds_rect"]
    n_pylons    = len(pylons)

    for ep in range(args.eval_episodes):
        obs, _    = eval_env.reset()
        target_idx = 0
        prev_dist  = None
        total_r    = 0.0
        pylons_passed = 0
        if viz:
            viz.clear_trail()

        for step in range(MAX_STEPS):
            state      = obs_to_state(obs, target_idx, pylons, bounds_rect)
            action_idx = agent.get_action(state, greedy=True)
            action     = np.array(DISCRETE_ACTIONS[action_idx], dtype=np.float32)
            obs, r_env, terminated, truncated, _ = eval_env.step(action)

            extra, passed = get_reward_shaping(obs, target_idx, prev_dist, pylons, bounds_rect)
            reward = extra - 0.02
            if terminated and r_env < 0:
                reward += r_env

            if passed:
                pylons_passed += 1
                target_idx = (target_idx + 1) % n_pylons
                prev_dist  = None
            else:
                tgt       = pylons[target_idx]
                prev_dist = np.sqrt((tgt[0]-obs[0])**2 + (tgt[1]-obs[1])**2)

            total_r += reward
            if viz:
                viz.update(obs, reward=reward, step_count=step,
                           laps=pylons_passed // n_pylons)
            if terminated or truncated:
                break

        print(f"  Eval ep {ep+1}: reward={total_r:.1f}  pylons={pylons_passed}  steps={step+1}")

    if viz:
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        viz.close()

    eval_env.close()
    if args.ros2:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
