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

TARGET_ALT_M  = PYLON_MID_HEIGHT_M   # 3.5 m
ALT_MARGIN_M  = 2.0                  # ± band around target altitude

# ---------------------------------------------------------------------------
# Action set — uses inner-loop stabilization
#
# Each discrete action specifies a *desired* (roll_deg, pitch_deg, throttle).
# A simple P+D controller computes the actual aileron/elevator/rudder from
# the current attitude and body rates, so the aircraft stays stable even
# with the corrected (weak) aerodynamic damping.
# ---------------------------------------------------------------------------
TRIM_PITCH_DEG = 2.0   # pitch for level cruise at ~7 m/s
TRIM_THROTTLE  = 0.35  # throttle for ~7 m/s cruise

# (target_roll_deg, target_pitch_deg, throttle)
DISCRETE_ACTIONS = [
    ( 0.0,  TRIM_PITCH_DEG,       TRIM_THROTTLE),       # 0: straight, level
    ( 20.0, TRIM_PITCH_DEG,       TRIM_THROTTLE),       # 1: bank right medium
    (-20.0, TRIM_PITCH_DEG,       TRIM_THROTTLE),       # 2: bank left medium
    ( 0.0,  TRIM_PITCH_DEG + 5.0, TRIM_THROTTLE + 0.1), # 3: climb
    ( 0.0,  TRIM_PITCH_DEG - 5.0, TRIM_THROTTLE - 0.05),# 4: descend
    ( 40.0, TRIM_PITCH_DEG,       TRIM_THROTTLE + 0.05),# 5: bank right hard
    (-40.0, TRIM_PITCH_DEG,       TRIM_THROTTLE + 0.05),# 6: bank left hard
    ( 10.0, TRIM_PITCH_DEG,       TRIM_THROTTLE),       # 7: slight right
    (-10.0, TRIM_PITCH_DEG,       TRIM_THROTTLE),       # 8: slight left
    ( 25.0, TRIM_PITCH_DEG + 3.0, TRIM_THROTTLE + 0.08),# 9: climb + turn right
    (-25.0, TRIM_PITCH_DEG + 3.0, TRIM_THROTTLE + 0.08),# 10: climb + turn left
]
N_ACTIONS = len(DISCRETE_ACTIONS)

# Inner-loop PD gains
_KP_PITCH = 0.08    # elevator per degree of pitch error
_KD_PITCH = 0.5     # elevator per rad/s of pitch rate
_KP_ROLL  = 0.04    # aileron per degree of roll error
_KD_ROLL  = 0.3     # aileron per rad/s of roll rate
_KP_YAW   = 0.02    # rudder per degree of roll (coordinated turn)


def action_to_controls(action_idx, obs):
    """Convert discrete action + current obs → (aileron, elevator, throttle, rudder)."""
    target_roll_deg, target_pitch_deg, throttle = DISCRETE_ACTIONS[action_idx]

    roll_deg  = np.degrees(obs[3])   # current roll
    pitch_deg = np.degrees(obs[4])   # current pitch (nose-up positive)
    p, q, r   = obs[12], obs[13], obs[14]  # body rates

    # In banked flight, vertical lift = L*cos(roll) = weight, so we need
    # more pitch (more CL) to compensate. Add pitch proportional to 1/cos(roll)-1.
    roll_rad = np.radians(roll_deg)
    cos_roll = max(np.cos(roll_rad), 0.3)  # clamp to avoid huge values at extreme bank
    bank_pitch_comp = (1.0 / cos_roll - 1.0) * 4.0  # degrees of extra pitch per bank

    # Pitch PD controller
    pitch_err = (target_pitch_deg + bank_pitch_comp) - pitch_deg
    elevator  = _KP_PITCH * pitch_err - _KD_PITCH * q
    elevator  = np.clip(elevator, -1.0, 1.0)

    # Roll PD controller
    roll_err = target_roll_deg - roll_deg
    aileron  = _KP_ROLL * roll_err - _KD_ROLL * p
    aileron  = np.clip(aileron, -1.0, 1.0)

    # Coordinated turn: rudder proportional to roll (keeps sideslip near zero)
    rudder = _KP_YAW * roll_deg
    rudder = np.clip(rudder, -1.0, 1.0)

    throttle = np.clip(throttle, 0.0, 1.0)

    return np.array([aileron, elevator, throttle, rudder], dtype=np.float32)

PASS_THRESHOLD_M = 6.0
MAX_STEPS        = 1500   # 1500 × 0.02 s = 30 s per episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def obs_to_state(obs, target_pylon_idx, pylons, bounds_rect):
    """
    State = (bearing_band, dist_band, alt_band)

    bearing_band (8 bins): relative bearing from velocity vector to target
      Encodes which direction to turn to reach the target.
    dist_band (3 bins): close / medium / far from target
    alt_band (3 bins): too low / on target / too high
    """
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]

    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

    # Bearing from velocity vector to target (signed angle)
    speed_xy = np.sqrt(vx*vx + vy*vy) + 1e-6
    # angle of velocity vector
    vel_angle = np.arctan2(vy, vx)
    # angle to target
    tgt_angle = np.arctan2(dy, dx)
    # relative bearing (positive = target is to the right)
    bearing = tgt_angle - vel_angle
    # wrap to [-pi, pi]
    bearing = (bearing + np.pi) % (2*np.pi) - np.pi

    # 8 bearing bins (each 45 degrees)
    bearing_band = int((bearing + np.pi) / (2*np.pi) * 8) % 8

    # Distance bands: close (<10m), medium (10-25m), far (>25m)
    if   dist < 10:  dist_band = 0
    elif dist < 25:  dist_band = 1
    else:            dist_band = 2

    # Altitude bands
    if   z < TARGET_ALT_M - ALT_MARGIN_M:  alt_band = 0
    elif z > TARGET_ALT_M + ALT_MARGIN_M:  alt_band = 2
    else:                                   alt_band = 1

    return (bearing_band, dist_band, alt_band)


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
        action     = action_to_controls(action_idx, obs)

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
            if (ep + 1) % 50 == 0 or ep == 0:
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
            action     = action_to_controls(action_idx, obs)
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
