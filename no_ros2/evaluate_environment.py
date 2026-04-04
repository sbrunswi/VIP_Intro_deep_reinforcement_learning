#!/usr/bin/env python3
"""
evaluate_environment.py
-----------------------
End-to-end pipeline: train an agent, then evaluate it in the mock and/or
ROS2 simulator.  State-action pairs from every evaluation step are saved
to a .npz log file.

Usage:
  # Train 400 episodes in mock, evaluate 5 in mock, save log
  python no_ros2/evaluate_environment.py --agent no_ros2/agents/example_agent.py

  # Train then evaluate in both mock and ROS2
  python no_ros2/evaluate_environment.py \
      --agent no_ros2/agents/example_agent.py \
      --mock-episodes 5 --ros2-episodes 5

  # Skip training, load saved policy
  python no_ros2/evaluate_environment.py \
      --agent no_ros2/agents/example_agent.py \
      --load-policy policy.npz --mock-episodes 10

Log file (.npz) keys:
  {env}_obs      float32  (episodes, steps, 15)  NaN-padded after episode ends
  {env}_actions  float32  (episodes, steps, 4)
  {env}_rewards  float32  (episodes, steps)
  {env}_pylons   int32    (episodes,)   pylons passed per episode
  {env}_steps    int32    (episodes,)   actual steps taken
  where env = "mock" and/or "ros2"
"""

import sys
import os
import argparse
import importlib.util
from datetime import datetime

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from no_ros2.environments.env_factory import make_pylon_env
from no_ros2.environments.pylon_course import get_course
from no_ros2.environments.pylon_wrapper import PylonRacingWrapper

MAX_STEPS   = 2000
TIME_PENALTY = 0.02


# ---------------------------------------------------------------------------
# Dynamic agent loader
# ---------------------------------------------------------------------------

def load_module(path: str, name: str):
    """Import any .py file by path and return the module."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_agent_module(agent_path: str):
    return load_module(agent_path, "_agent_module")


def make_env_from_file(env_path: str, course: str):
    """
    Load a mock environment from a .py file path.
    The file must expose a class named MockPylonRacingEnv.
    Returns a PylonRacingWrapper around the env.
    """
    mod = load_module(env_path, "_env_module")
    if not hasattr(mod, "MockPylonRacingEnv"):
        raise AttributeError(
            f"{env_path} must define a class named 'MockPylonRacingEnv'"
        )
    base = mod.MockPylonRacingEnv(course=course)
    return PylonRacingWrapper(base)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent_mod, agent, course_data):
    """
    Run one greedy episode using the agent module's obs_to_state /
    get_reward_shaping helpers (same interface as example_agent.py).

    Returns: obs_log (T,15), action_log (T,4), reward_log (T,), pylons_passed, steps
    """
    pylons      = course_data["pylons"]
    bounds_rect = course_data["bounds_rect"]
    n_pylons    = len(pylons)

    obs, _     = env.reset()
    target_idx = 0
    prev_dist  = None
    obs_log    = []
    action_log = []
    reward_log = []
    pylons_passed = 0

    for step in range(MAX_STEPS):
        state      = agent_mod.obs_to_state(obs, target_idx, pylons, bounds_rect)
        action_idx = agent.get_action(state, greedy=True)
        action     = np.array(agent_mod.DISCRETE_ACTIONS[action_idx], dtype=np.float32)

        obs_log.append(obs.copy())
        action_log.append(action.copy())

        obs, r_env, terminated, truncated, _ = env.step(action)

        extra, passed = agent_mod.get_reward_shaping(
            obs, target_idx, prev_dist, pylons, bounds_rect
        )
        reward = extra - TIME_PENALTY
        if terminated and r_env < 0:
            reward += r_env
        reward_log.append(float(reward))

        if passed:
            pylons_passed += 1
            target_idx = (target_idx + 1) % n_pylons
            prev_dist  = None
        else:
            tgt       = pylons[target_idx]
            prev_dist = np.sqrt((tgt[0] - obs[0])**2 + (tgt[1] - obs[1])**2)

        if terminated or truncated:
            break

    return (
        np.array(obs_log,    dtype=np.float32),
        np.array(action_log, dtype=np.float32),
        np.array(reward_log, dtype=np.float32),
        pylons_passed,
        step + 1,
    )


# ---------------------------------------------------------------------------
# Multi-episode evaluation
# ---------------------------------------------------------------------------

def evaluate(env, agent_mod, agent, course_data, n_episodes, label):
    results = []
    for ep in range(n_episodes):
        obs, act, rew, pylons, steps = run_episode(env, agent_mod, agent, course_data)
        total_r = float(np.sum(rew))
        print(f"  [{label}] ep {ep+1}/{n_episodes}: "
              f"reward={total_r:.1f}  pylons={pylons}  steps={steps}")
        results.append((obs, act, rew, pylons, steps))
    avg_r = np.mean([np.sum(r[2]) for r in results])
    avg_p = np.mean([r[3] for r in results])
    print(f"  [{label}] summary: avg_reward={avg_r:.1f}  avg_pylons={avg_p:.1f}\n")
    return results


def pad_episodes(results, max_steps, obs_dim=15, act_dim=4):
    n = len(results)
    obs_arr = np.full((n, max_steps, obs_dim), np.nan, dtype=np.float32)
    act_arr = np.full((n, max_steps, act_dim), np.nan, dtype=np.float32)
    rew_arr = np.full((n, max_steps),           np.nan, dtype=np.float32)
    for i, (obs, act, rew, _, steps) in enumerate(results):
        T = min(steps, max_steps)
        obs_arr[i, :T] = obs[:T]
        act_arr[i, :T] = act[:T]
        rew_arr[i, :T] = rew[:T]
    return obs_arr, act_arr, rew_arr


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def compare_logs(current: dict, baseline: dict):
    """Print a side-by-side comparison of current vs baseline log."""
    print("\n" + "=" * 60)
    print("  COMPARISON  (current vs baseline)")
    print("=" * 60)

    all_envs = sorted({k.split("_")[0] for k in list(current.keys()) + list(baseline.keys())
                       if k.endswith("_rewards")})

    for env in all_envs:
        key_r = f"{env}_rewards"
        key_p = f"{env}_pylons"
        key_s = f"{env}_steps"

        cur_has  = key_r in current
        base_has = key_r in baseline

        print(f"\n  [{env.upper()}]")
        print(f"  {'Metric':<22} {'Current':>12} {'Baseline':>12} {'Delta':>12}")
        print(f"  {'-'*58}")

        def row(label, cur_val, base_val, fmt=".1f"):
            if cur_val is None and base_val is None:
                return
            c_str = f"{cur_val:{fmt}}"  if cur_val  is not None else "  n/a"
            b_str = f"{base_val:{fmt}}" if base_val is not None else "  n/a"
            if cur_val is not None and base_val is not None:
                delta = cur_val - base_val
                sign  = "+" if delta >= 0 else ""
                d_str = f"{sign}{delta:{fmt}}"
            else:
                d_str = "  n/a"
            print(f"  {label:<22} {c_str:>12} {b_str:>12} {d_str:>12}")

        cur_avg_r  = float(np.nansum(current[key_r],  axis=1).mean())  if cur_has  else None
        base_avg_r = float(np.nansum(baseline[key_r], axis=1).mean())  if base_has else None
        cur_avg_p  = float(current[key_p].mean())   if cur_has  and key_p in current  else None
        base_avg_p = float(baseline[key_p].mean())  if base_has and key_p in baseline else None
        cur_avg_s  = float(current[key_s].mean())   if cur_has  and key_s in current  else None
        base_avg_s = float(baseline[key_s].mean())  if base_has and key_s in baseline else None

        cur_n  = current[key_r].shape[0]  if cur_has  else 0
        base_n = baseline[key_r].shape[0] if base_has else 0

        row("episodes",       float(cur_n),   float(base_n), fmt=".0f")
        row("avg total reward", cur_avg_r,  base_avg_r)
        row("avg pylons passed", cur_avg_p, base_avg_p, fmt=".2f")
        row("avg steps",      cur_avg_s,    base_avg_s, fmt=".0f")

        # Per-episode rewards
        if cur_has:
            ep_rewards = np.nansum(current[key_r], axis=1)
            print(f"  {'per-ep rewards':<22} "
                  f"{' '.join(f'{v:.0f}' for v in ep_rewards):>36}")
        if base_has:
            ep_rewards = np.nansum(baseline[key_r], axis=1)
            print(f"  {'baseline per-ep':<22} "
                  f"{' '.join(f'{v:.0f}' for v in ep_rewards):>36}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: train agent → evaluate in mock/ROS2 → save log"
    )
    parser.add_argument(
        "--agent", required=True, metavar="FILE",
        help="Path to agent .py file (e.g. no_ros2/agents/example_agent.py)"
    )
    parser.add_argument("--train",         type=int, default=400,            help="Training episodes (default 400, 0 = skip)")
    parser.add_argument("--load-policy",   metavar="FILE", default=None,     help="Load saved .npz Q-table, skip training")
    parser.add_argument("--save-policy",   metavar="FILE", default="policy.npz", help="Save Q-table after training (default policy.npz)")
    parser.add_argument("--env",           metavar="FILE", default=None,     help="Path to environment .py file (e.g. no_ros2/environments_v2/mock_pylon_env.py). Defaults to the factory default.")
    parser.add_argument("--mock-episodes", type=int, default=5,              help="Evaluation episodes in mock env (default 5)")
    parser.add_argument("--ros2-episodes", type=int, default=0,              help="Evaluation episodes in ROS2 env (default 0)")
    parser.add_argument("--course",        default="sample", choices=["sample", "purt"])
    parser.add_argument("--log",           default=None,                      help="Output log file. Defaults to logs/<agent>_<env>_<timestamp>.npz")
    parser.add_argument("--compare",       metavar="FILE",   default=None,    help="Compare results against a previous log .npz file")
    args = parser.parse_args()

    # Build default log path: logs/<agent>_<env>_<timestamp>.npz
    if args.log is None:
        logs_dir  = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        agent_stem = os.path.splitext(os.path.basename(args.agent))[0]
        if args.env:
            env_stem = os.path.splitext(os.path.basename(args.env))[0]
        elif args.ros2_episodes > 0 and args.mock_episodes == 0:
            env_stem = "ros2"
        else:
            env_stem = "mock_pylon_env"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = os.path.join(logs_dir, f"{agent_stem}_{env_stem}_{timestamp}.npz")

    # Load agent module
    print(f"Loading agent: {args.agent}")
    agent_mod = load_agent_module(args.agent)

    course_data = get_course(args.course)
    print(f"Course: {args.course} ({len(course_data['pylons'])} pylons)\n")

    # Build agent instance
    agent = agent_mod.QLearningAgent(
        alpha=0.15, gamma=0.98, epsilon=0.3, epsilon_min=0.05, epsilon_decay=0.997
    )

    # Load or train
    if args.load_policy:
        agent.load_policy(args.load_policy)
    elif args.train > 0:
        print(f"--- Training: {args.train} episodes in mock env ---")
        train_env = make_pylon_env(use_ros2=False, course=args.course)
        for ep in range(args.train):
            r, passed, steps = agent_mod.run_episode(
                train_env, agent, course_data, train=True
            )
            if (ep + 1) % 50 == 0 or ep == 0:
                print(f"  ep {ep+1}: reward={r:.1f}  pylons={passed}  "
                      f"steps={steps}  eps={agent.epsilon:.3f}")
        train_env.close()
        print("Training complete.\n")
        agent.save_policy(args.save_policy)

    save_dict = {}

    # ROS2 init (only if needed)
    if args.ros2_episodes > 0:
        import rclpy
        rclpy.init()

    # --- Mock evaluation ---
    if args.mock_episodes > 0:
        if args.env:
            env_label = os.path.basename(args.env)
            print(f"--- Evaluating {args.mock_episodes} episode(s) in {env_label} ---")
            mock_env = make_env_from_file(args.env, args.course)
        else:
            print(f"--- Evaluating {args.mock_episodes} episode(s) in MOCK env ---")
            mock_env = make_pylon_env(use_ros2=False, course=args.course)
        results  = evaluate(mock_env, agent_mod, agent, course_data, args.mock_episodes, "mock")
        mock_env.close()
        obs_arr, act_arr, rew_arr = pad_episodes(results, MAX_STEPS)
        save_dict["mock_obs"]     = obs_arr
        save_dict["mock_actions"] = act_arr
        save_dict["mock_rewards"] = rew_arr
        save_dict["mock_pylons"]  = np.array([r[3] for r in results], dtype=np.int32)
        save_dict["mock_steps"]   = np.array([r[4] for r in results], dtype=np.int32)

    # --- ROS2 evaluation ---
    if args.ros2_episodes > 0:
        print(f"--- Evaluating {args.ros2_episodes} episode(s) in ROS2 env ---")
        ros2_env = make_pylon_env(use_ros2=True, course=args.course)
        results  = evaluate(ros2_env, agent_mod, agent, course_data, args.ros2_episodes, "ros2")
        ros2_env.close()
        rclpy.shutdown()
        obs_arr, act_arr, rew_arr = pad_episodes(results, MAX_STEPS)
        save_dict["ros2_obs"]     = obs_arr
        save_dict["ros2_actions"] = act_arr
        save_dict["ros2_rewards"] = rew_arr
        save_dict["ros2_pylons"]  = np.array([r[3] for r in results], dtype=np.int32)
        save_dict["ros2_steps"]   = np.array([r[4] for r in results], dtype=np.int32)

    if save_dict:
        np.savez(args.log, **save_dict)
        print(f"Log saved → {args.log}  (keys: {list(save_dict.keys())})")

    if args.compare:
        if not os.path.isfile(args.compare):
            print(f"WARNING: --compare file not found: {args.compare}")
        elif not save_dict:
            print("WARNING: no current results to compare (run some episodes first).")
        else:
            baseline = dict(np.load(args.compare))
            compare_logs(save_dict, baseline)


if __name__ == "__main__":
    main()
