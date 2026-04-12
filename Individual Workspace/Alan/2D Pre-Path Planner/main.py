"""
main.py
-------
2D Pre-Path Q-learning waypoint optimizer with a unified small-step
action space. Adjusts all intermediate waypoints simultaneously on a
1-foot grid, clamped to +/-5 ft from their heuristic positions.

Usage:
    python main.py                        # purt course, 500 episodes
    python main.py --course sample        # larger 6-pylon course
    python main.py --episodes 1000        # more training
    python main.py --speed 5              # matches task cruise velocity
"""

import os
import sys
import argparse

# Add project root so we can import the shared course data
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

import numpy as np
from no_ros2.environments.pylon_course import get_course

# The rules-aware env and heuristic live in the parent "Alan" dir.
sys.path.insert(0, os.path.abspath(os.path.join(_script_dir, "..")))
from waypoint_env import PointMassEnv2D
from heuristic import heuristic_waypoints
from planner import PrePathQLearner
from visualizer import plot_dashboard
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="2D pre-path Q-learning waypoint optimizer")
    parser.add_argument("--course", default="purt", choices=["purt", "sample"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--speed", type=float, default=5.0,
                        help="Cruise speed (m/s). Task default = 5.")
    parser.add_argument("--save-policy", metavar="FILE", default=None)
    parser.add_argument("--load-policy", metavar="FILE", default=None)
    args = parser.parse_args()

    # Load course
    course = get_course(args.course)
    pylons_xy = course["pylons"][:, :2]  # drop z
    bounds = course["bounds_rect"]
    gates = course["gates"]
    n_pylons = len(pylons_xy)
    print(f"Course: {args.course} ({n_pylons} pylons)")
    print(f"Speed: {args.speed} m/s | Episodes: {args.episodes}")

    # Create simulator
    env = PointMassEnv2D(pylons_xy, bounds, speed=args.speed)

    # --- Step 1: Heuristic waypoints ---
    base_wps = heuristic_waypoints(pylons_xy, bounds=bounds)
    result_before = env.simulate(base_wps)
    status = "OK" if result_before["completed"] else "FAIL"
    print(f"\nBefore optimization:")
    print(f"  Time: {result_before['total_time']:.1f}s  "
          f"Distance: {result_before['total_distance']:.1f}m  "
          f"Smoothness: {result_before['smoothness']:.2f}  [{status}]")

    # --- Step 2: Q-learning optimization ---
    n_wps = len(base_wps)
    planner = PrePathQLearner(n_waypoints=n_wps)

    if args.load_policy:
        planner.load_policy(args.load_policy)

    print(f"\n--- Pre-Path Q-Learning: {args.episodes} episodes ---")
    best_wps, rewards = planner.run_optimization(
        base_wps, pylons_xy, env, gates, n_episodes=args.episodes)

    if args.save_policy:
        planner.save_policy(args.save_policy)

    # --- Step 3: Evaluate optimized path ---
    result_after = env.simulate(best_wps)
    status = "OK" if result_after["completed"] else "FAIL"
    print(f"\nAfter optimization:")
    print(f"  Time: {result_after['total_time']:.1f}s  "
          f"Distance: {result_after['total_distance']:.1f}m  "
          f"Smoothness: {result_after['smoothness']:.2f}  [{status}]")

    if result_before["completed"] and result_after["completed"]:
        dt = result_before["total_time"] - result_after["total_time"]
        dd = result_before["total_distance"] - result_after["total_distance"]
        print(f"\n  Improvement: {dt:+.2f}s time, {dd:+.1f}m distance")

    # --- Step 4: Visualize ---
    plot_dashboard(pylons_xy, gates, base_wps, best_wps,
                   result_before, result_after, rewards, bounds)
    plt.show()


if __name__ == "__main__":
    main()
