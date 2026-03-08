#!/usr/bin/env python3
"""Run the same PD altitude-hold agent as scripts/test_pylon_gym.py but against the mock env (no ROS2/Gazebo)."""
import sys
import os
# Workspace root and ros2_ws ROS package root so no_ros2 and auav_pylon_2026 are both importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_script_dir, '..'))
_auav_pkg = os.path.abspath(os.path.join(_root, 'ros2_ws', 'src', 'auav_pylon_2026'))
sys.path.insert(0, _root)
if os.path.isdir(_auav_pkg):
    sys.path.insert(0, _auav_pkg)

import numpy as np
from no_ros2.environments.env_factory import make_pylon_env


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pylon gym PD altitude-hold test (mock env)")
    parser.add_argument("--viz", action="store_true", help="Show 3D visualization (pylons + UAV)")
    parser.add_argument("--show-details", action="store_true", help="Show full state vector in viz (use with --viz)")
    args = parser.parse_args()

    print("Using Mock Pylon Environment (no ROS2)...")
    env = make_pylon_env(use_ros2=False)
    viz = None
    if args.viz:
        from no_ros2.viz_3d import PylonRacingViz3D
        viz = PylonRacingViz3D(show_state_vector=args.show_details)

    print("Testing Reset...")
    obs, info = env.reset()
    if viz is not None:
        viz.update(obs, reward=0.0, step_count=0, laps=0)

    target_alt = 7.0
    print(f"Initiating Takeoff & Hold at {target_alt}m (Press Ctrl+C to stop)...")

    current_elevator = -0.02
    step_count = 0

    try:
        while True:
            current_alt = obs[2]
            vz = obs[8]
            current_speed = np.sqrt(obs[6]**2 + obs[7]**2 + obs[8]**2)

            aileron = 0.0
            rudder = 0.0

            alt_error = target_alt - current_alt

            if current_speed < 4.0:
                target_elevator = -0.02
                throttle = 1.0
            elif current_speed >= 4.0 and current_alt < 2.0:
                target_elevator = 0.12
                throttle = 1.0
            else:
                target_elevator = np.clip((alt_error * 0.08) - (vz * 0.1), -0.25, 0.15)
                if alt_error > 1.0:
                    throttle = 0.8
                elif alt_error < -1.0:
                    throttle = 0.1
                else:
                    throttle = 0.45

            current_elevator += np.clip(target_elevator - current_elevator, -0.01, 0.01)

            action = np.array([aileron, current_elevator, throttle, rudder], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if viz is not None:
                viz.update(obs, reward=reward, step_count=step_count, laps=0)

            if step_count % 10 == 0:
                print(f"Step {step_count} | Speed: {current_speed:.2f} m/s | Alt: {current_alt:.2f} m | Elev: {current_elevator:.3f} | Thr: {throttle:.2f}")

            if terminated:
                print("Crashed! Resetting...")
                obs, info = env.reset()
                if viz is not None:
                    viz.clear_trail()
                    viz.update(obs, reward=0.0, step_count=step_count, laps=0)
                current_elevator = -0.02

            step_count += 1

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down test script...")
    finally:
        if viz is not None:
            viz.close()
        env.close()


if __name__ == '__main__':
    main()
