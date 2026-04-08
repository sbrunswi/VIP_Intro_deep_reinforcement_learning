#!/usr/bin/env python3
"""
waypoint_agent_v2.py
--------------------
Planning agent that uses XTrack_NAV_lookAhead (from cross_tracker_nav_sample.py)
to follow waypoints around the pylon course.  This matches the ROS2 pipeline
pattern in sim_tecs_ros_xtrack.py.

Architecture:
  1. Planner: builds ordered waypoints (gate midpoints) from the course.
  2. Navigator: XTrack_NAV_lookAhead instance computes desired velocity,
     flight-path angle, and heading via cross-track vector-field guidance.
  3. Heading-to-bank bridge: converts desired heading to a bank angle command.
  4. Inner-loop PD: converts (roll, pitch, throttle) to raw control surfaces.

Run:
  python no_ros2/agents/waypoint_agent_v2.py --viz
  python no_ros2/agents/waypoint_agent_v2.py --purt --viz --laps 3
  python no_ros2/agents/waypoint_agent_v2.py --sample --viz --show-details
"""

import os
import sys
import importlib.util

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _root)

import numpy as np

from no_ros2.environments.env_factory import make_pylon_env
from no_ros2.environments.pylon_course import get_course, PYLON_MID_HEIGHT_M
from no_ros2.environments.pylon_wrapper import PylonRacingWrapper
from no_ros2.environments_v2.cross_tracker_nav_sample import XTrack_NAV_lookAhead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_ALT_M   = 7.0     # target altitude (aircraft resets at 7m)
CRUISE_SPEED   = 7.0     # m/s
MAX_STEPS      = 6000    # 6000 * 0.02 = 120 s per episode
ENV_DT         = 0.02    # environment timestep (s)

# Inner-loop PD gains (same as physics_agent)
_KP_PITCH = 0.08
_KD_PITCH = 0.5
_KP_ROLL  = 0.04
_KD_ROLL  = 0.3
_KP_YAW   = 0.02

# Heading-to-bank bridge
_HEADING_GAIN  = 2.0     # bank deg per deg heading error
_MAX_BANK_DEG  = 30.0    # max bank angle
TRIM_PITCH_DEG = 2.0
TRIM_THROTTLE  = 0.35


# ---------------------------------------------------------------------------
# Waypoint planner
# ---------------------------------------------------------------------------

def plan_waypoints(course_data):
    """
    Build ordered waypoints from the course gate midpoints.
    Each gate becomes one waypoint at TARGET_ALT_M.
    Appends the first waypoint at the end to close the loop,
    giving XTrack_NAV_lookAhead a valid segment from last gate back to first.

    Returns list of (x, y, z) tuples.
    """
    pylons = course_data["pylons"]
    gates  = course_data["gates"]

    waypoints = []
    for p_a, p_b in gates:
        mid = (pylons[p_a][:2] + pylons[p_b][:2]) / 2.0
        waypoints.append((mid[0], mid[1], TARGET_ALT_M))

    waypoints.append(waypoints[0])  # close the loop
    return waypoints


# ---------------------------------------------------------------------------
# Heading-to-bank bridge + inner-loop PD controller
# ---------------------------------------------------------------------------

def heading_to_attitude(obs, des_heading, des_gamma, des_v):
    """
    Convert guidance outputs (des_heading, des_gamma, des_v) into
    attitude commands (target_roll_deg, target_pitch_deg, throttle).
    """
    vx, vy = obs[6], obs[7]
    V      = max(obs[9], 1.0)

    # Heading error -> bank angle
    current_heading = np.arctan2(vy, vx)
    heading_err = des_heading - current_heading
    heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
    heading_err_deg = np.degrees(heading_err)

    target_roll_deg = np.clip(
        _HEADING_GAIN * heading_err_deg, -_MAX_BANK_DEG, _MAX_BANK_DEG
    )

    # Flight path angle -> pitch
    # des_gamma is the desired climb/descend angle; map to pitch command
    target_pitch_deg = TRIM_PITCH_DEG + np.clip(
        np.degrees(des_gamma), -8.0, 8.0
    )

    # Speed hold + bank compensation
    speed_err = des_v - V
    bank_comp = 0.1 * abs(target_roll_deg) / max(_MAX_BANK_DEG, 1.0)
    throttle = TRIM_THROTTLE + 0.1 * speed_err + bank_comp

    return target_roll_deg, target_pitch_deg, throttle


def attitude_to_controls(target_roll_deg, target_pitch_deg, throttle, obs):
    """Convert attitude targets + current obs to raw (ail, elev, thr, rud)."""
    roll_deg  = np.degrees(obs[3])
    pitch_deg = np.degrees(obs[4])
    p, q, r   = obs[12], obs[13], obs[14]

    # Bank-pitch compensation
    roll_rad = np.radians(roll_deg)
    cos_roll = max(np.cos(roll_rad), 0.3)
    bank_pitch_comp = (1.0 / cos_roll - 1.0) * 6.0

    # Pitch PD
    pitch_err = (target_pitch_deg + bank_pitch_comp) - pitch_deg
    elevator  = np.clip(_KP_PITCH * pitch_err - _KD_PITCH * q, -1.0, 1.0)

    # Roll PD
    roll_err = target_roll_deg - roll_deg
    aileron  = np.clip(_KP_ROLL * roll_err - _KD_ROLL * p, -1.0, 1.0)

    # Coordinated turn rudder
    rudder = np.clip(_KP_YAW * roll_deg, -1.0, 1.0)

    throttle = np.clip(throttle, 0.0, 1.0)

    return np.array([aileron, elevator, throttle, rudder], dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _create_navigator(waypoints, start_idx):
    """Instantiate XTrack_NAV_lookAhead and configure for mock env."""
    nav = XTrack_NAV_lookAhead(ENV_DT, waypoints, start_idx)
    nav.v_cruise = CRUISE_SPEED  # override default 10.0 -> 7.0
    # When starting at index 0, prev_wpt defaults to (0,0,0) which is wrong.
    # Set it to the actual last gate (before the closing duplicate).
    if start_idx == 0:
        nav.prev_wpt = waypoints[-2]
    elif start_idx > 0:
        nav.prev_wpt = waypoints[start_idx - 1]
    return nav


def run_episode(env, course_data, max_steps=MAX_STEPS, viz=None,
                max_laps=None, show_details=False):
    waypoints = plan_waypoints(course_data)
    n_wp = len(waypoints)

    obs, _ = env.reset()

    # Pick the initial waypoint most aligned with current heading
    # (skip the closing duplicate at index n_wp-1)
    vx, vy = obs[6], obs[7]
    current_heading = np.arctan2(vy, vx)
    best_idx, best_score = 0, -1e9
    for i in range(n_wp - 1):
        wp = waypoints[i]
        dx, dy = wp[0] - obs[0], wp[1] - obs[1]
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        wp_heading = np.arctan2(dy, dx)
        err = abs((wp_heading - current_heading + np.pi) % (2*np.pi) - np.pi)
        score = -err - dist * 0.02
        if score > best_score:
            best_score = score
            best_idx = i

    # Instantiate navigator (matching ROS2 sim_tecs_ros_xtrack.py pattern)
    nav = _create_navigator(waypoints, best_idx)

    total_reward = 0.0
    gates_passed = 0
    laps = 0

    if viz:
        viz.clear_trail()

    for step in range(max_steps):
        # Navigator guidance
        V_array = [obs[6], obs[7], obs[8]]
        des_v, des_gamma, des_heading, along_err, xtrack_err = \
            nav.wp_tracker(waypoints, obs[0], obs[1], obs[2], V_array)

        # Convert guidance outputs to attitude commands
        target_roll, target_pitch, throttle = \
            heading_to_attitude(obs, des_heading, des_gamma, des_v)

        # Inner-loop PD -> raw controls
        action = attitude_to_controls(target_roll, target_pitch, throttle, obs)

        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Waypoint switching via check_arrived (matching ROS2 pattern)
        prev_wp_idx = nav.current_WP_ind
        nav.check_arrived(along_err, V_array)

        if nav.current_WP_ind > prev_wp_idx:
            gates_passed += 1

        # Looping: reinitialize when past last waypoint
        # (matching sim_tecs_ros_xtrack.py lines 399-417)
        if nav.current_WP_ind >= len(waypoints):
            laps += 1
            nav = _create_navigator(waypoints, 0)

        # Reward tracking
        if terminated and r_env < 0:
            total_reward += r_env
        elif obs[2] > 0.5:
            total_reward += 1.0

        if viz:
            viz.update(obs, reward=r_env, step_count=step, laps=laps)

        if show_details and step % 200 == 0:
            wp = waypoints[nav.current_WP_ind] if nav.current_WP_ind < n_wp else waypoints[0]
            dist = np.sqrt((wp[0]-obs[0])**2 + (wp[1]-obs[1])**2)
            print(f"    step={step:4d}  pos=({obs[0]:6.1f},{obs[1]:6.1f},{obs[2]:5.1f})  "
                  f"wp={nav.current_WP_ind}/{n_wp}  dist={dist:5.1f}  "
                  f"along={along_err:5.1f}  xtrack={xtrack_err:+5.1f}  "
                  f"roll={target_roll:5.1f}  V={obs[9]:.1f}  gates={gates_passed}")

        if done:
            break
        if max_laps and laps >= max_laps:
            break

    return total_reward, gates_passed, step + 1


# ---------------------------------------------------------------------------
# Dynamic env loader
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
    parser = argparse.ArgumentParser(
        description="Waypoint agent v2 — cross-track vector-field guidance")
    parser.add_argument("--viz",           action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--show-details",  action="store_true")
    parser.add_argument("--env",           metavar="FILE", default=None)
    parser.add_argument("--ros2",          action="store_true")
    parser.add_argument("--laps",          type=int, default=None)
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument("--purt",   action="store_true")
    course_group.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    course_name = "purt" if args.purt else "sample"
    course_data = get_course(course_name)
    print(f"Course: {course_name} ({len(course_data['pylons'])} pylons)")

    waypoints = plan_waypoints(course_data)
    n_gates = len(course_data["gates"])
    print(f"Planned {len(waypoints)} waypoints ({n_gates} gates + closing)")
    for i, wp in enumerate(waypoints):
        label = "close" if i == len(waypoints) - 1 else f"gate {i}"
        print(f"  WP {i} ({label}): ({wp[0]:7.2f}, {wp[1]:7.2f}, {wp[2]:4.1f})")

    if args.ros2:
        import rclpy
        rclpy.init()

    if args.env:
        print(f"Env: {os.path.basename(args.env)}")
        eval_env = _make_env_from_file(args.env, course_name)
    elif args.ros2:
        eval_env = make_pylon_env(use_ros2=True, course=course_name)
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

    print(f"\n--- Evaluation: {args.eval_episodes} episodes ---")
    total_gates = 0
    for ep in range(args.eval_episodes):
        reward, gates_passed, steps = run_episode(
            eval_env, course_data, max_steps=MAX_STEPS, viz=viz,
            max_laps=args.laps, show_details=args.show_details,
        )
        total_gates += gates_passed
        laps = gates_passed // (n_gates + 1)  # +1 for closing waypoint
        print(f"  Episode {ep+1}: reward={reward:7.1f}  gates={gates_passed}  "
              f"laps={laps}  steps={steps}")

    avg = total_gates / args.eval_episodes
    print(f"\nAvg gates/episode: {avg:.1f}")

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
