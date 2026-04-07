#!/usr/bin/env python3
"""
waypoint_agent.py
-----------------
Planning agent that generates waypoints from the pylon course and follows them
with pursuit guidance + PD inner-loop controller.

No learning required -- the agent computes a geometric flight plan from the
known course layout and tracks it in real time.

Architecture:
  1. Planner: builds ordered waypoints with approach points for smooth gate
     transitions.
  2. Guidance: proportional pursuit steers toward each waypoint in sequence.
  3. Inner-loop PD: converts attitude targets to raw control surfaces.

Run:
  python no_ros2/agents/waypoint_agent.py --viz
  python no_ros2/agents/waypoint_agent.py --purt --viz --laps 3
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use starting altitude as target (aircraft resets at 7m).
# Descending to 3.5m during flight causes altitude-control issues at bank.
TARGET_ALT_M   = 7.0
CRUISE_SPEED   = 7.0
MAX_STEPS      = 6000  # 6000 * 0.02 = 120 s per episode

# Waypoint acceptance radius
WP_ACCEPT_RADIUS = 14.0

# Approach waypoints: how far upstream of each gate to place them
APPROACH_OFFSET  = 12.0

# Inner-loop PD gains (same as physics_agent)
_KP_PITCH = 0.08
_KD_PITCH = 0.5
_KP_ROLL  = 0.04
_KD_ROLL  = 0.3
_KP_YAW   = 0.02

# Guidance limits
_HEADING_GAIN    = 2.0    # bank deg per deg heading error
_MAX_BANK_DEG    = 30.0   # max bank angle
_ALT_KP          = 3.0
_ALT_MAX_PITCH   = 8.0
TRIM_PITCH_DEG   = 2.0
TRIM_THROTTLE    = 0.35


# ---------------------------------------------------------------------------
# Waypoint planner
# ---------------------------------------------------------------------------

def plan_waypoints(course_data):
    """
    Build ordered waypoints: for each gate, an approach point and the gate
    midpoint.  Approach points create smooth heading transitions between gates.

    Returns list of (x, y, z) numpy arrays.
    """
    pylons = course_data["pylons"]
    gates  = course_data["gates"]
    n_gates = len(gates)

    # First compute all gate midpoints
    mids = []
    for p_a, p_b in gates:
        mids.append((pylons[p_a][:2] + pylons[p_b][:2]) / 2.0)

    waypoints = []
    for i in range(n_gates):
        mid = mids[i]
        prev_mid = mids[(i - 1) % n_gates]

        # Direction from previous gate to this gate
        approach_dir = mid - prev_mid
        dist = np.linalg.norm(approach_dir)
        if dist > 1e-3:
            approach_dir /= dist
        else:
            approach_dir = np.array([1.0, 0.0])

        # Approach point: upstream of the gate midpoint
        offset = min(APPROACH_OFFSET, dist * 0.35)
        approach = mid - approach_dir * offset
        waypoints.append(np.array([approach[0], approach[1], TARGET_ALT_M]))
        waypoints.append(np.array([mid[0], mid[1], TARGET_ALT_M]))

    return waypoints


# ---------------------------------------------------------------------------
# Inner-loop PD controller
# ---------------------------------------------------------------------------

def attitude_to_controls(target_roll_deg, target_pitch_deg, throttle, obs):
    """Convert attitude targets + current obs to raw (ail, elev, thr, rud)."""
    roll_deg  = np.degrees(obs[3])
    pitch_deg = np.degrees(obs[4])
    p, q, r   = obs[12], obs[13], obs[14]

    # Bank-pitch compensation
    roll_rad = np.radians(roll_deg)
    cos_roll = max(np.cos(roll_rad), 0.3)
    bank_pitch_comp = (1.0 / cos_roll - 1.0) * 6.0  # increased from 4.0

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
# Guidance
# ---------------------------------------------------------------------------

def pursuit_guidance(obs, wp_target):
    """
    Direct pursuit guidance toward waypoint.

    Returns (target_roll_deg, target_pitch_deg, throttle).
    """
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]
    V       = max(obs[9], 1.0)

    dx = wp_target[0] - x
    dy = wp_target[1] - y

    desired_heading = np.arctan2(dy, dx)
    current_heading = np.arctan2(vy, vx)

    heading_err = desired_heading - current_heading
    heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
    heading_err_deg = np.degrees(heading_err)

    target_roll_deg = np.clip(
        _HEADING_GAIN * heading_err_deg, -_MAX_BANK_DEG, _MAX_BANK_DEG
    )

    # Altitude hold
    alt_err = wp_target[2] - z
    pitch_correction = np.clip(_ALT_KP * alt_err, -_ALT_MAX_PITCH, _ALT_MAX_PITCH)
    target_pitch_deg = TRIM_PITCH_DEG + pitch_correction

    # Speed hold + bank compensation: increase throttle during banking
    # to generate enough lift for altitude maintenance
    speed_err = CRUISE_SPEED - V
    bank_comp = 0.1 * abs(target_roll_deg) / max(_MAX_BANK_DEG, 1.0)
    throttle = TRIM_THROTTLE + 0.1 * speed_err + bank_comp

    return target_roll_deg, target_pitch_deg, throttle


def should_advance_waypoint(obs, wp_curr, prev_dist):
    """Advance when within acceptance radius or closest approach is receding."""
    pos = np.array([obs[0], obs[1]])
    dist = np.linalg.norm(wp_curr[:2] - pos)

    if dist < WP_ACCEPT_RADIUS:
        return True, dist

    # Closest approach: if we were within 1.5x acceptance and now receding
    if prev_dist is not None and prev_dist < WP_ACCEPT_RADIUS * 1.5 and dist > prev_dist + 0.3:
        return True, dist

    return False, dist


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, course_data, max_steps=MAX_STEPS, viz=None,
                max_laps=None, show_details=False):
    pylons      = course_data["pylons"]
    n_pylons    = len(pylons)

    waypoints = plan_waypoints(course_data)
    n_wp = len(waypoints)

    obs, _ = env.reset()

    # Pick the initial waypoint most aligned with current heading
    vx, vy = obs[6], obs[7]
    current_heading = np.arctan2(vy, vx)
    best_idx, best_score = 0, -1e9
    for i, wp in enumerate(waypoints):
        dx, dy = wp[0] - obs[0], wp[1] - obs[1]
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        wp_heading = np.arctan2(dy, dx)
        err = abs((wp_heading - current_heading + np.pi) % (2*np.pi) - np.pi)
        # Score: prefer small heading error AND reasonable distance
        score = -err - dist * 0.02
        if score > best_score:
            best_score = score
            best_idx = i
    wp_idx = best_idx

    total_reward = 0.0
    gates_passed = 0
    laps = 0
    wp_prev_dist = None

    if viz:
        viz.clear_trail()

    for step in range(max_steps):
        wp_curr = waypoints[wp_idx]

        target_roll, target_pitch, throttle = pursuit_guidance(obs, wp_curr)
        action = attitude_to_controls(target_roll, target_pitch, throttle, obs)

        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        advanced, wp_prev_dist = should_advance_waypoint(obs, wp_curr, wp_prev_dist)
        if advanced:
            wp_prev_dist = None
            old_idx = wp_idx
            wp_idx = (wp_idx + 1) % n_wp
            # Every 2 waypoints = 1 gate (approach + midpoint)
            if wp_idx % 2 == 0:
                gates_passed += 1
                if gates_passed % n_pylons == 0:
                    laps += 1

        if terminated and r_env < 0:
            total_reward += r_env
        elif obs[2] > 0.5:
            total_reward += 1.0

        if viz:
            viz.update(obs, reward=r_env, step_count=step, laps=laps)

        if show_details and step % 200 == 0:
            dist = np.linalg.norm(wp_curr[:2] - np.array([obs[0], obs[1]]))
            wp_type = "app" if wp_idx % 2 == 0 else "gate"
            print(f"    step={step:4d}  pos=({obs[0]:6.1f},{obs[1]:6.1f},{obs[2]:5.1f})  "
                  f"wp={wp_idx}/{n_wp}({wp_type})  dist={dist:5.1f}  "
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
    parser = argparse.ArgumentParser(description="Waypoint planning agent for pylon racing")
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
    print(f"Planned {len(waypoints)} waypoints ({len(waypoints)//2} gates)")
    for i, wp in enumerate(waypoints):
        kind = "approach" if i % 2 == 0 else "gate   "
        print(f"  WP {i:2d} ({kind}): ({wp[0]:7.2f}, {wp[1]:7.2f}, {wp[2]:4.1f})")

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
        laps = gates_passed // len(course_data["pylons"])
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
