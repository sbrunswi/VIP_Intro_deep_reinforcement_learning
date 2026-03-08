#!/usr/bin/env python3
import rclpy
import numpy as np
import casadi as ca

# NEW IMPORTS: For RViz2 Path Visualization
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from auav_pylon_2026.pylon_env import PylonRacingEnv 
from auav_pylon_2026.tecs_controller_xtrack_sample import TECSControl_cub
from auav_pylon_2026.cross_tracker_nav_sample import XTrack_NAV_lookAhead

# Define the Race Track Waypoints
ALT = 7.0
CONTROL_POINT = [
    (-10.0, -5.0, ALT),
    (-30.0, -10.0, ALT),
    (-30.0, -40.0, ALT),
    (30.0, -30.0, ALT),
    (30.0, 5.0, ALT),
    (10.0, 5.0, ALT),
    (-10.0, -5.0, ALT),
]

def main():
    rclpy.init()
    print("Connecting to Pylon Racing Simulation...")
    env = PylonRacingEnv()
    
    # ---------------------------------------------------------
    # NEW: Setup RViz2 Path Publisher
    # We use the node already created by the Gym wrapper
    # ---------------------------------------------------------
    path_pub = env.node.create_publisher(Path, '/sim/ref_path', 10)
    
    # Pre-build the static path message
    ref_path_msg = Path()
    ref_path_msg.header.frame_id = 'map' # Standard RViz frame
    
    for wp in CONTROL_POINT:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(wp[0])
        pose.pose.position.y = float(wp[1])
        pose.pose.position.z = float(wp[2])
        ref_path_msg.poses.append(pose)
    # ---------------------------------------------------------

    print("Testing Reset...")
    obs, info = env.reset()
    
    dt = 0.1  
    time_elapsed = 0.0
    tecs_control = TECSControl_cub(dt, "sim")
    
    current_WP_ind = 0
    last_WP_ind = len(CONTROL_POINT)
    wpt_planner = XTrack_NAV_lookAhead(dt, CONTROL_POINT, current_WP_ind)
    
    wpt_planner.path_distance_buf = 5.0  
    
    # ---------------------------------------------------------
    # FIX: Increase switching distance to stop the jerking!
    # ---------------------------------------------------------
    wpt_planner.wpt_switching_distance = 15.0  
    wpt_planner.v_cruise = 10.0  
    
    flight_mode = "takeoff"
    end_cruise = False
    
    throttle = 0.7
    elev = 0.0
    aileron = 0.0
    rudder = 0.0
    
    step_count = 0

    try:
        while rclpy.ok():
            actual_data = {
                "x_est": obs[0], "y_est": obs[1], "z_est": obs[2],
                "roll_est": obs[3], "pitch_est": obs[4], "yaw_est": obs[5],
                "vx_est": obs[6], "vy_est": obs[7], "vz_est": obs[8],
                "v_est": obs[9], "gamma_est": obs[10], "vdot_est": obs[11],
                "p_est": obs[12], "q_est": obs[13], "r_est": obs[14],
            }

            time_elapsed += dt
            step_idx = int(time_elapsed / dt)

            if actual_data["z_est"] <= 1.0 and not end_cruise:
                flight_mode = "takeoff"
            else:
                flight_mode = "airborne"

            if flight_mode == "takeoff":
                throttle = float(ca.fmin(1.0, ca.fmax(0.7, throttle + 2.0 * dt)))
                rudder = 0.0
                aileron = 0.0
                
                v_to = 0.5  
                e_down = -0.02  
                e_up = 0.15  
                e_rate = 0.40  
                elev = float(ca.if_else(actual_data["v_est"] < v_to, e_down, ca.fmin(e_up, elev + e_rate * dt)))

            elif flight_mode == "airborne":
                if current_WP_ind == last_WP_ind:
                    current_WP_ind = 0
                    end_cruise = False
                    wpt_planner = XTrack_NAV_lookAhead(dt, CONTROL_POINT, current_WP_ind)
                else:
                    v_array = [actual_data["vx_est"], actual_data["vy_est"], actual_data["vz_est"]]
                    
                    des_v, des_gamma, des_heading, along_track_err, cross_track_err = wpt_planner.wp_tracker(
                        CONTROL_POINT, actual_data["x_est"], actual_data["y_est"], actual_data["z_est"], v_array, verbose=False
                    )

                    des_a = 1.0 * (des_v - np.abs(actual_data["v_est"]))
                    
                    ref_data = {
                        "des_v": des_v, 
                        "des_gamma": des_gamma, 
                        "des_heading": des_heading, 
                        "des_a": des_a
                    }

                    aileron, elev, throttle, rudder = tecs_control.compute_control(
                        step_idx, ref_data, actual_data
                    )
                    
                    current_WP_ind = wpt_planner.check_arrived(along_track_err, v_array, verbose=False)

            action = np.array([aileron, elev, throttle, rudder], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # ---------------------------------------------------------
            # NEW: Publish the RViz2 Path every 10 steps
            # ---------------------------------------------------------
            if step_count % 10 == 0:
                # Update the timestamp so RViz knows it's current
                ref_path_msg.header.stamp = env.node.get_clock().now().to_msg()
                path_pub.publish(ref_path_msg)
                
                print(f"Step {step_count} | Mode: {flight_mode.upper()} | WP: {current_WP_ind}/{last_WP_ind} | Alt: {actual_data['z_est']:.2f}m | Spd: {actual_data['v_est']:.2f}m/s")

            if terminated:
                print("Crash detected! Resetting simulation...")
                obs, info = env.reset()
                
                time_elapsed = 0.0
                current_WP_ind = 0
                wpt_planner = XTrack_NAV_lookAhead(dt, CONTROL_POINT, current_WP_ind)
                throttle = 0.7
                elev = 0.0

            step_count += 1

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down test script...")
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()