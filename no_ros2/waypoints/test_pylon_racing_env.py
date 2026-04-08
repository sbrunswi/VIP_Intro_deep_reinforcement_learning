import numpy as np
from pylon_racing_env import PylonRacingEnv

def test_autopilot():
    # Initialize the passthrough environment
    env = PylonRacingEnv()
    obs, info = env.reset()
    
    done = False
    print("Taking off...")
    
    while not done:
        # Pass a dummy action (it will be ignored)
        dummy_action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(dummy_action)
        
        # Print progress every 100 steps (1 second of sim time)
        if info["step"] % 100 == 0:
            print(f"Time: {info['sim_time']:5.1f}s | "
                  f"Phase: {info['flight_phase']:>8} | "
                  f"Heading to WP: {info['wp_idx']} | "
                  f"Alt: {info['actual_data']['z_est']:5.1f}m | "
                  f"Speed: {info['actual_data']['v_est']:4.1f}m/s | "
                  f"WP0 Bearing: {obs[4]:5.2f} rad")
            
        done = terminated or truncated

    print(f"Episode ended at {info['sim_time']:.1f} seconds.")
    if terminated:
        print(f"Termination reason: Crash or Out of Bounds. Final Alt: {info['actual_data']['z_est']:.1f}m")
    elif truncated:
        print("Termination reason: Reached maximum episode steps (success!).")

if __name__ == "__main__":
    test_autopilot()