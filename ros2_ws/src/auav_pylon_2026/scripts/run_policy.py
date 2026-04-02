#!/usr/bin/env python3
import rclpy
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Ensure the environment module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from auav_pylon_2026.pylon_env import PylonRacingEnv

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def run_policy():
    rclpy.init()
    env = PylonRacingEnv()
    
    state_dim = 15
    action_dim = 7
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(state_dim, action_dim).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), 'dqn_pylon.pth')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please wait for training to finish!")
        env.node.destroy_node()
        rclpy.shutdown()
        return
        
    print(f"Loading weights from {model_path}...")
    q_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    q_net.eval()
    
    print("Running Policy... Check RViz2 to see the agent fly!")
    
    try:
        while rclpy.ok():
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done and rclpy.ok():
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_net(state_t)
                    # argmax for pure exploitation
                    action = q_values.argmax().item()
                
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                total_reward += reward
                
                # Small sleep if you want to run exactly at 10Hz, 
                # but rclpy.spin_once inside env.step already handles loop rate mostly
                time.sleep(0.01)
                
            print(f"Episode Finished! Total Reward accumulated: {total_reward:.2f}")
    except KeyboardInterrupt:
        print("Stopping policy execution.")
    finally:
        try:
            env.node.destroy_node()
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    run_policy()
