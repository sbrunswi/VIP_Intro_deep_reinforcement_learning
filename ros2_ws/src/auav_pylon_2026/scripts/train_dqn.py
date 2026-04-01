#!/usr/bin/env python3
import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32)
        
    def __len__(self):
        return len(self.buffer)

def train():
    rclpy.init()
    env = PylonRacingEnv()
    
    state_dim = 15
    action_dim = 7
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(10000)
    
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 5000
    
    num_episodes = 500
    global_step = 0
    
    print("Starting Deep Q-Learning Training for Pylon Course...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_net(state_t)
                    action = q_values.argmax().item()
                    
            # Environment step
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            global_step += 1
            
            # Optimization step
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                curr_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * gamma * next_q
                    
                loss = nn.MSELoss()(curr_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Update target network
            if global_step % 100 == 0:
                target_net.load_state_dict(q_net.state_dict())
                
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}, Gate Idx Check: {env.current_gate_idx}")
        
    print("Training finished. Saving model weight to dqn_pylon.pth...")
    torch.save(q_net.state_dict(), "dqn_pylon.pth")
    
    # Optional cleanup, but keep it graceful
    try:
        env.node.destroy_node()
        rclpy.shutdown()
    except:
        pass

if __name__ == '__main__':
    train()
