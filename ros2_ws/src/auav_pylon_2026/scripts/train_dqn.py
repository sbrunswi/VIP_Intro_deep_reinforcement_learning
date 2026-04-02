#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
import sys
import os

# Import Pure CasADi physics (No ROS dependencies)
import casadi as ca
from tf_transformations import euler_from_quaternion

# We must ensure cyecca is reachable
sys.path.append(os.path.join(os.path.dirname(__file__), '../../cyecca'))
try:
    from cyecca.models import fixedwing_4ch
except ModuleNotFoundError:
    print("Could not import cyecca. Make sure you are in the correct environment.")
    sys.exit(1)


# ==========================================
# PURE PYTHON GYM ENVIRONMENT (1,000x Faster)
# ==========================================
class PurePythonPylonEnv:
    def __init__(self):
        # 1. Initialize CasADi model based exactly on fixedwing_sim.py
        self.dt = 0.01  # Internal 100Hz simulation rate
        self.control_steps = 10  # Agent runs at 10Hz (100Hz / 10)
        
        self.model = fixedwing_4ch.derive_model()
        
        self.p_dict = self.model["p_defaults"]
        self.p = np.array([self.p_dict[str(self.model["p"][i])] for i in range(self.model["p"].shape[0])], dtype=float)
        
        self.x0_dict = self.model["x0_defaults"]
        
        # ODE Integrator
        opts = {"abstol": 1e-2, "reltol": 1e-6, "fsens_err_con": True}
        self.integrator = ca.integrator("test", "cvodes", self.model["dae"], 0.0, self.dt, opts)
        
        # 2. RL Space Definitions
        self.discrete_action_map = {
            0: [ 0.0,  0.0, 0.7, 0.0], # Cruise
            1: [ 0.0,  0.5, 0.7, 0.0], # Pitch up
            2: [ 0.0, -0.5, 0.7, 0.0], # Pitch down
            3: [-0.5,  0.0, 0.7, 0.0], # Roll left
            4: [ 0.5,  0.0, 0.7, 0.0], # Roll right
            5: [ 0.0,  0.0, 1.0, 0.0], # Throttle up
            6: [ 0.0,  0.0, 0.0, 0.0]  # Throttle down
        }
        
        self.gates = np.array([
            [ 50.0,   0.0, 10.0],
            [ 50.0,  50.0, 10.0],
            [-50.0,  50.0, 10.0],
            [-50.0,   0.0, 10.0]
        ], dtype=np.float32)
        
        self.current_gate_idx = 0
        self.state = None
        self.prev_v = 0.0

    def sample_action(self):
        return np.random.randint(0, 7)

    def reset(self):
        # Instant state reset (No ROS service call)
        self.state = np.array(list(self.x0_dict.values()), dtype=float)
        self.current_gate_idx = 0
        self.prev_v = 0.0
        return self._get_obs(), {}

    def get_val(self, name):
        return self.state[self.model["x_index"][name]]

    def _get_obs(self):
        x = self.get_val("position_w_0")
        y = self.get_val("position_w_1")
        z = self.get_val("position_w_2")
        
        qw = self.get_val("quat_wb_0")
        qx = self.get_val("quat_wb_1")
        qy = self.get_val("quat_wb_2")
        qz = self.get_val("quat_wb_3")
        roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
        
        vx = self.get_val("velocity_b_0")
        vy = self.get_val("velocity_b_1")
        vz = self.get_val("velocity_b_2")
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        
        p_rate = self.get_val("omega_wb_b_0")
        q_rate = self.get_val("omega_wb_b_1")
        r_rate = self.get_val("omega_wb_b_2")
        
        denom = max(v, 1e-5)
        gamma = math.asin(max(-1.0, min(1.0, vz / denom)))
        
        # Calculate acceleration over the RL control step (0.1s)
        control_dt = self.dt * self.control_steps
        vdot = (v - self.prev_v) / control_dt
        self.prev_v = v
        
        obs = np.array([
            x, y, z, roll, pitch, yaw, 
            vx, vy, vz, v, gamma, vdot, 
            p_rate, q_rate, r_rate
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        cont_action = self.discrete_action_map[action]
        
        # Match ROS joy input parsing exactly 
        u = np.array([
            cont_action[0],
            -cont_action[1], # input_auto[1] = -msg.axes[1] in ROS
            cont_action[2],
            cont_action[3]
        ], dtype=float)
        
        # Integrate CasADi physics for `control_steps` to mimic ROS spin block
        for _ in range(self.control_steps):
            res = self.integrator(x0=self.state, z0=0.0, p=self.p, u=u)
            self.state = np.array(res["xf"]).reshape(-1)
        
        obs = self._get_obs()
        pos = obs[0:3]
        target_gate = self.gates[self.current_gate_idx]
        dist_to_gate = np.linalg.norm(pos - target_gate)
        
        reward = 0.0
        done = False
        
        # Gate Passing Logic
        if dist_to_gate < 10.0:
            reward += 100.0
            self.current_gate_idx = (self.current_gate_idx + 1) % len(self.gates)
            target_gate = self.gates[self.current_gate_idx]
            dist_to_gate = np.linalg.norm(pos - target_gate)
            
        # Dense shaping
        reward += -0.1 * dist_to_gate
        
        # Crash detection
        if obs[2] < 0.0:
            done = True
            reward -= 500.0
            
        return obs, float(reward), done, False, {}

# ==========================================
# NEURAL NETWORK SETUP
# ==========================================
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

# ==========================================
# TRAINING LOOP
# ==========================================
def train():
    env = PurePythonPylonEnv()
    
    state_dim = 15
    action_dim = 7
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(100000)
    
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 50000 # Increased for longer training
    
    # 5,000 episodes now finish very quickly vs ROS!
    num_episodes = 5000 
    global_step = 0
    
    print("Starting Pure-Python CasADi Deep Q-Learning Training...")
    print("Targeting 5,000 Episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * global_step / epsilon_decay)
            
            if random.random() < epsilon:
                action = env.sample_action()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_net(state_t)
                    action = q_values.argmax().item()
                    
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            global_step += 1
            
            # Learn
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
                
            if global_step % 500 == 0:
                target_net.load_state_dict(q_net.state_dict())
                
        # Only print occasionally to let it run lightning fast
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}, Gate Idx Check: {env.current_gate_idx}")
        
    print("Training finished. Saving model weights...")
    model_path = os.path.join(os.path.dirname(__file__), 'dqn_pylon.pth')
    torch.save(q_net.state_dict(), model_path)
    print(f"Weights saved safely to {model_path}.")

if __name__ == '__main__':
    train()
