#!/usr/bin/env python3
"""
train_dqn.py  –  Stage 1: Takeoff & Level Flight at 10 m
==========================================================
Goal: Teach the fixed-wing aircraft to climb off the ground and maintain
      a stable cruise at roughly 10 m AGL.  Gate / pylon logic is removed
      entirely so the agent solves the simplest possible subtask first.

Observation (15-D from cyecca state):
    [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r]

Actions (7 discrete):
    0 – Cruise          (moderate throttle, controls centred)
    1 – Pitch up        (pull elevator)
    2 – Pitch down      (push elevator)
    3 – Roll left       (left aileron)
    4 – Roll right      (right aileron)
    5 – Full throttle   (climb power)
    6 – Idle throttle   (let nose drop / speed-brake)
"""

import math
import os
import random
import sys
from collections import deque

import casadi as ca
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── locate cyecca ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
cyecca_path = os.path.abspath(os.path.join(script_dir, "../../cyecca"))
if cyecca_path not in sys.path:
    sys.path.append(cyecca_path)

try:
    from cyecca.models import fixedwing_4ch
except ModuleNotFoundError as e:
    print(f"Could not import cyecca.  Looked in {cyecca_path}\nError: {e}")
    sys.exit(1)

# ── quaternion → euler (no external deps) ─────────────────────────────────────
def euler_from_quaternion(q):
    x, y, z, w = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT  – Stage 1: Takeoff + Level Flight
# ══════════════════════════════════════════════════════════════════════════════
TARGET_ALT = 10.0       # desired cruise altitude (m)
MAX_STEPS   = 1000      # max steps per episode

DISCRETE_ACTIONS = {
    0: [ 0.0,  0.0, 0.7,  0.0],   # Cruise
    1: [ 0.0,  0.5, 0.7,  0.0],   # Pitch up
    2: [ 0.0, -0.5, 0.7,  0.0],   # Pitch down
    3: [-0.5,  0.0, 0.7,  0.0],   # Roll left
    4: [ 0.5,  0.0, 0.7,  0.0],   # Roll right
    5: [ 0.0,  0.0, 1.0,  0.0],   # Full throttle
    6: [ 0.0,  0.0, 0.0,  0.0],   # Idle throttle
}


class TakeoffEnv:
    def __init__(self):
        # ── CasADi physics ──────────────────────────────────────────────────
        self.dt           = 0.01          # 100 Hz inner loop
        self.control_hz   = 10            # 10 Hz RL control (10 inner steps)

        self.model  = fixedwing_4ch.derive_model()
        p_dict      = self.model["p_defaults"]
        self.p      = np.array(
            [p_dict[str(self.model["p"][i])] for i in range(self.model["p"].shape[0])],
            dtype=float,
        )
        self.x0_dict = self.model["x0_defaults"]

        opts = {"abstol": 1e-2, "reltol": 1e-6, "fsens_err_con": True}
        self.integrator = ca.integrator(
            "fw_int", "cvodes", self.model["dae"], 0.0, self.dt, opts
        )

        self.state      = None
        self.step_count = 0
        self.prev_v     = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get(self, name):
        return self.state[self.model["x_index"][name]]

    def _obs(self):
        x   = self._get("position_w_0")
        y   = self._get("position_w_1")
        z   = self._get("position_w_2")
        qw  = self._get("quat_wb_0")
        qx  = self._get("quat_wb_1")
        qy  = self._get("quat_wb_2")
        qz  = self._get("quat_wb_3")
        roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
        vx  = self._get("velocity_b_0")
        vy  = self._get("velocity_b_1")
        vz  = self._get("velocity_b_2")
        v   = math.sqrt(vx**2 + vy**2 + vz**2)
        pr  = self._get("omega_wb_b_0")
        qr  = self._get("omega_wb_b_1")
        rr  = self._get("omega_wb_b_2")
        denom = max(v, 1e-5)
        gamma = math.asin(max(-1.0, min(1.0, vz / denom)))
        vdot  = (v - self.prev_v) / (self.dt * self.control_hz)
        self.prev_v = v
        return np.array(
            [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, pr, qr, rr],
            dtype=np.float32,
        )

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self):
        self.state      = np.array(list(self.x0_dict.values()), dtype=float)
        # Start on the ground (z=0) with enough forward airspeed for the physics to work
        self.state[self.model["x_index"]["position_w_2"]]  = 0.0   # ground level
        self.state[self.model["x_index"]["velocity_b_0"]]  = 8.0   # 8 m/s takeoff roll speed
        self.step_count = 0
        self.prev_v     = 8.0
        return self._obs(), {}

    def sample(self):
        return random.randint(0, 6)

    def step(self, action: int):
        a = DISCRETE_ACTIONS[action]
        # NOTE: elevator is pre-negated to match auto_joy_callback sign convention
        u = np.array([a[0], -a[1], a[2], a[3]], dtype=float)

        try:
            for _ in range(self.control_hz):
                res        = self.integrator(x0=self.state, z0=0.0, p=self.p, u=u)
                self.state = np.array(res["xf"]).reshape(-1)
        except RuntimeError:
            # Physics blew up (extreme manoeuvre) → treat as crash
            return self._obs(), -200.0, True, False, {}

        obs  = self._obs()
        done = False

        z     = obs[2]
        roll  = obs[3]
        pitch = obs[4]
        v     = obs[9]

        # ── Orientation-aware crash detection ─────────────────────────────
        # A "crash" is defined as being on or below the ground AND in an attitude
        # that is not consistent with controlled flight (inverted or extreme pitch).
        # The plane sitting wings-level on the ground during the takeoff roll is NOT a crash.
        is_on_ground   = z < 0.3                          # very close to ground
        is_inverted    = abs(roll)  > math.radians(120)   # rolled past 120 deg
        is_nose_down   = pitch      < math.radians(-45)   # pitched hard into dirt
        crashed = is_on_ground and (is_inverted or is_nose_down)

        if crashed:
            return obs, -200.0, True, False, {}

        # ── Reward components ─────────────────────────────────────────────
        # 1. Altitude reward: bell-curve centred on TARGET_ALT
        alt_err    = z - TARGET_ALT
        r_altitude = 10.0 * math.exp(-0.5 * (alt_err / 3.0) ** 2)

        # 2. Wings-level penalty (quadratic in roll and pitch deviation)
        r_stability = -(roll**2 + pitch**2) * 5.0

        # 3. Forward speed reward
        r_speed = min(v, 20.0) * 0.3

        # 4. Small alive bonus so the agent doesn't prefer crashing
        r_alive = 1.0

        reward = r_altitude + r_stability + r_speed + r_alive

        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            done = True

        return obs, float(reward), done, False, {}





# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    def __init__(self, state_dim=15, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train():
    env = TakeoffEnv()

    device = torch.device("cpu")   # tiny MLP → CPU is fastest on M4 Pro
    print(f"Using device: {device}")

    q_net      = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=3e-4)
    buf       = ReplayBuffer(100_000)

    # Hyper-parameters
    EPISODES       = 3000
    BATCH          = 128
    GAMMA          = 0.99
    EPS_START      = 1.0
    EPS_END        = 0.05
    EPS_DECAY      = 30_000    # global steps to decay over
    TARGET_UPDATE  = 500       # update target every N global steps
    LEARN_START    = 500       # wait for buffer to fill before learning
    LOG_EVERY      = 20        # print every N episodes

    global_step = 0
    print(f"Stage-1 Training: takeoff + cruise at {TARGET_ALT} m — {EPISODES} episodes")

    for ep in range(EPISODES):
        state, _ = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-global_step / EPS_DECAY)

            if random.random() < eps:
                action = env.sample()
            else:
                with torch.no_grad():
                    st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = q_net(st).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            buf.push(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward
            global_step += 1

            # Learn
            if len(buf) >= LEARN_START:
                s, a, r, s2, d = buf.sample(BATCH)
                s  = torch.as_tensor(s,  device=device)
                a  = torch.as_tensor(a,  device=device)
                r  = torch.as_tensor(r,  device=device)
                s2 = torch.as_tensor(s2, device=device)
                d  = torch.as_tensor(d,  device=device)

                curr_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q   = target_net(s2).max(1)[0]
                    target_q = r + (1 - d) * GAMMA * next_q

                loss = nn.SmoothL1Loss()(curr_q, target_q)   # Huber loss (more stable)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        if ep % LOG_EVERY == 0:
            print(
                f"Ep {ep:4d}/{EPISODES} | "
                f"reward {ep_reward:8.1f} | "
                f"eps {eps:.3f} | "
                f"steps {env.step_count}"
            )

    save_path = os.path.join(script_dir, "dqn_pylon.pth")
    torch.save(q_net.state_dict(), save_path)
    print(f"\nWeights saved → {save_path}")


if __name__ == "__main__":
    train()
