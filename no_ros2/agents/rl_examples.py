#!/usr/bin/env python3
"""
rl_examples.py
--------------
Four example DQN agents demonstrating different action-space designs
for pylon racing. All share the same DQN infrastructure (NumpyMLP, Adam,
replay buffer) but differ in what the RL agent controls:

  Agent 1 — Waypoint Selector (discrete):
    Action = which predefined waypoint to target next.
    XTrack_NAV_lookAhead + PD inner loop handles all flying.

  Agent 2 — Guidance Commander (discrete):
    Action = (des_heading_offset, des_speed_offset) tuple.
    PD inner loop converts heading/speed to control surfaces.

  Agent 3 — Waypoint Coordinate (continuous via discretization):
    Action = (dx, dy) offset from current position as next waypoint.
    XTrack_NAV_lookAhead tracks that waypoint, PD inner loop flies.

  Agent 4 — Residual Planner (recommended):
    The heuristic waypoint planner (XTrack_NAV_lookAhead) flies the course.
    The RL agent learns small (heading, speed) adjustments on top of the
    planner's output. This is residual RL — the agent only needs to learn
    what the planner gets wrong (approach angles, corner speed, etc.).

All use a reduced 8D state vector (not all 15 obs are needed):
  [sin(bearing), cos(bearing), log(dist),
   cross_track_err, along_track_remaining,
   alt_err, roll, speed]

Run:
  python no_ros2/agents/rl_examples.py --agent 1 --purt --train 300
  python no_ros2/agents/rl_examples.py --agent 2 --sample --train 500
  python no_ros2/agents/rl_examples.py --agent 3 --purt --train 300
  python no_ros2/agents/rl_examples.py --agent 4 --purt --train 300 --viz
"""

import os
import sys
import importlib.util

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
sys.path.insert(0, _root)

import numpy as np
from collections import deque

from no_ros2.environments.env_factory import make_pylon_env
from no_ros2.environments.pylon_course import get_course, PYLON_MID_HEIGHT_M
from no_ros2.environments.pylon_wrapper import PylonRacingWrapper
from no_ros2.environments_v2.cross_tracker_nav_sample import XTrack_NAV_lookAhead

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TARGET_ALT_M   = 7.0
CRUISE_SPEED   = 7.0
MAX_STEPS      = 6000
ENV_DT         = 0.02
PASS_THRESH_M  = 6.0

# PD inner-loop gains
_KP_PITCH  = 0.08
_KD_PITCH  = 0.5
_KP_ROLL   = 0.04
_KD_ROLL   = 0.3
_KP_YAW    = 0.02
_HEADING_GAIN = 2.0
_MAX_BANK  = 30.0
TRIM_PITCH = 2.0
TRIM_THR   = 0.35


# ---------------------------------------------------------------------------
# State extraction — only what the RL agent needs
# ---------------------------------------------------------------------------
#
# From the 15D observation, the RL agent needs navigation-relevant info,
# NOT raw control states. The inner-loop PD handles stabilization using
# the full obs internally.
#
# 15D obs layout:
#   [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r]
#    0  1  2   3     4     5    6   7   8   9   10     11   12 13 14
#
# What the RL agent needs:
#   - Relative bearing to target (sin, cos) — direction to go
#   - Distance to target (log-scaled) — how far
#   - Cross-track error — lateral offset from ideal racing line
#   - Along-track remaining — progress on current path segment
#   - Altitude error — stay at altitude
#   - Speed — manage energy
#   - Roll — awareness of bank state (affects turn authority)
#
# What the RL agent does NOT need:
#   - Absolute position (x, y) — not generalizable
#   - Pitch, yaw — inner loop handles these
#   - vx, vy, vz — redundant with bearing + speed
#   - gamma, vdot — inner loop handles flight path
#   - p, q, r — angular rates for inner loop only

STATE_DIM = 8


def compute_path_errors(obs, prev_wpt, next_wpt):
    """
    Compute cross-track and along-track errors for the path segment
    prev_wpt -> next_wpt, matching XTrack_NAV_lookAhead's math.

    Returns (along_track_remaining, cross_track_err).
    """
    x, y = obs[0], obs[1]
    path_vect = np.array([next_wpt[0] - prev_wpt[0],
                          next_wpt[1] - prev_wpt[1]])
    path_len = np.linalg.norm(path_vect)
    if path_len < 1e-3:
        dx = next_wpt[0] - x
        dy = next_wpt[1] - y
        return np.sqrt(dx*dx + dy*dy), 0.0

    unit_along = path_vect / path_len
    unit_normal = np.array([-path_vect[1], path_vect[0]]) / path_len

    pose_vect = np.array([x - prev_wpt[0], y - prev_wpt[1]])
    along_from_w0 = np.dot(pose_vect, unit_along)
    cross_track = np.dot(pose_vect, unit_normal)
    along_remaining = max(0.0, path_len - np.clip(along_from_w0, 0.0, path_len))

    return along_remaining, cross_track


def make_state(obs, target_xy, prev_wpt=None, next_wpt=None):
    """
    8D state with path-relative awareness:
      [sin(bearing), cos(bearing), log(dist),
       cross_track_err, along_track_remaining,
       alt_err, roll, speed]

    target_xy: (x, y) of the current target gate.
    prev_wpt, next_wpt: path segment endpoints for cross-track computation.
    """
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]

    dx = target_xy[0] - x
    dy = target_xy[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

    vel_angle = np.arctan2(vy, vx)
    tgt_angle = np.arctan2(dy, dx)
    bearing = tgt_angle - vel_angle
    bearing = (bearing + np.pi) % (2*np.pi) - np.pi

    # Cross-track and along-track errors
    if prev_wpt is not None and next_wpt is not None:
        along_rem, xtrack = compute_path_errors(obs, prev_wpt, next_wpt)
    else:
        along_rem, xtrack = dist, 0.0

    return np.array([
        np.sin(bearing),              # turn direction
        np.cos(bearing),              # alignment (1 = heading at target)
        np.log1p(dist) / 5.0,        # log-scaled distance
        xtrack / 10.0,               # cross-track error (signed, normalized)
        along_rem / 20.0,            # along-track remaining (normalized)
        (z - TARGET_ALT_M) / 5.0,   # altitude error
        obs[3] / np.pi,             # roll normalized
        obs[9] / 10.0,              # speed normalized
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# PD inner-loop controller (shared by all agents)
# ---------------------------------------------------------------------------

def heading_to_controls(des_heading, des_speed, obs):
    """
    Convert desired heading + speed to raw [aileron, elevator, throttle, rudder].
    The RL agent outputs high-level commands; this function handles low-level control.
    """
    vx, vy = obs[6], obs[7]
    V = max(obs[9], 1.0)
    z = obs[2]
    roll_deg  = np.degrees(obs[3])
    pitch_deg = np.degrees(obs[4])
    p, q, r   = obs[12], obs[13], obs[14]

    # Heading error -> bank angle
    current_heading = np.arctan2(vy, vx)
    heading_err = des_heading - current_heading
    heading_err = (heading_err + np.pi) % (2*np.pi) - np.pi
    heading_err_deg = np.degrees(heading_err)
    target_roll_deg = np.clip(_HEADING_GAIN * heading_err_deg, -_MAX_BANK, _MAX_BANK)

    # Altitude hold -> pitch
    alt_err = TARGET_ALT_M - z
    alt_pitch = np.clip(3.0 * alt_err, -8.0, 8.0)

    # Bank-pitch compensation
    roll_rad = np.radians(roll_deg)
    cos_roll = max(np.cos(roll_rad), 0.3)
    bank_pitch_comp = (1.0 / cos_roll - 1.0) * 6.0
    target_pitch_deg = TRIM_PITCH + alt_pitch + bank_pitch_comp

    # Pitch PD
    pitch_err = target_pitch_deg - pitch_deg
    elevator = np.clip(_KP_PITCH * pitch_err - _KD_PITCH * q, -1.0, 1.0)

    # Roll PD
    roll_err = target_roll_deg - roll_deg
    aileron = np.clip(_KP_ROLL * roll_err - _KD_ROLL * p, -1.0, 1.0)

    # Rudder
    rudder = np.clip(_KP_YAW * roll_deg, -1.0, 1.0)

    # Throttle
    speed_err = des_speed - V
    bank_comp = 0.1 * abs(target_roll_deg) / max(_MAX_BANK, 1.0)
    throttle = np.clip(TRIM_THR + 0.1 * speed_err + bank_comp, 0.0, 1.0)

    return np.array([aileron, elevator, throttle, rudder], dtype=np.float32)


# ---------------------------------------------------------------------------
# Navigator helper (shared by agents 1 and 3)
# ---------------------------------------------------------------------------

def create_navigator(waypoints, start_idx):
    """Create and configure XTrack_NAV_lookAhead for the mock env."""
    nav = XTrack_NAV_lookAhead(ENV_DT, waypoints, start_idx)
    nav.v_cruise = CRUISE_SPEED
    if start_idx == 0 and len(waypoints) >= 2:
        nav.prev_wpt = waypoints[-2]
    elif start_idx > 0:
        nav.prev_wpt = waypoints[start_idx - 1]
    return nav


# ---------------------------------------------------------------------------
# Numpy MLP + Adam + Replay (shared infrastructure)
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(0, x)


def _he_init(fan_in, fan_out, rng):
    return rng.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)


class NumpyMLP:
    def __init__(self, input_dim, hidden1, hidden2, output_dim, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.W1 = _he_init(input_dim, hidden1, rng)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = _he_init(hidden1, hidden2, rng)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = _he_init(hidden2, output_dim, rng)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def copy_from(self, other):
        for attr in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            getattr(self, attr)[:] = getattr(other, attr)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        return (
            np.array([b[0] for b in batch], dtype=np.float32),
            np.array([b[1] for b in batch], dtype=np.int32),
            np.array([b[2] for b in batch], dtype=np.float32),
            np.array([b[3] for b in batch], dtype=np.float32),
            np.array([b[4] for b in batch], dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNCore:
    """Shared DQN training logic for all three agents."""

    def __init__(self, state_dim, n_actions, lr=3e-4, gamma=0.97,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.992,
                 batch_size=64, tau=0.005, double_dqn=True):
        rng = np.random.RandomState(42)
        self.q_net = NumpyMLP(state_dim, 128, 64, n_actions, rng)
        self.target_net = NumpyMLP(state_dim, 128, 64, n_actions, rng)
        self.target_net.copy_from(self.q_net)
        self.optimizer = AdamOptimizer(self.q_net.params(), lr=lr)
        self.replay = ReplayBuffer(capacity=50000)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.double_dqn = double_dqn
        self.train_steps = 0

    def get_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_net.forward(state).flatten()
        return int(np.argmax(q))

    def train_step(self):
        if len(self.replay) < self.batch_size * 2:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        batch = self.batch_size

        h1_pre = states @ self.q_net.W1 + self.q_net.b1
        h1 = _relu(h1_pre)
        h2_pre = h1 @ self.q_net.W2 + self.q_net.b2
        h2 = _relu(h2_pre)
        q_all = h2 @ self.q_net.W3 + self.q_net.b3
        q_values = q_all[np.arange(batch), actions]

        # Double DQN: online net selects action, target net evaluates
        if self.double_dqn:
            q_next_online = self.q_net.forward(next_states)
            best_actions = np.argmax(q_next_online, axis=1)
            q_next_target = self.target_net.forward(next_states)
            q_next_vals = q_next_target[np.arange(batch), best_actions]
        else:
            q_next = self.target_net.forward(next_states)
            q_next_vals = np.max(q_next, axis=1)

        q_target = rewards + self.gamma * q_next_vals * (1 - dones)

        td_error = q_target - q_values
        td_clipped = np.clip(td_error, -10.0, 10.0)

        d_out = np.zeros((batch, self.n_actions), dtype=np.float32)
        d_out[np.arange(batch), actions] = -td_clipped / batch

        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0)
        d_h2 = d_out @ self.q_net.W3.T
        d_h2 *= (h2_pre > 0).astype(np.float32)
        dW2 = h1.T @ d_h2
        db2 = d_h2.sum(axis=0)
        d_h1 = d_h2 @ self.q_net.W2.T
        d_h1 *= (h1_pre > 0).astype(np.float32)
        dW1 = states.T @ d_h1
        db1 = d_h1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > 5.0:
            grads = [g * (5.0 / total_norm) for g in grads]
        self.optimizer.step(grads)

        # Soft target update (Polyak averaging)
        self.train_steps += 1
        for p_online, p_target in zip(self.q_net.params(), self.target_net.params()):
            p_target[:] = self.tau * p_online + (1.0 - self.tau) * p_target
        return np.mean(td_error ** 2)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Reward shaping (shared)
# ---------------------------------------------------------------------------

def compute_reward(obs, target_xy, prev_dist, bounds_rect):
    """
    Returns (reward, passed, new_dist).
    """
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy = obs[6], obs[7]

    dx = target_xy[0] - x
    dy = target_xy[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

    if dist < PASS_THRESH_M:
        return 100.0, True, dist

    reward = 0.0

    # Distance shaping
    if prev_dist is not None:
        delta = prev_dist - dist
        reward += np.clip(delta * 3.0, -2.0, 2.0)

    # Alignment bonus
    speed_xy = np.sqrt(vx*vx + vy*vy) + 1e-6
    align = (dx/dist * vx + dy/dist * vy) / speed_xy
    reward += 0.5 * align

    # Proximity bonus
    if dist < 15.0:
        reward += 1.5 * (1.0 - dist / 15.0)

    # Bounds penalty
    g = bounds_rect
    if x < g["min_x"] or x > g["max_x"] or y < g["min_y"] or y > g["max_y"]:
        reward -= 5.0

    # Altitude penalty
    if abs(z - TARGET_ALT_M) > 4.0:
        reward -= 0.5

    return reward, False, dist


# ===================================================================
# AGENT 1: Waypoint Selector
# ===================================================================
#
# The RL agent picks WHICH predefined waypoint to fly toward.
# XTrack_NAV_lookAhead handles the path-following guidance,
# and the PD inner loop handles attitude control.
#
# Action space: N_GATES discrete actions (one per gate waypoint).
# The agent learns which gate to target given its current state.
# This is the simplest action space — the agent only decides strategy.

def plan_gate_midpoints(course_data):
    """Gate midpoints at target altitude (no closing waypoint for selector)."""
    pylons = course_data["pylons"]
    gates = course_data["gates"]
    waypoints = []
    for p_a, p_b in gates:
        mid = (pylons[p_a][:2] + pylons[p_b][:2]) / 2.0
        waypoints.append((mid[0], mid[1], TARGET_ALT_M))
    return waypoints


def run_agent1(env, agent, course_data, train=True, viz=None):
    """Waypoint Selector: agent picks which gate to target."""
    gates_wp = plan_gate_midpoints(course_data)
    n_gates = len(gates_wp)
    bounds_rect = course_data["bounds_rect"]

    obs, _ = env.reset()
    target_idx = 0
    prev_dist = None
    total_reward = 0.0
    gates_passed = 0

    # Build navigator waypoints (with closing waypoint for path tracking)
    nav_wps = list(gates_wp) + [gates_wp[0]]
    laps = 0

    if viz:
        viz.clear_trail()

    for step in range(MAX_STEPS):
        target_xy = gates_wp[target_idx]
        prev_wpt = gates_wp[(target_idx - 1) % n_gates]
        next_wpt = gates_wp[target_idx]
        state = make_state(obs, target_xy, prev_wpt, next_wpt)

        # Agent picks which waypoint to target
        action_idx = agent.get_action(state, greedy=not train)
        chosen_wp_idx = action_idx % n_gates  # wrap to valid range

        # Create navigator toward chosen waypoint
        nav = create_navigator(nav_wps, chosen_wp_idx)
        V_array = [obs[6], obs[7], obs[8]]
        des_v, des_gamma, des_heading, _, _ = \
            nav.wp_tracker(nav_wps, obs[0], obs[1], obs[2], V_array)

        # Inner-loop PD
        action = heading_to_controls(des_heading, des_v, obs)
        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Reward relative to the NEXT sequential gate (not the chosen one)
        reward, passed, dist = compute_reward(obs, target_xy, prev_dist, bounds_rect)
        if terminated and r_env < 0:
            reward += r_env

        if passed:
            gates_passed += 1
            target_idx = (target_idx + 1) % n_gates
            prev_dist = None
            if gates_passed % n_gates == 0:
                laps += 1
        else:
            prev_dist = dist

        new_prev = gates_wp[(target_idx - 1) % n_gates]
        new_next = gates_wp[target_idx]
        next_state = make_state(obs, gates_wp[target_idx], new_prev, new_next)

        if train:
            agent.replay.push(state, action_idx, np.clip(reward, -10, 10),
                              next_state, float(done))
            if step % 4 == 0:
                agent.train_step()

        total_reward += reward
        if viz:
            viz.update(obs, reward=r_env, step_count=step, laps=laps)
        if done:
            break

    if train:
        agent.decay_epsilon()
    return total_reward, gates_passed, step + 1


# ===================================================================
# AGENT 2: Guidance Commander
# ===================================================================
#
# The RL agent outputs high-level guidance commands:
#   (heading_offset, speed_offset) relative to bearing toward target.
# The PD inner loop converts these to control surfaces.
#
# This is similar to the existing dqn_agent.py but with bearing-relative
# heading offsets so the agent learns turning strategy, not absolute heading.
#
# Action space: 9 discrete actions (heading x speed grid).

AGENT2_ACTIONS = [
    # (heading_offset_deg, speed_boost)
    ( 0.0,  0.00),    # 0: straight at target, cruise
    (+30.0, 0.00),    # 1: aim right
    (-30.0, 0.00),    # 2: aim left
    (+60.0, 0.00),    # 3: wide right
    (-60.0, 0.00),    # 4: wide left
    ( 0.0, +0.15),    # 5: straight, speed up
    ( 0.0, -0.10),    # 6: straight, slow down
    (+30.0,+0.15),    # 7: right + speed up
    (-30.0,+0.15),    # 8: left + speed up
]


def run_agent2(env, agent, course_data, train=True, viz=None):
    """Guidance Commander: agent picks heading offset + speed."""
    gates_wp = plan_gate_midpoints(course_data)
    n_gates = len(gates_wp)
    bounds_rect = course_data["bounds_rect"]

    obs, _ = env.reset()
    target_idx = 0
    prev_dist = None
    total_reward = 0.0
    gates_passed = 0
    laps = 0

    if viz:
        viz.clear_trail()

    for step in range(MAX_STEPS):
        target_xy = gates_wp[target_idx]
        prev_wpt = gates_wp[(target_idx - 1) % n_gates]
        next_wpt = gates_wp[target_idx]
        state = make_state(obs, target_xy, prev_wpt, next_wpt)

        action_idx = agent.get_action(state, greedy=not train)
        heading_offset_deg, speed_boost = AGENT2_ACTIONS[action_idx]

        # Compute bearing to target
        x, y = obs[0], obs[1]
        vx, vy = obs[6], obs[7]
        dx, dy = target_xy[0] - x, target_xy[1] - y
        tgt_bearing = np.arctan2(dy, dx)

        # Apply heading offset
        des_heading = tgt_bearing + np.radians(heading_offset_deg)
        des_speed = CRUISE_SPEED + speed_boost

        # Inner-loop PD
        action = heading_to_controls(des_heading, des_speed, obs)
        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        reward, passed, dist = compute_reward(obs, target_xy, prev_dist, bounds_rect)
        if terminated and r_env < 0:
            reward += r_env

        if passed:
            gates_passed += 1
            target_idx = (target_idx + 1) % n_gates
            prev_dist = None
            if gates_passed % n_gates == 0:
                laps += 1
        else:
            prev_dist = dist

        new_prev = gates_wp[(target_idx - 1) % n_gates]
        new_next = gates_wp[target_idx]
        next_state = make_state(obs, gates_wp[target_idx], new_prev, new_next)

        if train:
            agent.replay.push(state, action_idx, np.clip(reward, -10, 10),
                              next_state, float(done))
            if step % 4 == 0:
                agent.train_step()

        total_reward += reward
        if viz:
            viz.update(obs, reward=r_env, step_count=step, laps=laps)
        if done:
            break

    if train:
        agent.decay_epsilon()
    return total_reward, gates_passed, step + 1


# ===================================================================
# AGENT 3: Waypoint Coordinate
# ===================================================================
#
# The RL agent outputs a (dx, dy) offset from its current position,
# defining a dynamic waypoint. XTrack_NAV_lookAhead tracks the path
# from current position to that waypoint, PD inner loop flies.
#
# This gives the agent full spatial control — it learns WHERE to go
# in 2D space, not just which predefined gate to pick.
#
# Action space: 8 discrete directions (N, NE, E, SE, S, SW, W, NW)
# at a fixed lookahead distance. Could be extended with variable
# distances for more expressiveness.

_WP_LOOKAHEAD = 15.0  # meters ahead for dynamic waypoint
AGENT3_DIRS = [
    ( 0.0,  1.0),   # 0: North
    ( 0.7,  0.7),   # 1: NE
    ( 1.0,  0.0),   # 2: East
    ( 0.7, -0.7),   # 3: SE
    ( 0.0, -1.0),   # 4: South
    (-0.7, -0.7),   # 5: SW
    (-1.0,  0.0),   # 6: West
    (-0.7,  0.7),   # 7: NW
]


def run_agent3(env, agent, course_data, train=True, viz=None):
    """Waypoint Coordinate: agent picks direction to place a dynamic waypoint."""
    gates_wp = plan_gate_midpoints(course_data)
    n_gates = len(gates_wp)
    bounds_rect = course_data["bounds_rect"]

    obs, _ = env.reset()
    target_idx = 0
    prev_dist = None
    total_reward = 0.0
    gates_passed = 0
    laps = 0

    if viz:
        viz.clear_trail()

    for step in range(MAX_STEPS):
        target_xy = gates_wp[target_idx]
        prev_wpt = gates_wp[(target_idx - 1) % n_gates]
        next_wpt = gates_wp[target_idx]
        state = make_state(obs, target_xy, prev_wpt, next_wpt)

        action_idx = agent.get_action(state, greedy=not train)
        dir_x, dir_y = AGENT3_DIRS[action_idx]

        # Place dynamic waypoint in chosen direction
        wp_x = obs[0] + dir_x * _WP_LOOKAHEAD
        wp_y = obs[1] + dir_y * _WP_LOOKAHEAD
        wp_z = TARGET_ALT_M
        dyn_wp = [(obs[0], obs[1], wp_z), (wp_x, wp_y, wp_z)]

        # Use navigator to track toward dynamic waypoint
        nav = XTrack_NAV_lookAhead(ENV_DT, dyn_wp, 1)
        nav.v_cruise = CRUISE_SPEED
        nav.prev_wpt = dyn_wp[0]
        V_array = [obs[6], obs[7], obs[8]]
        des_v, des_gamma, des_heading, _, _ = \
            nav.wp_tracker(dyn_wp, obs[0], obs[1], obs[2], V_array)

        # Inner-loop PD
        action = heading_to_controls(des_heading, des_v, obs)
        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        reward, passed, dist = compute_reward(obs, target_xy, prev_dist, bounds_rect)
        if terminated and r_env < 0:
            reward += r_env

        if passed:
            gates_passed += 1
            target_idx = (target_idx + 1) % n_gates
            prev_dist = None
            if gates_passed % n_gates == 0:
                laps += 1
        else:
            prev_dist = dist

        new_prev = gates_wp[(target_idx - 1) % n_gates]
        new_next = gates_wp[target_idx]
        next_state = make_state(obs, gates_wp[target_idx], new_prev, new_next)

        if train:
            agent.replay.push(state, action_idx, np.clip(reward, -10, 10),
                              next_state, float(done))
            if step % 4 == 0:
                agent.train_step()

        total_reward += reward
        if viz:
            viz.update(obs, reward=r_env, step_count=step, laps=laps)
        if done:
            break

    if train:
        agent.decay_epsilon()
    return total_reward, gates_passed, step + 1


# ===================================================================
# AGENT 4: Residual Planner
# ===================================================================
#
# The heuristic waypoint planner (XTrack_NAV_lookAhead) already flies
# the course well (~338 laps on purt). The RL agent learns small
# corrections on top of the planner's guidance output.
#
# Each step:
#   1. Navigator computes (des_heading, des_speed) — the baseline plan
#   2. RL agent picks a small (heading_adjust, speed_adjust) residual
#   3. Final command = baseline + residual
#   4. PD inner loop converts to control surfaces
#
# This is residual RL: the agent starts with good performance from
# step 1 and only needs to learn the delta that improves lap times
# or smoothness.
#
# Action space: 7 discrete residual adjustments.

AGENT4_RESIDUALS = [
    # (heading_adjust_deg, speed_adjust)
    ( 0.0,  0.00),    # 0: no correction (trust the planner)
    (+2.0,  0.00),    # 1: slight right
    (-2.0,  0.00),    # 2: slight left
    (+5.0,  0.00),    # 3: moderate right
    (-5.0,  0.00),    # 4: moderate left
    ( 0.0, +0.04),    # 5: speed up
    ( 0.0, -0.04),    # 6: slow down
]


def _plan_nav_waypoints(course_data):
    """Gate midpoints with closing waypoint for navigator."""
    gates_wp = plan_gate_midpoints(course_data)
    return list(gates_wp) + [gates_wp[0]]


def run_agent4(env, agent, course_data, train=True, viz=None):
    """Residual Planner: heuristic navigator + RL adjustments."""
    gates_wp = plan_gate_midpoints(course_data)
    n_gates = len(gates_wp)
    bounds_rect = course_data["bounds_rect"]

    # Navigator waypoints (with closing waypoint for looping)
    nav_wps = _plan_nav_waypoints(course_data)

    obs, _ = env.reset()

    # Pick initial waypoint aligned with heading (same as waypoint_agent_v2)
    vx, vy = obs[6], obs[7]
    current_heading = np.arctan2(vy, vx)
    best_idx, best_score = 0, -1e9
    for i in range(len(nav_wps) - 1):
        wp = nav_wps[i]
        dx, dy = wp[0] - obs[0], wp[1] - obs[1]
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        wp_heading = np.arctan2(dy, dx)
        err = abs((wp_heading - current_heading + np.pi) % (2*np.pi) - np.pi)
        score = -err - dist * 0.02
        if score > best_score:
            best_score = score
            best_idx = i

    nav = create_navigator(nav_wps, best_idx)

    # Track which sequential gate we're targeting for reward
    target_idx = best_idx % n_gates
    prev_dist = None
    total_reward = 0.0
    gates_passed = 0
    laps = 0

    if viz:
        viz.clear_trail()

    for step in range(MAX_STEPS):
        target_xy = gates_wp[target_idx]
        prev_wpt = gates_wp[(target_idx - 1) % n_gates]
        next_wpt = gates_wp[target_idx]
        state = make_state(obs, target_xy, prev_wpt, next_wpt)

        # Step 1: Navigator computes baseline guidance
        V_array = [obs[6], obs[7], obs[8]]
        des_v, des_gamma, des_heading, along_err, xtrack_err = \
            nav.wp_tracker(nav_wps, obs[0], obs[1], obs[2], V_array)

        # Step 2: RL agent picks a residual adjustment
        action_idx = agent.get_action(state, greedy=not train)
        heading_adj_deg, speed_adj = AGENT4_RESIDUALS[action_idx]

        # Step 3: Apply residual on top of baseline
        final_heading = des_heading + np.radians(heading_adj_deg)
        final_speed = des_v + speed_adj

        # Step 4: PD inner loop
        action = heading_to_controls(final_heading, final_speed, obs)
        obs, r_env, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Navigator waypoint switching
        prev_wp_idx = nav.current_WP_ind
        nav.check_arrived(along_err, V_array)
        if nav.current_WP_ind > prev_wp_idx:
            gates_passed += 1

        # Looping: reinitialize navigator
        if nav.current_WP_ind >= len(nav_wps):
            laps += 1
            nav = create_navigator(nav_wps, 0)

        # Update target index to match navigator
        target_idx = nav.current_WP_ind % n_gates

        # Reward
        reward, passed, dist = compute_reward(obs, gates_wp[target_idx],
                                              prev_dist, bounds_rect)
        if terminated and r_env < 0:
            reward += r_env

        # Penalize cross-track error (encourage staying on racing line)
        reward -= 0.15 * min(abs(xtrack_err), 5.0)

        # Gate passage bonus
        if nav.current_WP_ind > prev_wp_idx:
            reward += 20.0

        # Speed reward — faster is better for racing
        reward += 0.1 * (obs[9] - CRUISE_SPEED)

        # Penalty for corrections proportional to magnitude
        if action_idx != 0:
            h_adj = abs(AGENT4_RESIDUALS[action_idx][0])
            s_adj = abs(AGENT4_RESIDUALS[action_idx][1])
            reward -= 0.04 * h_adj + 1.0 * s_adj
        else:
            # Bonus for trusting the planner
            reward += 0.15

        if passed:
            prev_dist = None
        else:
            prev_dist = dist

        new_prev = gates_wp[(target_idx - 1) % n_gates]
        new_next = gates_wp[target_idx]
        next_state = make_state(obs, gates_wp[target_idx], new_prev, new_next)

        if train:
            agent.replay.push(state, action_idx, np.clip(reward, -25, 25),
                              next_state, float(done))
            if step % 4 == 0:
                agent.train_step()

        total_reward += reward
        if viz:
            viz.update(obs, reward=r_env, step_count=step, laps=laps)
        if done:
            break

    if train:
        agent.decay_epsilon()
    return total_reward, gates_passed, step + 1


# ---------------------------------------------------------------------------
# Dynamic env loader
# ---------------------------------------------------------------------------

def _make_env_from_file(env_path, course):
    env_path = os.path.abspath(env_path)
    spec = importlib.util.spec_from_file_location("_env_module", env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return PylonRacingWrapper(mod.MockPylonRacingEnv(course=course))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Example RL agents with 3 action-space designs")
    parser.add_argument("--agent", type=int, required=True, choices=[1, 2, 3, 4],
                        help="1=Waypoint Selector, 2=Guidance Commander, 3=Waypoint Coordinate, 4=Residual Planner")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--env", metavar="FILE", default=None)
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--viz", action="store_true")
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument("--purt", action="store_true")
    course_group.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    course_name = "purt" if args.purt else "sample"
    course_data = get_course(course_name)
    n_gates = len(course_data["gates"])
    print(f"Course: {course_name} ({n_gates} gates)")

    agent_names = {1: "Waypoint Selector", 2: "Guidance Commander",
                   3: "Waypoint Coordinate", 4: "Residual Planner"}
    print(f"Agent {args.agent}: {agent_names[args.agent]}")

    # Action counts per agent type
    if args.agent == 1:
        n_actions = n_gates
        run_fn = run_agent1
    elif args.agent == 2:
        n_actions = len(AGENT2_ACTIONS)
        run_fn = run_agent2
    elif args.agent == 3:
        n_actions = len(AGENT3_DIRS)
        run_fn = run_agent3
    else:
        n_actions = len(AGENT4_RESIDUALS)
        run_fn = run_agent4

    print(f"State dim: {STATE_DIM}, Actions: {n_actions}")

    if args.agent == 1:
        print(f"  Actions = gate indices [0..{n_gates-1}]")
    elif args.agent == 2:
        print(f"  Actions = (heading_offset, speed_boost) tuples:")
        for i, (h, s) in enumerate(AGENT2_ACTIONS):
            print(f"    {i}: heading={h:+.0f} deg, speed={s:+.2f}")
    elif args.agent == 3:
        print(f"  Actions = 8 compass directions at {_WP_LOOKAHEAD}m lookahead")
    else:
        print(f"  Actions = residual adjustments on top of heuristic planner:")
        for i, (h, s) in enumerate(AGENT4_RESIDUALS):
            print(f"    {i}: heading={h:+.0f} deg, speed={s:+.2f}")

    if args.ros2:
        import rclpy
        rclpy.init()

    agent = DQNCore(STATE_DIM, n_actions)

    if args.env:
        train_env = _make_env_from_file(args.env, course_name)
    elif args.ros2:
        train_env = make_pylon_env(use_ros2=True, course=course_name)
    else:
        train_env = make_pylon_env(use_ros2=False, course=course_name)

    viz = None
    if args.viz:
        from no_ros2.viz_3d import PylonRacingViz3D
        viz = PylonRacingViz3D(
            pylons=course_data["pylons"], gates=course_data["gates"],
            pylon_names=course_data["pylon_names"], bounds_rect=course_data["bounds_rect"],
            pylon_height_m=course_data["pylon_height_m"],
            pylon_radius_m=course_data["pylon_radius_m"],
        )

    # Training
    if args.train > 0:
        print(f"\n--- Training: {args.train} episodes ---")
        best_gates = 0
        window = deque(maxlen=50)
        for ep in range(args.train):
            r, gates, steps = run_fn(train_env, agent, course_data, train=True, viz=viz)
            window.append(gates)
            if gates > best_gates:
                best_gates = gates
            avg = sum(window) / len(window)
            if (ep+1) % 25 == 0 or ep == 0 or gates > 0:
                print(f"  ep {ep+1:4d}: reward={r:8.1f}  gates={gates}"
                      f"  steps={steps:4d}  eps={agent.epsilon:.3f}"
                      f"  best={best_gates}  avg50={avg:.1f}")
                sys.stdout.flush()

    # Evaluation
    print(f"\n--- Evaluation: {args.eval_episodes} episodes ---")
    eval_gates = []
    for ep in range(args.eval_episodes):
        r, gates, steps = run_fn(train_env, agent, course_data, train=False, viz=viz)
        eval_gates.append(gates)
        print(f"  eval {ep+1}: reward={r:8.1f}  gates={gates}  steps={steps}")
    print(f"  avg gates: {np.mean(eval_gates):.1f}")

    if viz:
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        viz.close()

    train_env.close()
    if args.ros2:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
