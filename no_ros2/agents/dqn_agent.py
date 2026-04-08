#!/usr/bin/env python3
"""
DQN agent for pylon racing with inner-loop stabilization.

Uses a small MLP to map a compact state → Q-values for discrete
high-level actions. An inner-loop PD controller converts each action
into raw (aileron, elevator, throttle, rudder) commands.

Pure numpy implementation (no PyTorch/TF).

Run standalone:
  python no_ros2/agents/dqn_agent.py --train 500
  python no_ros2/agents/dqn_agent.py --env no_ros2/environments_v2/mock_pylon_env.py
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

# ---------------------------------------------------------------------------
# Inner-loop stabilizer
# ---------------------------------------------------------------------------

TARGET_ALT_M   = PYLON_MID_HEIGHT_M
ALT_MARGIN_M   = 2.0
TRIM_PITCH_DEG = 2.0
TRIM_THROTTLE  = 0.35

# Actions: (heading_offset_deg, throttle_boost)
# heading_offset is relative to the bearing toward the current target pylon.
# 0 = fly directly at target; positive = aim right of target; negative = aim left
DISCRETE_ACTIONS = [
    ( 0.0,  0.00),    # 0: fly at target
    ( 20.0, 0.00),    # 1: aim 20 deg right of target
    (-20.0, 0.00),    # 2: aim 20 deg left of target
    ( 45.0, 0.00),    # 3: aim 45 deg right (wide approach)
    (-45.0, 0.00),    # 4: aim 45 deg left
    ( 0.0,  0.15),    # 5: fly at target + speed up
    ( 0.0, -0.10),    # 6: fly at target + slow down
]
N_ACTIONS = len(DISCRETE_ACTIONS)

_KP_PITCH  = 0.12
_KD_PITCH  = 0.6
_KP_ROLL   = 0.06
_KD_ROLL   = 0.3
_KP_YAW    = 0.02
_KP_ALT    = 2.0
_MAX_ALT_CORR = 8.0
_KP_HEADING = 1.0   # deg bank per deg heading error
_MAX_BANK   = 30.0  # max bank angle


def action_to_controls(action_idx, obs, bearing_rad=0.0):
    """
    Convert a discrete action + current bearing to raw controls.
    bearing_rad: bearing to target pylon in radians (positive = target is left).
    """
    heading_offset_deg, thr_boost = DISCRETE_ACTIONS[action_idx]
    roll_deg  = np.degrees(obs[3])
    pitch_deg = np.degrees(obs[4])
    z         = obs[2]
    p, q, r   = obs[12], obs[13], obs[14]
    speed     = obs[9]

    # Compute desired heading error (bearing + action offset)
    bearing_deg = np.degrees(bearing_rad)
    heading_err = bearing_deg + heading_offset_deg
    heading_err = (heading_err + 180.0) % 360.0 - 180.0

    # Roll sign: in this ENU env, positive roll = LEFT turn, negative roll = RIGHT turn
    # If heading_err > 0 (target is LEFT), we need positive roll (LEFT turn)
    # If heading_err < 0 (target is RIGHT), we need negative roll (RIGHT turn)
    if abs(heading_err) >= 90.0:
        target_roll_deg = np.sign(heading_err) * _MAX_BANK
    else:
        target_roll_deg = np.clip(_KP_HEADING * heading_err, -_MAX_BANK, _MAX_BANK)

    # Altitude controller
    alt_err = TARGET_ALT_M - z
    alt_pitch_corr = np.clip(_KP_ALT * alt_err, -_MAX_ALT_CORR, _MAX_ALT_CORR)
    bank_nose_down = -abs(roll_deg) * 0.1
    target_pitch = TRIM_PITCH_DEG + alt_pitch_corr + bank_nose_down
    pitch_err = target_pitch - pitch_deg
    elevator = np.clip(_KP_PITCH * pitch_err - _KD_PITCH * q, -1.0, 1.0)

    roll_err = target_roll_deg - roll_deg
    aileron = np.clip(_KP_ROLL * roll_err - _KD_ROLL * p, -1.0, 1.0)

    rudder = np.clip(_KP_YAW * roll_deg, -1.0, 1.0)

    thr = TRIM_THROTTLE + thr_boost
    thr += abs(roll_deg) / 90.0 * 0.15
    thr += (7.0 - speed) * 0.05
    throttle = np.clip(thr, 0.0, 1.0)

    return np.array([aileron, elevator, throttle, rudder], dtype=np.float32)


# ---------------------------------------------------------------------------
# Compact state: 7D
# ---------------------------------------------------------------------------

PASS_THRESHOLD_M = 6.0
MAX_STEPS        = 2000
STATE_DIM        = 7


def make_state(obs, target_pylon_idx, pylons):
    """
    Compact 7D state for navigation:
      [sin(bearing), cos(bearing), log(dist), alt_error, roll/pi, pitch/pi, speed/10]
    """
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]
    v       = obs[9]

    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

    vel_angle = np.arctan2(vy, vx)
    tgt_angle = np.arctan2(dy, dx)
    bearing = tgt_angle - vel_angle
    bearing = (bearing + np.pi) % (2*np.pi) - np.pi

    state = np.array([
        np.sin(bearing),                    # [-1, 1] turn direction
        np.cos(bearing),                    # [-1, 1] alignment (1 = heading right at it)
        np.log1p(dist) / 5.0,              # ~[0, 1] distance (log-scaled)
        (z - TARGET_ALT_M) / 5.0,          # altitude error
        obs[3] / np.pi,                    # roll normalized
        obs[4] / (np.pi/2),               # pitch normalized
        v / 10.0,                          # speed normalized
    ], dtype=np.float32)
    return state


# ---------------------------------------------------------------------------
# Numpy MLP with Adam optimizer
# ---------------------------------------------------------------------------

def _relu(x):
    return np.maximum(0, x)


def _he_init(fan_in, fan_out, rng):
    return rng.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)


class NumpyMLP:
    """2-hidden-layer MLP with ReLU."""

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

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        data = np.load(path)
        for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            setattr(self, k, data[k])


class AdamOptimizer:
    """Adam optimizer for a list of numpy arrays."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        states      = np.array([b[0] for b in batch], dtype=np.float32)
        actions     = np.array([b[1] for b in batch], dtype=np.int32)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones       = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS,
                 hidden1=64, hidden2=32,
                 lr=5e-4, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.985,
                 batch_size=64, target_update=50,
                 buffer_size=30000):
        rng = np.random.RandomState(42)
        self.q_net = NumpyMLP(state_dim, hidden1, hidden2, n_actions, rng)
        self.target_net = NumpyMLP(state_dim, hidden1, hidden2, n_actions, rng)
        self.target_net.copy_from(self.q_net)
        self.optimizer = AdamOptimizer(self.q_net.params(), lr=lr)

        self.replay = ReplayBuffer(buffer_size)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
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

        # Forward: current Q
        h1_pre = states @ self.q_net.W1 + self.q_net.b1
        h1 = _relu(h1_pre)
        h2_pre = h1 @ self.q_net.W2 + self.q_net.b2
        h2 = _relu(h2_pre)
        q_all = h2 @ self.q_net.W3 + self.q_net.b3
        q_values = q_all[np.arange(batch), actions]

        # Target Q
        q_next = self.target_net.forward(next_states)
        q_target = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # Huber loss gradient (clip TD error)
        td_error = q_target - q_values
        loss = np.mean(td_error ** 2)
        td_clipped = np.clip(td_error, -10.0, 10.0)

        # Backprop (mean gradients, not sum)
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

        # Global gradient clipping
        grads = [dW1, db1, dW2, db2, dW3, db3]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > 5.0:
            scale = 5.0 / total_norm
            grads = [g * scale for g in grads]

        self.optimizer.step(grads)

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.copy_from(self.q_net)

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self, path):
        self.q_net.save(path)
        print(f"Policy saved to {path}")

    def load_policy(self, path):
        self.q_net.load(path)
        self.target_net.copy_from(self.q_net)
        self.optimizer = AdamOptimizer(self.q_net.params(), lr=self.optimizer.lr)
        print(f"Policy loaded from {path}")


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def get_reward_shaping(obs, target_pylon_idx, prev_dist, pylons, bounds_rect):
    x, y, z = obs[0], obs[1], obs[2]
    vx, vy  = obs[6], obs[7]

    target = pylons[target_pylon_idx]
    dx = target[0] - x
    dy = target[1] - y
    dist = np.sqrt(dx*dx + dy*dy) + 1e-6

    passed = dist < PASS_THRESHOLD_M
    if passed:
        return 100.0, True

    base = 0.0

    # Distance shaping: reward getting closer
    if prev_dist is not None:
        delta = prev_dist - dist
        base += np.clip(delta * 3.0, -2.0, 2.0)

    # Alignment: reward flying toward target
    speed_xy = np.sqrt(vx*vx + vy*vy) + 1e-6
    align = (dx/dist * vx + dy/dist * vy) / speed_xy
    base += 0.5 * align

    # Proximity bonus
    if dist < 15.0:
        base += 1.5 * (1.0 - dist / 15.0)

    # Bounds penalty
    g = bounds_rect
    if x < g["min_x"] or x > g["max_x"] or y < g["min_y"] or y > g["max_y"]:
        base -= 5.0

    # Altitude
    alt_err = abs(z - TARGET_ALT_M)
    if alt_err > ALT_MARGIN_M * 2:
        base -= 0.5

    return base, False


# Compatibility with evaluate_environment.py
def obs_to_state(obs, target_pylon_idx, pylons, bounds_rect):
    return make_state(obs, target_pylon_idx, pylons)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, course_data, train=True, max_steps=MAX_STEPS):
    pylons      = course_data["pylons"]
    bounds_rect = course_data["bounds_rect"]
    n_pylons    = len(pylons)

    obs, _ = env.reset()
    target_idx = 0
    prev_dist  = None
    total_reward = 0.0
    pylons_passed = 0

    for step in range(max_steps):
        state = make_state(obs, target_idx, pylons)
        action_idx = agent.get_action(state, greedy=not train)

        # Compute bearing to target for inner-loop heading tracker
        x, y = obs[0], obs[1]
        vx, vy = obs[6], obs[7]
        tgt = pylons[target_idx]
        dx_t, dy_t = tgt[0] - x, tgt[1] - y
        bearing_rad = np.arctan2(dy_t, dx_t) - np.arctan2(vy, vx)
        bearing_rad = (bearing_rad + np.pi) % (2*np.pi) - np.pi

        controls = action_to_controls(action_idx, obs, bearing_rad)

        obs, r_env, terminated, truncated, _ = env.step(controls)
        done = terminated or truncated

        extra, passed = get_reward_shaping(obs, target_idx, prev_dist, pylons, bounds_rect)
        reward = extra
        if terminated and r_env < 0:
            reward += r_env

        if passed:
            pylons_passed += 1
            target_idx = (target_idx + 1) % n_pylons
            prev_dist = None
        else:
            tgt = pylons[target_idx]
            prev_dist = np.sqrt((tgt[0]-obs[0])**2 + (tgt[1]-obs[1])**2)

        next_state = make_state(obs, target_idx, pylons)

        if train:
            clipped_reward = np.clip(reward, -10.0, 10.0)
            agent.replay.push(state, action_idx, clipped_reward, next_state, float(done))
            if step % 4 == 0:
                agent.train_step()

        total_reward += reward
        if done:
            break

    if train:
        agent.decay_epsilon()

    return total_reward, pylons_passed, step + 1


# ---------------------------------------------------------------------------
# Env loader helper
# ---------------------------------------------------------------------------

def _make_env_from_file(env_path, course):
    env_path = os.path.abspath(env_path)
    spec = importlib.util.spec_from_file_location("_env_module", env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return PylonRacingWrapper(mod.MockPylonRacingEnv(course=course))


# Alias for evaluate_environment.py compatibility
class QLearningAgent(DQNAgent):
    """Alias so evaluate_environment.py can instantiate by the expected name."""
    def __init__(self, **kwargs):
        # Use DQN-appropriate defaults regardless of what evaluate_environment passes
        super().__init__(
            lr=5e-4, gamma=0.95, epsilon=1.0,
            epsilon_min=0.05, epsilon_decay=0.985,
            batch_size=64, target_update=50,
            hidden1=64, hidden2=32
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DQN agent for pylon racing")
    parser.add_argument("--train",         type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--env",           metavar="FILE", default=None)
    parser.add_argument("--save-policy",   metavar="FILE", default=None)
    parser.add_argument("--load-policy",   metavar="FILE", default=None)
    parser.add_argument("--ros2",          action="store_true")
    parser.add_argument("--viz",           action="store_true")
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument("--purt",   action="store_true")
    course_group.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    course_name = "purt" if args.purt else "sample"
    course_data = get_course(course_name)
    print(f"Course: {course_name} ({len(course_data['pylons'])} pylons)")

    if args.ros2:
        import rclpy
        rclpy.init()

    agent = DQNAgent(lr=5e-4, gamma=0.95, epsilon=1.0,
                     epsilon_min=0.05, epsilon_decay=0.985,
                     batch_size=64, target_update=50,
                     hidden1=64, hidden2=32)

    if args.load_policy:
        agent.load_policy(args.load_policy)

    # Training
    if not args.load_policy and args.train > 0:
        print(f"\n--- Training: {args.train} episodes ---")
        if args.env:
            train_env = _make_env_from_file(args.env, course_name)
        else:
            train_env = make_pylon_env(use_ros2=args.ros2, course=course_name)

        best_pylons = 0
        best_avg50 = 0.0
        total_pylons = 0
        window_pylons = deque(maxlen=50)
        for ep in range(args.train):
            r, passed, steps = run_episode(train_env, agent, course_data, train=True)
            total_pylons += passed
            window_pylons.append(passed)
            if passed > best_pylons:
                best_pylons = passed
            avg_p = sum(window_pylons) / len(window_pylons)
            # Save best checkpoint
            if len(window_pylons) >= 50 and avg_p > best_avg50:
                best_avg50 = avg_p
                if args.save_policy:
                    agent.q_net.save(args.save_policy + ".best")
            if (ep+1) % 25 == 0 or ep == 0 or passed > 0:
                print(f"  ep {ep+1:4d}: reward={r:8.1f}  pylons={passed}"
                      f"  steps={steps:4d}  eps={agent.epsilon:.3f}"
                      f"  best={best_pylons}  avg50={avg_p:.1f}")
                sys.stdout.flush()

        train_env.close()
        print(f"Training complete. Total pylons: {total_pylons}  best_avg50: {best_avg50:.1f}\n")
        if args.save_policy:
            agent.save_policy(args.save_policy)
            # Load best checkpoint for evaluation
            best_path = args.save_policy + ".best.npz"
            if os.path.exists(best_path):
                print(f"Loading best checkpoint (avg50={best_avg50:.1f})")
                agent.load_policy(best_path)

    # Evaluation
    print("--- Evaluation ---")
    if args.env:
        eval_env = _make_env_from_file(args.env, course_name)
    else:
        eval_env = make_pylon_env(use_ros2=args.ros2, course=course_name)

    eval_pylons = []
    for ep in range(args.eval_episodes):
        r, passed, steps = run_episode(eval_env, agent, course_data, train=False)
        eval_pylons.append(passed)
        print(f"  eval {ep+1}: reward={r:8.1f}  pylons={passed}  steps={steps}")
    print(f"  avg pylons: {np.mean(eval_pylons):.1f}")

    eval_env.close()
    if args.ros2:
        import rclpy
        rclpy.shutdown()


if __name__ == "__main__":
    main()
