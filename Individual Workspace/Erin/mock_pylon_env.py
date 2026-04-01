"""
Erin’s custom mock pylon racing environment for fixed-wing drones.
Uses basic RL strategies: sparse rewards for gate crossings, penalties for crashes.
Observation: 15D state + target gate vector (18D total).
Dynamics: simplified fixed-wing physics with stall detection.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from no_ros2.environments.pylon_course import DEFAULT_GATES, DEFAULT_PYLONS, FINISH_GATE_INDEX, get_pylon_tops

OBS_SIZE = 17  # 15D state + 2D target vector
MAX_ALT_M = 7.0

class MockPylonRacingEnv(gym.Env):
    def __init__(self, dt=0.1, seed=None):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(6, dtype=np.float32)  # x,y,z,vx,vy,vz
        self._heading = 0.0
        self._roll = 0.0
        self._turn_rate = 0.0
        self._prev_speed = 0.0
        self._gates = DEFAULT_GATES
        self._pylons = DEFAULT_PYLONS
        self._current_gate = 0
        self._laps = 0
        self._gate_crossings = set()

    def _get_obs(self):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        x, y, z = self._state[0], self._state[1], self._state[2]
        vx, vy, vz = self._state[3], self._state[4], self._state[5]
        v = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-9
        gamma = np.arcsin(np.clip(vz / v, -1.0, 1.0))
        vdot = (v - self._prev_speed) / self.dt
        yaw = self._heading
        pitch = gamma
        roll = self._roll
        r = self._turn_rate
        p = q = 0.0
        obs[0:15] = [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r]
        # Target: vector to next gate midpoint (2D in xy plane)
        g1, g2 = self._gates[self._current_gate]
        p1, p2 = self._pylons[g1][:2], self._pylons[g2][:2]
        gate_mid = (p1 + p2) / 2
        obs[15:17] = gate_mid - np.array([x, y])
        return obs

    def _integrate(self, action):
        aileron, elevator, throttle, rudder = action
        vx, vy, vz = self._state[3], self._state[4], self._state[5]
        speed_xy = np.sqrt(vx**2 + vy**2) + 1e-6
        self._prev_speed = np.sqrt(vx**2 + vy**2 + vz**2)
        self._heading = np.arctan2(vy, vx)
        # Fixed-wing: throttle controls speed, elevator pitch, rudder turn
        speed_cmd = 5.0 + 3.0 * max(0.0, throttle)
        pitch_rate = elevator * 2.0
        self._turn_rate = rudder * 1.0
        # Simple dynamics
        tau = 0.5
        speed_xy += (speed_cmd - speed_xy) * (self.dt / tau)
        speed_xy = max(2.0, speed_xy)  # min speed
        self._heading += self._turn_rate * self.dt
        pitch = np.arctan2(vz, speed_xy)
        pitch += pitch_rate * self.dt
        pitch = np.clip(pitch, -np.pi/4, np.pi/4)
        vz = speed_xy * np.sin(pitch)
        self._state[3] = speed_xy * np.cos(self._heading)
        self._state[4] = speed_xy * np.sin(self._heading)
        self._state[5] = vz
        self._state[0] += self._state[3] * self.dt
        self._state[1] += self._state[4] * self.dt
        self._state[2] += self._state[5] * self.dt
        # Roll from turn
        self._roll = np.arctan2(speed_xy * self._turn_rate, 9.81)
        self._roll = np.clip(self._roll, -np.pi/3, np.pi/3)
        # Clamps
        self._state[2] = np.clip(self._state[2], 0.0, MAX_ALT_M)

    def _check_gate_crossing(self):
        g1, g2 = self._gates[self._current_gate]
        p1, p2 = self._pylons[g1][:2], self._pylons[g2][:2]
        pos = self._state[:2]
        # Check if crossed line segment
        v1 = p2 - p1
        v2 = pos - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1.0:  # near line
            proj = np.dot(v2, v1) / np.dot(v1, v1)
            if 0 <= proj <= 1:
                gate_key = (self._current_gate, self._laps)
                if gate_key not in self._gate_crossings:
                    self._gate_crossings.add(gate_key)
                    self._current_gate = (self._current_gate + 1) % len(self._gates)
                    if self._current_gate == FINISH_GATE_INDEX:
                        self._laps += 1
                    return True
        return False

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._integrate(action)
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        # Gate reward
        if self._check_gate_crossing():
            reward += 10.0
        # Crash if too slow or ground
        v = obs[9]
        z = obs[2]
        if v < 3.0 or z < 0.5:
            terminated = True
            reward -= 10.0
        # Small penalty for deviation
        target_dist = np.linalg.norm(obs[15:18])
        reward -= 0.01 * target_dist
        return obs, reward, terminated, False, {"laps": self._laps}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = np.zeros(6, dtype=np.float32)
        self._state[2] = 3.0  # start at 3m
        self._state[3] = 5.0  # vx
        self._heading = 0.0
        self._roll = 0.0
        self._turn_rate = 0.0
        self._prev_speed = 5.0
        self._current_gate = 0
        self._laps = 0
        self._gate_crossings = set()
        return self._get_obs(), {}
