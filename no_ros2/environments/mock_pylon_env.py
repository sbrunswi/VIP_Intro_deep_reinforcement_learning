"""
Mock pylon racing environment with no ROS2 dependency.
Same action space and step/reset/done/reward semantics as PylonRacingEnv.
Observation is 15D in the same order as ROS2 PylonRacingEnv:
  [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r].
Simple deterministic dynamics for testing agents without Gazebo.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from no_ros2.environments.pylon_course import get_course, DEFAULT_COURSE, PYLON_HEIGHT_M

# Observation layout (15D, matches ROS2 PylonRacingEnv / TECS actual_data)
# [0:3]   x, y, z (position, m)
# [3:6]   roll, pitch, yaw (rad)
# [6:9]   vx, vy, vz (velocity, m/s)
# [9]     v = |velocity| (m/s)
# [10]    gamma = flight path angle (rad)
# [11]    vdot = d|v|/dt (m/s^2)
# [12:15] p, q, r = roll, pitch, yaw rate (rad/s)
OBS_SIZE = 15


class MockPylonRacingEnv(gym.Env):
    """Drop-in replacement for PylonRacingEnv with no rclpy/ROS2. Same action space; 15D observation."""

    def __init__(self, dt=0.1, seed=None, course=DEFAULT_COURSE):
        self._course = get_course(course)
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.dt = dt
        self._max_alt_m = self._course["pylon_height_m"] * 0.5
        self._rng = np.random.default_rng(seed)
        # Core state: x, y, z, vx, vy, vz
        self._state = np.zeros(6, dtype=np.float32)
        self._heading = 0.0  # rad, world frame: angle of velocity in xy plane
        self.has_taken_off = False
        # For 15D obs: prev values for finite differences
        self._prev_speed = 0.0
        self._prev_heading = 0.0
        self._roll = 0.0   # rad, from coordinated-turn model
        self._turn_rate = 0.0  # rad/s

    def _get_obs(self):
        x, y, z = self._state[0], self._state[1], self._state[2]
        vx, vy, vz = self._state[3], self._state[4], self._state[5]
        v = np.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
        gamma = np.arcsin(np.clip(vz / v, -1.0, 1.0))
        vdot = (v - self._prev_speed) / self.dt if self.dt > 0 else 0.0
        # yaw = heading; pitch ≈ gamma (nose-up positive); roll from coordinated turn
        yaw = self._heading
        pitch = gamma
        roll = self._roll
        # Angular rates: r = turn rate; p, q = 0 in this simple model
        r = self._turn_rate
        p = 0.0
        q = 0.0
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0], obs[1], obs[2] = x, y, z
        obs[3], obs[4], obs[5] = roll, pitch, yaw
        obs[6], obs[7], obs[8] = vx, vy, vz
        obs[9] = v
        obs[10] = gamma
        obs[11] = vdot
        obs[12], obs[13], obs[14] = p, q, r
        return obs

    def _integrate(self, action):
        aileron, elevator, throttle, rudder = action[0], action[1], action[2], action[3]
        vx, vy, vz = self._state[3], self._state[4], self._state[5]
        speed_xy = np.sqrt(vx * vx + vy * vy) + 1e-6
        self._prev_heading = self._heading
        self._prev_speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        self._heading = np.arctan2(vy, vx)
        # Commands: forward speed, climb rate, turn rate
        speed_xy_cmd = 2.0 + 3.0 * max(0.0, throttle)
        vz_cmd = elevator * 4.0 + (throttle - 0.5) * 2.0
        self._turn_rate = rudder * 1.5  # rad/s
        tau = 0.5
        speed_xy += (speed_xy_cmd - speed_xy) * (self.dt / tau)
        speed_xy = max(0.0, speed_xy)
        self._heading += self._turn_rate * self.dt
        self._state[5] += (vz_cmd - self._state[5]) * (self.dt / tau)
        self._state[3] = speed_xy * np.cos(self._heading)
        self._state[4] = speed_xy * np.sin(self._heading)
        self._state[0] += self._state[3] * self.dt
        self._state[1] += self._state[4] * self.dt
        self._state[2] += self._state[5] * self.dt
        # Coordinated-turn roll: roll = atan(v * r / g)
        v_after = np.sqrt(self._state[3]**2 + self._state[4]**2 + self._state[5]**2) + 1e-9
        self._roll = np.arctan2(v_after * self._turn_rate, 9.81)
        self._roll = np.clip(self._roll, -np.deg2rad(60), np.deg2rad(60))
        # Ground clamp
        if self._state[2] < 0.0:
            self._state[2] = 0.0
            self._state[5] = min(0.0, self._state[5])
        # Ceiling clamp (race altitude limit)
        if self._state[2] > self._max_alt_m:
            self._state[2] = self._max_alt_m
            self._state[5] = min(0.0, self._state[5])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._integrate(action)
        obs = self._get_obs()
        z = obs[2]
        if z > 1.0:
            self.has_taken_off = True
        terminated = bool(
            (self.has_taken_off and z < 0.1) or (z < -0.5)
        )
        if terminated:
            reward = -10.0
        elif self.has_taken_off:
            reward = 1.0
        else:
            reward = 0.0
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = np.zeros(6, dtype=np.float32)
        self._state[2] = 2.0  # start at 2 m altitude so "level" actions hold altitude (no need to climb from 0)
        self._state[3] = 4.0  # initial vx (m/s) so we have speed and defined heading
        self._state[5] = 0.0  # initial vz = 0 (level)
        self._heading = 0.0
        self._roll = 0.0
        self._turn_rate = 0.0
        self._prev_speed = 0.0
        self._prev_heading = 0.0
        self.has_taken_off = True  # already in the air
        return self._get_obs(), {}
