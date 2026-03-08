"""
Wrapper used by make_pylon_env. Passes through the base env's observation unchanged.
No ROS2 dependencies — safe to import when use_ros2=False.
Both mock and ROS2 base envs use 15D: [x, y, z, vx, vy, vz, roll, pitch, yaw, v, gamma, vdot, p, q, r].
"""
import gymnasium as gym


class PylonRacingWrapper(gym.Wrapper):
    """Pass-through wrapper. Exposes the base env's full observation (15D for mock and ROS2)."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
