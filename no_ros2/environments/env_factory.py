"""
Factory to create a wrapped pylon racing env (mock or ROS2).
Agent code uses make_pylon_env(use_ros2=...) and never touches the base env directly.
Both paths expose the same 15D observation layout as ROS2 PylonRacingEnv:
  [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r].
Action: 4D float (aileron, elevator, throttle, rudder), same semantics and topic when use_ros2=True.
When use_ros2=False, only no_ros2 and gym/numpy are used (no ROS2 or tf_transformations).
"""
from no_ros2.environments.mock_pylon_env import MockPylonRacingEnv
from no_ros2.environments.pylon_wrapper import PylonRacingWrapper


def make_pylon_env(use_ros2=False, **kwargs):
    """Build base env and wrap with PylonRacingWrapper. Caller must rclpy.init() before
    and rclpy.shutdown() after when use_ros2=True.
    """
    if use_ros2:
        from auav_pylon_2026.pylon_env import PylonRacingEnv
        base = PylonRacingEnv(**kwargs)
    else:
        base = MockPylonRacingEnv(**kwargs)
    return PylonRacingWrapper(base)
