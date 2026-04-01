import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf_transformations import euler_from_quaternion

class PylonRacingEnv(gym.Env):
    def __init__(self):
        super(PylonRacingEnv, self).__init__()
        self.action_space = spaces.Discrete(7)
        
        # Define 7 discrete actions mapping to [aileron, elevator, thrust, rudder]
        self.discrete_action_map = {
            0: [ 0.0,  0.0, 0.7, 0.0], # Cruise
            1: [ 0.0,  0.5, 0.7, 0.0], # Pitch up
            2: [ 0.0, -0.5, 0.7, 0.0], # Pitch down
            3: [-0.5,  0.0, 0.7, 0.0], # Roll left
            4: [ 0.5,  0.0, 0.7, 0.0], # Roll right
            5: [ 0.0,  0.0, 1.0, 0.0], # Throttle up
            6: [ 0.0,  0.0, 0.0, 0.0]  # Throttle down
        }
        
        # Pylon Gates: A set of waypoints to fly through (x, y, z)
        self.gates = np.array([
            [ 50.0,   0.0, 10.0],
            [ 50.0,  50.0, 10.0],
            [-50.0,  50.0, 10.0],
            [-50.0,   0.0, 10.0]
        ], dtype=np.float32)
        self.current_gate_idx = 0
        # UPGRADED: 15 Dimensions 
        # [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.node = rclpy.create_node('pylon_gym_node')
        
        self.pub = self.node.create_publisher(Joy, '/sim/auto_joy', 10)
        self.sub = self.node.create_subscription(Odometry, '/sim/odom', self._odom_cb, 10)
        
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')
        self.current_odom = None
        self.has_taken_off = False 
        
        # State tracking for finite differences and filters
        self.prev_t = None
        self.prev_state = None
        self.filtered_state = np.zeros(15, dtype=np.float32)
        self.last_filtered_state = np.zeros(15, dtype=np.float32)

    @staticmethod
    def _angdiff(a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def _odom_cb(self, msg):
        self.current_odom = msg
        
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_t is None:
            self.prev_t = t - 0.01
            
        dt = max(t - self.prev_t, 0.01)
        self.prev_t = t

        # Extract Raw Position & Orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.prev_state is None:
            self.prev_state = np.array([x, y, z, roll, pitch, yaw, 0.0]) # Last 0.0 is speed

        # Calculate Finite Differences
        vx = (x - self.prev_state[0]) / dt
        vy = (y - self.prev_state[1]) / dt
        vz = (z - self.prev_state[2]) / dt
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        p_rate = self._angdiff(roll, self.prev_state[3]) / dt
        q_rate = self._angdiff(pitch, self.prev_state[4]) / dt
        r_rate = self._angdiff(yaw, self.prev_state[5]) / dt

        denom = max(v, 1e-5)
        gamma = np.arcsin(np.clip(vz / denom, -1.0, 1.0))
        vdot = (v - self.prev_state[6]) # Acceleration

        # Assemble Raw 15D Vector
        raw_state = np.array([
            x, y, z, roll, pitch, yaw, 
            vx, vy, vz, v, gamma, vdot, 
            p_rate, q_rate, r_rate
        ], dtype=np.float32)

        # Low Pass Filter (10Hz)
        fc = 10.0  
        alpha = np.exp(-2 * np.pi * fc * dt)
        
        self.filtered_state = alpha * raw_state + (1.0 - alpha) * self.last_filtered_state
        self.last_filtered_state = self.filtered_state.copy()

        # Update previous state for next callback
        self.prev_state = np.array([x, y, z, roll, pitch, yaw, v])

    def _get_obs(self):
        if self.current_odom is None: 
            return np.zeros(15, dtype=np.float32)
        return self.filtered_state.copy()

    def step(self, action):
        cont_action = self.discrete_action_map[int(action)]
        joy_msg = Joy()
        # Negate elevator: fixedwing_sim.py auto_joy_callback negates axes[1]
        # (input_auto[1] = -msg.axes[1]), so we pre-negate so the agent's
        # positive elevator = climb convention is preserved end-to-end.
        joy_msg.axes = [
            float(action[0]),
            float(-action[1]),
            float(action[2]),
            float(action[3]),
            2000.0
        ]
        self.pub.publish(joy_msg)
        
        rclpy.spin_once(self.node, timeout_sec=0.1)
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
            
        # Dense shaping: penalize distance to current gate
        reward += -0.1 * dist_to_gate
        
        # Crash detection (Z altitude < 0 or negative)
        if obs[2] < 0.0:
            done = True
            reward -= 500.0
            
        return obs, float(reward), done, False, {}

    def reset(self, seed=None, options=None):
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            self.reset_client.call_async(Empty.Request())
        
        self.current_odom = None
        while self.current_odom is None:
            rclpy.spin_once(self.node)
            
        self.has_taken_off = False 
        self.current_gate_idx = 0
        self.prev_t = None
        self.prev_state = None
        self.filtered_state = np.zeros(15, dtype=np.float32)
        self.last_filtered_state = np.zeros(15, dtype=np.float32)
        
        return self._get_obs(), {}