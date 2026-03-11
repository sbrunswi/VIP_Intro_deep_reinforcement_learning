"""
pylon_env_physics.py
--------------------
Physics-constrained Gymnasium wrapper for the no_ros2 mock environment.

Wraps MockPylonRacingEnv (or the ROS2 env via env_factory) and adds:
  1. Actuator rate limiting  (servo speed)
  2. Stall detection & penalty
  3. Structural g-limit termination
  4. Overspeed termination
  5. Dryden wind/turbulence disturbance
  6. Shaped envelope reward terms
  7. Extended 21-D observation (15-D base + 6 flight-envelope states)

Observation layout (21D):
  [0-14]   base 15-D  (x,y,z,roll,pitch,yaw,vx,vy,vz,v,γ,v̇,p,q,r)
  [15]     alpha       – angle of attack   (rad)
  [16]     beta        – sideslip angle    (rad)
  [17]     nz          – load factor       (g)
  [18]     stall_frac  – 0=nominal, >1=stalled
  [19]     ige_factor  – ground-effect lift multiplier
  [20]     wind_head   – headwind component (m/s)

Action space (4D, same as base):
  [aileron, elevator, throttle, rudder]  –1..1  (throttle clamped 0..1)

Extra termination conditions:
  - Deep stall:         v < V_STALL * 0.7
  - Overspeed:          v > V_NE * 1.05
  - Over-g:             nz > G_LIMIT_POS or nz < G_LIMIT_NEG
  - Sustained inverted: |roll| > 170 deg for > 1 s

Usage
-----
    from no_ros2.environments.pylon_env_physics import PylonPhysicsEnv
    env = PylonPhysicsEnv()                        # pure mock, no wind
    env = PylonPhysicsEnv(wind_speed=3.0)          # with wind
    env = PylonPhysicsEnv(use_ros2=True)           # swap in real ROS2 base
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from no_ros2.environments.flight_dynamics import (
    FlightDynamics, EnvelopeReward,
    V_STALL_MS, V_NE_MS,
    G_LIMIT_POS, G_LIMIT_NEG,
    MASS_KG, G,
)

# Base 15-D observation index map
IDX = dict(
    x=0, y=1, z=2,
    roll=3, pitch=4, yaw=5,
    vx=6, vy=7, vz=8,
    v=9, gamma=10, vdot=11,
    p=12, q=13, r=14,
)


class PylonPhysicsEnv(gym.Env):
    """
    Physics-constrained wrapper around MockPylonRacingEnv (or ROS2 env).
    Drop-in replacement — same action/obs API as the base env, just extended.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        use_ros2: bool = False,
        wind_speed: float = 0.0,
        wind_dir_deg: float = 0.0,
        turbulence_intensity: float = 0.10,
        envelope_reward_weight: float = 1.0,
        dt: float = 0.1,
        seed: int = 42,
        **base_kwargs,
    ):
        super().__init__()

        # Build base env (mirrors env_factory.py pattern)
        if use_ros2:
            from auav_pylon_2026.pylon_env import PylonRacingEnv
            self._base = PylonRacingEnv(**base_kwargs)
        else:
            from no_ros2.environments.mock_pylon_env import MockPylonRacingEnv
            self._base = MockPylonRacingEnv(dt=dt, seed=seed, **base_kwargs)

        self._dynamics = FlightDynamics(
            wind_speed_ms=wind_speed,
            wind_dir_rad=np.deg2rad(wind_dir_deg),
            turbulence_intensity=turbulence_intensity,
            seed=seed,
        )

        self.envelope_reward_weight = envelope_reward_weight
        self._dt = dt

        # Spaces — same action space as base, extended 21-D obs
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0, 1.0,  1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # Internal state
        self._prev_action    = np.zeros(4, dtype=np.float32)
        self._inverted_timer = 0.0
        self._alpha_filt     = 0.0
        self._beta_filt      = 0.0
        self._step_count     = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs15, info = self._base.reset(seed=seed, options=options)
        self._prev_action    = np.zeros(4, dtype=np.float32)
        self._inverted_timer = 0.0
        self._alpha_filt     = 0.0
        self._beta_filt      = 0.0
        self._step_count     = 0
        return self._augment_obs(obs15, self._prev_action), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # 1. Rate-limit & clip actuators before passing to base env
        action = self._dynamics.enforce_actuator_limits(
            action, self._prev_action, self._dt
        )
        self._prev_action = action.copy()

        # 2. Step base env
        obs15, base_reward, terminated, truncated, info = self._base.step(action)

        # 3. Derive physics quantities
        v    = float(obs15[IDX['v']])
        z    = float(obs15[IDX['z']])
        roll = float(obs15[IDX['roll']])
        p    = float(obs15[IDX['p']])
        q    = float(obs15[IDX['q']])
        r    = float(obs15[IDX['r']])

        self._update_alpha_beta(obs15)

        aero     = self._dynamics.compute_aero_forces(
            v, self._alpha_filt, self._beta_filt,
            elevator=float(action[1]),
            throttle=float(action[2]),
        )
        lift_ige = self._dynamics.apply_ground_effect(aero['lift'], z)
        nz       = lift_ige / max(MASS_KG * G, 1e-3)

        _, stall_frac = self._dynamics.check_stall(v, self._alpha_filt)
        env_ok, env_info = self._dynamics.is_envelope_ok(v, self._alpha_filt, lift_ige)
        info.update(env_info)

        # 4. Extra termination checks
        if abs(roll) > np.deg2rad(170.0):
            self._inverted_timer += self._dt
        else:
            self._inverted_timer = 0.0

        physics_term = (
            not env_ok
            or v < V_STALL_MS * 0.7
            or v > V_NE_MS * 1.05
            or self._inverted_timer > 1.0
        )
        terminated = terminated or physics_term
        info['physics_termination'] = physics_term

        # 5. Envelope-shaped reward
        envelope_penalty = EnvelopeReward.compute_all(
            stall_frac, nz, v, z, p, q, r
        )
        reward = base_reward + self.envelope_reward_weight * envelope_penalty
        info['envelope_penalty'] = envelope_penalty

        # 6. Build augmented observation
        obs21 = self._augment_obs(obs15, action)

        self._step_count += 1
        return obs21, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if hasattr(self._base, 'node'):
            self._base.node.destroy_node()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_alpha_beta(self, obs15: np.ndarray):
        """Low-pass filter alpha/beta estimates at same 10 Hz cutoff as base env."""
        v     = float(obs15[IDX['v']])
        vy    = float(obs15[IDX['vy']])
        pitch = float(obs15[IDX['pitch']])
        gamma = float(obs15[IDX['gamma']])

        alpha_raw = pitch - gamma
        beta_raw  = np.arcsin(np.clip(vy / max(v, 1e-3), -1.0, 1.0))

        fc      = 10.0
        alpha_k = np.exp(-2 * np.pi * fc * self._dt)
        self._alpha_filt = alpha_k * self._alpha_filt + (1.0 - alpha_k) * alpha_raw
        self._beta_filt  = alpha_k * self._beta_filt  + (1.0 - alpha_k) * beta_raw

    def _augment_obs(self, obs15: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Append 6 physics states to the base 15-D vector → 21-D."""
        v   = float(obs15[IDX['v']])
        vx  = float(obs15[IDX['vx']])
        vy  = float(obs15[IDX['vy']])
        vz  = float(obs15[IDX['vz']])
        z   = float(obs15[IDX['z']])

        aero       = self._dynamics.compute_aero_forces(
            v, self._alpha_filt, self._beta_filt,
            elevator=float(action[1]),
            throttle=float(action[2]),
        )
        lift_ige   = self._dynamics.apply_ground_effect(aero['lift'], z)
        nz         = lift_ige / max(MASS_KG * G, 1e-3)
        ige_factor = lift_ige / max(aero['lift'], 1e-6)
        _, stall_frac = self._dynamics.check_stall(v, self._alpha_filt)

        wind = self._dynamics.get_wind_disturbance(v, self._dt)
        if v > 1e-3:
            vel_unit  = np.array([vx, vy, vz]) / v
            wind_head = float(np.dot(wind, vel_unit))
        else:
            wind_head = 0.0

        extra = np.array([
            self._alpha_filt,   # [15] alpha  rad
            self._beta_filt,    # [16] beta   rad
            nz,                 # [17] load factor (g)
            stall_frac,         # [18] stall fraction (>1 = stalled)
            ige_factor,         # [19] ground-effect lift multiplier
            wind_head,          # [20] headwind m/s
        ], dtype=np.float32)

        return np.concatenate([obs15, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    # Passthrough properties
    # ------------------------------------------------------------------

    @property
    def node(self):
        """Expose ROS2 node if using the ROS2 base env."""
        return getattr(self._base, 'node', None)

    @property
    def has_taken_off(self):
        return getattr(self._base, 'has_taken_off', True)