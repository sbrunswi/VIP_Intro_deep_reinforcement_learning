"""
FixedWingEnv — Gymnasium wrapper for the fixed-wing CasADi simulation.

Replaces the three-node ROS 2 pipeline
    fixedwing_sim.py  +  sim_tecs_ros_xtrack.py
with a single synchronous gymnasium.Env that drives the CasADi integrator
directly, with no publishers, subscribers, or ROS timers.

Data-flow inside step():
    action  ──►  XTrack_NAV_lookAhead (optional)
             ──►  TECSControl_cub     → AETR (4,)
             ──►  CasADi cvodes       → new state (13,)
             ──►  _state_to_obs()     → observation (15,)
"""

import numpy as np
import casadi as ca
import gymnasium
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from cyecca.models.fixedwing_4ch import derive_model
from .cross_tracker_nav_sample import XTrack_NAV_lookAhead
from .tecs_controller_xtrack_sample import TECSControl_cub


class FixedWingEnv(gymnasium.Env):
    """
    Gymnasium environment for a fixed-wing aircraft.

    Internal state vector — 13 elements (fixedwing_4ch.py convention):
        [0:3]   position_w   — world-frame position     [x, y, z]         (m)
        [3:6]   velocity_b   — body-frame velocity      [vx, vy, vz]      (m/s)
        [6:10]  quat_wb      — world→body quaternion    [qw, qx, qy, qz]
        [10:13] omega_wb_b   — body-frame angular rates [p, q, r]         (rad/s)

    Observation — 15 elements derived from state:
        [x, y, z, roll, pitch, yaw, vx_w, vy_w, vz_w, v, gamma, vdot, p, q, r]

    Action (use_tecs=True — high-level reference targets routed through TECS):
        [des_v (m/s),  des_gamma (rad),  des_heading (rad)]

        To bypass TECS and command AETR directly set use_tecs=False and replace
        action_space with the commented-out 4-D Box below.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        waypoints,
        dt: float = 0.01,
        use_tecs: bool = True,
        tecs_args: str = "sim",
        start_wp_index: int = 0,
    ):
        super().__init__()

        self.dt = dt
        self.use_tecs = use_tecs
        self.waypoints = list(waypoints)
        self._start_wp_index = start_wp_index

        # ── CasADi model ──────────────────────────────────────────────────────
        self.model = derive_model()

        # Build the integrator once; the ODE is autonomous so t0=0, tf=dt gives
        # the same result as the original t0=self.t, tf=self.t+dt used in
        # fixedwing_sim.py, while avoiding reconstruction overhead each step.
        _opts = {"abstol": 1e-2, "reltol": 1e-6, "fsens_err_con": True}
        self._integrator = ca.integrator(
            "fw_gym", "cvodes", self.model["dae"], 0.0, self.dt, _opts
        )

        # ── Default parameters & initial state ───────────────────────────────
        p_dict = self.model["p_defaults"]
        x0_dict = self.model["x0_defaults"]

        # Parameter vector: iterate in the order CasADi assigned to model["p"].
        # This matches the pattern in fixedwing_sim.py exactly.
        self._p0 = np.array(
            [p_dict[str(self.model["p"][i])] for i in range(self.model["p"].shape[0])],
            dtype=float,
        )  # shape (35,)

        # Initial state: dict values are stored in insertion order matching x_index.
        self._x0 = np.array(list(x0_dict.values()), dtype=float)  # shape (13,)

        # ── Optional nav tracker + TECS controller ────────────────────────────
        if self.use_tecs:
            self._nav = XTrack_NAV_lookAhead(dt, self.waypoints, start_wp_index)
            self._tecs = TECSControl_cub(dt, tecs_args)

        # ── Gymnasium spaces ─────────────────────────────────────────────────
        # ACTION — high-level reference targets (fed into TECS → AETR → physics).
        # TODO: tune V_MIN / V_MAX to match your flight envelope.
        _V_MIN, _V_MAX = 5.0, 25.0  # m/s
        self.action_space = spaces.Box(
            low=np.array([_V_MIN, -np.pi / 4, -np.pi], dtype=np.float32),
            high=np.array([_V_MAX,  np.pi / 4,  np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        # ── Direct AETR alternative (bypass TECS, set use_tecs=False):
        # self.action_space = spaces.Box(
        #     low =np.array([-1., -1., 0., -1.], dtype=np.float32),
        #     high=np.array([ 1.,  1., 1.,  1.], dtype=np.float32),
        #     dtype=np.float32,
        # )

        # OBSERVATION — 15-element derived state (see _state_to_obs for layout).
        # TODO: tighten bounds once you know the operating envelope.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # ── Runtime state (properly initialised in reset) ─────────────────────
        self.state: np.ndarray = self._x0.copy()    # (13,)  raw CasADi state
        self.p: np.ndarray = self._p0.copy()         # (35,)  vehicle parameters
        self.u: np.ndarray = np.zeros(4, dtype=float)  # AETR sent to integrator
        self.t: float = 0.0
        self._prev_v: float = 0.0    # last airspeed magnitude, for vdot finite-diff
        self._step_count: int = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self._x0.copy()
        self.p = self._p0.copy()
        self.u = np.zeros(4, dtype=float)
        self.t = 0.0
        self._step_count = 0

        obs = self._state_to_obs()
        self._prev_v = float(np.linalg.norm(obs[6:9]))  # world-frame speed at t=0

        if self.use_tecs:
            # Re-instantiate so the internal waypoint index and integrators reset.
            self._nav = XTrack_NAV_lookAhead(self.dt, self.waypoints, self._start_wp_index)

        return obs.astype(np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)

        # ── A) Resolve AETR from action ───────────────────────────────────────
        if self.use_tecs:
            # action = [des_v (m/s), des_gamma (rad), des_heading (rad)]
            des_v, des_gamma, des_heading = float(action[0]), float(action[1]), float(action[2])

            # des_a: desired acceleration (m/s²).
            # TODO: compute from a speed-error proportional term, or extend the
            # action space to a 4-D vector [des_v, des_gamma, des_heading, des_a].
            des_a = 0.0

            ref_data = {
                "des_v": des_v,
                "des_gamma": des_gamma,
                "des_heading": des_heading,
                "des_a": des_a,
            }
            actual_data = self._build_actual_data()

            # compute_control returns (ail_cmd, elev_cmd, throttle_cmd, rud_cmd).
            aetr = self._tecs.compute_control(self._step_count, ref_data, actual_data)
            ail_cmd, elev_cmd, throttle_cmd, rud_cmd = aetr

            # Optional: update waypoint index via the nav tracker.
            # The tracker's heading output is ignored here because the RL agent
            # supplies des_heading directly. Hook it back in if you want the
            # agent to only output speed/gamma while the tracker handles heading.
            # des_v_nav, des_gamma_nav, des_heading_nav, along_err, cross_err = \
            #     self._nav.wp_tracker(
            #         self.waypoints,
            #         actual_data["x_est"], actual_data["y_est"], actual_data["z_est"],
            #         [actual_data["vx_est"], actual_data["vy_est"], actual_data["vz_est"]],
            #     )
            # self._nav.check_arrived(along_err, [actual_data["vx_est"], ...])

        else:
            # Direct AETR — clip to hardware limits and pass straight through.
            ail_cmd      = float(np.clip(action[0], -1.0, 1.0))
            elev_cmd     = float(np.clip(action[1], -1.0, 1.0))
            throttle_cmd = float(np.clip(action[2],  0.0, 1.0))
            rud_cmd      = float(np.clip(action[3], -1.0, 1.0))

        self.u = np.array([ail_cmd, elev_cmd, throttle_cmd, rud_cmd], dtype=float)

        # ── B) Advance the CasADi integrator one step ─────────────────────────
        # This replicates integrate_simulation() from fixedwing_sim.py without
        # any ROS overhead. z0=0.0 matches the original call signature.
        res = self._integrator(x0=self.state, z0=0.0, p=self.p, u=self.u)
        self.state = np.array(res["xf"]).reshape(-1)   # (13,)
        self.t += self.dt
        self._step_count += 1

        # ── C) Build outputs ──────────────────────────────────────────────────
        obs = self._state_to_obs()
        self._prev_v = float(np.linalg.norm(obs[6:9]))   # update for next vdot

        reward     = self._compute_reward()
        terminated = self._check_terminated()
        truncated  = False
        info = {
            "t": self.t,
            "u": self.u.copy(),
            "state": self.state.copy(),
        }

        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        # TODO: implement a matplotlib or 3-D trajectory visualisation.
        pass

    def close(self):
        pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _state_to_obs(self) -> np.ndarray:
        """
        Derive the 15-element observation from the raw 13-element CasADi state.

        Layout:
            [0:3]   x, y, z         — world-frame position (m)
            [3:6]   roll, pitch, yaw — Euler ZYX angles (rad)
            [6:9]   vx_w, vy_w, vz_w — world-frame velocity (m/s)
            [9]     v               — airspeed magnitude (m/s)
            [10]    gamma           — flight-path angle, positive nose-up (rad)
            [11]    vdot            — airspeed rate finite-diff (m/s²)
            [12:15] p, q, r         — body-frame angular rates (rad/s)
        """
        pos_w   = self.state[0:3]    # world-frame position
        vel_b   = self.state[3:6]    # body-frame velocity
        quat_wb = self.state[6:10]   # [qw, qx, qy, qz], world→body
        omega_b = self.state[10:13]  # [p, q, r]

        # scipy uses scalar-last [qx, qy, qz, qw]; reorder from state's scalar-first.
        r_wb = Rotation.from_quat([quat_wb[1], quat_wb[2], quat_wb[3], quat_wb[0]])

        # Euler angles: ZYX decomposition gives (yaw, pitch, roll).
        yaw, pitch, roll = r_wb.as_euler("ZYX")

        # quat_wb rotates world→body: v_b = R_wb @ v_w → v_w = R_wb⁻¹ @ v_b
        vel_w = r_wb.inv().apply(vel_b)

        v     = float(np.linalg.norm(vel_w))
        gamma = float(np.arcsin(np.clip(vel_w[2] / max(v, 1e-3), -1.0, 1.0)))
        vdot  = (v - self._prev_v) / self.dt if self.dt > 0 else 0.0

        return np.array([
            pos_w[0], pos_w[1], pos_w[2],          # 0:3  position
            roll, pitch, yaw,                        # 3:6  Euler angles
            vel_w[0], vel_w[1], vel_w[2],           # 6:9  world-frame velocity
            v, gamma, vdot,                          # 9:12 speed / flight-path / accel
            omega_b[0], omega_b[1], omega_b[2],     # 12:15 angular rates
        ], dtype=np.float32)

    def _build_actual_data(self) -> dict:
        """
        Build the actual_data dict expected by TECSControl_cub.compute_control().
        All 15 keys must be present and match the names used in sim_tecs_ros_xtrack.py.
        """
        obs = self._state_to_obs()
        return {
            "x_est":     float(obs[0]),
            "y_est":     float(obs[1]),
            "z_est":     float(obs[2]),
            "roll_est":  float(obs[3]),
            "pitch_est": float(obs[4]),
            "yaw_est":   float(obs[5]),
            "vx_est":    float(obs[6]),
            "vy_est":    float(obs[7]),
            "vz_est":    float(obs[8]),
            "v_est":     float(obs[9]),
            "gamma_est": float(obs[10]),
            "vdot_est":  float(obs[11]),
            "p_est":     float(obs[12]),
            "q_est":     float(obs[13]),
            "r_est":     float(obs[14]),
        }

    def _compute_reward(self) -> float:
        """
        Placeholder reward — always returns 0.

        TODO: replace with a task-specific signal, for example:
            - Negative cross-track error from the active waypoint segment
            - Positive reward per waypoint reached
            - Energy-efficiency penalty (throttle magnitude)
            - Large negative reward on crash / divergence
        """
        return 0.0

    def _check_terminated(self) -> bool:
        """
        Placeholder termination — always returns False.

        TODO: add task-specific conditions, for example:
            - self.state[2] < 0.0          # ground collision (z is up-positive)
            - np.any(np.isnan(self.state)) # numerical blow-up
            - all waypoints visited
        """
        return False
