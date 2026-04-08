"""
PylonRacingEnv — Scratch Rebuilt Passthrough Environment
--------------------------------------------------------
A headless Gymnasium environment for fixed-wing UAV pylon racing, built to
execute entirely outside of ROS to enable high-speed RL training loops.

This is a "Passthrough" verification version: The agent action is 
completely ignored. The plane flies fully autonomously using:
    NAV -> TECS -> Physics 

CRITICAL FIXES IMPLEMENTED:
1. State/Parameter Mapping: Dictionary orders (like `list(dict.values())`) 
   are not preserved to match CasADi's required vector positions. We now 
   strictly use `x_index`, `p_index`, and `u_index` from `derive_model()`.
2. Stiff Dynamics Substepping: Ground forces are modeled heavily as stiff
   springs. RK4 at 100Hz explodes. We run an inner RK4 loop at 1000Hz (10x).
3. Quaternion Normalization: The default state provides `[0, 0.09, 0, 1]` 
   which is un-normalized. We normalize at reset and across all RK4 substeps 
   to prevent integration divergence.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Path Surgery: Inject ROS 2 package sources into PYTHONPATH 
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]   
# Depth: pylon_racing_env.py → waypoints/ → no_ros2/ → VIP_Intro_deep_reinforcement_learning/
_SRC = _REPO_ROOT / "ros2_ws" / "src"

for _p in [_SRC / "cyecca", _SRC / "auav_pylon_2026"]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# External Modules
from cyecca.models.fixedwing_4ch import derive_model              # type: ignore
from auav_pylon_2026.cross_tracker_nav_sample import XTrack_NAV_lookAhead  # type: ignore
from auav_pylon_2026.tecs_controller_xtrack_sample import TECSControl_cub  # type: ignore

# ---------------------------------------------------------------------------
# Course Definition (Rectangle Circuit Full Facility)
# ---------------------------------------------------------------------------
_ALT = 7.0
DEFAULT_WAYPOINTS: List[Tuple[float, float, float]] = [
    (-10.0,  -5.0, _ALT),
    (-30.0, -10.0, _ALT),
    (-30.0, -40.0, _ALT),
    ( 30.0, -30.0, _ALT),
    ( 30.0,   5.0, _ALT),
    ( 10.0,   5.0, _ALT),
    (-10.0,  -5.0, _ALT),   # closes the loop back to WP0
]

# 2D Gates (from sample_course_1.yaml)
PYLONS_XY: List[Tuple[float, float]] = [
    (-20.0,   5.0),   # P1
    (-20.0, -20.0),   # P2
    (-25.0, -45.0),   # P3
    ( 20.0, -25.0),   # P4
    ( 35.0,   0.0),   # P5
    ( 10.0,  -5.0),   # P6
]
GATES: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
BOUNDS = dict(min_x=-60.0, max_x=50.0, min_y=-60.0, max_y=30.0)

# Takeoff Constants
_V_TO = 0.5; _E_DOWN = -0.02; _E_UP = 0.15; _E_RATE = 0.40
_PHASE_TAKEOFF = 0.0
_PHASE_AIRBORNE = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap_pi(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)

def _lpf(prev: float, raw: float, alpha: float) -> float:
    """IIR low-pass filter: alpha ≈ exp(−2π·fc·dt)."""
    return alpha * raw + (1.0 - alpha) * prev

def _quat_to_euler_zyx(q: np.ndarray) -> Tuple[float, float, float]:
    """Quaternion [qx, qy, qz, qw] -> (roll, pitch, yaw)"""
    qx, qy, qz, qw = q
    roll  = np.arctan2(2*(qw*qx + qy*qz),  1 - 2*(qx*qx + qy*qy))
    sinp  = 2*(qw*qy - qz*qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw   = np.arctan2(2*(qw*qz + qx*qy),  1 - 2*(qy*qy + qz*qz))
    return float(roll), float(pitch), float(yaw)

def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2*(qy*vz - qz*vy)
    ty = 2*(qz*vx - qx*vz)
    tz = 2*(qx*vy - qy*vx)
    return np.array([
        vx + qw*tx + qy*tz - qz*ty,
        vy + qw*ty + qz*tx - qx*tz,
        vz + qw*tz + qx*ty - qy*tx,
    ])

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class PylonRacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        waypoints: Optional[List[Tuple[float, float, float]]] = None,
        dt: float = 0.01,
        max_episode_steps: int = 10_000,
        tecs_gain_file: str = "sim",
    ) -> None:
        super().__init__()
        self.waypoints = waypoints if waypoints is not None else DEFAULT_WAYPOINTS
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.tecs_gain_file = tecs_gain_file

        # Dummy Action Space (1D continuous, ignored entirely)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 12-D Egocentric Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # ── 1. Load Physics & Maps ───────────────────────────────────────────
        model = derive_model()
        self._f = model["f"]
        
        # Strict Index Mappings from CasADi
        self._p_index = model.get("p_index", {})
        self._x_index = model.get("x_index", {})
        self._u_index = model.get("u_index", {})
        self._x0_defaults = model.get("x0_defaults", {})

        # Construct parameter vector matching pure CasADi layout
        p_def = model["p_defaults"]
        self._p_vec = np.zeros(len(p_def), dtype=float)
        for name, idx in self._p_index.items():
            if name in p_def:
                self._p_vec[idx] = float(p_def[name])
                
        # Integrator Loop
        self._rk4 = self._build_rk4()

        # Trackers
        self._nav: Optional[XTrack_NAV_lookAhead] = None
        self._tecs: Optional[TECSControl_cub] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        # ── 1. Init Strict State Vector ──────────────────────────────────────
        self._x_state = np.zeros(len(self._x_index), dtype=float)
        for name, idx in self._x_index.items():
            if name in self._x0_defaults:
                self._x_state[idx] = float(self._x0_defaults[name])
                
        # Normalize the initial quaternion
        q_idx0 = self._x_index.get("quat_wb_0", 6)
        q = self._x_state[q_idx0 : q_idx0+4]
        self._x_state[q_idx0 : q_idx0+4] = q / np.linalg.norm(q)
        
        # Spawn at -0.5m (Underground catapult). 
        # This was proven stable in previous runs by providing a predictable upward spring force.
        self._x_state[self._x_index["position_w_2"]] = -0.5

        # ── 2. Init Sub-systems ──────────────────────────────────────────────
        self._current_wp_idx = 0
        self._nav = XTrack_NAV_lookAhead(dt=self.dt, waypoints=self.waypoints, start_WP_ind=0)
        self._nav.v_cruise = 10.0
        
        self._tecs = TECSControl_cub(self.dt, self.tecs_gain_file)

        # Bookkeeping
        self._step_count = 0
        self._sim_time = 0.0
        self._flight_phase = _PHASE_TAKEOFF
        self._takeoff_throttle = 0.7
        self._elev_ramp = _E_DOWN
        self._prev_speed = 0.0
        self._cross_err = 0.0
        self._along_err = 0.0
        self._actual_data = self._zero_actual_data()
        self._ref_data = self._zero_ref_data()
        self._lpf_state = {}

        # --- DIAGNOSTIC PROBE ---
        print("\n--- RESET DIAGNOSTICS ---")
        print(f"p_vec zeros count: {np.sum(self._p_vec == 0.0)} out of {len(self._p_vec)}")
        if np.sum(self._p_vec == 0.0) > 0:
            # Find which parameters failed to map
            zero_p = [name for name, idx in self._p_index.items() if self._p_vec[idx] == 0.0]
            print(f"WARNING: These parameters initialized to 0.0: {zero_p}")
            
        q_idx0 = self._x_index.get("quat_wb_0", 6)
        q_raw = self._x_state[q_idx0 : q_idx0+4]
        print(f"Initial Quaternion: {q_raw}")
        if np.linalg.norm(q_raw) == 0 or np.isnan(np.linalg.norm(q_raw)):
            print("WARNING: QUATERNION IS ZERO OR NAN! Normalization will fail.")
        print("-------------------------\n")

        return self._build_observation(), self._build_info()

    def step(self, action: np.ndarray):
        # ── 1. NAV & TECS Autopilot ──────────────────────────────────────────
        actuators = self._compute_autopilot()  # ail, elev, thr, rud
        
        # Build u_vec strictly matching u_index
        u_vec = np.zeros(len(self._u_index), dtype=float)
        u_vec[self._u_index["ail_cmd"]] = actuators[0]
        u_vec[self._u_index["elev_cmd"]] = actuators[1]
        u_vec[self._u_index["throttle_cmd"]] = actuators[2]
        u_vec[self._u_index["rud_cmd"]] = actuators[3]

        # ── 2. Physics ───────────────────────────────────────────────────────
        self._x_state = self._rk4(self._x_state, u_vec, self._p_vec)

        # ── 3. Data Extraction ───────────────────────────────────────────────
        self._update_actual_data()

        # Air/Ground logic: Transition to airborne when altitude clears 1.0m (Parity with ROS)
        if self._flight_phase == _PHASE_TAKEOFF:
            if self._actual_data["z_est"] > 1.0:
                self._flight_phase = _PHASE_AIRBORNE

        # ── 4. Navigation Loop ───────────────────────────────────────────────
        if self._flight_phase == _PHASE_AIRBORNE:
            self._update_navigation()

        # ── 5. Clock & Reward ────────────────────────────────────────────────
        self._sim_time += self.dt
        self._step_count += 1
        
        reward = -1.0 * self.dt
        terminated = False
        
        # Crash termination
        if self._flight_phase == _PHASE_AIRBORNE and self._actual_data["z_est"] < 0:
            terminated = True
            
        truncated = self._step_count >= self.max_episode_steps

        return self._build_observation(), reward, terminated, truncated, self._build_info()

    # =========================================================================
    # Internal Simulation Flow
    # =========================================================================

    def _build_rk4(self):
        """Standard explicit RK4 padded with 100 inner integration substeps."""
        f = self._f
        dt_sub = self.dt / 100.0
        q_idx0 = self._x_index.get("quat_wb_0", 6)

        def rk4(x, u, p):
            for _ in range(100):
                k1 = np.array(f(x, u, p)).flatten()
                k2 = np.array(f(x + 0.5*dt_sub*k1, u, p)).flatten()
                k3 = np.array(f(x + 0.5*dt_sub*k2, u, p)).flatten()
                k4 = np.array(f(x + dt_sub*k3, u, p)).flatten()
                
                x = x + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
                # Immediate constraint projection (normalization)
                q = x[q_idx0 : q_idx0+4]
                x[q_idx0 : q_idx0+4] = q / np.linalg.norm(q)
            return x
        return rk4

    def _compute_autopilot(self) -> np.ndarray:
        """Determines logic block to ping for surface controls."""
        if self._flight_phase == _PHASE_TAKEOFF:
            self._takeoff_throttle = float(np.clip(self._takeoff_throttle + 2.0*self.dt, 0.7, 1.0))
            if self._actual_data.get("v_est", 0) < _V_TO:
                self._elev_ramp = _E_DOWN
            else:
                self._elev_ramp = float(np.clip(self._elev_ramp + _E_RATE*self.dt, _E_DOWN, _E_UP))
            return np.array([0.0, self._elev_ramp, self._takeoff_throttle, 0.0])
        else:
            t_step = int(self._sim_time / self.dt)
            a, e, t, r = self._tecs.compute_control(t=t_step, 
                                                    ref_data=self._ref_data, 
                                                    actual_data=self._actual_data)
            # Invert TECS elevator to match CasADi (Positive = Nose UP)
            e = -e 
            return np.array([a, e, t, r])

    def _update_navigation(self):
        ad = self._actual_data
        v_arr = [ad["vx_est"], ad["vy_est"], ad["vz_est"]]
        
        # Ping NAV lookAhead
        des_v, des_g, des_h, a_err, c_err = self._nav.wp_tracker(
            waypoints=self.waypoints,
            x_est=ad["x_est"], y_est=ad["y_est"], z_est=ad["z_est"],
            V_array=v_arr, verbose=False
        )
        
        self._ref_data = {
            "des_v": des_v, 
            "des_gamma": des_g, 
            "des_heading": des_h, 
            "des_a": 1.0 * (des_v - abs(ad["v_est"])) 
        }
        self._along_err = float(a_err)
        self._cross_err = float(c_err)

        self._current_wp_idx = self._nav.check_arrived(along_track_err=a_err, V_array=v_arr)
        if self._current_wp_idx >= len(self.waypoints):
            self._current_wp_idx = 0
            self._nav = XTrack_NAV_lookAhead(dt=self.dt, waypoints=self.waypoints, start_WP_ind=0)
            self._nav.v_cruise = 10.0

    def _update_actual_data(self):
        """Map CasADi states back to named variables for TECS."""
        x = self._x_state
        px = x[self._x_index["position_w_0"]]
        py = x[self._x_index["position_w_1"]]
        pz = x[self._x_index["position_w_2"]]
        
        q_idx = self._x_index["quat_wb_0"]
        q = x[q_idx : q_idx+4]
        roll, pitch, yaw = _quat_to_euler_zyx(q)
        
        v_idx = self._x_index["velocity_b_0"]
        vel_b = x[v_idx : v_idx+3]
        vel_w = _quat_rotate(q, vel_b)
        vx, vy, vz = vel_w
        speed = max(float(np.linalg.norm(vel_w)), 1e-5)
        
        gamma = float(np.arcsin(np.clip(vz / speed, -1.0, 1.0)))
        vdot = speed - self._prev_speed
        self._prev_speed = speed
        
        o_idx = self._x_index["omega_wb_b_0"]
        p_, q_, r_ = x[o_idx], x[o_idx+1], x[o_idx+2]
            
        alpha = float(np.exp(-2 * np.pi * 10.0 * self.dt))
        sigs = {"x": px, "y": py, "z": pz, "roll": roll, "pitch": pitch, "yaw": yaw,
                "vx": vx, "vy": vy, "vz": vz, "v": speed, "gamma": gamma, "vdot": vdot,
                "p": p_, "q": q_, "r": r_}
        
        for name, raw in sigs.items():
            key = f"{name}_est"
            prev = self._lpf_state.get(key, raw)
            self._lpf_state[key] = _lpf(prev, raw, alpha)
            self._actual_data[key] = self._lpf_state[key]

    def _build_observation(self) -> np.ndarray:
        ad = self._actual_data
        x, y, z, yaw = ad["x_est"], ad["y_est"], ad["z_est"], ad["yaw_est"]
        N = len(self.waypoints)
        
        # WP0
        wp0 = self.waypoints[self._current_wp_idx % N]
        dx0, dy0 = wp0[0] - x, wp0[1] - y
        w0_dist = np.linalg.norm([dx0, dy0, wp0[2]-z])
        w0_bear = _wrap_pi(np.arctan2(dy0, dx0) - yaw)
        w0_elev = wp0[2] - z

        # WP1
        wp1 = self.waypoints[(self._current_wp_idx + 1) % N]
        dx1, dy1 = wp1[0] - x, wp1[1] - y
        w1_dist = np.linalg.norm([dx1, dy1, wp1[2]-z])
        w1_bear = _wrap_pi(np.arctan2(dy1, dx1) - yaw)
        w1_elev = wp1[2] - z
        
        # Gate Pylons
        gate_idx = self._current_wp_idx % len(GATES)
        p1 = PYLONS_XY[GATES[gate_idx][0]]
        p2 = PYLONS_XY[GATES[gate_idx][1]]
        dp1 = np.linalg.norm([x - p1[0], y - p1[1]])
        dp2 = np.linalg.norm([x - p2[0], y - p2[1]])

        return np.array([
            ad["v_est"], ad["roll_est"], ad["pitch_est"],
            w0_dist, w0_bear, w0_elev,
            w1_dist, w1_bear, w1_elev,
            self._cross_err, dp1, dp2
        ], dtype=np.float32)

    def _build_info(self):
        return {
            "sim_time": self._sim_time,
            "step": self._step_count,
            "flight_phase": "takeoff" if self._flight_phase == _PHASE_TAKEOFF else "airborne",
            "wp_idx": self._current_wp_idx,
            "actual_data": dict(self._actual_data)
        }

    def _zero_actual_data(self):
        return {k: 0.0 for k in ["x_est", "y_est", "z_est", "roll_est", "pitch_est", "yaw_est",
                                 "vx_est", "vy_est", "vz_est", "v_est", "gamma_est", "vdot_est",
                                 "p_est", "q_est", "r_est"]}

    def _zero_ref_data(self):
        return {"des_v": 0.0, "des_gamma": 0.0, "des_heading": 0.0, "des_a": 0.0}

