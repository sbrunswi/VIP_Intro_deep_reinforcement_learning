"""
flight_dynamics.py
------------------
Physics-based flight constraint model for PylonRacingEnv.

Models the HobbyHobby Sport Cub S2 (57 g, 617 mm span) as used by the
Cyecca fixedwing_4ch simulator (fixedwing_4ch.py p_defaults).  All values
are SI units (m, m/s, rad, rad/s, N, kg).

Constraints modelled:
  - Stall / max airspeed envelope
  - Load factor (structural g-limits)
  - Angular rate limits (servo + airframe)
  - Actuator rate limits & deflection limits
  - Energy / throttle lag
  - Ground effect near z=0
  - Wind disturbance model (turbulence + steady)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Airframe constants — HobbyHobby Sport Cub S2 (matches fixedwing_4ch.py)
# ---------------------------------------------------------------------------

# Mass & geometry
MASS_KG          = 1.5          # kg
WING_AREA_M2     = 0.24         # m²
WINGSPAN_M       = 1.2          # m
INERTIA          = np.diag([2e-4, 2.6e-4, 3.2e-4])  # Ixx, Iyy, Izz  kg·m²
CHORD_M = 0.20  # Average width of the wing (m)

# Aerodynamic coefficients (from fixedwing_4ch.py p_defaults)
CL0              = 0.20         # lift at zero AoA
CL_ALPHA         = 5.2          # 1/rad
CD0              = 0.09         # parasitic drag
K_INDUCED        = 0.062        # induced drag factor (CDCLS in fixedwing_4ch)
CM_ALPHA         = -0.8         # pitch stiffness (approximate, not in p_defaults)
CM_Q             = -12.0        # pitch damping (approximate)
CY_BETA          = -0.35        # side-force due to sideslip (approximate)
CL_P             = -0.5         # roll damping (approximate)
CN_R             = -0.12        # yaw damping (approximate)

# Control effectiveness (from fixedwing_4ch.py p_defaults, per radian)
CL_AILERON       = 0.10         # Clda — roll from aileron
CM_ELEVATOR      = 0.90         # Cmde — pitch from elevator (positive = nose up)
CN_RUDDER        = 0.12         # Cndr — yaw from rudder

# Propulsion
MAX_THRUST_N     = 0.56         # N  (thr_max from fixedwing_4ch.py)
THRUST_LAG_TC    = 0.10         # s  (small brushless motor spools quickly)

# Atmosphere
RHO              = 1.225        # kg/m³  sea-level density
G                = 9.81         # m/s²

# ---------------------------------------------------------------------------
# Flight envelope limits — Sport Cub S2
# ---------------------------------------------------------------------------

# Stall speed: sqrt(2*m*g / (rho*S*CL_max)) ≈ sqrt(2*0.057*9.81/(1.225*0.05553*1.5)) ≈ 3.3 m/s
V_STALL_MS       = 3.5          # m/s  (conservative, clean)
V_STALL_LAND_MS  = 3.0          # m/s  (low-speed approach)
V_NE_MS          = 15.0         # m/s  never-exceed for this micro-UAV
V_CRUISE_MS      = 7.0          # m/s  nominal competition cruise
V_MAX_CLIMB_MS   = 10.0         # m/s  best climb band

ALPHA_MAX_RAD    = np.deg2rad(18.0)   # stall AoA
ALPHA_MIN_RAD    = np.deg2rad(-8.0)   # negative stall AoA
BETA_MAX_RAD     = np.deg2rad(25.0)   # sideslip limit

# Structural limits
G_LIMIT_POS      = 6.0          # max positive load factor (light airframe)
G_LIMIT_NEG      = -3.0         # max negative load factor

# Angular rate limits  rad/s
P_MAX            = np.deg2rad(360)    # roll rate  (agile micro-UAV)
Q_MAX            = np.deg2rad(180)    # pitch rate
R_MAX            = np.deg2rad(120)    # yaw rate

# Actuator deflection limits (from fixedwing_4ch.py p_defaults)
AILERON_MAX_RAD  = np.deg2rad(30.0)   # max_defl_ail
ELEVATOR_MAX_RAD = np.deg2rad(24.0)   # max_defl_elev
RUDDER_MAX_RAD   = np.deg2rad(20.0)   # max_defl_rud

# Actuator rate limits  rad/s  (small digital servos ~300°/s)
AILERON_RATE_MAX = np.deg2rad(300)
ELEVATOR_RATE_MAX= np.deg2rad(300)
RUDDER_RATE_MAX  = np.deg2rad(300)

# ---------------------------------------------------------------------------
# Helper: dynamic pressure
# ---------------------------------------------------------------------------

def dynamic_pressure(v: float) -> float:
    """q = 0.5 * rho * v²"""
    return 0.5 * RHO * v * v


def q_bar(v: float) -> float:
    return dynamic_pressure(v)


# ---------------------------------------------------------------------------
# FlightDynamics  –  stateless constraint / physics helper
# ---------------------------------------------------------------------------

class FlightDynamics:
    """
    Provides:
      - enforce_actuator_limits()   – clips & rate-limits actuator commands
      - compute_aero_forces()       – lift, drag, side-force in body frame
      - compute_moments()           – roll, pitch, yaw moments
      - check_stall()               – bool + stall fraction
      - check_structural()          – bool, load factor
      - is_envelope_ok()            – combined envelope check
      - apply_ground_effect()       – modifies lift near ground
      - get_wind_disturbance()      – Dryden turbulence + steady wind
    """

    def __init__(self, wind_speed_ms: float = 2.0, wind_dir_rad: float = 0.0,
                 turbulence_intensity: float = 0.15, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.wind_steady = np.array([
            wind_speed_ms * np.cos(wind_dir_rad),
            wind_speed_ms * np.sin(wind_dir_rad),
            0.0
        ])
        self.turb_intensity = turbulence_intensity

        # Dryden filter state (1st order shaping filter per axis)
        self._turb_state = np.zeros(3)
        self._turb_tau   = np.array([1.5, 1.5, 0.5])   # time constants  s

    # ------------------------------------------------------------------
    # Actuator management
    # ------------------------------------------------------------------

    def enforce_actuator_limits(
        self,
        cmd: np.ndarray,          # [aileron, elevator, throttle, rudder]  -1..1
        prev_cmd: np.ndarray,     # previous step command
        dt: float
    ) -> np.ndarray:
        """
        Returns clipped command respecting:
          • deflection limits
          • rate limits (servo speed)
          • throttle 0..1 only
        """
        cmd = cmd.copy().astype(np.float32)
        prev_cmd = prev_cmd.copy().astype(np.float32)

        # --- aileron [0] ---
        max_d_ail = AILERON_RATE_MAX * dt / AILERON_MAX_RAD   # in norm units
        cmd[0] = prev_cmd[0] + np.clip(cmd[0] - prev_cmd[0], -max_d_ail, max_d_ail)
        cmd[0] = np.clip(cmd[0], -1.0, 1.0)

        # --- elevator [1] ---
        max_d_elev = ELEVATOR_RATE_MAX * dt / ELEVATOR_MAX_RAD
        cmd[1] = prev_cmd[1] + np.clip(cmd[1] - prev_cmd[1], -max_d_elev, max_d_elev)
        cmd[1] = np.clip(cmd[1], -1.0, 1.0)

        # --- throttle [2]: 0..1, rate limited (spool) ---
        max_d_thr = dt / THRUST_LAG_TC          # ~6.7 per second full range
        cmd[2] = prev_cmd[2] + np.clip(cmd[2] - prev_cmd[2], -max_d_thr, max_d_thr)
        cmd[2] = np.clip(cmd[2], 0.0, 1.0)

        # --- rudder [3] ---
        max_d_rud = RUDDER_RATE_MAX * dt / RUDDER_MAX_RAD
        cmd[3] = prev_cmd[3] + np.clip(cmd[3] - prev_cmd[3], -max_d_rud, max_d_rud)
        cmd[3] = np.clip(cmd[3], -1.0, 1.0)

        return cmd

    # ------------------------------------------------------------------
    # Aerodynamics
    # ------------------------------------------------------------------

    def compute_aero_forces(
        self,
        v: float,           # airspeed m/s
        alpha: float,       # AoA rad
        beta: float,        # sideslip rad
        elevator: float,    # norm -1..1
        throttle: float
    ) -> dict:
        """
        Returns dict with keys: lift, drag, side_force, thrust (all Newtons)
        and cl, cd, alpha_eff (for diagnostics).
        """
        alpha = np.clip(alpha, ALPHA_MIN_RAD, ALPHA_MAX_RAD)
        qbar  = q_bar(v)
        S     = WING_AREA_M2

        # Lift
        cl    = CL0 + CL_ALPHA * alpha + CM_ELEVATOR * elevator * 0.1
        cl    = np.clip(cl, -1.8, 1.8)
        lift  = qbar * S * cl

        # Drag
        cd    = CD0 + K_INDUCED * cl**2
        drag  = qbar * S * cd

        # Side force (from sideslip)
        side_force = qbar * S * CY_BETA * beta

        # Thrust
        thrust = MAX_THRUST_N * throttle

        return dict(lift=lift, drag=drag, side_force=side_force,
                    thrust=thrust, cl=cl, cd=cd, alpha_eff=alpha)

    def compute_moments(
    self,
    v: float,
    alpha: float,      # <--- Added alpha here
    q_rate: float,      # pitch rate rad/s
    p_rate: float,      # roll rate  rad/s
    r_rate: float,      # yaw rate   rad/s
    aileron: float,     # norm -1..1
    elevator: float,    # norm -1..1
    rudder: float       # norm -1..1
) -> np.ndarray:
    """Returns [roll_moment, pitch_moment, yaw_moment] in N·m."""
    qbar = q_bar(v)
    S    = WING_AREA_M2
    b    = WINGSPAN_M
    c    = CHORD_M  # Using chord for pitch stability
    
    # Safe velocity for division to avoid NaN/Inf at low speeds
    v_safe = max(v, 1.0)

    # 1. Roll Moment (L) - Influenced by roll rate and ailerons
    roll_moment = qbar * S * b * (
        CL_P * (p_rate * b / (2 * v_safe)) + 
        CL_AILERON * aileron * AILERON_MAX_RAD
    )

    # 2. Pitch Moment (M) - The "Self-Righting" Logic
    # We now multiply CM_ALPHA by the actual alpha (AoA)
    pitch_moment = qbar * S * c * (
        CM_ALPHA * alpha + 
        CM_Q * (q_rate * c / (2 * v_safe)) + 
        CM_ELEVATOR * elevator * ELEVATOR_MAX_RAD
    )

    # 3. Yaw Moment (N) - Influenced by yaw rate and rudder
    yaw_moment = qbar * S * b * (
        CN_R * (r_rate * b / (2 * v_safe)) + 
        CN_RUDDER * rudder * RUDDER_MAX_RAD
    )

    return np.array([roll_moment, pitch_moment, yaw_moment])

    # ------------------------------------------------------------------
    # Envelope checks
    # ------------------------------------------------------------------

    def check_stall(self, v: float, alpha: float) -> tuple[bool, float]:
        """
        Returns (is_stalled: bool, stall_fraction: float 0..1+).
        stall_fraction > 1.0 means deep stall.
        """
        # Speed-based: stall speed increases with load factor but approximate here
        speed_margin  = (v - V_STALL_MS) / max(V_STALL_MS, 1e-3)
        alpha_fraction = abs(alpha) / ALPHA_MAX_RAD
        stall = (v < V_STALL_MS) or (abs(alpha) > ALPHA_MAX_RAD)
        frac  = max(1.0 - speed_margin, alpha_fraction)
        return stall, frac

    def check_structural(self, lift: float) -> tuple[bool, float]:
        """Returns (over_limit: bool, load_factor: float)."""
        nz = lift / max(MASS_KG * G, 1e-3)
        over = nz > G_LIMIT_POS or nz < G_LIMIT_NEG
        return over, nz

    def check_speed_envelope(self, v: float) -> tuple[bool, str]:
        """Returns (inside: bool, status_str)."""
        if v < V_STALL_MS:
            return False, "STALL"
        if v > V_NE_MS:
            return False, "OVERSPEED"
        return True, "OK"

    def is_envelope_ok(self, v: float, alpha: float, lift: float) -> tuple[bool, dict]:
        stalled, stall_frac = self.check_stall(v, alpha)
        over_g,  load_fac   = self.check_structural(lift)
        spd_ok,  spd_str    = self.check_speed_envelope(v)
        ok = (not stalled) and (not over_g) and spd_ok
        info = dict(stalled=stalled, stall_fraction=stall_frac,
                    load_factor=load_fac, speed_status=spd_str)
        return ok, info

    # ------------------------------------------------------------------
    # Ground effect  (increases effective lift within ~1 wingspan)
    # ------------------------------------------------------------------

    def apply_ground_effect(self, lift: float, z_agl: float) -> float:
        """
        Increases lift when z_agl < wingspan using a simple IGE factor.
        Factor approaches 1.0 at altitude and ~1.25 at z→0.
        """
        ratio = max(z_agl, 0.01) / WINGSPAN_M
        ige   = 1.0 + 0.25 * np.exp(-2.0 * ratio)   # asymptotes to 1.0
        return lift * ige

    # ------------------------------------------------------------------
    # Wind / turbulence  (Dryden simplified, per-step call)
    # ------------------------------------------------------------------

    def get_wind_disturbance(self, v: float, dt: float) -> np.ndarray:
        """
        Returns 3-D wind vector in NED [vx, vy, vz] m/s.
        Combines steady wind + Dryden-like turbulence.
        """
        sigma = self.turb_intensity * max(v, V_STALL_MS)
        white = self.rng.normal(0.0, sigma, 3)
        alpha_filt = dt / (self._turb_tau + dt)
        self._turb_state += alpha_filt * (white - self._turb_state)
        return self.wind_steady + self._turb_state


# ---------------------------------------------------------------------------
# EnvelopeReward  –  shaped reward terms related to flight envelope
# ---------------------------------------------------------------------------

class EnvelopeReward:
    """
    Reward / penalty shaping utilities so that the RL agent is trained
    to respect real flight constraints.
    """

    @staticmethod
    def stall_penalty(stall_fraction: float) -> float:
        """Quadratic penalty that grows rapidly beyond stall."""
        excess = max(stall_fraction - 0.85, 0.0)   # start penalising near stall
        return -8.0 * excess ** 2

    @staticmethod
    def overspeed_penalty(v: float) -> float:
        excess = max(v - V_NE_MS, 0.0)
        return -5.0 * excess ** 2

    @staticmethod
    def load_factor_penalty(nz: float) -> float:
        excess_pos = max(nz - G_LIMIT_POS, 0.0)
        excess_neg = max(G_LIMIT_NEG - nz, 0.0)
        return -3.0 * (excess_pos ** 2 + excess_neg ** 2)

    @staticmethod
    def altitude_floor_penalty(z: float, floor: float = 1.5) -> float:
        """Penalise flight below safe floor (ground proximity)."""
        deficit = max(floor - z, 0.0)
        return -10.0 * deficit ** 2

    @staticmethod
    def angular_rate_penalty(p: float, q: float, r: float) -> float:
        def _pen(rate, limit): return max(abs(rate) - limit, 0.0) ** 2
        return -2.0 * (_pen(p, P_MAX) + _pen(q, Q_MAX) + _pen(r, R_MAX))

    @classmethod
    def compute_all(cls, stall_fraction, nz, v, z, p, q, r) -> float:
        """Single call that returns total envelope penalty for one step."""
        return (
            cls.stall_penalty(stall_fraction)
            + cls.overspeed_penalty(v)
            + cls.load_factor_penalty(nz)
            + cls.altitude_floor_penalty(z)
            + cls.angular_rate_penalty(p, q, r)
        )

def update_state(self, state, action, dt=0.02):
    """
    state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    action: [ail, elev, thr, rud] normalized -1..1
    """
    # 1. Unpack State
    pos = state[0:3]    # NED Position (m)
    vel_bus = state[3:6] # Velocity in Body Frame (u, v, w)
    att = state[6:9]    # Euler Angles (rad)
    omega = state[9:12] # Angular Rates (p, q, r)

    # 2. Get Forces and Moments from our helper
    # (Using the compute_moments fix we just did)
    forces = self.dynamics.compute_aero_forces(
        v=np.linalg.norm(vel_bus), 
        alpha=state[7], # theta as proxy for alpha in simple cases
        beta=0.0, 
        elevator=action[1], 
        throttle=action[2]
    )
    
    moments = self.dynamics.compute_moments(
        v=np.linalg.norm(vel_bus),
        alpha=state[7],
        q_rate=omega[1], p_rate=omega[0], r_rate=omega[2],
        aileron=action[0], elevator=action[1], rudder=action[3]
    )

    # 3. Linear Acceleration (F = ma -> a = F/m)
    # Add gravity: remember to rotate gravity into the body frame!
    g_body = np.array([
        -G * np.sin(att[1]),
        G * np.sin(att[0]) * np.cos(att[1]),
        G * np.cos(att[0]) * np.cos(att[1])
    ])
    
    accel_body = (np.array([forces['thrust'] - forces['drag'], 
                           forces['side_force'], 
                           -forces['lift']]) / MASS_KG) + g_body

    # 4. Angular Acceleration (Euler's Equations)
    # alpha = I^-1 * (Moments - omega x (I * omega))
    mag_moment = moments - np.cross(omega, INERTIA @ omega)
    alpha_body = np.linalg.inv(INERTIA) @ mag_moment

    # 5. Integration (The "Semi-Implicit" part)
    # Update velocities FIRST
    new_vel_bus = vel_bus + accel_body * dt
    new_omega = omega + alpha_body * dt

    # 6. Transform Body Velocity to World (NED) Position
    # You need a Rotation Matrix (R_body_to_ned) here!
    # pos_ned = pos_ned + (R @ new_vel_bus) * dt
    
    return np.concatenate([pos, new_vel_bus, att, new_omega])