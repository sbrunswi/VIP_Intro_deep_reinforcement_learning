"""
Mock pylon racing environment — 6-DOF physics matched to fixedwing_4ch.py (cyecca).

Implements the same aerodynamic model, parameters, and equations of motion as
ros2_ws/src/cyecca/cyecca/models/fixedwing_4ch.py, but in numpy (no CasADi).

Frame conventions (same as fixedwing_4ch.py):
  World: ENU-like, z = altitude (up), x/y are horizontal.
  Body:  x = forward (nose), y = right (starboard), z = up (top of aircraft).
  alpha = atan2(-w_body, u_body)   [positive when nose is above velocity vector]
  beta  = asin(v_body / V)         [positive = nose-left sideslip]

State (12D internal):
  [x, y, z, u, v, w, phi, theta, psi, p, q, r]
  where x/y/z = world position, u/v/w = body velocity, phi/theta/psi = Euler angles,
  p/q/r = body angular rates.

Observation (15D, same layout as ROS2 PylonRacingEnv):
  [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from no_ros2.environments.pylon_course import get_course, DEFAULT_COURSE

OBS_SIZE = 15

# --- Aerodynamic / physical parameters (matches fixedwing_4ch.py p_defaults) ---
_PARAMS = {
    "thr_max": 0.56,    # Maximum thrust (N)
    "m":       0.057,   # Mass (kg)
    "XCG":     0.25,    # CG position (non-dimensional chord)
    "XAC":     0.25,    # Aerodynamic centre (non-dimensional chord)
    "S":       0.05553, # Wing area (m^2)
    "rho":     1.225,   # Air density (kg/m^3)
    "g":       9.81,    # Gravitational acceleration (m/s^2)
    "Jx":      2.0e-4,  # Roll moment of inertia (kg·m^2)
    "Jy":      2.6e-4,  # Pitch moment of inertia (kg·m^2)
    "Jz":      3.2e-4,  # Yaw moment of inertia (kg·m^2)
    "Jxz":     0.0e-4,  # Product of inertia (kg·m^2)
    "cbar":    0.09,    # Mean aerodynamic chord (m)
    "span":    0.617,   # Wingspan (m)
    # Control effectiveness
    "Cm0":     0.0314,  # Zero-lift pitching moment coefficient
    "Clda":    0.10,    # Aileron roll effectiveness (per rad)
    "Cldr":    0.05,    # Rudder roll effectiveness (per rad)
    "Cmde":    0.9,     # Elevator pitch effectiveness (per rad)
    "Cndr":    0.12,    # Rudder yaw effectiveness (per rad)
    "Cnda":    0.03,    # Aileron yaw effectiveness (per rad)
    "CYda":    0.02,    # Sideforce due to aileron (per rad)
    "CYdr":   -0.12,    # Sideforce due to rudder (per rad)
    # Longitudinal stability
    "CL0":     0.20,    # Lift coefficient at zero AoA
    "CLa":     5.2,     # Lift slope (per rad)
    "Cma":    -0.60,    # Pitching moment due to AoA (per rad)
    "Cmq":   -18.0,     # Pitch damping (per rad/s)
    "CD0":     0.09,    # Parasitic drag coefficient
    "CDCLS":   0.062,   # Lift-induced drag coefficient
    # Lateral-directional stability
    "Cnb":     0.10,    # Yaw stiffness (per rad)
    "Clp":    -1.30,    # Roll damping (per rad/s)
    "Cnr":    -0.12,    # Yaw damping (per rad/s)
    "Cnp":    -0.10,    # Yaw due to roll rate (per rad/s)
    "Clr":     0.10,    # Roll due to yaw rate (per rad/s)
    "CYb":    -0.65,    # Sideforce due to sideslip (per rad)
    "CYr":     0.25,    # Sideforce due to yaw rate (per rad/s)
    "CYp":     0.15,    # Sideforce due to roll rate (per rad/s)
}

_DEG2RAD = np.pi / 180.0
_MAX_AIL  = 30.0 * _DEG2RAD   # maximum aileron deflection (rad)
_MAX_ELEV = 24.0 * _DEG2RAD   # maximum elevator deflection (rad)
_MAX_RUD  = 20.0 * _DEG2RAD   # maximum rudder deflection (rad)
_ALPHA_STALL = 20.0 * _DEG2RAD
_TOL_V = 1e-3


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rot_bw(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler → 3×3 rotation matrix R (body → world, ENU, body z-up)."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [ cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [ sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,     ct*sp,             ct*cp            ],
    ])


def _euler_rates(phi: float, theta: float,
                 p: float, q: float, r: float) -> np.ndarray:
    """Body rates [p,q,r] → Euler angle rates [phi_dot, theta_dot, psi_dot]."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct = np.cos(theta)
    if abs(ct) < 1e-6:
        ct = np.copysign(1e-6, ct)
    tt = np.sin(theta) / ct
    return np.array([
        p + sp * tt * q + cp * tt * r,
        cp * q - sp * r,
        sp / ct * q + cp / ct * r,
    ])


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _aero(velocity_b: np.ndarray, omega_b: np.ndarray,
          action: np.ndarray, pr: dict):
    """Aerodynamic forces and moments in body frame."""
    u, v, w = velocity_b
    P, Q, R = omega_b

    # Control surface deflections — signs match the full ROS2 pipeline:
    #   Aileron (action[0]): no sign flip in pylon_env → ail_rad = +MAX * action[0]
    #   Elevator (action[1]): pylon_env.py negates it (axes[1] = -action[1]) before
    #     Gazebo, then fixedwing_sim negates once more, so net: elev_cmd = action[1].
    #     But in cyecca elev_rad = MAX * u[1] where u[1] came from the first negation,
    #     so: elev_rad = -MAX * action[1]   (positive agent cmd = nose-up = negative elev)
    #   Rudder (action[3]): cyecca applies -1 multiplier → rud_rad = -MAX * action[3]
    ail_rad  =  _MAX_AIL  * action[0]
    elev_rad = -_MAX_ELEV * action[1]   # negated: positive action = nose-up in obs
    rud_rad  = -_MAX_RUD  * action[3]   # sign flip from fixedwing_4ch.py line 246

    V = float(np.sqrt(u*u + v*v + w*w))
    V = max(V, _TOL_V)

    # Aerodynamic angles
    # ENU body (z-up): for pitched-up nose the freestream has downward body component
    # → w < 0 → alpha = atan2(-w, u) > 0  (matches fixedwing_4ch.py line 218)
    alpha = np.arctan2(-w, u)
    beta  = np.arcsin(np.clip(v / V, -1.0, 1.0))

    # Force coefficients
    CL = pr["CL0"] + pr["CLa"] * alpha
    if abs(alpha) >= _ALPHA_STALL:
        CL = 0.0
    CD = pr["CD0"] + pr["CDCLS"] * CL * CL

    b, c, S = pr["span"], pr["cbar"], pr["S"]
    CC = (
        -pr["CYb"] * beta
        + pr["CYda"] * ail_rad / _MAX_AIL
        + pr["CYdr"] * rud_rad / _MAX_RUD
        + pr["CYp"] * b / (2.0 * V) * P
        + pr["CYr"] * b / (2.0 * V) * R
    )

    # Moment coefficients
    Cl_c = pr["Clda"] * ail_rad + (-1.0) * pr["Cldr"] * rud_rad
    Cm_c = (
        pr["Cm0"] + pr["Cma"] * alpha + pr["Cmde"] * elev_rad
        + (pr["XAC"] - pr["XCG"]) * CL
    )
    Cn_c = pr["Cnb"] * beta + pr["Cndr"] * rud_rad + (-1.0) * pr["Cnda"] * ail_rad

    qbar = 0.5 * pr["rho"] * V * V
    D_mag = CD * qbar * S
    L_mag = CL * qbar * S
    Fs    = CC * qbar * S

    # --- Forces in body frame ---
    # Drag: directly opposing velocity direction
    v_hat = velocity_b / V
    D_b = -D_mag * v_hat

    # Lift & crosswind: rotate from wind frame to body frame.
    # Wind-to-body rotation R_bn uses standard aerodynamic convention.
    # R_nb (body→wind) = Rz(-beta) @ Ry(alpha)
    # R_bn = R_nb^T = Ry(-alpha) @ Rz(beta)
    #
    # Ry(-alpha) = [[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]]
    #
    # Wind-frame z-axis expressed in body: R_bn @ [0,0,1]
    #   Rz(beta)  @ [0,0,1] = [0, 0, 1]
    #   Ry(-alpha)@ [0,0,1] = [-sin(alpha), 0, cos(alpha)]
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    z_wind_in_body = np.array([-sa,       0.0, ca      ])
    # Wind-frame y-axis expressed in body: R_bn @ [0,1,0]
    #   Rz(beta)  @ [0,1,0] = [-sin(beta), cos(beta), 0]
    #   Ry(-alpha)@ [-sb, cb, 0] = [-sb*cos(alpha), cb, -sb*sin(alpha)]
    y_wind_in_body = np.array([-sb * ca, cb, -sb * sa ])

    L_b = L_mag * z_wind_in_body
    C_b = Fs    * y_wind_in_body

    FA_b = D_b + L_b + C_b

    # --- Moments in body frame ---
    # Aerodynamic moments from control surfaces and stability derivatives:
    MA_b = np.array([
        qbar * S * b * Cl_c,
        qbar * S * c * Cm_c,
        qbar * S * b * Cn_c,
    ])

    # Aerodynamic damping (matches fixedwing_4ch.py lines 286-290).
    # These are added as raw values *after* scaling, matching the CasADi model.
    MA_b[0] += pr["Clp"] * b / (2.0 * V) * P   # Roll damping
    MA_b[0] += pr["Clr"] * b / (2.0 * V) * R   # Roll due to yaw rate
    MA_b[1] += pr["Cmq"] * c / (2.0 * V) * Q   # Pitch damping
    MA_b[2] += pr["Cnp"] * b / (2.0 * V) * P   # Yaw due to roll rate
    MA_b[2] += pr["Cnr"] * b / (2.0 * V) * R   # Yaw damping

    return FA_b, MA_b, alpha, beta, CL, CD


def _ground(position_w: np.ndarray, velocity_b: np.ndarray,
            omega_b: np.ndarray, R_bw: np.ndarray):
    """Ground-contact spring/damper (matches fixedwing_4ch.py lines 296-321)."""
    R_wb = R_bw.T
    wheel_positions_b = [
        np.array([ 0.1,  0.1, -0.1]),   # left main wheel
        np.array([ 0.1, -0.1, -0.1]),   # right main wheel
        np.array([-0.4,  0.0,  0.0]),   # tail wheel
    ]
    FG_w_total = np.zeros(3)
    MG_b = np.zeros(3)

    for wb in wheel_positions_b:
        wheel_w   = R_bw @ wb
        pos_w_whl = position_w + wheel_w
        if pos_w_whl[2] < 0.0:
            vel_whl_b = velocity_b + np.cross(omega_b, wb)
            vel_whl_w = R_bw @ vel_whl_b
            # Vertical spring/damper + horizontal friction
            fz = np.clip(-pos_w_whl[2] * 10.0, -100.0, 100.0) \
                 - vel_whl_w[2] * 1.10
            fx = -vel_whl_w[0] * 0.001
            fy = -vel_whl_w[1] * 0.001
            force_w = np.array([fx, fy, fz])
            FG_w_total += force_w
            MG_b += np.cross(wb, R_wb @ force_w)

    FG_b = R_wb @ FG_w_total
    return FG_b, MG_b


def _derivatives(state: np.ndarray, action: np.ndarray, pr: dict) -> np.ndarray:
    """
    Compute d(state)/dt for state = [x,y,z, u,v,w, phi,theta,psi, p,q,r].
    Matches the equations in fixedwing_4ch.py lines 342-353.
    """
    x_p, y_p, z_p     = state[0:3]
    u, v, w            = state[3:6]
    phi, theta, psi    = state[6:9]
    p_b, q_b, r_b      = state[9:12]

    velocity_b = np.array([u, v, w])
    omega_b    = np.array([p_b, q_b, r_b])
    position_w = np.array([x_p, y_p, z_p])

    R_bw = _rot_bw(phi, theta, psi)   # body → world
    R_wb = R_bw.T                      # world → body

    # --- Forces ---
    FA_b, MA_b, *_ = _aero(velocity_b, omega_b, action, pr)
    FG_b, MG_b     = _ground(position_w, velocity_b, omega_b, R_bw)

    throttle = max(float(action[2]), 1e-3)
    FT_b = np.array([pr["thr_max"] * throttle, 0.0, 0.0])

    # Gravity in body frame: world gravity = [0,0,-mg] (ENU, z-up)
    FW_b = R_wb @ np.array([0.0, 0.0, -pr["m"] * pr["g"]])

    F_b = FA_b + FG_b + FT_b + FW_b
    M_b = MA_b + MG_b

    # --- Translational dynamics (Newton, body frame) ---
    # dv/dt = F/m - omega × v  (Coriolis term for rotating frame)
    dv_b = F_b / pr["m"] - np.cross(omega_b, velocity_b)

    # --- Position kinematics ---
    dp_w = R_bw @ velocity_b

    # --- Euler angle kinematics ---
    deuler = _euler_rates(phi, theta, p_b, q_b, r_b)

    # --- Rotational dynamics (Euler's equation) ---
    Jx, Jy, Jz, Jxz = pr["Jx"], pr["Jy"], pr["Jz"], pr["Jxz"]
    J = np.array([
        [Jx,  0.0, Jxz],
        [0.0, Jy,  0.0],
        [Jxz, 0.0, Jz ],
    ])
    domega_b = np.linalg.solve(J, M_b - np.cross(omega_b, J @ omega_b))

    return np.concatenate([dp_w, dv_b, deuler, domega_b])


def _rk4_step(state: np.ndarray, action: np.ndarray, dt: float, pr: dict) -> np.ndarray:
    """Single 4th-order Runge-Kutta step."""
    k1 = _derivatives(state,               action, pr)
    k2 = _derivatives(state + 0.5*dt*k1,   action, pr)
    k3 = _derivatives(state + 0.5*dt*k2,   action, pr)
    k4 = _derivatives(state +     dt*k3,   action, pr)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def _rk4(state: np.ndarray, action: np.ndarray, dt: float, pr: dict,
         n_substeps: int = 4) -> np.ndarray:
    """
    Integrate over dt using n_substeps of RK4.

    The roll-damping eigenvalue is ~294 rad/s, requiring dt_sub < 9.5 ms
    for RK4 stability.  With n_substeps=4 and a typical action dt=0.02 s,
    each sub-step is 5 ms — within the stability region.
    """
    dt_sub = dt / n_substeps
    s = state
    for _ in range(n_substeps):
        s = _rk4_step(s, action, dt_sub, pr)
    return s


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class MockPylonRacingEnv(gym.Env):
    """
    Drop-in replacement for PylonRacingEnv with full 6-DOF physics (no ROS2).

    Uses the same aerodynamic model as fixedwing_4ch.py so agents trained here
    should transfer more reliably to the ROS2/Gazebo simulation.

    Observation (15D) layout matches ROS2 PylonRacingEnv exactly:
      [x, y, z, roll, pitch, yaw, vx, vy, vz, v, gamma, vdot, p, q, r]
    Action (4D): [aileron, elevator, throttle, rudder] in [-1, 1]
    """

    def __init__(self, dt: float = 0.02, seed=None, course=DEFAULT_COURSE):
        self._course = get_course(course)
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.dt = dt
        self._params = _PARAMS.copy()
        self._rng = np.random.default_rng(seed)

        # Internal 12D state: [x,y,z, u,v,w, phi,theta,psi, p,q,r]
        self._state = np.zeros(12, dtype=np.float64)
        self._prev_speed = 0.0
        self.has_taken_off = False

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        x, y, z       = self._state[0:3]
        u, v_b, w     = self._state[3:6]
        phi, theta, psi = self._state[6:9]
        p, q, r       = self._state[9:12]

        R_bw = _rot_bw(phi, theta, psi)
        vel_world = R_bw @ np.array([u, v_b, w])
        vx, vy, vz = vel_world

        V = float(np.sqrt(u*u + v_b*v_b + w*w))
        V = max(V, 1e-9)
        gamma = np.arcsin(np.clip(vz / V, -1.0, 1.0))
        vdot = (V - self._prev_speed) / self.dt

        # NOTE on sign convention: internally, the cyecca-matched model uses the
        # same frame as fixedwing_4ch.py where *negative* theta = nose-up.
        # To produce obs consistent with ROS2 pylon_env (positive pitch = nose up)
        # we negate theta in the observation.
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0],  obs[1],  obs[2]  = x, y, z
        obs[3],  obs[4],  obs[5]  = phi, -theta, psi   # negate theta → nose-up positive
        obs[6],  obs[7],  obs[8]  = vx, vy, vz
        obs[9]  = V
        obs[10] = gamma
        obs[11] = vdot
        obs[12], obs[13], obs[14] = p, q, r
        return obs

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        u, v_b, w = self._state[3], self._state[4], self._state[5]
        self._prev_speed = float(np.sqrt(u*u + v_b*v_b + w*w))

        self._state = _rk4(self._state, action, self.dt, self._params)

        # Safety clamp: prevent NaN / overflow from extreme states
        if not np.all(np.isfinite(self._state)):
            self._state = np.nan_to_num(self._state, nan=0.0, posinf=0.0, neginf=0.0)
        V_body = float(np.linalg.norm(self._state[3:6]))
        if V_body > 50.0:   # hard cap at 50 m/s (well above physical max)
            self._state[3:6] *= 50.0 / V_body

        # Ground clamp: z cannot go below 0
        if self._state[2] < 0.0:
            self._state[2] = 0.0
            if self._state[5] < 0.0:   # w (upward body velocity)
                self._state[5] = 0.0

        obs = self._get_obs()
        z = float(obs[2])

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

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Trim initial conditions for level cruise at 7 m/s, z=7 m altitude.
        #
        # Lift = weight requires:
        #   CL_trim = m*g / (qbar*S) = 0.337  →  alpha_trim = 0.026 rad
        #
        # In the cyecca frame convention (matching fixedwing_4ch.py):
        #   - negative theta = nose-up (body z-up, Ry(theta) rotates x toward +z_world)
        #   - w_body < 0 for positive angle of attack (freestream downward in body z-up)
        #
        # Trim (Cm=0) at alpha=0.026 requires elevator offset:
        #   0 = Cm0 + Cma*alpha + Cmde*elev_rad  →  elev_rad = -0.0176 rad
        #   Because elev_rad = -MAX_ELEV * action[1]:  action[1]_trim = 0.042
        _alpha = 0.026    # trim angle of attack (rad)
        _V     = 7.0      # trim airspeed (m/s)
        self._state = np.zeros(12, dtype=np.float64)
        self._state[0] = 0.0                      # x
        self._state[1] = 0.0                      # y
        self._state[2] = 7.0                      # z (altitude m)
        # Randomize starting heading so the agent learns all orientations
        psi = self._rng.uniform(-np.pi, np.pi)
        self._state[3] = _V * np.cos(_alpha)      # u (forward body speed)
        self._state[5] = -_V * np.sin(_alpha)     # w (z-up body: negative = nose-up AoA)
        self._state[7] = -_alpha                  # theta (cyecca: negative = nose up)
        self._state[8] = psi                      # yaw (random heading)
        self._trim_elevator = 0.042               # elevator action for level trim

        self._prev_speed = 7.0
        self.has_taken_off = True
        return self._get_obs(), {}
