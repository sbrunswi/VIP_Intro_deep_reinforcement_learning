#!/usr/bin/env python3
"""
test_dynamics_match.py
----------------------
Compares the state derivatives (xdot) of mock_pylon_env.py vs fixedwing_4ch.py
at identical states and actions to verify the dynamics match.

Uses CasADi to evaluate the ROS2 model directly, and numpy for the mock env.
Prints per-component errors and a PASS/FAIL summary.

Usage:
    python no_ros2/test_dynamics_match.py
"""

import sys, os
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

# --- Import mock env dynamics ---
from no_ros2.environments_v2.mock_pylon_env import (
    _derivatives, _aero, _rot_bw, _PARAMS, _MAX_AIL, _MAX_ELEV, _MAX_RUD,
)

# --- Import CasADi model (bypass cyecca __init__ which needs Python 3.12+) ---
import importlib.util as _ilu

def _load_module_from_path(fpath, name):
    spec = _ilu.spec_from_file_location(name, fpath)
    mod = _ilu.module_from_spec(spec)
    # Pre-load cyecca.lie so the model file can import it
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load cyecca.lie first (the model depends on it)
_cyecca_root = os.path.join(_root, "ros2_ws", "src", "cyecca")
sys.path.insert(0, _cyecca_root)
# Only import the lie subpackage, not the full cyecca package
import cyecca.lie as _lie  # noqa: this works because cyecca/lie/__init__.py is self-contained

_fw4ch = _load_module_from_path(
    os.path.join(_cyecca_root, "cyecca", "models", "fixedwing_4ch.py"),
    "fixedwing_4ch",
)
derive_model = _fw4ch.derive_model


def euler_to_quat_zyx(phi, theta, psi):
    """ZYX Euler angles -> quaternion [w, x, y, z] (Hamilton, scalar-first).
    Matches cyecca SO3Quat convention: identity = [1, 0, 0, 0]."""
    cp, sp = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    # ZYX convention
    w = cp * ct * cy + sp * st * sy
    x = sp * ct * cy - cp * st * sy
    y = cp * st * cy + sp * ct * sy
    z = cp * ct * sy - sp * st * cy
    return np.array([w, x, y, z])  # scalar-first for cyecca


def run_comparison(state_12, action_4, label=""):
    """
    state_12: [x,y,z, u,v,w, phi,theta,psi, p,q,r]
    action_4: [aileron, elevator, throttle, rudder] in [-1,1]

    Returns dict with per-component absolute errors.
    """
    x, y, z     = state_12[0:3]
    u, v, w     = state_12[3:6]
    phi, theta, psi = state_12[6:9]
    pb, qb, rb  = state_12[9:12]

    # ---- Mock env derivatives (numpy) ----
    xdot_mock = _derivatives(state_12, action_4, _PARAMS)
    # xdot_mock layout: [dx,dy,dz, du,dv,dw, dphi,dtheta,dpsi, dp,dq,dr]

    # ---- CasADi model derivatives ----
    model = derive_model()
    f = model["f"]
    p_defaults = model["p_defaults"]
    p_vals = [p_defaults[model["p"][i].name()] for i in range(model["p"].shape[0])]

    # Build CasADi state: [position_w(3), velocity_b(3), quat_wb(4), omega_wb_b(3)]
    quat = euler_to_quat_zyx(phi, theta, psi)
    x_cas = np.concatenate([
        [x, y, z],
        [u, v, w],
        quat,
        [pb, qb, rb],
    ])

    # CasADi action: [ail_cmd, elev_cmd, throttle_cmd, rud_cmd]
    # The mock env sign convention:
    #   ail_rad  =  MAX_AIL  * action[0]
    #   elev_rad = -MAX_ELEV * action[1]   (mock negates)
    #   rud_rad  = -MAX_RUD  * action[3]   (mock negates)
    # CasADi (fixedwing_4ch.py lines 244-246):
    #   ail_rad  = max_defl_ail * DEG2RAD * u[0] * 1
    #   elev_rad = max_defl_elev * DEG2RAD * u[1] * 1
    #   rud_rad  = max_defl_rud * DEG2RAD * u[3] * -1
    # So CasADi: elev_rad = +MAX_ELEV * u[1], but mock: elev_rad = -MAX_ELEV * action[1]
    # To get the same elev_rad: u_cas[1] = -action[1]
    # CasADi: rud_rad = -MAX_RUD * u[3], mock: rud_rad = -MAX_RUD * action[3]
    # Same sign, so u_cas[3] = action[3]
    u_cas = np.array([
        action_4[0],     # aileron: same
        -action_4[1],    # elevator: negate to match mock's sign flip
        action_4[2],     # throttle: same
        action_4[3],     # rudder: same
    ])

    xdot_cas_raw = np.array(f(x_cas, u_cas, p_vals)).flatten()
    # CasADi xdot: [d_pos_w(3), d_vel_b(3), d_quat(4), d_omega(3)]

    # Also get CasADi Info for force diagnostics
    Info = model["Info"]
    info_res = Info(x_cas, u_cas, p_vals)
    cas_L_b = np.array(info_res[0]).flatten()  # L_b
    cas_D_b = np.array(info_res[1]).flatten()  # D_b
    cas_C_b = np.array(info_res[2]).flatten()  # C_b
    cas_FW_b = np.array(info_res[3]).flatten() # FW_b (gravity)
    cas_FT_b = np.array(info_res[4]).flatten() # FT_b (thrust)
    cas_CL = float(info_res[7])
    cas_CD = float(info_res[8])
    cas_alpha = float(info_res[9])
    cas_beta = float(info_res[11])

    # Mock env force breakdown
    velocity_b = state_12[3:6]
    omega_b = state_12[9:12]
    mock_FA_b, mock_MA_b, mock_alpha, mock_beta, mock_CL, mock_CD = _aero(
        velocity_b, omega_b, action_4, _PARAMS)
    R_bw = _rot_bw(phi, theta, psi)
    R_wb = R_bw.T
    mock_FW_b = R_wb @ np.array([0, 0, -_PARAMS["m"] * _PARAMS["g"]])
    throttle = max(float(action_4[2]), 1e-3)
    mock_FT_b = np.array([_PARAMS["thr_max"] * throttle, 0, 0])

    # Extract comparable components
    dpos_cas   = xdot_cas_raw[0:3]   # world position rate
    dvel_cas   = xdot_cas_raw[3:6]   # body velocity rate
    dquat_cas  = xdot_cas_raw[6:10]  # quaternion rate
    domega_cas = xdot_cas_raw[10:13] # angular rate derivative

    dpos_mock   = xdot_mock[0:3]
    dvel_mock   = xdot_mock[3:6]
    deuler_mock = xdot_mock[6:9]
    domega_mock = xdot_mock[9:12]

    # --- Compare position derivatives (world velocity) ---
    err_pos = np.abs(dpos_cas - dpos_mock)

    # --- Compare velocity derivatives (body acceleration) ---
    err_vel = np.abs(dvel_cas - dvel_mock)

    # --- Compare angular acceleration ---
    err_omega = np.abs(domega_cas - domega_mock)

    # --- Convert CasADi quat rate to Euler rate for comparison ---
    # d(quat)/dt = 0.5 * quat * omega  (quaternion kinematics)
    # We can't directly compare quat rate vs euler rate, but we can compare
    # the angular velocity they represent. Instead, let's just compare
    # the angular acceleration (domega) which is the physically meaningful part.
    # The Euler rate and quat rate are just different parameterizations of the same rotation.

    # For Euler rate comparison, compute Euler rate from quaternion rate:
    # This is complex, so skip and focus on the dynamics (forces/moments).

    results = {
        "label": label,
        "dpos_err": err_pos,
        "dvel_err": err_vel,
        "domega_err": err_omega,
        "dpos_cas": dpos_cas,
        "dpos_mock": dpos_mock,
        "dvel_cas": dvel_cas,
        "dvel_mock": dvel_mock,
        "domega_cas": domega_cas,
        "domega_mock": domega_mock,
        # Force diagnostics
        "cas_L_b": cas_L_b, "cas_D_b": cas_D_b, "cas_C_b": cas_C_b,
        "cas_FW_b": cas_FW_b, "cas_FT_b": cas_FT_b,
        "cas_FA_b": cas_L_b + cas_D_b + cas_C_b,
        "mock_FA_b": mock_FA_b, "mock_FW_b": mock_FW_b, "mock_FT_b": mock_FT_b,
        "cas_alpha": cas_alpha, "mock_alpha": mock_alpha,
        "cas_beta": cas_beta, "mock_beta": mock_beta,
        "cas_CL": cas_CL, "mock_CL": mock_CL,
        "cas_CD": cas_CD, "mock_CD": mock_CD,
        "mock_MA_b": mock_MA_b,
    }
    return results


def print_results(r, tol=1e-4):
    label = r["label"]
    print(f"\n{'='*60}")
    print(f"  Test case: {label}")
    print(f"{'='*60}")

    names_pos = ["dx/dt", "dy/dt", "dz/dt"]
    names_vel = ["du/dt", "dv/dt", "dw/dt"]
    names_omg = ["dp/dt", "dq/dt", "dr/dt"]

    all_pass = True
    for group, names, cas_key, mock_key, err_key in [
        ("Position rate (world vel)", names_pos, "dpos_cas", "dpos_mock", "dpos_err"),
        ("Velocity rate (body accel)", names_vel, "dvel_cas", "dvel_mock", "dvel_err"),
        ("Angular accel", names_omg, "domega_cas", "domega_mock", "domega_err"),
    ]:
        print(f"\n  {group}:")
        print(f"  {'Component':<12} {'CasADi':>12} {'Mock':>12} {'Error':>12} {'Status':>8}")
        print(f"  {'-'*56}")
        for i, name in enumerate(names):
            cas_val  = r[cas_key][i]
            mock_val = r[mock_key][i]
            err_val  = r[err_key][i]
            # Use relative tolerance for large values, absolute for small
            ref = max(abs(cas_val), abs(mock_val), 1e-6)
            rel_err = err_val / ref
            ok = rel_err < tol or err_val < 1e-8
            status = "OK" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  {name:<12} {cas_val:>12.6f} {mock_val:>12.6f} {err_val:>12.2e} {status:>8}")

    # Diagnostics for failing cases
    if not all_pass:
        print(f"\n  --- Force Diagnostics ---")
        print(f"  alpha:  CasADi={r['cas_alpha']:.6f}  Mock={r['mock_alpha']:.6f}")
        print(f"  beta:   CasADi={r['cas_beta']:.6f}  Mock={r['mock_beta']:.6f}")
        print(f"  CL:     CasADi={r['cas_CL']:.6f}  Mock={r['mock_CL']:.6f}")
        print(f"  CD:     CasADi={r['cas_CD']:.6f}  Mock={r['mock_CD']:.6f}")
        for comp, i in [("x", 0), ("y", 1), ("z", 2)]:
            print(f"  FA_b[{comp}]: CasADi={r['cas_FA_b'][i]:>10.6f}  Mock={r['mock_FA_b'][i]:>10.6f}  err={abs(r['cas_FA_b'][i]-r['mock_FA_b'][i]):.2e}")
        for comp, i in [("x", 0), ("y", 1), ("z", 2)]:
            print(f"  FW_b[{comp}]: CasADi={r['cas_FW_b'][i]:>10.6f}  Mock={r['mock_FW_b'][i]:>10.6f}  err={abs(r['cas_FW_b'][i]-r['mock_FW_b'][i]):.2e}")
        for comp, i in [("x", 0), ("y", 1), ("z", 2)]:
            print(f"  FT_b[{comp}]: CasADi={r['cas_FT_b'][i]:>10.6f}  Mock={r['mock_FT_b'][i]:>10.6f}  err={abs(r['cas_FT_b'][i]-r['mock_FT_b'][i]):.2e}")
        # Lift/Drag components
        for comp, i in [("x", 0), ("y", 1), ("z", 2)]:
            print(f"  L_b[{comp}]:  CasADi={r['cas_L_b'][i]:>10.6f}")
        for comp, i in [("x", 0), ("y", 1), ("z", 2)]:
            print(f"  D_b[{comp}]:  CasADi={r['cas_D_b'][i]:>10.6f}")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    DEG = np.pi / 180

    test_cases = [
        {
            "label": "Level cruise (trim-like)",
            "state": np.array([0, 0, 7,  7.0, 0, -0.18,  0, -0.026, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.0, 0.042, 0.5, 0.0], dtype=np.float32),
        },
        {
            "label": "Banked turn (15 deg roll, 5 deg pitch)",
            "state": np.array([5, 3, 10,  6.5, 0.3, -0.5,  15*DEG, -5*DEG, 30*DEG,  0.1, 0.05, 0.02], dtype=np.float64),
            "action": np.array([0.3, 0.1, 0.6, 0.1], dtype=np.float32),
        },
        {
            "label": "Aggressive maneuver",
            "state": np.array([10, -5, 5,  8.0, 1.0, -1.0,  30*DEG, -10*DEG, 45*DEG,  0.5, -0.3, 0.2], dtype=np.float64),
            "action": np.array([0.8, -0.5, 0.9, -0.4], dtype=np.float32),
        },
        {
            "label": "Near stall (high AoA, low speed)",
            "state": np.array([0, 0, 3,  3.0, 0.0, -1.5,  0, -25*DEG, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.0, 0.5, 0.3, 0.0], dtype=np.float32),
        },
        {
            "label": "Pure sideslip",
            "state": np.array([0, 0, 7,  6.5, 1.5, 0.0,  0, 0, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.0, 0.0, 0.5, 0.3], dtype=np.float32),
        },
        {
            "label": "Inverted flight (180 deg roll)",
            "state": np.array([0, 0, 10,  7.0, 0, -0.2,  180*DEG, -0.03, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.0, -0.3, 0.7, 0.0], dtype=np.float32),
        },
        {
            "label": "High angular rates (spinning)",
            "state": np.array([0, 0, 7,  5.0, 0.5, -0.3,  10*DEG, -5*DEG, 0,  2.0, 1.0, 0.5], dtype=np.float64),
            "action": np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float32),
        },
        {
            "label": "Ground contact (z < wheel height)",
            "state": np.array([0, 0, 0.05,  5.0, 0, 0,  0, 0, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32),
        },
        {
            "label": "Combined sideslip + AoA + rates",
            "state": np.array([0, 0, 7,  6.0, 1.0, -0.8,  20*DEG, -8*DEG, 90*DEG,  0.3, -0.2, 0.4], dtype=np.float64),
            "action": np.array([-0.5, 0.3, 0.4, 0.6], dtype=np.float32),
        },
        {
            "label": "Steep dive (45 deg nose-down)",
            "state": np.array([0, 0, 15,  7.0, 0, 0,  0, 45*DEG, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.0, 0.0, 0.3, 0.0], dtype=np.float32),
        },
        {
            "label": "Asymmetric ground contact (rolled on ground)",
            "state": np.array([0, 0, 0.02,  3.0, 0.5, 0,  15*DEG, 0, 0,  0, 0, 0], dtype=np.float64),
            "action": np.array([0.2, 0.0, 0.5, 0.1], dtype=np.float32),
        },
    ]

    all_pass = True
    for tc in test_cases:
        r = run_comparison(tc["state"], tc["action"], tc["label"])
        ok = print_results(r, tol=0.01)  # 1% relative tolerance
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    print(f"  FINAL: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
