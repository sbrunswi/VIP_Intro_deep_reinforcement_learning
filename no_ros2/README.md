# No-ROS2 Pylon Racing Stack

A **no-ROS2** pipeline for developing and testing pylon racing agents and environments without Gazebo or ROS2. The same gym interface and 15D state vector are used so agents can be trained here and, if desired, run against the real ROS2 sim later.

---

## Overview

- **Mock environment** — Same action/observation spaces and step/reset/done/reward semantics as the ROS2 `PylonRacingEnv`; no `rclpy` or `tf_transformations`.
- **15D state vector** — Full observation aligned with the TECS controller’s `actual_data`: position, velocity, attitude, speed, flight path angle, acceleration, and body rates.
- **Unified factory** — `make_pylon_env(use_ros2=False)` uses only this folder and gym/numpy; `use_ros2=True` uses the ROS2 env (caller must handle `rclpy.init`/`shutdown`).
- **Agents** — Q-learning (tabular, 15D discretized state).
- **3D visualization** — Matplotlib-based viewer with chase camera, trail, and HUD (time, laps, alt, speed, reward, step). Use `--show-details` with `--viz` to show axis labels/ticks and the full 15D state vector panel.

---

## Layout: 15D Observation

Same layout as ROS2 `PylonRacingEnv` (so agents can be swapped in without code changes):

| Index | Symbol  | Meaning              | Units   | Notes / constraints (TECS-aligned) |
|-------|---------|----------------------|---------|-------------------------------------|
| 0–2   | x, y, z | Position             | m       | z ≥ 0; course bounds for x, y       |
| 3–5   | roll, pitch, yaw | Attitude (Euler) | rad | roll ±30° (phi_lim); pitch ±20° (TECS) |
| 6–8   | vx, vy, vz | Velocity         | m/s     | —                                   |
| 9     | v       | Speed \|v\|          | m/s     | Typical cruise ~5–10               |
| 10    | gamma   | Flight path angle    | rad     | asin(vz/v), [−π/2, π/2]            |
| 11    | vdot    | d\|v\|/dt            | m/s²    | From finite difference              |
| 12–14 | p, q, r | Roll/pitch/yaw rate  | rad/s   | Body angular rates                  |

**Interface contract:** `observation_space.shape == (15,)`, `action_space.shape == (4,)` (aileron, elevator, throttle, rudder). With `use_ros2=True`, the same env subscribes to `/sim/odom` and publishes to `/sim/auto_joy` (Joy.axes = [a, e, t, r, 2000]); the no_ros2 script is a drop-in replacement for the TECS node.

---

## Files

| File | Purpose |
|------|--------|
| **`environments/mock_pylon_env.py`** | `MockPylonRacingEnv`: gym.Env with 15D obs, 4D action (aileron, elevator, throttle, rudder). Simple dynamics: heading + rudder turn rate, throttle/elevator for speed and climb. Exports `OBS_SIZE = 15`. |
| **`environments/pylon_wrapper.py`** | `PylonRacingWrapper`: pass-through wrapper (no slicing); exposes base env’s full 15D. No ROS2 deps. |
| **`environments/env_factory.py`** | `make_pylon_env(use_ros2=False, **kwargs)` → mock + wrapper; `use_ros2=True` → ROS2 `PylonRacingEnv` + same wrapper. |
| **`environments/pylon_course.py`** | Course data: `DEFAULT_PYLONS`, `DEFAULT_GATES`, `PYLON_HEIGHT_M`, `PYLON_MID_HEIGHT_M`, `BOUNDS_RECT`, `PYLON_NAMES`, `FINISH_GATE_INDEX`. Matches `purt_course.yaml`. |
| **`viz_3d.py`** | `PylonRacingViz3D`: 3D matplotlib viewer — pylons, gates, UAV glyph (beige), body axes, trail, chase camera. HUD: Time (M:SS), Laps, Alt, Speed, Reward, Step. With `--show-details` (and `--viz`): axis labels/ticks and full 15D state vector panel. `update(obs, reward=..., step_count=..., laps=...)`; `clear_trail()` resets time and laps. |
| **`test_pylon_gym.py`** | PD altitude-hold test script; optional `--viz`. Uses `make_pylon_env(use_ros2=False)`. |
| **`agents/example_agent.py`** | Q-learning agent: discretized 15D state (region, alt, heading to target, target pylon, roll, pitch, speed, gamma, yaw rate), 11 discrete actions. Reward shaping: pylon pass, approach, alignment, altitude hold, and 15D envelope (roll/pitch/speed/gamma). `STATE_BOUNDS` aligned with TECS. CLI: `--ros2` for direct swap with sim. |

---

## Running

From the **repository root** (so `no_ros2` and `ros2_ws/src/auav_pylon_2026` are on the path):

```bash
# Q-learning: train then eval with viz
python no_ros2/agents/example_agent.py --train 400 --viz

# Q-learning: no training, just show greedy policy
python no_ros2/agents/example_agent.py --no-train --viz --eval-episodes 3

# Viz with axis labels and full state vector (--show-details)
python no_ros2/agents/example_agent.py --no-train --viz --show-details
python no_ros2/test_pylon_gym.py --viz --show-details

# PD test with 3D viz
python no_ros2/test_pylon_gym.py --viz
```

Optional: add the repo root and `ros2_ws/src/auav_pylon_2026` to `PYTHONPATH` (e.g. in WSL) if imports fail.

---

## Direct swap with ROS2 controller

**Yes — the agents can be used as a direct swap** for the TECS + waypoint stack in the ROS2 sim:

- **Same interface:** 15D observation from `/sim/odom` (via `PylonRacingEnv`), 4D action (aileron, elevator, throttle, rudder) published to `/sim/auto_joy`. No ref_data or waypoints; the policy maps state → action.
- **How to run:** Start the Gazebo (or other) sim so `/sim/odom` and `/sim/auto_joy` are available. **Do not** run the TECS node (`sim_tecs_ros_xtrack` or the ROS2 `test_pylon_gym.py`). From the repo root with the ROS2 workspace sourced, run for example:
  ```bash
  source ros2_ws/install/setup.bash # or setup.zsh

  # Option 1: Fast Training (Mock) -> Eval in ROS 2
  # Trains 400 episodes instantly in the headless Python mock env, 
  # then connects to ROS 2 to fly the final trained model.
  python3 no_ros2/agents/example_agent.py --ros2

  # Option 2: Slow Training (ROS 2) -> Eval in ROS 2
  # connects to ROS 2 immediately. You will watch all 400 training 
  # episodes happen live in the physics engine (useful for debugging).
  python3 no_ros2/agents/example_agent.py --train-in-ros2

  # Option 3: No Training -> Eval in ROS 2 (with 3D Viz)
  # Skips training entirely and just evaluates a greedy policy in the sim.
  python3 no_ros2/agents/example_agent.py --no-train --ros2 --viz
  ```
  The script calls `rclpy.init()` when `--ros2` is set, creates `make_pylon_env(use_ros2=True)` (same 15D obs), and publishes actions to `/sim/auto_joy`; it calls `rclpy.shutdown()` on exit.

- **Caveats:** Policies are trained on the **mock** env (simplified dynamics). The real sim (and hardware) differ, so performance may drop without fine-tuning or training on the sim. Observation filtering in `PylonRacingEnv` may differ slightly from the TECS node’s `actual_data`; the 15D layout is the same.

---

## Controller Alignment

- **Roll:** ±30° from `sim.yaml` `phi_lim_deg`. Mock uses coordinated-turn roll `atan(v*r/g)`, clipped to ±60°.
- **Pitch:** TECS clips desired pitch to ±20°; mock uses pitch ≈ gamma (flight path angle).
- **Actions:** Same as ROS2: aileron, elevator, throttle, rudder in [−1, 1] (throttle [0, 1] in practice); fifth axis “mode” set to 2000 for auto.
- **State bounds** used in Q-learning (`STATE_BOUNDS` in `agents/example_agent.py`) match these limits and typical envelopes (speed, gamma, angular rates) so discretization and reward shaping are consistent with the TECS controller.

---

## Dependencies

- **Python 3** with **gymnasium**, **numpy**
- **matplotlib** (for `viz_3d.py`)

Install from the `no_ros2` folder:

```bash
pip install -r no_ros2/requirements.txt
```

No ROS2, tf_transformations, or Gazebo required for the no_ros2 path.
