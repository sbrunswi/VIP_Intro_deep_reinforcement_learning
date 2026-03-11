# Erin's Custom Pylon Environment

A **minimal deep RL environment** for training fixed-wing drones to navigate pylon gates.
Experiments stay completely inside `Individual Workspace/Erin`; no modifications to upstream code.

## Environment Features

- **Sparse rewards:** +10 per gate crossing, -10 for crash/stall, -0.01 × distance to target
- **Fixed-wing physics:** coordinated turns, min speed (2 m/s), pitch control, stall detection
- **Gate tracking:** automatically switches waypoints and counts laps
- **17D observations:** 15D state (position, attitude, velocity, rates) + 2D vector to next gate
- **4D actions:** aileron, elevator, throttle, rudder

## Structure

- `mock_pylon_env.py` – RL-ready environment with gate crossing detection and sparse rewards
- `env_factory.py` – factory with `use_erin` flag to select this environment
- `train.py` – minimal Q-learning trainer for Erin's environment
- `test_pylon_gym.py` – demo script (manual PD control; see RL training examples below)

## Quick Start

From the **repository root**, set up the path and run:

```powershell
# Set up path (run this first)
$env:PYTHONPATH = "$PWD\Individual Workspace;$env:PYTHONPATH"

# Train RL agent
python "Individual Workspace\Erin\train.py" --train 100 --viz

# Test manual control
python "Individual Workspace\Erin\test_pylon_gym.py" --viz
```

### Alternative: Inline setup

```powershell
# One-liner setup and run (from repo root)
$env:PYTHONPATH = "$PWD\Individual Workspace;$env:PYTHONPATH"; python "Individual Workspace\Erin\train.py" --train 50 --viz
```

## How It Works

### Environment (`mock_pylon_env.py`)

The environment simulates a fixed-wing drone racing through 4 pylons in a loop:

1. **Physics Model:**
   - **Actions:** 4D continuous (aileron, elevator, throttle, rudder) in [-1, 1]
   - **Dynamics:** Simplified fixed-wing with coordinated turns, minimum speed (2 m/s), stall detection
   - **State:** 6D core state (x,y,z,vx,vy,vz) + derived 15D observations (position, attitude, velocity, rates)

2. **Observations (17D):**
   - **0-14:** Standard 15D state (x,y,z,roll,pitch,yaw,vx,vy,vz,v,gamma,vdot,p,q,r)
   - **15-16:** 2D vector to next gate midpoint (guides RL agent toward target)

3. **Rewards:**
   - **+10:** Gate crossing (sparse reward for navigation)
   - **-10:** Crash or stall (v < 3 m/s or z < 0.5 m)
   - **-0.01 × distance:** Small penalty for being far from target gate

4. **Termination:**
   - Episode ends on crash/stall or after 500 steps
   - Gate crossings automatically advance to next waypoint
   - Lap counter tracks complete circuits

### Training (`train.py`)

Uses **Q-learning** with epsilon-greedy exploration:

1. **State Discretization:** Converts 17D continuous observations to discrete states (position bins + heading bins)
2. **Q-Table:** Learns action values for each state (4 actions: random samples from action space)
3. **Training Loop:**
   - Epsilon-greedy: 80% exploit learned Q-values, 20% random exploration
   - Q-learning update: Q(s,a) += α × (r + γ × max(Q(s',a')) - Q(s,a))
   - Parameters: α=0.1 (learning rate), γ=0.99 (discount), ε=0.2 (exploration)

4. **Persistence:**
   - **Checkpoints:** Q-table saved to `q_table.pkl` after training
   - **Metrics:** Episode rewards/laps/steps logged to `metrics.csv`
   - **Resume:** Load saved model with `--load-model q_table.pkl`

### Manual Control (`test_pylon_gym.py`)

Demonstrates PD altitude-hold controller (no RL):

- **Throttle:** Maintains speed ~5 m/s
- **Elevator:** PD control for altitude hold at 7m
- **Aileron/Rudder:** Zero (straight flight)
- **Result:** Drone flies straight and holds altitude, but doesn't navigate gates

## Usage Examples

### Basic Training

```powershell
# Train from scratch
$env:PYTHONPATH = "$PWD\Individual Workspace;$env:PYTHONPATH"
python "Individual Workspace\Erin\train.py" --train 100 --viz
```

This trains for 100 episodes, saves the model, and runs 3 evaluation episodes with 3D visualization.

### Resume Training

```powershell
# Continue training from saved checkpoint
python "Individual Workspace\Erin\train.py" --train 50 --load-model q_table.pkl --viz
```

### Evaluate Only

```powershell
# Load and evaluate saved model (no training)
python "Individual Workspace\Erin\train.py" --eval-only --load-model q_table.pkl --viz
```

### Custom Training Script

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'Individual Workspace'))

from Erin.env_factory import make_pylon_env
import numpy as np

env = make_pylon_env(use_erin=True)

for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # random actions for demo
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode}: Reward={total_reward:.1f}, Laps={info['laps']}")

env.close()
```

## Files Created During Training

- **`q_table.pkl`** – Trained Q-table (binary, ~100KB)
- **`metrics.csv`** – Training history:
  ```
  Episode,Total_Reward,Laps,Steps
  1,-5.2,0,45
  2,15.8,1,120
  ...
  ```

## Troubleshooting

**`ModuleNotFoundError: No module named 'Erin'`**

Run from repository root and set PYTHONPATH:

```powershell
$env:PYTHONPATH = "$PWD\Individual Workspace;$env:PYTHONPATH"
```

**`ValueError: could not broadcast...`**

Check that `OBS_SIZE = 17` in `mock_pylon_env.py` matches the observation array.

**Drone flies straight and doesn't navigate gates**

The `test_pylon_gym.py` uses manual PD control (no RL). Train an RL agent with `train.py` to learn gate navigation.

**Training not improving**

- Increase `--train` episodes (try 500+)
- Adjust discretization in `discretize_obs()` (finer bins)
- Tune Q-learning parameters (α, γ, ε) in `train.py`

## Differences from Stock `no_ros2`

| Feature         | Stock            | Erin                                |
| --------------- | ---------------- | ----------------------------------- |
| Rewards         | +1/step airborne | +10/gate, -10/crash, -0.01/distance |
| Observation     | 15D              | 17D (15D + 2D gate vector)          |
| Gate tracking   | None             | Yes; auto-switches waypoints        |
| Min speed       | 0 m/s            | 2 m/s (stall)                       |
| Target guidance | None             | Vector to next gate in obs          |

## Extending the Environment

Edit `mock_pylon_env.py` to customize:

- **Physics:** Modify `_integrate()` for more realistic aerodynamics
- **Rewards:** Change gate bonus, crash penalty, distance penalty in `step()`
- **Observations:** Add features to `_get_obs()` (e.g., angular acceleration)
- **Gates:** Update `self._gates` and `self._pylons` for different courses

Example custom reward:

```python
# Add heading alignment bonus
target_heading = np.arctan2(obs[16], obs[15])
heading_error = abs(obs[5] - target_heading)
reward -= 0.05 * heading_error  # penalty for wrong heading
```

All modifications stay local to `Individual Workspace/Erin`.
