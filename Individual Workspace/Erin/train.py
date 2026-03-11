#!/usr/bin/env python3
r"""
Minimal Q-learning trainer for Erin's pylon environment with checkpoint saving and metrics logging.
Usage: python Individual\Workspace\Erin\train.py [--train N] [--viz] [--load-model FILE] [--eval-only]
"""
import sys, os, pickle, csv
sys.path.insert(0, os.path.join(os.getcwd(), 'Individual Workspace'))

import numpy as np
from collections import defaultdict
from Erin.env_factory import make_pylon_env
from Erin.mock_pylon_env import OBS_SIZE

CHECKPOINT_FILE = "q_table.pkl"
METRICS_FILE = "metrics.csv"

def discretize_obs(obs):
    """Convert 17D observation to a discrete state."""
    x, y, z = obs[0], obs[1], obs[2]
    # Simple: bin position and heading
    region = (int(x / 5) % 10, int(y / 5) % 10, int(obs[5] / 0.5) % 12)
    return region

def load_model(filepath):
    """Load Q-table from checkpoint."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            print(f"✓ Loaded model from {filepath}")
            return pickle.load(f)
    return defaultdict(lambda: np.zeros(4))

def save_model(Q, filepath):
    """Save Q-table to checkpoint."""
    with open(filepath, 'wb') as f:
        pickle.dump(Q, f)
    print(f"✓ Saved model to {filepath}")

def log_metrics(episode, total_reward, laps, steps):
    """Append metrics to CSV file."""
    file_exists = os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'Total_Reward', 'Laps', 'Steps'])
        writer.writerow([episode, total_reward, laps, steps])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=50, help="Training episodes")
    parser.add_argument("--viz", action="store_true", help="Visualize")
    parser.add_argument("--load-model", type=str, help="Load Q-table from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate loaded model (no training)")
    args = parser.parse_args()

    env = make_pylon_env(use_erin=True)
    viz = None
    if args.viz:
        from no_ros2.viz_3d import PylonRacingViz3D
        viz = PylonRacingViz3D()

    # Load or initialize Q-table
    if args.load_model:
        Q = load_model(args.load_model)
    else:
        Q = defaultdict(lambda: np.zeros(4))
        print(f"✓ Initialized new Q-table")

    # Training phase (skipped if --eval-only)
    if not args.eval_only and args.train > 0:
        alpha, gamma, epsilon = 0.1, 0.99, 0.2
        print(f"\nTraining for {args.train} episodes...")

        for episode in range(args.train):
            obs, _ = env.reset()
            state = discretize_obs(obs)
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < 500:
                # Epsilon-greedy
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = Q[state]

                obs, reward, done, _, info = env.step(action)
                next_state = discretize_obs(obs)

                # Q-learning update
                Q[state] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state])
                state = next_state
                total_reward += reward
                steps += 1

                if viz:
                    viz.update(obs, reward=reward, step_count=steps, laps=info.get('laps', 0))

            laps = info.get('laps', 0)
            log_metrics(episode + 1, total_reward, laps, steps)
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{args.train} | Reward: {total_reward:.1f} | Laps: {laps} | Steps: {steps}")

        # Save trained model
        save_model(Q, CHECKPOINT_FILE)

    # Evaluation phase
    print(f"\nEvaluating model...")
    eval_episodes = 3
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 500:
            state = discretize_obs(obs)
            action = Q[state]
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            if viz:
                viz.update(obs, reward=reward, step_count=steps, laps=info.get('laps', 0))
        print(f"  Eval {ep+1}/{eval_episodes}: Reward {total_reward:.1f}, Laps {info.get('laps', 0)}, Steps {steps}")

    print(f"\n✓ Metrics saved to {METRICS_FILE}")
    if viz:
        viz.close()
    env.close()
