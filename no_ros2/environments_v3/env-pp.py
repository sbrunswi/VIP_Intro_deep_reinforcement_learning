### This environment should be for a simple path-planning task in a 2-D grid in order to compare DP with Q-learning
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

### Define the path and visualize it
### create some visualization for thinking ### 

_repo_root = pathlib.Path(__file__).resolve().parents[0]
print(f"Repo root: {_repo_root}")
figure_dir = _repo_root / "figures"
os.makedirs(figure_dir, exist_ok=True)

# 1. Define corners (Poles) starting at min X
corners = np.array([[5, 10], [10, 45], [35, 40], [45, 15]])

# 2. Gate Waypoints (Midpoints of sides)
g1, g2 = (corners[0]+corners[1])/2, (corners[1]+corners[2])/2
g3, g4 = (corners[2]+corners[3])/2, (corners[3]+corners[0])/2

# 3. External Turning Waypoints
t1, t2 = np.array([35, 45]), np.array([15, 5])

# 4. Clockwise Sequence
sequence = [g1, g2, t1, g3, g4, t2]
path = np.array(sequence + [sequence[0]])

fig, ax = plt.subplots(figsize=(8, 8))

# Draw Poles and Gates
for i in range(4):
    p1, p2 = corners[i], corners[(i+1)%4]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=3, alpha=0.3)
ax.scatter(corners[:,0], corners[:,1], color='red', s=150, zorder=5)

# Draw Gaussian Band (Active State Space)
for i, wp in enumerate(sequence):
    color = 'blue' if i in [0,1,3,4] else 'orange'
    ax.add_patch(plt.Circle(wp, 6, color=color, alpha=0.15))
    ax.scatter(wp[0], wp[1], color=color, marker='x', s=40)

# Draw Path
ax.plot(path[:,0], path[:,1], 'g--', lw=2)

ax.set_xlim(-5, 50); ax.set_ylim(-5, 50); ax.set_aspect('equal')
save_path = figure_dir / "filtered-state-space.png"
fig.savefig(save_path)
print(f"Saved to: {save_path}")

plt.show(block=False)   # show without blocking
plt.pause(10)          # keep window open for 10 seconds
plt.close(fig)          # close figure
