import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add project root so we can import the shared course data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _root)

from no_ros2.environments.pylon_course import get_course
from heuristic import default_directions


def plot_course(ax, course_name):
    """Plot a single course on the given axes."""
    course = get_course(course_name)
    pylons_xy = course["pylons"][:, :2]
    bounds = course["bounds_rect"]
    n = len(pylons_xy)
    pylon_names = course["pylon_names"]
    gate_pairs = course["gates"]
    R = course.get("pylon_radius_m", 0.25)

    # Draw bounds rectangle
    rect = plt.Rectangle(
        (bounds["min_x"], bounds["min_y"]),
        bounds["max_x"] - bounds["min_x"],
        bounds["max_y"] - bounds["min_y"],
        linewidth=1.5, edgecolor="gray", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

    # Draw gates
    for k, (i1, i2) in enumerate(gate_pairs):
        p1, p2 = pylons_xy[i1], pylons_xy[i2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                "k--", linewidth=1, alpha=0.5)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, f"G{k+1}", fontsize=9, color="gray",
                ha="center", va="bottom")

    # Draw pylons
    for j in range(n):
        circle = plt.Circle(pylons_xy[j], R, color="red", zorder=5)
        ax.add_patch(circle)
        ax.text(pylons_xy[j][0], pylons_xy[j][1] + 0.6, pylon_names[j],
                fontsize=10, ha="center", va="bottom", fontweight="bold")

    # Annotate IN/OUT on each gate
    directions = default_directions(n)
    for i in range(n):
        p1, p2 = pylons_xy[i], pylons_xy[(i + 1) % n]
        mx, my = (p1 + p2) / 2
        label = "IN" if directions[i] > 0 else "OUT"
        ax.text(mx, my - 0.8, label, fontsize=8, color="darkgreen",
                ha="center", va="top", fontweight="bold")

    ax.set_xlim(bounds["min_x"] - 1, bounds["max_x"] + 1)
    ax.set_ylim(bounds["min_y"] - 1, bounds["max_y"] + 1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{course_name} course ({n} pylons)")


# Plot both courses side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("white")

for ax, name in zip(axes, ["purt", "sample"]):
    ax.set_facecolor("white")
    plot_course(ax, name)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "pylon_course.png"), dpi=150)
plt.show()
