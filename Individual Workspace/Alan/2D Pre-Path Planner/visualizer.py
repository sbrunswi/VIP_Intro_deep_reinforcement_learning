"""
visualizer.py
-------------
Light-theme matplotlib dashboard matching the visual style of
Individual Workspace/Alan/plot_pylons.py:
- white background
- red pylons (filled circles with labels)
- gray dashed bounds rectangle
- black dashed gates with gray labels
- blue squares for heuristic waypoints
- green squares for optimized waypoints
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


PYLON_RADIUS = 0.25
# TRANSITION_RADIUS = 2.0  # metres — probabilistic transition region


def _draw_course(ax, pylons_xy, gates, waypoints, trajectory, bounds,
                 result, wp_color, wp_marker, path_color, title):
    """Draw one 'before' or 'after' panel in the light theme."""
    ax.set_facecolor("white")

    # Bounds rectangle
    if bounds:
        rect = Rectangle(
            (bounds["min_x"], bounds["min_y"]),
            bounds["max_x"] - bounds["min_x"],
            bounds["max_y"] - bounds["min_y"],
            linewidth=1.5, edgecolor="gray", facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    # Gates (dashed lines between pylon pairs with labels)
    for gi, (i1, i2) in enumerate(gates):
        p1 = pylons_xy[i1]
        p2 = pylons_xy[i2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                "k--", linewidth=1, alpha=0.5)
        mx = (p1[0] + p2[0]) / 2.0
        my = (p1[1] + p2[1]) / 2.0
        ax.text(mx, my, f"G{gi+1}", fontsize=9, color="gray",
                ha="center", va="bottom")

    # Trajectory (simulated flight path)
    if trajectory and len(trajectory) > 1:
        traj = np.asarray(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color=path_color,
                linewidth=1.8, alpha=0.75, zorder=3)

    # Waypoint path (dashed line connecting waypoints)
    if waypoints and len(waypoints) > 0:
        wps = np.asarray(waypoints)
        wps_closed = np.vstack([wps, wps[0:1]])
        ax.plot(wps_closed[:, 0], wps_closed[:, 1],
                color=wp_color, linewidth=1.0, linestyle="--",
                alpha=0.6, zorder=2)
        # All waypoints with transition circles
        # for i, wp in enumerate(waypoints):
        #     # Transparent transition circle
        #     tc = Circle(wp, TRANSITION_RADIUS, color=wp_color,
        #                 alpha=0.12, zorder=3)
        #     ax.add_patch(tc)
        #     # Waypoint marker
        #     ax.plot(wp[0], wp[1], wp_marker, color=wp_color,
        #             markersize=7, markeredgecolor="black",
        #             markeredgewidth=0.5, zorder=6)
        #     ax.text(wp[0] + 0.4, wp[1] + 0.4, f"W{i}",
        #             fontsize=8, color=wp_color, ha="left",
        #             va="bottom", fontweight="bold")
                # All waypoints
        for i, wp in enumerate(waypoints):
            # Waypoint marker
            ax.plot(wp[0], wp[1], wp_marker, color=wp_color)

    # Pylons (red filled circles with labels)
    for i, p in enumerate(pylons_xy):
        circle = Circle((p[0], p[1]), PYLON_RADIUS,
                            color="red", zorder=5)
        ax.add_patch(circle)
        ax.text(p[0], p[1] + 0.6, f"P{i+1}", fontsize=10,
                ha="center", va="bottom", fontweight="bold", color="black")

    # Stats box
    if result:
        status = "completed" if result["completed"] else "incomplete"
        stats = (f"Time: {result['total_time']:.1f}s\n"
                 f"Dist: {result['total_distance']:.1f}m\n"
                 f"Status: {status}")
        props = dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="gray", alpha=0.9)
        ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", color="black",
                bbox=props, zorder=10)

    # Axes
    ax.set_title(title, fontsize=11, pad=8, color="black")
    ax.set_xlabel("x (m)", color="black")
    ax.set_ylabel("y (m)", color="black")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("gray")

    if bounds:
        pad = 2.0
        ax.set_xlim(bounds["min_x"] - pad, bounds["max_x"] + pad)
        ax.set_ylim(bounds["min_y"] - pad, bounds["max_y"] + pad)


def plot_dashboard(pylons_xy, gates, base_wps, best_wps,
                   result_before, result_after, rewards, bounds=None):
    """
    Produce a 2x2 gridspec dashboard:
      top-left   : heuristic (before) course panel
      top-right  : optimized (after) course panel
      bottom-row : learning curve spanning both columns
    """
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle("Pre-Path Planner Dashboard",
                 color="black", fontsize=15, fontweight="bold", y=0.97)

    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3],
                          hspace=0.28, wspace=0.22,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)

    pylons_arr = np.asarray(pylons_xy)

    # Top-left: before
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_course(ax1, pylons_arr, gates, base_wps,
                 result_before["trajectory"], bounds, result_before,
                 wp_color="blue", wp_marker="s", path_color="blue",
                 title="Before  (Heuristic)")

    # Top-right: after
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_course(ax2, pylons_arr, gates, best_wps,
                 result_after["trajectory"], bounds, result_after,
                 wp_color="green", wp_marker="s", path_color="green",
                 title="After  (Q-Learning)")

    # Bottom: learning curve
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("white")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.tick_params(colors="black", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color("gray")

    episodes = np.arange(1, len(rewards) + 1)
    r_arr = np.asarray(rewards)
    ax3.fill_between(episodes, rewards, alpha=0.15, color="#6c5ce7")
    ax3.plot(episodes, rewards, color="#6c5ce7", alpha=0.4, linewidth=0.8)

    window = min(30, max(len(rewards) // 5, 2))
    if window > 1 and len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax3.plot(np.arange(window, len(rewards) + 1), avg,
                 color="#2d3436", linewidth=2.2,
                 label=f"Rolling avg ({window})")

    # Zoom vertical axis into the meaningful region
    r_lo = np.percentile(r_arr, 10)
    r_hi = r_arr.max()
    r_pad = max((r_hi - r_lo) * 0.1, 0.3)
    ax3.set_ylim(r_lo - r_pad, r_hi + r_pad)

    ax3.set_xlabel("Episode", fontsize=9, color="black")
    ax3.set_ylabel("Best Reward", fontsize=9, color="black")
    ax3.set_title("Optimization Progress", fontsize=11, pad=8, color="black")
    ax3.legend(loc="lower right", fontsize=8)

    # Improvement annotation
    if result_before["completed"] and result_after["completed"]:
        dt = result_before["total_time"] - result_after["total_time"]
        dd = result_before["total_distance"] - result_after["total_distance"]
        note = f"Time: {dt:+.2f}s   Dist: {dd:+.1f}m"
        ax3.annotate(note, xy=(0.98, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=9, color="green",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="green", alpha=0.9))

    return fig
