"""
Default pylon course layout for mock env and 3D visualization.
Matches purt_course.yaml (x, y in meters; z=0 base; height 7 m).
"""
import numpy as np

PYLON_HEIGHT_M = 7.0
PYLON_RADIUS_M = 0.25
# Vertical midpoint of pylons (base z=0 to top z=PYLON_HEIGHT_M)
PYLON_MID_HEIGHT_M = PYLON_HEIGHT_M * 0.5

# Pylons (x, y, z_base); z_base is typically 0
DEFAULT_PYLONS = np.array([
    [3.984, -9.587, 0.0],   # P1
    [-9.1245, -0.0397, 0.0],  # P2
    [4.055, 8.704, 0.0],    # P3
    [19.872, -0.236, 0.0], # P4
], dtype=np.float64)

# Gates: (p1_idx, p2_idx) for line segment between pylons
# Gate 0 (G1) is the finish line
DEFAULT_GATES = [(0, 1), (1, 2), (2, 3), (3, 0)]
FINISH_GATE_INDEX = 0

# Pylon display names for labels
PYLON_NAMES = ["P1", "P2", "P3", "P4"]

# Axis-aligned bounds (meters), from purt_course.yaml (min/max x and y)
BOUNDS_RECT = {"min_x": -13.0, "max_x": 24.0, "min_y": -15.0, "max_y": 13.0}


def get_pylon_tops(pylons=None, height=PYLON_HEIGHT_M):
    """Return (N, 3) array of pylon top positions (x, y, z_top)."""
    if pylons is None:
        pylons = DEFAULT_PYLONS
    tops = np.array(pylons, dtype=np.float64)
    tops[:, 2] += height
    return tops
