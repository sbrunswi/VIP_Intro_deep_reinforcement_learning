"""
Pylon course layouts for mock env and 3D visualization.
Select a course by name with get_course(name).
Available courses: "purt", "sample"
"""
import numpy as np

PYLON_HEIGHT_M = 7.0
PYLON_RADIUS_M = 0.25
PYLON_MID_HEIGHT_M = PYLON_HEIGHT_M * 0.5
FINISH_GATE_INDEX = 0

# ---------------------------------------------------------------------------
# Course definitions
# ---------------------------------------------------------------------------

COURSES = {
    "purt": {
        "pylons": np.array([
            [ 3.984,  -9.587, 0.0],  # P1
            [-9.1245, -0.0397, 0.0], # P2
            [ 4.055,   8.704, 0.0],  # P3
            [19.872,  -0.236, 0.0],  # P4
        ], dtype=np.float64),
        "gates": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "pylon_names": ["P1", "P2", "P3", "P4"],
        "bounds_rect": {"min_x": -13.0, "max_x": 24.0, "min_y": -15.0, "max_y": 13.0},
        "pylon_height_m": 7.0,
        "pylon_radius_m": 0.25,
    },
    "sample": {
        "pylons": np.array([
            [-20.0,   5.0, 0.0],  # P1
            [-20.0, -20.0, 0.0],  # P2
            [-25.0, -45.0, 0.0],  # P3
            [ 20.0, -25.0, 0.0],  # P4
            [ 35.0,   0.0, 0.0],  # P5
            [ 10.0,  -5.0, 0.0],  # P6
        ], dtype=np.float64),
        "gates": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        "pylon_names": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "bounds_rect": {"min_x": -60.0, "max_x": 50.0, "min_y": -60.0, "max_y": 30.0},
        "pylon_height_m": 7.0,
        "pylon_radius_m": 0.25,
    },
}

DEFAULT_COURSE = "sample"


def get_course(name: str) -> dict:
    """Return course data dict for the given course name. Raises ValueError for unknown names."""
    if name not in COURSES:
        raise ValueError(f"Unknown course '{name}'. Available: {list(COURSES.keys())}")
    return COURSES[name]


# ---------------------------------------------------------------------------
# Module-level defaults (point at DEFAULT_COURSE for backward compatibility)
# ---------------------------------------------------------------------------

def _default(key):
    return COURSES[DEFAULT_COURSE][key]

DEFAULT_PYLONS = _default("pylons")
DEFAULT_GATES  = _default("gates")
PYLON_NAMES    = _default("pylon_names")
BOUNDS_RECT    = _default("bounds_rect")


def get_pylon_tops(pylons=None, height=PYLON_HEIGHT_M):
    """Return (N, 3) array of pylon top positions (x, y, z_top)."""
    if pylons is None:
        pylons = DEFAULT_PYLONS
    tops = np.array(pylons, dtype=np.float64)
    tops[:, 2] += height
    return tops
