"""
waypoint_env.py
---------------
Barebone 2D point-mass simulator for waypoint path evaluation.

The agent flies at constant speed with rate-limited heading changes.
Sharp turns cost time, creating optimization pressure for good waypoint placement.
"""

import numpy as np


class PointMassEnv2D:
    def __init__(self, pylons_xy, bounds, speed=7.0, max_turn_rate=2.0,
                 dt=0.05, max_steps=2000, capture_radius=1.5):
        self.pylons_xy = np.array(pylons_xy)
        self.bounds = bounds
        self.speed = speed
        self.max_turn_rate = max_turn_rate
        self.dt = dt
        self.max_steps = max_steps
        self.capture_radius = capture_radius

    def simulate(self, waypoints):
        """
        Run the point-mass through the waypoint sequence.

        Parameters
        ----------
        waypoints : list of (x, y)
            Ordered waypoints to visit (includes pylon positions as checkpoints).

        Returns
        -------
        dict with keys:
            total_time      : float  — seconds to complete (or max time if incomplete)
            total_distance  : float  — path length in metres
            smoothness      : float  — sum of |heading_change| (lower = smoother)
            completed       : bool   — True if all waypoints visited
            trajectory      : list of (x, y)
        """
        wps = [np.array(w, dtype=np.float64) for w in waypoints]
        n_wps = len(wps)
        if n_wps < 2:
            return self._empty_result()

        # Start at first waypoint, heading toward second
        x, y = wps[0]
        dx = wps[1][0] - x
        dy = wps[1][1] - y
        heading = np.arctan2(dy, dx)

        wp_idx = 1  # targeting this waypoint next
        trajectory = [(x, y)]
        total_dist = 0.0
        smoothness = 0.0

        for step in range(self.max_steps):
            # Desired heading toward current target
            target = wps[wp_idx]
            desired = np.arctan2(target[1] - y, target[0] - x)

            # Rate-limited turn
            err = self._angle_wrap(desired - heading)
            max_delta = self.max_turn_rate * self.dt
            delta = np.clip(err, -max_delta, max_delta)
            heading += delta
            smoothness += abs(delta)

            # Advance position
            vx = self.speed * np.cos(heading)
            vy = self.speed * np.sin(heading)
            x += vx * self.dt
            y += vy * self.dt
            total_dist += self.speed * self.dt
            trajectory.append((x, y))

            # Check waypoint capture
            dist_to_wp = np.hypot(target[0] - x, target[1] - y)
            if dist_to_wp < self.capture_radius:
                wp_idx += 1
                if wp_idx >= n_wps:
                    # All waypoints visited
                    total_time = (step + 1) * self.dt
                    return {
                        "total_time": total_time,
                        "total_distance": total_dist,
                        "smoothness": smoothness,
                        "completed": True,
                        "trajectory": trajectory,
                    }

            # Bounds check
            b = self.bounds
            if x < b["min_x"] or x > b["max_x"] or y < b["min_y"] or y > b["max_y"]:
                total_time = (step + 1) * self.dt
                return {
                    "total_time": total_time,
                    "total_distance": total_dist,
                    "smoothness": smoothness,
                    "completed": False,
                    "trajectory": trajectory,
                }

        # Ran out of steps
        return {
            "total_time": self.max_steps * self.dt,
            "total_distance": total_dist,
            "smoothness": smoothness,
            "completed": False,
            "trajectory": trajectory,
        }

    @staticmethod
    def _angle_wrap(a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _empty_result(self):
        return {
            "total_time": 0.0,
            "total_distance": 0.0,
            "smoothness": 0.0,
            "completed": False,
            "trajectory": [],
        }
