"""
3D visualization for pylon racing: pylons and UAV.
Uses matplotlib (mpl_toolkits.mplot3d). Run with --viz from test_pylon_gym.py.
"""
import time
import numpy as np
from no_ros2.environments.pylon_course import (
    DEFAULT_PYLONS,
    DEFAULT_GATES,
    PYLON_HEIGHT_M,
    PYLON_RADIUS_M,
    PYLON_NAMES,
    FINISH_GATE_INDEX,
    BOUNDS_RECT,
)


# Beige/off-white for UAV body
UAV_BEIGE = "#e8dcc4"
UAV_BEIGE_DARK = "#c4b896"
# Axis line colors (body frame: forward, right, up)
UAV_AXIS_FWD = "#c45c5c"
UAV_AXIS_RIGHT = "#5c9e5c"
UAV_AXIS_UP = "#5c5cc4"


def _body_axes_from_velocity(vx, vy, vz):
    """Return unit vectors (forward, right, up) in world frame for a fixed-wing with given velocity."""
    speed = np.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
    forward = np.array([vx / speed, vy / speed, vz / speed], dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    rn = np.sqrt(np.dot(right, right)) + 1e-9
    right /= rn
    up = np.cross(right, forward)
    up /= (np.sqrt(np.dot(up, up)) + 1e-9)
    return forward, right, up


def _uav_line_vertices(x, y, z, forward, right, up, scale=0.6):
    """Return list of (xs, ys, zs) for fixed-wing body + axis lines. scale = half-size in m."""
    c = np.array([x, y, z])
    # Fuselage: tail to nose along forward
    fuselage = np.array([
        c - scale * 0.8 * forward,
        c + scale * 0.8 * forward,
    ]).T  # (3, 2)
    # Wings: left tip to right tip along right
    wings = np.array([
        c - scale * 1.0 * right,
        c + scale * 1.0 * right,
    ]).T  # (3, 2)
    # Vertical tail: center to top along up
    tail_up = np.array([
        c,
        c + scale * 0.5 * up,
    ]).T  # (3, 2)
    # Nose wedge (two short lines from nose for visibility)
    nose = c + scale * 0.8 * forward
    wing_nose_l = nose - scale * 0.25 * right
    wing_nose_r = nose + scale * 0.25 * right
    nose_line = np.array([wing_nose_l, wing_nose_r]).T  # (3, 2)
    # Axis lines (from center, shorter)
    ax_scale = scale * 0.4
    fwd_axis = np.array([c, c + ax_scale * forward]).T
    right_axis = np.array([c, c + ax_scale * right]).T
    up_axis = np.array([c, c + ax_scale * up]).T
    return {
        "fuselage": fuselage,
        "wings": wings,
        "tail_up": tail_up,
        "nose_line": nose_line,
        "axis_forward": fwd_axis,
        "axis_right": right_axis,
        "axis_up": up_axis,
    }


def _cylinder_mesh(x0, y0, z0, z1, r, n=8):
    """Return (xx, yy, zz) for a cylinder from (x0,y0,z0) to (x0,y0,z1) with radius r."""
    theta = np.linspace(0, 2 * np.pi, n)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    z_bot = np.full_like(theta, z0)
    z_top = np.full_like(theta, z1)
    return x, y, z_bot, z_top


class PylonRacingViz3D:
    """Real-time 3D plot of pylons and UAV. Call update(obs, reward=..., step_count=...) each step."""

    def __init__(
        self,
        pylons=None,
        gates=None,
        trail_len=60,
        target_alt_m=7.0,
        follow_camera=True,
        camera_margin=20.0,
        draw_every_n_steps=5,
        velocity_arrow_length=2.0,
        show_state_vector=False,
    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        self._plt = plt
        self.pylons = pylons if pylons is not None else DEFAULT_PYLONS.copy()
        self.gates = gates if gates is not None else list(DEFAULT_GATES)
        self.trail_len = trail_len
        self.target_alt_m = target_alt_m
        self.follow_camera = follow_camera
        self.camera_margin = camera_margin
        self.draw_every_n_steps = draw_every_n_steps
        self.velocity_arrow_length = velocity_arrow_length
        self._show_state_vector = show_state_vector
        self._draw_counter = 0
        self._trail_x = []
        self._trail_y = []
        self._trail_z = []
        self._last_reward = 0.0
        self._last_step_count = 0
        self._last_speed = 0.0
        self._last_azim = -65  # fallback when UAV has no velocity
        self._start_time = None  # set on first update; reset in clear_trail()
        self._last_laps = 0

        self._fig = plt.figure(figsize=(6, 4.5), facecolor="#1a1a1a")
        self._ax = self._fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")
        self._fig.canvas.manager.set_window_title("Pylon Racing (no ROS2)")

        # Dark theme: axis panes
        self._ax.xaxis.pane.fill = False
        self._ax.yaxis.pane.fill = False
        self._ax.zaxis.pane.fill = False
        self._ax.xaxis.pane.set_edgecolor("#333")
        self._ax.yaxis.pane.set_edgecolor("#333")
        self._ax.zaxis.pane.set_edgecolor("#333")
        self._ax.tick_params(colors="#aaa")
        self._ax.xaxis.label.set_color("#aaa")
        self._ax.yaxis.label.set_color("#aaa")
        self._ax.zaxis.label.set_color("#aaa")
        self._ax.grid(False)

        g = BOUNDS_RECT
        # Bounds rectangle at z=0
        bx = [g["min_x"], g["max_x"], g["max_x"], g["min_x"], g["min_x"]]
        by = [g["min_y"], g["min_y"], g["max_y"], g["max_y"], g["min_y"]]
        self._ax.plot(bx, by, [0] * 5, color="#555", linestyle="-", linewidth=1.5, zorder=1)

        # Pylon cylinders and labels (bright for dark bg)
        z_mid = np.mean([self.pylons[:, 2].min(), self.pylons[:, 2].max()]) + PYLON_HEIGHT_M * 0.5
        for i in range(len(self.pylons)):
            px, py, z0 = self.pylons[i, 0], self.pylons[i, 1], self.pylons[i, 2]
            z1 = z0 + PYLON_HEIGHT_M
            xc, yc, zb, zt = _cylinder_mesh(px, py, z0, z1, PYLON_RADIUS_M)
            self._ax.plot(xc, yc, zb, color="#ffa726", linewidth=2, zorder=5)
            self._ax.plot(xc, yc, zt, color="#ffa726", linewidth=2, zorder=5)
            for j in range(len(xc)):
                self._ax.plot(
                    [xc[j], xc[j]], [yc[j], yc[j]], [zb[j], zt[j]],
                    color="#ffa726", linewidth=1.5, alpha=0.9, zorder=5
                )
            name = PYLON_NAMES[i] if i < len(PYLON_NAMES) else f"P{i+1}"
            self._ax.text(px, py, z1 + 0.5, name, fontsize=8, color="#ffd54f", zorder=6)

        # Gate lines: finish gate bright green, others muted
        for gi, (i, j) in enumerate(self.gates):
            p1, p2 = self.pylons[i], self.pylons[j]
            color = "#66bb6a" if gi == FINISH_GATE_INDEX else "#555"
            lw = 2.0 if gi == FINISH_GATE_INDEX else 1.0
            self._ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [z_mid, z_mid],
                color=color, linestyle="--", alpha=0.8, linewidth=lw, zorder=2
            )

        # UAV: fixed-wing body + axis lines (drawn each update in update())
        self._uav_artists = []
        self._trail_line, = self._ax.plot([], [], [], color="#4dd0e1", alpha=0.7, linewidth=2, zorder=3)

        # Default view (chase: slightly above, behind)
        self._ax.elev = 18
        self._ax.azim = self._last_azim

        # Axis limits (will be updated if follow_camera)
        margin = self.camera_margin
        xs, ys = self.pylons[:, 0], self.pylons[:, 1]
        self._ax.set_xlim(xs.min() - margin, xs.max() + margin)
        self._ax.set_ylim(ys.min() - margin, ys.max() + margin)
        self._ax.set_zlim(0, PYLON_HEIGHT_M + 5)
        if show_state_vector:
            self._ax.set_xlabel("X (m)")
            self._ax.set_ylabel("Y (m)")
            self._ax.set_zlabel("Z (m)")
        else:
            self._ax.set_xlabel("")
            self._ax.set_ylabel("")
            self._ax.set_zlabel("")

            # No axis lines or tick numbers on any axis (cleaner view)
            self._ax.set_xticklabels([])
            self._ax.set_yticklabels([])
            self._ax.set_zticklabels([])
            self._ax.xaxis.line.set_visible(False)
            self._ax.yaxis.line.set_visible(False)
            self._ax.zaxis.line.set_visible(False)
            self._ax.xaxis.pane.set_edgecolor("none")
            self._ax.yaxis.pane.set_edgecolor("none")
            self._ax.zaxis.pane.set_edgecolor("none")

        # HUD text (updated in update) — light on dark
        self._hud_text = self._fig.text(0.02, 0.02, "", fontsize=8, family="monospace",
                                       color="#ddd", verticalalignment="bottom", wrap=False)
        # State vector panel (right side), only when show_state_vector=True
        self._state_text = None
        if show_state_vector:
            self._state_text = self._fig.text(0.98, 0.98, "", fontsize=7, family="monospace",
                                             color="#bbb", verticalalignment="top", horizontalalignment="right", wrap=False)

        self._plt.ion()
        self._plt.show(block=False)

    def update(self, obs, reward=None, step_count=None, laps=None):
        """Update plot with current observation. Expects 15D ROS2 layout [x,y,z, roll,pitch,yaw, vx,vy,vz, ...]; supports 6D [x,y,z,vx,vy,vz] for legacy."""
        if self._start_time is None:
            self._start_time = time.time()
        x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
        if len(obs) >= 9:
            vx, vy, vz = float(obs[6]), float(obs[7]), float(obs[8])
        else:
            vx = float(obs[3]) if len(obs) > 3 else 0.0
            vy = float(obs[4]) if len(obs) > 4 else 0.0
            vz = float(obs[5]) if len(obs) > 5 else 0.0
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        if reward is not None:
            self._last_reward = reward
        if step_count is not None:
            self._last_step_count = step_count
        if laps is not None:
            self._last_laps = int(laps)
        self._last_speed = speed

        self._trail_x.append(x)
        self._trail_y.append(y)
        self._trail_z.append(z)
        if len(self._trail_x) > self.trail_len:
            self._trail_x.pop(0)
            self._trail_y.pop(0)
            self._trail_z.pop(0)

        # Fixed-wing UAV body + axis lines (beige body, colored axis lines)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9
        if speed < 0.1:
            forward = np.array([1.0, 0.0, 0.0])
            right = np.array([0.0, 1.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])
        else:
            forward, right, up = _body_axes_from_velocity(vx, vy, vz)
        verts = _uav_line_vertices(x, y, z, forward, right, up, scale=0.6)
        for a in self._uav_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._uav_artists = []
        # Body (beige)
        for key in ("fuselage", "wings", "tail_up", "nose_line"):
            xs, ys, zs = verts[key][0], verts[key][1], verts[key][2]
            ln, = self._ax.plot(xs, ys, zs, color=UAV_BEIGE, linewidth=2.5, zorder=10)
            self._uav_artists.append(ln)
        # Axis lines (forward, right, up)
        ln_f, = self._ax.plot(verts["axis_forward"][0], verts["axis_forward"][1], verts["axis_forward"][2],
                              color=UAV_AXIS_FWD, linewidth=1.5, zorder=11)
        ln_r, = self._ax.plot(verts["axis_right"][0], verts["axis_right"][1], verts["axis_right"][2],
                              color=UAV_AXIS_RIGHT, linewidth=1.5, zorder=11)
        ln_u, = self._ax.plot(verts["axis_up"][0], verts["axis_up"][1], verts["axis_up"][2],
                              color=UAV_AXIS_UP, linewidth=1.5, zorder=11)
        self._uav_artists.extend([ln_f, ln_r, ln_u])

        # Trail line
        if self._trail_x:
            self._trail_line.set_data_3d(self._trail_x, self._trail_y, self._trail_z)
        else:
            self._trail_line.set_data_3d([], [], [])

        # Chase camera:
        if self.follow_camera:
            m = self.camera_margin
            self._ax.set_xlim(x - m, x + m)
            self._ax.set_ylim(y - m, y + m)
            z_min = max(0, z - 5)
            z_max = max(PYLON_HEIGHT_M + 5, z + 10)
            self._ax.set_zlim(z_min, z_max)
            # Azimuth: camera behind UAV = opposite to horizontal velocity
            if speed > 0.3:
                # atan2(vy,vx) = direction UAV is going; add 180 to be behind
                heading_deg = np.degrees(np.arctan2(vy, vx))
                self._last_azim = heading_deg + 180
            self._ax.azim = self._last_azim
            self._ax.elev = 18  # slightly above for chase view

        # HUD: time (M:SS), laps, alt, speed, reward, step
        elapsed = time.time() - self._start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_str = f"{mins}:{secs:02d}"
        self._hud_text.set_text(
            f"Time: {time_str}  |  Laps: {self._last_laps}  |  Alt: {z:.1f} m  |  Speed: {speed:.1f} m/s  |  Reward: {self._last_reward:.0f}  |  Step: {self._last_step_count}"
        )

        # State vector (right side): all 15 components when available (only if enabled)
        if self._state_text is not None:
            STATE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "v", "gamma", "vdot", "p", "q", "r")
            STATE_UNITS = ("m", "m", "m", "deg", "deg", "deg", "m/s", "m/s", "m/s", "m/s", "deg", "m/s²", "rad/s", "rad/s", "rad/s")
            lines = ["State vector:"]
            n = min(len(obs), 15)
            for i in range(n):
                val = float(obs[i])
                name = STATE_NAMES[i]
                unit = STATE_UNITS[i] if i < len(STATE_UNITS) else ""
                if unit in ("deg",) and name in ("roll", "pitch", "yaw", "gamma"):
                    val_show = np.degrees(val)
                    lines.append(f"  {name}: {val_show:7.2f} {unit}")
                else:
                    lines.append(f"  {name}: {val:7.3f} {unit}")
            if n < 15:
                lines.append(f"  (only {n} dims)")
            self._state_text.set_text("\n".join(lines))

        # Throttle redraws
        self._draw_counter += 1
        if self._draw_counter >= self.draw_every_n_steps:
            self._draw_counter = 0
            self._fig.canvas.draw_idle()
            self._plt.pause(0.001)

    def clear_trail(self):
        """Clear the UAV path trail and reset session timer (e.g. after env reset)."""
        self._trail_x.clear()
        self._trail_y.clear()
        self._trail_z.clear()
        self._trail_line.set_data_3d([], [], [])
        self._start_time = time.time()  # reset elapsed time for new run
        self._last_laps = 0

    def close(self):
        self._plt.ioff()
        self._plt.close(self._fig)
