#!/usr/bin/env python3
import os
import math
import time
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime
from pathlib import Path as FsPath
import shutil

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
import csv
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32, Int32, String
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

PKG_NAME = "pylon_2026_world"

Vec2 = Tuple[float, float]

EPS = 1e-9


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _seg_intersect_t(a, b, c, d):
    """Return (hit, t_ab) where t_ab in [0,1] along AB if segments AB and CD intersect."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = _cross(r, s)
    qp = (c[0] - a[0], c[1] - a[1])

    if abs(denom) < EPS:
        # Parallel: check colinear overlap
        if abs(_cross(qp, r)) < EPS:
            r2 = r[0] * r[0] + r[1] * r[1]
            if r2 < EPS:
                return (False, None)
            t0 = ((c[0] - a[0]) * r[0] + (c[1] - a[1]) * r[1]) / r2
            t1 = ((d[0] - a[0]) * r[0] + (d[1] - a[1]) * r[1]) / r2
            t_min = max(0.0, min(t0, t1))
            t_max = min(1.0, max(t0, t1))
            if t_min <= t_max + EPS:
                return (True, max(0.0, min(1.0, 0.5 * (t_min + t_max))))
        return (False, None)

    t = _cross(qp, s) / denom
    u = _cross(qp, r) / denom
    if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
        return (True, max(0.0, min(1.0, t)))
    return (False, None)


# ---------------- Path resolution ----------------
def resolve_params_path(path_param: str, pkg_name: str = PKG_NAME) -> str:
    """
    Resolve a params YAML path:
      - absolute path: use as-is (with ~ expansion)
      - relative path: try <share/pkg>/<path> then <share/pkg>/config/<path>
    """
    p = os.path.expanduser(path_param or "")
    if os.path.isabs(p) and os.path.exists(p):
        return p

    pkg_share = get_package_share_directory(pkg_name)
    cand = os.path.join(pkg_share, p)
    cand2 = os.path.join(pkg_share, "config", os.path.basename(p) if p else "")
    tried = []
    for c in [cand, cand2]:
        if c:
            tried.append(c)
            if os.path.exists(c):
                return c

    raise FileNotFoundError(
        f"Could not resolve params_file '{path_param}'. Tried:\n  " + "\n  ".join(tried)
    )


# --- resolve best-score path (1-line CSV) ---
def _resolve_best_log_path(user_param: Optional[str], default_dir: str) -> str:
    """
    If user_param is:
      - absolute DIR  -> use <dir>/best_score.csv
      - absolute FILE -> use it directly
      - relative name -> put it inside default_dir
      - empty         -> use <default_dir>/best_score.csv
    """
    base = FsPath(default_dir)
    base.mkdir(parents=True, exist_ok=True)

    if user_param:
        p = FsPath(user_param).expanduser()
        if p.is_absolute():
            if p.is_dir():
                p.mkdir(parents=True, exist_ok=True)
                return str(p / "best_score.csv")
            else:
                (p.parent or FsPath(".")).mkdir(parents=True, exist_ok=True)
                return str(p)
        # relative name → inside default_dir
        return str(base / p.name)

    return str(base / "best_score.csv")

# ---------------- path to container log ------------
def _read_best_score_from_csv(path: str) -> Optional[float]:
    """Read lap_score (3rd column) from a best_score.csv (header on first line)."""
    try:
        with open(path, "r") as f:
            rows = [r for r in csv.reader(f) if r]  # keep non-empty
        if len(rows) <= 1:
            return None  # header only / empty
        last = rows[-1]
        return float(last[2]) if len(last) >= 3 else None
    except Exception:
        return None

def _maybe_update_submit_best(src_path: str,
                              env_var: str = "JUPYTERHUB_USER",
                              submit_root: str = "/submit") -> tuple[Optional[str], Optional[bool]]:
    """
    Ensure /submit/<email>/logs/best_score.csv contains the *better* score.
    Returns (dest_path, replaced?) where replaced? is:
        True  -> copied new file,
        False -> kept existing (it was better or equal),
        None  -> not attempted (no env or permisssion error).
    """
    email = os.environ.get(env_var, "").strip()
    if not email:
        return (None, None)

    dst_dir = os.path.join(submit_root, email, "logs")
    dst = os.path.join(dst_dir, "best_score.csv")
    os.makedirs(dst_dir, exist_ok=True)

    new_score = _read_best_score_from_csv(src_path)
    if new_score is None:
        # If new file unreadable, just copy to be safe
        shutil.copyfile(src_path, dst)
        return (dst, True)

    if os.path.exists(dst):
        old_score = _read_best_score_from_csv(dst)
        if old_score is not None and old_score <= new_score:
            # Existing submit score is already better or equal — keep it
            return (dst, False)

    shutil.copyfile(src_path, dst)
    return (dst, True)



# ---------------- Small vec helpers ----------------
def v2(x: float, y: float) -> Vec2:
    return (x, y)


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def mul(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def perp(a: Vec2) -> Vec2:
    return (-a[1], a[0])  # +90° in XY plane


# ---------------- Data ----------------
@dataclass
class Pylon:
    x: float
    y: float
    z: float
    r: float
    name: str


@dataclass
class Gate:
    p1: int
    p2: int
    name: str


PKG_NAME = "pylon_2026_world"


def _resolve_lap_log_path(user_param: str | None) -> str:
    """
    Prefer saving directly in src/pylon_2026_world/logs.
    Fallback to share/<pkg>/logs, then ~/.ros if needed.
    - ABSOLUTE dir -> create timestamped CSV inside it
    - ABSOLUTE file -> use it directly
    - RELATIVE/empty -> 'name' inside preferred logs dir
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"pylon_laps_{stamp}.csv"

    # Explicit absolute path overrides
    if user_param:
        p = FsPath(user_param).expanduser()
        if p.is_absolute():
            if p.is_dir():
                p.mkdir(parents=True, exist_ok=True)
                return str(p / default_name)
            else:
                (p.parent or FsPath(".")).mkdir(parents=True, exist_ok=True)
                return str(p)
        name = p.name  # relative filename
    else:
        name = default_name

    # 1) Source tree logs dir next to the package root: src/<pkg>/logs
    try:
        # course_node.py -> .../src/pylon_2026_world/pylon_2026_world/course_node.py
        # parents[1] -> .../src/pylon_2026_world
        src_logs = FsPath(__file__).resolve().parents[1] / "logs"
        src_logs.mkdir(parents=True, exist_ok=True)
        return str(src_logs / name)
    except Exception:
        pass  # not writable or not a dev workspace

    # 2) Installed share/<pkg>/logs
    try:
        share_logs = FsPath(get_package_share_directory(PKG_NAME)) / "logs"
        share_logs.mkdir(parents=True, exist_ok=True)
        return str(share_logs / name)
    except Exception:
        pass

    # 3) Final fallback
    home_logs = FsPath("~/.ros").expanduser()
    home_logs.mkdir(parents=True, exist_ok=True)
    return str(home_logs / name)

def _copy_submit_lap_log(src_path: str,
                         env_var: str = "JUPYTERHUB_USER",
                         submit_root: str = "/submit") -> str | None:
    """
    Copy the current lap log (e.g., pylon_laps_YYYYMMDD_HHMMSS.csv)
    to /submit/<email>/logs/ with the SAME basename.
    """
    email = os.environ.get(env_var, "").strip()
    if not email:
        return None
    dst_dir = os.path.join(submit_root, email, "logs")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copyfile(src_path, dst)  # overwrite each time
    return dst


# ---------------- Node ----------------
class CourseNode(Node):
    def __init__(self):
        super().__init__("pylon_course")

        # Parameters
        self.declare_parameter("params_file", "config/purt_course.yaml")
        self.declare_parameter("pose_topic", "cub1/pose")  # change if needed

        # Resolve YAML relative to package share/
        raw_param = self.get_parameter("params_file").get_parameter_value().string_value
        params_file = resolve_params_path(raw_param, PKG_NAME)

        with open(params_file, "r") as f:
            p = yaml.safe_load(f)

        self.frame_id = p.get("frame_id", "map")
        self.gate_margin = float(p.get("gate_margin_m", 0.5))
        self.missed_gate_penalty_s = float(p.get("missed_gate_penalty_s", 3.0))
        self.start_on_finish_cross = bool(p.get("start_on_finish_cross", True))
        self.centerline_closed = bool(p.get("centerline_closed", True))
        self.pylons: List[Pylon] = [Pylon(**d) for d in p["pylons"]]
        self.gates: List[Gate] = [Gate(**g) for g in p["gates"]]

        # ----------- Penalties and Scoring -----------
        self.missed_gate_penalty_s = float(p.get("missed_gate_penalty_s", 3.0))
        self.penalties_this_lap = 0.0  # initialize
        #  Out-of-bounds handling
        self.out_of_bounds_penalty_s = float(p.get("out_of_bounds_penalty_s", 0.0))
        self.bounds_rect = p.get("bounds_rect", None)
        self.bounds_line_z = float(p.get("bounds_line_z", 0.2))
        self.bounds_line_width = float(p.get("bounds_line_width", 0.08))
        self.in_bounds_prev = (
            None  # track in/out state to penalize only on leaving events
        )

        # Visualization pubs (latched so RViz sees past messages)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.markers_pub = self.create_publisher(
            MarkerArray, "pylon_course/markers", latched
        )
        self.path_pub = self.create_publisher(Path, "pylon_course/centerline", latched)

        # Race result pubs
        self.score_pub = self.create_publisher(Float32, "race/score", 10)
        self.lap_time_pub = self.create_publisher(Float32, "race/lap_time", 10)
        self.lap_count_pub = self.create_publisher(Int32, "race/lap_count", 10)
        self.status_pub = self.create_publisher(String, "race/status", 10)

        # Pose sub
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.on_pose, 20
        )

        # State
        self.prev_xy: Optional[Vec2] = None
        self.curr_gate_idx = 0
        self.race_started = False
        self.lap_start_monotonic: Optional[float] = None
        self.lap_count = 0

        # from YAML (still allows override there, see below)
        # self.pylon_height_m = float(p.get("pylon_height_m", self.get_parameter("pylon_height_m").value))

        # Publish static visuals once
        self.publish_course_markers()
        self.publish_centerline()
        self.status(
            f"READY (params: {os.path.basename(params_file)}) — waiting for {pose_topic} and finish crossing."
        )

        # logging params
        self.declare_parameter(
            "laps_out_file", ""
        )  # empty => use package logs/ (resolved to src in symlink-install)
        user_path = (
            self.get_parameter("laps_out_file").get_parameter_value().string_value
        )

        # LOG: where to write lap CSV (already resolved)
        self.lap_log_path = _resolve_lap_log_path(user_path)
        self.status(f"Logging laps to: {self.lap_log_path}")

        # lap CSV
        if not FsPath(self.lap_log_path).exists():
            with open(self.lap_log_path, "w") as f:
                f.write("lap,lap_time_s,lap_score,penalties_pylon,penalties_oob\n")

        # per-lap penalty counters
        self.missed_gates_count = 0  # +1 per gate missed
        self.oob_events_count = 0  # +1 each time you leave bounds

        # best metrics and best CSV (single-line)
        self.best_lap_time = None
        self.best_score = None
        self.best_lap_num = None
        self.best_penalties_count = None

        self.declare_parameter("best_out_file", "")  # optional override
        best_user = self.get_parameter("best_out_file").value
        best_dir = str(FsPath(self.lap_log_path).parent)
        self.best_log_path = _resolve_best_log_path(best_user, best_dir)

        ### Save to Geddes Leaderboard
        self.save_to_geddes = True
        if self.save_to_geddes:
            try:
                dst_path, replaced = _maybe_update_submit_best(self.best_log_path)
                if dst_path is None:
                    self.status("Submit copy not created (JUPYTERHUB_USER unset).")
                elif replaced:
                    self.status(f"Submit best initialized → {dst_path}")
                else:
                    self.status(f"Submit best kept (existing is as good or better) → {dst_path}")
            except Exception as e:
                self.get_logger().warn(f"Submit copy init failed: {e}")
            # also save to /submit/<email>/logs/ with same filename
            try:
                _copy_submit_lap_log(self.lap_log_path)
            except Exception as e:
                self.get_logger().warn(f"Submit lap log init copy failed: {e}")



        if not FsPath(self.best_log_path).exists():
            with open(self.best_log_path, "w") as f:
                f.write("lap,lap_time_s,lap_score,penalties_pylon,penalties_oob\n")

    def _gates_crossed_ordered(self, a_xy, b_xy):
        """Return all gates crossed by segment a->b as [(t_along_AB, gate_index), ...], sorted by t."""
        events = []
        for idx, g in enumerate(self.gates):
            p1 = (self.pylons[g.p1].x, self.pylons[g.p1].y)
            p2 = (self.pylons[g.p2].x, self.pylons[g.p2].y)
            hit, t = _seg_intersect_t(a_xy, b_xy, p1, p2)
            if hit:
                events.append((t, idx))
        events.sort(key=lambda x: x[0])
        return events

    # -------- Bounds checking --------
    def _in_bounds_rect(self, xy):
        if not self.bounds_rect:
            return True
        x, y = xy
        r = self.bounds_rect
        return (r["min_x"] <= x <= r["max_x"]) and (r["min_y"] <= y <= r["max_y"])

    def _bounds_marker(self, now):
        from geometry_msgs.msg import Point

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = now
        m.ns = ""
        m.id = 4000
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = self.bounds_line_width
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = 1.0, 0.3, 0.3  # reddish

        r = self.bounds_rect
        z = self.bounds_line_z
        corners = [
            (r["min_x"], r["min_y"]),
            (r["max_x"], r["min_y"]),
            (r["max_x"], r["max_y"]),
            (r["min_x"], r["max_y"]),
            (r["min_x"], r["min_y"]),  # close loop
        ]
        for cx, cy in corners:
            m.points.append(Point(x=cx, y=cy, z=z))
        return m

    def _bounds_label(self, now):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = now
        m.ns = ""
        m.id = 4001
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.scale.z = 0.5
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = 1.0, 0.9, 0.2  # yellowish
        r = self.bounds_rect
        m.pose.position.x = 0.5 * (r["min_x"] + r["max_x"])
        m.pose.position.y = 0.5 * (r["min_y"] + r["max_y"])
        m.pose.position.z = self.bounds_line_z + 0.6
        m.text = ""
        return m

    # -------- Visualization --------
    def publish_course_markers(self):
        now = self.get_clock().now().to_msg()
        ns = "course"
        arr = MarkerArray()

        # Pylons (cylinders) + labels
        for i, py in enumerate(self.pylons):
            cyl = Marker()
            cyl.header.frame_id = self.frame_id
            cyl.header.stamp = now
            cyl.ns = ns
            cyl.id = 1000 + i
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = py.x
            cyl.pose.position.y = py.y
            cyl.pose.position.z = py.z + 5.5
            cyl.scale.x = py.r * 2.0
            cyl.scale.y = py.r * 2.0
            cyl.scale.z = 11.0
            cyl.color.a = 0.9
            if i == 0:  # emphasize pylon near finish
                cyl.color.r, cyl.color.g, cyl.color.b = 1.0, 0.3, 0.3
            else:
                cyl.color.r, cyl.color.g, cyl.color.b = 0.2, 0.5, 1.0
            arr.markers.append(cyl)

            lbl = Marker()
            lbl.header.frame_id = self.frame_id
            lbl.header.stamp = now
            lbl.ns = ns
            lbl.id = 2000 + i
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = py.x
            lbl.pose.position.y = py.y
            lbl.pose.position.z = py.z + 1.4
            lbl.scale.z = 0.35
            lbl.color.a = 1.0
            lbl.color.r = lbl.color.g = lbl.color.b = 1.0
            lbl.text = py.name
            arr.markers.append(lbl)

        # Gates (LINE_LIST) + labels at midpoints
        gates = Marker()
        gates.header.frame_id = self.frame_id
        gates.header.stamp = now
        gates.ns = ns
        gates.id = 3000
        gates.type = Marker.LINE_LIST
        gates.action = Marker.ADD
        gates.scale.x = 0.05
        gates.color.a = 1.0
        gates.color.r = gates.color.g = gates.color.b = 1.0

        for gi, g in enumerate(self.gates):
            p1, p2 = self.pylons[g.p1], self.pylons[g.p2]
            gates.points.extend(
                [
                    Point(x=p1.x, y=p1.y, z=p1.z + 0.2),
                    Point(x=p2.x, y=p2.y, z=p2.z + 0.2),
                ]
            )

            midx = 0.5 * (p1.x + p2.x)
            midy = 0.5 * (p1.y + p2.y)
            midz = 0.5 * (p1.z + p2.z)

            glbl = Marker()
            glbl.header.frame_id = self.frame_id
            glbl.header.stamp = now
            glbl.ns = ns
            glbl.id = 3100 + gi
            glbl.type = Marker.TEXT_VIEW_FACING
            glbl.action = Marker.ADD
            glbl.pose.position.x = midx
            glbl.pose.position.y = midy
            glbl.pose.position.z = midz + 0.8
            glbl.scale.z = 0.35
            glbl.color.a = 1.0
            if gi == 0:
                glbl.color.r, glbl.color.g, glbl.color.b = 1.0, 0.3, 0.3  # finish
            else:
                glbl.color.r, glbl.color.g, glbl.color.b = 0.7, 1.0, 0.7
            glbl.text = g.name
            arr.markers.append(glbl)

        arr.markers.append(gates)
        if self.bounds_rect:
            arr.markers.append(self._bounds_marker(now))
            arr.markers.append(self._bounds_label(now))
        self.markers_pub.publish(arr)

    def publish_centerline(self):
        if not self.pylons:
            return
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        for idx in range(len(self.pylons)):
            ps = PoseStamped()
            py = self.pylons[idx]
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = py.x
            ps.pose.position.y = py.y
            ps.pose.position.z = py.z + 0.1
            path.poses.append(ps)
        if self.centerline_closed:
            path.poses.append(path.poses[0])
        self.path_pub.publish(path)

    # -------- Race logic --------
    def on_pose(self, msg):
        # assuming PoseStamped
        pos = msg.pose.position
        a_xy = self.prev_xy
        b_xy = (pos.x, pos.y)

        if a_xy is None:
            self.prev_xy = b_xy
            return

        events = self._gates_crossed_ordered(a_xy, b_xy)
        now = time.monotonic()
        n = len(self.gates)

        # --- Out-of-bounds penalty: add when leaving the rectangle ---
        if self.bounds_rect and self.race_started:
            # initialize the previous state on first use
            if self.in_bounds_prev is None:
                self.in_bounds_prev = self._in_bounds_rect(a_xy)

            in_now = self._in_bounds_rect(b_xy)

            if self.in_bounds_prev and (not in_now):
                pen = float(self.out_of_bounds_penalty_s)
                if pen > 0.0:
                    self.penalties_this_lap += pen
                    self.oob_events_count += 1  # <-- NEW
                    self.status(
                        f"Out of bounds: +{pen:.1f}s (total {self.penalties_this_lap:.1f}s)."
                    )

            # update state
            self.in_bounds_prev = in_now

        # --- Process gate crossing events ---
        for _, gidx in events:
            # START only on G1
            if not self.race_started:
                if self.start_on_finish_cross and gidx == 0:
                    self.race_started = True
                    self.lap_start_monotonic = now
                    self.penalties_this_lap = 0.0
                    self.curr_gate_idx = 1  # after G1, expect G2
                    self.status("START on G1")
                continue  # ignore any other gate before start

            expected = self.curr_gate_idx
            prev_gate = (expected - 1 + n) % n
            self.status(f"Crossed gate G{gidx + 1}")

            if gidx == expected:
                # normal progression: only advance by one
                self.curr_gate_idx = (expected + 1) % n

            elif gidx == prev_gate:
                # previous gate (backtrack/jitter): ignore completely
                continue

            else:
                # forward jump: penalize missed gates and advance
                missed = (gidx - expected) % n
                if missed > 0:
                    add_pen = missed * self.missed_gate_penalty_s
                    self.missed_gates_count += missed  # count missed gates
                    self.penalties_this_lap += add_pen
                    self.status(
                        f"Missed {missed} gate(s): +{add_pen:.1f}s (total {self.penalties_this_lap:.1f}s)."
                    )
                    self.curr_gate_idx = (gidx + 1) % n
                # if missed == 0, it's the same gate re-crossed after we advanced; ignore.

            if gidx == 0:
                lap_t = now - self.lap_start_monotonic
                final_score = lap_t + self.penalties_this_lap
                self.lap_count += 1

                self.lap_time_pub.publish(Float32(data=float(lap_t)))
                self.lap_count_pub.publish(Int32(data=self.lap_count))
                self.score_pub.publish(Float32(data=float(final_score)))
                self.status(
                    f"LAP {self.lap_count} COMPLETE: {lap_t:.3f}s + {self.penalties_this_lap:.1f}s = {final_score:.3f}s"
                )

                # append to lap CSV
                with open(self.lap_log_path, "a") as f:
                    f.write(
                        f"{self.lap_count},{lap_t:.6f},{final_score:.6f},"
                        f"{self.missed_gates_count},{self.oob_events_count}\n"
                    )
                if self.save_to_geddes:
                    # mirror the updated lap log to /submit/<email>/logs/pylon_laps_...csv
                    try:
                        _copy_submit_lap_log(self.lap_log_path)
                    except Exception as e:
                        self.get_logger().warn(f"Failed to mirror lap log: {e}")


                # update best and overwrite single-line CSV if improved
                if (self.best_score is None) or (final_score < self.best_score):
                    self.best_score = final_score
                    self.best_lap_time = lap_t
                    self.best_lap_num = self.lap_count
                    try:
                        with open(self.best_log_path, "w") as f:
                            f.write("lap,lap_time_s,lap_score,penalties_pylon,penalties_oob\n")
                            f.write(
                                f"{self.best_lap_num},{self.best_lap_time:.6f},{self.best_score:.6f},"
                                f"{self.missed_gates_count},{self.oob_events_count}\n"
                            )
                        self.status(f"Best score updated → {self.best_log_path}")

                        # Only update /submit/<email>/logs/best_score.csv if the new score is better
                        if self.save_to_geddes:
                            try:
                                dst_path, replaced = _maybe_update_submit_best(self.best_log_path)
                                if dst_path is not None:
                                    if replaced:
                                        self.status(f"Submit best updated → {dst_path}")
                                    else:
                                        self.status("Submit best kept (existing is better or equal).")
                            except Exception as e:
                                self.get_logger().warn(f"Failed to mirror best_score.csv: {e}")

                    except Exception as e:
                        self.get_logger().warn(f"Failed to write best score: {e}")


                # reset for next lap
                self.lap_start_monotonic = now
                self.penalties_this_lap = 0.0
                self.missed_gates_count = 0  # reset counter
                self.oob_events_count = 0  # reset counter
                self.curr_gate_idx = 1  # after G1, expect G2

        self.prev_xy = b_xy

    def _segment_crosses_gate(self, a_xy: Vec2, b_xy: Vec2, gate_idx: int) -> bool:
        g = self.gates[gate_idx]
        p1 = v2(self.pylons[g.p1].x, self.pylons[g.p1].y)
        p2 = v2(self.pylons[g.p2].x, self.pylons[g.p2].y)

        axis = sub(p2, p1)
        L = max(norm(axis), 1e-6)
        u = mul(axis, 1.0 / L)  # along-gate unit vector
        n = perp(u)  # gate normal (perpendicular)
        mid = mul(add(p1, p2), 0.5)

        # Signed distances to the gate line (through midpoint, normal n)
        da = dot(sub(a_xy, mid), n)
        db = dot(sub(b_xy, mid), n)

        # Must be opposite signs (strictly), else no crossing
        if da == 0.0 or db == 0.0:
            return False
        if (da > 0) == (db > 0):
            return False

        # Interpolate intersection and check it's within gate segment +/- margin
        t = abs(da) / (abs(da) + abs(db))
        x_int = a_xy[0] + (b_xy[0] - a_xy[0]) * t
        y_int = a_xy[1] + (b_xy[1] - a_xy[1]) * t
        along = abs(dot(sub((x_int, y_int), mid), u))
        self.get_logger().info(
            f"Gate {gate_idx} crossing at along={along:.2f} (L/2={0.5*L:.2f} ± {self.gate_margin:.2f})"
        )
        return along <= (0.5 * L + self.gate_margin)

    def status(self, text: str):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))


def main():
    rclpy.init()
    node = CourseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
