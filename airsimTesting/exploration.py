#!/usr/bin/env python3
"""Autonomous frontier-based exploration with live SLAM mapping.

Uses LiveSLAM from finalMappingPipeline.py for real-time mapping and an
``ExplorationPlanner`` that uses the Wavefront Frontier Detection (WFD)
algorithm to drive the drone toward un-mapped regions.

The planner operates on a full 3-D voxel grid with three cell states:
observed-free, observed-occupied, and unknown.  At each decision step it:

1. Uses **raycasting** through actual LiDAR point-cloud data to mark voxels
   along each sensor ray as *observed*.  This replaces the fixed-radius
   sphere approach, giving an accurate observed region that reflects real
   sensor coverage (limited by walls, range, and field of view).
2. Builds a 3-D occupied grid from the SLAM map and derives the *frontier*
   mask — free voxels with at least one 6-connected unknown neighbour.
3. Runs a BFS from the drone through free space (outer wavefront) and
   extracts connected frontier clusters via inner BFS sweeps.
4. Scores each reachable frontier cluster by information-gain / travel-cost
   and returns the best 3-D centroid as the next waypoint.
"""

from __future__ import annotations

import numpy as np
import time
import os
import glob
import threading
import bisect

from scipy import ndimage
from scipy.spatial.transform import Rotation

from finalMappingPipeline import LiveSLAM, SLAMConfig, SLAMPipeline, _QualityPlot
from RegistrationComparison import resolve_recording_dir, filter_valid, xform_pts
from obstacleAvoidance import (
    PathPlanner, PathFollower, FollowerState, FlightResult,
    get_drone_position, sample_near_obstacle_goal, straight_line_free,
)


# ══════════════════════════════════════════════════════════════════════════════
# BufferedSLAM — wraps LiveSLAM with delayed registration
# ══════════════════════════════════════════════════════════════════════════════

class BufferedSLAM:
    """Wraps a ``LiveSLAM`` instance with a scan buffer for delayed registration.

    Raw LiDAR scans, poses, and sensor data are collected at the configured
    scan rate into a FIFO queue.  A scan is only fed to the SLAM pipeline
    once ``delay_scans`` additional scans have arrived *after* it, giving the
    high-rate state/IMU/GPS buffers (in ``saveFlightData.py`` style) time to
    bracket the LiDAR timestamp from both sides.

    Parameters
    ----------
    live : LiveSLAM
        The underlying live SLAM driver (already connected).
    delay_scans : int
        How many newer scans must arrive before a queued scan is registered
        into the SLAM map.  Higher = more temporal coverage for interpolation,
        but adds latency to the map.
    max_buffer : int
        Hard cap on buffer length (safety).
    """

    def __init__(self, live: LiveSLAM, delay_scans: int = 3, max_buffer: int = 200):
        self.live = live
        self.delay_scans = delay_scans
        self.max_buffer = max_buffer

        # Buffered raw scans: list of dicts with all data needed by process_frame
        self._scan_buf: list[dict] = []
        # How many scans have been registered into the pipeline so far
        self._n_registered: int = 0
        # Total scans collected
        self._n_collected: int = 0

        # High-rate GPS ring buffer (populated by _gps_loop on its own
        # thread, interpolated in drain_ready).  Protected by _gps_lock.
        self._gps_buf_ts: list[float] = []   # sorted AirSim timestamps (ns)
        self._gps_buf_geo: list[np.ndarray] = []  # [lat, lon, alt]
        self._GPS_BUF_MAX = 2000             # trim safety cap
        self._gps_lock = threading.Lock()    # guards _gps_buf_*

        # Optional planner reference — when set, each registered scan's
        # world-frame points are forwarded via planner.feed_scan() so the
        # planner can raycast through real LiDAR data.
        self._planner: "ExplorationPlanner | None" = None

        # Threading — two threads:
        #   _thread      : LiDAR collect + drain (SLAM processing)
        #   _gps_thread  : high-rate GPS polling (never blocked by SLAM)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gps_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._thread_client = None   # collection thread AirSim client
        self._gps_client = None      # GPS thread AirSim client

        # ── Deferred overlay queue ─────────────────────────────────────
        # Path / target overlays are queued and released after the same
        # wall-clock delay as the scan buffer so the viewer stays in sync
        # with the (delayed) SLAM map.
        self._overlay_queue: list[tuple[float, dict]] = []
        self._overlay_lock = threading.Lock()

    @property
    def overlay_delay(self) -> float:
        """Wall-clock seconds that overlays should be delayed.

        Matches the scan buffer latency: ``delay_scans / scan_rate``.
        """
        hz = self.live._min_interval  # seconds between scans
        return self.delay_scans * hz

    # ── Overlay queueing ──────────────────────────────────────────────

    def queue_overlay(self, **kwargs) -> None:
        """Schedule an overlay update to appear after the buffer delay.

        Accepted keyword arguments (all optional):
          target_pos, path_points, frontier_points, candidate_points

        The update is timestamped with ``time.time()`` and will be pushed
        to the viewer by ``drain_overlays()`` once ``overlay_delay``
        seconds have elapsed.
        """
        with self._overlay_lock:
            self._overlay_queue.append((time.time(), dict(kwargs)))

    def drain_overlays(self) -> None:
        """Push any overlay updates whose delay has elapsed to the viewer.

        Call this frequently (e.g. in every wait-loop iteration) so that
        queued overlays are released in sync with the delayed SLAM map.
        """
        now = time.time()
        delay = self.overlay_delay
        ready: list[dict] = []
        with self._overlay_lock:
            while self._overlay_queue and (now - self._overlay_queue[0][0]) >= delay:
                _, data = self._overlay_queue.pop(0)
                ready.append(data)

        # Apply each update in order; later updates overwrite earlier ones
        for data in ready:
            if "target_pos" in data:
                self.live.set_target(data["target_pos"])
            if "path_points" in data:
                self.live.set_path_points(data["path_points"])
            if "frontier_points" in data:
                self.live.set_frontier_points(data["frontier_points"])
            if "candidate_points" in data:
                self.live.set_candidate_points(data["candidate_points"])
        if ready:
            self.live.refresh_overlays()

    def set_planner(self, planner: "ExplorationPlanner") -> None:
        """Attach an ExplorationPlanner to receive per-scan world-frame data."""
        self._planner = planner

    # ── Collect one scan (rate-limited by LiveSLAM) ──────────────────────

    def collect_once(self) -> dict | None:
        """Grab one LiDAR scan + pose from AirSim and buffer it.

        Does NOT register the scan in the SLAM pipeline yet.
        Returns the raw scan dict, or None if rate-limited / no data.
        Uses the thread-local AirSim client when running in background
        mode, otherwise falls back to ``self.live.client``.
        """
        import cosysairsim as airsim
        import time as _time

        now = _time.time()
        if now - self.live._last_scan_time < self.live._min_interval:
            return None

        client = self._thread_client if self._thread_client is not None else self.live.client
        if client is None:
            raise RuntimeError("Call live.connect() first")

        # Sample state BEFORE LiDAR
        state_before = client.getMultirotorState()

        # Get sensor data
        lidar_data = client.getLidarData()

        # Sample state AFTER LiDAR
        state_after = client.getMultirotorState()

        if len(lidar_data.point_cloud) < 9:
            return None

        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape((-1, 3))

        # Interpolate pose to LiDAR timestamp
        pos, ori = self.live._interpolate_pose(state_before, state_after, lidar_data)

        # LiDAR mount offset
        lpos = lidar_data.pose.position
        lori = lidar_data.pose.orientation
        lidar_position = np.array([lpos.x_val, lpos.y_val, lpos.z_val])
        lidar_orientation = np.array([lori.w_val, lori.x_val, lori.y_val, lori.z_val])

        # GPS — grabbed at high rate by _sample_gps(); NOT grabbed here.
        # A placeholder None is stored; drain_ready() will interpolate
        # from the high-rate GPS buffer.

        lidar_ts = float(lidar_data.time_stamp)
        state_before_ts = float(state_before.timestamp)
        state_after_ts = float(state_after.timestamp)

        scan = {
            "points": points,
            "position": pos,
            "orientation": ori,
            "lidar_position": lidar_position,
            "lidar_orientation": lidar_orientation,
            "gps": None,  # filled by drain_ready via GPS buffer interpolation
            "lidar_ts": lidar_ts,
            "state_before_ts": state_before_ts,
            "state_after_ts": state_after_ts,
            "collect_time": now,
            "frame_label": self.live._frame_count,
        }

        self._scan_buf.append(scan)
        self._n_collected += 1
        self.live._frame_count += 1
        self.live._last_scan_time = now

        # Save raw frame to disk if recording is enabled
        if self.live.save_dir is not None:
            self.live._save_frame(points, pos, ori, lidar_position,
                                  lidar_orientation, None)

        # Safety trim
        if len(self._scan_buf) > self.max_buffer:
            self._scan_buf = self._scan_buf[-self.max_buffer:]

        return scan

    # ── Register buffered scans that are old enough ──────────────────────

    def drain_ready(self) -> list[dict]:
        """Register any buffered scans that have enough newer scans after them.

        A scan at index *i* is ready when there are at least ``delay_scans``
        newer scans in the buffer (i.e. scans at indices i+1 … i+delay_scans).

        Returns a list of pipeline results for newly registered scans.
        """
        results = []
        ready_count = len(self._scan_buf) - self.delay_scans

        while self._n_registered < ready_count and self._n_registered < len(self._scan_buf):
            scan = self._scan_buf[self._n_registered]

            # ── Interpolate GPS from the high-rate buffer ─────────────
            lidar_ts = scan["lidar_ts"]
            gps_interp = None
            gps_bracket_ms = None
            with self._gps_lock:
                buf_ts = list(self._gps_buf_ts)    # snapshot
                buf_geo = list(self._gps_buf_geo)
            if len(buf_ts) >= 2:
                idx = bisect.bisect_right(buf_ts, lidar_ts)
                i_before = max(idx - 1, 0)
                i_after = min(idx, len(buf_ts) - 1)
                ts_a = buf_ts[i_before]
                ts_b = buf_ts[i_after]
                near_ms = min(abs(lidar_ts - ts_a),
                              abs(lidar_ts - ts_b)) / 1e6
                far_ms = max(abs(lidar_ts - ts_a),
                             abs(lidar_ts - ts_b)) / 1e6
                gps_bracket_ms = (near_ms, far_ms)
                if ts_b != ts_a:
                    t = np.clip((lidar_ts - ts_a) / (ts_b - ts_a), 0.0, 1.0)
                    gps_interp = ((1 - t) * buf_geo[i_before]
                                  + t * buf_geo[i_after])
                else:
                    gps_interp = buf_geo[i_before].copy()
            scan["gps"] = gps_interp

            # Print timing: how far the bracketing state samples are from LiDAR
            state_near_ms = min(abs(scan["lidar_ts"] - scan["state_before_ts"]),
                                abs(scan["lidar_ts"] - scan["state_after_ts"])) / 1e6
            state_far_ms = max(abs(scan["lidar_ts"] - scan["state_before_ts"]),
                               abs(scan["lidar_ts"] - scan["state_after_ts"])) / 1e6
            gps_str = (f"GPS: {gps_bracket_ms[0]:5.1f}/{gps_bracket_ms[1]:5.1f} ms"
                       if gps_bracket_ms else "GPS: n/a")
            buf_depth = len(self._scan_buf) - self._n_registered

            # ── GPS quality gate ──────────────────────────────────────
            # Reject scan if the nearest GPS sample is >20 ms away or
            # the furthest bracket sample is >40 ms away.
            GPS_NEAR_MAX_MS = 10.0
            GPS_FAR_MAX_MS  = 40.0
            if gps_bracket_ms is not None:
                gps_near, gps_far = gps_bracket_ms
                if gps_near > GPS_NEAR_MAX_MS or gps_far > GPS_FAR_MAX_MS:
                    self._n_gps_rejected = getattr(self, '_n_gps_rejected', 0) + 1
                    if self._n_gps_rejected <= 5 or self._n_gps_rejected % 20 == 0:
                        print(f"  [buf] REJECTED scan {scan['frame_label']:03d}  "
                              f"| state: {state_near_ms:5.1f}/{state_far_ms:5.1f} ms  "
                              f"| {gps_str}  "
                              f"| reason: GPS bracket too wide  "
                              f"(total rejected: {self._n_gps_rejected})")
                    self._n_registered += 1
                    continue
            elif gps_interp is None:
                # No GPS data at all — also reject
                self._n_gps_rejected = getattr(self, '_n_gps_rejected', 0) + 1
                if self._n_gps_rejected <= 5 or self._n_gps_rejected % 20 == 0:
                    print(f"  [buf] REJECTED scan {scan['frame_label']:03d}  "
                          f"| state: {state_near_ms:5.1f}/{state_far_ms:5.1f} ms  "
                          f"| GPS: n/a  "
                          f"| reason: no GPS data  "
                          f"(total rejected: {self._n_gps_rejected})")
                self._n_registered += 1
                continue

            print(f"  [buf] registering scan {scan['frame_label']:03d}  "
                  f"| state: {state_near_ms:5.1f}/{state_far_ms:5.1f} ms  "
                  f"| {gps_str}  "
                  f"| buf depth: {buf_depth}")

            # Feed to SLAM pipeline (drone pos updated below after
            # all per-scan work so the viewer gets one consistent update)
            result = self.live.pipeline.process_frame(
                scan["points"],
                scan["position"],
                scan["orientation"],
                gps=scan["gps"],
                lidar_position=scan["lidar_position"],
                lidar_orientation=scan["lidar_orientation"],
                frame_label=scan["frame_label"],
            )
            results.append(result)
            self._n_registered += 1

            # Forward world-frame points to the planner for raycasting
            if self._planner is not None:
                pts = scan["points"][np.any(scan["points"] != 0, axis=1)]
                if len(pts) > 0:
                    lp = scan["lidar_position"]
                    lo = scan["lidar_orientation"]
                    R_l = Rotation.from_quat(
                        [lo[1], lo[2], lo[3], lo[0]]).as_matrix()
                    body = (R_l @ pts.T).T + lp
                    ori = scan["orientation"]
                    R_b = Rotation.from_quat(
                        [ori[1], ori[2], ori[3], ori[0]]).as_matrix()
                    world_pts = (R_b @ body.T).T + scan["position"]
                    self._planner.feed_scan(
                        scan["position"].copy(),
                        world_pts.astype(np.float32),
                    )
                    # Push updated frontier overlay to the viewer
                    frontier_pts = self._planner.get_frontier_points()
                    self.live.pipeline.set_frontier_points(
                        frontier_pts if len(frontier_pts) > 0 else None)

            # Single authoritative drone-position update per scan.
            # This is the ONLY place that should call set_drone_pos
            # during live collection so the viewer marker doesn't jump.
            self.live.pipeline.set_drone_pos(scan["position"])
            self.live.pipeline.refresh_overlays()

        # Trim fully-registered scans from the front of the buffer,
        # but keep at least delay_scans entries for context
        keep = max(self.delay_scans, 10)
        if self._n_registered > keep:
            trim = self._n_registered - keep
            self._scan_buf = self._scan_buf[trim:]
            self._n_registered -= trim

        return results

    # ── Combined: collect + drain (drop-in for process_once) ─────────────

    def process_once(self) -> dict | None:
        """Collect a scan (if rate allows) and register any ready scans.

        This is a drop-in replacement for ``live.process_once()`` in any
        control loop.
        """
        self.collect_once()
        results = self.drain_ready()
        return results[-1] if results else None

    # ── Flush: force-register remaining buffered scans ───────────────────

    def flush(self):
        """Register all remaining buffered scans (call at end of flight)."""
        remaining = len(self._scan_buf) - self._n_registered
        if remaining > 0:
            print(f"  [buf] flushing {remaining} remaining scans...")
        # Temporarily set delay to 0 to drain everything
        old_delay = self.delay_scans
        self.delay_scans = 0
        self.drain_ready()
        self.delay_scans = old_delay

    @property
    def buffer_size(self) -> int:
        """Number of scans currently in the buffer."""
        return len(self._scan_buf)

    @property
    def pending(self) -> int:
        """Number of scans collected but not yet registered."""
        return len(self._scan_buf) - self._n_registered

    # ── Background-thread scan collection ────────────────────────────

    def start_collection(self) -> None:
        """Launch background threads for scan collection and GPS polling.

        Two threads are started, each with its own AirSim client:
          * **collection thread** — grabs LiDAR + state, runs SLAM
          * **GPS thread** — polls GPS at high rate (~5 ms) so the ring
            buffer always brackets every LiDAR timestamp tightly, even
            while SLAM registration is blocking the collection thread.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._gps_thread = threading.Thread(
            target=self._gps_loop, name="BufferedSLAM-GPS", daemon=True)
        self._gps_thread.start()
        self._thread = threading.Thread(
            target=self._collection_loop, name="BufferedSLAM", daemon=True)
        self._thread.start()

    def stop_collection(self, timeout: float = 5.0) -> None:
        """Signal both background threads to stop and wait."""
        self._stop_event.set()
        if self._gps_thread is not None:
            self._gps_thread.join(timeout=timeout)
            self._gps_thread = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._thread_client = None
        self._gps_client = None

    def _collection_loop(self) -> None:
        """Background thread: LiDAR collection + SLAM registration."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        import cosysairsim as airsim
        self._thread_client = airsim.MultirotorClient()
        self._thread_client.confirmConnection()
        print("  [BufferedSLAM] Collection thread AirSim client connected")

        while not self._stop_event.is_set():
            try:
                self.collect_once()
                self.drain_ready()
            except Exception as e:
                print(f"  [BufferedSLAM] Error: {e}")
            # Sleep just under the scan interval so we never miss a window
            time.sleep(self.live._min_interval * 0.5)

    def _gps_loop(self) -> None:
        """Background thread: high-rate GPS polling.

        Runs on its own AirSim client so it is never blocked by the
        SLAM processing in ``drain_ready()``.
        """
        import cosysairsim as airsim
        self._gps_client = airsim.MultirotorClient()
        self._gps_client.confirmConnection()
        print("  [BufferedSLAM] GPS thread AirSim client connected")

        while not self._stop_event.is_set():
            self._sample_gps()
            time.sleep(0.005)  # ~200 Hz

    def _sample_gps(self) -> None:
        """Poll GPS once and append to the high-rate ring buffer."""
        client = self._gps_client if self._gps_client is not None else self.live.client
        if client is None:
            return
        try:
            gps_data = client.getGpsData()
            ts = float(gps_data.time_stamp)
            gp = gps_data.gnss.geo_point
            geo = np.array([gp.latitude, gp.longitude, gp.altitude])
            with self._gps_lock:
                self._gps_buf_ts.append(ts)
                self._gps_buf_geo.append(geo)
                # Trim if too large
                if len(self._gps_buf_ts) > self._GPS_BUF_MAX:
                    trim = len(self._gps_buf_ts) - self._GPS_BUF_MAX
                    self._gps_buf_ts = self._gps_buf_ts[trim:]
                    self._gps_buf_geo = self._gps_buf_geo[trim:]
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ExplorationPlanner
# ══════════════════════════════════════════════════════════════════════════════

class ExplorationPlanner:
    """3-D Wavefront Frontier Detection (WFD) planner for autonomous exploration.

    Uses the Wavefront Frontier Detection algorithm (Keidar & Kaminka, 2012)
    adapted for 3-D voxel grids.  Three voxel states are distinguished:

    - *Observed*: voxels through which a LiDAR ray has passed (raycasting).
    - *Occupied*: observed voxels that contain SLAM map points.
    - *Free*: observed voxels that are **not** occupied.
    - *Unknown*: voxels that no LiDAR ray has ever traversed.

    The *observed* region is determined by **raycasting** from each scan's
    sensor origin through the actual LiDAR point cloud.  Every voxel on the
    line-of-sight from sensor to hit-point is marked as observed, giving an
    accurate representation of what the LiDAR has actually seen — walls
    block the rays, and unscanned directions remain *unknown*.

    A *frontier* voxel is a free voxel with at least one 6-connected unknown
    neighbour.  The algorithm performs a BFS from the drone's current grid
    cell through free space, grouping contiguous frontier voxels into
    clusters as they are discovered.  Only frontiers that are **reachable**
    from the drone through known free space are returned.

    Parameters
    ----------
    bounds : tuple of six floats ``(xmin, xmax, ymin, ymax, zmin, zmax)``
        Axis-aligned bounding box of the exploration volume (NED frame).
    resolution : float
        Voxel edge length in metres for the 3-D planning grid (default 1 m).
    min_frontier_size : int
        Minimum voxels in a frontier cluster for it to be a valid target.
    ray_subsample : float
        Voxel size (m) for downsampling LiDAR endpoints before raycasting.
        Smaller → more rays → slower but higher fidelity.  A value equal
        to ``resolution`` is a good starting point (one ray per planning
        voxel).
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float, float, float],
        resolution: float = 1.0,
        min_frontier_size: int = 3,
        ray_subsample: float | None = None,
        waypoint_exclusion_radius: float = 3.0,
        unknown_gain_radius: int = 3,
        distance_exponent: float = 0.5,
        lidar_altitude_offset: float = 0.0,
        min_target_distance: float = 3.0,
        inflation_margin: float = 2.0,
        # ── Random waypoint parameters ────────────────────────────
        use_random: bool = False,
        random_max_attempts: int = 50,
        # ── NBV parameters ────────────────────────────────────────
        use_nbv: bool = False,
        nbv_sensor_half_angle: float = 45.0,
        nbv_cruise_altitude: float | None = None,
        nbv_n_unknown_columns: int = 10,
        nbv_n_local_samples: int = 15,
        nbv_local_radius: float = 10.0,
        above_grid_margin: float = 0.0,
        nbv_lidar_max_range: float = 40.0,
        nbv_unknown_block_size: int = 4,
        nbv_n_unknown_blocks: int = 8,
        nbv_volumetric_radius: float = 8.0,
        nbv_volumetric_weight: float = 0.3,
        nbv_ray_max_targets: int = 500,
        nbv_use_ray_tracing: bool = True,
    ):
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = bounds
        self.resolution = resolution
        self.min_frontier_size = min_frontier_size
        self.ray_subsample = ray_subsample if ray_subsample is not None else resolution
        self.waypoint_exclusion_radius = waypoint_exclusion_radius
        self.unknown_gain_radius = unknown_gain_radius
        self.distance_exponent = distance_exponent
        self.lidar_altitude_offset = lidar_altitude_offset
        self.min_target_distance = min_target_distance
        self.inflation_margin = inflation_margin

        # Random waypoint settings
        self.use_random = use_random
        self.random_max_attempts = random_max_attempts

        # Reference to the PathPlanner (set by run_live) so that random
        # target selection can use sample_near_obstacle_goal().
        self._path_planner: "PathPlanner | None" = None

        # NBV settings
        self.use_nbv = use_nbv
        self.nbv_sensor_half_angle = nbv_sensor_half_angle
        self.nbv_cruise_altitude = nbv_cruise_altitude
        self.nbv_n_unknown_columns = nbv_n_unknown_columns
        self.nbv_n_local_samples = nbv_n_local_samples
        self.nbv_local_radius = nbv_local_radius
        self.above_grid_margin = above_grid_margin
        self.nbv_lidar_max_range = nbv_lidar_max_range
        self.nbv_unknown_block_size = nbv_unknown_block_size
        self.nbv_n_unknown_blocks = nbv_n_unknown_blocks
        self.nbv_volumetric_radius = nbv_volumetric_radius
        self.nbv_volumetric_weight = nbv_volumetric_weight
        self.nbv_ray_max_targets = nbv_ray_max_targets
        self.nbv_use_ray_tracing = nbv_use_ray_tracing

        self.nx = int(np.ceil((self.xmax - self.xmin) / resolution))
        self.ny = int(np.ceil((self.ymax - self.ymin) / resolution))
        self.nz = int(np.ceil((self.zmax - self.zmin) / resolution))

        # Persistent observed grid — once observed, always observed
        self._observed = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

        # Last occupied grid (updated each next_target call)
        self._last_occupied_grid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

        # Raw scan data fed from BufferedSLAM: (sensor_origin(3,), world_pts(N,3))
        self._scan_data: list[tuple[np.ndarray, np.ndarray]] = []
        # How many scans have already been raycasted into _observed
        self._n_scans_raycasted: int = 0

        # History of all selected waypoints — used to avoid revisiting
        self._waypoint_history: list[np.ndarray] = []

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _world_to_grid(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        ix = int(np.clip(int((x - self.xmin) / self.resolution), 0, self.nx - 1))
        iy = int(np.clip(int((y - self.ymin) / self.resolution), 0, self.ny - 1))
        iz = int(np.clip(int((z - self.zmin) / self.resolution), 0, self.nz - 1))
        return ix, iy, iz

    def _grid_to_world(self, ix: float, iy: float, iz: float) -> tuple[float, float, float]:
        x = self.xmin + (ix + 0.5) * self.resolution
        y = self.ymin + (iy + 0.5) * self.resolution
        z = self.zmin + (iz + 0.5) * self.resolution
        return x, y, z

    # ── Scan data input ───────────────────────────────────────────────────

    def feed_scan(self, sensor_origin: np.ndarray, world_pts: np.ndarray) -> None:
        """Store a LiDAR scan and immediately raycast it into the observed grid.

        Also marks the scan's hit-points as occupied so the frontier overlay
        (observed & not-occupied neighbours unknown) stays current between
        planning calls.

        Parameters
        ----------
        sensor_origin : ndarray, shape (3,)
            Position of the sensor in NED world frame when the scan was taken.
        world_pts : ndarray, shape (N, 3)
            LiDAR hit-points already in NED world frame.
        """
        wpts = np.asarray(world_pts, dtype=np.float32)
        self._scan_data.append((
            np.asarray(sensor_origin, dtype=np.float32).ravel()[:3],
            wpts,
        ))
        # Raycast immediately so the observed grid stays current.
        self._raycast_update_observed()

        # Mark hit-points as occupied so frontier derivation is up-to-date
        if len(wpts) > 0:
            res = self.resolution
            ix = np.clip(((wpts[:, 0] - self.xmin) / res).astype(np.intp),
                         0, self.nx - 1)
            iy = np.clip(((wpts[:, 1] - self.ymin) / res).astype(np.intp),
                         0, self.ny - 1)
            iz = np.clip(((wpts[:, 2] - self.zmin) / res).astype(np.intp),
                         0, self.nz - 1)
            self._last_occupied_grid[ix, iy, iz] = True

    # ── Frontier overlay (cheap, no BFS) ──────────────────────────────────

    def get_frontier_points(self) -> np.ndarray:
        """Return world-frame frontier voxel centres from the current grids.

        Cheap to call — only uses the already-maintained ``_observed`` and
        ``_last_occupied_grid`` arrays (no pipeline query, no BFS).

        Returns (N, 3) float64 or empty (0, 3).
        """
        free = self._observed & ~self._last_occupied_grid
        unknown = ~self._observed
        struct = ndimage.generate_binary_structure(3, 1)
        unknown_adj = ndimage.binary_dilation(unknown, structure=struct)
        is_frontier = free & unknown_adj
        frt = np.argwhere(is_frontier)
        if len(frt) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.column_stack([
            self.xmin + (frt[:, 0] + 0.5) * self.resolution,
            self.ymin + (frt[:, 1] + 0.5) * self.resolution,
            self.zmin + (frt[:, 2] + 0.5) * self.resolution,
        ])

    # ── Raycasting-based observed-zone update ─────────────────────────────

    def _raycast_update_observed(self) -> None:
        """Mark voxels along LiDAR rays as *observed* for all new scans.

        For each unprocessed scan the method:
        1. Down-samples the hit-points to ~1 per ``ray_subsample`` voxel to
           limit the number of rays.
        2. Densely samples points along each ray (origin → hit-point) at
           ``resolution / 2`` spacing.
        3. Converts the sample coordinates to grid indices and marks those
           cells as observed.

        This is a vectorised NumPy implementation — no per-ray Python loop.
        """
        res = self.resolution
        half_step = res * 0.5          # sampling interval along rays
        sub_inv = 1.0 / self.ray_subsample

        for origin, world_pts in self._scan_data[self._n_scans_raycasted:]:
            if len(world_pts) == 0:
                self._n_scans_raycasted += 1
                continue

            # ── Subsample endpoints: keep one per subsample-voxel ─────
            ijk = np.floor(world_pts * sub_inv).astype(np.int32)
            _, idx = np.unique(ijk, axis=0, return_index=True)
            endpoints = world_pts[idx]            # (M, 3)

            # ── Dense sampling along each ray ─────────────────────────
            dirs = endpoints - origin              # (M, 3)
            lengths = np.linalg.norm(dirs, axis=1) # (M,)

            # Max sample count across all rays (ensures no cell is missed)
            max_steps = max(2, int(np.ceil(lengths.max() / half_step)) + 1)
            t_vals = np.linspace(0.0, 1.0, max_steps)  # (S,)

            # Broadcast: (M, S, 3) = (1,1,3) + (1,S,1) * (M,1,3)
            samples = (origin[None, None, :]
                       + t_vals[None, :, None] * dirs[:, None, :])

            # Flatten to (M*S, 3) and convert to grid indices
            flat = samples.reshape(-1, 3)
            gx = np.clip(((flat[:, 0] - self.xmin) / res).astype(np.intp),
                          0, self.nx - 1)
            gy = np.clip(((flat[:, 1] - self.ymin) / res).astype(np.intp),
                          0, self.ny - 1)
            gz = np.clip(((flat[:, 2] - self.zmin) / res).astype(np.intp),
                          0, self.nz - 1)

            self._observed[gx, gy, gz] = True
            self._n_scans_raycasted += 1

    # ── Wavefront Frontier Detection BFS ─────────────────────────────────

    def _wfd_bfs(
        self,
        start: tuple[int, int, int],
        free: np.ndarray,
        is_frontier: np.ndarray,
    ) -> list[np.ndarray]:
        """Run Wavefront Frontier Detection BFS from *start* through *free*.

        The outer BFS expands through free voxels starting at the drone's
        position.  When a frontier voxel is reached, an inner BFS extracts
        the entire connected frontier cluster before the outer BFS continues.

        Parameters
        ----------
        start : (ix, iy, iz)
            Grid index of the drone's current cell.
        free : bool ndarray (nx, ny, nz)
            True for free (observed & not occupied) voxels.
        is_frontier : bool ndarray (nx, ny, nz)
            True for frontier voxels (free with ≥1 unknown 6-neighbour).

        Returns
        -------
        list of ndarray, each shape (N, 3) int
            All frontier clusters found (no size filter applied here).
        """
        from collections import deque

        NBRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        _MAP_OPEN = 1;  _MAP_CLOSE = 2
        _FRT_OPEN = 3;  _FRT_CLOSE = 4

        nx, ny, nz = self.nx, self.ny, self.nz
        cell_state = np.zeros((nx, ny, nz), dtype=np.int8)

        sx, sy, sz = start
        queue_m: deque[tuple[int, int, int]] = deque([(sx, sy, sz)])
        cell_state[sx, sy, sz] = _MAP_OPEN

        clusters: list[np.ndarray] = []

        while queue_m:
            px, py, pz = queue_m.popleft()
            if cell_state[px, py, pz] == _MAP_CLOSE:
                continue

            if is_frontier[px, py, pz]:
                # ── Inner BFS: extract connected frontier cluster ────
                queue_f: deque[tuple[int, int, int]] = deque([(px, py, pz)])
                cluster: list[tuple[int, int, int]] = []
                cell_state[px, py, pz] = _FRT_OPEN

                while queue_f:
                    qx, qy, qz = queue_f.popleft()
                    if cell_state[qx, qy, qz] in (_MAP_CLOSE, _FRT_CLOSE):
                        continue
                    if is_frontier[qx, qy, qz]:
                        cluster.append((qx, qy, qz))
                        for dx, dy, dz in NBRS:
                            rx, ry, rz = qx + dx, qy + dy, qz + dz
                            if (0 <= rx < nx and 0 <= ry < ny and 0 <= rz < nz
                                    and cell_state[rx, ry, rz] not in
                                    (_FRT_CLOSE, _MAP_CLOSE, _FRT_OPEN)):
                                queue_f.append((rx, ry, rz))
                                cell_state[rx, ry, rz] = _FRT_OPEN
                    cell_state[qx, qy, qz] = _FRT_CLOSE

                if cluster:
                    clusters.append(np.array(cluster, dtype=int))
                # Mark frontier voxels as closed for the outer BFS
                for fx, fy, fz in cluster:
                    cell_state[fx, fy, fz] = _MAP_CLOSE

            # ── Expand outer BFS through free space ──────────────────
            for dx, dy, dz in NBRS:
                vx, vy, vz = px + dx, py + dy, pz + dz
                if (0 <= vx < nx and 0 <= vy < ny and 0 <= vz < nz
                        and cell_state[vx, vy, vz] not in (_MAP_CLOSE, _MAP_OPEN)
                        and free[vx, vy, vz]):
                    queue_m.append((vx, vy, vz))
                    cell_state[vx, vy, vz] = _MAP_OPEN

            cell_state[px, py, pz] = _MAP_CLOSE

        return clusters

    # ── Next-Best-View (NBV) target selection ─────────────────────────────

    def _nbv_generate_candidates(
        self,
        current_pos: np.ndarray,
        occupied_grid: np.ndarray,
        free: np.ndarray,
        unknown: np.ndarray,
        is_frontier: np.ndarray,
        start: tuple[int, int, int],
    ) -> np.ndarray:
        """Generate candidate viewpoints from multiple heuristics.

        Heuristic 1 — **Frontier cluster viewpoints** at multiple altitudes:
            For each WFD frontier cluster centroid, generate candidates at
            the frontier Z, at half/full/double ``lidar_altitude_offset``
            above it, and at ``nbv_cruise_altitude``.  The multi-altitude
            spread lets the drone fly under overhangs at lower Z or scan
            wide from higher up.

        Heuristic 2 — **High-unknown columns**:
            Find the ``nbv_n_unknown_columns`` columns (ix, iy) with the
            most unknown voxels.  Place a candidate at the topmost free
            voxel in each column (and at cruise altitude).

        Heuristic 3 — **Local neighbourhood samples**:
            Random positions within ``nbv_local_radius`` of the drone in
            free space, for immediate opportunities.

        Heuristic 4 — **Dense unknown region centroids**:
            Divides the grid into coarse blocks of
            ``nbv_unknown_block_size`` voxels, counts unknown voxels per
            block, and generates candidates at viewable positions around
            the densest blocks.  This targets interior unknown regions
            that have no frontier boundary (e.g. behind walls or in
            entirely unseen areas).

        Returns
        -------
        candidates : ndarray (N, 3)
            De-duplicated candidate viewpoints in NED world frame.
        """
        raw: list[np.ndarray] = []
        margin = self.inflation_margin
        cruise_z = (self.nbv_cruise_altitude
                    if self.nbv_cruise_altitude is not None
                    else current_pos[2])

        # ── H1: Frontier cluster viewpoints at multiple altitudes ─────
        clusters = self._wfd_bfs(start, free, is_frontier)
        valid_clusters = [c for c in clusters
                          if len(c) >= self.min_frontier_size]

        z_offsets = [0.0]
        if self.lidar_altitude_offset > 0:
            z_offsets += [
                self.lidar_altitude_offset * 0.5,
                self.lidar_altitude_offset,
                self.lidar_altitude_offset * 2.0,
            ]

        for cells in valid_clusters:
            cx_ = float(cells[:, 0].mean())
            cy_ = float(cells[:, 1].mean())
            cz_ = float(cells[:, 2].mean())
            wx, wy, wz_f = self._grid_to_world(cx_, cy_, cz_)
            for zo in z_offsets:
                raw.append(np.array([wx, wy, wz_f - zo]))
            raw.append(np.array([wx, wy, cruise_z]))

        # ── H2: Columns with most unknown voxels ─────────────────────
        unknown_per_col = unknown.sum(axis=2)              # (nx, ny)
        n_cols = min(self.nbv_n_unknown_columns,
                     self.nx * self.ny)
        flat_top = np.argsort(unknown_per_col.ravel())[::-1][:n_cols]

        for fi in flat_top:
            if unknown_per_col.ravel()[fi] == 0:
                break
            ix_, iy_ = np.unravel_index(fi, unknown_per_col.shape)
            wx, wy, _ = self._grid_to_world(ix_, iy_, 0)
            # Topmost free voxel in column (smallest iz ⇒ most negative z)
            for iz_ in range(self.nz):
                if free[ix_, iy_, iz_]:
                    _, _, wz = self._grid_to_world(ix_, iy_, iz_)
                    raw.append(np.array([wx, wy, wz]))
                    break
            raw.append(np.array([wx, wy, cruise_z]))

        # ── H3: Local random samples around drone ────────────────────
        for _ in range(self.nbv_n_local_samples):
            offset = np.random.uniform(
                -self.nbv_local_radius, self.nbv_local_radius, 3)
            cand = current_pos[:3] + offset
            if not (self.xmin + margin <= cand[0] <= self.xmax - margin and
                    self.ymin + margin <= cand[1] <= self.ymax - margin):
                continue
            gx, gy, gz = self._world_to_grid(*cand)
            if occupied_grid[gx, gy, gz]:
                continue
            raw.append(cand)

        # ── H4: Dense unknown region centroids ───────────────────────
        # Divide grid into coarse blocks and find the densest unknown
        # regions.  These may have NO frontier boundary at all.
        bs = self.nbv_unknown_block_size
        n_bx = max(1, self.nx // bs)
        n_by = max(1, self.ny // bs)
        n_bz = max(1, self.nz // bs)
        # Trim the unknown grid to an even multiple of bs for reshaping
        unk_trimmed = unknown[:n_bx * bs, :n_by * bs, :n_bz * bs]
        block_counts = unk_trimmed.reshape(
            n_bx, bs, n_by, bs, n_bz, bs
        ).sum(axis=(1, 3, 5))  # (n_bx, n_by, n_bz)

        n_blocks = min(self.nbv_n_unknown_blocks, n_bx * n_by * n_bz)
        flat_top_blk = np.argsort(block_counts.ravel())[::-1][:n_blocks]

        for fi in flat_top_blk:
            if block_counts.ravel()[fi] == 0:
                break
            bix, biy, biz = np.unravel_index(fi, block_counts.shape)
            # Block centroid in grid coords
            gcx = (bix + 0.5) * bs
            gcy = (biy + 0.5) * bs
            gcz = (biz + 0.5) * bs
            wx, wy, wz = self._grid_to_world(gcx, gcy, gcz)

            # Candidate at cruise altitude directly above the block
            raw.append(np.array([wx, wy, cruise_z]))
            # Candidate at the LiDAR offset above the block centroid
            if self.lidar_altitude_offset > 0:
                raw.append(np.array([wx, wy, wz - self.lidar_altitude_offset]))
                raw.append(np.array([wx, wy, wz - self.lidar_altitude_offset * 2.0]))
            # Candidate at the block's own Z (useful for enclosed regions)
            raw.append(np.array([wx, wy, wz]))
            # Offset candidates approaching from four horizontal sides
            offset_d = bs * self.resolution * 0.5 + margin
            for dx_, dy_ in [(offset_d, 0), (-offset_d, 0),
                             (0, offset_d), (0, -offset_d)]:
                sx_ = wx + dx_
                sy_ = wy + dy_
                if (self.xmin + margin <= sx_ <= self.xmax - margin and
                        self.ymin + margin <= sy_ <= self.ymax - margin):
                    raw.append(np.array([sx_, sy_, cruise_z]))

        if not raw:
            return np.empty((0, 3), dtype=np.float64)

        # ── De-duplicate: snap to half-resolution grid ───────────────
        pts = np.array(raw, dtype=np.float64)
        snap = self.resolution * 0.5
        keys = np.round(pts / snap).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        pts = pts[np.sort(idx)]

        # ── Clamp Z to valid OMPL range ──────────────────────────────
        # The PathPlanner extends zmin upward by above_grid_margin.
        # In NED more negative = higher, so the plannable ceiling is
        # zmin - above_grid_margin.  Keep a 0.5 m buffer inside.
        z_ceil = self.zmin - self.above_grid_margin + 0.5
        z_floor = self.zmax - 0.5
        pts[:, 2] = np.clip(pts[:, 2], z_ceil, z_floor)
        return pts

    def _nbv_score_candidate(
        self,
        cand: np.ndarray,
        occupied_grid: np.ndarray,
        unknown: np.ndarray,
    ) -> tuple[int, int]:
        """Score a viewpoint — dispatches to ray-traced or columnar method."""
        if self.nbv_use_ray_tracing:
            return self._nbv_score_candidate_raytrace(cand, occupied_grid, unknown)
        else:
            return self._nbv_score_candidate_columnar(cand, occupied_grid, unknown)

    def _nbv_score_candidate_columnar(
        self,
        cand: np.ndarray,
        occupied_grid: np.ndarray,
        unknown: np.ndarray,
    ) -> tuple[int, int]:
        """Score *one* viewpoint using fast columnar (vertical) occlusion.

        For each column (ix, iy) within the cone footprint, walks downward
        from the candidate's Z level and counts unknown voxels until an
        occupied voxel occludes the rest of the column.  Fast but does not
        account for lateral obstacles (walls at the same altitude).

        Returns
        -------
        frustum_gain : int
            Visible unknown voxels inside the LiDAR cone.
        volumetric_unknown : int
            Total unknown voxels within ``nbv_volumetric_radius``.
        """
        cx, cy, cz = cand
        tan_ha = np.tan(np.radians(self.nbv_sensor_half_angle))
        res = self.resolution
        max_range = self.nbv_lidar_max_range
        max_range_sq = max_range * max_range

        # Grid iz just at or above the candidate altitude
        ciz_float = (cz - self.zmin) / res
        ciz = int(np.clip(np.floor(ciz_float), -1, self.nz - 1))
        iz_start = max(ciz + 1, 0)
        if iz_start >= self.nz:
            return 0, 0

        # Column world coordinates
        col_x = self.xmin + (np.arange(self.nx) + 0.5) * res
        col_y = self.ymin + (np.arange(self.ny) + 0.5) * res
        dx2 = (col_x - cx) ** 2
        dy2 = (col_y - cy) ** 2
        dist_sq_xy = dx2[:, np.newaxis] + dy2[np.newaxis, :]

        # Per-column first-occupied below candidate
        first_occ = np.full((self.nx, self.ny), self.nz, dtype=np.intp)
        for iz in range(iz_start, self.nz):
            mask = occupied_grid[:, :, iz] & (first_occ == self.nz)
            first_occ[mask] = iz

        wz_levels = self.zmin + (np.arange(iz_start, self.nz) + 0.5) * res
        depths = wz_levels - cz
        cone_r_sq = np.where(depths > 0, (depths * tan_ha) ** 2, -1.0)

        dz2 = depths ** 2
        dist3d_sq = (dist_sq_xy[:, :, np.newaxis]
                     + dz2[np.newaxis, np.newaxis, :])

        iz_abs = np.arange(iz_start, self.nz)
        in_cone = dist_sq_xy[:, :, np.newaxis] <= cone_r_sq[np.newaxis, np.newaxis, :]
        in_range = dist3d_sq <= max_range_sq
        not_occ = iz_abs[np.newaxis, np.newaxis, :] < first_occ[:, :, np.newaxis]
        unk = unknown[:, :, iz_start:]

        frustum_gain = int((in_cone & in_range & not_occ & unk).sum())

        # Volumetric unknown density — columnar occlusion applied within
        # a sphere around the candidate so unknowns behind walls are not
        # counted.  We reuse the columnar first-occupied map but extend
        # to all Z levels (not just below the candidate).
        vol_r = self.nbv_volumetric_radius
        vol_r_vox = int(np.ceil(vol_r / res))
        gxc = int(np.clip((cx - self.xmin) / res, 0, self.nx - 1))
        gyc = int(np.clip((cy - self.ymin) / res, 0, self.ny - 1))
        gzc = int(np.clip((cz - self.zmin) / res, 0, self.nz - 1))
        x0 = max(gxc - vol_r_vox, 0); x1 = min(gxc + vol_r_vox + 1, self.nx)
        y0 = max(gyc - vol_r_vox, 0); y1 = min(gyc + vol_r_vox + 1, self.ny)
        z0 = max(gzc - vol_r_vox, 0); z1 = min(gzc + vol_r_vox + 1, self.nz)
        vol_unk_box = unknown[x0:x1, y0:y1, z0:z1]
        # Distance filter within the box to make it a sphere
        vx_ = self.xmin + (np.arange(x0, x1) + 0.5) * res
        vy_ = self.ymin + (np.arange(y0, y1) + 0.5) * res
        vz_ = self.zmin + (np.arange(z0, z1) + 0.5) * res
        vdx2 = (vx_ - cx) ** 2
        vdy2 = (vy_ - cy) ** 2
        vdz2 = (vz_ - cz) ** 2
        vdist_sq = (vdx2[:, None, None] + vdy2[None, :, None]
                    + vdz2[None, None, :])
        in_sphere = vdist_sq <= vol_r * vol_r
        # Columnar occlusion: for each column, find the first occupied
        # voxel below the candidate inside the sub-box, and mask
        # everything beneath it.
        occ_box = occupied_grid[x0:x1, y0:y1, z0:z1]
        col_first_occ = np.full((x1 - x0, y1 - y0), z1 - z0, dtype=np.intp)
        # iz relative to box z0; real iz = z0 + local_iz
        # Walk downward from candidate altitude
        local_iz_start = max(0, ciz + 1 - z0)
        for liz in range(local_iz_start, z1 - z0):
            m = occ_box[:, :, liz] & (col_first_occ == (z1 - z0))
            col_first_occ[m] = liz
        local_iz_arr = np.arange(z1 - z0)
        not_occ_vol = local_iz_arr[None, None, :] < col_first_occ[:, :, None]
        volumetric_unknown = int((vol_unk_box & in_sphere & not_occ_vol).sum())

        return frustum_gain, volumetric_unknown

    # ── Helper: 3-D Bresenham ray-march visibility check ─────────
    def _ray_march_visible(
        self,
        sample_ijs: np.ndarray,
        ox: float, oy: float, oz: float,
        occupied_grid: np.ndarray,
    ) -> int:
        """Count how many *sample_ijs* voxels are visible via 3-D ray march.

        A Bresenham-style ray is marched from ``(ox, oy, oz)`` (candidate
        grid position, float) to each target voxel.  If any occupied cell
        lies along the ray the target is considered occluded.

        Returns the number of *unoccluded* targets.
        """
        n_visible = 0
        occ = occupied_grid
        nx, ny, nz = self.nx, self.ny, self.nz

        for ti in range(len(sample_ijs)):
            tx = int(sample_ijs[ti, 0])
            ty = int(sample_ijs[ti, 1])
            tz = int(sample_ijs[ti, 2])
            rdx = tx - ox
            rdy = ty - oy
            rdz = tz - oz
            steps = max(abs(int(round(rdx))), abs(int(round(rdy))),
                        abs(int(round(rdz))), 1)
            inv_steps = 1.0 / steps
            sx = rdx * inv_steps
            sy = rdy * inv_steps
            sz = rdz * inv_steps

            occluded = False
            px, py, pz = ox, oy, oz
            for s in range(1, steps):
                px += sx; py += sy; pz += sz
                ix = int(px)
                iy = int(py)
                iz = int(pz)
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    if occ[ix, iy, iz]:
                        occluded = True
                        break
            if not occluded:
                n_visible += 1

        return n_visible

    def _nbv_score_candidate_raytrace(
        self,
        cand: np.ndarray,
        occupied_grid: np.ndarray,
        unknown: np.ndarray,
    ) -> tuple[int, int]:
        """Score *one* viewpoint using 3-D ray-traced visibility.

        For each unknown voxel within the LiDAR cone/range (frustum) or
        within ``nbv_volumetric_radius`` (volumetric sphere), a 3-D
        Bresenham-style ray is marched from the candidate to the voxel.
        If any occupied voxel lies along that ray the unknown voxel is
        considered occluded and is not counted.

        Both sets are independently downsampled to
        ``nbv_ray_max_targets`` for speed and then scaled back up.

        Returns
        -------
        frustum_gain : int
            Estimated *visible* unknown voxels inside the LiDAR cone.
        volumetric_unknown : int
            Estimated *visible* unknown voxels within
            ``nbv_volumetric_radius`` of the candidate.
        """
        cx, cy, cz = cand
        res = self.resolution
        tan_ha = np.tan(np.radians(self.nbv_sensor_half_angle))
        max_range = self.nbv_lidar_max_range
        max_range_sq = max_range * max_range

        # Candidate grid position (may be outside grid, that's OK)
        gx0 = (cx - self.xmin) / res
        gy0 = (cy - self.ymin) / res
        gz0 = (cz - self.zmin) / res

        # ── Collect all unknown voxels ────────────────────────────────
        unk_ijs = np.argwhere(unknown)  # (N, 3) grid indices
        if len(unk_ijs) == 0:
            return 0, 0

        # World centres of unknown voxels
        wx = self.xmin + (unk_ijs[:, 0] + 0.5) * res
        wy = self.ymin + (unk_ijs[:, 1] + 0.5) * res
        wz = self.zmin + (unk_ijs[:, 2] + 0.5) * res

        dx = wx - cx
        dy = wy - cy
        dz = wz - cz
        dist_sq = dx * dx + dy * dy + dz * dz

        # ── Frustum mask: cone + range ───────────────────────────────
        in_range = dist_sq <= max_range_sq
        dist_xy_sq = dx * dx + dy * dy
        cone_ok = dist_xy_sq <= (dz * tan_ha) ** 2
        below = dz > 0
        frustum_mask = in_range & cone_ok & below
        n_frustum_total = int(frustum_mask.sum())

        # ── Volumetric mask: sphere ──────────────────────────────────
        vol_r = self.nbv_volumetric_radius
        vol_mask = dist_sq <= vol_r * vol_r
        n_vol_total = int(vol_mask.sum())

        budget = self.nbv_ray_max_targets

        # ── Ray-trace frustum set ────────────────────────────────────
        if n_frustum_total == 0:
            frustum_gain = 0
        else:
            frustum_ijs = unk_ijs[frustum_mask]
            if len(frustum_ijs) > budget:
                chosen = np.random.choice(len(frustum_ijs), budget, replace=False)
                sample = frustum_ijs[chosen]
                scale = n_frustum_total / budget
            else:
                sample = frustum_ijs
                scale = 1.0
            n_vis = self._ray_march_visible(sample, gx0, gy0, gz0,
                                            occupied_grid)
            frustum_gain = int(round(n_vis * scale))

        # ── Ray-trace volumetric set ─────────────────────────────────
        if n_vol_total == 0:
            volumetric_unknown = 0
        else:
            vol_ijs = unk_ijs[vol_mask]
            if len(vol_ijs) > budget:
                chosen = np.random.choice(len(vol_ijs), budget, replace=False)
                sample = vol_ijs[chosen]
                scale = n_vol_total / budget
            else:
                sample = vol_ijs
                scale = 1.0
            n_vis = self._ray_march_visible(sample, gx0, gy0, gz0,
                                            occupied_grid)
            volumetric_unknown = int(round(n_vis * scale))

        return frustum_gain, volumetric_unknown

    def _nbv_select_target(
        self,
        current_pos: np.ndarray,
        occupied_grid: np.ndarray,
        free: np.ndarray,
        unknown: np.ndarray,
        is_frontier: np.ndarray,
        start: tuple[int, int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Select next target using heuristic-generated candidates + frustum scoring.

        1. Generate candidate viewpoints via ``_nbv_generate_candidates``.
           Candidates come from frontier cluster centroids at multiple
           altitudes, high-unknown columns, and random local samples.
        2. Filter candidates (occupied, too close, revisited).
        3. Score each surviving candidate with ``_nbv_score_candidate``
           (downward cone raycasting with per-candidate occlusion).
        4. Return the candidate with the highest gain / distance^exp.

        Parameters
        ----------
        current_pos, occupied_grid, free, unknown, is_frontier, start
            Same grids / masks built by ``next_target``.

        Returns
        -------
        target : ndarray (3,) or None
        evaluated_candidates : ndarray (M, 3) or None
            World-space positions of all candidates that passed filtering
            and were scored (for visualization).
        """
        candidates = self._nbv_generate_candidates(
            current_pos, occupied_grid, free, unknown, is_frontier, start)

        # Per-candidate debug records: list of dicts with scoring details
        debug_records: list[dict] = []

        if len(candidates) == 0:
            print("    [nbv] 0 candidates generated")
            self._last_nbv_debug = debug_records
            return None, None

        cur_xyz = current_pos[:3]
        best_score = -np.inf
        best_candidate = None
        best_idx = -1
        n_evaluated = 0
        evaluated_list: list[np.ndarray] = []

        for ci, cand in enumerate(candidates):
            dist = float(np.linalg.norm(cand - cur_xyz))

            if dist < self.min_target_distance:
                debug_records.append(dict(
                    pos=cand.copy(), dist=dist, frustum=0, vol=0,
                    combined_gain=0.0, score=0.0, status="too_close"))
                continue

            if self._waypoint_history:
                prev = np.array(self._waypoint_history)
                if np.any(np.linalg.norm(prev - cand, axis=1)
                          < self.waypoint_exclusion_radius):
                    debug_records.append(dict(
                        pos=cand.copy(), dist=dist, frustum=0, vol=0,
                        combined_gain=0.0, score=0.0, status="revisited"))
                    continue

            # Check that the candidate voxel (if inside grid) is not occupied
            if (self.xmin <= cand[0] <= self.xmax and
                    self.ymin <= cand[1] <= self.ymax and
                    self.zmin <= cand[2] <= self.zmax):
                gx, gy, gz = self._world_to_grid(*cand)
                if occupied_grid[gx, gy, gz]:
                    debug_records.append(dict(
                        pos=cand.copy(), dist=dist, frustum=0, vol=0,
                        combined_gain=0.0, score=0.0, status="occupied"))
                    continue

            frustum_gain, vol_unknown = self._nbv_score_candidate(
                cand, occupied_grid, unknown)
            # The frustum gain is the ray-traced count of unknown voxels
            # actually *visible* from this candidate position.  If it is
            # zero, the candidate cannot observe any new information
            # (unknowns are fully occluded by walls / obstacles), so it
            # must be skipped — the volumetric bonus must NOT rescue a
            # candidate with zero visibility.
            if frustum_gain <= 0:
                debug_records.append(dict(
                    pos=cand.copy(), dist=dist, frustum=frustum_gain,
                    vol=vol_unknown, combined_gain=0.0, score=0.0,
                    status="zero_frustum"))
                continue
            # Volumetric unknown density serves as a tiebreaker among
            # candidates that all have non-zero frustum visibility,
            # biasing toward large unexplored regions.
            combined_gain = (frustum_gain
                             + self.nbv_volumetric_weight * vol_unknown)

            score = combined_gain / max(dist, 0.01) ** self.distance_exponent
            n_evaluated += 1
            evaluated_list.append(cand.copy())

            debug_records.append(dict(
                pos=cand.copy(), dist=dist, frustum=frustum_gain,
                vol=vol_unknown, combined_gain=combined_gain,
                score=score, status="scored"))

            if score > best_score:
                best_score = score
                best_candidate = cand.copy()
                best_idx = len(debug_records) - 1

        # Mark the winning record
        if best_idx >= 0:
            debug_records[best_idx]["status"] = "selected"

        self._last_nbv_debug = debug_records

        evaluated_pts = (np.array(evaluated_list, dtype=np.float64)
                         if evaluated_list else None)

        if best_candidate is not None:
            best_rec = debug_records[best_idx]
            print(f"    [nbv] {len(candidates)} candidates, "
                  f"{n_evaluated} scored, best score={best_score:.1f} "
                  f"(frustum={best_rec['frustum']}, vol={best_rec['vol']})")
        else:
            print(f"    [nbv] {len(candidates)} candidates, "
                  f"{n_evaluated} scored, no valid target")

        if best_candidate is not None:
            self._waypoint_history.append(best_candidate.copy())
        return best_candidate, evaluated_pts

    # ── Random waypoint selection ─────────────────────────────────────────

    def _random_select_target(
        self,
        current_pos: np.ndarray,
        occupied_grid: np.ndarray,
        path_planner: "PathPlanner | None" = None,
    ) -> np.ndarray | None:
        """Select a random goal near obstacles that forces non-trivial planning.

        Uses the same strategy as ``sample_near_obstacle_goal`` from
        obstacleAvoidance.py: picks goals 2–5 m from occupied voxels that
        are traversable but whose straight line from the drone is blocked
        by obstacles, guaranteeing the path planner must route around them.

        Falls back to simple random sampling if no PathPlanner is available
        or no near-obstacle goal can be found.
        """
        cur_xyz = current_pos[:3]

        # ── Primary: near-obstacle goal via PathPlanner ───────────────
        if path_planner is not None and path_planner.occupied is not None:
            goal = sample_near_obstacle_goal(
                path_planner,
                cur_xyz,
                min_dist_from_obstacle=2.0,
                max_dist_from_obstacle=5.0,
                min_dist_from_drone=self.min_target_distance,
                max_attempts=self.random_max_attempts,
            )
            if goal is not None:
                # Check proximity to previous waypoints
                if self._waypoint_history:
                    prev = np.array(self._waypoint_history)
                    if np.any(np.linalg.norm(prev - goal, axis=1)
                              < self.waypoint_exclusion_radius):
                        goal = None
            if goal is not None:
                self._waypoint_history.append(goal.copy())
                print(f"    [random] near-obstacle goal selected")
                return goal
            print(f"    [random] near-obstacle sampling failed, "
                  f"falling back to free-space random")

        # ── Fallback: simple random in observed free space ────────────
        margin = self.inflation_margin
        inflate_r = max(1, int(np.ceil(margin / self.resolution)))
        struct = ndimage.generate_binary_structure(3, 1)
        inflated = ndimage.binary_dilation(
            occupied_grid, structure=struct, iterations=inflate_r)

        for attempt in range(self.random_max_attempts):
            x = np.random.uniform(self.xmin + margin, self.xmax - margin)
            y = np.random.uniform(self.ymin + margin, self.ymax - margin)
            z = np.random.uniform(self.zmin + margin, self.zmax - margin)
            cand = np.array([x, y, z])

            gx, gy, gz = self._world_to_grid(x, y, z)
            if not self._observed[gx, gy, gz]:
                continue
            if inflated[gx, gy, gz]:
                continue

            dist = float(np.linalg.norm(cand - cur_xyz))
            if dist < self.min_target_distance:
                continue

            if self._waypoint_history:
                prev = np.array(self._waypoint_history)
                if np.any(np.linalg.norm(prev - cand, axis=1)
                          < self.waypoint_exclusion_radius):
                    continue

            self._waypoint_history.append(cand.copy())
            print(f"    [random] free-space fallback after "
                  f"{attempt + 1}/{self.random_max_attempts} attempts")
            return cand

        print(f"    [random] failed to find valid waypoint after "
              f"{self.random_max_attempts} attempts")
        return None

    # ── Core planner ──────────────────────────────────────────────────────

    def next_target(
        self,
        pipeline: SLAMPipeline,
        current_pos,
    ) -> tuple[np.ndarray | None, dict]:
        """Determine the next waypoint via Wavefront Frontier Detection.

        Parameters
        ----------
        pipeline : SLAMPipeline
            The live SLAM pipeline whose map and pose history are queried.
        current_pos : array-like, shape (3,)
            Drone's current NED position ``[x, y, z]``.

        Returns
        -------
        target : ndarray shape (3,) **or** ``None``
            Next waypoint in NED.  ``None`` when no reachable frontiers remain.
        info : dict
            ``n_frontier_cells``, ``n_frontier_cells_raw``,
            ``n_occupied_cells``, ``n_clusters``, ``frontier_world_pts``.
        """
        current_pos = np.asarray(current_pos, dtype=float)

        # 1) Observed grid is updated continuously via feed_scan() on the
        #    collection thread — no batch raycast needed here.

        # 2) Build 3-D occupied grid from SLAM map points ────────────────
        occupied_grid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        vis_pts = pipeline.get_map_points()

        info: dict = {
            "n_frontier_cells": 0,
            "n_frontier_cells_raw": 0,
            "n_occupied_cells": 0,
            "n_clusters": 0,
            "frontier_world_pts": np.empty((0, 3), dtype=np.float64),
        }

        if len(vis_pts) == 0:
            return None, info          # no map yet — nothing to explore

        ix = np.clip(
            ((vis_pts[:, 0] - self.xmin) / self.resolution).astype(int),
            0, self.nx - 1,
        )
        iy = np.clip(
            ((vis_pts[:, 1] - self.ymin) / self.resolution).astype(int),
            0, self.ny - 1,
        )
        iz = np.clip(
            ((vis_pts[:, 2] - self.zmin) / self.resolution).astype(int),
            0, self.nz - 1,
        )
        occupied_grid[ix, iy, iz] = True
        self._last_occupied_grid = occupied_grid
        info["n_occupied_cells"] = int(occupied_grid.sum())

        # 3) Derive free / unknown / frontier masks ──────────────────────
        free = self._observed & ~occupied_grid
        unknown = ~self._observed

        n_observed = int(self._observed.sum())
        n_free = int(free.sum())
        n_unknown = int(unknown.sum())
        n_total = self.nx * self.ny * self.nz
        print(f"    [debug] scans raycasted: {self._n_scans_raycasted} | "
              f"observed: {n_observed}/{n_total} | "
              f"free: {n_free} | occupied: {info['n_occupied_cells']} | "
              f"unknown: {n_unknown}")

        # Frontier = free voxels with ≥ 1 unknown 6-connected neighbour
        struct = ndimage.generate_binary_structure(3, 1)   # 6-connected
        unknown_adjacent = ndimage.binary_dilation(unknown, structure=struct)
        is_frontier = free & unknown_adjacent

        # Ensure a small neighbourhood around the drone is observed+free
        # so the BFS can always connect outward to the scanned region.
        # The drone is physically present, so a few-voxel radius is safe.
        start = self._world_to_grid(*current_pos[:3])
        sx, sy, sz = start
        _SEED_R = 2  # voxel radius around drone to force free
        for dx in range(-_SEED_R, _SEED_R + 1):
            for dy in range(-_SEED_R, _SEED_R + 1):
                for dz in range(-_SEED_R, _SEED_R + 1):
                    nx_, ny_, nz_ = sx + dx, sy + dy, sz + dz
                    if (0 <= nx_ < self.nx and 0 <= ny_ < self.ny
                            and 0 <= nz_ < self.nz):
                        self._observed[nx_, ny_, nz_] = True
                        occupied_grid[nx_, ny_, nz_] = False
                        free[nx_, ny_, nz_] = True
        # Recompute frontier mask after seeding (new free cells may be
        # adjacent to unknown voxels)
        unknown = ~self._observed
        unknown_adjacent = ndimage.binary_dilation(unknown, structure=struct)
        is_frontier = free & unknown_adjacent

        # ── Branch: Random, NBV, or WFD ────────────────────────────────
        if self.use_random:
            # ── 4r) Random waypoint — uses PathPlanner if available ──
            best_target = self._random_select_target(
                current_pos, occupied_grid, self._path_planner)

            # Still populate frontier overlay for the viewer
            frt_ijs = np.argwhere(is_frontier)
            if len(frt_ijs) > 0:
                frontier_world = np.column_stack([
                    self.xmin + (frt_ijs[:, 0] + 0.5) * self.resolution,
                    self.ymin + (frt_ijs[:, 1] + 0.5) * self.resolution,
                    self.zmin + (frt_ijs[:, 2] + 0.5) * self.resolution,
                ])
                info["frontier_world_pts"] = frontier_world
                info["n_frontier_cells"] = len(frt_ijs)
                info["n_frontier_cells_raw"] = len(frt_ijs)

            return best_target, info

        if self.use_nbv:
            # ── 4a) NBV — frustum-based viewpoint scoring ────────────
            best_target, evaluated_candidates = self._nbv_select_target(
                current_pos, occupied_grid, free, unknown,
                is_frontier, start)

            # Store evaluated candidate positions for viewer overlay
            if evaluated_candidates is not None:
                info["candidate_world_pts"] = evaluated_candidates

            # Attach per-candidate debug records for _NBVDebugPlot
            info["nbv_debug"] = getattr(self, "_last_nbv_debug", [])
            info["nbv_volumetric_weight"] = self.nbv_volumetric_weight
            info["nbv_distance_exponent"] = self.distance_exponent

            # Populate frontier overlay from the frontier mask so the
            # viewer still shows frontier voxels even in NBV mode.
            frt_ijs = np.argwhere(is_frontier)  # (N, 3)
            if len(frt_ijs) > 0:
                frontier_world = np.column_stack([
                    self.xmin + (frt_ijs[:, 0] + 0.5) * self.resolution,
                    self.ymin + (frt_ijs[:, 1] + 0.5) * self.resolution,
                    self.zmin + (frt_ijs[:, 2] + 0.5) * self.resolution,
                ])
                info["frontier_world_pts"] = frontier_world
                info["n_frontier_cells"] = len(frt_ijs)
                info["n_frontier_cells_raw"] = len(frt_ijs)

            return best_target, info

        # ── 4b) Wavefront Frontier Detection BFS (original) ──────────
        all_clusters = self._wfd_bfs(start, free, is_frontier)

        # Raw count (all clusters, any size)
        n_raw = sum(len(c) for c in all_clusters)
        info["n_frontier_cells_raw"] = n_raw

        # Filter by minimum cluster size
        valid_clusters = [c for c in all_clusters if len(c) >= self.min_frontier_size]
        n_valid = sum(len(c) for c in valid_clusters)
        info["n_frontier_cells"] = n_valid
        info["n_clusters"] = len(valid_clusters)

        # Build world coordinates for viewer overlay
        if n_valid > 0:
            all_cells = np.vstack(valid_clusters)
            frontier_world = np.column_stack([
                self.xmin + (all_cells[:, 0] + 0.5) * self.resolution,
                self.ymin + (all_cells[:, 1] + 0.5) * self.resolution,
                self.zmin + (all_cells[:, 2] + 0.5) * self.resolution,
            ])
            info["frontier_world_pts"] = frontier_world

        if n_valid == 0:
            return None, info

        # 5) Score frontier clusters ──────────────────────────────────────
        #    gain  = unknown voxels in a box around the frontier centroid
        #            (estimates how much NEW space will be revealed, not
        #            how much boundary we already know about)
        #    cost  = distance^exponent  (exponent < 1 softens the bias
        #            toward nearby frontiers so the drone pushes outward)
        #    The waypoint is offset upward (more negative Z in NED) so the
        #    downward-facing LiDAR can scan the frontier region from above.
        cur_xyz = current_pos[:3]
        best_score = -np.inf
        best_target = None
        r = self.unknown_gain_radius
        # Safe Z clamp: keep waypoints well inside bounds so OMPL doesn't
        # reject them due to the inflated obstacle boundary.
        z_lo = self.zmin + self.inflation_margin
        z_hi = self.zmax - self.inflation_margin

        for cells in valid_clusters:
            cx_mean = float(cells[:, 0].mean())
            cy_mean = float(cells[:, 1].mean())
            cz_mean = float(cells[:, 2].mean())
            wx, wy, wz_frontier = self._grid_to_world(cx_mean, cy_mean, cz_mean)

            # Offset waypoint above the frontier for downward LiDAR coverage
            wz = wz_frontier - self.lidar_altitude_offset

            # Keep target inside safe bounds
            wx = float(np.clip(wx, self.xmin + self.inflation_margin, self.xmax - self.inflation_margin))
            wy = float(np.clip(wy, self.ymin + self.inflation_margin, self.ymax - self.inflation_margin))
            wz = float(np.clip(wz, z_lo, z_hi))

            # Reject if the target voxel itself is occupied
            tix, tiy, tiz = self._world_to_grid(wx, wy, wz)
            if occupied_grid[tix, tiy, tiz]:
                continue

            # Reject if too close to any previous waypoint
            candidate = np.array([wx, wy, wz])
            if self._waypoint_history:
                prev = np.array(self._waypoint_history)
                if np.any(np.linalg.norm(prev - candidate, axis=1)
                          < self.waypoint_exclusion_radius):
                    continue

            # Reject if target is too close to the drone's current position
            # (prevents picking "already here" waypoints)
            dist = max(float(np.linalg.norm(candidate - cur_xyz)), 0.01)
            if dist < self.min_target_distance:
                continue

            # Count unknown voxels near the frontier centroid — this
            # measures exploration potential (how much new space is behind
            # this frontier) rather than the raw frontier size.
            gx, gy, gz = self._world_to_grid(wx, wy, wz_frontier)
            x0, x1 = max(0, gx - r), min(self.nx, gx + r + 1)
            y0, y1 = max(0, gy - r), min(self.ny, gy + r + 1)
            z0, z1 = max(0, gz - r), min(self.nz, gz + r + 1)
            unknown_nearby = int(unknown[x0:x1, y0:y1, z0:z1].sum())

            gain = float(unknown_nearby)
            score = gain / (dist ** self.distance_exponent)

            if score > best_score:
                best_score = score
                best_target = candidate

        # Fallback: nearest frontier voxel if all cluster centroids rejected
        if best_target is None and n_valid > 0:
            all_cells = np.vstack(valid_clusters)
            fw = np.column_stack([
                self.xmin + (all_cells[:, 0] + 0.5) * self.resolution,
                self.ymin + (all_cells[:, 1] + 0.5) * self.resolution,
                self.zmin + (all_cells[:, 2] + 0.5) * self.resolution,
            ])
            # Offset Z for LiDAR coverage (fly above the frontier)
            fw[:, 2] -= self.lidar_altitude_offset
            fw[:, 0] = np.clip(fw[:, 0], self.xmin + self.inflation_margin, self.xmax - self.inflation_margin)
            fw[:, 1] = np.clip(fw[:, 1], self.ymin + self.inflation_margin, self.ymax - self.inflation_margin)
            fw[:, 2] = np.clip(fw[:, 2], self.zmin + self.inflation_margin, self.zmax - self.inflation_margin)
            # Filter out frontier voxels too close to previous waypoints
            if self._waypoint_history:
                prev = np.array(self._waypoint_history)
                keep = np.ones(len(fw), dtype=bool)
                for wp in prev:
                    keep &= np.linalg.norm(fw - wp, axis=1) >= self.waypoint_exclusion_radius
                fw = fw[keep]
            # Filter out points too close to the drone
            if len(fw) > 0:
                fb_dists = np.linalg.norm(fw - cur_xyz.reshape(1, 3), axis=1)
                fw = fw[fb_dists >= self.min_target_distance]
            if len(fw) > 0:
                dists = np.linalg.norm(fw - cur_xyz.reshape(1, 3), axis=1)
                nearest = fw[np.argmin(dists)]
                best_target = np.array([nearest[0], nearest[1], nearest[2]])

        # Record chosen waypoint in history
        if best_target is not None:
            self._waypoint_history.append(best_target.copy())

        return best_target, info


# ══════════════════════════════════════════════════════════════════════════════
# _NBVDebugPlot — live candidate-comparison dashboard
# ══════════════════════════════════════════════════════════════════════════════

class _NBVDebugPlot:
    """Live matplotlib dashboard showing *why* each NBV candidate was
    selected or rejected.

    Three panels:

    1. **Left (large) — Top-down spatial map**
       A top-down (X–Y) projection of the SLAM map points rendered as a
       grey density background.  Candidate viewpoints are overlaid as
       circles whose **size** is proportional to how preferable they are
       (final score).  Lines are drawn from the drone to each scored
       candidate.  Filtered-out candidates are shown as small faded
       markers.  The selected winner gets a bright lime star.

    2. **Top-right — Gain components (stacked bar)**
       Frustum vs weighted-volumetric gain for each scored candidate.

    3. **Bottom-right — Frustum gain vs Distance (scatter)**
       Scored candidates at (distance, frustum_gain) coloured by score.
    """

    _STATUS_COLORS = {
        "scored":       "#1f77b4",   # mpl blue
        "selected":     "#2ca02c",   # green
        "too_close":    "#d62728",   # red
        "revisited":    "#ff7f0e",   # orange
        "occupied":     "#7f7f7f",   # grey
        "zero_frustum": "#9467bd",   # purple
    }

    def __init__(self):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
        except Exception:
            self._ok = False
            return

        self._ok = True
        self._plt = plt
        plt.ion()
        # Layout: left half = spatial map (spans both rows),
        #         right column = gain bars (top) + scatter (bottom)
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
        self.ax_map   = self.fig.add_subplot(gs[:, 0])   # left, full height
        self.ax_stack = self.fig.add_subplot(gs[0, 1])    # top-right
        self.ax_scat  = self.fig.add_subplot(gs[1, 1])    # bottom-right
        self.fig.suptitle("NBV Candidate Debug", fontsize=14)

        for ax in [self.ax_map, self.ax_stack, self.ax_scat]:
            ax.grid(True, alpha=0.3)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.01)

        self._iter = 0
        self._cbar = None  # track colorbar so we can remove on redraw

    # ──────────────────────────────────────────────────────────────────
    def update(
        self,
        info: dict,
        drone_pos: np.ndarray | None = None,
        map_points: np.ndarray | None = None,
    ) -> None:
        """Redraw the dashboard from the latest ``info`` dict.

        Parameters
        ----------
        info : dict
            The info dict returned by ``ExplorationPlanner.next_target``.
            Must contain ``nbv_debug`` (list of per-candidate dicts).
        drone_pos : ndarray (3,), optional
            Current drone NED position.
        map_points : ndarray (N, 3), optional
            SLAM map points (NED) used as a grey background projection.
        """
        if not self._ok:
            return
        records = info.get("nbv_debug", [])
        if not records:
            return

        self._iter += 1
        plt = self._plt

        # ── Separate scored vs filtered records ───────────────────────
        scored   = [r for r in records if r["status"] in ("scored", "selected")]
        filtered = [r for r in records if r["status"] not in ("scored", "selected")]
        selected = [r for r in records if r["status"] == "selected"]

        # Sort scored by score descending
        scored.sort(key=lambda r: r["score"], reverse=True)

        # ── Hypothetical winners: frustum-only and volume-only ────────
        # Compute what the selection *would* have been if only one
        # component were used, so you can see which term is pulling.
        dist_exp = info.get("nbv_distance_exponent", 1.0)
        frustum_only_winner = None
        vol_only_winner = None
        if scored:
            best_f_score = -np.inf
            best_v_score = -np.inf
            for r in scored:
                d = max(r["dist"], 0.01)
                f_score = r["frustum"] / d ** dist_exp
                v_score = r["vol"] / d ** dist_exp
                if f_score > best_f_score:
                    best_f_score = f_score
                    frustum_only_winner = r
                if v_score > best_v_score:
                    best_v_score = v_score
                    vol_only_winner = r

        # ── [left] Top-down spatial map with SLAM background ──────────
        ax = self.ax_map
        ax.clear()

        # Background: SLAM map points projected to XY
        if map_points is not None and len(map_points) > 0:
            # Subsample for performance if very large
            pts = np.asarray(map_points, dtype=np.float32)
            if len(pts) > 60_000:
                idx = np.random.choice(len(pts), 60_000, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, 1], pts[:, 0], c="#c0c0c0", s=0.4,
                       alpha=0.30, rasterized=True, zorder=1)

        # Frontier overlay (light blue)
        frt = info.get("frontier_world_pts")
        if frt is not None and len(frt) > 0:
            frt = np.asarray(frt)
            ax.scatter(frt[:, 1], frt[:, 0], c="#87ceeb", s=1.5,
                       alpha=0.35, rasterized=True, zorder=2, label="frontier")

        # Compute score range for marker sizing (scored candidates)
        if scored:
            all_scores = np.array([r["score"] for r in scored])
            s_min, s_max = all_scores.min(), all_scores.max()
            s_range = s_max - s_min if s_max > s_min else 1.0
            MIN_SIZE, MAX_SIZE = 30, 350

        # Filtered candidates: small faded dots by status
        for status, color in self._STATUS_COLORS.items():
            if status in ("scored", "selected"):
                continue
            pts_f = [r["pos"] for r in records if r["status"] == status]
            if not pts_f:
                continue
            pts_f = np.array(pts_f)
            ax.scatter(pts_f[:, 1], pts_f[:, 0],
                       c=color, s=15, alpha=0.30, edgecolors="none",
                       label=status, zorder=3)

        # Scored candidates: size ∝ score, lines from drone
        if scored:
            for r in scored:
                sz = MIN_SIZE + (r["score"] - s_min) / s_range * (MAX_SIZE - MIN_SIZE)
                color = self._STATUS_COLORS[r["status"]]
                pos = r["pos"]
                ax.scatter([pos[1]], [pos[0]], c=color, s=sz,
                           edgecolors="k", linewidths=0.5, alpha=0.85,
                           zorder=5)
                # Line from drone to candidate
                if drone_pos is not None:
                    ax.plot([drone_pos[1], pos[1]], [drone_pos[0], pos[0]],
                            color="#888888", linewidth=0.4, alpha=0.4,
                            zorder=4)

        # Winner star
        if selected:
            s = selected[0]["pos"]
            ax.scatter([s[1]], [s[0]], marker="*", s=400, c="lime",
                       edgecolors="k", linewidths=1.3, zorder=7,
                       label="selected")
            # Bold line from drone to winner
            if drone_pos is not None:
                ax.plot([drone_pos[1], s[1]], [drone_pos[0], s[0]],
                        color="lime", linewidth=1.5, alpha=0.7, zorder=6)

        # Hypothetical winners: frustum-only (cyan diamond) & vol-only (orange diamond)
        if frustum_only_winner is not None:
            fp = frustum_only_winner["pos"]
            ax.scatter([fp[1]], [fp[0]], marker="D", s=180, c="cyan",
                       edgecolors="k", linewidths=1.0, zorder=9,
                       label="frustum-only pick")
            if drone_pos is not None:
                ax.plot([drone_pos[1], fp[1]], [drone_pos[0], fp[0]],
                        color="cyan", linewidth=1.0, linestyle="--",
                        alpha=0.6, zorder=6)
        if vol_only_winner is not None:
            vp = vol_only_winner["pos"]
            ax.scatter([vp[1]], [vp[0]], marker="D", s=180, c="#ff7f0e",
                       edgecolors="k", linewidths=1.0, zorder=9,
                       label="vol-only pick")
            if drone_pos is not None:
                ax.plot([drone_pos[1], vp[1]], [drone_pos[0], vp[0]],
                        color="#ff7f0e", linewidth=1.0, linestyle="--",
                        alpha=0.6, zorder=6)

        # Drone marker
        if drone_pos is not None:
            ax.scatter([drone_pos[1]], [drone_pos[0]], marker="P", s=200,
                       c="white", edgecolors="k", linewidths=1.5,
                       zorder=8, label="drone")

        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.set_aspect("equal", adjustable="datalim")
        # Filter summary annotation
        from collections import Counter
        filt_counts = Counter(r["status"] for r in filtered)
        filt_str = ", ".join(f"{v} {k}" for k, v in filt_counts.items())
        ax.set_title(f"Candidates on Map  (iter {self._iter})\n"
                     f"filtered: {filt_str if filt_str else 'none'}",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper left", ncol=2,
                  markerscale=0.6, handletextpad=0.3)

        # ── [top-right] Stacked gain components ──────────────────────
        ax = self.ax_stack
        ax.clear()
        if scored:
            labels = [f"({r['pos'][0]:.0f},{r['pos'][1]:.0f},{r['pos'][2]:.0f})"
                      for r in scored]
            frustums = np.array([r["frustum"] for r in scored], dtype=float)
            vols     = np.array([r["vol"] for r in scored], dtype=float)
            w = info.get("nbv_volumetric_weight", 0.0)
            weighted_vols = vols * w if w else vols
            y_pos = np.arange(len(scored))
            ax.barh(y_pos, frustums, color="#1f77b4",
                    label="Frustum gain", edgecolor="k", linewidth=0.3)
            ax.barh(y_pos, weighted_vols, left=frustums, color="#ff7f0e",
                    label=f"Vol ×{w:.2f}" if w else "Vol (unweighted)",
                    edgecolor="k", linewidth=0.3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Gain (voxels)")
            ax.legend(fontsize=8, loc="lower right")

            # Mark hypothetical winners on the bar chart
            labels_list = [f"({r['pos'][0]:.0f},{r['pos'][1]:.0f},{r['pos'][2]:.0f})"
                           for r in scored]
            if frustum_only_winner is not None:
                fow_label = (f"({frustum_only_winner['pos'][0]:.0f},"
                             f"{frustum_only_winner['pos'][1]:.0f},"
                             f"{frustum_only_winner['pos'][2]:.0f})")
                if fow_label in labels_list:
                    fi = labels_list.index(fow_label)
                    ax.annotate("\u25C6 frustum-only", xy=(frustums[fi], fi),
                                xytext=(frustums[fi] + frustums.max() * 0.05, fi),
                                fontsize=7, color="cyan", fontweight="bold",
                                va="center")
            if vol_only_winner is not None:
                vow_label = (f"({vol_only_winner['pos'][0]:.0f},"
                             f"{vol_only_winner['pos'][1]:.0f},"
                             f"{vol_only_winner['pos'][2]:.0f})")
                if vow_label in labels_list:
                    vi = labels_list.index(vow_label)
                    bar_end = frustums[vi] + (weighted_vols[vi] if len(weighted_vols) > vi else 0)
                    ax.annotate("\u25C6 vol-only", xy=(bar_end, vi),
                                xytext=(bar_end + frustums.max() * 0.05, vi),
                                fontsize=7, color="#ff7f0e", fontweight="bold",
                                va="center")
        ax.set_title("Gain Components", fontsize=10)
        ax.grid(True, alpha=0.3)

        # ── [bottom-right] Frustum gain vs distance scatter ──────────
        ax = self.ax_scat
        ax.clear()
        if scored:
            dists    = [r["dist"] for r in scored]
            frustums = [r["frustum"] for r in scored]
            sc_vals  = [r["score"] for r in scored]
            sp = ax.scatter(dists, frustums, c=sc_vals, cmap="viridis",
                            edgecolors="k", linewidths=0.4, s=50)
            if selected:
                s = selected[0]
                ax.scatter([s["dist"]], [s["frustum"]], marker="*",
                           s=250, c="lime", edgecolors="k", linewidths=1,
                           zorder=5, label="selected")
                ax.legend(fontsize=8)
            try:
                if self._cbar is not None:
                    self._cbar.remove()
                self._cbar = self.fig.colorbar(
                    sp, ax=ax, pad=0.02, aspect=30, label="score")
            except Exception:
                pass
        ax.set_xlabel("Distance to drone (m)")
        ax.set_ylabel("Frustum gain (visible unknown voxels)")
        ax.set_title("Frustum Gain vs Distance", fontsize=10)
        ax.grid(True, alpha=0.3)

        # ── Redraw ────────────────────────────────────────────────────
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path: str) -> None:
        """Save the current figure to *path*."""
        if not self._ok:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SAVE_DIR = os.path.join(os.path.dirname(__file__), "flight_recordings")

# ── Mode selection ────────────────────────────────────────────────────────
# Set REPLAY_DIR to a recording directory (or parent) to run offline on
# saved LiDAR data.  Leave empty ("") for live AirSim flight.

REPLAY_DIR      = "flight_recordings"            # e.g. "flight_recordings/flight_1771909992"
#REPLAY_DIR      = ""  
# ── Exploration parameters (shared by both modes) ────────────────────────
EXPLORE_BOUNDS  = (-13, 27, -35, 5, -14, 0.15)   # (xmin,xmax,ymin,ymax,zmin,zmax) NED
TAKEOFF_HEIGHT  = EXPLORE_BOUNDS[4] -5         
VELOCITY        = 3             # m/s (live mode only)
SCAN_HZ         = 1 / 1.5      # scans per second (live mode only)
PLANNER_RES     = 1.0           # planning grid voxel size (m)
MAX_TARGETS     = 10000            # safety cap on autonomous waypoints
SCAN_DELAY      = 3             # register a scan only after N newer scans (live mode)
PLAN_EVERY      = 3             # run planner every N frames (replay mode)
FRAME_SKIP      = 1             # process every Nth frame (replay mode, 1 = all)

# ── Path-planning parameters (live mode) ─────────────────────────────────
INFLATION_RADIUS    = 1.5       # safety margin (m) inflated around obstacles
ABOVE_GRID_MARGIN   = 10.0      # metres of free airspace above the grid ceiling (Z)
PATH_PLANNER_TYPE   = "ABITstar"
PATH_SOLVE_TIMEOUT  = 2.0       # seconds per OMPL solve
FLIGHT_MODE         = "velocity" # "path" (moveOnPathAsync) or "velocity" (pure-pursuit)
POLL_HZ             = 20.0
MIN_WP_SPACING      = 1.0

# ── Exploration scoring (live mode) ──────────────────────────────────────
UNKNOWN_GAIN_RADIUS = 3         # voxels: box radius for counting unknown neighbours
DISTANCE_EXPONENT   = 1.0       # softer distance penalty (1.0 = original linear)
LIDAR_ALT_OFFSET    = 2.0       # fly this many metres above frontier centroid (NED)
MIN_TARGET_DIST     = 3.0       # skip targets closer than this (m) to avoid re-visiting
WP_EXCLUSION_RADIUS = 3.0       # avoid re-selecting waypoints within this radius (m)

# ── Target selection algorithm ───────────────────────────────────────────
# Set USE_RANDOM = True for random waypoint selection (baseline comparison).
# Set USE_NBV = True to use Next-Best-View frustum scoring instead of WFD
# frontier-centroid scoring.  NBV generates candidate viewpoints on a grid
# at cruise altitude and picks the one that maximises the number of unknown
# voxels visible through a simulated downward-facing LiDAR cone.
# Priority: USE_RANDOM > USE_NBV > WFD (only the first enabled mode runs).
USE_RANDOM              = False
RANDOM_MAX_ATTEMPTS     = 50
USE_NBV                 = True
NBV_SENSOR_HALF_ANGLE   = 45.0  # half-angle (deg) of the LiDAR cone
NBV_CRUISE_ALTITUDE     = TAKEOFF_HEIGHT  # one of the altitude options for candidates (NED)
NBV_N_UNKNOWN_COLUMNS   = 10    # top-K columns by unknown-voxel count to sample
NBV_N_LOCAL_SAMPLES     = 15    # random free-space samples near the drone
NBV_LOCAL_RADIUS        = 10.0  # world-space radius for local samples (m)
NBV_LIDAR_MAX_RANGE     = 40.0  # LiDAR max range (m); voxels beyond this are not scored
NBV_UNKNOWN_BLOCK_SIZE  = 4     # coarse block size (voxels) for unknown-region heuristic
NBV_N_UNKNOWN_BLOCKS    = 8     # top-K densest unknown blocks to generate candidates from
NBV_VOLUMETRIC_RADIUS   = 8.0   # radius (m) for volumetric unknown density bonus
NBV_VOLUMETRIC_WEIGHT   = 0.0   # weight of volumetric unknown term vs. frustum gain
NBV_RAY_MAX_TARGETS     = 500   # max unknown voxels to ray-trace per candidate (downsample budget)
NBV_USE_RAY_TRACING     = True  # True = 3D ray tracing, False = fast columnar occlusion
NBV_SHOW_CANDIDATES     = True  # show evaluated candidate positions as magenta points in viewer

# ══════════════════════════════════════════════════════════════════════════════
# Replay mode — offline exploration on saved LiDAR data
# ══════════════════════════════════════════════════════════════════════════════

def run_replay(recording_dir: str):
    """Replay saved flight data through the SLAM pipeline + exploration planner.

    Loads frame_*.npz files, feeds each through the SLAM pipeline, and runs
    the WFD planner periodically so you can see frontier detection and target
    selection in the Open3D viewer without a running simulator.
    """
    recording_dir = resolve_recording_dir(recording_dir)
    all_frames = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    if not all_frames:
        print(f"No frame_*.npz files found in {recording_dir}")
        return

    frames = all_frames[::FRAME_SKIP]
    print(f"Replaying {len(frames)} frames from {recording_dir}")
    print(f"  (available: {len(all_frames)}, skip={FRAME_SKIP})")

    # ── Set up SLAM pipeline + viewer ────────────────────────────────────
    cfg = SLAMConfig(
        registration="vgicp",
        octo_resolution=0.15,
        frame_skip=1,
        enable_viewer=True,
    )
    pipeline = SLAMPipeline(cfg)
    pipeline.start_viewer()

    # ── Set up planner ───────────────────────────────────────────────────
    planner = ExplorationPlanner(
        bounds=EXPLORE_BOUNDS,
        resolution=PLANNER_RES,
        min_frontier_size=3,
        use_random=USE_RANDOM,
        random_max_attempts=RANDOM_MAX_ATTEMPTS,
        use_nbv=USE_NBV,
        nbv_sensor_half_angle=NBV_SENSOR_HALF_ANGLE,
        nbv_cruise_altitude=NBV_CRUISE_ALTITUDE,
        nbv_n_unknown_columns=NBV_N_UNKNOWN_COLUMNS,
        nbv_n_local_samples=NBV_N_LOCAL_SAMPLES,
        nbv_local_radius=NBV_LOCAL_RADIUS,
        nbv_unknown_block_size=NBV_UNKNOWN_BLOCK_SIZE,
        nbv_n_unknown_blocks=NBV_N_UNKNOWN_BLOCKS,
        nbv_volumetric_radius=NBV_VOLUMETRIC_RADIUS,
        nbv_volumetric_weight=NBV_VOLUMETRIC_WEIGHT,
        nbv_ray_max_targets=NBV_RAY_MAX_TARGETS,
        nbv_use_ray_tracing=NBV_USE_RAY_TRACING,
    )

    strategy = "random" if USE_RANDOM else ("NBV" if USE_NBV else "WFD")

    # NBV candidate comparison dashboard (replay mode)
    nbv_debug_plot = _NBVDebugPlot() if USE_NBV else None

    print(f"\n{'='*60}")
    print(f"Exploration planner replay  (strategy: {strategy})")
    print(f"  Bounds: x=[{EXPLORE_BOUNDS[0]}, {EXPLORE_BOUNDS[1]}], "
          f"y=[{EXPLORE_BOUNDS[2]}, {EXPLORE_BOUNDS[3]}], "
          f"z=[{EXPLORE_BOUNDS[4]}, {EXPLORE_BOUNDS[5]}]")
    print(f"  Grid:  {planner.nx} x {planner.ny} x {planner.nz} voxels @ {PLANNER_RES} m")
    print(f"  Observation: raycasted from LiDAR  |  Plan every {PLAN_EVERY} frames")
    print(f"{'='*60}\n")

    wp_count = 0
    targets_chosen: list[np.ndarray] = []

    for i, path in enumerate(frames):
        t_load = time.perf_counter()
        data = np.load(path)
        pts = data["points"]
        pos = data["position"] if "position" in data.files else np.zeros(3)
        ori = (data["orientation"] if "orientation" in data.files
               else np.array([1, 0, 0, 0], dtype=float))
        lp = data["lidar_position"] if "lidar_position" in data.files else None
        lo = data["lidar_orientation"] if "lidar_orientation" in data.files else None
        gps = data["gps"] if "gps" in data.files else None

        # ── Feed frame to SLAM pipeline ──────────────────────────────────
        pipeline.set_drone_pos(pos)
        result = pipeline.process_frame(
            pts, pos, ori,
            gps=gps,
            lidar_position=lp,
            lidar_orientation=lo,
            frame_label=i,
        )

        # ── Feed scan to planner for raycasting ─────────────────────────
        valid_pts = filter_valid(pts)
        if len(valid_pts) > 0:
            lp_arr = lp if lp is not None else np.zeros(3)
            lo_arr = lo if lo is not None else np.array([1, 0, 0, 0], dtype=float)
            R_l = Rotation.from_quat(
                [lo_arr[1], lo_arr[2], lo_arr[3], lo_arr[0]]).as_matrix()
            body = (R_l @ valid_pts.T).T + lp_arr

            ori_arr = np.asarray(ori, dtype=float)
            R_b = Rotation.from_quat(
                [ori_arr[1], ori_arr[2], ori_arr[3], ori_arr[0]]).as_matrix()
            world_pts = (R_b @ body.T).T + np.asarray(pos, dtype=float)
            planner.feed_scan(pos.copy(), world_pts.astype(np.float32))

        # ── Run planner periodically ─────────────────────────────────────
        if (i + 1) % PLAN_EVERY == 0 or i == len(frames) - 1:
            current_pos = np.asarray(pos, dtype=float)
            t_plan = time.perf_counter()
            target, info = planner.next_target(pipeline, current_pos)
            dt_plan = time.perf_counter() - t_plan

            # Update NBV debug dashboard
            if nbv_debug_plot is not None and "nbv_debug" in info:
                nbv_debug_plot.update(
                    info, drone_pos=current_pos,
                    map_points=pipeline.get_map_points())

            # Update viewer overlays
            pipeline.set_frontier_points(info.get("frontier_world_pts"))

            print(f"  [frame {i+1:03d}/{len(frames)}] "
                  f"Occupied: {info['n_occupied_cells']} | "
                  f"Frontiers: {info['n_frontier_cells']}/{info['n_frontier_cells_raw']} "
                  f"in {info['n_clusters']} clusters | "
                  f"plan: {dt_plan*1e3:.0f}ms")

            if target is not None:
                wp_count += 1
                targets_chosen.append(target.copy())
                pipeline.set_target_pos(target.tolist())
                print(f"       -> target #{wp_count}: "
                      f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
            else:
                print(f"       -> no frontiers remain")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Replay complete — {len(frames)} frames processed, "
          f"{wp_count} targets selected")
    if targets_chosen:
        print(f"  Target waypoints:")
        for j, t in enumerate(targets_chosen):
            print(f"    #{j+1}: ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})")
    print(f"{'='*60}")

    pipeline.get_corrected_map_points()
    pipeline.print_summary()

    # Save the final SLAM map directly from the viewer's shared-memory
    # buffer — this is exactly what was displayed, avoiding any GTSAM
    # recomposition artefacts.
    map_out_dir = os.path.join(os.path.dirname(__file__), "savedMaps",
                               f"slam_map_{int(time.time())}")
    pipeline._viewer.export_map(
        map_out_dir,
        bounds=np.array(EXPLORE_BOUNDS, dtype=np.float64),
        resolution=PLANNER_RES,
        source="replay",
    )

    print("\n  Viewer is still open — close the Open3D window or press Enter to exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    pipeline.stop_viewer()
    print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Live mode — autonomous flight with AirSim
# ══════════════════════════════════════════════════════════════════════════════

def run_live():
    """Connect to AirSim, fly autonomously using frontier exploration."""
    import cosysairsim as airsim

    cfg = SLAMConfig(
        registration="vgicp",
        octo_resolution=0.15,
        frame_skip=0,
        live_max_hz=SCAN_HZ,
        enable_viewer=True,
    )

    live = LiveSLAM(cfg)

    # Start the 3-D viewer as early as possible so the user can see
    # the map being populated from the very first scan.
    live.pipeline.start_viewer()

    live.connect()

    out_dir = os.path.join(SAVE_DIR, f"exploration_{int(time.time())}")
    live.enable_recording(out_dir)

    buf = BufferedSLAM(live, delay_scans=SCAN_DELAY)
    print(f"  Scan buffer delay: {SCAN_DELAY} scans "
          f"({SCAN_DELAY / SCAN_HZ:.1f}s at {SCAN_HZ:.2f} Hz)")

    # Live quality plot (RMSE / fitness / rejections)
    quality_plot = _QualityPlot(cfg.registration)

    # NBV candidate comparison dashboard (only active when USE_NBV=True)
    nbv_debug_plot = _NBVDebugPlot() if USE_NBV else None

    # Start background scan collection immediately so the viewer
    # and SLAM map are populated from the very first moment.
    buf.start_collection()

    # ── Exploration planner (must be created before takeoff so scans are forwarded)
    exploration = ExplorationPlanner(
        bounds=EXPLORE_BOUNDS,
        resolution=PLANNER_RES,
        min_frontier_size=3,
        unknown_gain_radius=UNKNOWN_GAIN_RADIUS,
        distance_exponent=DISTANCE_EXPONENT,
        lidar_altitude_offset=LIDAR_ALT_OFFSET,
        min_target_distance=MIN_TARGET_DIST,
        waypoint_exclusion_radius=WP_EXCLUSION_RADIUS,
        inflation_margin=INFLATION_RADIUS + 0.5,
        use_random=USE_RANDOM,
        random_max_attempts=RANDOM_MAX_ATTEMPTS,
        use_nbv=USE_NBV,
        nbv_sensor_half_angle=NBV_SENSOR_HALF_ANGLE,
        nbv_cruise_altitude=NBV_CRUISE_ALTITUDE,
        nbv_n_unknown_columns=NBV_N_UNKNOWN_COLUMNS,
        nbv_n_local_samples=NBV_N_LOCAL_SAMPLES,
        nbv_local_radius=NBV_LOCAL_RADIUS,
        above_grid_margin=ABOVE_GRID_MARGIN,
        nbv_lidar_max_range=NBV_LIDAR_MAX_RANGE,
        nbv_unknown_block_size=NBV_UNKNOWN_BLOCK_SIZE,
        nbv_n_unknown_blocks=NBV_N_UNKNOWN_BLOCKS,
        nbv_volumetric_radius=NBV_VOLUMETRIC_RADIUS,
        nbv_volumetric_weight=NBV_VOLUMETRIC_WEIGHT,
        nbv_ray_max_targets=NBV_RAY_MAX_TARGETS,
        nbv_use_ray_tracing=NBV_USE_RAY_TRACING,
    )
    buf.set_planner(exploration)

    # ── OMPL path planner (obstacle avoidance) ───────────────────────────
    path_planner = PathPlanner(
        inflation_radius=INFLATION_RADIUS,
        planner_type=PATH_PLANNER_TYPE,
        solve_timeout=PATH_SOLVE_TIMEOUT,
        ground_z=0.0,
        above_grid_margin=ABOVE_GRID_MARGIN,
    )
    # Give the exploration planner a reference to the path planner so
    # random mode can use sample_near_obstacle_goal().
    exploration._path_planner = path_planner

    # ── Takeoff ──────────────────────────────────────────────────────────
    # Scan collection thread is already running (started above).

    print("Taking off...")
    live.client.takeoffAsync().join()
    live.client.hoverAsync().join()

    print(f"Rising to altitude z={TAKEOFF_HEIGHT} ...")
    live.client.moveToPositionAsync(0, 0, TAKEOFF_HEIGHT, velocity=VELOCITY).join()
    live.client.hoverAsync().join()

    # Hover at cruise altitude and collect several scans so the planner
    # has a proper observed region centred at the operating height.
    # Scan collection is running in the background thread already.
    MIN_INITIAL_SCANS = 5
    INITIAL_TIMEOUT   = 30.0   # seconds — safety cap
    print(f"  Collecting initial scans at cruise altitude (need {MIN_INITIAL_SCANS})...")
    t0 = time.time()
    while buf._n_collected < MIN_INITIAL_SCANS and (time.time() - t0) < INITIAL_TIMEOUT:
        quality_plot.update(live.pipeline)
        time.sleep(0.05)
    print(f"  Collected {buf._n_collected} scans in {time.time() - t0:.1f}s")

    # Do NOT flush — scans must stay in the buffer for the full delay
    # so GPS coordinates have time to settle.  The background thread
    # drains them naturally once enough newer scans arrive.
    print(f"  Initial scans forwarded to planner: {exploration._n_scans_raycasted} raycasted, "
          f"{len(exploration._scan_data)} total")

    # ── Path follower (threaded flight executor) ─────────────────────────
    def _viewer_tick(pos, _follower):
        """Called at POLL_HZ on the follower thread.

        Drone position in the viewer is managed exclusively by the
        BufferedSLAM collection thread so the marker doesn't jump
        between competing position sources.  This callback is kept
        for any future per-tick work but does NOT touch the viewer."""
        pass

    follower = PathFollower(
        live.client, planner=path_planner,
        mode=FLIGHT_MODE, velocity=VELOCITY,
        poll_hz=POLL_HZ, min_spacing=MIN_WP_SPACING,
        on_tick=_viewer_tick,
    )
    follower.start()

    mode_label = ("moveOnPathAsync" if FLIGHT_MODE == "path"
                  else "pure-pursuit velocity")

    print(f"\n{'='*60}")
    print(f"Autonomous 3-D exploration started")
    print(f"  Bounds: x=[{EXPLORE_BOUNDS[0]}, {EXPLORE_BOUNDS[1]}], "
          f"y=[{EXPLORE_BOUNDS[2]}, {EXPLORE_BOUNDS[3]}], "
          f"z=[{EXPLORE_BOUNDS[4]}, {EXPLORE_BOUNDS[5]}]")
    print(f"  Grid:  {exploration.nx} x {exploration.ny} x {exploration.nz} voxels @ {PLANNER_RES} m")
    strategy = "random" if USE_RANDOM else ("NBV" if USE_NBV else "WFD")
    print(f"  Target strategy: {strategy}")
    print(f"  Path planner: {PATH_PLANNER_TYPE}  |  Inflation: {INFLATION_RADIUS} m")
    print(f"  Flight mode: {mode_label}  |  Max waypoints: {MAX_TARGETS}")
    print(f"{'='*60}\n")

    wp_count = 0

    while wp_count < MAX_TARGETS:
        # The PathFollower background thread automatically holds altitude
        # between paths — no explicit hover command needed here.

        buf.drain_overlays()
        quality_plot.update(live.pipeline)

        state = live.client.getMultirotorState()
        p = state.kinematics_estimated.position
        current_pos = np.array([p.x_val, p.y_val, p.z_val])

        # ── Update PathPlanner's map BEFORE target selection so that
        #    random mode has access to the current traversability grid.
        origin = np.array([exploration.xmin, exploration.ymin, exploration.zmin])
        path_planner.update_map(
            exploration._observed.copy(),
            exploration._last_occupied_grid.copy(),
            origin, exploration.resolution,
        )
        vis_pts = live.pipeline.get_map_points()
        if len(vis_pts) > 0:
            path_planner.points = vis_pts.astype(np.float32)

        target, info = exploration.next_target(live.pipeline, current_pos)

        # Update the NBV debug dashboard
        if nbv_debug_plot is not None and "nbv_debug" in info:
            nbv_debug_plot.update(
                info, drone_pos=current_pos,
                map_points=vis_pts if len(vis_pts) > 0 else None)

        # Queue overlays for deferred display (syncs with SLAM buffer delay)
        buf.queue_overlay(frontier_points=info.get("frontier_world_pts"))
        if NBV_SHOW_CANDIDATES:
            buf.queue_overlay(candidate_points=info.get("candidate_world_pts"))

        print(f"  [{wp_count + 1:02d}] Occupied: {info['n_occupied_cells']} | "
              f"Frontiers: {info['n_frontier_cells']}/{info['n_frontier_cells_raw']} "
              f"(after visited filter) in {info['n_clusters']} clusters")

        if target is None:
            print("\n  Exploration complete — no unvisited map-edge frontiers remain")
            break

        print(f"       -> target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

        buf.queue_overlay(target_pos=target.tolist())
        wp_count += 1

        # ── Plan a collision-free path to the target ─────────────────
        t_plan = time.time()
        path = path_planner.plan(current_pos, target)
        dt_plan = time.time() - t_plan

        if path is not None:
            print(f"       Path: {len(path)} waypoints, {dt_plan:.2f}s")

            # Queue the planned path for deferred display
            buf.queue_overlay(path_points=np.asarray(path, dtype=np.float64))

            # Fly the planned path with the threaded PathFollower
            follower.follow(path, goal=target)

            # Wait for the follower to finish — scans are collected
            # continuously by the BufferedSLAM background thread.
            while follower.is_busy:
                buf.drain_overlays()
                quality_plot.update(live.pipeline)
                time.sleep(0.05)

            # Log the flight result
            result = follower.last_result
            if result is not None:
                status = ("ARRIVED" if result.success
                          else ("COLLISION" if result.collided else "MISSED"))
                clearance_str = (
                    f"{result.min_obstacle_clearance:.2f} m"
                    if result.min_obstacle_clearance < float('inf') else "n/a")
                print(f"       {status} — flight: {result.flight_time:.1f}s, "
                      f"error: {result.arrival_error:.2f} m, "
                      f"clearance: {clearance_str}")
        else:
            print(f"       No path found ({dt_plan:.2f}s) — skipping target")
            buf.queue_overlay(path_points=None)  # clear path line from viewer
            # Do NOT fall back to straight-line flight — the path planner
            # couldn't find a route, so flying direct would hit obstacles.
            # Instead, skip this target and let the next iteration pick
            # a different one.
            continue

        # Brief pause to let the scan thread populate the map
        time.sleep(0.5)

    print(f"\nExploration finished after {wp_count} waypoints.")

    # ── Finalise ─────────────────────────────────────────────────────────
    # Flush any remaining deferred overlays so the viewer is up to date
    with buf._overlay_lock:
        for _, data in buf._overlay_queue:
            if "target_pos" in data:
                live.set_target(data["target_pos"])
            if "path_points" in data:
                live.set_path_points(data["path_points"])
            if "frontier_points" in data:
                live.set_frontier_points(data["frontier_points"])
            if "candidate_points" in data:
                live.set_candidate_points(data["candidate_points"])
        buf._overlay_queue.clear()
    live.refresh_overlays()

    buf.stop_collection()
    follower.stop()
    buf.flush()
    print(f"  Buffer stats: {buf._n_collected} total scans collected")

    live.pipeline.get_corrected_map_points()

    # Save the final SLAM map directly from the viewer's shared-memory
    # buffer — this is exactly what was displayed, avoiding any GTSAM
    # recomposition artefacts.
    map_out_dir = os.path.join(os.path.dirname(__file__), "savedMaps",
                               f"slam_map_{int(time.time())}")
    live.pipeline._viewer.export_map(
        map_out_dir,
        bounds=np.array(EXPLORE_BOUNDS, dtype=np.float64),
        resolution=PLANNER_RES,
        source="live",
    )

    # Save the quality plot
    if out_dir:
        qp_path = os.path.join(out_dir, "quality.png")
        quality_plot.save(qp_path, live.pipeline)
        print(f"  Quality plot saved to {qp_path}")

    # Save the NBV debug dashboard
    if out_dir and nbv_debug_plot is not None:
        nbv_path = os.path.join(out_dir, "nbv_debug.png")
        nbv_debug_plot.save(nbv_path)
        print(f"  NBV debug plot saved to {nbv_path}")

    if out_dir:
        bt_path = os.path.join(out_dir, "map.bt")
        try:
            live.pipeline.save_octomap(bt_path)
        except Exception as e:
            print(f"  (could not save OctoMap: {e})")
    live.pipeline.print_summary()
    print(f"\nOutputs saved to {out_dir}/")

    print("\n  Viewer is still open — close the Open3D window or press Enter here to exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    live.pipeline.stop_viewer()
    print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    # Command line override: pass a recording directory as argument
    replay = sys.argv[1] if len(sys.argv) > 1 else REPLAY_DIR

    if replay:
        # Resolve relative paths from this script's directory
        if not os.path.isabs(replay):
            replay = os.path.join(os.path.dirname(__file__), replay)
        run_replay(replay)
    else:
        run_live()
