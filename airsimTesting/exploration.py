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

from finalMappingPipeline import LiveSLAM, SLAMConfig, SLAMPipeline
from RegistrationComparison import resolve_recording_dir, filter_valid, xform_pts
from obstacleAvoidance import (
    PathPlanner, PathFollower, FollowerState, FlightResult,
    get_drone_position,
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
            GPS_NEAR_MAX_MS = 20.0
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
        # ── NBV parameters ────────────────────────────────────────
        use_nbv: bool = False,
        nbv_sensor_half_angle: float = 45.0,
        nbv_cruise_altitude: float | None = None,
        nbv_n_unknown_columns: int = 10,
        nbv_n_local_samples: int = 15,
        nbv_local_radius: float = 10.0,
        above_grid_margin: float = 0.0,
        nbv_lidar_max_range: float = 40.0,
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

        # NBV settings
        self.use_nbv = use_nbv
        self.nbv_sensor_half_angle = nbv_sensor_half_angle
        self.nbv_cruise_altitude = nbv_cruise_altitude
        self.nbv_n_unknown_columns = nbv_n_unknown_columns
        self.nbv_n_local_samples = nbv_n_local_samples
        self.nbv_local_radius = nbv_local_radius
        self.above_grid_margin = above_grid_margin
        self.nbv_lidar_max_range = nbv_lidar_max_range

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
    ) -> int:
        """Score *one* viewpoint by simulating its downward-facing LiDAR cone.

        Works at *any* altitude (above the grid, inside it, etc.).  For each
        column (ix, iy) within the cone footprint the method walks downward
        from the candidate's Z level and counts unknown voxels until an
        occupied voxel occludes the rest of the column.  This correctly
        handles overhangs: a candidate *below* an overhang can see the
        voxels beneath it that a higher candidate cannot.

        Returns the total visible-unknown-voxel count (information gain).
        """
        cx, cy, cz = cand
        tan_ha = np.tan(np.radians(self.nbv_sensor_half_angle))
        res = self.resolution
        max_range = self.nbv_lidar_max_range
        max_range_sq = max_range * max_range

        # Grid iz just at or above the candidate altitude
        ciz_float = (cz - self.zmin) / res
        ciz = int(np.clip(np.floor(ciz_float), -1, self.nz - 1))
        # Start scanning from ciz+1 (first layer strictly below)
        iz_start = max(ciz + 1, 0)
        if iz_start >= self.nz:
            return 0  # candidate below entire grid in NED → nothing to see

        # Column world coordinates
        col_x = self.xmin + (np.arange(self.nx) + 0.5) * res  # (nx,)
        col_y = self.ymin + (np.arange(self.ny) + 0.5) * res  # (ny,)
        dx2 = (col_x - cx) ** 2                               # (nx,)
        dy2 = (col_y - cy) ** 2                               # (ny,)
        dist_sq_xy = dx2[:, np.newaxis] + dy2[np.newaxis, :]   # (nx, ny)

        # Per-column first-occupied below candidate
        first_occ = np.full((self.nx, self.ny), self.nz, dtype=np.intp)
        for iz in range(iz_start, self.nz):
            mask = occupied_grid[:, :, iz] & (first_occ == self.nz)
            first_occ[mask] = iz

        # Z centres for levels below the candidate
        wz_levels = self.zmin + (np.arange(iz_start, self.nz) + 0.5) * res
        depths = wz_levels - cz                               # positive = below
        cone_r_sq = np.where(depths > 0,
                             (depths * tan_ha) ** 2, -1.0)    # (n_levels,)

        # 3-D distance² from candidate to each voxel centre:
        # dist² = dx² + dy² + dz²  where dz = depth for each level.
        # Voxels beyond the LiDAR max range produce unreliable returns.
        dz2 = depths ** 2                                      # (n_levels,)
        dist3d_sq = (dist_sq_xy[:, :, np.newaxis]
                     + dz2[np.newaxis, np.newaxis, :])         # (nx,ny,n_levels)

        iz_abs = np.arange(iz_start, self.nz)                 # absolute indices
        # Broadcast: (nx, ny, 1) vs (1, 1, n_levels)
        in_cone = (dist_sq_xy[:, :, np.newaxis]
                   <= cone_r_sq[np.newaxis, np.newaxis, :])    # (nx,ny,n_levels)
        in_range = dist3d_sq <= max_range_sq                   # (nx,ny,n_levels)
        not_occ = (iz_abs[np.newaxis, np.newaxis, :]
                   < first_occ[:, :, np.newaxis])              # (nx,ny,n_levels)
        unk = unknown[:, :, iz_start:]                         # (nx,ny,n_levels)

        return int((in_cone & in_range & not_occ & unk).sum())

    def _nbv_select_target(
        self,
        current_pos: np.ndarray,
        occupied_grid: np.ndarray,
        free: np.ndarray,
        unknown: np.ndarray,
        is_frontier: np.ndarray,
        start: tuple[int, int, int],
    ) -> np.ndarray | None:
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
        """
        candidates = self._nbv_generate_candidates(
            current_pos, occupied_grid, free, unknown, is_frontier, start)

        if len(candidates) == 0:
            print("    [nbv] 0 candidates generated")
            return None

        cur_xyz = current_pos[:3]
        best_score = -np.inf
        best_candidate = None
        n_evaluated = 0

        for cand in candidates:
            dist = float(np.linalg.norm(cand - cur_xyz))
            if dist < self.min_target_distance:
                continue

            if self._waypoint_history:
                prev = np.array(self._waypoint_history)
                if np.any(np.linalg.norm(prev - cand, axis=1)
                          < self.waypoint_exclusion_radius):
                    continue

            # Check that the candidate voxel (if inside grid) is not occupied
            if (self.xmin <= cand[0] <= self.xmax and
                    self.ymin <= cand[1] <= self.ymax and
                    self.zmin <= cand[2] <= self.zmax):
                gx, gy, gz = self._world_to_grid(*cand)
                if occupied_grid[gx, gy, gz]:
                    continue

            gain = self._nbv_score_candidate(cand, occupied_grid, unknown)
            if gain == 0:
                continue

            score = gain / max(dist, 0.01) ** self.distance_exponent
            n_evaluated += 1

            if score > best_score:
                best_score = score
                best_candidate = cand.copy()

        print(f"    [nbv] {len(candidates)} candidates, "
              f"{n_evaluated} scored, best gain/cost={best_score:.1f}")

        if best_candidate is not None:
            self._waypoint_history.append(best_candidate.copy())
        return best_candidate

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

        # ── Branch: NBV or WFD ────────────────────────────────────────
        if self.use_nbv:
            # ── 4a) NBV — frustum-based viewpoint scoring ────────────
            best_target = self._nbv_select_target(
                current_pos, occupied_grid, free, unknown,
                is_frontier, start)

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
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SAVE_DIR = os.path.join(os.path.dirname(__file__), "flight_recordings")

# ── Mode selection ────────────────────────────────────────────────────────
# Set REPLAY_DIR to a recording directory (or parent) to run offline on
# saved LiDAR data.  Leave empty ("") for live AirSim flight.

#REPLAY_DIR      = "flight_recordings"            # e.g. "flight_recordings/flight_1771909992"
REPLAY_DIR      = ""  
# ── Exploration parameters (shared by both modes) ────────────────────────
EXPLORE_BOUNDS  = (-20, 20, -50, 10, -20, 0)   # (xmin,xmax,ymin,ymax,zmin,zmax) NED
TAKEOFF_HEIGHT  = EXPLORE_BOUNDS[4] - 5         # NED z, 5 m above grid ceiling (LiDAR points down)
VELOCITY        = 3             # m/s (live mode only)
SCAN_HZ         = 1 / 1.5      # scans per second (live mode only)
PLANNER_RES     = 1.0           # planning grid voxel size (m)
MAX_TARGETS     = 50            # safety cap on autonomous waypoints
SCAN_DELAY      = 3             # register a scan only after N newer scans (live mode)
PLAN_EVERY      = 3             # run planner every N frames (replay mode)
FRAME_SKIP      = 1             # process every Nth frame (replay mode, 1 = all)

# ── Path-planning parameters (live mode) ─────────────────────────────────
INFLATION_RADIUS    = 1.5       # safety margin (m) inflated around obstacles
ABOVE_GRID_MARGIN   = 10.0      # metres of free airspace above the grid ceiling
PATH_PLANNER_TYPE   = "ABITstar"
PATH_SOLVE_TIMEOUT  = 2.0       # seconds per OMPL solve
FLIGHT_MODE         = "velocity" # "path" (moveOnPathAsync) or "velocity" (pure-pursuit)
POLL_HZ             = 20.0
MIN_WP_SPACING      = 1.0

# ── Exploration scoring (live mode) ──────────────────────────────────────
UNKNOWN_GAIN_RADIUS = 3         # voxels: box radius for counting unknown neighbours
DISTANCE_EXPONENT   = 0.5       # softer distance penalty (1.0 = original linear)
LIDAR_ALT_OFFSET    = 2.0       # fly this many metres above frontier centroid (NED)
MIN_TARGET_DIST     = 3.0       # skip targets closer than this (m) to avoid re-visiting
WP_EXCLUSION_RADIUS = 3.0       # avoid re-selecting waypoints within this radius (m)

# ── Target selection algorithm ───────────────────────────────────────────
# Set USE_NBV = True to use Next-Best-View frustum scoring instead of WFD
# frontier-centroid scoring.  NBV generates candidate viewpoints on a grid
# at cruise altitude and picks the one that maximises the number of unknown
# voxels visible through a simulated downward-facing LiDAR cone.
USE_NBV                 = True
NBV_SENSOR_HALF_ANGLE   = 45.0  # half-angle (deg) of the LiDAR cone
NBV_CRUISE_ALTITUDE     = TAKEOFF_HEIGHT  # one of the altitude options for candidates (NED)
NBV_N_UNKNOWN_COLUMNS   = 10    # top-K columns by unknown-voxel count to sample
NBV_N_LOCAL_SAMPLES     = 15    # random free-space samples near the drone
NBV_LOCAL_RADIUS        = 10.0  # world-space radius for local samples (m)
NBV_LIDAR_MAX_RANGE     = 40.0  # LiDAR max range (m); voxels beyond this are not scored

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
        use_nbv=USE_NBV,
        nbv_sensor_half_angle=NBV_SENSOR_HALF_ANGLE,
        nbv_cruise_altitude=NBV_CRUISE_ALTITUDE,
        nbv_n_unknown_columns=NBV_N_UNKNOWN_COLUMNS,
        nbv_n_local_samples=NBV_N_LOCAL_SAMPLES,
        nbv_local_radius=NBV_LOCAL_RADIUS,
    )

    print(f"\n{'='*60}")
    print(f"Exploration planner replay")
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
        frame_skip=1,
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
        use_nbv=USE_NBV,
        nbv_sensor_half_angle=NBV_SENSOR_HALF_ANGLE,
        nbv_cruise_altitude=NBV_CRUISE_ALTITUDE,
        nbv_n_unknown_columns=NBV_N_UNKNOWN_COLUMNS,
        nbv_n_local_samples=NBV_N_LOCAL_SAMPLES,
        nbv_local_radius=NBV_LOCAL_RADIUS,
        above_grid_margin=ABOVE_GRID_MARGIN,
        nbv_lidar_max_range=NBV_LIDAR_MAX_RANGE,
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
    print(f"  Path planner: {PATH_PLANNER_TYPE}  |  Inflation: {INFLATION_RADIUS} m")
    print(f"  Flight mode: {mode_label}  |  Max waypoints: {MAX_TARGETS}")
    print(f"{'='*60}\n")

    wp_count = 0

    while wp_count < MAX_TARGETS:
        # The PathFollower background thread automatically holds altitude
        # between paths — no explicit hover command needed here.

        state = live.client.getMultirotorState()
        p = state.kinematics_estimated.position
        current_pos = np.array([p.x_val, p.y_val, p.z_val])

        target, info = exploration.next_target(live.pipeline, current_pos)

        live.set_frontier_points(info.get("frontier_world_pts"))

        print(f"  [{wp_count + 1:02d}] Occupied: {info['n_occupied_cells']} | "
              f"Frontiers: {info['n_frontier_cells']}/{info['n_frontier_cells_raw']} "
              f"(after visited filter) in {info['n_clusters']} clusters")

        if target is None:
            print("\n  Exploration complete — no unvisited map-edge frontiers remain")
            break

        print(f"       -> target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

        live.set_target(target.tolist())
        wp_count += 1

        # ── Update PathPlanner's map from exploration grids ──────────
        origin = np.array([exploration.xmin, exploration.ymin, exploration.zmin])
        path_planner.update_map(
            exploration._observed.copy(),
            exploration._last_occupied_grid.copy(),
            origin, exploration.resolution,
        )
        # Store occupied world-frame points for viewer / clearance tracking
        vis_pts = live.pipeline.get_map_points()
        if len(vis_pts) > 0:
            path_planner.points = vis_pts.astype(np.float32)

        # ── Plan a collision-free path to the frontier target ────────
        t_plan = time.time()
        path = path_planner.plan(current_pos, target)
        dt_plan = time.time() - t_plan

        if path is not None:
            print(f"       Path: {len(path)} waypoints, {dt_plan:.2f}s")

            # Show the planned path in the 3D viewer (cyan line)
            live.set_path_points(np.asarray(path, dtype=np.float64))

            # Fly the planned path with the threaded PathFollower
            follower.follow(path, goal=target)

            # Wait for the follower to finish — scans are collected
            # continuously by the BufferedSLAM background thread.
            while follower.is_busy:
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
            print(f"       No path found ({dt_plan:.2f}s) — flying direct")
            live.set_path_points(None)  # clear path line from viewer
            # Fall back to straight-line flight routed through the
            # PathFollower so it uses the correct thread-local client.
            straight_path = np.array([current_pos, target], dtype=np.float64)
            follower.follow(straight_path, goal=target)
            while follower.is_busy:
                time.sleep(0.05)

        # Brief pause to let the scan thread populate the map
        time.sleep(0.5)

    print(f"\nExploration finished after {wp_count} waypoints.")

    # ── Finalise ─────────────────────────────────────────────────────────
    buf.stop_collection()
    follower.stop()
    buf.flush()
    print(f"  Buffer stats: {buf._n_collected} total scans collected")

    live.pipeline.get_corrected_map_points()
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
