#!/usr/bin/env python3
"""OMPL-based obstacle-avoidance path planning for a drone in AirSim.

Loads a voxel map (ground-truth or incrementally built) from an .npz file,
builds an occupancy grid with an inflation layer for safety, then uses
OMPL's RRT* planner to find collision-free paths to random near-obstacle
goal locations.

The map is stored as a ``VoxelMap`` that can be **hot-reloaded** or
**updated in-place** so the same planner pipeline works for both:
  - static ground-truth maps, and
  - incrementally expanding maps during live exploration.

Unknown space is treated as traversable (optimistic), since the drone must
be able to fly into unexplored regions.  The inflation (safety margin)
only applies to *known* occupied voxels.
"""

from __future__ import annotations

import cosysairsim as airsim
import math
import numpy as np
import os
import sys
import time
from collections import deque

from ompl import base as ob
from ompl import geometric as og

from sensorFeed import Viewer3D


# ══════════════════════════════════════════════════════════════════════════════
# GPS-based drone localisation
# ══════════════════════════════════════════════════════════════════════════════

def gps_to_ned(
    gps_lat: float, gps_lon: float, gps_alt: float,
    home_lat: float, home_lon: float, home_alt: float,
) -> np.ndarray:
    """Convert GPS geodetic coordinates to local NED relative to *home*.

    Uses a flat-earth approximation with the WGS-84 semi-major axis.
    Accurate to < 1 m for distances up to several kilometres.

    Returns
    -------
    ndarray, shape (3,)
        Position in NED (North-East-Down) metres, where
        X = North, Y = East, Z = Down.
    """
    R_EARTH = 6_378_137.0  # WGS-84 semi-major axis (m)

    d_lat = math.radians(gps_lat - home_lat)
    d_lon = math.radians(gps_lon - home_lon)

    north = d_lat * R_EARTH
    east  = d_lon * R_EARTH * math.cos(math.radians(home_lat))
    down  = -(gps_alt - home_alt)

    return np.array([north, east, down])


def get_drone_position(client: airsim.MultirotorClient) -> np.ndarray:
    """Return the drone's NED position derived from its simulated GPS.

    AirSim's state-estimator (``kinematics_estimated``) reports position
    relative to the player-start origin, but this can be incorrect if
    the drone spawns at an unexpected location.  The simulated GPS, on
    the other hand, reflects the drone's *true* physics state.  By
    converting the GPS reading back to NED via the home geo-point we
    get the correct position in the map's coordinate frame.
    """
    home = client.getHomeGeoPoint()
    gps  = client.getGpsData().gnss.geo_point
    return gps_to_ned(
        gps.latitude, gps.longitude, gps.altitude,
        home.latitude, home.longitude, home.altitude,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VoxelMap — occupancy representation with safety inflation
# ══════════════════════════════════════════════════════════════════════════════

class VoxelMap:
    """3-D occupancy grid built from a voxel point cloud.

    Parameters
    ----------
    resolution : float
        Voxel edge length in metres used for the planning grid.
        May differ from the source map resolution (points are re-binned).
    inflation_radius : float
        Safety margin (metres) added around every occupied voxel.
    """

    def __init__(self, resolution: float = 0.5, inflation_radius: float = 1.0):
        self.resolution = resolution
        self.inflation_radius = inflation_radius

        # Grid arrays (allocated on first load / update)
        self.occupied: np.ndarray | None = None   # bool (nx, ny, nz)
        self.inflated: np.ndarray | None = None   # bool (nx, ny, nz) — collision grid

        # World-frame origin (minimum corner)
        self.origin = np.zeros(3)
        self.nx = self.ny = self.nz = 0

        # The raw occupied points (world frame) — kept for viewer display
        self.points: np.ndarray = np.empty((0, 3), dtype=np.float32)

    # ── Load from .npz (displayGroundTruth format) ────────────────────────

    def load_npz(self, path: str) -> None:
        """Load a voxel map from a ``.npz`` file saved by displayGroundTruth.py.

        Expected keys: ``points`` (N, 3) float32 occupied voxel centres,
        plus optional metadata (``center``, ``grid_size``, ``resolution``).
        """
        data = np.load(path)
        pts = data["points"].astype(np.float32)
        self._build_from_points(pts)
        print(f"  [VoxelMap] Loaded {len(pts):,} occupied points from {path}")
        print(f"  [VoxelMap] Grid: {self.nx}×{self.ny}×{self.nz} @ {self.resolution} m")
        print(f"  [VoxelMap] Bounds: ({self.origin[0]:.1f}..{self.origin[0]+self.nx*self.resolution:.1f}, "
              f"{self.origin[1]:.1f}..{self.origin[1]+self.ny*self.resolution:.1f}, "
              f"{self.origin[2]:.1f}..{self.origin[2]+self.nz*self.resolution:.1f})")

    def update_points(self, pts: np.ndarray) -> None:
        """Replace the map with a new set of occupied world-frame points.

        Use this for live / incrementally expanding maps.
        """
        pts = np.asarray(pts, dtype=np.float32)
        self._build_from_points(pts)

    # ── Internal grid construction ────────────────────────────────────────

    def _build_from_points(self, pts: np.ndarray) -> None:
        self.points = pts
        if len(pts) == 0:
            self.occupied = np.zeros((1, 1, 1), dtype=bool)
            self.inflated = np.zeros((1, 1, 1), dtype=bool)
            self.origin = np.zeros(3)
            self.nx = self.ny = self.nz = 1
            return

        res = self.resolution
        # Pad the bounding box by the inflation radius + one extra voxel
        pad = self.inflation_radius + res
        mins = pts.min(axis=0) - pad
        maxs = pts.max(axis=0) + pad
        self.origin = mins.copy()

        self.nx = int(np.ceil((maxs[0] - mins[0]) / res))
        self.ny = int(np.ceil((maxs[1] - mins[1]) / res))
        self.nz = int(np.ceil((maxs[2] - mins[2]) / res))

        # Build occupied grid
        self.occupied = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        ix = np.clip(((pts[:, 0] - self.origin[0]) / res).astype(int), 0, self.nx - 1)
        iy = np.clip(((pts[:, 1] - self.origin[1]) / res).astype(int), 0, self.ny - 1)
        iz = np.clip(((pts[:, 2] - self.origin[2]) / res).astype(int), 0, self.nz - 1)
        self.occupied[ix, iy, iz] = True

        # Inflate: dilate occupied voxels by a spherical structuring element
        self._inflate()

    def _inflate(self) -> None:
        from scipy import ndimage

        r_vox = max(1, int(np.ceil(self.inflation_radius / self.resolution)))
        # Build spherical structuring element
        diam = 2 * r_vox + 1
        struct = np.zeros((diam, diam, diam), dtype=bool)
        c = r_vox
        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dz in range(-r_vox, r_vox + 1):
                    if dx * dx + dy * dy + dz * dz <= r_vox * r_vox:
                        struct[c + dx, c + dy, c + dz] = True

        self.inflated = ndimage.binary_dilation(self.occupied, structure=struct)
        n_occ = int(self.occupied.sum())
        n_inf = int(self.inflated.sum())
        print(f"  [VoxelMap] Occupied voxels: {n_occ:,}  |  Inflated: {n_inf:,}  "
              f"(r={self.inflation_radius} m, {r_vox} vox)")

    # ── Query helpers ─────────────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        res = self.resolution
        ix = int(np.clip(int((x - self.origin[0]) / res), 0, self.nx - 1))
        iy = int(np.clip(int((y - self.origin[1]) / res), 0, self.ny - 1))
        iz = int(np.clip(int((z - self.origin[2]) / res), 0, self.nz - 1))
        return ix, iy, iz

    def grid_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return self.origin + np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * self.resolution

    def is_occupied(self, x: float, y: float, z: float) -> bool:
        """Check the *raw* occupied grid (no inflation)."""
        ix, iy, iz = self.world_to_grid(x, y, z)
        return bool(self.occupied[ix, iy, iz])

    def is_collision(self, x: float, y: float, z: float) -> bool:
        """Check the *inflated* collision grid."""
        ix, iy, iz = self.world_to_grid(x, y, z)
        return bool(self.inflated[ix, iy, iz])

    def in_bounds(self, x: float, y: float, z: float) -> bool:
        """True if the point falls inside the grid's bounding box."""
        return (self.origin[0] <= x <= self.origin[0] + self.nx * self.resolution
                and self.origin[1] <= y <= self.origin[1] + self.ny * self.resolution
                and self.origin[2] <= z <= self.origin[2] + self.nz * self.resolution)

    @property
    def bounds_min(self) -> np.ndarray:
        return self.origin.copy()

    @property
    def bounds_max(self) -> np.ndarray:
        return self.origin + np.array([self.nx, self.ny, self.nz]) * self.resolution


# ══════════════════════════════════════════════════════════════════════════════
# OMPL Planner
# ══════════════════════════════════════════════════════════════════════════════

class OMPLPlanner:
    """OMPL-based 3-D path planner using RRT* with a VoxelMap for collision.

    Unknown space (outside the map or not yet mapped) is treated as
    **free**, so the planner can operate with partial / expanding maps.

    Parameters
    ----------
    voxel_map : VoxelMap
        The occupancy grid to collision-check against.
    planner_type : str
        OMPL planner name: ``"RRTstar"``, ``"RRTConnect"``, ``"BITstar"``, etc.
    solve_timeout : float
        Maximum seconds for the planner to search per query.
    path_resolution : float
        Interpolation resolution (metres) for collision checking along edges.
    """

    def __init__(
        self,
        voxel_map: VoxelMap,
        planner_type: str = "RRTstar",
        solve_timeout: float = 5.0,
        path_resolution: float = 0.25,
        bounds_padding: float = 20.0,
    ):
        self.vmap = voxel_map
        self.planner_type = planner_type
        self.solve_timeout = solve_timeout
        self.path_resolution = path_resolution
        self.bounds_padding = bounds_padding

        # Build OMPL state space — padded well beyond the map so the
        # drone's actual position is always inside the valid state space.
        # The validity checker already treats out-of-map space as free.
        self._space = ob.RealVectorStateSpace(3)
        self._apply_bounds()

        # Space information + validity checker
        self._si = ob.SpaceInformation(self._space)
        self._si.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_valid))
        self._si.setStateValidityCheckingResolution(
            path_resolution / self._space.getMaximumExtent()
        )
        self._si.setup()

    def _apply_bounds(self) -> None:
        """Set OMPL state-space bounds = map bounds + generous padding."""
        bounds = ob.RealVectorBounds(3)
        bmin = self.vmap.bounds_min
        bmax = self.vmap.bounds_max
        pad = self.bounds_padding
        for i in range(3):
            bounds.setLow(i, float(bmin[i]) - pad)
            bounds.setHigh(i, float(bmax[i]) + pad)
        self._space.setBounds(bounds)
        lo = [float(bmin[i]) - pad for i in range(3)]
        hi = [float(bmax[i]) + pad for i in range(3)]
        print(f"  [OMPL] State space bounds: "
              f"({lo[0]:.1f}..{hi[0]:.1f}, {lo[1]:.1f}..{hi[1]:.1f}, {lo[2]:.1f}..{hi[2]:.1f})")

    def _is_valid(self, state) -> bool:
        """OMPL state validity checker — free if NOT in the inflated grid.

        Points outside the known map are treated as free (optimistic for
        exploration into unknown space).
        """
        x, y, z = state[0], state[1], state[2]
        if not self.vmap.in_bounds(x, y, z):
            return True  # unknown / out-of-map → optimistically free
        return not self.vmap.is_collision(x, y, z)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        simplify: bool = True,
    ) -> np.ndarray | None:
        """Plan a collision-free path from *start* to *goal*.

        Parameters
        ----------
        start, goal : ndarray, shape (3,)
            World-frame positions (NED).
        simplify : bool
            If True, run OMPL's path simplifier to smooth/shorten the result.

        Returns
        -------
        ndarray, shape (N, 3) or None
            Waypoints along the collision-free path, or ``None`` if planning fails.
        """
        # Dynamically expand bounds if start or goal is outside
        self._ensure_in_bounds(start)
        self._ensure_in_bounds(goal)

        pdef = ob.ProblemDefinition(self._si)

        s = ob.State(self._space)
        s[0], s[1], s[2] = float(start[0]), float(start[1]), float(start[2])
        g = ob.State(self._space)
        g[0], g[1], g[2] = float(goal[0]), float(goal[1]), float(goal[2])

        pdef.setStartAndGoalStates(s, g)
        pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(self._si))

        planner = self._make_planner()
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(self.solve_timeout)

        if not solved:
            return None

        path = pdef.getSolutionPath()

        if simplify:
            simplifier = og.PathSimplifier(self._si)
            simplifier.simplifyMax(path)

        path.interpolate()   # densify for smooth flight

        # Extract waypoints
        states = path.getStates()
        waypoints = np.array([[st[0], st[1], st[2]] for st in states], dtype=np.float64)
        return waypoints

    def _make_planner(self):
        planners = {
            "RRTstar": og.RRTstar,
            "RRTConnect": og.RRTConnect,
            "RRT": og.RRT,
            "PRM": og.PRM,
            "BITstar": og.BITstar,
            "InformedRRTstar": og.InformedRRTstar,
        }
        cls = planners.get(self.planner_type, og.RRTstar)
        return cls(self._si)

    def _ensure_in_bounds(self, point: np.ndarray) -> None:
        """Expand OMPL bounds if *point* is outside the current state space."""
        bounds = self._space.getBounds()
        expanded = False
        for i in range(3):
            v = float(point[i])
            if v < bounds.low[i]:
                print(f"  [OMPL] Expanding axis {i} low: {bounds.low[i]:.2f} → {v - 5:.2f}")
                bounds.setLow(i, v - 5.0)
                expanded = True
            if v > bounds.high[i]:
                print(f"  [OMPL] Expanding axis {i} high: {bounds.high[i]:.2f} → {v + 5:.2f}")
                bounds.setHigh(i, v + 5.0)
                expanded = True
        if expanded:
            self._space.setBounds(bounds)
            self._si.setup()

    def rebuild(self) -> None:
        """Rebuild OMPL internals after the VoxelMap has been updated.

        Call this when the underlying map changes (e.g. new exploration data)
        so the planner bounds and validity checker reflect the new geometry.
        """
        self._apply_bounds()
        self._si.setup()


# ══════════════════════════════════════════════════════════════════════════════
# Nearest free-space finder (for escaping from inside obstacles)
# ══════════════════════════════════════════════════════════════════════════════

def find_nearest_free(
    vmap: VoxelMap,
    pos: np.ndarray,
    max_radius: float = 10.0,
    step: float = 0.5,
) -> np.ndarray | None:
    """BFS-style search outward from *pos* to find the nearest collision-free point.

    Searches in a grid of ``step``-spaced candidates expanding in shells
    of increasing radius.  Returns the nearest point that is inside the
    map bounds and NOT in the inflated collision grid, or ``None`` if no
    free point is found within ``max_radius``.
    """
    # Try going straight up first (most common escape)
    for dz in np.arange(-step, -max_radius, -step):
        cand = pos + np.array([0.0, 0.0, dz])
        if not vmap.in_bounds(*cand) or not vmap.is_collision(*cand):
            return cand

    # Expand in axis-aligned directions
    for r in np.arange(step, max_radius + step, step):
        for dx in [-r, 0, r]:
            for dy in [-r, 0, r]:
                for dz in [-r, 0, r]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cand = pos + np.array([dx, dy, dz])
                    if not vmap.in_bounds(*cand):
                        return cand  # out-of-map = free
                    if not vmap.is_collision(*cand):
                        return cand
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Goal sampler — random positions near obstacles
# ══════════════════════════════════════════════════════════════════════════════

def sample_near_obstacle_goal(
    vmap: VoxelMap,
    current_pos: np.ndarray,
    min_dist_from_obstacle: float = 2.0,
    max_dist_from_obstacle: float = 5.0,
    min_dist_from_drone: float = 5.0,
    max_attempts: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Sample a random collision-free goal near occupied voxels.

    Strategy:
    1. Pick a random occupied voxel.
    2. Offset it by a random direction at a distance in
       [min_dist_from_obstacle, max_dist_from_obstacle].
    3. Verify the candidate is collision-free (inflated grid) and
       at least ``min_dist_from_drone`` from the current position.

    Returns None if no valid goal is found after ``max_attempts`` tries.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get indices of occupied voxels
    occ_indices = np.argwhere(vmap.occupied)
    if len(occ_indices) == 0:
        return None

    for _ in range(max_attempts):
        # 1. Pick a random occupied voxel
        idx = rng.integers(0, len(occ_indices))
        occ_ijk = occ_indices[idx]
        occ_world = vmap.grid_to_world(*occ_ijk)

        # 2. Random offset direction
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        dist = rng.uniform(min_dist_from_obstacle, max_dist_from_obstacle)
        candidate = occ_world + direction * dist

        # 3. Must be inside the map bounds
        if not vmap.in_bounds(*candidate):
            continue

        # 4. Must NOT be in the inflated collision zone
        if vmap.is_collision(*candidate):
            continue

        # 5. Must be far enough from the drone
        if np.linalg.norm(candidate - current_pos) < min_dist_from_drone:
            continue

        return candidate

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Flight executor — dynamics-based path following in Unreal Engine
# ══════════════════════════════════════════════════════════════════════════════

class FlightResult:
    """Result of a single path-following flight."""
    def __init__(self):
        self.success: bool = False
        self.collided: bool = False
        self.collision_info: object | None = None
        self.flight_time: float = 0.0
        self.path_length: float = 0.0
        self.arrival_error: float = 0.0
        self.min_obstacle_clearance: float = float('inf')
        self.trajectory: list[np.ndarray] = []  # actual positions over time


def _subsample_path(waypoints: np.ndarray, min_spacing: float = 1.0) -> np.ndarray:
    """Sub-sample a dense OMPL path so consecutive waypoints are ≥ min_spacing apart.

    Always keeps the first and last waypoint. This avoids feeding hundreds
    of closely-spaced points to moveOnPathAsync which can cause jittery flight.
    """
    if len(waypoints) <= 2:
        return waypoints

    kept = [waypoints[0]]
    for wp in waypoints[1:-1]:
        if np.linalg.norm(wp - kept[-1]) >= min_spacing:
            kept.append(wp)
    kept.append(waypoints[-1])
    return np.array(kept)


def fly_path_on_path(
    client: airsim.MultirotorClient,
    waypoints: np.ndarray,
    velocity: float = 3.0,
    lookahead: float = -1,
    adaptive_lookahead: int = 1,
    min_spacing: float = 1.0,
    viewer: Viewer3D | None = None,
    vmap: VoxelMap | None = None,
    map_points: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    poll_hz: float = 20.0,
) -> FlightResult:
    """Fly the drone through waypoints using AirSim's moveOnPathAsync.

    This uses AirSim's built-in path-following controller which runs the
    full multirotor dynamics simulation inside Unreal Engine — the drone
    physically accelerates, banks, and decelerates through the waypoints.

    During flight the function polls the drone's actual position at
    ``poll_hz`` to:
    - Update the 3-D viewer with the real trajectory
    - Monitor for collisions via ``simGetCollisionInfo``
    - Track minimum clearance to known obstacles

    Parameters
    ----------
    client : MultirotorClient
        Connected and armed AirSim client.
    waypoints : ndarray (N, 3)
        Path waypoints in NED world frame (from OMPL planner).
    velocity : float
        Desired cruise speed in m/s.
    lookahead / adaptive_lookahead
        AirSim path-following parameters.  -1 = auto lookahead.
    min_spacing : float
        Minimum distance (m) between consecutive waypoints sent to AirSim.
    viewer : Viewer3D or None
        If provided, the drone position is updated in real-time.
    vmap : VoxelMap or None
        If provided, obstacle clearance is tracked during flight.
    map_points : ndarray or None
        The voxel map points for viewer updates.
    goal : ndarray or None
        Current goal for viewer target marker.
    poll_hz : float
        How often (Hz) to poll position/collision during flight.

    Returns
    -------
    FlightResult
        Detailed result including collision info and actual trajectory.
    """
    result = FlightResult()
    result.path_length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))

    # Sub-sample for smooth dynamics
    wps = _subsample_path(waypoints, min_spacing=min_spacing)

    # Build AirSim Vector3r path
    path_vec = [airsim.Vector3r(float(wp[0]), float(wp[1]), float(wp[2])) for wp in wps]

    # Configure yaw to face direction of travel
    yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)

    # Compute a generous timeout from path length + margin
    timeout_sec = max(result.path_length / max(velocity, 0.5) * 3.0, 30.0)

    t0 = time.time()
    poll_interval = 1.0 / poll_hz

    # Reset collision state
    client.simGetCollisionInfo()

    # Launch the path-following task (non-blocking)
    future = client.moveOnPathAsync(
        path_vec,
        velocity=velocity,
        timeout_sec=timeout_sec,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=yaw_mode,
        lookahead=lookahead,
        adaptive_lookahead=adaptive_lookahead,
    )

    # ── Real-time monitoring loop ────────────────────────────────────
    while not future._set_flag:
        pos = get_drone_position(client)
        result.trajectory.append(pos.copy())

        # Collision check
        cinfo = client.simGetCollisionInfo()
        if cinfo.has_collided:
            result.collided = True
            result.collision_info = cinfo
            print(f"    !! COLLISION with '{cinfo.object_name}' "
                  f"at ({cinfo.position.x_val:.1f}, {cinfo.position.y_val:.1f}, "
                  f"{cinfo.position.z_val:.1f})")

        # Obstacle clearance (check against raw occupied grid)
        if vmap is not None and vmap.in_bounds(*pos):
            ix, iy, iz = vmap.world_to_grid(*pos)
            if vmap.is_collision(*pos):
                result.min_obstacle_clearance = 0.0
            else:
                # Approximate clearance: distance to nearest occupied voxel
                # in a small neighbourhood (fast local check)
                r = 5  # check 5-voxel radius
                xlo = max(0, ix - r); xhi = min(vmap.nx, ix + r + 1)
                ylo = max(0, iy - r); yhi = min(vmap.ny, iy + r + 1)
                zlo = max(0, iz - r); zhi = min(vmap.nz, iz + r + 1)
                local_occ = vmap.occupied[xlo:xhi, ylo:yhi, zlo:zhi]
                if local_occ.any():
                    occ_local = np.argwhere(local_occ)
                    occ_local[:, 0] += xlo
                    occ_local[:, 1] += ylo
                    occ_local[:, 2] += zlo
                    occ_world = np.column_stack([
                        vmap.origin[0] + (occ_local[:, 0] + 0.5) * vmap.resolution,
                        vmap.origin[1] + (occ_local[:, 1] + 0.5) * vmap.resolution,
                        vmap.origin[2] + (occ_local[:, 2] + 0.5) * vmap.resolution,
                    ])
                    dists = np.linalg.norm(occ_world - pos, axis=1)
                    clearance = float(dists.min())
                    result.min_obstacle_clearance = min(result.min_obstacle_clearance, clearance)

        # Update viewer
        if viewer is not None and map_points is not None:
            viewer.update(
                map_points,
                drone_pos=pos,
                target_pos=goal,
                frontier_points=waypoints,  # show planned path as overlay
            )

        time.sleep(poll_interval)

    result.flight_time = time.time() - t0

    # Final position
    final_pos = get_drone_position(client)
    result.trajectory.append(final_pos.copy())
    result.arrival_error = float(np.linalg.norm(final_pos - waypoints[-1]))
    result.success = (result.arrival_error < 3.0 and not result.collided)

    return result


def fly_path_velocity(
    client: airsim.MultirotorClient,
    waypoints: np.ndarray,
    velocity: float = 3.0,
    arrival_threshold: float = 1.5,
    min_spacing: float = 1.0,
    viewer: Viewer3D | None = None,
    vmap: VoxelMap | None = None,
    map_points: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    poll_hz: float = 20.0,
) -> FlightResult:
    """Pure-pursuit velocity controller — fly the path using moveByVelocityAsync.

    This is a fallback/alternative to moveOnPathAsync that gives more explicit
    control over the drone's velocity vector.  At each step the controller
    computes a velocity toward the next waypoint, advancing to the following
    waypoint when within ``arrival_threshold``.

    The drone physically simulates all dynamics in Unreal Engine.
    """
    result = FlightResult()
    result.path_length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))

    wps = _subsample_path(waypoints, min_spacing=min_spacing)
    poll_interval = 1.0 / poll_hz
    cmd_duration = poll_interval * 3  # velocity command duration

    t0 = time.time()
    wp_idx = 0

    client.simGetCollisionInfo()  # reset

    while wp_idx < len(wps):
        pos = get_drone_position(client)
        result.trajectory.append(pos.copy())

        target_wp = wps[wp_idx]
        to_target = target_wp - pos
        dist = np.linalg.norm(to_target)

        # Advance to next waypoint if close enough
        if dist < arrival_threshold and wp_idx < len(wps) - 1:
            wp_idx += 1
            target_wp = wps[wp_idx]
            to_target = target_wp - pos
            dist = np.linalg.norm(to_target)

        # Final waypoint — tighter threshold
        if wp_idx == len(wps) - 1 and dist < arrival_threshold * 0.5:
            break

        # Velocity toward target (scale speed by distance for smooth decel)
        speed = min(velocity, max(dist * 0.8, 0.5))
        direction = to_target / max(dist, 1e-6)
        vx, vy, vz = direction * speed

        # Yaw toward direction of travel
        yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        client.moveByVelocityAsync(
            float(vx), float(vy), float(vz),
            duration=cmd_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
        )

        # Collision check
        cinfo = client.simGetCollisionInfo()
        if cinfo.has_collided:
            result.collided = True
            result.collision_info = cinfo
            print(f"    !! COLLISION with '{cinfo.object_name}' "
                  f"at ({cinfo.position.x_val:.1f}, {cinfo.position.y_val:.1f}, "
                  f"{cinfo.position.z_val:.1f})")

        # Obstacle clearance
        if vmap is not None and vmap.in_bounds(*pos):
            if vmap.is_collision(*pos):
                result.min_obstacle_clearance = 0.0

        # Update viewer
        if viewer is not None and map_points is not None:
            viewer.update(
                map_points,
                drone_pos=pos,
                target_pos=goal,
                frontier_points=waypoints,
            )

        # Timeout safety
        if time.time() - t0 > result.path_length / max(velocity, 0.5) * 5.0 + 60:
            print("    !! Flight timeout")
            break

        time.sleep(poll_interval)

    # Stop the drone
    client.moveByVelocityAsync(0, 0, 0, duration=1.0).join()

    result.flight_time = time.time() - t0
    final_pos = get_drone_position(client)
    result.trajectory.append(final_pos.copy())
    result.arrival_error = float(np.linalg.norm(final_pos - wps[-1]))
    result.success = (result.arrival_error < 3.0 and not result.collided)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────────
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "flight_recordings")

# The most recent ground-truth recording (or set to a specific path)
MAP_NPZ = ""  # leave empty to auto-detect latest ground_truth_* recording

PLANNING_RESOLUTION = 0.5   # voxel size (m) for the planning grid
INFLATION_RADIUS    = 1.5   # safety margin (m) around obstacles
PLANNER_TYPE        = "RRTstar"
SOLVE_TIMEOUT       = 5.0   # seconds per planning query
PATH_RESOLUTION     = 0.25  # collision-check resolution along edges

VELOCITY            = 3.0   # flight speed (m/s)
TAKEOFF_HEIGHT      = -10.0 # NED z for initial ascent (must be within map Z range)
NUM_GOALS           = 10    # number of random near-obstacle goals to visit
BOUNDS_PADDING      = 20.0  # extra metres around map for OMPL state space

NEAR_OBS_MIN        = 2.0   # min distance from obstacle for goal sampling
NEAR_OBS_MAX        = 5.0   # max distance from obstacle for goal sampling
MIN_GOAL_DIST       = 5.0   # min distance from drone to sampled goal

FLIGHT_MODE         = "path"  # "path" = moveOnPathAsync, "velocity" = pure-pursuit
POLL_HZ             = 20.0   # real-time monitoring rate during flight
MIN_WP_SPACING      = 1.0    # min metres between sub-sampled waypoints


def find_latest_ground_truth() -> str:
    """Find the most recent ground_truth_*/ground_truth.npz recording."""
    import glob
    pattern = os.path.join(RECORDINGS_DIR, "ground_truth_*", "ground_truth.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No ground truth recordings found matching {pattern}\n"
            f"Run displayGroundTruth.py first to create one."
        )
    return matches[-1]


def main():
    t_start = time.time()

    # ── 1) Load voxel map ────────────────────────────────────────────────
    map_path = MAP_NPZ if MAP_NPZ else find_latest_ground_truth()
    print(f"[1/7] Loading voxel map: {map_path}")
    vmap = VoxelMap(resolution=PLANNING_RESOLUTION, inflation_radius=INFLATION_RADIUS)
    vmap.load_npz(map_path)

    # ── 2) Set up OMPL planner ───────────────────────────────────────────
    print(f"[2/7] Setting up OMPL planner ({PLANNER_TYPE}, timeout={SOLVE_TIMEOUT}s)")
    planner = OMPLPlanner(
        vmap,
        planner_type=PLANNER_TYPE,
        solve_timeout=SOLVE_TIMEOUT,
        path_resolution=PATH_RESOLUTION,
        bounds_padding=BOUNDS_PADDING,
    )

    # ── 3) Connect to AirSim ────────────────────────────────────────────
    print("[3/7] Connecting to AirSim ...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    # ── 4) GPS localisation sanity check ──────────────────────────────
    print("[4/7] Checking GPS localisation ...")
    gps_pos   = get_drone_position(client)
    state_est = client.getMultirotorState()
    p_est     = state_est.kinematics_estimated.position
    kin_pos   = np.array([p_est.x_val, p_est.y_val, p_est.z_val])
    offset    = gps_pos - kin_pos

    home = client.getHomeGeoPoint()
    gps  = client.getGpsData().gnss.geo_point
    print(f"  Home GPS : lat={home.latitude:.7f}  lon={home.longitude:.7f}  alt={home.altitude:.2f}")
    print(f"  Drone GPS: lat={gps.latitude:.7f}  lon={gps.longitude:.7f}  alt={gps.altitude:.2f}")
    print(f"  Kinematics NED : ({kin_pos[0]:.2f}, {kin_pos[1]:.2f}, {kin_pos[2]:.2f})")
    print(f"  GPS-derived NED: ({gps_pos[0]:.2f}, {gps_pos[1]:.2f}, {gps_pos[2]:.2f})")
    print(f"  Offset (GPS-Kin): ({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f})")
    if np.linalg.norm(offset) > 1.0:
        print(f"  ⚠  Significant offset detected ({np.linalg.norm(offset):.2f} m) — "
              f"using GPS position for planning")

    # ── 5) Launch viewer ─────────────────────────────────────────────────
    print(f"[5/7] Launching viewer with {len(vmap.points):,} occupied voxels")
    viewer = Viewer3D()
    viewer.start(initial_points=vmap.points)

    # ── 6) Takeoff + escape to collision-free position ─────────────────
    print(f"[6/7] Taking off to z={TAKEOFF_HEIGHT} ...")
    client.takeoffAsync().join()
    time.sleep(1)
    client.moveToPositionAsync(0, 0, TAKEOFF_HEIGHT, velocity=VELOCITY).join()
    time.sleep(1)

    # After takeoff, check if the drone is inside an obstacle (e.g. tree
    # at spawn).  If so, ascend further until clear, then move laterally.
    esc_pos = get_drone_position(client)
    if vmap.in_bounds(*esc_pos) and vmap.is_collision(*esc_pos):
        print(f"  ⚠  Drone is inside inflated obstacle at ({esc_pos[0]:.1f}, "
              f"{esc_pos[1]:.1f}, {esc_pos[2]:.1f}) — escaping ...")
        # Try ascending in 2 m steps
        escape_z = esc_pos[2]
        for _ in range(10):
            escape_z -= 2.0  # go UP in NED (more negative)
            if not vmap.in_bounds(esc_pos[0], esc_pos[1], escape_z):
                break
            if not vmap.is_collision(esc_pos[0], esc_pos[1], escape_z):
                break
        client.moveToPositionAsync(
            float(esc_pos[0]), float(esc_pos[1]), float(escape_z),
            velocity=VELOCITY,
        ).join()
        time.sleep(0.5)
        esc_pos = get_drone_position(client)
        print(f"  Escaped to ({esc_pos[0]:.1f}, {esc_pos[1]:.1f}, {esc_pos[2]:.1f})  "
              f"collision={vmap.is_collision(*esc_pos) if vmap.in_bounds(*esc_pos) else 'out-of-bounds'}")

    # Select flight executor based on mode
    fly_fn = fly_path_on_path if FLIGHT_MODE == "path" else fly_path_velocity
    mode_label = "moveOnPathAsync" if FLIGHT_MODE == "path" else "pure-pursuit velocity"

    # ── Main loop: sample goals near obstacles and fly to them ───────────
    rng = np.random.default_rng(42)
    goals_reached = 0
    goals_failed = 0
    total_collisions = 0
    all_results: list[FlightResult] = []

    print(f"\n{'='*60}")
    print(f"Obstacle-avoidance path planning — {NUM_GOALS} random goals")
    print(f"  Grid: {vmap.nx}×{vmap.ny}×{vmap.nz} @ {PLANNING_RESOLUTION} m")
    print(f"  Inflation: {INFLATION_RADIUS} m  |  Planner: {PLANNER_TYPE}")
    print(f"  Goal sampling: {NEAR_OBS_MIN}-{NEAR_OBS_MAX} m from obstacles")
    print(f"  Flight mode: {mode_label}  |  Poll: {POLL_HZ} Hz")
    print(f"{'='*60}\n")

    for goal_i in range(NUM_GOALS):
        # Current drone position (from GPS)
        current_pos = get_drone_position(client)

        # Update viewer with drone position
        viewer.update(vmap.points, drone_pos=current_pos)

        # Sample a goal near obstacles
        print(f"[Goal {goal_i+1}/{NUM_GOALS}] Sampling near-obstacle goal ...")
        goal = sample_near_obstacle_goal(
            vmap, current_pos,
            min_dist_from_obstacle=NEAR_OBS_MIN,
            max_dist_from_obstacle=NEAR_OBS_MAX,
            min_dist_from_drone=MIN_GOAL_DIST,
            rng=rng,
        )

        if goal is None:
            print(f"  SKIP — could not sample a valid goal after max attempts")
            goals_failed += 1
            continue

        dist = np.linalg.norm(goal - current_pos)
        print(f"  Goal: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})  |  "
              f"Distance: {dist:.1f} m")

        # Show goal + drone in viewer
        viewer.update(vmap.points, drone_pos=current_pos, target_pos=goal)

        # ── Plan path with OMPL ──────────────────────────────────────
        # If the drone is currently inside an obstacle (e.g. after a
        # collision or spawning in a tree), find the nearest free-space
        # point and use that as the planner start instead.
        plan_start = current_pos.copy()
        if vmap.in_bounds(*plan_start) and vmap.is_collision(*plan_start):
            free_pt = find_nearest_free(vmap, plan_start)
            if free_pt is None:
                print(f"  SKIP — drone stuck inside obstacle, no free point found")
                goals_failed += 1
                continue
            print(f"  ⚠  Start in collision — offsetting to free point "
                  f"({free_pt[0]:.1f}, {free_pt[1]:.1f}, {free_pt[2]:.1f})")
            # Physically move the drone to the free point first
            client.moveToPositionAsync(
                float(free_pt[0]), float(free_pt[1]), float(free_pt[2]),
                velocity=VELOCITY,
            ).join()
            time.sleep(0.5)
            plan_start = get_drone_position(client)

        t_plan = time.time()
        path = planner.plan(plan_start, goal)
        dt_plan = time.time() - t_plan

        if path is None:
            print(f"  FAIL — planner could not find a path ({dt_plan:.2f}s)")
            goals_failed += 1
            continue

        print(f"  Path found: {len(path)} waypoints, {dt_plan:.2f}s")

        # Show planned path in viewer (orange frontier overlay)
        viewer.update(vmap.points, drone_pos=current_pos,
                      target_pos=goal, frontier_points=path)
        time.sleep(0.3)  # brief pause to see the planned path before flight

        # ── Execute path with full dynamics ───────────────────────────
        print(f"  Flying ({mode_label}) ...")
        flight = fly_fn(
            client, path,
            velocity=VELOCITY,
            min_spacing=MIN_WP_SPACING,
            viewer=viewer,
            vmap=vmap,
            map_points=vmap.points,
            goal=goal,
            poll_hz=POLL_HZ,
        )
        all_results.append(flight)

        # ── Report ───────────────────────────────────────────────────
        status = "ARRIVED" if flight.success else ("COLLISION" if flight.collided else "MISSED")
        if flight.collided:
            total_collisions += 1
        if flight.success or not flight.collided:
            goals_reached += 1
        else:
            goals_failed += 1

        clearance_str = (f"{flight.min_obstacle_clearance:.2f} m"
                         if flight.min_obstacle_clearance < float('inf')
                         else "n/a")
        traj_len = len(flight.trajectory)
        print(f"  {status} — flight: {flight.flight_time:.1f}s, "
              f"error: {flight.arrival_error:.2f} m, "
              f"clearance: {clearance_str}, "
              f"trajectory: {traj_len} samples")

        if flight.collided:
            ci = flight.collision_info
            print(f"    Collided with: '{ci.object_name}' "
                  f"(penetration: {ci.penetration_depth:.3f})")

        print()

        # Brief pause between goals
        final_pos = get_drone_position(client)
        viewer.update(vmap.points, drone_pos=final_pos, target_pos=goal)
        time.sleep(0.5)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Obstacle avoidance complete")
    print(f"  Goals reached:  {goals_reached}/{NUM_GOALS}")
    print(f"  Goals failed:   {goals_failed}/{NUM_GOALS}")
    print(f"  Collisions:     {total_collisions}/{NUM_GOALS}")
    print(f"  Total time:     {total_time:.1f}s")
    if all_results:
        avg_err = np.mean([r.arrival_error for r in all_results])
        avg_time = np.mean([r.flight_time for r in all_results])
        clearances = [r.min_obstacle_clearance for r in all_results
                      if r.min_obstacle_clearance < float('inf')]
        min_clear = min(clearances) if clearances else float('nan')
        print(f"  Avg arrival err: {avg_err:.2f} m")
        print(f"  Avg flight time: {avg_time:.1f}s")
        print(f"  Min clearance:   {min_clear:.2f} m")
    print(f"{'='*60}")

    print("\nViewer is still open — press Enter to land and exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    # Land
    print("Landing ...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    viewer.stop()
    print("Done.")

    # Suppress the OMPL cleanup crash (known pip package bug)
    os._exit(0)


if __name__ == "__main__":
    main()
