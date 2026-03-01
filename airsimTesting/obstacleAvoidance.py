#!/usr/bin/env python3
"""PathPlanner — OMPL-based 3-D path planning through three-state voxel maps.

Provides a :class:`PathPlanner` that accepts a map with three voxel states
(**free**, **occupied**, **unknown**) and plans collision-free paths through
known-free space using OMPL's sampling-based planners.

**Key design**: Unknown voxels are treated as **obstacles** (conservative
planning).  The planner will only route paths through voxels that have been
observed and found to be empty.  This is the correct behaviour for autonomous
exploration where the drone must not fly into unscanned regions.

Quick usage from ``exploration.py``::

    from obstacleAvoidance import PathPlanner

    planner = PathPlanner(inflation_radius=1.5, ground_z=0.0)
    planner.update_map(observed, occupied,
                       origin=np.array([xmin, ymin, zmin]),
                       resolution=1.0)
    path = planner.plan(current_pos, target_pos)
    if path is not None:
        print(f"Found path with {len(path)} waypoints")

The module also includes AirSim flight-executor utilities and a standalone
test that loads a ground-truth voxel map and flies random obstacle-avoidance
missions in the simulator (``python obstacleAvoidance.py``).
"""

from __future__ import annotations

import math
import numpy as np
import os
import time
from typing import TYPE_CHECKING

from ompl import base as ob
from ompl import geometric as og

if TYPE_CHECKING:
    import cosysairsim as airsim


# ══════════════════════════════════════════════════════════════════════════════
# PathPlanner
# ══════════════════════════════════════════════════════════════════════════════

class PathPlanner:
    """OMPL-based 3-D path planner for three-state voxel maps.

    Accepts a map with three voxel states:

    - **Free**: observed and unoccupied — the drone may fly here.
    - **Occupied**: contains an obstacle.
    - **Unknown**: not yet observed by any sensor — treated as impassable.

    An inflation (safety margin) is applied around occupied voxels so
    planned paths maintain a safe clearance distance.

    Parameters
    ----------
    inflation_radius : float
        Safety margin (metres) inflated around occupied voxels.
    planner_type : str
        OMPL planner algorithm.  ``"ABITstar"`` (near-optimal, fast) is
        the default.  Other options: ``"RRTstar"``, ``"RRTConnect"``
        (fastest but non-optimal), ``"BITstar"``, ``"InformedRRTstar"``.
    solve_timeout : float
        Maximum time (seconds) for a single planning query.  Anytime
        planners stop early once a first feasible solution is found, then
        refine for up to ``refine_time`` additional seconds.
    refine_time : float
        Time (seconds) for optional post-solve path refinement.
    ground_z : float
        Maximum allowed NED Z (more positive = deeper underground).
        States below this are always invalid.  Set to ``np.inf`` to disable.
    simplify_time : float
        Time limit (seconds) for OMPL path simplification.
    range_m : float
        Maximum edge length for tree-based planners (metres).
    """

    def __init__(
        self,
        inflation_radius: float = 1.5,
        planner_type: str = "ABITstar",
        solve_timeout: float = 2.0,
        refine_time: float = 0.5,
        ground_z: float = 0.0,
        simplify_time: float = 0.5,
        range_m: float = 5.0,
    ):
        self.inflation_radius = inflation_radius
        self.planner_type = planner_type
        self.solve_timeout = solve_timeout
        self.refine_time = refine_time
        self.ground_z = ground_z
        self.simplify_time = simplify_time
        self.range_m = range_m

        # Map state (set by update_map)
        self._traversable: np.ndarray | None = None   # observed & ~inflated
        self._occupied_grid: np.ndarray | None = None  # raw occupied
        self._observed_grid: np.ndarray | None = None  # raw observed
        self._inflated_grid: np.ndarray | None = None  # inflated occupied
        self._origin = np.zeros(3, dtype=np.float64)
        self._resolution: float = 1.0
        self._nx = self._ny = self._nz = 0

        # Raw occupied points (world frame) — kept for viewer display
        self._points: np.ndarray = np.empty((0, 3), dtype=np.float32)

        # OMPL internals (built on first update_map)
        self._space: ob.RealVectorStateSpace | None = None
        self._si: ob.SpaceInformation | None = None

    # ── Map update ────────────────────────────────────────────────────────

    def update_map(
        self,
        observed: np.ndarray,
        occupied: np.ndarray,
        origin: np.ndarray | tuple | list,
        resolution: float,
    ) -> None:
        """Update the planner's map from three-state voxel grids.

        Call this whenever the SLAM map changes (new scans registered).
        The method re-inflates occupied voxels and rebuilds the OMPL
        state space.

        Parameters
        ----------
        observed : bool ndarray, shape (nx, ny, nz)
            True for voxels that have been observed by the sensor.
        occupied : bool ndarray, shape (nx, ny, nz)
            True for voxels that contain obstacles.  Should generally be
            a subset of *observed*, though this is not enforced.
        origin : array-like, shape (3,)
            World-frame NED position of the grid's minimum corner
            ``(x_min, y_min, z_min)``.
        resolution : float
            Voxel edge length in metres.
        """
        self._observed_grid = np.asarray(observed, dtype=bool)
        self._occupied_grid = np.asarray(occupied, dtype=bool)
        self._origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
        self._resolution = float(resolution)
        self._nx, self._ny, self._nz = self._observed_grid.shape

        # Inflate occupied voxels with a spherical safety margin
        self._inflate()

        # Traversable = observed AND not in inflated obstacle zone
        self._traversable = self._observed_grid & ~self._inflated_grid

        # Rebuild OMPL state space and validity checker
        self._build_ompl()

        n_free = int(self._traversable.sum())
        n_obs = int(self._observed_grid.sum())
        n_occ = int(self._occupied_grid.sum())
        n_total = self._nx * self._ny * self._nz
        print(f"  [PathPlanner] Grid {self._nx}x{self._ny}x{self._nz} "
              f"@ {resolution} m  |  observed {n_obs}/{n_total}  "
              f"occ {n_occ}  free {n_free}")

    # ── Path planning ─────────────────────────────────────────────────────

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        simplify: bool = True,
    ) -> np.ndarray | None:
        """Plan a collision-free path through known-free space.

        Parameters
        ----------
        start, goal : array-like, shape (3,)
            Start and goal positions in NED world frame.
        simplify : bool
            Run OMPL's path simplifier to smooth/shorten the result.

        Returns
        -------
        ndarray, shape (N, 3) or None
            Waypoints (NED) along the path, or ``None`` if no path is found.
        """
        if self._si is None:
            raise RuntimeError("Call update_map() before plan()")

        start = np.asarray(start, dtype=float).ravel()[:3]
        goal = np.asarray(goal, dtype=float).ravel()[:3]

        # Auto-fix invalid start: if outside grid or not traversable,
        # snap to the nearest valid position.
        if not self._is_valid_point(start):
            fixed = self.find_nearest_free(start, margin=3.0)
            if fixed is None:
                fixed = self.find_nearest_free(start)  # try without margin
            if fixed is None:
                print(f"  [plan] Cannot fix invalid start "
                      f"({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})")
                return None
            print(f"  [plan] Auto-adjusted start from "
                  f"({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f}) to "
                  f"({fixed[0]:.1f}, {fixed[1]:.1f}, {fixed[2]:.1f})")
            start = fixed

        pdef = ob.ProblemDefinition(self._si)

        s = ob.State(self._space)
        s[0], s[1], s[2] = float(start[0]), float(start[1]), float(start[2])
        g = ob.State(self._space)
        g[0], g[1], g[2] = float(goal[0]), float(goal[1]), float(goal[2])

        pdef.setStartAndGoalStates(s, g)
        pdef.setOptimizationObjective(
            ob.PathLengthOptimizationObjective(self._si))

        planner = self._make_planner()
        planner.setProblemDefinition(pdef)
        planner.setup()

        # ── Solve with early termination for anytime planners ────────
        FEASIBILITY_PLANNERS = {"RRT", "RRTConnect", "PRM"}
        try:
            if self.planner_type in FEASIBILITY_PLANNERS:
                solved = planner.solve(self.solve_timeout)
            else:
                # Phase 1: find ANY feasible solution (or timeout)
                ptc_exact = ob.exactSolnPlannerTerminationCondition(pdef)
                ptc_time = ob.timedPlannerTerminationCondition(
                    self.solve_timeout)
                ptc = ob.plannerOrTerminationCondition(ptc_exact, ptc_time)
                solved = planner.solve(ptc)
                # Phase 2: brief refinement
                if solved:
                    planner.solve(ob.timedPlannerTerminationCondition(
                        min(self.refine_time, self.solve_timeout * 0.25)))
        except (AttributeError, TypeError):
            # Fallback if termination helpers are unavailable
            solved = planner.solve(self.solve_timeout)

        if not solved:
            return None

        path = pdef.getSolutionPath()

        if simplify:
            simplifier = og.PathSimplifier(self._si)
            simplifier.simplify(path, self.simplify_time)

        path.interpolate()   # densify for smooth flight

        states = path.getStates()
        return np.array([[st[0], st[1], st[2]] for st in states],
                        dtype=np.float64)

    # ── Query helpers ─────────────────────────────────────────────────────

    def _is_valid_point(self, pt: np.ndarray) -> bool:
        """True if the point would be accepted as an OMPL start/goal."""
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        if not self.is_in_grid(x, y, z):
            return False
        return self.is_traversable(x, y, z)

    def is_traversable(self, x: float, y: float, z: float) -> bool:
        """True if the point is in known-free (observed & not inflated) space."""
        if self._traversable is None:
            return False
        ix, iy, iz = self._world_to_grid_unchecked(x, y, z)
        if not (0 <= ix < self._nx and 0 <= iy < self._ny
                and 0 <= iz < self._nz):
            return False
        return bool(self._traversable[ix, iy, iz])

    def is_collision(self, x: float, y: float, z: float) -> bool:
        """True if the point is in the inflated-occupied zone."""
        if self._inflated_grid is None:
            return False
        ix, iy, iz = self._world_to_grid_unchecked(x, y, z)
        if not (0 <= ix < self._nx and 0 <= iy < self._ny
                and 0 <= iz < self._nz):
            return False
        return bool(self._inflated_grid[ix, iy, iz])

    def is_occupied(self, x: float, y: float, z: float) -> bool:
        """True if the point is in a raw-occupied voxel (no inflation)."""
        if self._occupied_grid is None:
            return False
        ix, iy, iz = self._world_to_grid_unchecked(x, y, z)
        if not (0 <= ix < self._nx and 0 <= iy < self._ny
                and 0 <= iz < self._nz):
            return False
        return bool(self._occupied_grid[ix, iy, iz])

    def is_in_grid(self, x: float, y: float, z: float) -> bool:
        """True if the point falls inside the grid's bounding box."""
        o, r = self._origin, self._resolution
        return (o[0] <= x < o[0] + self._nx * r
                and o[1] <= y < o[1] + self._ny * r
                and o[2] <= z < o[2] + self._nz * r)

    def world_to_grid(
        self, x: float, y: float, z: float,
    ) -> tuple[int, int, int]:
        """Convert world NED to grid indices (clipped to bounds)."""
        res = self._resolution
        ix = int(np.clip(int((x - self._origin[0]) / res), 0, self._nx - 1))
        iy = int(np.clip(int((y - self._origin[1]) / res), 0, self._ny - 1))
        iz = int(np.clip(int((z - self._origin[2]) / res), 0, self._nz - 1))
        return ix, iy, iz

    def grid_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        """Convert grid indices to world NED (voxel centre)."""
        return (self._origin
                + np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * self._resolution)

    def find_nearest_free(
        self, pos: np.ndarray, max_radius: float = 10.0,
        margin: float = 0.0,
    ) -> np.ndarray | None:
        """Find the nearest traversable point via expanding search.

        Tries ascending first (most common escape in NED), then expands
        in all directions.  Returns ``None`` if nothing found within
        *max_radius*.

        Parameters
        ----------
        margin : float
            Minimum distance (m) from the grid boundary that the returned
            point must satisfy.  Use this to avoid picking edge voxels
            that the drone may overshoot.
        """
        pos = np.asarray(pos, dtype=float).ravel()[:3]
        step = self._resolution

        def _inside_margin(pt: np.ndarray) -> bool:
            if margin <= 0:
                return True
            o, r = self._origin, self._resolution
            return (o[0] + margin <= pt[0] <= o[0] + self._nx * r - margin
                    and o[1] + margin <= pt[1] <= o[1] + self._ny * r - margin
                    and o[2] + margin <= pt[2] <= o[2] + self._nz * r - margin)

        for dz in np.arange(-step, -max_radius, -step):
            cand = pos + np.array([0.0, 0.0, dz])
            if self.is_traversable(*cand) and _inside_margin(cand):
                return cand

        for r in np.arange(step, max_radius + step, step):
            for dx in [-r, 0.0, r]:
                for dy in [-r, 0.0, r]:
                    for dz in [-r, 0.0, r]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        cand = pos + np.array([dx, dy, dz])
                        if self.is_traversable(*cand) and _inside_margin(cand):
                            return cand
        return None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def origin(self) -> np.ndarray:
        """Grid origin (minimum corner in world frame)."""
        return self._origin.copy()

    @property
    def resolution(self) -> float:
        """Grid voxel size in metres."""
        return self._resolution

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Grid dimensions ``(nx, ny, nz)``."""
        return (self._nx, self._ny, self._nz)

    @property
    def occupied(self) -> np.ndarray | None:
        """Raw occupied grid ``(nx, ny, nz)`` — no inflation."""
        return self._occupied_grid

    @property
    def observed(self) -> np.ndarray | None:
        """Observed (sensor-covered) grid ``(nx, ny, nz)``."""
        return self._observed_grid

    @property
    def traversable(self) -> np.ndarray | None:
        """Traversable grid ``(nx, ny, nz)`` — observed & not inflated."""
        return self._traversable

    @property
    def points(self) -> np.ndarray:
        """Raw occupied points in world frame (for viewer display)."""
        return self._points

    @points.setter
    def points(self, value: np.ndarray) -> None:
        self._points = np.asarray(value, dtype=np.float32)

    # ── Internal ──────────────────────────────────────────────────────────

    def _world_to_grid_unchecked(
        self, x: float, y: float, z: float,
    ) -> tuple[int, int, int]:
        """Convert world coords to grid indices without boundary clamping."""
        res = self._resolution
        return (int((x - self._origin[0]) / res),
                int((y - self._origin[1]) / res),
                int((z - self._origin[2]) / res))

    def _inflate(self) -> None:
        """Inflate occupied voxels by a spherical structuring element."""
        from scipy import ndimage

        r_vox = max(1, int(np.ceil(self.inflation_radius / self._resolution)))
        diam = 2 * r_vox + 1
        struct = np.zeros((diam, diam, diam), dtype=bool)
        c = r_vox
        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dz in range(-r_vox, r_vox + 1):
                    if dx * dx + dy * dy + dz * dz <= r_vox * r_vox:
                        struct[c + dx, c + dy, c + dz] = True

        self._inflated_grid = ndimage.binary_dilation(
            self._occupied_grid, structure=struct)

    def _build_ompl(self) -> None:
        """Build OMPL state space, bounds, and validity checker."""
        self._space = ob.RealVectorStateSpace(3)

        bounds = ob.RealVectorBounds(3)
        pad = max(self._resolution * 4, 2.0)  # generous padding for edge overshoot
        for i in range(3):
            bounds.setLow(i, float(self._origin[i]) - pad)
        bounds.setHigh(
            0, float(self._origin[0]) + self._nx * self._resolution + pad)
        bounds.setHigh(
            1, float(self._origin[1]) + self._ny * self._resolution + pad)
        z_max = (float(self._origin[2])
                 + self._nz * self._resolution + pad)
        bounds.setHigh(2, min(z_max, self.ground_z))

        self._space.setBounds(bounds)

        self._si = ob.SpaceInformation(self._space)
        self._si.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._is_valid))
        self._si.setStateValidityCheckingResolution(
            self._resolution / self._space.getMaximumExtent())
        self._si.setup()

    def _is_valid(self, state) -> bool:
        """OMPL validity checker — True only for known-free space."""
        x, y, z = state[0], state[1], state[2]

        # Ground plane constraint
        if z > self.ground_z:
            return False

        # Grid index (no clamp)
        res = self._resolution
        ix = int((x - self._origin[0]) / res)
        iy = int((y - self._origin[1]) / res)
        iz = int((z - self._origin[2]) / res)

        # Out of grid -> unknown -> invalid
        if not (0 <= ix < self._nx and 0 <= iy < self._ny
                and 0 <= iz < self._nz):
            return False

        return bool(self._traversable[ix, iy, iz])

    def _make_planner(self):
        """Instantiate the configured OMPL planner."""
        planners = {
            "RRTstar": og.RRTstar,
            "RRTConnect": og.RRTConnect,
            "RRT": og.RRT,
            "PRM": og.PRM,
            "BITstar": og.BITstar,
            "ABITstar": og.ABITstar,
            "AITstar": og.AITstar,
            "InformedRRTstar": og.InformedRRTstar,
        }
        cls = planners.get(self.planner_type, og.RRTstar)
        p = cls(self._si)
        if hasattr(p, 'setRange'):
            p.setRange(self.range_m)
        return p


# ══════════════════════════════════════════════════════════════════════════════
# Path utilities
# ══════════════════════════════════════════════════════════════════════════════

def subsample_path(
    waypoints: np.ndarray, min_spacing: float = 1.0,
) -> np.ndarray:
    """Sub-sample a dense path so consecutive waypoints are >= *min_spacing* apart.

    Always keeps the first and last waypoint.
    """
    if len(waypoints) <= 2:
        return waypoints

    kept = [waypoints[0]]
    for wp in waypoints[1:-1]:
        if np.linalg.norm(wp - kept[-1]) >= min_spacing:
            kept.append(wp)
    kept.append(waypoints[-1])
    return np.array(kept)


def straight_line_free(
    planner: PathPlanner,
    start: np.ndarray,
    end: np.ndarray,
    step: float = 0.3,
) -> bool:
    """True if the straight line from *start* to *end* is fully traversable.

    Marches along the segment and checks each sample against the planner's
    traversability grid.  Fails if any point is unknown, occupied, or in
    the inflated zone.
    """
    diff = end - start
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return True
    direction = diff / dist
    n_samples = int(np.ceil(dist / step)) + 1
    for i in range(n_samples):
        t = min(i * step, dist)
        pt = start + direction * t
        if not planner.is_traversable(*pt):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# GPS-based drone localisation (no cosysairsim import needed)
# ══════════════════════════════════════════════════════════════════════════════

def gps_to_ned(
    gps_lat: float, gps_lon: float, gps_alt: float,
    home_lat: float, home_lon: float, home_alt: float,
) -> np.ndarray:
    """Convert GPS geodetic coordinates to local NED relative to *home*.

    Uses a flat-earth approximation with the WGS-84 semi-major axis.
    """
    R_EARTH = 6_378_137.0
    d_lat = math.radians(gps_lat - home_lat)
    d_lon = math.radians(gps_lon - home_lon)
    north = d_lat * R_EARTH
    east = d_lon * R_EARTH * math.cos(math.radians(home_lat))
    down = -(gps_alt - home_alt)
    return np.array([north, east, down])


def get_drone_position(client: airsim.MultirotorClient) -> np.ndarray:
    """Return the drone's NED position derived from its simulated GPS."""
    home = client.getHomeGeoPoint()
    gps = client.getGpsData().gnss.geo_point
    return gps_to_ned(
        gps.latitude, gps.longitude, gps.altitude,
        home.latitude, home.longitude, home.altitude,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Flight executors (require cosysairsim at call time)
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
        self.trajectory: list[np.ndarray] = []


# ══════════════════════════════════════════════════════════════════════════════
# PathFollower — threaded path executor
# ══════════════════════════════════════════════════════════════════════════════

import threading
from enum import Enum, auto
from typing import Callable


class FollowerState(Enum):
    """Current state of the PathFollower."""
    IDLE = auto()       # No path; hovering or waiting
    FOLLOWING = auto()  # Actively flying a path
    ARRIVED = auto()    # Reached the last waypoint
    PREEMPTED = auto()  # Path was replaced before completion
    STOPPED = auto()    # Follower thread shut down


class PathFollower:
    """Threaded path-following executor for AirSim drones.

    Runs its own background thread that autonomously follows waypoint
    paths using either ``moveOnPathAsync`` or pure-pursuit velocity
    control.  New paths can be submitted at any time, preempting the
    current one.

    Quick usage::

        follower = PathFollower(client, planner, mode="velocity")
        follower.start()

        # Submit a path (non-blocking — returns immediately)
        follower.follow(waypoints, goal=goal_pos)

        # Meanwhile the main thread can do SLAM, replan, etc.
        while follower.state == FollowerState.FOLLOWING:
            buf.process_once()
            time.sleep(0.01)

        result = follower.last_result

        # Submit a new path (preempts if still flying)
        follower.follow(new_waypoints, goal=new_goal)

        # When done:
        follower.stop()

    Parameters
    ----------
    client : airsim.MultirotorClient
        Connected AirSim client (must already have API control).
    planner : PathPlanner | None
        If provided, used for collision/clearance monitoring.
    mode : str
        ``"path"`` for ``moveOnPathAsync``, ``"velocity"`` for
        pure-pursuit velocity control.
    velocity : float
        Target flight speed (m/s).
    arrival_threshold : float
        Distance (m) to final waypoint to count as arrived.
    poll_hz : float
        Control loop frequency (Hz).
    min_spacing : float
        Minimum spacing between waypoints after sub-sampling.
    viewer : object | None
        Optional ``Viewer3D`` for real-time display updates.
    on_tick : callable | None
        Called once per control tick with signature
        ``on_tick(pos: np.ndarray, follower: PathFollower)``.
        Use this for SLAM scan collection or any per-tick work.
    """

    def __init__(
        self,
        client,
        planner: PathPlanner | None = None,
        mode: str = "velocity",
        velocity: float = 3.0,
        arrival_threshold: float = 1.5,
        poll_hz: float = 20.0,
        min_spacing: float = 1.0,
        viewer=None,
        on_tick: Callable[[np.ndarray, 'PathFollower'], None] | None = None,
    ):
        self.client = client
        self.planner = planner
        self.mode = mode
        self.velocity = velocity
        self.arrival_threshold = arrival_threshold
        self.poll_hz = poll_hz
        self.min_spacing = min_spacing
        self.viewer = viewer
        self.on_tick = on_tick

        # ── Thread state ──────────────────────────────────────────────
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._new_path_event = threading.Event()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._done_event.set()  # initially "done" (no flight pending)

        # Current mission (protected by _lock)
        self._waypoints: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._map_points: np.ndarray | None = None
        self._preempt = False

        # Public readable state
        self._state = FollowerState.IDLE
        self._last_result: FlightResult | None = None
        self._position: np.ndarray = np.zeros(3)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def state(self) -> FollowerState:
        """Current follower state (thread-safe read)."""
        return self._state

    @property
    def last_result(self) -> FlightResult | None:
        """Result from the most recently completed (or preempted) path."""
        return self._last_result

    @property
    def position(self) -> np.ndarray:
        """Most recent drone position (NED), updated each tick."""
        return self._position.copy()

    @property
    def is_busy(self) -> bool:
        """True if actively following a path."""
        return self._state == FollowerState.FOLLOWING

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background follower thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._state = FollowerState.IDLE
        self._thread = threading.Thread(
            target=self._run_loop, name="PathFollower", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the follower to stop and wait for the thread to exit."""
        self._stop_event.set()
        self._new_path_event.set()  # wake the thread if it's waiting
        self._done_event.set()     # unblock any pending wait()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._state = FollowerState.STOPPED

    def follow(
        self,
        waypoints: np.ndarray,
        goal: np.ndarray | None = None,
        map_points: np.ndarray | None = None,
    ) -> None:
        """Submit a new path to follow (non-blocking, preempts current).

        Parameters
        ----------
        waypoints : ndarray (N, 3)
            Waypoints in NED world frame.
        goal : ndarray (3,) | None
            Final goal position (for viewer display; defaults to last wp).
        map_points : ndarray (M, 3) | None
            Occupied points for viewer display.  If ``None`` and a planner
            is set, uses ``planner.points``.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if goal is None:
            goal = waypoints[-1].copy()
        if map_points is None and self.planner is not None:
            map_points = self.planner.points

        self._done_event.clear()
        with self._lock:
            self._waypoints = waypoints
            self._goal = goal
            self._map_points = map_points
            self._preempt = True
        self._state = FollowerState.FOLLOWING
        self._new_path_event.set()

    def wait(self, timeout: float | None = None) -> FlightResult | None:
        """Block until the current path completes or is preempted.

        Returns the ``FlightResult``, or ``None`` on timeout.
        """
        if not self._done_event.wait(timeout=timeout):
            return None  # timed out
        return self._last_result

    # ── Main thread loop ─────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Background thread: wait for paths, fly them, repeat."""
        # AirSim's msgpack-rpc client binds its asyncio transport to
        # the event loop of the thread that created it.  A background
        # thread therefore needs its OWN client connection.
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        import cosysairsim as airsim
        self._thread_client = airsim.MultirotorClient()
        self._thread_client.confirmConnection()
        self._thread_client.enableApiControl(True)
        self._thread_client.armDisarm(True)
        print("  [PathFollower] Thread-local AirSim client connected")

        while not self._stop_event.is_set():
            # Wait for a path or stop signal
            self._new_path_event.wait(timeout=0.5)
            self._new_path_event.clear()

            if self._stop_event.is_set():
                break

            # Grab the pending path
            with self._lock:
                if self._waypoints is None:
                    continue
                waypoints = self._waypoints.copy()
                goal = self._goal.copy() if self._goal is not None else None
                map_points = (self._map_points.copy()
                              if self._map_points is not None else None)
                self._waypoints = None
                self._preempt = False

            self._state = FollowerState.FOLLOWING

            try:
                if self.mode == "path":
                    result = self._fly_on_path(waypoints, goal, map_points)
                else:
                    result = self._fly_velocity(waypoints, goal, map_points)
            except Exception as e:
                print(f"  [PathFollower] Flight error: {e}")
                result = FlightResult()

            self._last_result = result

            # Was this preempted? (a new path arrived mid-flight)
            with self._lock:
                if self._preempt or self._waypoints is not None:
                    self._state = FollowerState.PREEMPTED
                    self._done_event.set()
                    continue

            self._state = (FollowerState.ARRIVED if result.success
                           else FollowerState.IDLE)
            self._done_event.set()

    # ── Flight implementations ───────────────────────────────────────

    def _fly_on_path(
        self, waypoints: np.ndarray, goal: np.ndarray | None,
        map_points: np.ndarray | None,
    ) -> FlightResult:
        """Follow waypoints using ``moveOnPathAsync``."""
        import cosysairsim as airsim

        result = FlightResult()
        result.path_length = float(np.sum(np.linalg.norm(
            np.diff(waypoints, axis=0), axis=1)))

        wps = subsample_path(waypoints, min_spacing=self.min_spacing)
        path_vec = [airsim.Vector3r(float(wp[0]), float(wp[1]), float(wp[2]))
                    for wp in wps]
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        timeout_sec = max(
            result.path_length / max(self.velocity, 0.5) * 3.0, 30.0)

        poll_interval = 1.0 / self.poll_hz
        t0 = time.time()

        self._thread_client.simGetCollisionInfo()  # reset

        future = self._thread_client.moveOnPathAsync(
            path_vec, velocity=self.velocity, timeout_sec=timeout_sec,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=yaw_mode, lookahead=-1, adaptive_lookahead=1,
        )

        while not future._set_flag:
            if self._stop_event.is_set() or self._preempt_check():
                try:
                    self._thread_client.cancelLastTask()
                except Exception:
                    self._thread_client.moveByVelocityAsync(0, 0, 0, 1).join()
                break

            pos = get_drone_position(self._thread_client)
            self._position = pos
            result.trajectory.append(pos.copy())

            self._check_collision(result)
            self._track_clearance(result, pos)
            self._update_viewer(pos, goal, map_points, waypoints)

            if self.on_tick is not None:
                try:
                    self.on_tick(pos, self)
                except Exception:
                    pass

            time.sleep(poll_interval)

        result.flight_time = time.time() - t0
        final_pos = get_drone_position(self._thread_client)
        self._position = final_pos
        result.trajectory.append(final_pos.copy())
        result.arrival_error = float(np.linalg.norm(final_pos - wps[-1]))
        result.success = (result.arrival_error < self.arrival_threshold * 2
                          and not result.collided)
        return result

    def _fly_velocity(
        self, waypoints: np.ndarray, goal: np.ndarray | None,
        map_points: np.ndarray | None,
    ) -> FlightResult:
        """Follow waypoints using pure-pursuit velocity commands."""
        import cosysairsim as airsim

        result = FlightResult()
        result.path_length = float(np.sum(np.linalg.norm(
            np.diff(waypoints, axis=0), axis=1)))

        wps = subsample_path(waypoints, min_spacing=self.min_spacing)
        poll_interval = 1.0 / self.poll_hz
        cmd_duration = poll_interval * 3

        t0 = time.time()
        wp_idx = 0
        self._thread_client.simGetCollisionInfo()

        while wp_idx < len(wps):
            if self._stop_event.is_set() or self._preempt_check():
                break

            pos = get_drone_position(self._thread_client)
            self._position = pos
            result.trajectory.append(pos.copy())

            target_wp = wps[wp_idx]
            to_target = target_wp - pos
            dist = np.linalg.norm(to_target)

            if dist < self.arrival_threshold and wp_idx < len(wps) - 1:
                wp_idx += 1
                target_wp = wps[wp_idx]
                to_target = target_wp - pos
                dist = np.linalg.norm(to_target)

            if (wp_idx == len(wps) - 1
                    and dist < self.arrival_threshold * 0.5):
                break

            speed = min(self.velocity, max(dist * 0.8, 0.5))
            direction = to_target / max(dist, 1e-6)
            vx, vy, vz = direction * speed

            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

            self._thread_client.moveByVelocityAsync(
                float(vx), float(vy), float(vz),
                duration=cmd_duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
            )

            self._check_collision(result)
            self._track_clearance(result, pos)
            self._update_viewer(pos, goal, map_points, waypoints)

            if self.on_tick is not None:
                try:
                    self.on_tick(pos, self)
                except Exception:
                    pass

            if (time.time() - t0
                    > result.path_length / max(self.velocity, 0.5) * 5 + 60):
                print("    !! Flight timeout")
                break

            time.sleep(poll_interval)

        self._thread_client.moveByVelocityAsync(0, 0, 0, duration=1.0).join()

        result.flight_time = time.time() - t0
        final_pos = get_drone_position(self._thread_client)
        self._position = final_pos
        result.trajectory.append(final_pos.copy())
        result.arrival_error = float(np.linalg.norm(final_pos - wps[-1]))
        result.success = (result.arrival_error < self.arrival_threshold * 2
                          and not result.collided)
        return result

    # ── Helpers ───────────────────────────────────────────────────────

    def _preempt_check(self) -> bool:
        """True if a new path has been submitted (should preempt)."""
        with self._lock:
            return self._preempt or self._waypoints is not None

    def _check_collision(self, result: FlightResult) -> None:
        cinfo = self._thread_client.simGetCollisionInfo()
        if cinfo.has_collided:
            result.collided = True
            result.collision_info = cinfo
            print(f"    !! COLLISION with '{cinfo.object_name}' "
                  f"at ({cinfo.position.x_val:.1f}, "
                  f"{cinfo.position.y_val:.1f}, "
                  f"{cinfo.position.z_val:.1f})")

    def _track_clearance(
        self, result: FlightResult, pos: np.ndarray,
    ) -> None:
        planner = self.planner
        if planner is None or not planner.is_in_grid(*pos):
            return
        if planner.is_collision(*pos):
            result.min_obstacle_clearance = 0.0
            return
        ix, iy, iz = planner.world_to_grid(*pos)
        nx, ny, nz = planner.grid_shape
        occ = planner.occupied
        if occ is None:
            return
        r = 5
        xlo, xhi = max(0, ix - r), min(nx, ix + r + 1)
        ylo, yhi = max(0, iy - r), min(ny, iy + r + 1)
        zlo, zhi = max(0, iz - r), min(nz, iz + r + 1)
        local_occ = occ[xlo:xhi, ylo:yhi, zlo:zhi]
        if not local_occ.any():
            return
        occ_local = np.argwhere(local_occ)
        occ_local[:, 0] += xlo
        occ_local[:, 1] += ylo
        occ_local[:, 2] += zlo
        orig = planner.origin
        res = planner.resolution
        occ_world = np.column_stack([
            orig[0] + (occ_local[:, 0] + 0.5) * res,
            orig[1] + (occ_local[:, 1] + 0.5) * res,
            orig[2] + (occ_local[:, 2] + 0.5) * res,
        ])
        clearance = float(np.linalg.norm(occ_world - pos, axis=1).min())
        result.min_obstacle_clearance = min(
            result.min_obstacle_clearance, clearance)

    def _update_viewer(
        self, pos: np.ndarray, goal: np.ndarray | None,
        map_points: np.ndarray | None, waypoints: np.ndarray,
    ) -> None:
        if self.viewer is not None and map_points is not None:
            self.viewer.update(
                map_points, drone_pos=pos, target_pos=goal,
                frontier_points=waypoints)


# ══════════════════════════════════════════════════════════════════════════════
# ThreadedPathPlanner — background planning while flying
# ══════════════════════════════════════════════════════════════════════════════

class PlanResult:
    """Result of a background planning request."""
    def __init__(self):
        self.path: np.ndarray | None = None
        self.goal: np.ndarray = np.zeros(3)
        self.start: np.ndarray = np.zeros(3)
        self.plan_time: float = 0.0
        self.success: bool = False


class ThreadedPathPlanner:
    """Threaded wrapper around :class:`PathPlanner` for concurrent planning.

    Plans paths on a dedicated background thread so the drone can fly one
    path while the next is being computed — eliminating pauses between
    waypoint segments.

    Quick usage::

        tp = ThreadedPathPlanner(planner)
        tp.start()

        # Submit a plan request (returns immediately)
        tp.request_plan(start_pos, goal_pos)

        # ... fly the current path ...

        # Pick up the result (blocks until ready)
        result = tp.get_result(timeout=10.0)
        if result and result.success:
            follower.follow(result.path, goal=result.goal)

        tp.stop()

    Parameters
    ----------
    planner : PathPlanner
        The underlying synchronous planner instance.
    """

    def __init__(self, planner: PathPlanner):
        self.planner = planner

        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()  # protects request/result state
        self._plan_lock = threading.Lock()  # serialises plan() + update_map()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._result_event = threading.Event()

        # Request state
        self._req_start: np.ndarray | None = None
        self._req_goal: np.ndarray | None = None

        # Result state
        self._result: PlanResult | None = None
        self._is_planning = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background planner thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ThreadedPlanner", daemon=True)
        self._thread.start()
        print("  [ThreadedPathPlanner] Background thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background planner thread."""
        self._stop_event.set()
        self._request_event.set()  # wake the thread
        self._result_event.set()   # unblock any pending get_result()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── Public API ────────────────────────────────────────────────────

    def request_plan(
        self, start: np.ndarray, goal: np.ndarray,
    ) -> None:
        """Submit a planning request (non-blocking, returns immediately).

        If a previous request is still being computed it will finish
        first; the new request is queued after it.
        """
        with self._lock:
            self._req_start = np.asarray(start, dtype=float).ravel()[:3].copy()
            self._req_goal = np.asarray(goal, dtype=float).ravel()[:3].copy()
            self._result = None
            self._result_event.clear()
        self._request_event.set()

    def get_result(self, timeout: float | None = None) -> PlanResult | None:
        """Block until the current plan result is available.

        Returns ``None`` on timeout.
        """
        if not self._result_event.wait(timeout=timeout):
            return None
        return self._result

    @property
    def result_ready(self) -> bool:
        """True if a plan result is available without blocking."""
        return self._result_event.is_set()

    @property
    def last_result(self) -> PlanResult | None:
        """Most recent plan result (may be ``None``)."""
        return self._result

    @property
    def is_planning(self) -> bool:
        """True if the background thread is currently planning."""
        return self._is_planning

    def update_map(self, *args, **kwargs) -> None:
        """Thread-safe proxy for ``planner.update_map(...)``.

        Acquires the plan lock so an in-progress plan finishes before
        the map is swapped.
        """
        with self._plan_lock:
            self.planner.update_map(*args, **kwargs)

    def plan_sync(self, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        """Synchronous (blocking) plan — useful for the first path.

        Runs on the calling thread but still acquires the plan lock so
        it will not collide with a background request.
        """
        result = PlanResult()
        result.start = np.asarray(start, dtype=float).ravel()[:3].copy()
        result.goal = np.asarray(goal, dtype=float).ravel()[:3].copy()
        t0 = time.time()
        with self._plan_lock:
            path = self.planner.plan(result.start, result.goal)
        result.plan_time = time.time() - t0
        result.path = path
        result.success = path is not None
        return result

    # ── Background thread ─────────────────────────────────────────────

    def _run(self) -> None:
        """Background loop: wait for requests, compute paths."""
        while not self._stop_event.is_set():
            self._request_event.wait(timeout=0.5)
            self._request_event.clear()

            if self._stop_event.is_set():
                break

            # Grab the pending request
            with self._lock:
                if self._req_start is None or self._req_goal is None:
                    continue
                start = self._req_start.copy()
                goal = self._req_goal.copy()
                self._req_start = None
                self._req_goal = None

            self._is_planning = True
            result = PlanResult()
            result.start = start
            result.goal = goal

            t0 = time.time()
            try:
                with self._plan_lock:
                    path = self.planner.plan(start, goal)
                result.plan_time = time.time() - t0
                result.path = path
                result.success = path is not None
                if result.success:
                    print(f"  [ThreadedPlanner] Path ready: "
                          f"{len(path)} wps in {result.plan_time:.2f}s")
                else:
                    print(f"  [ThreadedPlanner] No path found "
                          f"({result.plan_time:.2f}s)")
            except Exception as e:
                result.plan_time = time.time() - t0
                result.success = False
                print(f"  [ThreadedPlanner] Error: {e}")

            self._is_planning = False
            self._result = result
            self._result_event.set()


# ══════════════════════════════════════════════════════════════════════════════
# Goal sampler (for standalone testing)
# ══════════════════════════════════════════════════════════════════════════════

def sample_near_obstacle_goal(
    planner: PathPlanner,
    current_pos: np.ndarray,
    min_dist_from_obstacle: float = 2.0,
    max_dist_from_obstacle: float = 5.0,
    min_dist_from_drone: float = 5.0,
    max_attempts: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Sample a random collision-free goal near occupied voxels.

    The goal must NOT be reachable via a straight line from the drone,
    forcing the planner to navigate around obstacles.
    """
    if rng is None:
        rng = np.random.default_rng()

    occ = planner.occupied
    if occ is None:
        return None
    occ_indices = np.argwhere(occ)
    if len(occ_indices) == 0:
        return None

    for _ in range(max_attempts):
        idx = rng.integers(0, len(occ_indices))
        occ_ijk = occ_indices[idx]
        occ_world = planner.grid_to_world(*occ_ijk)

        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        dist = rng.uniform(min_dist_from_obstacle, max_dist_from_obstacle)
        candidate = occ_world + direction * dist

        # Must be traversable (observed, not occupied, not inflated)
        if not planner.is_traversable(*candidate):
            continue

        # Must be above the ground plane
        if candidate[2] > planner.ground_z:
            continue

        # Must be far enough from the drone
        if np.linalg.norm(candidate - current_pos) < min_dist_from_drone:
            continue

        # Straight line must be BLOCKED -- forces non-trivial planning
        if straight_line_free(planner, current_pos, candidate, step=0.3):
            continue

        return candidate

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth map loader (for standalone testing)
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth_map(
    npz_path: str,
    resolution: float = 0.5,
    inflation_radius: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a ground-truth .npz and build observed/occupied grids.

    For ground truth, every voxel in the bounding box is treated as
    *observed* (the entire volume was scanned).

    Returns
    -------
    observed, occupied : bool ndarray (nx, ny, nz)
    origin : ndarray (3,)
    resolution : float
    """
    data = np.load(npz_path)
    pts = data["points"].astype(np.float32)

    res = resolution
    pad = inflation_radius + res
    mins = pts.min(axis=0) - pad
    maxs = pts.max(axis=0) + pad
    origin = mins.copy()

    nx = int(np.ceil((maxs[0] - mins[0]) / res))
    ny = int(np.ceil((maxs[1] - mins[1]) / res))
    nz = int(np.ceil((maxs[2] - mins[2]) / res))

    occupied = np.zeros((nx, ny, nz), dtype=bool)
    ix = np.clip(((pts[:, 0] - origin[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - origin[1]) / res).astype(int), 0, ny - 1)
    iz = np.clip(((pts[:, 2] - origin[2]) / res).astype(int), 0, nz - 1)
    occupied[ix, iy, iz] = True

    # Ground truth: the full volume has been scanned
    observed = np.ones((nx, ny, nz), dtype=bool)

    print(f"  [load_ground_truth_map] {len(pts):,} points -> "
          f"{nx}x{ny}x{nz} grid @ {res} m")

    return observed, occupied, origin, res


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test -- obstacle avoidance missions in AirSim
# ══════════════════════════════════════════════════════════════════════════════

# -- Configuration ---------------------------------------------------------
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "flight_recordings")
MAP_NPZ = ""  # leave empty to auto-detect latest ground_truth_*

PLANNING_RESOLUTION = 0.5
INFLATION_RADIUS    = 1.5
PLANNER_TYPE        = "ABITstar"
SOLVE_TIMEOUT       = 2.0
GROUND_Z            = 0.0

VELOCITY            = 3.0
TAKEOFF_HEIGHT      = -10.0
NUM_GOALS           = 10

NEAR_OBS_MIN        = 2.0
NEAR_OBS_MAX        = 5.0
MIN_GOAL_DIST       = 5.0

FLIGHT_MODE         = "path"   # "path" or "velocity"
POLL_HZ             = 20.0
MIN_WP_SPACING      = 1.0


def find_latest_ground_truth() -> str:
    """Find the most recent ground_truth_*/ground_truth.npz recording."""
    import glob
    pattern = os.path.join(RECORDINGS_DIR, "ground_truth_*", "ground_truth.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No ground truth recordings found matching {pattern}\n"
            f"Run displayGroundTruth.py first to create one.")
    return matches[-1]


def main():
    import cosysairsim as airsim
    from sensorFeed import Viewer3D

    t_start = time.time()

    # -- 1) Load voxel map -------------------------------------------------
    map_path = MAP_NPZ if MAP_NPZ else find_latest_ground_truth()
    print(f"[1/7] Loading voxel map: {map_path}")
    observed, occupied, origin, res = load_ground_truth_map(
        map_path, resolution=PLANNING_RESOLUTION,
        inflation_radius=INFLATION_RADIUS)

    # -- 2) Set up planner -------------------------------------------------
    print(f"[2/7] Setting up PathPlanner ({PLANNER_TYPE}, "
          f"timeout={SOLVE_TIMEOUT}s)")
    planner = PathPlanner(
        inflation_radius=INFLATION_RADIUS,
        planner_type=PLANNER_TYPE,
        solve_timeout=SOLVE_TIMEOUT,
        ground_z=GROUND_Z,
    )
    planner.update_map(observed, occupied, origin, res)

    # Store occupied points for viewer display
    data = np.load(map_path)
    planner.points = data["points"].astype(np.float32)

    # -- 3) Connect to AirSim ----------------------------------------------
    print("[3/7] Connecting to AirSim ...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    # -- 4) GPS localisation sanity check ----------------------------------
    print("[4/7] Checking GPS localisation ...")
    gps_pos = get_drone_position(client)
    state_est = client.getMultirotorState()
    p_est = state_est.kinematics_estimated.position
    kin_pos = np.array([p_est.x_val, p_est.y_val, p_est.z_val])
    offset = gps_pos - kin_pos

    home = client.getHomeGeoPoint()
    gps = client.getGpsData().gnss.geo_point
    print(f"  Home GPS : lat={home.latitude:.7f}  lon={home.longitude:.7f}  "
          f"alt={home.altitude:.2f}")
    print(f"  Drone GPS: lat={gps.latitude:.7f}  lon={gps.longitude:.7f}  "
          f"alt={gps.altitude:.2f}")
    print(f"  Kinematics NED : ({kin_pos[0]:.2f}, {kin_pos[1]:.2f}, "
          f"{kin_pos[2]:.2f})")
    print(f"  GPS-derived NED: ({gps_pos[0]:.2f}, {gps_pos[1]:.2f}, "
          f"{gps_pos[2]:.2f})")
    print(f"  Offset (GPS-Kin): ({offset[0]:.2f}, {offset[1]:.2f}, "
          f"{offset[2]:.2f})")
    if np.linalg.norm(offset) > 1.0:
        print(f"  Warning: significant offset ({np.linalg.norm(offset):.2f} m)")

    # -- 5) Launch viewer --------------------------------------------------
    print(f"[5/7] Launching viewer with {len(planner.points):,} "
          f"occupied voxels")
    viewer = Viewer3D()
    viewer.start(initial_points=planner.points)

    # -- 6) Set up ThreadedPathPlanner early (plans during takeoff) -------
    threaded_planner = ThreadedPathPlanner(planner)
    threaded_planner.start()

    rng = np.random.default_rng(42)

    # Sample the first goal using the *expected* post-takeoff position
    # and submit it to the background planner BEFORE takeoff, so the
    # planning runs concurrently with the takeoff manoeuvre.
    expected_takeoff_pos = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
    print("[6/8] Sampling first goal and planning during takeoff ...")
    first_goal = None
    for _ in range(20):
        first_goal = sample_near_obstacle_goal(
            planner, expected_takeoff_pos,
            min_dist_from_obstacle=NEAR_OBS_MIN,
            max_dist_from_obstacle=NEAR_OBS_MAX,
            min_dist_from_drone=MIN_GOAL_DIST,
            rng=rng,
        )
        if first_goal is not None:
            break
    if first_goal is not None:
        dist = np.linalg.norm(first_goal - expected_takeoff_pos)
        print(f"  First goal: ({first_goal[0]:.1f}, {first_goal[1]:.1f}, "
              f"{first_goal[2]:.1f})  |  Distance: {dist:.1f} m")
        threaded_planner.request_plan(expected_takeoff_pos, first_goal)
    else:
        print("  Warning: could not sample first goal pre-takeoff")

    # -- 7) Takeoff + escape (planning runs concurrently) ------------------
    print(f"[7/8] Taking off to z={TAKEOFF_HEIGHT} ...")
    client.takeoffAsync().join()
    time.sleep(1)
    client.moveToPositionAsync(
        0, 0, TAKEOFF_HEIGHT, velocity=VELOCITY).join()
    client.hoverAsync().join()

    esc_pos = get_drone_position(client)
    escaped = False
    if planner.is_in_grid(*esc_pos) and planner.is_collision(*esc_pos):
        print(f"  Warning: drone inside inflated obstacle at "
              f"({esc_pos[0]:.1f}, {esc_pos[1]:.1f}, {esc_pos[2]:.1f}) "
              f"-- escaping ...")
        escape_z = esc_pos[2]
        for _ in range(10):
            escape_z -= 2.0
            if not planner.is_in_grid(esc_pos[0], esc_pos[1], escape_z):
                break
            if not planner.is_collision(esc_pos[0], esc_pos[1], escape_z):
                break
        client.moveToPositionAsync(
            float(esc_pos[0]), float(esc_pos[1]), float(escape_z),
            velocity=VELOCITY).join()
        client.hoverAsync().join()
        time.sleep(0.5)
        esc_pos = get_drone_position(client)
        in_coll = (planner.is_collision(*esc_pos)
                   if planner.is_in_grid(*esc_pos) else "out-of-bounds")
        print(f"  Escaped to ({esc_pos[0]:.1f}, {esc_pos[1]:.1f}, "
              f"{esc_pos[2]:.1f})  collision={in_coll}")
        escaped = True

    mode_label = ("moveOnPathAsync" if FLIGHT_MODE == "path"
                  else "pure-pursuit velocity")

    # -- 8) Set up PathFollower --------------------------------------------
    follower = PathFollower(
        client, planner=planner, mode=FLIGHT_MODE,
        velocity=VELOCITY, poll_hz=POLL_HZ,
        min_spacing=MIN_WP_SPACING, viewer=viewer,
    )
    follower.start()

    # -- 9) Main loop: pipelined plan-while-fly ----------------------------
    #
    # Strategy: while the drone flies path N, the planner thread is
    # already computing path N+1.  When the flight finishes the next
    # path is (usually) ready immediately — no pause.
    #
    # If the drone's actual arrival position differs significantly from
    # the assumed start of the pre-planned path, a quick synchronous
    # replan is done as a fallback.
    #
    REPLAN_THRESHOLD = 3.0  # m — replan if arrival drifts more than this

    goals_reached = 0
    goals_failed = 0
    total_collisions = 0
    all_results: list[FlightResult] = []

    nx, ny, nz = planner.grid_shape
    print(f"\n{'='*60}")
    print(f"Obstacle-avoidance path planning -- {NUM_GOALS} random goals")
    print(f"  Grid: {nx}x{ny}x{nz} @ {PLANNING_RESOLUTION} m")
    print(f"  Inflation: {INFLATION_RADIUS} m  |  Planner: {PLANNER_TYPE}")
    print(f"  Goal sampling: {NEAR_OBS_MIN}-{NEAR_OBS_MAX} m from obstacles")
    print(f"  Flight mode: {mode_label}  |  Poll: {POLL_HZ} Hz")
    print(f"  Pipelined planning: ON (plan-while-fly)")
    print(f"{'='*60}\n")

    # ── Helper: recover drone into valid grid position ────────────────
    def recover_start(pos: np.ndarray) -> np.ndarray | None:
        """Try to move the drone to a valid start position."""
        plan_start = pos.copy()
        RECOVERY_MARGIN = 5.0
        for _recovery in range(3):
            need_recovery = False
            reason = ""
            if not planner.is_in_grid(*plan_start):
                need_recovery = True
                reason = "outside grid"
            elif planner.is_collision(*plan_start):
                need_recovery = True
                reason = "in collision"
            if not need_recovery:
                return plan_start

            free_pt = planner.find_nearest_free(
                plan_start, margin=RECOVERY_MARGIN)
            if free_pt is None:
                free_pt = planner.find_nearest_free(plan_start)
            if free_pt is None:
                return None

            print(f"  Warning: start {reason} -- flying to "
                  f"({free_pt[0]:.1f}, {free_pt[1]:.1f}, {free_pt[2]:.1f})")
            client.moveToPositionAsync(
                float(free_pt[0]), float(free_pt[1]), float(free_pt[2]),
                velocity=VELOCITY).join()
            time.sleep(1.0)
            plan_start = get_drone_position(client)

        # 3 recovery attempts exhausted
        if not planner.is_in_grid(*plan_start):
            return None
        return plan_start

    # ── Helper: process a completed flight ────────────────────────────
    def process_flight(flight: FlightResult | None, goal_i: int,
                       goal: np.ndarray) -> bool:
        """Log flight result. Returns True if goal was reached."""
        nonlocal goals_reached, goals_failed, total_collisions

        if flight is None:
            print(f"  FAIL -- flight returned no result")
            goals_failed += 1
            return False

        all_results.append(flight)

        status = ("ARRIVED" if flight.success
                  else ("COLLISION" if flight.collided else "MISSED"))
        if flight.collided:
            total_collisions += 1
        if flight.success or not flight.collided:
            goals_reached += 1
        else:
            goals_failed += 1

        clearance_str = (
            f"{flight.min_obstacle_clearance:.2f} m"
            if flight.min_obstacle_clearance < float('inf') else "n/a")
        print(f"  {status} -- flight: {flight.flight_time:.1f}s, "
              f"error: {flight.arrival_error:.2f} m, "
              f"clearance: {clearance_str}, "
              f"trajectory: {len(flight.trajectory)} samples")

        if flight.collided:
            ci = flight.collision_info
            print(f"    Collided with: '{ci.object_name}' "
                  f"(penetration: {ci.penetration_depth:.3f})")
        print()
        return flight.success or not flight.collided

    # ── Collect the pre-planned first path ────────────────────────────
    # The background planner was started before takeoff.  Collect it now.
    current_pos = get_drone_position(client)
    viewer.update(planner.points, drone_pos=current_pos)

    first_path = None
    if first_goal is not None:
        pr = threaded_planner.get_result(timeout=30.0)
        if pr is not None and pr.success:
            # Check if escape moved us far from the planned start
            drift = np.linalg.norm(current_pos - pr.start)
            if drift <= REPLAN_THRESHOLD and not escaped:
                first_path = pr.path
                print(f"  Pre-planned first path ready: "
                      f"{len(first_path)} wps, {pr.plan_time:.2f}s "
                      f"(drift {drift:.2f} m)")
            else:
                # Escape moved us — replan from actual position
                print(f"  Drift {drift:.2f} m from planned start "
                      f"— replanning from actual position ...")
                plan_start = recover_start(current_pos)
                if plan_start is not None:
                    pr2 = threaded_planner.plan_sync(plan_start, first_goal)
                    if pr2.success:
                        first_path = pr2.path
                        print(f"  Replan OK: {len(first_path)} wps, "
                              f"{pr2.plan_time:.2f}s")
        else:
            print(f"  Pre-plan failed — trying synchronous plan ...")
            plan_start = recover_start(current_pos)
            if plan_start is not None:
                pr2 = threaded_planner.plan_sync(plan_start, first_goal)
                if pr2.success:
                    first_path = pr2.path

    # If pre-plan didn't work, try sampling fresh goals
    if first_path is None:
        for _attempt in range(NUM_GOALS):
            if first_goal is None:
                print(f"[Goal 1/{NUM_GOALS}] Sampling near-obstacle goal ...")
                first_goal = sample_near_obstacle_goal(
                    planner, current_pos,
                    min_dist_from_obstacle=NEAR_OBS_MIN,
                    max_dist_from_obstacle=NEAR_OBS_MAX,
                    min_dist_from_drone=MIN_GOAL_DIST,
                    rng=rng,
                )
            if first_goal is None:
                print(f"  SKIP -- could not sample a valid goal")
                goals_failed += 1
                continue

            plan_start = recover_start(current_pos)
            if plan_start is None:
                print(f"  SKIP -- could not recover into grid")
                goals_failed += 1
                first_goal = None
                continue

            pr = threaded_planner.plan_sync(plan_start, first_goal)
            print(f"  Path {'found' if pr.success else 'FAILED'}: "
                  f"{len(pr.path) if pr.path is not None else 0} wps, "
                  f"{pr.plan_time:.2f}s")
            if pr.success:
                first_path = pr.path
                break
            else:
                goals_failed += 1
                first_goal = None

    if first_path is None or first_goal is None:
        print("  Could not plan any initial path — aborting.")
    else:
        # Start flying the first path
        viewer.update(planner.points, drone_pos=current_pos,
                      target_pos=first_goal, frontier_points=first_path)
        print(f"  Flying ({mode_label}) ...")
        follower.follow(first_path, goal=first_goal,
                        map_points=planner.points)

        # Keep track of the "expected arrival position" so the planner
        # thread can start the next plan from it.
        prev_goal = first_goal.copy()
        goals_sampled = 1  # how many goals consumed from NUM_GOALS budget

        # ── Pipeline: plan next while flying current ─────────────────
        while goals_sampled < NUM_GOALS:
            next_goal_idx = goals_sampled + 1  # 1-based display index

            # (a) Sample + request next plan while drone is still flying
            print(f"[Goal {next_goal_idx}/{NUM_GOALS}] "
                  f"Sampling next goal (planner thread) ...")
            next_goal = sample_near_obstacle_goal(
                planner, prev_goal,  # use expected arrival as reference
                min_dist_from_obstacle=NEAR_OBS_MIN,
                max_dist_from_obstacle=NEAR_OBS_MAX,
                min_dist_from_drone=MIN_GOAL_DIST,
                rng=rng,
            )
            if next_goal is not None:
                dist = np.linalg.norm(next_goal - prev_goal)
                print(f"  Goal: ({next_goal[0]:.1f}, {next_goal[1]:.1f}, "
                      f"{next_goal[2]:.1f})  |  Distance: {dist:.1f} m")
                threaded_planner.request_plan(prev_goal, next_goal)
            else:
                print(f"  SKIP -- could not sample a valid goal")

            # (b) Wait for current flight to finish
            flight = follower.wait()
            process_flight(flight, goals_sampled, prev_goal)
            goals_sampled += 1

            if next_goal is None:
                goals_failed += 1  # already counted skip above
                # Need a fresh synchronous plan from actual position
                current_pos = get_drone_position(client)
                viewer.update(planner.points, drone_pos=current_pos)
                # Try to sample again from actual position
                next_goal = sample_near_obstacle_goal(
                    planner, current_pos,
                    min_dist_from_obstacle=NEAR_OBS_MIN,
                    max_dist_from_obstacle=NEAR_OBS_MAX,
                    min_dist_from_drone=MIN_GOAL_DIST,
                    rng=rng,
                )
                if next_goal is None:
                    prev_goal = current_pos
                    continue
                plan_start = recover_start(current_pos)
                if plan_start is None:
                    prev_goal = current_pos
                    continue
                pr = threaded_planner.plan_sync(plan_start, next_goal)
                if not pr.success:
                    prev_goal = current_pos
                    continue
                follower.follow(pr.path, goal=next_goal,
                                map_points=planner.points)
                prev_goal = next_goal.copy()
                continue

            # (c) Collect the pre-planned result
            plan_result = threaded_planner.get_result(timeout=30.0)

            actual_pos = get_drone_position(client)
            viewer.update(planner.points, drone_pos=actual_pos,
                          target_pos=next_goal)

            path = None

            if plan_result is not None and plan_result.success:
                drift = np.linalg.norm(actual_pos - plan_result.start)
                if drift <= REPLAN_THRESHOLD:
                    # Pre-planned path is usable
                    path = plan_result.path
                    print(f"  Using pre-planned path "
                          f"(drift {drift:.2f} m <= {REPLAN_THRESHOLD} m)")
                else:
                    # Arrival drifted — quick synchronous replan
                    print(f"  Drift {drift:.2f} m > {REPLAN_THRESHOLD} m "
                          f"— replanning from actual position ...")
                    plan_start = recover_start(actual_pos)
                    if plan_start is not None:
                        pr = threaded_planner.plan_sync(plan_start, next_goal)
                        if pr.success:
                            path = pr.path
                            print(f"  Replan OK: {len(path)} wps, "
                                  f"{pr.plan_time:.2f}s")
            else:
                # Background plan failed — try synchronous from actual pos
                print(f"  Pre-plan failed — replanning synchronously ...")
                plan_start = recover_start(actual_pos)
                if plan_start is not None:
                    pr = threaded_planner.plan_sync(plan_start, next_goal)
                    if pr.success:
                        path = pr.path

            if path is None:
                print(f"  FAIL -- no usable path to goal")
                goals_failed += 1
                prev_goal = actual_pos
                continue

            # (d) Immediately start flying the next path (no pause!)
            viewer.update(planner.points, drone_pos=actual_pos,
                          target_pos=next_goal, frontier_points=path)
            print(f"  Flying ({mode_label}) ...")
            follower.follow(path, goal=next_goal,
                            map_points=planner.points)
            prev_goal = next_goal.copy()

        # Wait for the very last flight to finish
        if follower.is_busy:
            flight = follower.wait()
            process_flight(flight, goals_sampled, prev_goal)

        final_pos = get_drone_position(client)
        viewer.update(planner.points, drone_pos=final_pos)

    # -- Summary -----------------------------------------------------------
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

    follower.stop()
    threaded_planner.stop()

    print("\nViewer is still open -- press Enter to land and exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

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
