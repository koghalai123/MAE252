#!/usr/bin/env python3
"""Autonomous frontier-based exploration with live SLAM mapping.

Uses LiveSLAM from finalMappingPipeline.py for real-time mapping and an
``ExplorationPlanner`` that identifies frontiers directly from the voxel map
to drive the drone toward un-mapped regions.

The planner works on a coarse 2-D occupancy grid projected from the 3-D voxel
map.  At each decision step it:

1. Projects occupied voxels from the SLAM map onto a 2-D grid.
2. Finds *frontier* cells — unoccupied cells that are neighbours of occupied
   cells (i.e. the edges of the known map).
3. Removes frontiers the drone has already visited (within ``visited_radius``
   of any past scan pose).
4. Clusters the remaining frontiers, scores each cluster by
   information-gain / travel-cost, and returns the best centroid as the
   next waypoint.
"""

from __future__ import annotations

import cosysairsim as airsim
import numpy as np
import time
import os

from scipy import ndimage

from finalMappingPipeline import LiveSLAM, SLAMConfig, SLAMPipeline


# ══════════════════════════════════════════════════════════════════════════════
# ExplorationPlanner
# ══════════════════════════════════════════════════════════════════════════════

class ExplorationPlanner:
    """Map-edge frontier planner for autonomous drone exploration.

    Frontiers are derived entirely from the live voxel map: any unoccupied
    grid cell that neighbours an occupied cell is a frontier.  Cells that the
    drone has already flown near (within ``visited_radius``) are excluded so
    the planner does not re-target already-scanned edges.

    Parameters
    ----------
    bounds : tuple of four floats ``(xmin, xmax, ymin, ymax)``
        Horizontal bounding box of the area to explore (NED frame).
    resolution : float
        Grid cell size in metres for the 2-D planning grid (default 1 m).
    flight_height : float
        NED *z*-coordinate used for all generated waypoints (default -15).
    min_frontier_size : int
        Minimum cells in a frontier cluster for it to be a valid target.
    visited_radius : float
        Cells within this distance (m) of any past scan pose are marked as
        *visited* and excluded from frontier detection.  Increase to avoid
        revisiting; decrease to allow re-scanning the same edges.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        resolution: float = 1.0,
        flight_height: float = -15.0,
        min_frontier_size: int = 3,
        visited_radius: float = 3.0,
    ):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.resolution = resolution
        self.flight_height = flight_height
        self.min_frontier_size = min_frontier_size
        self.visited_radius = visited_radius

        self.nx = int(np.ceil((self.xmax - self.xmin) / resolution))
        self.ny = int(np.ceil((self.ymax - self.ymin) / resolution))

        # Pre-compute circular kernel for visited-zone stamping
        r_cells = int(np.ceil(visited_radius / resolution))
        y_k, x_k = np.ogrid[-r_cells : r_cells + 1, -r_cells : r_cells + 1]
        self._visit_kernel = (
            (x_k ** 2 + y_k ** 2) <= (visited_radius / resolution) ** 2
        )

        # Persistent visited grid — grows monotonically as the drone flies
        self._visited = np.zeros((self.nx, self.ny), dtype=bool)

        # How many past poses we've already stamped into _visited
        self._n_poses_processed: int = 0

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.xmin) / self.resolution)
        iy = int((y - self.ymin) / self.resolution)
        return (
            int(np.clip(ix, 0, self.nx - 1)),
            int(np.clip(iy, 0, self.ny - 1)),
        )

    def _grid_to_world(self, ix: float, iy: float) -> tuple[float, float]:
        x = self.xmin + (ix + 0.5) * self.resolution
        y = self.ymin + (iy + 0.5) * self.resolution
        return x, y

    # ── Visited-zone stamping ─────────────────────────────────────────────

    def _stamp_visited(self, pos) -> None:
        """Mark grid cells within *visited_radius* of *pos* as visited."""
        cx, cy = self._world_to_grid(float(pos[0]), float(pos[1]))
        r = self._visit_kernel.shape[0] // 2

        x0, x1 = max(cx - r, 0), min(cx + r + 1, self.nx)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, self.ny)

        kx0 = x0 - (cx - r)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (cy - r)
        ky1 = ky0 + (y1 - y0)

        self._visited[x0:x1, y0:y1] |= self._visit_kernel[kx0:kx1, ky0:ky1]

    # ── Core planner ──────────────────────────────────────────────────────

    def next_target(
        self,
        pipeline: SLAMPipeline,
        current_pos,
    ) -> tuple[np.ndarray | None, dict]:
        """Determine the next exploration waypoint from the live voxel map.

        Frontiers are unoccupied cells adjacent to occupied cells, minus any
        cells the drone has already visited.

        Parameters
        ----------
        pipeline : SLAMPipeline
            The live SLAM pipeline whose map and pose history are queried.
        current_pos : array-like, shape (3,)
            Drone's current NED position ``[x, y, z]``.

        Returns
        -------
        target : ndarray shape (3,) **or** ``None``
            Next waypoint ``[x, y, z]`` in NED.  ``None`` when no frontiers
            remain (map edges fully visited).
        info : dict
            ``n_frontier_cells``, ``n_frontier_cells_raw`` (before visited
            filter), ``n_occupied_cells``, ``n_clusters``.
        """
        current_pos = np.asarray(current_pos, dtype=float)

        # 1) Update visited grid from new scan poses ──────────────────────
        raw_poses = pipeline._scan_T_raw
        for T in raw_poses[self._n_poses_processed :]:
            self._stamp_visited(T[:3, 3])
        self._n_poses_processed = len(raw_poses)
        self._stamp_visited(current_pos)

        # 2) Build 2-D occupied grid from 3-D voxel map ──────────────────
        occupied_grid = np.zeros((self.nx, self.ny), dtype=bool)
        vis_pts = pipeline.get_map_points()

        info: dict = {
            "n_frontier_cells": 0,
            "n_frontier_cells_raw": 0,
            "n_occupied_cells": 0,
            "n_clusters": 0,
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
        occupied_grid[ix, iy] = True
        info["n_occupied_cells"] = int(occupied_grid.sum())

        # 3) Find map-edge frontiers ──────────────────────────────────────
        #    frontier = unoccupied cells that are 4-connected neighbours of
        #    at least one occupied cell.
        struct = ndimage.generate_binary_structure(2, 1)   # 4-connected
        dilated = ndimage.binary_dilation(occupied_grid, structure=struct)
        frontier_raw = dilated & ~occupied_grid            # map edges

        n_frontier_raw = int(frontier_raw.sum())
        info["n_frontier_cells_raw"] = n_frontier_raw

        # 4) Remove already-visited frontier cells ────────────────────────
        frontier_mask = frontier_raw & ~self._visited

        n_frontier = int(frontier_mask.sum())
        info["n_frontier_cells"] = n_frontier

        # Build 3-D world coordinates for all frontier cells (for viewer overlay)
        fc_all = np.argwhere(frontier_mask)
        if len(fc_all) > 0:
            frontier_world = np.column_stack([
                self.xmin + (fc_all[:, 0] + 0.5) * self.resolution,
                self.ymin + (fc_all[:, 1] + 0.5) * self.resolution,
                np.full(len(fc_all), self.flight_height),
            ])
            info["frontier_world_pts"] = frontier_world
        else:
            info["frontier_world_pts"] = np.empty((0, 3), dtype=np.float64)

        if n_frontier == 0:
            return None, info

        # 5) Cluster frontiers and score ──────────────────────────────────
        labeled, n_clusters = ndimage.label(frontier_mask, structure=struct)
        info["n_clusters"] = n_clusters

        cur_xy = current_pos[:2]
        best_score = -np.inf
        best_target = None

        for cid in range(1, n_clusters + 1):
            cells = np.argwhere(labeled == cid)
            if len(cells) < self.min_frontier_size:
                continue

            # Cluster centroid → world coords
            cx_mean = float(cells[:, 0].mean())
            cy_mean = float(cells[:, 1].mean())
            wx, wy = self._grid_to_world(cx_mean, cy_mean)

            # Keep target inside bounds with a small margin
            wx = float(np.clip(wx, self.xmin + 1, self.xmax - 1))
            wy = float(np.clip(wy, self.ymin + 1, self.ymax - 1))

            # Reject if the target cell itself is occupied
            tix, tiy = self._world_to_grid(wx, wy)
            if occupied_grid[tix, tiy]:
                continue

            gain = float(len(cells))                # information gain
            dist = max(float(np.linalg.norm(np.array([wx, wy]) - cur_xy)), 0.5)
            score = gain / dist                     # benefit / cost

            if score > best_score:
                best_score = score
                best_target = np.array([wx, wy, self.flight_height])

        # Fallback: nearest individual frontier cell if all clusters too small
        if best_target is None and n_frontier > 0:
            fc = np.argwhere(frontier_mask)
            fw = np.column_stack([
                self.xmin + (fc[:, 0] + 0.5) * self.resolution,
                self.ymin + (fc[:, 1] + 0.5) * self.resolution,
            ])
            dists = np.linalg.norm(fw - cur_xy.reshape(1, 2), axis=1)
            nearest = fw[np.argmin(dists)]
            best_target = np.array([nearest[0], nearest[1], self.flight_height])

        return best_target, info


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SAVE_DIR = os.path.join(os.path.dirname(__file__), "flight_recordings")

# Area to fully explore (xmin, xmax, ymin, ymax) in NED
EXPLORE_BOUNDS  = (-15, 30, -30, 5)
FLIGHT_HEIGHT   = -15.0         # NED z for waypoints
VELOCITY        = 2             # m/s
SCAN_HZ         = 1 / 2.5      # scans per second
PLANNER_RES     = 1.0           # planning grid cell size (m)
VISITED_RADIUS  = 3.0           # exclude map edges within this range of past poses (m)
MAX_TARGETS     = 50            # safety cap on autonomous waypoints

# ══════════════════════════════════════════════════════════════════════════════
# Set up SLAM pipeline
# ══════════════════════════════════════════════════════════════════════════════

cfg = SLAMConfig(
    registration="vgicp",
    octo_resolution=0.15,
    frame_skip=1,
    live_max_hz=SCAN_HZ,
    enable_viewer=True,
)

live = LiveSLAM(cfg)
live.connect()

out_dir = os.path.join(SAVE_DIR, f"exploration_{int(time.time())}")
live.enable_recording(out_dir)
live.pipeline.start_viewer()

# ══════════════════════════════════════════════════════════════════════════════
# Takeoff and rise to exploration altitude
# ══════════════════════════════════════════════════════════════════════════════

print("Taking off...")
live.client.takeoffAsync().join()
time.sleep(1)

print(f"Rising to altitude z={FLIGHT_HEIGHT} ...")
future = live.client.moveToPositionAsync(0, 0, FLIGHT_HEIGHT, velocity=VELOCITY)
while not future._set_flag:
    live.process_once()
    time.sleep(0.001)
# Collect a few scans at the starting position
for _ in range(5):
    live.process_once()
    time.sleep(0.1)

# ══════════════════════════════════════════════════════════════════════════════
# Autonomous frontier exploration loop
# ══════════════════════════════════════════════════════════════════════════════

planner = ExplorationPlanner(
    bounds=EXPLORE_BOUNDS,
    resolution=PLANNER_RES,
    flight_height=FLIGHT_HEIGHT,
    min_frontier_size=3,
    visited_radius=VISITED_RADIUS,
)

print(f"\n{'='*60}")
print(f"Autonomous exploration started")
print(f"  Bounds: x=[{EXPLORE_BOUNDS[0]}, {EXPLORE_BOUNDS[1]}], "
      f"y=[{EXPLORE_BOUNDS[2]}, {EXPLORE_BOUNDS[3]}]")
print(f"  Grid:  {planner.nx} x {planner.ny} cells @ {PLANNER_RES} m")
print(f"  Visited radius: {VISITED_RADIUS} m  |  Max waypoints: {MAX_TARGETS}")
print(f"{'='*60}\n")

wp_count = 0

while wp_count < MAX_TARGETS:
    # Current drone position
    state = live.client.getMultirotorState()
    p = state.kinematics_estimated.position
    current_pos = np.array([p.x_val, p.y_val, p.z_val])

    # Ask the planner
    target, info = planner.next_target(live.pipeline, current_pos)

    # Send frontier overlay to the 3-D viewer (orange points)
    live.set_frontier_points(info.get("frontier_world_pts"))

    print(f"  [{wp_count + 1:02d}] Occupied: {info['n_occupied_cells']} | "
          f"Frontiers: {info['n_frontier_cells']}/{info['n_frontier_cells_raw']} "
          f"(after visited filter) in {info['n_clusters']} clusters")

    if target is None:
        print("\n  Exploration complete — no unvisited map-edge frontiers remain")
        break

    print(f"       -> target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

    # Show markers in viewer
    live.set_target(target.tolist())
    wp_count += 1

    # Fly to the target
    future = live.client.moveToPositionAsync(
        float(target[0]), float(target[1]), float(target[2]),
        velocity=VELOCITY,
    )
    while not future._set_flag:
        live.process_once()
        time.sleep(0.001)

    # Hover and scan at the waypoint to fill in detail
    for _ in range(5):
        live.process_once()
        time.sleep(0.1)

print(f"\nExploration finished after {wp_count} waypoints.")

# ══════════════════════════════════════════════════════════════════════════════
# Finalise — save outputs but keep the viewer open for inspection
# ══════════════════════════════════════════════════════════════════════════════

# Correct the map and save, but do NOT stop the viewer
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
