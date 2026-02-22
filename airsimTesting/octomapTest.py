#!/usr/bin/env python3
"""
OctoMap test — replay a recorded flight into a pyoctomap octree and visualise
the occupied voxel-centres via the existing Viewer3D (Open3D in a subprocess).

Usage:
    python octomapTest.py                      # uses latest recording
    python octomapTest.py /path/to/flight_dir  # explicit directory

This is a standalone verification before integrating octomap into the full
ICP mapping pipeline.
"""

import os, sys, glob, time
import numpy as np
import pyoctomap
from scipy.spatial.transform import Rotation

from sensorFeed import Viewer3D          # reuse existing visualiser

# ── Configuration ─────────────────────────────────────────────────────────────
OCTO_RESOLUTION = 0.10           # voxel (leaf) size in metres
VIS_EVERY       = 1              # extract & update viewer every N frames
RECORDING_DIR   = ""             # empty = latest in flight_recordings/
MAX_RANGE       = -1.0           # ray max range for insertPointCloud (-1=unlimited)
USE_RAYCASTING  = False          # True  → insertPointCloud (ray carving, slower)
                                 # False → updateNodes       (just mark occupied, fast)
MAX_FRAMES      = 00             # 0 = all frames, >0 = cap at this many
FRAME_SKIP      = 3              # process every Nth frame (1 = no skip)

# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_recording_dir(path: str) -> str:
    """Resolve to the most recent flight_* folder if *path* is empty."""
    if path:
        return path
    base = os.path.join(os.path.dirname(__file__), "flight_recordings")
    dirs = sorted(glob.glob(os.path.join(base, "flight_*")))
    if not dirs:
        raise FileNotFoundError(f"No flight_* directories under {base}")
    return dirs[-1]


def filter_valid_points(points: np.ndarray) -> np.ndarray:
    """Remove zero / invalid LiDAR returns."""
    return points[np.any(points != 0, axis=1)]


def transform_points(points: np.ndarray, position: np.ndarray,
                     orientation: np.ndarray) -> np.ndarray:
    """Rotate + translate *points* by a pose given as position + [w,x,y,z] quat."""
    rot = Rotation.from_quat([orientation[1], orientation[2],
                              orientation[3], orientation[0]])
    R = rot.as_matrix()
    return (R @ points.T).T + position


# ── Main ──────────────────────────────────────────────────────────────────────

def run_replay(recording_dir: str = ""):
    recording_dir = resolve_recording_dir(recording_dir)
    all_frames = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    if not all_frames:
        print(f"No frame_*.npz files found in {recording_dir}")
        return

    # Apply frame skip then cap
    frames = all_frames[::FRAME_SKIP]
    if MAX_FRAMES > 0:
        frames = frames[:MAX_FRAMES]

    print(f"Replaying {len(frames)} frames from {recording_dir}")
    print(f"  (total available: {len(all_frames)}, skip={FRAME_SKIP}, max={MAX_FRAMES})")
    print(f"OctoMap resolution: {OCTO_RESOLUTION}m  "
          f"raycasting={'ON' if USE_RAYCASTING else 'OFF'}  "
          f"vis_every={VIS_EVERY}  "
          f"skip={FRAME_SKIP}  max={MAX_FRAMES or 'all'}")

    tree = pyoctomap.OcTree(OCTO_RESOLUTION)
    viewer = Viewer3D()

    # Incremental voxel tracking — avoids slow tree.extractPointCloud()
    inv_res = 1.0 / OCTO_RESOLUTION
    half_res = OCTO_RESOLUTION * 0.5
    seen_keys = set()                                  # set of (ix, iy, iz)
    VIS_BUF_SIZE = 2_000_000
    vis_buf = np.zeros((VIS_BUF_SIZE, 3), dtype=np.float32)
    vis_len = 0

    def _quantize_and_append(pts: np.ndarray):
        """Quantize pts to voxel centres, deduplicate, append new ones."""
        nonlocal vis_buf, vis_len
        # Vectorised quantization to integer voxel keys
        ijk = np.floor(pts * inv_res).astype(np.int32)
        # Deduplicate within this batch
        unique_ijk = set(map(tuple, ijk))
        new_keys = unique_ijk - seen_keys
        if not new_keys:
            return 0
        seen_keys.update(new_keys)
        # Convert keys to voxel centres
        new_arr = np.array(list(new_keys), dtype=np.float32)
        centres = new_arr * OCTO_RESOLUTION + half_res
        n = len(centres)
        # Grow buffer if needed
        if vis_len + n > len(vis_buf):
            new_size = max(len(vis_buf) * 2, vis_len + n)
            new_buf = np.zeros((new_size, 3), dtype=np.float32)
            new_buf[:vis_len] = vis_buf[:vis_len]
            vis_buf = new_buf
        vis_buf[vis_len:vis_len + n] = centres
        vis_len += n
        return n

    def _get_vis_pts():
        return vis_buf[:vis_len]

    t_insert_total  = 0.0
    t_extract_total = 0.0
    t_vis_total     = 0.0
    raw_pts_total   = 0

    for i, path in enumerate(frames):
        data = np.load(path)
        points = filter_valid_points(data["points"])
        if len(points) == 0:
            continue

        # ── sensor-local → body → world ──────────────────────────────────
        lidar_pos = (data["lidar_position"]
                     if "lidar_position" in data.files else np.zeros(3))
        lidar_ori = (data["lidar_orientation"]
                     if "lidar_orientation" in data.files
                     else np.array([1, 0, 0, 0], dtype=float))
        body_pts = transform_points(points, lidar_pos, lidar_ori)

        position = data["position"] if "position" in data.files else np.zeros(3)
        orientation = (data["orientation"] if "orientation" in data.files
                       else np.array([1, 0, 0, 0], dtype=float))
        world_pts = transform_points(body_pts, position, orientation)
        raw_pts_total += len(world_pts)

        # ── insert into octree ───────────────────────────────────────────
        world_f64 = world_pts.astype(np.float64)
        t0 = time.perf_counter()
        if USE_RAYCASTING:
            sensor_origin = position.astype(np.float64)
            tree.insertPointCloud(world_f64, sensor_origin, max_range=MAX_RANGE)
        else:
            tree.updateNodes(world_f64, True)
        t_insert = time.perf_counter() - t0
        t_insert_total += t_insert

        # ── incremental voxel tracking ────────────────────────────────────
        t0 = time.perf_counter()
        new_voxels = _quantize_and_append(world_f64)
        t_extract = time.perf_counter() - t0
        t_extract_total += t_extract

        # ── visualise ────────────────────────────────────────────────────
        if (i + 1) % VIS_EVERY == 0 or i == len(frames) - 1:
            t0 = time.perf_counter()
            vis_pts = _get_vis_pts()
            if i == 0:
                viewer.start(initial_points=vis_pts)
            else:
                viewer.update(vis_pts)
            t_vis = time.perf_counter() - t0
            t_vis_total += t_vis

            print(f"  frame {i+1:3d}/{len(frames)}  "
                  f"raw={len(world_pts):6,}  new_vox={new_voxels:6,}  "
                  f"total_vox={vis_len:8,}  "
                  f"mem={tree.memoryUsage()/1e6:.1f}MB  "
                  f"insert={t_insert*1000:.1f}ms  "
                  f"vox={t_extract*1000:.1f}ms  "
                  f"vis={t_vis*1000:.1f}ms",
                  flush=True)
        else:
            print(f"  frame {i+1:3d}/{len(frames)}  "
                  f"raw={len(world_pts):6,}  new_vox={new_voxels:6,}  "
                  f"insert={t_insert*1000:.1f}ms  vox={t_extract*1000:.1f}ms",
                  flush=True)

    # ── Final summary ────────────────────────────────────────────────────
    ratio = vis_len / raw_pts_total * 100 if raw_pts_total else 0

    print(f"\n{'='*60}")
    print(f"OctoMap Summary")
    print(f"  Resolution:       {OCTO_RESOLUTION} m")
    print(f"  Raw points total: {raw_pts_total:,}")
    print(f"  Occupied voxels:  {vis_len:,}")
    print(f"  Compression:      {ratio:.1f}% of raw ({raw_pts_total/max(vis_len,1):.1f}x)")
    print(f"  Tree nodes:       {tree.size():,}")
    print(f"  Memory (tree):    {tree.memoryUsage()/1e6:.1f} MB")
    print(f"  Insert total:     {t_insert_total:.3f}s  ({t_insert_total/len(frames)*1000:.1f}ms/frame)")
    print(f"  Voxel track total:{t_extract_total:.3f}s  ({t_extract_total/len(frames)*1000:.1f}ms/frame)")
    print(f"  Vis total:        {t_vis_total:.3f}s")
    print(f"{'='*60}")

    print("\nClose the Open3D window or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


if __name__ == "__main__":
    _wall_start = time.perf_counter()
    rec = sys.argv[1] if len(sys.argv) > 1 else RECORDING_DIR
    run_replay(rec)
    _wall_elapsed = time.perf_counter() - _wall_start
    print(f"\nTotal wall-clock time: {_wall_elapsed:.3f}s ({_wall_elapsed/60:.1f}min)")
