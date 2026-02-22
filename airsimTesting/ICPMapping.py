"""3D Mapping with ICP refinement — stitch LiDAR scans into a global point cloud.

Extends basicMapping3D.py by using Open3D's point-to-plane ICP to refine the
drone pose before each scan is merged.  The workflow is:

  1.  Transform the new scan to an *initial* world pose using the recorded
      vehicle state (same as basicMapping3D).
  2.  Down-sample the new scan and the current global map to voxel grids.
  3.  Estimate surface normals on both clouds.
  4.  Run point-to-plane ICP with the state-derived pose as the initial guess.
  5.  Apply the ICP-refined 4×4 transform to the *full-resolution* new scan
      and merge it into the global map.

This corrects the residual timing-mismatch errors that cause individual scans
to be slightly tilted relative to the rest of the map.

Modes
-----
REPLAY : load a flight_recordings/ directory recorded by saveFlightData.py
LIVE   : fly waypoints in Unreal / AirSim, accumulate scans in real time

Toggle the mode with USE_RECORDING / RECORDING_DIR below.
"""

import cosysairsim as airsim
import numpy as np
import time
import os
import glob
import open3d as o3d
from scipy.spatial.transform import Rotation
from sensorFeed import Viewer3D

# ── Configuration ────────────────────────────────────────────────────────────
USE_RECORDING = True
RECORDING_DIR = "/home/koghalai/MAE252/airsimTesting/flight_recordings/"

MAX_SCAN_HZ = 10  # live-mode scan rate limit

WAYPOINTS = [
    (0, 0, -10),
    (10, 0, -10),
    (10, 10, -10),
    (0, 10, -10),
    (0, 0, -10),
    (5, 5, -15),
    (0, 0, -10),
]
FLY_VELOCITY = 3  # m/s

# ── ICP parameters ───────────────────────────────────────────────────────────
ICP_VOXEL_SIZE      = 0.25   # down-sample resolution for ICP matching (metres)
ICP_MAX_CORR_DIST   = 1.0    # maximum correspondence distance
ICP_MAX_ITERATION   = 30     # ICP iteration cap
ICP_RELATIVE_FITNESS = 1e-6  # convergence threshold (fitness change)
ICP_RELATIVE_RMSE    = 1e-6  # convergence threshold (RMSE change)
ICP_MIN_MAP_POINTS  = 5000   # skip ICP until the map has this many points
NORMAL_RADIUS       = 0.50   # radius for normal estimation
NORMAL_MAX_NN       = 20     # max neighbours for normal estimation
MAP_VOXEL_SIZE      = 0.05   # voxel size for thinning the stored global map
                              # (0 = keep all points, costs more RAM)
MAP_THIN_TRIGGER    = 1_500_000  # thin when map exceeds this many points
MAP_BUFFER_SIZE     = 20_000_000 # pre-allocated point buffer size
ICP_LOCAL_RADIUS    = 30.0   # only use map points within this radius of scan
                              # centroid for ICP (metres).  0 = use whole map.
USE_GPU             = True   # use CUDA-accelerated tensor ICP when available


# ── Utility functions ────────────────────────────────────────────────────────

def filter_valid_points(points):
    """Remove zero / invalid LiDAR returns (all-zero rows)."""
    return points[np.any(points != 0, axis=1)]


def transform_points(points, position, orientation):
    """Rotate + translate *points* by a pose given as position + [w,x,y,z] quat."""
    rot = Rotation.from_quat([orientation[1], orientation[2],
                              orientation[3], orientation[0]])
    R = rot.as_matrix()
    return (R @ points.T).T + position


def apply_T(points, T):
    """Apply a 4×4 homogeneous transform to Nx3 points."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    return (T @ pts_h.T).T[:, :3]


def fit_plane(points, label=""):
    """Fit a plane via SVD and print its slope.  Returns (normal, slope_x, slope_y)."""
    if len(points) < 10:
        return None
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] != 0:
        slope_x = -normal[0] / normal[2]
        slope_y = -normal[1] / normal[2]
    else:
        slope_x = slope_y = float("nan")
    print(f"  {label}plane slope  x: {slope_x:+.4f}  y: {slope_y:+.4f}  "
          f"normal: [{normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f}]")
    return normal, slope_x, slope_y


def resolve_recording_dir(path):
    """If *path* is the parent recordings folder, pick the latest flight_* subfolder."""
    if os.path.isfile(os.path.join(path, "frame_00000.npz")):
        return path
    flights = sorted(glob.glob(os.path.join(path, "flight_*")))
    if flights:
        return flights[-1]
    return path


# ── Detect GPU at import time ────────────────────────────────────────────────
_HAS_CUDA = False
try:
    if USE_GPU and o3d.core.cuda.is_available():
        _HAS_CUDA = True
        _DEVICE = o3d.core.Device("CUDA:0")
        print("ICP: using CUDA GPU acceleration")
except Exception:
    pass
if not _HAS_CUDA:
    _DEVICE = o3d.core.Device("CPU:0")
    print("ICP: using CPU")


# ── ICP helpers ──────────────────────────────────────────────────────────────

def _to_pcd(points):
    """Numpy Nx3 → Open3D legacy PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def _to_tpcd(points, device=None):
    """Numpy Nx3 → Open3D *tensor* PointCloud on the given device."""
    if device is None:
        device = _DEVICE
    tpcd = o3d.t.geometry.PointCloud(device)
    tpcd.point.positions = o3d.core.Tensor(points.astype(np.float64), device=device)
    return tpcd


def _estimate_normals_tensor(tpcd):
    """Estimate normals on a tensor PointCloud (in-place)."""
    tpcd.estimate_normals(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    return tpcd


def _downsample_tensor(tpcd, voxel_size):
    """Voxel-downsample a tensor PointCloud."""
    if voxel_size > 0:
        return tpcd.voxel_down_sample(voxel_size)
    return tpcd


def _extract_local_window(map_pts, centroid, radius):
    """Return map points within *radius* of *centroid* (fast numpy mask)."""
    if radius <= 0:
        return map_pts
    diff = map_pts - centroid
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    return map_pts[dist_sq <= radius * radius]


def _downsample_legacy(pcd, voxel_size):
    """Voxel-downsample a legacy point cloud."""
    if voxel_size > 0:
        return pcd.voxel_down_sample(voxel_size)
    return pcd


def _estimate_normals_legacy(pcd):
    """Estimate surface normals on a legacy PointCloud in-place."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    return pcd


def run_icp(source_pts, target_pts, init_T=np.eye(4)):
    """Run point-to-plane ICP.  Returns (4×4 transform, fitness, rmse, timing_dict).

    Uses CUDA tensor ICP when available, otherwise falls back to the legacy
    CPU pipeline.  A local spatial window is extracted from the target map
    to avoid processing millions of irrelevant points.

    Parameters
    ----------
    source_pts : Nx3 float — the *new* scan (already roughly placed in world frame)
    target_pts : Mx3 float — the accumulated global map
    init_T     : 4×4        — starting transform (identity = scan already placed)
    """
    td = {}  # timing breakdown

    # ── Extract local window from target map ──────────────────────────────
    t0 = time.perf_counter()
    centroid = source_pts.mean(axis=0)
    local_target = _extract_local_window(target_pts, centroid, ICP_LOCAL_RADIUS)
    if len(local_target) < 100:
        local_target = target_pts  # fallback to full map
    td["local_window"] = time.perf_counter() - t0

    init_T_64 = init_T.astype(np.float64)

    if _HAS_CUDA:
        T, fitness, rmse, sub_td = _run_icp_tensor(source_pts, local_target, init_T_64)
    else:
        T, fitness, rmse, sub_td = _run_icp_legacy(source_pts, local_target, init_T_64)

    td.update(sub_td)
    return T, fitness, rmse, td


def _run_icp_tensor(source_pts, target_pts, init_T):
    """GPU-accelerated tensor ICP."""
    td = {}

    t0 = time.perf_counter()
    src = _downsample_tensor(_to_tpcd(source_pts), ICP_VOXEL_SIZE)
    tgt = _downsample_tensor(_to_tpcd(target_pts), ICP_VOXEL_SIZE)
    td["downsample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _estimate_normals_tensor(src)
    _estimate_normals_tensor(tgt)
    td["normals"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_MAX_ITERATION,
        relative_fitness=ICP_RELATIVE_FITNESS,
        relative_rmse=ICP_RELATIVE_RMSE,
    )

    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

    result = o3d.t.pipelines.registration.icp(
        source=src,
        target=tgt,
        max_correspondence_distance=ICP_MAX_CORR_DIST,
        init_source_to_target=o3d.core.Tensor(init_T, device=_DEVICE),
        estimation_method=estimation,
        criteria=criteria,
    )
    td["icp_solve"] = time.perf_counter() - t0

    T = result.transformation.cpu().numpy()
    return T, result.fitness, result.inlier_rmse, td


def _run_icp_legacy(source_pts, target_pts, init_T):
    """CPU legacy ICP fallback."""
    td = {}

    t0 = time.perf_counter()
    src_pcd = _downsample_legacy(_to_pcd(source_pts), ICP_VOXEL_SIZE)
    tgt_pcd = _downsample_legacy(_to_pcd(target_pts), ICP_VOXEL_SIZE)
    td["downsample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _estimate_normals_legacy(src_pcd)
    _estimate_normals_legacy(tgt_pcd)
    td["normals"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_MAX_ITERATION,
        relative_fitness=ICP_RELATIVE_FITNESS,
        relative_rmse=ICP_RELATIVE_RMSE,
    )

    result = o3d.pipelines.registration.registration_icp(
        source=src_pcd,
        target=tgt_pcd,
        max_correspondence_distance=ICP_MAX_CORR_DIST,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=criteria,
    )
    td["icp_solve"] = time.perf_counter() - t0

    return result.transformation, result.fitness, result.inlier_rmse, td


def voxel_thin(points, voxel_size):
    """Thin a point cloud via voxel down-sampling, returning Nx3 numpy array."""
    if voxel_size <= 0 or len(points) == 0:
        return points
    pcd = _to_pcd(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)


# ── Replay mode ──────────────────────────────────────────────────────────────

def run_replay(recording_dir):
    recording_dir = resolve_recording_dir(recording_dir)
    frames = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    if not frames:
        print(f"No frame_*.npz files found in {recording_dir}")
        return

    print(f"Replaying {len(frames)} frames from {recording_dir}")
    print(f"ICP params: voxel={ICP_VOXEL_SIZE}, max_corr={ICP_MAX_CORR_DIST}, "
          f"iter={ICP_MAX_ITERATION}, map_voxel={MAP_VOXEL_SIZE}")

    viewer = Viewer3D()

    # Pre-allocated buffer — avoids np.vstack copies (was 66% of runtime)
    map_buf = np.zeros((MAP_BUFFER_SIZE, 3), dtype=np.float32)
    map_len = 0  # number of valid points in map_buf

    def _get_map():
        """Return a view of the filled portion of the buffer."""
        return map_buf[:map_len]

    def _append(pts):
        """Append pts to buffer, growing if needed."""
        nonlocal map_buf, map_len
        n = len(pts)
        if map_len + n > len(map_buf):
            new_size = max(len(map_buf) * 2, map_len + n)
            new_buf = np.zeros((new_size, 3), dtype=np.float32)
            new_buf[:map_len] = map_buf[:map_len]
            map_buf = new_buf
        map_buf[map_len:map_len + n] = pts
        map_len += n

    def _replace(pts):
        """Replace buffer contents (after thinning)."""
        nonlocal map_buf, map_len
        n = len(pts)
        if n > len(map_buf):
            map_buf = np.zeros((n + 1_000_000, 3), dtype=np.float32)
        map_buf[:n] = pts
        map_len = n

    # ── Timing accumulators ───────────────────────────────────────────
    timings = {
        "load":        [],
        "transform":   [],
        "local_window":[],
        "downsample":  [],
        "normals":     [],
        "icp_solve":   [],
        "apply_T":     [],
        "fit_plane":   [],
        "merge":       [],
        "visualize":   [],
        "total":       [],
    }

    for i, path in enumerate(frames):
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        data = np.load(path)
        points = filter_valid_points(data["points"])
        if len(points) == 0:
            continue
        timings["load"].append(time.perf_counter() - t0)

        # ── Step 1: state-based transform (initial guess) ────────────────
        t0 = time.perf_counter()
        lidar_pos = data["lidar_position"] if "lidar_position" in data.files else np.zeros(3)
        lidar_ori = (data["lidar_orientation"] if "lidar_orientation" in data.files
                     else np.array([1, 0, 0, 0], dtype=float))
        body_pts = transform_points(points, lidar_pos, lidar_ori)

        position = data["position"] if "position" in data.files else np.zeros(3)
        orientation = (data["orientation"] if "orientation" in data.files
                       else np.array([1, 0, 0, 0], dtype=float))
        world_pts_initial = transform_points(body_pts, position, orientation)
        timings["transform"].append(time.perf_counter() - t0)

        # ── Step 2: ICP refinement against current map ───────────────────
        if map_len >= ICP_MIN_MAP_POINTS:
            T_refine, fitness, rmse, icp_breakdown = run_icp(world_pts_initial, _get_map())

            # Record ICP sub-step timings
            timings["local_window"].append(icp_breakdown.get("local_window", 0))
            timings["downsample"].append(icp_breakdown.get("downsample", 0))
            timings["normals"].append(icp_breakdown.get("normals", 0))
            timings["icp_solve"].append(icp_breakdown.get("icp_solve", 0))

            t0 = time.perf_counter()
            world_pts = apply_T(world_pts_initial, T_refine).astype(np.float32)
            timings["apply_T"].append(time.perf_counter() - t0)

            # Decompose correction for logging
            correction_t = T_refine[:3, 3]
            correction_R = Rotation.from_matrix(T_refine[:3, :3])
            correction_euler = correction_R.as_euler("xyz", degrees=True)
            print(f"  ICP frame {i:03d}: fitness={fitness:.4f}  rmse={rmse:.4f}  "
                  f"Δt={np.linalg.norm(correction_t):.4f}m  "
                  f"Δr=({correction_euler[0]:+.2f}, {correction_euler[1]:+.2f}, "
                  f"{correction_euler[2]:+.2f})°")
        else:
            world_pts = world_pts_initial.astype(np.float32)
            print(f"  frame {i:03d}: map too small for ICP ({map_len} pts), using state pose")
            for k in ["local_window", "downsample", "normals", "icp_solve", "apply_T"]:
                timings[k].append(0.0)

        # ── Step 3: per-scan plane diagnostic ────────────────────────────
        t0 = time.perf_counter()
        fit_plane(world_pts, label=f"frame {i:03d} ")
        timings["fit_plane"].append(time.perf_counter() - t0)

        # ── Step 4: merge into global map ────────────────────────────────
        t0 = time.perf_counter()
        _append(world_pts)

        # Periodically thin the map to keep RAM in check
        if MAP_VOXEL_SIZE > 0 and map_len > MAP_THIN_TRIGGER:
            pre = map_len
            _replace(voxel_thin(_get_map(), MAP_VOXEL_SIZE))
            print(f"  map thinned {pre:,} → {map_len:,}")
        timings["merge"].append(time.perf_counter() - t0)

        # ── Visualize ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if i == 0:
            viewer.start(initial_points=_get_map())
        else:
            viewer.update(_get_map())

        extra = ""
        if "gps" in data.files:
            gps = data["gps"]
            extra = f" | GPS ({gps[0]:.6f}, {gps[1]:.6f}, {gps[2]:.1f})"
        print(f"  frame {i+1}/{len(frames)} | map pts {map_len:,}{extra}",
              flush=True)
        timings["visualize"].append(time.perf_counter() - t0)
        timings["total"].append(time.perf_counter() - t_total_start)

        # Per-frame timing summary
        print(f"  ⏱  load={timings['load'][-1]:.3f}  xform={timings['transform'][-1]:.3f}  "
              f"window={timings['local_window'][-1]:.3f}  ds={timings['downsample'][-1]:.3f}  "
              f"normals={timings['normals'][-1]:.3f}  icp={timings['icp_solve'][-1]:.3f}  "
              f"apply={timings['apply_T'][-1]:.3f}  plane={timings['fit_plane'][-1]:.3f}  "
              f"merge={timings['merge'][-1]:.3f}  vis={timings['visualize'][-1]:.3f}  "
              f"TOTAL={timings['total'][-1]:.3f}s\n")

    # Final thin + plane fit
    if MAP_VOXEL_SIZE > 0:
        _replace(voxel_thin(_get_map(), MAP_VOXEL_SIZE))
    viewer.update(_get_map())
    print(f"\nFinal global map: {map_len:,} points")
    fit_plane(_get_map(), label="FINAL ")

    print(f"\nReplay complete — {map_len:,} points in map.")

    # ── Timing summary ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TIMING SUMMARY (seconds)")
    print(f"{'step':<15} {'total':>8} {'mean':>8} {'max':>8} {'%':>6}")
    print("-"*50)
    grand_total = sum(timings["total"])
    for key in ["load", "transform", "local_window", "downsample", "normals",
                "icp_solve", "apply_T", "fit_plane", "merge", "visualize", "total"]:
        vals = timings[key]
        if not vals:
            continue
        s = sum(vals)
        pct = 100.0 * s / grand_total if grand_total > 0 else 0
        print(f"{key:<15} {s:>8.3f} {s/len(vals):>8.4f} {max(vals):>8.4f} {pct:>5.1f}%")
    print("="*70)

    print("Close the Open3D window or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


# ── Live mode ────────────────────────────────────────────────────────────────

def run_live():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)

    lidar = client.getLidarData()
    init_pts = np.array(lidar.point_cloud, dtype=np.float32).reshape((-1, 3))

    viewer = Viewer3D()
    viewer.start(initial_points=init_pts)
    map_buf = np.zeros((MAP_BUFFER_SIZE, 3), dtype=np.float32)
    map_len = 0

    def _get_map():
        return map_buf[:map_len]

    def _append(pts):
        nonlocal map_buf, map_len
        n = len(pts)
        if map_len + n > len(map_buf):
            new_size = max(len(map_buf) * 2, map_len + n)
            new_buf = np.zeros((new_size, 3), dtype=np.float32)
            new_buf[:map_len] = map_buf[:map_len]
            map_buf = new_buf
        map_buf[map_len:map_len + n] = pts
        map_len += n

    def _replace(pts):
        nonlocal map_buf, map_len
        n = len(pts)
        if n > len(map_buf):
            map_buf = np.zeros((n + 1_000_000, 3), dtype=np.float32)
        map_buf[:n] = pts
        map_len = n

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)

    min_interval = 1.0 / MAX_SCAN_HZ
    last_scan = 0.0
    scan_count = 0

    try:
        for i, wp in enumerate(WAYPOINTS):
            print(f"\nWaypoint {i+1}/{len(WAYPOINTS)}: {wp}")
            future = client.moveToPositionAsync(wp[0], wp[1], wp[2],
                                                velocity=FLY_VELOCITY)

            while not future._set_flag:
                now = time.time()
                if now - last_scan < min_interval:
                    time.sleep(0.01)
                    continue
                last_scan = now

                lidar = client.getLidarData()
                points = np.array(lidar.point_cloud, dtype=np.float32).reshape((-1, 3))
                points = filter_valid_points(points)
                if len(points) == 0:
                    continue

                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                ori = state.kinematics_estimated.orientation
                position = np.array([pos.x_val, pos.y_val, pos.z_val])
                orientation = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])

                lpos = lidar.pose.position
                lori = lidar.pose.orientation
                lidar_position = np.array([lpos.x_val, lpos.y_val, lpos.z_val])
                lidar_orientation = np.array([lori.w_val, lori.x_val, lori.y_val, lori.z_val])

                # sensor local → body → world (initial guess)
                body_pts = transform_points(points, lidar_position, lidar_orientation)
                world_pts_initial = transform_points(body_pts, position, orientation)

                # ICP refinement
                if map_len >= ICP_MIN_MAP_POINTS:
                    T_refine, fitness, rmse, _ = run_icp(world_pts_initial, _get_map())
                    world_pts = apply_T(world_pts_initial, T_refine).astype(np.float32)
                else:
                    world_pts = world_pts_initial.astype(np.float32)

                _append(world_pts)

                # Periodic thinning
                if MAP_VOXEL_SIZE > 0 and map_len > MAP_THIN_TRIGGER:
                    _replace(voxel_thin(_get_map(), MAP_VOXEL_SIZE))

                viewer.update(_get_map())
                scan_count += 1
                print(f"\r  scans {scan_count} | map pts {map_len:,}",
                      end="", flush=True)

        # Final thin
        if MAP_VOXEL_SIZE > 0:
            _replace(voxel_thin(_get_map(), MAP_VOXEL_SIZE))
        viewer.update(_get_map())
        fit_plane(_get_map(), label="FINAL ")
        print(f"\n\nMapping complete — {map_len:,} points, {scan_count} scans.")
        print("Close the Open3D window or Ctrl+C to exit.")
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        viewer.stop()
        print("Done.")


if __name__ == "__main__":
    _wall_start = time.perf_counter()
    if USE_RECORDING:
        run_replay(RECORDING_DIR)
    else:
        run_live()
    _wall_elapsed = time.perf_counter() - _wall_start
    print(f"\nTotal wall-clock time: {_wall_elapsed:.3f}s ({_wall_elapsed/60:.1f}min)")
