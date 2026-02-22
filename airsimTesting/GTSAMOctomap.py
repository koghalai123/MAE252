#!/usr/bin/env python3
"""GTSAM pose-graph SLAM + OctoMap storage + Viewer3D visualisation.

Pipeline per frame:
  1. Load scan, transform to world frame via recorded vehicle state.
  2. If not the first frame, run point-to-plane ICP (source=new scan,
     target=octomap voxel centres within a local window) to get relative
     transform between current and previous frame.
  3. Add ICP between-factor + optional GPS prior to a GTSAM iSAM2 graph.
  4. Every LC_SEARCH_EVERY frames, try loop closure: ICP-match current scan
     against stored older scans; if fitness + RMSE pass thresholds, add a
     loop-closure between-factor → iSAM2 corrects accumulated drift.
  5. Insert ICP-refined points into the octree (updateNodes) and incrementally
     track new voxel centres for fast O(new) visualisation.

Replay mode only — reads flight_recordings/ directories.

Usage:
    python GTSAMOctomap.py                      # latest recording
    python GTSAMOctomap.py /path/to/flight_dir  # explicit directory
"""

import os, sys, glob, time, copy
import numpy as np
import open3d as o3d
import pyoctomap
import gtsam
from gtsam import symbol
from scipy.spatial.transform import Rotation
from sensorFeed import Viewer3D

# ── Recording ─────────────────────────────────────────────────────────────────
RECORDING_DIR = ""          # empty = latest
MAX_FRAMES    = 0           # 0 = all
FRAME_SKIP    = 1           # every Nth frame

# ── OctoMap ───────────────────────────────────────────────────────────────────
OCTO_RESOLUTION = 0.10      # leaf voxel size (m)

# ── ICP ───────────────────────────────────────────────────────────────────────
ICP_VOXEL_SIZE       = 0.25
ICP_MAX_CORR_DIST    = 1.0
ICP_MAX_ITERATION    = 30
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE    = 1e-6
ICP_MIN_VOXELS       = 3000   # skip ICP until this many voxels in map
NORMAL_RADIUS        = 0.50
NORMAL_MAX_NN        = 20
ICP_LOCAL_RADIUS     = 30.0   # 0 = whole map
USE_GPU              = True

# ── GTSAM / Loop Closure ─────────────────────────────────────────────────────
GPS_SIGMA              = 3.0   # metres — GPS position noise
ICP_NOISE_SCALE        = 1.0   # multiplier on ICP-derived noise
LC_SEARCH_EVERY        = 10    # try loop closure every N frames
LC_MIN_INDEX_SEP       = 15    # ignore candidates within this many frames
LC_FITNESS_THRESH      = 0.20  # minimum ICP fitness to accept a loop closure
LC_MAX_SPATIAL_DIST    = 50.0  # metres — skip candidates farther than this

# ── Visualisation ─────────────────────────────────────────────────────────────
VIS_EVERY = 1

# ══════════════════════════════════════════════════════════════════════════════
# GPU detection
# ══════════════════════════════════════════════════════════════════════════════
_HAS_CUDA = False
try:
    if USE_GPU and o3d.core.cuda.is_available():
        _HAS_CUDA = True
        _DEVICE = o3d.core.Device("CUDA:0")
        print("ICP: CUDA GPU")
except Exception:
    pass
if not _HAS_CUDA:
    _DEVICE = o3d.core.Device("CPU:0")
    print("ICP: CPU")

# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def resolve_recording_dir(path: str) -> str:
    if path and os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "frame_00000.npz")):
            return path
        flights = sorted(glob.glob(os.path.join(path, "flight_*")))
        if flights:
            return flights[-1]
        return path
    base = os.path.join(os.path.dirname(__file__), "flight_recordings")
    dirs = sorted(glob.glob(os.path.join(base, "flight_*")))
    if not dirs:
        raise FileNotFoundError(f"No flight_* dirs under {base}")
    return dirs[-1]


def filter_valid_points(pts):
    return pts[np.any(pts != 0, axis=1)]


def transform_points(pts, position, orientation):
    """Rotate + translate by position + [w,x,y,z] quaternion."""
    R = Rotation.from_quat([orientation[1], orientation[2],
                            orientation[3], orientation[0]]).as_matrix()
    return (R @ pts.T).T + position


def apply_T(pts, T):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


def fit_plane(pts, label=""):
    if len(pts) < 10:
        return None
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[-1]
    sx = -n[0] / n[2] if n[2] != 0 else float("nan")
    sy = -n[1] / n[2] if n[2] != 0 else float("nan")
    print(f"  {label}plane slope  x:{sx:+.4f}  y:{sy:+.4f}  "
          f"n:[{n[0]:+.4f},{n[1]:+.4f},{n[2]:+.4f}]")
    return n, sx, sy


def pose3_from_pos_quat(pos, quat_wxyz):
    """Create gtsam.Pose3 from position (3,) and [w,x,y,z] quaternion."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    rot = gtsam.Rot3.Quaternion(w, x, y, z)
    t = gtsam.Point3(float(pos[0]), float(pos[1]), float(pos[2]))
    return gtsam.Pose3(rot, t)


def pose3_to_T(p: gtsam.Pose3) -> np.ndarray:
    """Pose3 → 4×4 numpy."""
    T = np.eye(4)
    T[:3, :3] = p.rotation().matrix()
    t = p.translation()
    T[:3, 3] = np.array(t).reshape(3)
    return T


def T_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(T)


# ══════════════════════════════════════════════════════════════════════════════
# ICP helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd

def _to_tpcd(pts, device=None):
    if device is None:
        device = _DEVICE
    tpcd = o3d.t.geometry.PointCloud(device)
    tpcd.point.positions = o3d.core.Tensor(pts.astype(np.float64), device=device)
    return tpcd

def _ds_tensor(tpcd, vs):
    return tpcd.voxel_down_sample(vs) if vs > 0 else tpcd

def _normals_tensor(tpcd):
    tpcd.estimate_normals(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    return tpcd

def _ds_legacy(pcd, vs):
    return pcd.voxel_down_sample(vs) if vs > 0 else pcd

def _normals_legacy(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))
    return pcd

def _extract_local(pts, centroid, radius):
    if radius <= 0:
        return pts
    d = pts - centroid
    return pts[np.einsum('ij,ij->i', d, d) <= radius * radius]


def run_icp(source_pts, target_pts, init_T=np.eye(4)):
    """Point-to-plane ICP → (T_4x4, fitness, rmse, timing_dict)."""
    td = {}
    t0 = time.perf_counter()
    centroid = source_pts.mean(axis=0)
    local = _extract_local(target_pts, centroid, ICP_LOCAL_RADIUS)
    if len(local) < 100:
        local = target_pts
    td["local_window"] = time.perf_counter() - t0

    i64 = init_T.astype(np.float64)
    if _HAS_CUDA:
        T, f, r, s = _icp_tensor(source_pts, local, i64)
    else:
        T, f, r, s = _icp_legacy(source_pts, local, i64)
    td.update(s)
    return T, f, r, td


def _icp_tensor(src, tgt, init_T):
    td = {}
    t0 = time.perf_counter()
    s = _ds_tensor(_to_tpcd(src), ICP_VOXEL_SIZE)
    t = _ds_tensor(_to_tpcd(tgt), ICP_VOXEL_SIZE)
    td["downsample"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _normals_tensor(s); _normals_tensor(t)
    td["normals"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_MAX_ITERATION,
        relative_fitness=ICP_RELATIVE_FITNESS, relative_rmse=ICP_RELATIVE_RMSE)
    e = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    r = o3d.t.pipelines.registration.icp(
        source=s, target=t, max_correspondence_distance=ICP_MAX_CORR_DIST,
        init_source_to_target=o3d.core.Tensor(init_T, device=_DEVICE),
        estimation_method=e, criteria=c)
    td["icp_solve"] = time.perf_counter() - t0
    return r.transformation.cpu().numpy(), r.fitness, r.inlier_rmse, td


def _icp_legacy(src, tgt, init_T):
    td = {}
    t0 = time.perf_counter()
    sp = _ds_legacy(_to_pcd(src), ICP_VOXEL_SIZE)
    tp = _ds_legacy(_to_pcd(tgt), ICP_VOXEL_SIZE)
    td["downsample"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _normals_legacy(sp); _normals_legacy(tp)
    td["normals"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_MAX_ITERATION,
        relative_fitness=ICP_RELATIVE_FITNESS, relative_rmse=ICP_RELATIVE_RMSE)
    r = o3d.pipelines.registration.registration_icp(
        source=sp, target=tp, max_correspondence_distance=ICP_MAX_CORR_DIST,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=c)
    td["icp_solve"] = time.perf_counter() - t0
    return r.transformation, r.fitness, r.inlier_rmse, td


def _icp_noise(rmse):
    """Build a conservative 6-DOF noise model from ICP RMSE."""
    rot_s = max(0.05, rmse * 0.5) * ICP_NOISE_SCALE
    tra_s = max(0.10, rmse * 1.5) * ICP_NOISE_SCALE
    sigma = float(max(rot_s, tra_s))
    return gtsam.noiseModel.Isotropic.Sigma(6, sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Replay
# ══════════════════════════════════════════════════════════════════════════════

def run_replay(recording_dir: str = ""):
    recording_dir = resolve_recording_dir(recording_dir)
    all_frames = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    if not all_frames:
        print(f"No frames in {recording_dir}"); return

    frames = all_frames[::FRAME_SKIP]
    if MAX_FRAMES > 0:
        frames = frames[:MAX_FRAMES]

    print(f"Replaying {len(frames)} frames from {recording_dir}")
    print(f"  available={len(all_frames)} skip={FRAME_SKIP} max={MAX_FRAMES or 'all'}")
    print(f"  octo_res={OCTO_RESOLUTION}  icp_voxel={ICP_VOXEL_SIZE}  "
          f"icp_corr={ICP_MAX_CORR_DIST}  local_r={ICP_LOCAL_RADIUS}")
    print(f"  gpu={'yes' if _HAS_CUDA else 'no'}  "
          f"lc_every={LC_SEARCH_EVERY}  lc_min_sep={LC_MIN_INDEX_SEP}  "
          f"lc_fit_thresh={LC_FITNESS_THRESH}")

    # ── GTSAM setup ───────────────────────────────────────────────────────
    isam = gtsam.ISAM2()
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

    # Per-scan storage for loop closure
    scan_pcds = []      # downsampled Open3D pcds per frame (for loop-closure ICP)
    scan_T_raw = []     # raw 4×4 world pose per frame (from state)
    pose_count = 0
    loop_closures = 0

    # ── OctoMap + viewer ──────────────────────────────────────────────────
    tree = pyoctomap.OcTree(OCTO_RESOLUTION)
    viewer = Viewer3D()

    inv_res = 1.0 / OCTO_RESOLUTION
    half_res = OCTO_RESOLUTION * 0.5
    seen_keys: set = set()
    VIS_BUF = 2_000_000
    vis_buf = np.zeros((VIS_BUF, 3), dtype=np.float32)
    vis_len = 0

    def _voxelise_and_append(pts):
        nonlocal vis_buf, vis_len
        ijk = np.floor(pts * inv_res).astype(np.int32)
        unique = set(map(tuple, ijk))
        new = unique - seen_keys
        if not new:
            return 0
        seen_keys.update(new)
        arr = np.array(list(new), dtype=np.float32)
        centres = arr * OCTO_RESOLUTION + half_res
        n = len(centres)
        if vis_len + n > len(vis_buf):
            ns = max(len(vis_buf) * 2, vis_len + n)
            nb = np.zeros((ns, 3), dtype=np.float32)
            nb[:vis_len] = vis_buf[:vis_len]
            vis_buf = nb
        vis_buf[vis_len:vis_len + n] = centres
        vis_len += n
        return n

    def _get_vis():
        return vis_buf[:vis_len]

    def _rebuild_octree_from_optimized():
        """After a loop closure, re-insert all scans with optimized poses into
        a fresh octree + voxel display buffer.  This corrects drift."""
        nonlocal tree, seen_keys, vis_buf, vis_len
        values = isam.calculateEstimate()
        tree = pyoctomap.OcTree(OCTO_RESOLUTION)
        seen_keys = set()
        vis_buf = np.zeros((VIS_BUF, 3), dtype=np.float32)
        vis_len = 0
        for j in range(pose_count):
            key_j = symbol('x', j)
            try:
                opt_pose = values.atPose3(key_j)
            except Exception:
                continue
            T_opt = pose3_to_T(opt_pose)
            # Re-transform raw scan (stored in body frame)
            raw_pts = np.asarray(scan_pcds[j].points)
            world_pts = apply_T(raw_pts, T_opt).astype(np.float64)
            tree.updateNodes(world_pts, True)
            _voxelise_and_append(world_pts)

    # ── Timing ────────────────────────────────────────────────────────────
    tkeys = ["load", "transform", "icp", "gtsam", "octo_insert",
             "vox_track", "vis", "loop_closure", "total"]
    timings = {k: [] for k in tkeys}
    raw_total = 0

    for i, path in enumerate(frames):
        t_frame = time.perf_counter()

        # ── Load ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        data = np.load(path)
        pts = filter_valid_points(data["points"])
        if len(pts) == 0:
            continue
        timings["load"].append(time.perf_counter() - t0)

        # ── Transform to world (initial guess from recorded state) ────────
        t0 = time.perf_counter()
        lp = data["lidar_position"] if "lidar_position" in data.files else np.zeros(3)
        lo = (data["lidar_orientation"] if "lidar_orientation" in data.files
              else np.array([1, 0, 0, 0], dtype=float))
        body = transform_points(pts, lp, lo)
        pos = data["position"] if "position" in data.files else np.zeros(3)
        ori = (data["orientation"] if "orientation" in data.files
               else np.array([1, 0, 0, 0], dtype=float))
        world_init = transform_points(body, pos, ori)
        timings["transform"].append(time.perf_counter() - t0)
        raw_total += len(world_init)

        # Raw world-pose 4×4 from the recorded state
        T_raw = np.eye(4)
        R_raw = Rotation.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_matrix()
        T_raw[:3, :3] = R_raw
        T_raw[:3, 3] = pos

        # ── ICP against octomap voxel centres ─────────────────────────────
        t0 = time.perf_counter()
        if vis_len >= ICP_MIN_VOXELS and pose_count > 0:
            target_pts = _get_vis()
            T_icp, fitness, rmse, _ = run_icp(world_init, target_pts)
            world_pts = apply_T(world_init, T_icp).astype(np.float32)

            ct = T_icp[:3, 3]
            ce = Rotation.from_matrix(T_icp[:3, :3]).as_euler("xyz", degrees=True)
            print(f"  ICP {i:03d}: fit={fitness:.4f} rmse={rmse:.4f} "
                  f"Δt={np.linalg.norm(ct):.4f}m "
                  f"Δr=({ce[0]:+.2f},{ce[1]:+.2f},{ce[2]:+.2f})°")
        else:
            world_pts = world_init.astype(np.float32)
            T_icp = np.eye(4)
            fitness = 0.0
            rmse = 0.0
            print(f"  frame {i:03d}: too few voxels ({vis_len}), state pose only")
        timings["icp"].append(time.perf_counter() - t0)

        # ── GTSAM factor graph ────────────────────────────────────────────
        t0 = time.perf_counter()
        key_curr = symbol('x', pose_count)

        # Build Pose3 for current frame (state-derived, ICP-corrected)
        pose3_curr = pose3_from_pos_quat(pos, ori)
        if np.any(T_icp[:3, :3] != np.eye(3)) or np.any(T_icp[:3, 3] != 0):
            # Compose ICP correction onto state pose
            corrected_T = T_icp @ T_raw
            pose3_curr = T_to_pose3(corrected_T)

        if pose_count == 0:
            # Anchor first pose with prior
            graph.add(gtsam.PriorFactorPose3(key_curr, pose3_curr, prior_noise))
            initial_estimates.insert(key_curr, pose3_curr)
        else:
            # Between factor: relative transform from previous to current
            key_prev = symbol('x', pose_count - 1)
            try:
                prev_est = isam.calculateEstimate().atPose3(key_prev)
            except Exception:
                prev_est = T_to_pose3(scan_T_raw[-1])
            relative_pose = prev_est.between(pose3_curr)
            noise = _icp_noise(rmse) if rmse > 0 else gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
            graph.add(gtsam.BetweenFactorPose3(key_prev, key_curr, relative_pose, noise))
            initial_estimates.insert(key_curr, pose3_curr)

        # GPS prior if available
        if "gps" in data.files:
            gps = data["gps"]
            gps_pt = gtsam.Point3(float(gps[0]), float(gps[1]), float(gps[2]))
            gps_noise = gtsam.noiseModel.Isotropic.Sigma(3, GPS_SIGMA)
            graph.add(gtsam.GPSFactor(key_curr, gps_pt, gps_noise))

        # Update iSAM2
        isam.update(graph, initial_estimates)
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()
        timings["gtsam"].append(time.perf_counter() - t0)

        # Store scan for loop closure (downsampled, in world frame)
        pcd_ds = _to_pcd(world_pts)
        if ICP_VOXEL_SIZE > 0:
            pcd_ds = pcd_ds.voxel_down_sample(ICP_VOXEL_SIZE)
        scan_pcds.append(pcd_ds)
        scan_T_raw.append(T_raw.copy())
        pose_count += 1

        # ── Loop closure ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        did_loop_close = False
        if pose_count > LC_MIN_INDEX_SEP and (pose_count - 1) % LC_SEARCH_EVERY == 0:
            curr_pos = pos
            curr_pcd = pcd_ds
            # Try to match against older scans
            for j in range(max(0, pose_count - LC_MIN_INDEX_SEP - 1)):
                if abs(pose_count - 1 - j) < LC_MIN_INDEX_SEP:
                    continue
                # Quick spatial distance check
                try:
                    values = isam.calculateEstimate()
                    pose_j = values.atPose3(symbol('x', j))
                    T_j = pose3_to_T(pose_j)
                    dist = np.linalg.norm(T_j[:3, 3] - curr_pos)
                    if dist > LC_MAX_SPATIAL_DIST:
                        continue
                except Exception:
                    continue

                # ICP between current and candidate
                cand = scan_pcds[j]
                _normals_legacy(curr_pcd)
                _normals_legacy(cand)
                lc_res = o3d.pipelines.registration.registration_icp(
                    curr_pcd, cand, ICP_MAX_CORR_DIST * 1.5, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))

                if (lc_res.fitness > LC_FITNESS_THRESH and
                        lc_res.inlier_rmse < ICP_MAX_CORR_DIST * 0.5):
                    rel_pose_lc = T_to_pose3(lc_res.transformation)
                    noise_lc = _icp_noise(lc_res.inlier_rmse)
                    lc_graph = gtsam.NonlinearFactorGraph()
                    lc_graph.add(gtsam.BetweenFactorPose3(
                        symbol('x', j), key_curr, rel_pose_lc, noise_lc))
                    isam.update(lc_graph, gtsam.Values())
                    loop_closures += 1
                    did_loop_close = True
                    print(f"  ✓ LOOP CLOSURE x{j}↔x{pose_count-1}  "
                          f"fit={lc_res.fitness:.3f} rmse={lc_res.inlier_rmse:.3f}")
                    break
        timings["loop_closure"].append(time.perf_counter() - t0)

        # If loop closure happened, rebuild the octree with optimised poses
        if did_loop_close:
            t0 = time.perf_counter()
            _rebuild_octree_from_optimized()
            print(f"  octree rebuilt with optimized poses ({vis_len:,} voxels, "
                  f"{time.perf_counter()-t0:.2f}s)")
        else:
            # ── Insert into octree (normal path) ──────────────────────────
            t0 = time.perf_counter()
            wf64 = world_pts.astype(np.float64)
            tree.updateNodes(wf64, True)
            timings["octo_insert"].append(time.perf_counter() - t0)

            # ── Incremental voxel tracking ────────────────────────────────
            t0 = time.perf_counter()
            new_vox = _voxelise_and_append(wf64)
            timings["vox_track"].append(time.perf_counter() - t0)

        # ── Visualise ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if (i + 1) % VIS_EVERY == 0 or i == len(frames) - 1:
            vp = _get_vis()
            if not viewer._proc:
                viewer.start(initial_points=vp)
            else:
                viewer.update(vp)
        timings["vis"].append(time.perf_counter() - t0)

        timings["total"].append(time.perf_counter() - t_frame)

        # ── Per-frame log ─────────────────────────────────────────────────
        print(f"  {i+1:3d}/{len(frames)} | raw={len(pts):6,} voxels={vis_len:8,} "
              f"mem={tree.memoryUsage()/1e6:.1f}MB nodes={tree.size():,} "
              f"lc={loop_closures} | "
              f"tot={timings['total'][-1]*1e3:.0f}ms\n", flush=True)

    # ── Final ─────────────────────────────────────────────────────────────
    fit_plane(_get_vis(), label="FINAL ")
    ratio = vis_len / raw_total * 100 if raw_total else 0

    print(f"\n{'='*65}")
    print(f"GTSAM + OctoMap Summary")
    print(f"  Octo resolution:  {OCTO_RESOLUTION} m")
    print(f"  Raw pts total:    {raw_total:,}")
    print(f"  Occupied voxels:  {vis_len:,}  ({ratio:.1f}% → {raw_total/max(vis_len,1):.1f}x)")
    print(f"  Tree nodes:       {tree.size():,}  mem={tree.memoryUsage()/1e6:.1f}MB")
    print(f"  Pose graph nodes: {pose_count}")
    print(f"  Loop closures:    {loop_closures}")
    gt = sum(timings["total"])
    print(f"\n  {'step':<14} {'total':>7} {'mean':>7} {'max':>7} {'%':>5}")
    print(f"  {'-'*44}")
    for k in tkeys:
        v = timings[k]
        if not v:
            continue
        s = sum(v)
        pct = 100 * s / gt if gt else 0
        print(f"  {k:<14} {s:>7.3f} {s/len(v):>7.4f} {max(v):>7.4f} {pct:>4.1f}%")
    print(f"  {'='*44}")

    print("\nClose the Open3D window or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_replay(sys.argv[1] if len(sys.argv) > 1 else RECORDING_DIR)
    print(f"\nWall-clock: {time.perf_counter()-t0:.1f}s")
