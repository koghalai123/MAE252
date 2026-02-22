#!/usr/bin/env python3
"""ICP + OctoMap mapping — combines scan-to-map ICP alignment with OctoMap storage.

The pipeline per frame:
  1. Transform new scan to world frame via recorded vehicle state (initial guess).
  2. Run point-to-plane ICP against the *octomap voxel centres* as the target map
     (far fewer points than raw accumulation → faster ICP).
  3. Apply ICP-refined transform to the full-resolution scan.
  4. Insert refined points into the octree (updateNodes — fast, no raycasting).
  5. Incrementally track new voxel centres for O(1)-amortised visualisation.

Replay mode only (reads flight_recordings/).
"""

import os, sys, glob, time
import numpy as np
import open3d as o3d
import pyoctomap
from scipy.spatial.transform import Rotation
from sensorFeed import Viewer3D

# ── Recording ─────────────────────────────────────────────────────────────────
RECORDING_DIR   = ""             # empty = latest in flight_recordings/
MAX_FRAMES      = 0              # 0 = all, >0 = cap
FRAME_SKIP      = 2              # process every Nth frame

# ── OctoMap ───────────────────────────────────────────────────────────────────
OCTO_RESOLUTION = 0.10           # leaf voxel size (m)

# ── ICP parameters ────────────────────────────────────────────────────────────
ICP_VOXEL_SIZE       = 0.25      # downsample for ICP matching
ICP_MAX_CORR_DIST    = 1.0       # max correspondence distance
ICP_MAX_ITERATION    = 30
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE    = 1e-6
ICP_MIN_VOXELS       = 3000      # skip ICP until octree has this many voxels
NORMAL_RADIUS        = 0.50
NORMAL_MAX_NN        = 20
ICP_LOCAL_RADIUS     = 30.0      # local window radius (m), 0 = whole map
USE_GPU              = True      # prefer CUDA tensor ICP

# ── Visualisation ─────────────────────────────────────────────────────────────

# ── GPU detection ─────────────────────────────────────────────────────────────
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
    """Apply 4×4 homogeneous transform to Nx3 points."""
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


# ══════════════════════════════════════════════════════════════════════════════
# ICP helpers (same as ICPMapping.py)
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
    """Point-to-plane ICP.  Returns (T, fitness, rmse, timing_dict)."""
    td = {}
    t0 = time.perf_counter()
    centroid = source_pts.mean(axis=0)
    local = _extract_local(target_pts, centroid, ICP_LOCAL_RADIUS)
    if len(local) < 100:
        local = target_pts
    td["local_window"] = time.perf_counter() - t0

    init64 = init_T.astype(np.float64)
    if _HAS_CUDA:
        T, fit, rms, sub = _icp_tensor(source_pts, local, init64)
    else:
        T, fit, rms, sub = _icp_legacy(source_pts, local, init64)
    td.update(sub)
    return T, fit, rms, td


def _icp_tensor(src_pts, tgt_pts, init_T):
    td = {}
    t0 = time.perf_counter()
    src = _ds_tensor(_to_tpcd(src_pts), ICP_VOXEL_SIZE)
    tgt = _ds_tensor(_to_tpcd(tgt_pts), ICP_VOXEL_SIZE)
    td["downsample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _normals_tensor(src); _normals_tensor(tgt)
    td["normals"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crit = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_MAX_ITERATION,
        relative_fitness=ICP_RELATIVE_FITNESS,
        relative_rmse=ICP_RELATIVE_RMSE)
    est = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
    res = o3d.t.pipelines.registration.icp(
        source=src, target=tgt,
        max_correspondence_distance=ICP_MAX_CORR_DIST,
        init_source_to_target=o3d.core.Tensor(init_T, device=_DEVICE),
        estimation_method=est, criteria=crit)
    td["icp_solve"] = time.perf_counter() - t0
    return res.transformation.cpu().numpy(), res.fitness, res.inlier_rmse, td


def _icp_legacy(src_pts, tgt_pts, init_T):
    td = {}
    t0 = time.perf_counter()
    src = _ds_legacy(_to_pcd(src_pts), ICP_VOXEL_SIZE)
    tgt = _ds_legacy(_to_pcd(tgt_pts), ICP_VOXEL_SIZE)
    td["downsample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _normals_legacy(src); _normals_legacy(tgt)
    td["normals"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=ICP_MAX_ITERATION,
        relative_fitness=ICP_RELATIVE_FITNESS,
        relative_rmse=ICP_RELATIVE_RMSE)
    res = o3d.pipelines.registration.registration_icp(
        source=src, target=tgt,
        max_correspondence_distance=ICP_MAX_CORR_DIST,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=crit)
    td["icp_solve"] = time.perf_counter() - t0
    return res.transformation, res.fitness, res.inlier_rmse, td


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
    print(f"  (available={len(all_frames)} skip={FRAME_SKIP} max={MAX_FRAMES or 'all'})")
    print(f"  octo_res={OCTO_RESOLUTION}m  icp_voxel={ICP_VOXEL_SIZE}  "
          f"icp_corr={ICP_MAX_CORR_DIST}  local_r={ICP_LOCAL_RADIUS}  "
          f"gpu={'yes' if _HAS_CUDA else 'no'}")

    tree = pyoctomap.OcTree(OCTO_RESOLUTION)
    viewer = Viewer3D()

    # ── Incremental voxel display buffer ──────────────────────────────────
    inv_res = 1.0 / OCTO_RESOLUTION
    half_res = OCTO_RESOLUTION * 0.5
    seen_keys: set = set()
    VIS_BUF = 2_000_000
    vis_buf = np.zeros((VIS_BUF, 3), dtype=np.float32)
    vis_len = 0

    def _voxelise_and_append(pts):
        """Hash new voxel centres into the display buffer. Returns # new."""
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

    # ── Timing accumulators ───────────────────────────────────────────────
    keys = ["load", "transform", "local_window", "downsample", "normals",
            "icp_solve", "apply_T", "octo_insert", "vox_track", "vis", "total"]
    timings = {k: [] for k in keys}
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

        # ── Transform to world (initial guess) ───────────────────────────
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

        # ── ICP against octomap voxel centres ─────────────────────────────
        if vis_len >= ICP_MIN_VOXELS:
            target_pts = _get_vis()  # octomap voxel centres as ICP target
            T, fitness, rmse, icp_td = run_icp(world_init, target_pts)
            timings["local_window"].append(icp_td.get("local_window", 0))
            timings["downsample"].append(icp_td.get("downsample", 0))
            timings["normals"].append(icp_td.get("normals", 0))
            timings["icp_solve"].append(icp_td.get("icp_solve", 0))

            t0 = time.perf_counter()
            world_pts = apply_T(world_init, T).astype(np.float32)
            timings["apply_T"].append(time.perf_counter() - t0)

            ct = T[:3, 3]
            ce = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            print(f"  ICP {i:03d}: fit={fitness:.4f} rmse={rmse:.4f} "
                  f"Δt={np.linalg.norm(ct):.4f}m "
                  f"Δr=({ce[0]:+.2f},{ce[1]:+.2f},{ce[2]:+.2f})°")
        else:
            world_pts = world_init.astype(np.float32)
            print(f"  frame {i:03d}: too few voxels for ICP ({vis_len}), state pose")
            for k in ["local_window", "downsample", "normals", "icp_solve", "apply_T"]:
                timings[k].append(0.0)

        # ── Insert into octree ────────────────────────────────────────────
        wf64 = world_pts.astype(np.float64)
        t0 = time.perf_counter()
        tree.updateNodes(wf64, True)
        timings["octo_insert"].append(time.perf_counter() - t0)

        # ── Incremental voxel tracking ────────────────────────────────────
        t0 = time.perf_counter()
        new_vox = _voxelise_and_append(wf64)
        timings["vox_track"].append(time.perf_counter() - t0)

        # ── Visualise ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        vp = _get_vis()
        if i == 0:
            viewer.start(initial_points=vp)
        else:
            viewer.update(vp)
        timings["vis"].append(time.perf_counter() - t0)

        timings["total"].append(time.perf_counter() - t_frame)

        # ── Per-frame log ─────────────────────────────────────────────────
        print(f"  {i+1:3d}/{len(frames)} | raw={len(pts):6,} new_vox={new_vox:5,} "
              f"voxels={vis_len:8,} mem={tree.memoryUsage()/1e6:.1f}MB | "
              f"load={timings['load'][-1]*1e3:.0f} xform={timings['transform'][-1]*1e3:.0f} "
              f"icp={sum(timings[k][-1] for k in ['local_window','downsample','normals','icp_solve'])*1e3:.0f} "
              f"ins={timings['octo_insert'][-1]*1e3:.0f} "
              f"vox={timings['vox_track'][-1]*1e3:.0f} "
              f"vis={timings['vis'][-1]*1e3:.0f} "
              f"tot={timings['total'][-1]*1e3:.0f}ms\n", flush=True)

    # ── Final ─────────────────────────────────────────────────────────────
    fit_plane(_get_vis(), label="FINAL ")
    ratio = vis_len / raw_total * 100 if raw_total else 0

    print(f"\n{'='*65}")
    print(f"ICP + OctoMap Summary")
    print(f"  Octo resolution:  {OCTO_RESOLUTION} m")
    print(f"  Raw pts total:    {raw_total:,}")
    print(f"  Occupied voxels:  {vis_len:,}  ({ratio:.1f}% → {raw_total/max(vis_len,1):.1f}x)")
    print(f"  Tree nodes:       {tree.size():,}  mem={tree.memoryUsage()/1e6:.1f}MB")
    gt = sum(timings["total"])
    print(f"\n  {'step':<14} {'total':>7} {'mean':>7} {'max':>7} {'%':>5}")
    print(f"  {'-'*44}")
    for k in keys:
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
