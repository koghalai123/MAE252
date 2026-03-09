#!/usr/bin/env python3
"""Compare LiDAR scan registration methods for 3D mapping.

Runs multiple registration algorithms on the same flight recording and
produces side-by-side comparison plots + a metrics summary table.

Methods compared:
  1. State-only       – raw vehicle pose, zero correction (baseline)
  2. Point-to-plane ICP – standard ICP (same as ICPMapping / ICPAndOctomap)
  3. GICP             – Generalized ICP (plane-to-plane covariances)
  4. NDT              – Normal Distributions Transform (custom implementation)
  5. FPFH + RANSAC    – Feature-based global registration + ICP refinement
  6. ScanContext + ICP – ICP odometry + scan-context place-recognition loop closure

All methods use identical voxel-grid storage so the only variable is the
registration / loop-closure strategy.

Usage:
    python RegistrationComparison.py                       # latest flight
    python RegistrationComparison.py /path/to/flight_dir   # explicit dir
"""

import os, sys, glob, time
import multiprocessing
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.optimize import minimize

try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt

# ── Recording ─────────────────────────────────────────────────────────────────
RECORDING_DIR = ""          # empty → latest in flight_recordings/
MAX_FRAMES    = 0           # 0 = all
FRAME_SKIP    = 2           # process every Nth frame

# ── Voxel display resolution ─────────────────────────────────────────────────
VOXEL_RES = 0.10            # metres — same as OCTO_RESOLUTION in other scripts

# ── Common ICP / registration parameters ──────────────────────────────────────
ICP_VOXEL        = 0.25     # voxel downsample for ICP matching
ICP_CORR         = 1.0      # max correspondence distance
ICP_ITER         = 30
ICP_FIT_TOL      = 1e-6
ICP_RMSE_TOL     = 1e-6
MIN_VOXELS       = 3000     # skip registration until map has this many voxels
NORM_R           = 0.50     # normal-estimation search radius
NORM_NN          = 20       # normal-estimation max neighbours
LOCAL_R          = 30.0     # local window radius (m) — 0 = whole map

# ── NDT-specific ──────────────────────────────────────────────────────────────
NDT_VOXEL        = 1.0      # NDT grid cell size (m)
NDT_MIN_PTS      = 5        # min points per cell to form a distribution
NDT_MAX_SRC      = 8000     # randomly subsample source for speed
NDT_MAX_ITER     = 50       # scipy optimizer iterations

# ── FPFH-specific ─────────────────────────────────────────────────────────────
FPFH_VOXEL       = 0.25      # voxel size for FPFH feature computation
FPFH_MAX_NN      = 100
RANSAC_N         = 4        # RANSAC sample size
RANSAC_ITER      = 100_000
RANSAC_CONF      = 0.999

# ── Scan Context (place recognition) ─────────────────────────────────────────
SC_RINGS         = 20
SC_SECTORS       = 60
SC_MAX_RANGE     = 80.0     # metres
SC_EVERY         = 10       # check for loop closure every N frames
SC_MIN_SEP       = 15       # minimum frame index separation
SC_DIST_THRESH   = 0.3      # scan-context cosine distance threshold
SC_ICP_CORR      = 2.0      # ICP corr dist for loop-closure refinement

# ── GPU ───────────────────────────────────────────────────────────────────────
USE_GPU = True
_CUDA = False
try:
    if USE_GPU and o3d.core.cuda.is_available():
        _CUDA = True
        _DEV = o3d.core.Device("CUDA:0")
except Exception:
    pass
if not _CUDA:
    _DEV = o3d.core.Device("CPU:0")


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def resolve_recording_dir(path: str) -> str:
    if path and os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "frame_00000.npz")):
            return path
        # Look for flight_* or exploration_* subdirectories
        subdirs = sorted(
            glob.glob(os.path.join(path, "flight_*"))
            + glob.glob(os.path.join(path, "exploration_*")))
        if subdirs:
            return subdirs[-1]
        return path
    base = os.path.join(os.path.dirname(__file__), "flight_recordings")
    dirs = sorted(
        glob.glob(os.path.join(base, "flight_*"))
        + glob.glob(os.path.join(base, "exploration_*")))
    if not dirs:
        raise FileNotFoundError(f"No flight_* or exploration_* dirs under {base}")
    return dirs[-1]


def filter_valid(pts):
    return pts[np.any(pts != 0, axis=1)]


def xform_pts(pts, position, orientation):
    """Rotate + translate by position + [w,x,y,z] quaternion."""
    R = Rotation.from_quat([orientation[1], orientation[2],
                            orientation[3], orientation[0]]).as_matrix()
    return (R @ pts.T).T + position


def apply_T(pts, T):
    """Apply 4×4 homogeneous transform to Nx3 points."""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


def evaluate_registration(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    T: np.ndarray,
    max_correspondence_distance: float = 0.25,
    downsample_voxel: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute registration quality metrics using **all** source points.

    Unlike Open3D’s ``inlier_rmse``, which only measures the RMSE of
    source points that fall within ``max_correspondence_distance`` of a
    target point (masking badly misaligned regions), this function returns
    metrics that reflect the *entire* source cloud.

    Parameters
    ----------
    src_pts : ndarray (N, 3)
        Source points in their **original** frame (before ``T``).
    tgt_pts : ndarray (M, 3)
        Target points (the existing map).
    T : ndarray (4, 4)
        Homogeneous transform produced by the registration algorithm.
    max_correspondence_distance : float
        Distance threshold for the inlier/fitness calculation.
    downsample_voxel : float
        If > 0, voxel-downsample both clouds before evaluation to save
        time on large clouds.  0 = use all points.

    Returns
    -------
    fitness : float
        Fraction of transformed source points that have a target
        neighbour within ``max_correspondence_distance``.
    inlier_rmse : float
        Classic inlier RMSE (only inlier distances).
    full_rmse : float
        RMSE over **all** transformed source points (including outliers).
        This is strictly ≥ inlier_rmse and spikes when alignment is bad.
    mean_dist : float
        Mean nearest-neighbour distance (all points).  More intuitive
        than RMSE for spotting drift.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    tgt = np.asarray(tgt_pts, dtype=np.float64)
    if src.shape[1] > 3:
        src = src[:, :3]
    if tgt.shape[1] > 3:
        tgt = tgt[:, :3]

    # Optional downsample for speed
    if downsample_voxel > 0:
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src)
        src = np.asarray(src_pcd.voxel_down_sample(downsample_voxel).points)
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt)
        tgt = np.asarray(tgt_pcd.voxel_down_sample(downsample_voxel).points)

    # Transform source
    transformed = apply_T(src, T)

    # Nearest-neighbour distances
    tree = cKDTree(tgt)
    dists, _ = tree.query(transformed, k=1)

    inlier_mask = dists <= max_correspondence_distance
    fitness = float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0
    inlier_rmse = (float(np.sqrt(np.mean(dists[inlier_mask] ** 2)))
                   if np.any(inlier_mask) else 0.0)
    full_rmse = float(np.sqrt(np.mean(dists ** 2))) if len(dists) else 0.0
    mean_dist = float(np.mean(dists)) if len(dists) else 0.0

    return fitness, inlier_rmse, full_rmse, mean_dist


def fit_plane(pts):
    """Fit a plane via SVD.  Returns (normal, slope_x, slope_y, residual_std)."""
    if len(pts) < 10:
        return None
    c = pts.mean(axis=0)
    centered = pts - c
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    n = Vt[-1]
    res_std = float(np.std(centered @ n))
    sx = -n[0] / n[2] if n[2] != 0 else float("nan")
    sy = -n[1] / n[2] if n[2] != 0 else float("nan")
    return n, sx, sy, res_std


def _to_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def _to_tpcd(pts, dev=None):
    if dev is None:
        dev = _DEV
    tpcd = o3d.t.geometry.PointCloud(dev)
    tpcd.point.positions = o3d.core.Tensor(pts.astype(np.float64), device=dev)
    return tpcd


def _local(pts, centroid, radius):
    if radius <= 0:
        return pts
    d = pts - centroid
    return pts[np.einsum('ij,ij->i', d, d) <= radius * radius]


# ══════════════════════════════════════════════════════════════════════════════
# Load all frames once (shared across methods)
# ══════════════════════════════════════════════════════════════════════════════

def load_frames(recording_dir):
    paths = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    paths = paths[::FRAME_SKIP]
    if MAX_FRAMES > 0:
        paths = paths[:MAX_FRAMES]
    out = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        pts = filter_valid(d["points"])
        if len(pts) == 0:
            continue
        lp = d["lidar_position"]    if "lidar_position"    in d.files and np.asarray(d["lidar_position"]).ndim > 0    else np.zeros(3)
        lo = d["lidar_orientation"] if "lidar_orientation" in d.files and np.asarray(d["lidar_orientation"]).ndim > 0 else np.array([1,0,0,0], dtype=float)
        body = xform_pts(pts, lp, lo)
        pos = d["position"]    if "position"    in d.files and np.asarray(d["position"]).ndim > 0    else np.zeros(3)
        ori = d["orientation"] if "orientation" in d.files and np.asarray(d["orientation"]).ndim > 0 else np.array([1,0,0,0], dtype=float)
        # Skip frames where position/orientation are missing (saved as None)
        if pos is None or ori is None or np.asarray(pos).ndim == 0 or np.asarray(ori).ndim == 0:
            continue
        world = xform_pts(body, pos, ori)
        out.append(dict(world_pts=world.astype(np.float32),
                        position=pos.copy(), orientation=ori.copy(),
                        num_raw=len(pts)))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. State-only (baseline — no registration)
# ══════════════════════════════════════════════════════════════════════════════

def register_state_only(src, tgt, init_T=np.eye(4)):
    return np.eye(4), 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Point-to-plane ICP  (multi-scale coarse→fine)
#
#   A single tight correspondence distance cannot recover rotational error:
#   1° at 30 m ≈ 0.5 m displacement.  Three passes (wide → medium → tight)
#   let ICP progressively lock on.
# ══════════════════════════════════════════════════════════════════════════════

ICP_MULTI_CORR = [0.5, 0.25, 0.1]   # coarse → fine correspondence distances
ICP_MULTI_ITER = [35, 15,  7 ]   # iterations per pass


def register_icp_p2plane(src, tgt, init_T=np.eye(4)):
    t0 = time.perf_counter()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt

    if _CUDA:
        s = _to_tpcd(src).voxel_down_sample(ICP_VOXEL)
        t = _to_tpcd(local).voxel_down_sample(ICP_VOXEL)
        s.estimate_normals(radius=NORM_R, max_nn=NORM_NN)
        t.estimate_normals(radius=NORM_R, max_nn=NORM_NN)
        T_cur = o3d.core.Tensor(init_T.astype(np.float64), device=_DEV)
        for corr, iters in zip(ICP_MULTI_CORR, ICP_MULTI_ITER):
            r = o3d.t.pipelines.registration.icp(
                s, t, corr, T_cur,
                o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.t.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=ICP_FIT_TOL, relative_rmse=ICP_RMSE_TOL,
                    max_iteration=iters))
            T_cur = r.transformation
        return T_cur.cpu().numpy(), r.fitness, r.inlier_rmse, time.perf_counter() - t0
    else:
        sp = _to_pcd(src).voxel_down_sample(ICP_VOXEL)
        tp = _to_pcd(local).voxel_down_sample(ICP_VOXEL)
        sp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
        tp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
        T_cur = init_T.astype(np.float64)
        for corr, iters in zip(ICP_MULTI_CORR, ICP_MULTI_ITER):
            r = o3d.pipelines.registration.registration_icp(
                sp, tp, corr, T_cur,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=ICP_FIT_TOL, relative_rmse=ICP_RMSE_TOL,
                    max_iteration=iters))
            T_cur = np.asarray(r.transformation)
        return T_cur, r.fitness, r.inlier_rmse, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Generalized ICP  (plane-to-plane — uses local surface covariances)
#    Multi-scale coarse→fine, same as point-to-plane ICP above.
# ══════════════════════════════════════════════════════════════════════════════

def register_gicp(src, tgt, init_T=np.eye(4)):
    t0 = time.perf_counter()
    _t = time.perf_counter
    sub = {}  # detailed sub-timings (seconds)

    _ts = _t()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt
    sub['local_crop'] = _t() - _ts

    _ts = _t()
    sp = _to_pcd(src).voxel_down_sample(ICP_VOXEL)
    sub['src_downsample'] = _t() - _ts

    _ts = _t()
    tp = _to_pcd(local).voxel_down_sample(ICP_VOXEL)
    sub['tgt_downsample'] = _t() - _ts

    _ts = _t()
    sp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
    sub['src_normals'] = _t() - _ts

    _ts = _t()
    tp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
    sub['tgt_normals'] = _t() - _ts

    sub['src_pts'] = len(sp.points)
    sub['tgt_pts'] = len(tp.points)

    T_cur = init_T.astype(np.float64)
    for idx, (corr, iters) in enumerate(zip(ICP_MULTI_CORR, ICP_MULTI_ITER)):
        _ts = _t()
        r = o3d.pipelines.registration.registration_generalized_icp(
            sp, tp, corr, T_cur,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=ICP_FIT_TOL, relative_rmse=ICP_RMSE_TOL,
                max_iteration=iters))
        T_cur = np.asarray(r.transformation)
        sub[f'gicp_pass{idx}'] = _t() - _ts

    sub['total'] = _t() - t0
    return T_cur, r.fitness, r.inlier_rmse, sub


# ══════════════════════════════════════════════════════════════════════════════
# 4. NDT — Normal Distributions Transform
#
#   The target map is discretised into 3-D cells.  Each cell stores a Gaussian
#   (mean + covariance).  The registration maximises the summed probability of
#   the transformed source points under these distributions.
#
#   Because Open3D has no built-in NDT, the optimisation is done with
#   scipy.optimize backed by vectorised NumPy / KDTree lookups.
# ══════════════════════════════════════════════════════════════════════════════

class NDTGrid:
    """Build a Normal-Distribution-Transform representation of a point cloud."""

    def __init__(self, points, voxel_size=1.0, min_pts=5):
        self.voxel_size = voxel_size
        inv = 1.0 / voxel_size
        ijk = np.floor(points * inv).astype(np.int64)

        # Group points by voxel key
        keys, inverse, counts = np.unique(ijk, axis=0,
                                          return_inverse=True,
                                          return_counts=True)
        means, covs_inv = [], []
        for v in range(len(keys)):
            if counts[v] < min_pts:
                continue
            vpts = points[inverse == v]
            mu = vpts.mean(axis=0)
            cov = np.cov(vpts.T)
            if cov.ndim < 2 or cov.shape != (3, 3):
                continue
            cov += np.eye(3) * 1e-3          # regularise
            try:
                ci = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                continue
            means.append(mu)
            covs_inv.append(ci)

        if means:
            self.means    = np.array(means)       # (M, 3)
            self.covs_inv = np.array(covs_inv)    # (M, 3, 3)
            self._tree    = cKDTree(self.means)
        else:
            self.means    = np.zeros((0, 3))
            self.covs_inv = np.zeros((0, 3, 3))
            self._tree    = None

    def score(self, points):
        """Negative sum of Gaussian likelihoods (lower = better alignment)."""
        if self._tree is None or len(points) == 0:
            return 0.0
        _, idx = self._tree.query(points, k=1)
        diffs  = points - self.means[idx]                    # (N, 3)
        maha   = np.einsum('ij,ijk,ik->i', diffs,
                           self.covs_inv[idx], diffs)        # (N,)
        return -np.sum(np.exp(-0.5 * maha))


def _ndt_objective(params, src_pts, grid):
    R = Rotation.from_rotvec(params[3:6]).as_matrix()
    transformed = (R @ src_pts.T).T + params[:3]
    return grid.score(transformed)


def register_ndt(src, tgt, init_T=np.eye(4)):
    t0 = time.perf_counter()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt

    grid = NDTGrid(local.astype(np.float64), voxel_size=NDT_VOXEL,
                   min_pts=NDT_MIN_PTS)
    if grid._tree is None:
        return init_T.copy(), 0.0, 0.0, time.perf_counter() - t0

    # Sub-sample source for speed
    src64 = src.astype(np.float64)
    if len(src64) > NDT_MAX_SRC:
        idx = np.random.choice(len(src64), NDT_MAX_SRC, replace=False)
        src_ds = src64[idx]
    else:
        src_ds = src64

    # Initial parameters [tx, ty, tz, rx, ry, rz] from init_T
    rv0 = Rotation.from_matrix(init_T[:3, :3]).as_rotvec()
    x0  = np.concatenate([init_T[:3, 3], rv0])

    res = minimize(_ndt_objective, x0, args=(src_ds, grid),
                   method='Powell',
                   options={'maxiter': NDT_MAX_ITER, 'maxfev': 600})

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(res.x[3:6]).as_matrix()
    T[:3, 3]  = res.x[:3]

    # Sanity check — reject if correction is unreasonably large
    dt = np.linalg.norm(T[:3, 3])
    dr = np.linalg.norm(res.x[3:6])
    if dt > 5.0 or dr > 0.3:          # >5 m or ~17°
        T = init_T.copy()

    # Compute fitness / RMSE on down-sampled source for consistency
    ds_pts = np.asarray(_to_pcd(src).voxel_down_sample(ICP_VOXEL).points)
    transformed = apply_T(ds_pts, T)
    tree = cKDTree(local.astype(np.float64))
    dists, _ = tree.query(transformed, k=1)
    inlier = dists < ICP_CORR
    fitness = float(np.mean(inlier))
    rmse = float(np.sqrt(np.mean(dists[inlier] ** 2))) if np.any(inlier) else 0.0

    return T, fitness, rmse, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
# 5. FPFH + RANSAC — Feature-based global registration + ICP refinement
#
#   FPFH (Fast Point Feature Histogram) encodes local 3-D geometry into a
#   33-dimensional descriptor per point.  RANSAC then finds a rigid transform
#   that maximises feature correspondences.  A final ICP pass refines the
#   alignment.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_fpfh(pcd, voxel_size):
    ds = pcd.voxel_down_sample(voxel_size)
    ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        ds, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=FPFH_MAX_NN))
    return ds, fpfh


def register_fpfh_ransac(src, tgt, init_T=np.eye(4)):
    t0 = time.perf_counter()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt

    src_pcd = _to_pcd(src)
    tgt_pcd = _to_pcd(local)
    src_ds, src_fpfh = _compute_fpfh(src_pcd, FPFH_VOXEL)
    tgt_ds, tgt_fpfh = _compute_fpfh(tgt_pcd, FPFH_VOXEL)

    dist_thresh = FPFH_VOXEL * 1.5
    est  = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    chk  = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)]
    crit = o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_ITER, RANSAC_CONF)

    # API changed between Open3D versions — try with mutual_filter first
    try:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_ds, tgt_ds, src_fpfh, tgt_fpfh,
            True,               # mutual_filter  (O3D ≥ 0.15)
            dist_thresh, est, RANSAC_N, chk, crit)
    except TypeError:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_ds, tgt_ds, src_fpfh, tgt_fpfh,
            dist_thresh, est, RANSAC_N, chk, crit)

    # Refine with point-to-plane ICP
    src_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
    tgt_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
    icp = o3d.pipelines.registration.registration_icp(
        src_ds, tgt_ds, ICP_CORR, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=ICP_FIT_TOL, relative_rmse=ICP_RMSE_TOL,
            max_iteration=ICP_ITER))

    T = np.asarray(icp.transformation)

    # Sanity check
    dt = np.linalg.norm(T[:3, 3])
    dr = np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec())
    if dt > 10.0 or dr > 0.5:
        T = np.eye(4)

    return T, icp.fitness, icp.inlier_rmse, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Scan Context — place-recognition descriptor for loop closure
#
#   Each scan is encoded as a (rings × sectors) matrix of maximum heights.
#   Column-shifted cosine distance detects revisits.  When a match is found,
#   ICP refines the loop-closure transform.
#
#   Frame-to-frame odometry still uses point-to-plane ICP (same as method 2).
# ══════════════════════════════════════════════════════════════════════════════

class ScanContext:
    def __init__(self, num_rings=SC_RINGS, num_sectors=SC_SECTORS,
                 max_range=SC_MAX_RANGE):
        self.nr = num_rings
        self.ns = num_sectors
        self.mr = max_range

    def descriptor(self, points, center=None):
        """Create a scan-context descriptor centred at *center*."""
        pts = points - center if center is not None else points
        xy  = pts[:, :2]
        rng = np.linalg.norm(xy, axis=1)
        mask = rng < self.mr
        rng = rng[mask]
        ang = np.arctan2(xy[mask, 1], xy[mask, 0]) + np.pi     # [0, 2π]
        hgt = pts[mask, 2]

        ring_step = self.mr / self.nr
        sect_step = 2 * np.pi / self.ns

        ri = np.clip((rng / ring_step).astype(int), 0, self.nr - 1)
        si = np.clip((ang / sect_step).astype(int), 0, self.ns - 1)

        desc = np.full((self.nr, self.ns), -np.inf)
        np.maximum.at(desc, (ri, si), hgt)
        desc[desc == -np.inf] = 0.0
        return desc

    def distance(self, d1, d2):
        """Column-shifted cosine distance (lower = more similar)."""
        f1 = d1.flatten()
        n1 = np.linalg.norm(f1)
        if n1 == 0:
            return float('inf')
        # Stack all column-shifted versions of d2
        shifts = np.array([np.roll(d2, s, axis=1).flatten()
                           for s in range(self.ns)])     # (S, M)
        norms = np.linalg.norm(shifts, axis=1)           # (S,)
        valid = norms > 0
        if not np.any(valid):
            return float('inf')
        cos_sims = (shifts[valid] @ f1) / (norms[valid] * n1)
        return float(1.0 - np.max(cos_sims))


# ══════════════════════════════════════════════════════════════════════════════
# 7. small_gicp — Parallel C++ GICP (multi-threaded, much faster)
#
#   Uses the small_gicp library which provides optimised C++/pybind11
#   implementations of GICP with parallel tree construction and matching.
#   Multi-scale coarse→fine, same as the Open3D GICP above.
# ══════════════════════════════════════════════════════════════════════════════

try:
    import small_gicp as _sg
    _HAS_SMALL_GICP = True
except ImportError:
    _HAS_SMALL_GICP = False

_NUM_THREADS = min(multiprocessing.cpu_count(), 16)


def _sg_fitness_rmse(src_np, T, tgt_np, max_dist):
    """Compute Open3D-compatible fitness & inlier RMSE after alignment."""
    pts = src_np[:, :3] if src_np.shape[1] > 3 else src_np
    tgt = tgt_np[:, :3] if tgt_np.shape[1] > 3 else tgt_np
    h = np.hstack([pts, np.ones((len(pts), 1))])
    transformed = (T @ h.T).T[:, :3]
    tree = cKDTree(tgt)
    dists, _ = tree.query(transformed, k=1)
    inlier = dists <= max_dist
    fitness = float(np.mean(inlier)) if len(inlier) else 0.0
    rmse = float(np.sqrt(np.mean(dists[inlier] ** 2))) if np.any(inlier) else 0.0
    return fitness, rmse


def register_small_gicp(src, tgt, init_T=np.eye(4)):
    """Parallel GICP via small_gicp — multi-scale coarse→fine."""
    if not _HAS_SMALL_GICP:
        raise RuntimeError("small_gicp is not installed — pip install small_gicp")

    t0 = time.perf_counter()
    _t = time.perf_counter
    sub = {}

    # Local crop
    _ts = _t()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt
    sub['local_crop'] = _t() - _ts

    # Downsample + preprocess source (returns pointcloud + kdtree)
    _ts = _t()
    src_sg, _ = _sg.preprocess_points(src.astype(np.float64),
                                      downsampling_resolution=ICP_VOXEL,
                                      num_neighbors=NORM_NN,
                                      num_threads=_NUM_THREADS)
    sub['src_downsample'] = _t() - _ts

    # Downsample + preprocess target
    _ts = _t()
    tgt_sg, tgt_tree = _sg.preprocess_points(local.astype(np.float64),
                                             downsampling_resolution=ICP_VOXEL,
                                             num_neighbors=NORM_NN,
                                             num_threads=_NUM_THREADS)
    sub['tgt_downsample'] = _t() - _ts

    # No separate normal step — preprocess_points handles normals+covs
    sub['src_normals'] = 0.0
    sub['tgt_normals'] = 0.0

    sub['src_pts'] = src_sg.size()
    sub['tgt_pts'] = tgt_sg.size()

    # Multi-scale coarse → fine
    T_cur = init_T.astype(np.float64)
    for idx, (corr, iters) in enumerate(zip(ICP_MULTI_CORR, ICP_MULTI_ITER)):
        _ts = _t()
        result = _sg.align(tgt_sg, src_sg, tgt_tree,
                           init_T_target_source=T_cur,
                           registration_type='GICP',
                           max_correspondence_distance=corr,
                           num_threads=_NUM_THREADS,
                           max_iterations=iters)
        T_cur = result.T_target_source
        sub[f'gicp_pass{idx}'] = _t() - _ts

    # Compute fitness/RMSE consistent with Open3D
    src_ds_np = np.asarray(src_sg.points())
    tgt_ds_np = np.asarray(tgt_sg.points())
    fitness, rmse = _sg_fitness_rmse(src_ds_np, T_cur, tgt_ds_np,
                                     ICP_MULTI_CORR[-1])

    sub['total'] = _t() - t0
    return T_cur, fitness, rmse, sub


# ══════════════════════════════════════════════════════════════════════════════
# 8. small_gicp VGICP — Voxelised GICP (fastest for large target clouds)
#
#   Instead of building a KD-tree on the target, the target is inserted into
#   a Gaussian Voxel Map.  The source is matched against voxel distributions.
#   This avoids the expensive KD-tree query on large targets.
# ══════════════════════════════════════════════════════════════════════════════

def register_small_vgicp(src, tgt, init_T=np.eye(4)):
    """Voxelised GICP via small_gicp — fastest for large target maps."""
    if not _HAS_SMALL_GICP:
        raise RuntimeError("small_gicp is not installed — pip install small_gicp")

    t0 = time.perf_counter()
    _t = time.perf_counter
    sub = {}

    # Local crop
    _ts = _t()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt
    sub['local_crop'] = _t() - _ts

    # Downsample + preprocess source (returns pointcloud + kdtree)
    _ts = _t()
    src_sg, _ = _sg.preprocess_points(src.astype(np.float64),
                                      downsampling_resolution=ICP_VOXEL,
                                      num_neighbors=NORM_NN,
                                      num_threads=_NUM_THREADS)
    sub['src_downsample'] = _t() - _ts

    # Build Gaussian Voxel Map from target (replaces downsample + KD-tree)
    _ts = _t()
    tgt_sg, _ = _sg.preprocess_points(local.astype(np.float64),
                                      downsampling_resolution=ICP_VOXEL,
                                      num_neighbors=NORM_NN,
                                      num_threads=_NUM_THREADS)
    voxelmap = _sg.GaussianVoxelMap(ICP_VOXEL)
    voxelmap.insert(tgt_sg)
    sub['tgt_downsample'] = _t() - _ts

    sub['src_normals'] = 0.0
    sub['tgt_normals'] = 0.0

    sub['src_pts'] = src_sg.size()
    sub['tgt_pts'] = len(voxelmap)

    # Multi-scale coarse → fine
    T_cur = init_T.astype(np.float64)
    for idx, (corr, iters) in enumerate(zip(ICP_MULTI_CORR, ICP_MULTI_ITER)):
        _ts = _t()
        result = _sg.align(voxelmap, src_sg,
                           init_T_target_source=T_cur,
                           max_correspondence_distance=corr,
                           num_threads=_NUM_THREADS,
                           max_iterations=iters)
        T_cur = result.T_target_source
        sub[f'gicp_pass{idx}'] = _t() - _ts

    # Compute fitness/RMSE
    src_ds_np = np.asarray(src_sg.points())
    tgt_ds_np = np.asarray(tgt_sg.points())
    fitness, rmse = _sg_fitness_rmse(src_ds_np, T_cur, tgt_ds_np,
                                     ICP_MULTI_CORR[-1])

    sub['total'] = _t() - t0
    return T_cur, fitness, rmse, sub


# ══════════════════════════════════════════════════════════════════════════════
# 9. KISS-ICP — Adaptive-threshold ICP with voxel hash map
#
#   Designed for LiDAR odometry.  Uses an adaptive correspondence threshold
#   and a voxel hash map for the target.  Very fast but returns a 4x4
#   transform only (no fitness/RMSE — we compute them post-hoc).
# ══════════════════════════════════════════════════════════════════════════════

try:
    from kiss_icp.registration import Registration as _KISSRegistration
    from kiss_icp.mapping import VoxelHashMap as _KISSVoxelMap
    _HAS_KISS_ICP = True
except ImportError:
    _HAS_KISS_ICP = False


def register_kiss_icp(src, tgt, init_T=np.eye(4)):
    """KISS-ICP adaptive-threshold registration."""
    if not _HAS_KISS_ICP:
        raise RuntimeError("kiss_icp is not installed — pip install kiss_icp")

    t0 = time.perf_counter()
    _t = time.perf_counter
    sub = {}

    # Local crop
    _ts = _t()
    local = _local(tgt, src.mean(axis=0), LOCAL_R)
    if len(local) < 100:
        local = tgt
    sub['local_crop'] = _t() - _ts

    # Downsample source
    _ts = _t()
    sp = _to_pcd(src).voxel_down_sample(ICP_VOXEL)
    src_ds = np.asarray(sp.points)
    sub['src_downsample'] = _t() - _ts

    # Build KISS voxel hash map from target
    _ts = _t()
    voxel_size = ICP_VOXEL
    max_dist = ICP_MULTI_CORR[0]  # use widest correspondence distance
    voxel_map = _KISSVoxelMap(voxel_size=voxel_size,
                              max_distance=max_dist,
                              max_points_per_voxel=20)
    tp = _to_pcd(local).voxel_down_sample(ICP_VOXEL)
    tgt_ds = np.asarray(tp.points)
    voxel_map.update(tgt_ds, np.eye(4))
    sub['tgt_downsample'] = _t() - _ts

    sub['src_normals'] = 0.0
    sub['tgt_normals'] = 0.0

    sub['src_pts'] = len(src_ds)
    sub['tgt_pts'] = len(tgt_ds)

    # Registration
    _ts = _t()
    reg = _KISSRegistration(max_num_iterations=sum(ICP_MULTI_ITER),
                            convergence_criterion=ICP_FIT_TOL,
                            max_num_threads=_NUM_THREADS)
    # sigma is an adaptive threshold — use widest corr dist as conservative value
    T_result = reg.align_points_to_map(
        points=src_ds,
        voxel_map=voxel_map,
        initial_guess=init_T.astype(np.float64),
        max_correspondance_distance=max_dist,
        kernel=1.0)
    sub['gicp_pass0'] = _t() - _ts
    sub['gicp_pass1'] = 0.0
    sub['gicp_pass2'] = 0.0

    # Compute fitness/RMSE
    fitness, rmse = _sg_fitness_rmse(src_ds, T_result, tgt_ds,
                                     ICP_MULTI_CORR[-1])

    sub['total'] = _t() - t0
    return T_result, fitness, rmse, sub


# ══════════════════════════════════════════════════════════════════════════════
# Registration registry — importable by other scripts
# ══════════════════════════════════════════════════════════════════════════════

REGISTRATION_METHODS = {
    "state_only":    register_state_only,
    "icp":           register_icp_p2plane,
    "p2plane":       register_icp_p2plane,
    "gicp":          register_gicp,
    "ndt":           register_ndt,
    "fpfh":          register_fpfh_ransac,
    "fpfh_ransac":   register_fpfh_ransac,
}
if _HAS_SMALL_GICP:
    REGISTRATION_METHODS["small_gicp"] = register_small_gicp
    REGISTRATION_METHODS["vgicp"]      = register_small_vgicp
if _HAS_KISS_ICP:
    REGISTRATION_METHODS["kiss_icp"]   = register_kiss_icp


def get_register_fn(method: str):
    """Return a registration function by keyword.

    Valid keywords: state_only, icp, p2plane, gicp, ndt, fpfh, fpfh_ransac,
                    small_gicp, vgicp, kiss_icp

    Each function has signature:
        register(source_pts, target_pts, init_T=np.eye(4))
        -> (T_4x4, fitness, inlier_rmse, detail_dict_or_seconds)
    """
    key = method.lower().strip()
    if key not in REGISTRATION_METHODS:
        raise ValueError(
            f"Unknown registration method '{method}'. "
            f"Choose from: {', '.join(sorted(REGISTRATION_METHODS))}")
    return REGISTRATION_METHODS[key]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline: run one method on all frames
# ══════════════════════════════════════════════════════════════════════════════

def run_method(name, register_fn, frame_data, use_scan_context=False):
    """Run a registration method on every frame. Returns a results dict."""

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

    inv_res  = 1.0 / VOXEL_RES
    half_res = VOXEL_RES * 0.5
    seen: set = set()
    VIS_CAP   = 2_000_000
    vis_buf   = np.zeros((VIS_CAP, 3), dtype=np.float32)
    vis_len   = 0

    def _vox(pts):
        nonlocal vis_buf, vis_len
        ijk = np.floor(pts * inv_res).astype(np.int32)
        new = set(map(tuple, ijk)) - seen
        if not new:
            return 0
        seen.update(new)
        arr     = np.array(list(new), dtype=np.float32)
        centres = arr * VOXEL_RES + half_res
        n = len(centres)
        if vis_len + n > len(vis_buf):
            ns = max(len(vis_buf) * 2, vis_len + n)
            nb = np.zeros((ns, 3), dtype=np.float32)
            nb[:vis_len] = vis_buf[:vis_len]
            vis_buf = nb
        vis_buf[vis_len:vis_len + n] = centres
        vis_len += n
        return n

    def _map():
        return vis_buf[:vis_len]

    # Scan-context state
    sc        = ScanContext() if use_scan_context else None
    sc_descs  = []
    sc_scans  = []       # down-sampled scans for loop-closure ICP
    sc_pos    = []
    lc_count  = 0

    per_frame_t = []
    fit_list    = []
    rmse_list   = []
    raw_total   = 0

    for i, fd in enumerate(frame_data):
        tf = time.perf_counter()
        world_init = fd['world_pts']
        raw_total += fd['num_raw']

        # ── Registration ─────────────────────────────────────────────────
        if register_fn is None or vis_len < MIN_VOXELS:
            world_pts = world_init
            fit_list.append(0.0)
            rmse_list.append(0.0)
        else:
            T, fitness, rmse, _ = register_fn(
                world_init.astype(np.float64),
                _map().astype(np.float64))
            world_pts = apply_T(world_init, T).astype(np.float32)
            fit_list.append(float(fitness))
            rmse_list.append(float(rmse))

        # ── Scan-context loop closure ────────────────────────────────────
        if use_scan_context and sc is not None:
            desc = sc.descriptor(world_pts, center=fd['position'])
            sc_descs.append(desc)
            sc_pos.append(fd['position'].copy())

            # Store down-sampled scan for loop-closure ICP
            ds_pcd = _to_pcd(world_pts).voxel_down_sample(ICP_VOXEL)
            sc_scans.append(np.asarray(ds_pcd.points).astype(np.float32))

            if (len(sc_descs) > SC_MIN_SEP
                    and i % SC_EVERY == 0):
                best_d, best_j = float('inf'), -1
                for j in range(len(sc_descs) - SC_MIN_SEP):
                    d = sc.distance(desc, sc_descs[j])
                    if d < best_d:
                        best_d, best_j = d, j

                if best_d < SC_DIST_THRESH and best_j >= 0:
                    # Refine with ICP
                    sp = _to_pcd(sc_scans[-1])
                    tp = _to_pcd(sc_scans[best_j])
                    sp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
                    tp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
                    lc = o3d.pipelines.registration.registration_icp(
                        sp, tp, SC_ICP_CORR, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            relative_fitness=ICP_FIT_TOL, relative_rmse=ICP_RMSE_TOL,
                            max_iteration=ICP_ITER))
                    if lc.fitness > 0.20:
                        world_pts = apply_T(world_pts, lc.transformation).astype(np.float32)
                        lc_count += 1
                        print(f"    LC {i}<->{best_j}  sc_dist={best_d:.3f}  "
                              f"fit={lc.fitness:.3f}  rmse={lc.inlier_rmse:.3f}")

        # ── Insert into voxel map ────────────────────────────────────────
        _vox(world_pts.astype(np.float64))

        elapsed = time.perf_counter() - tf
        per_frame_t.append(elapsed)

        if (i + 1) % 20 == 0 or i == len(frame_data) - 1:
            extra = f"  lc={lc_count}" if use_scan_context else ""
            print(f"    {i+1:3d}/{len(frame_data)} | "
                  f"voxels={vis_len:,} | {elapsed*1e3:.0f}ms{extra}",
                  flush=True)

    # ── Collect results ──────────────────────────────────────────────────
    voxels = _map().copy()
    plane  = fit_plane(voxels)
    total  = sum(per_frame_t)

    print(f"  => {vis_len:,} voxels | plane_res={plane[3]:.4f}m | "
          f"time={total:.1f}s"
          + (f" | loop_closures={lc_count}" if use_scan_context else ""))

    return dict(
        voxel_centres  = voxels,
        plane_residual = plane[3] if plane else float('inf'),
        plane_normal   = plane[0] if plane else np.zeros(3),
        total_time     = total,
        per_frame_time = per_frame_t,
        fitness_list   = fit_list,
        rmse_list      = rmse_list,
        num_voxels     = vis_len,
        loop_closures  = lc_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3D interactive viewer (one Open3D window per method, each in its own process)
# ══════════════════════════════════════════════════════════════════════════════

def _viewer_process(title, points, z_min, z_max):
    """Open an interactive Open3D window showing *points*, coloured by height."""
    import open3d as _o3d
    import numpy as _np

    pcd = _o3d.geometry.PointCloud()
    pcd.points = _o3d.utility.Vector3dVector(points.astype(_np.float64))

    # Colour by Z (height) using a blue→green→red gradient
    z = points[:, 2].astype(_np.float64)
    span = z_max - z_min if z_max != z_min else 1.0
    t = _np.clip((z - z_min) / span, 0, 1)            # normalised [0,1]
    colors = _np.zeros((len(t), 3), dtype=_np.float64)
    colors[:, 0] = t            # R increases with height
    colors[:, 1] = 1.0 - _np.abs(t - 0.5) * 2  # G peaks in the middle
    colors[:, 2] = 1.0 - t      # B decreases with height
    pcd.colors = _o3d.utility.Vector3dVector(colors)

    vis = _o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = _np.array([0.1, 0.1, 0.1])
    vis.run()           # blocks until the user closes the window
    vis.destroy_window()


def launch_3d_viewer(title, points, z_range=None):
    """Spawn a non-blocking process that shows an interactive 3D view."""
    if len(points) == 0:
        return None
    # Sub-sample for responsiveness when point counts are very large
    if len(points) > 500_000:
        idx = np.random.choice(len(points), 500_000, replace=False)
        pts = points[idx].copy()
    else:
        pts = points.copy()
    if z_range is None:
        zlo, zhi = float(np.percentile(pts[:, 2], 2)), float(np.percentile(pts[:, 2], 98))
    else:
        zlo, zhi = z_range
    p = multiprocessing.Process(target=_viewer_process,
                                args=(title, pts, zlo, zhi), daemon=True)
    p.start()
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results, out_dir):
    names = list(results.keys())
    n     = len(names)

    # ── Figure 1: top-down map views ─────────────────────────────────────
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    axes = np.atleast_2d(axes)

    # Unified Z colour range
    z_all = np.concatenate([r['voxel_centres'][:, 2] for r in results.values()])
    zlo, zhi = np.percentile(z_all, [2, 98])

    for idx, name in enumerate(names):
        ax  = axes[idx // cols, idx % cols]
        pts = results[name]['voxel_centres']
        # Sub-sample if huge
        if len(pts) > 200_000:
            sel = np.random.choice(len(pts), 200_000, replace=False)
            pts = pts[sel]
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2],
                        cmap='viridis', s=0.1, vmin=zlo, vmax=zhi,
                        rasterized=True)
        pr = results[name]['plane_residual']
        ax.set_title(f"{name}\n(plane res={pr:.4f}m, "
                     f"voxels={results[name]['num_voxels']:,})",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=8)
        if idx % cols == 0:
            ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cax, label='Z (m)')
    fig.suptitle("Registration Method Comparison — Top-Down Map Views",
                 fontsize=13, fontweight='bold', y=1.01)
    maps_path = os.path.join(out_dir, "comparison_maps.png")
    fig.savefig(maps_path, dpi=150, bbox_inches='tight')
    print(f"\nMap plot saved: {maps_path}")

    # ── Figure 2: metric bar charts ──────────────────────────────────────
    metric_data = {
        'Plane Residual (m)': [results[n]['plane_residual'] for n in names],
        'Total Time (s)':     [results[n]['total_time']     for n in names],
        'Voxel Count':        [results[n]['num_voxels']     for n in names],
        'Mean Fitness':       [np.mean(results[n]['fitness_list'])
                               if results[n]['fitness_list'] else 0
                               for n in names],
    }
    nm = len(metric_data)
    fig2, axes2 = plt.subplots(1, nm, figsize=(5 * nm, 4.5))
    colours = plt.cm.Set2(np.linspace(0, 1, n))

    for m_idx, (mlabel, vals) in enumerate(metric_data.items()):
        ax = axes2[m_idx]
        bars = ax.bar(range(n), vals, color=colours)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7)
        ax.set_title(mlabel, fontsize=10)
        ax.tick_params(labelsize=8)
        for bar, v in zip(bars, vals):
            txt = f'{v:.4f}' if v < 10 else (f'{v:.1f}' if v < 1000 else f'{v:,.0f}')
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    txt, ha='center', va='bottom', fontsize=7)

    fig2.suptitle("Metric Comparison", fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    metrics_path = os.path.join(out_dir, "comparison_metrics.png")
    fig2.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"Metrics plot saved: {metrics_path}")

    # ── Figure 3: per-frame fitness time-series ──────────────────────────
    fig3, (ax_f, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for name in names:
        fl = results[name]['fitness_list']
        if any(f > 0 for f in fl):
            ax_f.plot(fl, label=name, linewidth=0.8)
    ax_f.set_ylabel('Fitness')
    ax_f.set_title('Per-Frame Fitness')
    ax_f.legend(fontsize=7, ncol=3)

    for name in names:
        rl = results[name]['rmse_list']
        if any(r > 0 for r in rl):
            ax_r.plot(rl, label=name, linewidth=0.8)
    ax_r.set_ylabel('Inlier RMSE')
    ax_r.set_xlabel('Frame')
    ax_r.set_title('Per-Frame RMSE')
    ax_r.legend(fontsize=7, ncol=3)

    fig3.tight_layout()
    ts_path = os.path.join(out_dir, "comparison_timeseries.png")
    fig3.savefig(ts_path, dpi=150, bbox_inches='tight')
    print(f"Time-series plot saved: {ts_path}")

    try:
        plt.show()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rec = resolve_recording_dir(
        sys.argv[1] if len(sys.argv) > 1 else RECORDING_DIR)

    print(f"Loading frames from {rec} ...")
    frames = load_frames(rec)
    print(f"  {len(frames)} frames loaded  (skip={FRAME_SKIP}  "
          f"max={MAX_FRAMES or 'all'}  gpu={'CUDA' if _CUDA else 'CPU'})")

    methods = [
        ("State-only",      None,                  False),
        ("P2Plane ICP",     register_icp_p2plane,  False),
        ("GICP",            register_gicp,          False),
        ("NDT",             register_ndt,           False),
        ("FPFH+RANSAC",     register_fpfh_ransac,   False),
        ("ScanContext+ICP", register_icp_p2plane,   True),
    ]

    # Compute a shared Z colour range from the state-only pass so all
    # 3D windows use the same colour mapping.
    print("\nRunning State-only first to establish Z colour range...")
    results = {}
    results["State-only"] = run_method("State-only", None, frames, use_scan_context=False)
    z_all = results["State-only"]['voxel_centres'][:, 2]
    z_range = (float(np.percentile(z_all, 2)), float(np.percentile(z_all, 98)))

    viewer_procs = []
    viewer_procs.append(launch_3d_viewer("State-only",
                                         results["State-only"]['voxel_centres'],
                                         z_range))

    for label, fn, sc in methods:
        if label == "State-only":
            continue      # already ran above
        results[label] = run_method(label, fn, frames, use_scan_context=sc)
        vp = launch_3d_viewer(label, results[label]['voxel_centres'], z_range)
        viewer_procs.append(vp)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("REGISTRATION COMPARISON RESULTS")
    print(f"{'='*90}")
    hdr = (f"{'Method':<20} {'Voxels':>10} {'Plane Res(m)':>13} "
           f"{'Time(s)':>9} {'Mean Fit':>10} {'Mean RMSE':>10} {'LCs':>5}")
    print(hdr)
    print(f"{'─'*90}")
    for name in results:
        r  = results[name]
        mf = np.mean(r['fitness_list']) if r['fitness_list'] else 0
        mr = np.mean([x for x in r['rmse_list'] if x > 0]) if any(
            x > 0 for x in r['rmse_list']) else 0
        print(f"{name:<20} {r['num_voxels']:>10,} {r['plane_residual']:>13.4f} "
              f"{r['total_time']:>9.1f} {mf:>10.4f} {mr:>10.4f} "
              f"{r['loop_closures']:>5}")
    print(f"{'='*90}")

    # ── Plots ────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    plot_comparison(results, out_dir)

    # ── Wait for all 3D viewer windows to be closed ──────────────────────
    active = [p for p in viewer_procs if p is not None and p.is_alive()]
    if active:
        print(f"\n{len(active)} interactive 3D windows are open.")
        print("Close them (or Ctrl+C) to finish.")
        try:
            for p in active:
                p.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nTotal wall-clock: {time.perf_counter() - t0:.1f}s")
