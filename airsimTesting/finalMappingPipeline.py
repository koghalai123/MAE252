#!/usr/bin/env python3
"""SLAM pipeline — GTSAM pose-graph + deferred OctoMap + swappable registration.

Combines:
  - Frame-to-map registration (default: GICP, swappable via REGISTRATION)
  - GTSAM iSAM2 pose-graph optimisation with between-factors + GPS priors
  - Deferred OctoMap: fast voxel hash grid during live operation, OctoMap
    built once at the end for .bt file export (~100x faster than Bayesian)
  - Submap architecture: local voxel stores in anchor-keyframe coordinates;
    after pose correction only anchor-to-world transforms are updated

Replay mode only — reads flight_recordings/ directories.

Usage:
    python finalMapping.py                          # latest recording
    python finalMapping.py /path/to/flight_dir      # explicit directory
"""

import os, sys, glob, time
import numpy as np
import open3d as o3d
import pyoctomap
import gtsam
from gtsam import symbol
from scipy.spatial.transform import Rotation
from sensorFeed import Viewer3D
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from RegistrationComparison import (
    get_register_fn, resolve_recording_dir, filter_valid, xform_pts,
    apply_T, fit_plane,
    ICP_VOXEL, NORM_NN, LOCAL_R, MIN_VOXELS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Registration algorithm: "gicp" | "small_gicp" | "vgicp" | "kiss_icp" | "icp" | "fpfh_ransac" | "ndt" | "state_only"
REGISTRATION = "vgicp"

# Recording
RECORDING_DIR = ""           # empty → latest in flight_recordings/
MAX_FRAMES    = 0            # 0 = all
FRAME_SKIP    = 3            # process every Nth frame

# OctoMap / voxel grid
OCTO_RESOLUTION   = 0.15     # leaf voxel size (m)
OCTO_INSERT_VOXEL = 0.15     # downsample before submap insertion (0 = skip)

# Submaps
SUBMAP_FRAMES  = 10          # frames per submap

# Registration outlier rejection
REJECT_OUTLIERS     = True
REJECT_WARMUP       = 4      # frames before rejection kicks in
REJECT_RMSE_CI      = 0.80   # confidence level for RMSE jump detection
REJECT_ROT_CI       = 0.0    # confidence level for rotation jump detection

# Registration target cropping
REG_LOCAL_RADIUS    = 40.0   # crop target cloud to this radius around drone (m, 0 = no crop)

# GTSAM
GPS_SIGMA           = 1.0    # GPS position noise (m) — lower = trust GPS more
ICP_NOISE_SCALE     = 1.0    # multiplier on ICP-derived noise model


# ══════════════════════════════════════════════════════════════════════════════
# VoxelHashGrid — ultra-fast voxel storage
# ══════════════════════════════════════════════════════════════════════════════

class VoxelHashGrid:
    """O(1)-insert voxel grid backed by a Python set of integer keys.

    Points are quantised to a regular grid; the set gives instant dedup.
    No raycasting, no tree traversal — just hash operations.
    """

    __slots__ = ('resolution', 'inv_res', 'half_res', 'keys')

    def __init__(self, resolution: float):
        self.resolution = resolution
        self.inv_res    = 1.0 / resolution
        self.half_res   = resolution * 0.5
        self.keys: set  = set()

    def insert_points(self, pts: np.ndarray):
        """Add points to the grid.  O(n) in number of input points."""
        if len(pts) == 0:
            return
        ijk = np.floor(np.asarray(pts) * self.inv_res).astype(np.int64)
        self.keys.update(map(tuple, ijk))

    def get_centers(self) -> np.ndarray:
        """Return occupied voxel centres as Nx3 float64."""
        if not self.keys:
            return np.empty((0, 3), dtype=np.float64)
        arr = np.array(list(self.keys), dtype=np.float64)
        return arr * self.resolution + self.half_res

    def size(self) -> int:
        return len(self.keys)

    def memoryUsage(self) -> int:
        """Rough byte estimate."""
        return len(self.keys) * 120


# ══════════════════════════════════════════════════════════════════════════════
# Submap — local voxel storage in anchor-keyframe coordinates
# ══════════════════════════════════════════════════════════════════════════════

class Submap:
    """A VoxelHashGrid in a local (anchor keyframe) frame.

    When GTSAM corrects the anchor pose, world-frame points are obtained by
    re-transforming the local occupied voxels — no re-insertion needed.
    """

    def __init__(self, anchor_index: int, anchor_T: np.ndarray,
                 resolution: float = OCTO_RESOLUTION):
        self.anchor_index = anchor_index
        self.anchor_T     = anchor_T.copy()
        self.resolution   = resolution
        self.frame_indices: list[int] = []
        self.grid = VoxelHashGrid(resolution)

    @property
    def _T_inv(self):
        return np.linalg.inv(self.anchor_T)

    def _to_local(self, world_pts: np.ndarray) -> np.ndarray:
        return apply_T(world_pts, self._T_inv)

    def insert(self, world_pts: np.ndarray):
        """Insert a scan (world-frame) into the local voxel grid."""
        local_pts = self._to_local(world_pts).astype(np.float64)
        self.grid.insert_points(local_pts)

    def get_world_points(self, updated_T: np.ndarray | None = None
                         ) -> np.ndarray:
        """Return occupied voxel centres in the world frame."""
        occ = self.grid.get_centers()
        if len(occ) == 0:
            return np.empty((0, 3), dtype=np.float64)
        T = updated_T if updated_T is not None else self.anchor_T
        return apply_T(occ, T)

    def memoryUsage(self) -> int:
        return self.grid.memoryUsage()

    def node_count(self) -> int:
        return self.grid.size()


def _downsample_for_insert(pts: np.ndarray) -> np.ndarray:
    """Voxel-downsample points before submap insertion for speed."""
    if OCTO_INSERT_VOXEL <= 0:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(OCTO_INSERT_VOXEL)
    return np.asarray(pcd.points)


# ══════════════════════════════════════════════════════════════════════════════
# GTSAM helpers
# ══════════════════════════════════════════════════════════════════════════════

def pose3_from_pos_quat(pos, quat_wxyz):
    """Build a gtsam.Pose3 from a position vector and w-x-y-z quaternion."""
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]),
                   float(quat_wxyz[2]), float(quat_wxyz[3]))
    return gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z),
                       gtsam.Point3(float(pos[0]), float(pos[1]), float(pos[2])))


def pose3_to_T(p: gtsam.Pose3) -> np.ndarray:
    """Convert a gtsam.Pose3 to a 4x4 homogeneous transform matrix."""
    T = np.eye(4)
    T[:3, :3] = p.rotation().matrix()
    T[:3, 3] = np.array(p.translation()).reshape(3)
    return T


def T_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    """Convert a 4x4 homogeneous transform to a gtsam.Pose3."""
    return gtsam.Pose3(T)


def _icp_noise(rmse):
    """Build an isotropic noise model scaled from registration RMSE."""
    rot_s = max(0.05, rmse * 0.5) * ICP_NOISE_SCALE
    tra_s = max(0.10, rmse * 1.5) * ICP_NOISE_SCALE
    sigma = float(max(rot_s, tra_s))
    return gtsam.noiseModel.Isotropic.Sigma(6, sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Registration outlier detector
# ══════════════════════════════════════════════════════════════════════════════

class RegistrationOutlierDetector:
    """Detect bad registrations via confidence-interval spike detection.

    Maintains a running history of RMSE and rotation-correction magnitudes.
    A frame is flagged as an outlier when:
      1. Its RMSE exceeds the upper bound of a *rmse_ci* confidence interval
      2. Its rotation correction exceeds the upper bound of a *rot_ci* CI

    Both conditions must be true simultaneously.
    The detector is dormant for the first *warmup* accepted frames.
    """

    def __init__(self, warmup: int = REJECT_WARMUP,
                 rmse_ci: float = REJECT_RMSE_CI,
                 rot_ci: float = REJECT_ROT_CI):
        self.warmup  = max(3, warmup)
        self.rmse_ci = rmse_ci
        self.rot_ci  = rot_ci
        self._rmse_hist: list[float] = []
        self._rot_hist:  list[float] = []
        self.rejected_count = 0

    @staticmethod
    def _z(ci: float) -> float:
        """One-sided z-score for a given confidence level."""
        from scipy.stats import norm
        return float(norm.ppf((1.0 + ci) / 2.0))

    def check(self, rmse: float, dr_mag: float) -> tuple[bool, str]:
        """Return (is_outlier, reason_string).

        If the frame is *not* an outlier the caller should pass it to
        ``accept()`` so it enters the running statistics.
        """
        if rmse == 0.0:
            return False, ""

        n = len(self._rmse_hist)
        if n < self.warmup:
            return False, ""

        rmse_arr = np.array(self._rmse_hist)
        rot_arr  = np.array(self._rot_hist)

        rmse_mean, rmse_std = float(rmse_arr.mean()), float(rmse_arr.std(ddof=1))
        rot_mean,  rot_std  = float(rot_arr.mean()),  float(rot_arr.std(ddof=1))

        z_rmse = self._z(self.rmse_ci)
        z_rot  = self._z(self.rot_ci)

        rmse_thresh = rmse_mean + z_rmse * rmse_std
        rot_thresh  = rot_mean  + z_rot  * rot_std

        rmse_bad = rmse > rmse_thresh
        rot_bad  = dr_mag > rot_thresh

        if rmse_bad and rot_bad:
            reason = (f"RMSE {rmse:.4f} > {rmse_thresh:.4f} "
                      f"({self.rmse_ci*100:.0f}% CI, \u03bc={rmse_mean:.4f} \u03c3={rmse_std:.4f}) "
                      f"AND \u0394r {dr_mag:.2f}\u00b0 > {rot_thresh:.2f}\u00b0 "
                      f"({self.rot_ci*100:.0f}% CI, \u03bc={rot_mean:.2f} \u03c3={rot_std:.2f})")
            self.rejected_count += 1
            return True, reason

        return False, ""

    def accept(self, rmse: float, dr_mag: float):
        """Record an accepted frame's metrics into the running history."""
        if rmse > 0:
            self._rmse_hist.append(rmse)
            self._rot_hist.append(dr_mag)

    def p_values(self, rmse: float, dr_mag: float) -> tuple[float, float]:
        """Return (p_rmse, p_rot) — upper-tail probabilities.

        p = P(X >= observed) under the running normal model.
        Returns (1.0, 1.0) during warmup (no evidence of outlier).
        """
        from scipy.stats import norm
        n = len(self._rmse_hist)
        if n < self.warmup or rmse == 0.0:
            return 1.0, 1.0
        rmse_arr = np.array(self._rmse_hist)
        rot_arr  = np.array(self._rot_hist)
        rmse_mean, rmse_std = float(rmse_arr.mean()), float(rmse_arr.std(ddof=1))
        rot_mean,  rot_std  = float(rot_arr.mean()),  float(rot_arr.std(ddof=1))
        z_rmse = (rmse - rmse_mean) / max(rmse_std, 1e-12)
        z_rot  = (dr_mag - rot_mean) / max(rot_std, 1e-12)
        p_rmse = float(norm.sf(z_rmse))
        p_rot  = float(norm.sf(z_rot))
        return p_rmse, p_rot

    @property
    def history_len(self) -> int:
        return len(self._rmse_hist)


# ══════════════════════════════════════════════════════════════════════════════
# Replay pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_replay(recording_dir: str = ""):
    recording_dir = resolve_recording_dir(recording_dir)
    all_frames = sorted(glob.glob(os.path.join(recording_dir, "frame_*.npz")))
    if not all_frames:
        print(f"No frames in {recording_dir}"); return

    frames = all_frames[::FRAME_SKIP]
    if MAX_FRAMES > 0:
        frames = frames[:MAX_FRAMES]

    register_fn = get_register_fn(REGISTRATION)

    print(f"Replaying {len(frames)} frames from {recording_dir}")
    print(f"  available={len(all_frames)} skip={FRAME_SKIP} max={MAX_FRAMES or 'all'}")
    print(f"  registration={REGISTRATION}  octo_res={OCTO_RESOLUTION}m  mode=deferred")
    print(f"  submaps: enabled  frames_per_submap={SUBMAP_FRAMES}")

    # ── GTSAM iSAM2 ──────────────────────────────────────────────────────
    isam = gtsam.ISAM2()
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

    scan_T_raw = []      # raw 4x4 world poses (for GTSAM fallback)
    pose_count = 0

    # ── Submaps + viewer ──────────────────────────────────────────────────
    submaps: list       = []
    current_submap      = None

    viewer = Viewer3D()

    inv_res  = 1.0 / OCTO_RESOLUTION
    half_res = OCTO_RESOLUTION * 0.5
    seen_keys: set = set()
    VIS_BUF = 2_000_000
    vis_buf = np.zeros((VIS_BUF, 3), dtype=np.float32)
    vis_len = 0

    def _voxelise_and_append(pts):
        """Quantise points to voxel grid and append new voxels to vis_buf."""
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
        """Return the current occupied voxel centres for visualisation."""
        return vis_buf[:vis_len]

    def _recomposite_from_submaps():
        """Rebuild vis_buf from all submaps using updated GTSAM poses.

        O(total_occupied_voxels) — just point transforms, no raycasting.
        """
        nonlocal vis_buf, vis_len, seen_keys
        values = isam.calculateEstimate()
        seen_keys = set()
        vis_buf = np.zeros((VIS_BUF, 3), dtype=np.float32)
        vis_len = 0
        for sm in submaps:
            try:
                updated_T = pose3_to_T(
                    values.atPose3(symbol('x', sm.anchor_index)))
            except Exception:
                updated_T = sm.anchor_T
            world_pts = sm.get_world_points(updated_T)
            if len(world_pts) > 0:
                _voxelise_and_append(world_pts.astype(np.float64))

    # ── Timing ────────────────────────────────────────────────────────────
    tkeys = ["load", "transform", "register", "gtsam",
             "octo_insert", "vox_track", "vis", "total"]
    timings = {k: [] for k in tkeys}
    raw_total = 0

    # Detailed registration sub-timings
    reg_sub_keys = ["local_crop", "src_downsample", "tgt_downsample",
                    "src_normals", "tgt_normals",
                    "gicp_pass0", "gicp_pass1", "gicp_pass2"]
    reg_sub_timings = {k: [] for k in reg_sub_keys}
    reg_src_pts = []   # source point count after downsample
    reg_tgt_pts = []   # target point count after downsample

    # ── Detailed timing log file ──────────────────────────────────────────
    timing_log_path = os.path.join(recording_dir, "timing_log.txt")
    timing_log_file = open(timing_log_path, "w")
    timing_log_file.write(
        f"# Timing log for {recording_dir}\n"
        f"# registration={REGISTRATION}  octo_res={OCTO_RESOLUTION}  "
        f"insert_voxel={OCTO_INSERT_VOXEL}\n"
        f"# submaps=True  submap_frames={SUBMAP_FRAMES}  "
        f"frame_skip={FRAME_SKIP}\n"
        f"# ICP_VOXEL={ICP_VOXEL}  LOCAL_R={LOCAL_R}  NORM_NN={NORM_NN}\n"
        f"#\n"
    )
    timing_log_file.write(
        f"{'frame':>6} {'status':>10} {'raw_pts':>8} {'voxels':>9} "
        f"{'load_ms':>8} {'xform_ms':>9} {'reg_ms':>8} "
        f"{'gtsam_ms':>9} {'octo_ms':>8} {'vox_ms':>8} "
        f"{'vis_ms':>8} {'total_ms':>9} "
        f"{'fitness':>8} {'rmse':>8} {'dt_m':>8} {'dr_deg':>8} "
        f"{'p_rmse':>8} {'p_dr':>8} {'cum_voxels':>10} {'submaps':>8}\n"
    )
    timing_log_file.write(f"{'='*180}\n")
    timing_log_file.flush()
    print(f"  Timing log: {timing_log_path}")

    # ── Registration outlier detector ─────────────────────────────────────
    outlier_det = RegistrationOutlierDetector()

    # ── Live quality plot ─────────────────────────────────────────────────
    q_frames   = []
    q_fitness  = []
    q_rmse     = []
    q_dt       = []
    q_dr       = []
    q_p_rmse   = []
    q_p_dr     = []
    q_rejected = []

    plt.ion()
    fig_q, axes_q = plt.subplots(2, 2, figsize=(12, 7))
    fig_q.suptitle(f"Registration Quality  ({REGISTRATION})", fontsize=13)
    ax_fit, ax_rmse, ax_dt, ax_dr = axes_q.flat

    line_fit,  = ax_fit.plot([], [], 'b-', lw=1)
    ax_fit.set_ylabel('Fitness'); ax_fit.set_xlabel('Frame')
    ax_fit.set_title('Fitness (higher = better overlap)')
    ax_fit.axhline(0.5, color='g', ls='--', lw=0.7, label='good')
    ax_fit.axhline(0.2, color='r', ls='--', lw=0.7, label='poor')
    ax_fit.legend(fontsize=8)

    line_rmse, = ax_rmse.plot([], [], 'r-', lw=1)
    ax_rmse.set_ylabel('Inlier RMSE (m)'); ax_rmse.set_xlabel('Frame')
    ax_rmse.set_title('RMSE (lower = tighter fit)')
    ax_rmse2 = ax_rmse.twinx()
    line_p_rmse, = ax_rmse2.plot([], [], color='orange', ls='--', lw=0.8,
                                 alpha=0.7, label='p-value')
    ax_rmse2.set_ylabel('p-value', color='orange')
    ax_rmse2.tick_params(axis='y', labelcolor='orange')
    ax_rmse2.set_ylim(-0.05, 1.05)
    scat_rej_rmse = ax_rmse.scatter([], [], c='red', marker='x', s=40,
                                    zorder=5, linewidths=1.5, label='rejected')

    line_dt,   = ax_dt.plot([], [], 'm-', lw=1)
    ax_dt.set_ylabel('\u0394t (m)'); ax_dt.set_xlabel('Frame')
    ax_dt.set_title('Translation correction')

    line_dr,   = ax_dr.plot([], [], 'c-', lw=1)
    ax_dr.set_ylabel('\u0394r (\u00b0)'); ax_dr.set_xlabel('Frame')
    ax_dr.set_title('Rotation correction')
    ax_dr2 = ax_dr.twinx()
    line_p_dr, = ax_dr2.plot([], [], color='orange', ls='--', lw=0.8,
                             alpha=0.7, label='p-value')
    ax_dr2.set_ylabel('p-value', color='orange')
    ax_dr2.tick_params(axis='y', labelcolor='orange')
    ax_dr2.set_ylim(-0.05, 1.05)
    scat_rej_dr = ax_dr.scatter([], [], c='red', marker='x', s=40,
                                zorder=5, linewidths=1.5, label='rejected')

    def _update_quality_plot():
        """Refresh all quality plot lines and rejected-frame markers."""
        try:
            line_fit.set_data(q_frames, q_fitness)
            line_rmse.set_data(q_frames, q_rmse)
            line_p_rmse.set_data(q_frames, q_p_rmse)
            line_dt.set_data(q_frames, q_dt)
            line_dr.set_data(q_frames, q_dr)
            line_p_dr.set_data(q_frames, q_p_dr)
            if q_rejected:
                rej_xy_rmse = np.column_stack(
                    ([q_frames[j] for j in q_rejected],
                     [q_rmse[j]   for j in q_rejected]))
                rej_xy_dr = np.column_stack(
                    ([q_frames[j] for j in q_rejected],
                     [q_dr[j]     for j in q_rejected]))
                scat_rej_rmse.set_offsets(rej_xy_rmse)
                scat_rej_dr.set_offsets(rej_xy_dr)
            for ax in axes_q.flat:
                ax.relim(); ax.autoscale_view()
            ax_rmse2.set_ylim(-0.05, 1.05)
            ax_dr2.set_ylim(-0.05, 1.05)
            fig_q.canvas.draw_idle()
            fig_q.canvas.flush_events()
        except Exception:
            pass

    for ax in axes_q.flat:
        ax.grid(True, alpha=0.3)
    fig_q.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    PLOT_UPDATE_EVERY = 1

    # ══════════════════════════════════════════════════════════════════════
    # Main frame loop
    # ══════════════════════════════════════════════════════════════════════
    for i, path in enumerate(frames):
        t_frame = time.perf_counter()

        # ── Load ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        data = np.load(path)
        pts = filter_valid(data["points"])
        if len(pts) == 0:
            continue
        timings["load"].append(time.perf_counter() - t0)

        # ── Transform to world (initial guess) ───────────────────────────
        t0 = time.perf_counter()
        lp = data["lidar_position"] if "lidar_position" in data.files else np.zeros(3)
        lo = (data["lidar_orientation"] if "lidar_orientation" in data.files
              else np.array([1, 0, 0, 0], dtype=float))
        body = xform_pts(pts, lp, lo)
        pos = data["position"] if "position" in data.files else np.zeros(3)
        ori = (data["orientation"] if "orientation" in data.files
               else np.array([1, 0, 0, 0], dtype=float))
        world_init = xform_pts(body, pos, ori)
        timings["transform"].append(time.perf_counter() - t0)
        raw_total += len(world_init)

        # Raw world-pose 4x4
        T_raw = np.eye(4)
        R_raw = Rotation.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_matrix()
        T_raw[:3, :3] = R_raw
        T_raw[:3, 3] = pos

        # ── Registration against voxel map ────────────────────────────────
        t0 = time.perf_counter()
        if vis_len >= MIN_VOXELS and pose_count > 0 and REGISTRATION != "state_only":
            target_pts = _get_vis()
            # Crop target to local sphere around drone for speed
            if REG_LOCAL_RADIUS > 0:
                dists = np.linalg.norm(target_pts - pos.reshape(1, 3), axis=1)
                mask = dists <= REG_LOCAL_RADIUS
                target_pts = target_pts[mask]
            T_reg, fitness, rmse, reg_detail = register_fn(
                world_init.astype(np.float64),
                target_pts.astype(np.float64))
            world_pts = apply_T(world_init, T_reg).astype(np.float32)

            # Record registration sub-timings
            if isinstance(reg_detail, dict):
                for sk in reg_sub_keys:
                    reg_sub_timings[sk].append(reg_detail.get(sk, 0.0))
                reg_src_pts.append(reg_detail.get('src_pts', 0))
                reg_tgt_pts.append(reg_detail.get('tgt_pts', 0))

            ct = T_reg[:3, 3]
            ce = Rotation.from_matrix(T_reg[:3, :3]).as_euler("xyz", degrees=True)
            dt_mag = float(np.linalg.norm(ct))
            dr_mag = float(np.linalg.norm(ce))
            src_n = reg_detail.get('src_pts', 0) if isinstance(reg_detail, dict) else 0
            tgt_n = reg_detail.get('tgt_pts', 0) if isinstance(reg_detail, dict) else 0
            print(f"  REG {i:03d}: fit={fitness:.4f} rmse={rmse:.4f} "
                  f"\u0394t={dt_mag:.4f}m "
                  f"\u0394r=({ce[0]:+.2f},{ce[1]:+.2f},{ce[2]:+.2f})\u00b0"
                  f"  src={src_n:,} tgt={tgt_n:,}")
        else:
            world_pts = world_init.astype(np.float32)
            T_reg = np.eye(4)
            fitness = 0.0
            rmse = 0.0
            dt_mag = 0.0
            dr_mag = 0.0
            print(f"  frame {i:03d}: "
                  f"{'baseline' if REGISTRATION == 'state_only' else f'too few voxels ({vis_len})'}"
                  f", state pose only")
        timings["register"].append(time.perf_counter() - t0)

        # Record quality metrics
        q_frames.append(i)
        q_fitness.append(fitness)
        q_rmse.append(rmse)
        q_dt.append(dt_mag)
        q_dr.append(dr_mag)

        # ── Outlier rejection + p-value computation ───────────────────────
        p_rmse_val, p_dr_val = outlier_det.p_values(rmse, dr_mag)
        q_p_rmse.append(p_rmse_val)
        q_p_dr.append(p_dr_val)

        if REJECT_OUTLIERS and rmse > 0:
            is_outlier, reason = outlier_det.check(rmse, dr_mag)
            if is_outlier:
                print(f"  \u2717 REJECTED frame {i:03d}: {reason}")
                q_rejected.append(len(q_frames) - 1)
                timings["total"].append(time.perf_counter() - t_frame)
                _update_quality_plot()
                _tl = lambda k: timings[k][-1] * 1e3 if timings[k] else 0.0
                timing_log_file.write(
                    f"{i:6d} {'rejected':>10} {len(pts):8d} {vis_len:9d} "
                    f"{_tl('load'):8.2f} {_tl('transform'):9.2f} {_tl('register'):8.2f} "
                    f"{'--':>9} {'--':>8} {'--':>8} "
                    f"{'--':>8} {_tl('total'):9.2f} "
                    f"{fitness:8.4f} {rmse:8.4f} {dt_mag:8.4f} {dr_mag:8.4f} "
                    f"{q_p_rmse[-1]:8.4f} {q_p_dr[-1]:8.4f} "
                    f"{vis_len:10d} {len(submaps):8d}\n"
                )
                timing_log_file.flush()
                continue
            outlier_det.accept(rmse, dr_mag)

        # ── GTSAM factor graph ────────────────────────────────────────────
        t0 = time.perf_counter()
        key_curr = symbol('x', pose_count)

        pose3_curr = pose3_from_pos_quat(pos, ori)
        if np.any(T_reg[:3, :3] != np.eye(3)) or np.any(T_reg[:3, 3] != 0):
            corrected_T = T_reg @ T_raw
            pose3_curr = T_to_pose3(corrected_T)

        if pose_count == 0:
            graph.add(gtsam.PriorFactorPose3(key_curr, pose3_curr, prior_noise))
            initial_estimates.insert(key_curr, pose3_curr)
        else:
            key_prev = symbol('x', pose_count - 1)
            try:
                prev_est = isam.calculateEstimate().atPose3(key_prev)
            except Exception:
                prev_est = T_to_pose3(scan_T_raw[-1])
            relative_pose = prev_est.between(pose3_curr)
            noise = (_icp_noise(rmse) if rmse > 0
                     else gtsam.noiseModel.Isotropic.Sigma(6, 0.5))
            graph.add(gtsam.BetweenFactorPose3(
                key_prev, key_curr, relative_pose, noise))
            initial_estimates.insert(key_curr, pose3_curr)

        # GPS prior
        if "gps" in data.files:
            gps = data["gps"]
            gps_pt = gtsam.Point3(float(gps[0]), float(gps[1]), float(gps[2]))
            graph.add(gtsam.GPSFactor(
                key_curr, gps_pt,
                gtsam.noiseModel.Isotropic.Sigma(3, GPS_SIGMA)))

        isam.update(graph, initial_estimates)
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()
        timings["gtsam"].append(time.perf_counter() - t0)

        scan_T_raw.append(T_raw.copy())
        T_est_curr = (T_reg @ T_raw) if (
            np.any(T_reg[:3, :3] != np.eye(3))
            or np.any(T_reg[:3, 3] != 0)) else T_raw
        pose_count += 1

        # ── Submap management / voxel insertion ───────────────────────────
        if (current_submap is None
                or len(current_submap.frame_indices) >= SUBMAP_FRAMES):
            current_submap = Submap(
                anchor_index=pose_count - 1,
                anchor_T=T_est_curr.copy())
            submaps.append(current_submap)
        current_submap.frame_indices.append(pose_count - 1)

        t0 = time.perf_counter()
        wf64 = world_pts.astype(np.float64)
        ds_pts = _downsample_for_insert(wf64)
        current_submap.insert(ds_pts)
        timings["octo_insert"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _voxelise_and_append(wf64)
        timings["vox_track"].append(time.perf_counter() - t0)

        # ── Visualise ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        vp = _get_vis()
        if not viewer._proc:
            viewer.start(initial_points=vp)
        else:
            viewer.update(vp)
        timings["vis"].append(time.perf_counter() - t0)

        timings["total"].append(time.perf_counter() - t_frame)

        # ── Update live quality plot ──────────────────────────────────────
        if len(q_frames) % PLOT_UPDATE_EVERY == 0:
            _update_quality_plot()

        # ── Per-frame console log ─────────────────────────────────────────
        total_mem = sum(sm.memoryUsage() for sm in submaps)
        print(f"  {i+1:3d}/{len(frames)} | raw={len(pts):6,} "
              f"voxels={vis_len:8,} submaps={len(submaps)} "
              f"mem={total_mem/1e6:.1f}MB | "
              f"tot={timings['total'][-1]*1e3:.0f}ms\n", flush=True)

        # ── Write detailed timing to log file ─────────────────────────────
        _tl = lambda k: timings[k][-1] * 1e3 if timings[k] else 0.0
        timing_log_file.write(
            f"{i:6d} {'ok':>10} {len(pts):8d} {vis_len:9d} "
            f"{_tl('load'):8.2f} {_tl('transform'):9.2f} {_tl('register'):8.2f} "
            f"{_tl('gtsam'):9.2f} {_tl('octo_insert'):8.2f} {_tl('vox_track'):8.2f} "
            f"{_tl('vis'):8.2f} {_tl('total'):9.2f} "
            f"{fitness:8.4f} {rmse:8.4f} {dt_mag:8.4f} {dr_mag:8.4f} "
            f"{q_p_rmse[-1]:8.4f} {q_p_dr[-1]:8.4f} "
            f"{vis_len:10d} {len(submaps):8d}\n"
        )
        timing_log_file.flush()

    # ══════════════════════════════════════════════════════════════════════
    # Post-loop: save outputs and print summary
    # ══════════════════════════════════════════════════════════════════════

    # ── Save final quality plot ────────────────────────────────────────────
    try:
        bad_idx = [j for j, f in enumerate(q_fitness) if 0 < f < 0.3]
        if bad_idx:
            ax_fit.scatter([q_frames[j] for j in bad_idx],
                           [q_fitness[j] for j in bad_idx],
                           c='red', s=25, zorder=5, label='fit < 0.3')
            ax_fit.legend(fontsize=8)

        if q_rejected:
            rej_frames_rmse = [q_frames[j] for j in q_rejected]
            rej_rmse_vals   = [q_rmse[j]   for j in q_rejected]
            rej_dr_vals     = [q_dr[j]     for j in q_rejected]
            ax_rmse.scatter(rej_frames_rmse, rej_rmse_vals,
                            c='red', marker='x', s=40, zorder=5,
                            linewidths=1.5, label='rejected')
            ax_rmse.legend(fontsize=8, loc='upper left')
            ax_dr.scatter(rej_frames_rmse, rej_dr_vals,
                          c='red', marker='x', s=40, zorder=5,
                          linewidths=1.5, label='rejected')
            ax_dr.legend(fontsize=8, loc='upper left')

        for ax in axes_q.flat:
            ax.relim(); ax.autoscale_view()
        ax_rmse2.set_ylim(-0.05, 1.05)
        ax_dr2.set_ylim(-0.05, 1.05)
        png_path = os.path.join(recording_dir, "reg_quality.png")
        fig_q.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"  Quality plot saved to {png_path}")
    except Exception as e:
        print(f"  (could not save quality plot: {e})")

    # ── Build final OctoMap for .bt export ─────────────────────────────────
    print("  Building final OctoMap for export...")
    try:
        merged = pyoctomap.OcTree(OCTO_RESOLUTION)
        values = isam.calculateEstimate()
        for sm in submaps:
            try:
                T = pose3_to_T(
                    values.atPose3(symbol('x', sm.anchor_index)))
            except Exception:
                T = sm.anchor_T
            wpts = sm.get_world_points(T)
            if len(wpts) > 0:
                merged.updateNodes(wpts.astype(np.float64), True)
        bt_path = os.path.join(recording_dir, "map.bt")
        merged.writeBinary(bt_path)
        print(f"  OctoMap saved to {bt_path} "
              f"({merged.size():,} nodes, "
              f"{merged.memoryUsage()/1e6:.1f}MB)")
    except Exception as e:
        print(f"  (could not save OctoMap: {e})")

    # ── Final vis recomposite ─────────────────────────────────────────────
    _recomposite_from_submaps()

    plane = fit_plane(_get_vis())
    ratio = vis_len / raw_total * 100 if raw_total else 0

    print(f"\n{'='*65}")
    print(f"Final Mapping Summary  (registration={REGISTRATION})")
    print(f"{'='*65}")
    print(f"  Octo resolution:  {OCTO_RESOLUTION} m")
    print(f"  Submaps:          {len(submaps)} submaps, "
          f"{SUBMAP_FRAMES} frames/submap")
    if REJECT_OUTLIERS:
        print(f"  Outlier rejection: {outlier_det.rejected_count} frames rejected "
              f"(warmup={REJECT_WARMUP}, RMSE CI={REJECT_RMSE_CI*100:.0f}%, "
              f"Rot CI={REJECT_ROT_CI*100:.0f}%)")
    print(f"  Raw pts total:    {raw_total:,}")
    print(f"  Occupied voxels:  {vis_len:,}  "
          f"({ratio:.1f}% -> {raw_total/max(vis_len,1):.1f}x)")
    total_mem = sum(sm.memoryUsage() for sm in submaps)
    total_nodes = sum(sm.node_count() for sm in submaps)
    print(f"  Map nodes:        {total_nodes:,}  "
          f"mem={total_mem/1e6:.1f}MB  ({len(submaps)} submaps)")
    print(f"  Pose graph nodes: {pose_count}")
    if plane:
        n, sx, sy, res = plane
        print(f"  Plane residual:   {res:.4f} m")
        print(f"  Plane slope:      x={sx:+.4f}  y={sy:+.4f}")

    gt = sum(timings["total"])
    summary_lines = []
    summary_lines.append(
        f"\n  {'step':<20} {'total':>7} {'mean':>7} {'max':>7} "
        f"{'min':>7} {'std':>7} {'%':>5}  {'count':>5}")
    summary_lines.append(f"  {'-'*76}")
    for k in tkeys:
        v = timings[k]
        if not v:
            continue
        s = sum(v)
        pct = 100 * s / gt if gt else 0
        va = np.array(v)
        summary_lines.append(
            f"  {k:<20} {s:>7.3f} {va.mean():>7.4f} {va.max():>7.4f} "
            f"{va.min():>7.4f} {va.std():>7.4f} {pct:>4.1f}%  {len(v):>5d}")
        # Expand registration sub-timings inline
        if k == "register":
            for sk in reg_sub_keys:
                sv = reg_sub_timings[sk]
                if not sv:
                    continue
                ss = sum(sv)
                sp = 100 * ss / gt if gt else 0
                sa = np.array(sv)
                summary_lines.append(
                    f"    {sk:<18} {ss:>7.3f} {sa.mean():>7.4f} {sa.max():>7.4f} "
                    f"{sa.min():>7.4f} {sa.std():>7.4f} {sp:>4.1f}%  {len(sv):>5d}")
            if reg_src_pts:
                sa = np.array(reg_src_pts)
                summary_lines.append(
                    f"    {'src_pts':<18} {'':>7} {sa.mean():>7.0f} {sa.max():>7d} "
                    f"{sa.min():>7d} {sa.std():>7.0f} {'':>5}  {len(sa):>5d}")
            if reg_tgt_pts:
                ta = np.array(reg_tgt_pts)
                summary_lines.append(
                    f"    {'tgt_pts':<18} {'':>7} {ta.mean():>7.0f} {ta.max():>7d} "
                    f"{ta.min():>7d} {ta.std():>7.0f} {'':>5}  {len(ta):>5d}")
    summary_lines.append(f"  {'='*76}")
    for line in summary_lines:
        print(line)

    # Write summary to timing log
    timing_log_file.write(f"\n{'='*70}\n")
    timing_log_file.write(f"SUMMARY\n")
    timing_log_file.write(f"{'='*70}\n")
    for line in summary_lines:
        timing_log_file.write(line + "\n")
    timing_log_file.write(f"\nTotal wall-clock: {gt:.3f}s\n")
    timing_log_file.write(f"Frames processed: {len(timings['total'])}\n")
    timing_log_file.write(f"Frames rejected:  {outlier_det.rejected_count}\n")
    timing_log_file.close()
    print(f"\n  Detailed timing log saved to {timing_log_path}")

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
