#!/usr/bin/env python3
"""SLAM pipeline — GTSAM pose-graph + deferred OctoMap + swappable registration.

Object-oriented design for both **replay** and **live AirSim** mapping.

Core class:
    SLAMPipeline  — source-agnostic.  Feed it frames via ``process_frame()``
                     and it handles registration, pose-graph optimisation,
                     submap management, and visualisation.

Adapters:
    ReplaySLAM    — reads flight_recordings/ directories, feeds SLAMPipeline.
    LiveSLAM      — connects to AirSim, grabs LiDAR + pose in real time.

Usage (replay):
    python finalMappingPipeline.py                       # latest recording
    python finalMappingPipeline.py /path/to/flight_dir   # explicit directory

Usage (live — from another script):
    from finalMappingPipeline import SLAMPipeline, SLAMConfig, LiveSLAM
    live = LiveSLAM(SLAMConfig(registration="vgicp"))
    live.run()          # blocking — Ctrl+C to stop

Usage (embed in your own loop):
    from finalMappingPipeline import SLAMPipeline, SLAMConfig
    cfg  = SLAMConfig(registration="vgicp", octo_resolution=0.15)
    slam = SLAMPipeline(cfg)
    slam.start_viewer()
    for pts, pos, ori, gps in my_data_source():
        slam.process_frame(pts, pos, ori, gps=gps)
    slam.save_octomap("map.bt")
    slam.print_summary()
"""

from __future__ import annotations

import os, sys, glob, time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import open3d as o3d
import pyoctomap
import gtsam
from gtsam import symbol
from scipy.spatial.transform import Rotation
from sensorFeed import Viewer3D

from RegistrationComparison import (
    get_register_fn, resolve_recording_dir, filter_valid, xform_pts,
    apply_T, fit_plane, evaluate_registration,
    ICP_VOXEL, NORM_NN, LOCAL_R, MIN_VOXELS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SLAMConfig:
    """All tuneable knobs for the SLAM pipeline."""

    # Registration algorithm
    # "gicp" | "small_gicp" | "vgicp" | "kiss_icp" | "icp" | "fpfh_ransac" | "ndt" | "state_only"
    registration: str = "vgicp"

    # OctoMap / voxel grid
    octo_resolution: float   = 0.15   # leaf voxel size (m)
    octo_insert_voxel: float = 0.15   # downsample before submap insertion (0 = skip)

    # Submaps
    submap_frames: int = 10           # frames per submap

    # Registration outlier rejection
    reject_outliers: bool  = True
    reject_warmup: int     = 7
    reject_dt_ci: float    = 0.80     # translation CI threshold
    reject_rot_ci: float   = 0.80    # rotation CI threshold

    # Registration target cropping
    reg_local_radius: float = 40.0    # crop target cloud to this radius (m, 0 = no crop)

    # GTSAM noise
    gps_sigma: float       = 0.2     # GPS position noise (m)
    icp_noise_scale: float = 1.0     # multiplier on ICP-derived noise model

    # Visualisation
    enable_viewer: bool = True
    vis_buffer_size: int = 2_000_000

    # Replay-specific
    recording_dir: str = ""
    max_frames: int    = 0            # 0 = all
    frame_skip: int    = 3

    # Live-specific
    live_max_hz: float = 4.0         # max sensor polling rate


# ══════════════════════════════════════════════════════════════════════════════
# VoxelHashGrid — ultra-fast voxel storage
# ══════════════════════════════════════════════════════════════════════════════

class VoxelHashGrid:
    """O(1)-insert voxel grid backed by a Python set of integer keys."""

    __slots__ = ('resolution', 'inv_res', 'half_res', 'keys')

    def __init__(self, resolution: float):
        self.resolution = resolution
        self.inv_res    = 1.0 / resolution
        self.half_res   = resolution * 0.5
        self.keys: set  = set()

    def insert_points(self, pts: np.ndarray):
        if len(pts) == 0:
            return
        ijk = np.floor(np.asarray(pts) * self.inv_res).astype(np.int64)
        self.keys.update(map(tuple, ijk))

    def get_centers(self) -> np.ndarray:
        if not self.keys:
            return np.empty((0, 3), dtype=np.float64)
        arr = np.array(list(self.keys), dtype=np.float64)
        return arr * self.resolution + self.half_res

    def size(self) -> int:
        return len(self.keys)

    def memory_usage(self) -> int:
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
                 resolution: float):
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
        local_pts = self._to_local(world_pts).astype(np.float64)
        self.grid.insert_points(local_pts)

    def get_world_points(self, updated_T: np.ndarray | None = None) -> np.ndarray:
        occ = self.grid.get_centers()
        if len(occ) == 0:
            return np.empty((0, 3), dtype=np.float64)
        T = updated_T if updated_T is not None else self.anchor_T
        return apply_T(occ, T)

    def memory_usage(self) -> int:
        return self.grid.memory_usage()

    def node_count(self) -> int:
        return self.grid.size()


# ══════════════════════════════════════════════════════════════════════════════
# GTSAM helpers
# ══════════════════════════════════════════════════════════════════════════════

def pose3_from_pos_quat(pos, quat_wxyz):
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]),
                   float(quat_wxyz[2]), float(quat_wxyz[3]))
    return gtsam.Pose3(gtsam.Rot3.Quaternion(w, x, y, z),
                       gtsam.Point3(float(pos[0]), float(pos[1]), float(pos[2])))


def pose3_to_T(p: gtsam.Pose3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = p.rotation().matrix()
    T[:3, 3]  = np.array(p.translation()).reshape(3)
    return T


def T_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(T)


def _icp_noise(rmse: float, scale: float = 1.0):
    rot_s = max(0.05, rmse * 0.5) * scale
    tra_s = max(0.10, rmse * 1.5) * scale
    sigma = float(max(rot_s, tra_s))
    return gtsam.noiseModel.Isotropic.Sigma(6, sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Registration outlier detector
# ══════════════════════════════════════════════════════════════════════════════

class RegistrationOutlierDetector:
    """Detect bad registrations via confidence-interval spike detection.

    Rejects frames where **both** the translation magnitude and the
    rotation magnitude exceed their respective CI thresholds.
    """

    def __init__(self, warmup: int = 4, dt_ci: float = 0.70,
                 rot_ci: float = 0.70):
        self.warmup = max(3, warmup)
        self.dt_ci  = dt_ci
        self.rot_ci = rot_ci
        self._dt_hist:  list[float] = []
        self._rot_hist: list[float] = []
        self.rejected_count = 0

    @staticmethod
    def _z(ci: float) -> float:
        from scipy.stats import norm
        return float(norm.ppf((1.0 + ci) / 2.0))

    def check(self, dt_mag: float, dr_mag: float) -> tuple[bool, str]:
        n = len(self._dt_hist)
        if n < self.warmup:
            return False, ""
        dt_arr  = np.array(self._dt_hist)
        rot_arr = np.array(self._rot_hist)
        dt_mean,  dt_std  = float(dt_arr.mean()),  float(dt_arr.std(ddof=1))
        rot_mean, rot_std = float(rot_arr.mean()), float(rot_arr.std(ddof=1))
        z_dt  = self._z(self.dt_ci)
        z_rot = self._z(self.rot_ci)
        dt_thresh  = dt_mean  + z_dt  * dt_std
        rot_thresh = rot_mean + z_rot * rot_std
        dt_bad  = dt_mag > dt_thresh
        rot_bad = dr_mag > rot_thresh
        if dt_bad and rot_bad:
            reason = (f"\u0394t {dt_mag:.4f}m > {dt_thresh:.4f}m "
                      f"({self.dt_ci*100:.0f}% CI, \u03bc={dt_mean:.4f} \u03c3={dt_std:.4f}) "
                      f"AND \u0394r {dr_mag:.2f}\u00b0 > {rot_thresh:.2f}\u00b0 "
                      f"({self.rot_ci*100:.0f}% CI, \u03bc={rot_mean:.2f} \u03c3={rot_std:.2f})")
            self.rejected_count += 1
            return True, reason
        return False, ""

    def accept(self, dt_mag: float, dr_mag: float):
        self._dt_hist.append(dt_mag)
        self._rot_hist.append(dr_mag)

    def p_values(self, dt_mag: float, dr_mag: float) -> tuple[float, float]:
        from scipy.stats import norm
        n = len(self._dt_hist)
        if n < self.warmup:
            return 1.0, 1.0
        dt_arr  = np.array(self._dt_hist)
        rot_arr = np.array(self._rot_hist)
        dt_mean,  dt_std  = float(dt_arr.mean()),  float(dt_arr.std(ddof=1))
        rot_mean, rot_std = float(rot_arr.mean()), float(rot_arr.std(ddof=1))
        z_dt  = (dt_mag - dt_mean)  / max(dt_std,  1e-12)
        z_rot = (dr_mag - rot_mean) / max(rot_std, 1e-12)
        p_dt  = float(norm.sf(z_dt))
        p_rot = float(norm.sf(z_rot))
        return p_dt, p_rot

    @property
    def history_len(self) -> int:
        return len(self._dt_hist)


# ══════════════════════════════════════════════════════════════════════════════
# SLAMPipeline — core engine (source-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

class SLAMPipeline:
    """GTSAM pose-graph SLAM with submap-based voxel storage.

    Feed frames one at a time via ``process_frame()``.  The pipeline handles:
      - Frame-to-map registration (swappable algorithm)
      - GTSAM iSAM2 pose-graph with between-factors + GPS priors
      - Submap management with anchor-pose correction
      - Real-time visualisation (optional)

    Works identically for replay and live data.
    """

    def __init__(self, config: SLAMConfig | None = None):
        self.cfg = config or SLAMConfig()

        # Registration function
        self._register_fn = get_register_fn(self.cfg.registration)

        # GTSAM iSAM2
        self._isam = gtsam.ISAM2()
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial_estimates = gtsam.Values()
        self._prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

        # Pose bookkeeping
        self._scan_T_raw: list[np.ndarray] = []
        self._pose_count = 0

        # Submaps
        self._submaps: list[Submap] = []
        self._current_submap: Submap | None = None

        # Visualisation buffer (fast voxel dedup for viewer)
        self._inv_res  = 1.0 / self.cfg.octo_resolution
        self._half_res = self.cfg.octo_resolution * 0.5
        self._seen_keys: set = set()
        self._vis_buf = np.zeros((self.cfg.vis_buffer_size, 3), dtype=np.float32)
        self._vis_len = 0

        # Viewer
        self._viewer: Viewer3D | None = None
        self._drone_pos: np.ndarray | None = None
        self._target_pos: np.ndarray | None = None
        self._frontier_points: np.ndarray | None = None
        self._path_points: np.ndarray | None = None
        self._candidate_points: np.ndarray | None = None

        # Outlier detector
        self._outlier_det = RegistrationOutlierDetector(
            warmup=self.cfg.reject_warmup,
            dt_ci=self.cfg.reject_dt_ci,
            rot_ci=self.cfg.reject_rot_ci,
        )

        # Counters / stats
        self._raw_total = 0
        self._frame_index = 0  # total frames submitted (including rejected)
        self._frames_accepted = 0

        # Per-frame quality metrics (for plotting / analysis)
        self.q_frames:   list[int]   = []
        self.q_fitness:  list[float] = []
        self.q_rmse:     list[float] = []
        self.q_inlier_rmse: list[float] = []
        self.q_dt:       list[float] = []
        self.q_dr:       list[float] = []
        self.q_p_dt:     list[float] = []
        self.q_p_dr:     list[float] = []
        self.q_rejected: list[int]   = []

        # Timing
        self._tkeys = ["load", "transform", "register", "gtsam",
                        "octo_insert", "vox_track", "vis", "total"]
        self.timings = {k: [] for k in self._tkeys}

        # Callbacks (optional hooks for logging / plotting / custom actions)
        self._on_frame_callbacks: list[Callable] = []

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def pose_count(self) -> int:
        return self._pose_count

    @property
    def voxel_count(self) -> int:
        return self._vis_len

    @property
    def submap_count(self) -> int:
        return len(self._submaps)

    @property
    def rejected_count(self) -> int:
        return self._outlier_det.rejected_count

    @property
    def submaps(self) -> list[Submap]:
        return self._submaps

    @property
    def isam(self) -> gtsam.ISAM2:
        return self._isam

    # ── Viewer management ─────────────────────────────────────────────────

    def start_viewer(self):
        """Launch the Open3D viewer in a separate process."""
        if self._viewer is None:
            self._viewer = Viewer3D()

    def stop_viewer(self):
        """Stop the viewer process."""
        if self._viewer is not None:
            self._viewer.stop()
            self._viewer = None

    def set_drone_pos(self, pos):
        """Set the drone position for the viewer marker."""
        self._drone_pos = np.asarray(pos, dtype=float) if pos is not None else None

    def set_target_pos(self, pos):
        """Set the target waypoint for the viewer marker."""
        self._target_pos = np.asarray(pos, dtype=float) if pos is not None else None

    def set_frontier_points(self, pts):
        """Set frontier overlay points (Nx3) shown as orange in the viewer."""
        if pts is not None and len(pts) > 0:
            self._frontier_points = np.asarray(pts, dtype=np.float64)
        else:
            self._frontier_points = None

    def set_path_points(self, pts):
        """Set planned-path overlay points (Nx3) shown as cyan line in the viewer."""
        if pts is not None and len(pts) > 0:
            self._path_points = np.asarray(pts, dtype=np.float64)
        else:
            self._path_points = None

    def set_candidate_points(self, pts):
        """Set NBV candidate overlay points (Nx3) shown as magenta in the viewer."""
        if pts is not None and len(pts) > 0:
            self._candidate_points = np.asarray(pts, dtype=np.float64)
        else:
            self._candidate_points = None

    def refresh_overlays(self):
        """Push current drone/target/path/candidate overlays to the viewer without a new scan.

        This is much cheaper than a full ``update()`` because it skips the
        point-cloud copy.  Call at high frequency (e.g. 20 Hz) for smooth
        marker tracking.
        """
        if self._viewer is not None and self._viewer._proc:
            self._viewer.update_overlays(
                drone_pos=self._drone_pos,
                target_pos=self._target_pos,
                path_points=self._path_points,
                candidate_points=self._candidate_points,
            )

    # ── Callback registration ─────────────────────────────────────────────

    def on_frame(self, callback: Callable):
        """Register a callback invoked after each accepted frame.

        Signature: callback(pipeline, frame_result_dict)
        """
        self._on_frame_callbacks.append(callback)

    # ── Core frame processing ─────────────────────────────────────────────

    def process_frame(
        self,
        points: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
        *,
        gps: np.ndarray | None = None,
        lidar_position: np.ndarray | None = None,
        lidar_orientation: np.ndarray | None = None,
        frame_label: int | None = None,
    ) -> dict:
        """Process a single scan frame.

        Parameters
        ----------
        points : (N, 3) float — raw LiDAR points in sensor frame.
        position : (3,) float — vehicle world position.
        orientation : (4,) float — vehicle orientation as (w, x, y, z) quaternion.
        gps : (3,) float, optional — GPS measurement (lat/lon/alt or local XYZ).
        lidar_position : (3,) float, optional — LiDAR mount offset on vehicle.
        lidar_orientation : (4,) float, optional — LiDAR mount orientation (w,x,y,z).
        frame_label : int, optional — frame number for logging (auto-incremented if omitted).

        Returns
        -------
        dict with keys: accepted, fitness, rmse, dt, dr, voxels, pose_count, etc.
        """
        t_frame = time.perf_counter()
        idx = frame_label if frame_label is not None else self._frame_index
        self._frame_index += 1

        result = {
            "accepted": False, "frame": idx, "fitness": 0.0, "rmse": 0.0,
            "dt": 0.0, "dr": 0.0, "voxels": self._vis_len,
            "pose_count": self._pose_count, "submaps": len(self._submaps),
        }

        # ── Filter valid points ───────────────────────────────────────────
        pts = filter_valid(points)
        if len(pts) == 0:
            return result

        # ── Transform to world (initial guess) ───────────────────────────
        t0 = time.perf_counter()
        lp = lidar_position if lidar_position is not None else np.zeros(3)
        lo = lidar_orientation if lidar_orientation is not None else np.array([1, 0, 0, 0], dtype=float)
        body = xform_pts(pts, lp, lo)

        pos = np.asarray(position, dtype=float)
        ori = np.asarray(orientation, dtype=float)
        world_init = xform_pts(body, pos, ori)
        self.timings["transform"].append(time.perf_counter() - t0)
        self._raw_total += len(world_init)

        # Raw world-pose 4x4
        T_raw = np.eye(4)
        R_raw = Rotation.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_matrix()
        T_raw[:3, :3] = R_raw
        T_raw[:3, 3]  = pos

        # ── Registration against voxel map ────────────────────────────────
        t0 = time.perf_counter()
        if (self._vis_len >= MIN_VOXELS
                and self._pose_count > 0
                and self.cfg.registration != "state_only"):
            target_pts = self._get_vis()
            if self.cfg.reg_local_radius > 0:
                dists = np.linalg.norm(target_pts - pos.reshape(1, 3), axis=1)
                target_pts = target_pts[dists <= self.cfg.reg_local_radius]

            T_reg, fitness, rmse, reg_detail = self._register_fn(
                world_init.astype(np.float64),
                target_pts.astype(np.float64))
            world_pts = apply_T(world_init, T_reg).astype(np.float32)

            # Re-evaluate with all-points RMSE — Open3D's inlier_rmse only
            # measures points within max_correspondence_distance, which
            # stays deceptively low even on badly misaligned registrations.
            _, inlier_rmse, full_rmse, mean_dist = evaluate_registration(
                world_init.astype(np.float64),
                target_pts.astype(np.float64),
                T_reg,
                max_correspondence_distance=ICP_VOXEL,
                downsample_voxel=ICP_VOXEL,
            )
            rmse = full_rmse  # use all-points RMSE for quality gating
            inlier_rmse_val = inlier_rmse  # keep for plotting

            ct = T_reg[:3, 3]
            ce = Rotation.from_matrix(T_reg[:3, :3]).as_euler("xyz", degrees=True)
            dt_mag = float(np.linalg.norm(ct))
            dr_mag = float(np.linalg.norm(ce))
            src_n = reg_detail.get('src_pts', 0) if isinstance(reg_detail, dict) else 0
            tgt_n = reg_detail.get('tgt_pts', 0) if isinstance(reg_detail, dict) else 0
            print(f"  REG {idx:03d}: fit={fitness:.4f} rmse={full_rmse:.4f} "
                  f"(inlier={inlier_rmse:.4f} mean={mean_dist:.4f}) "
                  f"\u0394t={dt_mag:.4f}m "
                  f"\u0394r=({ce[0]:+.2f},{ce[1]:+.2f},{ce[2]:+.2f})\u00b0"
                  f"  src={src_n:,} tgt={tgt_n:,}")
        else:
            world_pts = world_init.astype(np.float32)
            T_reg = np.eye(4)
            fitness = rmse = dt_mag = dr_mag = 0.0
            inlier_rmse_val = 0.0
            print(f"  frame {idx:03d}: "
                  f"{'baseline' if self.cfg.registration == 'state_only' else f'too few voxels ({self._vis_len})'}"
                  f", state pose only")
        self.timings["register"].append(time.perf_counter() - t0)

        # Record quality metrics
        self.q_frames.append(idx)
        self.q_fitness.append(fitness)
        self.q_rmse.append(rmse)
        self.q_inlier_rmse.append(inlier_rmse_val)
        self.q_dt.append(dt_mag)
        self.q_dr.append(dr_mag)

        # ── Outlier rejection ─────────────────────────────────────────────
        p_dt_val, p_dr_val = self._outlier_det.p_values(dt_mag, dr_mag)
        self.q_p_dt.append(p_dt_val)
        self.q_p_dr.append(p_dr_val)

        if self.cfg.reject_outliers and dt_mag > 0:
            is_outlier, reason = self._outlier_det.check(dt_mag, dr_mag)
            if is_outlier:
                print(f"  \u2717 REJECTED frame {idx:03d}: {reason}")
                self.q_rejected.append(len(self.q_frames) - 1)
                self.timings["total"].append(time.perf_counter() - t_frame)
                result.update(fitness=fitness, rmse=rmse, dt=dt_mag, dr=dr_mag)
                return result
            self._outlier_det.accept(dt_mag, dr_mag)

        # ── GTSAM factor graph ────────────────────────────────────────────
        t0 = time.perf_counter()
        key_curr = symbol('x', self._pose_count)

        pose3_curr = pose3_from_pos_quat(pos, ori)
        if np.any(T_reg[:3, :3] != np.eye(3)) or np.any(T_reg[:3, 3] != 0):
            corrected_T = T_reg @ T_raw
            pose3_curr = T_to_pose3(corrected_T)

        if self._pose_count == 0:
            self._graph.add(gtsam.PriorFactorPose3(key_curr, pose3_curr, self._prior_noise))
            self._initial_estimates.insert(key_curr, pose3_curr)
        else:
            key_prev = symbol('x', self._pose_count - 1)
            try:
                prev_est = self._isam.calculateEstimate().atPose3(key_prev)
            except Exception:
                prev_est = T_to_pose3(self._scan_T_raw[-1])
            relative_pose = prev_est.between(pose3_curr)
            noise = (_icp_noise(rmse, self.cfg.icp_noise_scale) if rmse > 0
                     else gtsam.noiseModel.Isotropic.Sigma(6, 0.5))
            self._graph.add(gtsam.BetweenFactorPose3(
                key_prev, key_curr, relative_pose, noise))
            self._initial_estimates.insert(key_curr, pose3_curr)

        # GPS prior
        if gps is not None:
            gps_pt = gtsam.Point3(float(gps[0]), float(gps[1]), float(gps[2]))
            self._graph.add(gtsam.GPSFactor(
                key_curr, gps_pt,
                gtsam.noiseModel.Isotropic.Sigma(3, self.cfg.gps_sigma)))

        self._isam.update(self._graph, self._initial_estimates)
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial_estimates = gtsam.Values()
        self.timings["gtsam"].append(time.perf_counter() - t0)

        self._scan_T_raw.append(T_raw.copy())
        T_est_curr = (T_reg @ T_raw) if (
            np.any(T_reg[:3, :3] != np.eye(3))
            or np.any(T_reg[:3, 3] != 0)) else T_raw
        self._pose_count += 1
        self._frames_accepted += 1

        # ── Submap management / voxel insertion ───────────────────────────
        if (self._current_submap is None
                or len(self._current_submap.frame_indices) >= self.cfg.submap_frames):
            self._current_submap = Submap(
                anchor_index=self._pose_count - 1,
                anchor_T=T_est_curr.copy(),
                resolution=self.cfg.octo_resolution)
            self._submaps.append(self._current_submap)
        self._current_submap.frame_indices.append(self._pose_count - 1)

        t0 = time.perf_counter()
        wf64 = world_pts.astype(np.float64)
        ds_pts = self._downsample_for_insert(wf64)
        self._current_submap.insert(ds_pts)
        self.timings["octo_insert"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._voxelise_and_append(wf64)
        self.timings["vox_track"].append(time.perf_counter() - t0)

        # ── Visualise ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if self._viewer is not None and self.cfg.enable_viewer:
            vp = self._get_vis()
            if not self._viewer._proc:
                self._viewer.start(initial_points=vp)
            else:
                self._viewer.update(vp,
                                    drone_pos=self._drone_pos,
                                    target_pos=self._target_pos,
                                    frontier_points=self._frontier_points,
                                    path_points=self._path_points,
                                    candidate_points=self._candidate_points)
        self.timings["vis"].append(time.perf_counter() - t0)

        self.timings["total"].append(time.perf_counter() - t_frame)

        # ── Console log ───────────────────────────────────────────────────
        total_mem = sum(sm.memory_usage() for sm in self._submaps)
        print(f"  {self._frames_accepted:3d} | raw={len(pts):6,} "
              f"voxels={self._vis_len:8,} submaps={len(self._submaps)} "
              f"mem={total_mem/1e6:.1f}MB | "
              f"tot={self.timings['total'][-1]*1e3:.0f}ms\n", flush=True)

        # ── Build result ──────────────────────────────────────────────────
        result.update(
            accepted=True, fitness=fitness, rmse=rmse,
            dt=dt_mag, dr=dr_mag,
            voxels=self._vis_len,
            pose_count=self._pose_count,
            submaps=len(self._submaps),
        )

        # ── Fire callbacks ────────────────────────────────────────────────
        for cb in self._on_frame_callbacks:
            try:
                cb(self, result)
            except Exception as e:
                print(f"  (callback error: {e})")

        return result

    # ── Map access ────────────────────────────────────────────────────────

    def get_map_points(self) -> np.ndarray:
        """Return current occupied voxel centres (Nx3 float32)."""
        return self._get_vis()

    def get_corrected_map_points(self) -> np.ndarray:
        """Recomposite all submaps using latest GTSAM poses."""
        self._recomposite_from_submaps()
        return self._get_vis()

    def get_optimised_poses(self) -> list[np.ndarray]:
        """Return list of optimised 4x4 poses from iSAM2."""
        values = self._isam.calculateEstimate()
        poses = []
        for i in range(self._pose_count):
            try:
                poses.append(pose3_to_T(values.atPose3(symbol('x', i))))
            except Exception:
                poses.append(self._scan_T_raw[i] if i < len(self._scan_T_raw) else np.eye(4))
        return poses

    # ── OctoMap export ────────────────────────────────────────────────────

    def save_octomap(self, path: str) -> str:
        """Build a final OctoMap from corrected submaps and save as .bt."""
        print("  Building final OctoMap for export...")
        merged = pyoctomap.OcTree(self.cfg.octo_resolution)
        values = self._isam.calculateEstimate()
        for sm in self._submaps:
            try:
                T = pose3_to_T(values.atPose3(symbol('x', sm.anchor_index)))
            except Exception:
                T = sm.anchor_T
            wpts = sm.get_world_points(T)
            if len(wpts) > 0:
                merged.updateNodes(wpts.astype(np.float64), True)
        merged.writeBinary(path)
        print(f"  OctoMap saved to {path} "
              f"({merged.size():,} nodes, {merged.memoryUsage()/1e6:.1f}MB)")
        return path

    # ── Summary / stats ───────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a dictionary of pipeline statistics."""
        ratio = self._vis_len / self._raw_total * 100 if self._raw_total else 0
        total_mem = sum(sm.memory_usage() for sm in self._submaps)
        total_nodes = sum(sm.node_count() for sm in self._submaps)
        plane = fit_plane(self._get_vis()) if self._vis_len > 10 else None
        return {
            "registration": self.cfg.registration,
            "octo_resolution": self.cfg.octo_resolution,
            "submap_count": len(self._submaps),
            "submap_frames": self.cfg.submap_frames,
            "rejected_count": self._outlier_det.rejected_count,
            "raw_total": self._raw_total,
            "voxel_count": self._vis_len,
            "compression_ratio": ratio,
            "map_nodes": total_nodes,
            "map_memory_mb": total_mem / 1e6,
            "pose_count": self._pose_count,
            "plane": plane,
            "timings": {k: list(v) for k, v in self.timings.items()},
        }

    def print_summary(self):
        """Print a formatted summary to stdout."""
        s = self.get_summary()
        ratio = s["compression_ratio"]
        print(f"\n{'='*65}")
        print(f"Final Mapping Summary  (registration={s['registration']})")
        print(f"{'='*65}")
        print(f"  Octo resolution:  {s['octo_resolution']} m")
        print(f"  Submaps:          {s['submap_count']} submaps, "
              f"{s['submap_frames']} frames/submap")
        if self.cfg.reject_outliers:
            print(f"  Outlier rejection: {s['rejected_count']} frames rejected "
                  f"(warmup={self.cfg.reject_warmup}, \u0394t CI={self.cfg.reject_dt_ci*100:.0f}%, "
                  f"\u0394r CI={self.cfg.reject_rot_ci*100:.0f}%)")
        print(f"  Raw pts total:    {s['raw_total']:,}")
        print(f"  Occupied voxels:  {s['voxel_count']:,}  "
              f"({ratio:.1f}% -> {s['raw_total']/max(s['voxel_count'],1):.1f}x)")
        print(f"  Map nodes:        {s['map_nodes']:,}  "
              f"mem={s['map_memory_mb']:.1f}MB  ({s['submap_count']} submaps)")
        print(f"  Pose graph nodes: {s['pose_count']}")
        if s["plane"]:
            n, sx, sy, res = s["plane"]
            print(f"  Plane residual:   {res:.4f} m")
            print(f"  Plane slope:      x={sx:+.4f}  y={sy:+.4f}")

        gt = sum(self.timings["total"])
        print(f"\n  {'step':<20} {'total':>7} {'mean':>7} {'max':>7} "
              f"{'min':>7} {'std':>7} {'%':>5}  {'count':>5}")
        print(f"  {'-'*76}")
        for k in self._tkeys:
            v = self.timings[k]
            if not v:
                continue
            s_sum = sum(v)
            pct = 100 * s_sum / gt if gt else 0
            va = np.array(v)
            print(f"  {k:<20} {s_sum:>7.3f} {va.mean():>7.4f} {va.max():>7.4f} "
                  f"{va.min():>7.4f} {va.std():>7.4f} {pct:>4.1f}%  {len(v):>5d}")
        print(f"  {'='*76}")

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all state and start fresh (keeps config and viewer)."""
        self._isam = gtsam.ISAM2()
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial_estimates = gtsam.Values()
        self._scan_T_raw.clear()
        self._pose_count = 0
        self._submaps.clear()
        self._current_submap = None
        self._seen_keys.clear()
        self._vis_buf = np.zeros((self.cfg.vis_buffer_size, 3), dtype=np.float32)
        self._vis_len = 0
        self._outlier_det = RegistrationOutlierDetector(
            warmup=self.cfg.reject_warmup,
            dt_ci=self.cfg.reject_dt_ci,
            rot_ci=self.cfg.reject_rot_ci)
        self._raw_total = 0
        self._frame_index = 0
        self._frames_accepted = 0
        self.q_frames.clear()
        self.q_fitness.clear()
        self.q_rmse.clear()
        self.q_inlier_rmse.clear()
        self.q_dt.clear()
        self.q_dr.clear()
        self.q_p_dt.clear()
        self.q_p_dr.clear()
        self.q_rejected.clear()
        self.timings = {k: [] for k in self._tkeys}

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _downsample_for_insert(self, pts: np.ndarray) -> np.ndarray:
        if self.cfg.octo_insert_voxel <= 0:
            return pts
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.cfg.octo_insert_voxel)
        return np.asarray(pcd.points)

    def _voxelise_and_append(self, pts: np.ndarray):
        ijk = np.floor(pts * self._inv_res).astype(np.int32)
        unique = set(map(tuple, ijk))
        new = unique - self._seen_keys
        if not new:
            return 0
        self._seen_keys.update(new)
        arr = np.array(list(new), dtype=np.float32)
        centres = arr * self.cfg.octo_resolution + self._half_res
        n = len(centres)
        if self._vis_len + n > len(self._vis_buf):
            ns = max(len(self._vis_buf) * 2, self._vis_len + n)
            nb = np.zeros((ns, 3), dtype=np.float32)
            nb[:self._vis_len] = self._vis_buf[:self._vis_len]
            self._vis_buf = nb
        self._vis_buf[self._vis_len:self._vis_len + n] = centres
        self._vis_len += n
        return n

    def _get_vis(self) -> np.ndarray:
        return self._vis_buf[:self._vis_len]

    def _recomposite_from_submaps(self):
        values = self._isam.calculateEstimate()
        self._seen_keys = set()
        self._vis_buf = np.zeros((self.cfg.vis_buffer_size, 3), dtype=np.float32)
        self._vis_len = 0
        for sm in self._submaps:
            try:
                updated_T = pose3_to_T(
                    values.atPose3(symbol('x', sm.anchor_index)))
            except Exception:
                updated_T = sm.anchor_T
            world_pts = sm.get_world_points(updated_T)
            if len(world_pts) > 0:
                self._voxelise_and_append(world_pts.astype(np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# ReplaySLAM — run on recorded flight data
# ══════════════════════════════════════════════════════════════════════════════

class ReplaySLAM:
    """Feed recorded flight data to ``SLAMPipeline``.

    Usage::

        replay = ReplaySLAM(SLAMConfig(registration="vgicp"))
        replay.run()                          # latest recording
        replay.run("/path/to/flight_dir")     # explicit directory
    """

    def __init__(self, config: SLAMConfig | None = None,
                 pipeline: SLAMPipeline | None = None):
        self.cfg = config or SLAMConfig()
        self.pipeline = pipeline or SLAMPipeline(self.cfg)
        self.recording_dir: str = ""

    def run(self, recording_dir: str = "") -> SLAMPipeline:
        """Replay all frames and return the pipeline (with map, poses, etc.)."""
        self.recording_dir = resolve_recording_dir(recording_dir or self.cfg.recording_dir)
        all_frames = sorted(glob.glob(os.path.join(self.recording_dir, "frame_*.npz")))
        if not all_frames:
            print(f"No frames in {self.recording_dir}")
            return self.pipeline

        frames = all_frames[::self.cfg.frame_skip]
        if self.cfg.max_frames > 0:
            frames = frames[:self.cfg.max_frames]

        print(f"Replaying {len(frames)} frames from {self.recording_dir}")
        print(f"  available={len(all_frames)} skip={self.cfg.frame_skip} "
              f"max={self.cfg.max_frames or 'all'}")
        print(f"  registration={self.cfg.registration}  "
              f"octo_res={self.cfg.octo_resolution}m  mode=deferred")
        print(f"  submaps: enabled  frames_per_submap={self.cfg.submap_frames}")

        # Start viewer
        if self.cfg.enable_viewer:
            self.pipeline.start_viewer()

        # Optionally set up live quality plot
        quality_plot = _QualityPlot(self.cfg.registration)

        for i, path in enumerate(frames):
            t0 = time.perf_counter()
            data = np.load(path)
            pts = data["points"]
            load_time = time.perf_counter() - t0
            self.pipeline.timings["load"].append(load_time)

            pos = data["position"] if "position" in data.files else np.zeros(3)
            ori = (data["orientation"] if "orientation" in data.files
                   else np.array([1, 0, 0, 0], dtype=float))
            lp = data["lidar_position"] if "lidar_position" in data.files else None
            lo = data["lidar_orientation"] if "lidar_orientation" in data.files else None
            gps = data["gps"] if "gps" in data.files else None

            result = self.pipeline.process_frame(
                pts, pos, ori,
                gps=gps,
                lidar_position=lp,
                lidar_orientation=lo,
                frame_label=i,
            )

            quality_plot.update(self.pipeline)

        # ── Post-loop ─────────────────────────────────────────────────────
        self.pipeline.get_corrected_map_points()

        # Save OctoMap
        bt_path = os.path.join(self.recording_dir, "map.bt")
        try:
            self.pipeline.save_octomap(bt_path)
        except Exception as e:
            print(f"  (could not save OctoMap: {e})")

        # Save quality plot
        try:
            png_path = os.path.join(self.recording_dir, "reg_quality.png")
            quality_plot.save(png_path, self.pipeline)
            print(f"  Quality plot saved to {png_path}")
        except Exception as e:
            print(f"  (could not save quality plot: {e})")

        self.pipeline.print_summary()
        return self.pipeline

    def wait_for_exit(self):
        """Block until Ctrl+C, then stop viewer."""
        print("\nClose the Open3D window or Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop_viewer()


# ══════════════════════════════════════════════════════════════════════════════
# LiveSLAM — real-time mapping with AirSim
# ══════════════════════════════════════════════════════════════════════════════

class LiveSLAM:
    """Connect to AirSim and map in real time.

    Usage::

        live = LiveSLAM(SLAMConfig(registration="vgicp"))
        live.run()        # blocking — Ctrl+C to stop and save

    Or drive from your own control loop::

        live = LiveSLAM(config)
        live.connect()
        while flying:
            live.process_once()    # grab one scan and feed to pipeline
        live.finish("/path/to/output")
    """

    def __init__(self, config: SLAMConfig | None = None,
                 pipeline: SLAMPipeline | None = None):
        self.cfg = config or SLAMConfig()
        self.pipeline = pipeline or SLAMPipeline(self.cfg)
        self.client = None
        self._frame_count = 0
        self._min_interval = 1.0 / max(self.cfg.live_max_hz, 0.1)
        self._last_scan_time = 0.0
        self._target_pos = None

        # Optional: save directory for recording frames alongside mapping
        self.save_dir: str | None = None

    def set_target(self, pos):
        """Set the current target waypoint (shown as green sphere in viewer)."""
        self._target_pos = pos
        self.pipeline.set_target_pos(pos)

    def set_frontier_points(self, pts):
        """Set frontier overlay points (Nx3) shown as orange in the viewer."""
        self.pipeline.set_frontier_points(pts)

    def set_path_points(self, pts):
        """Set planned-path overlay (Nx3) shown as cyan line in the viewer."""
        self.pipeline.set_path_points(pts)

    def set_candidate_points(self, pts):
        """Set NBV candidate overlay (Nx3) shown as magenta in the viewer."""
        self.pipeline.set_candidate_points(pts)

    def refresh_overlays(self):
        """Push current overlays to the viewer without a new scan."""
        self.pipeline.refresh_overlays()

    def connect(self):
        """Connect to AirSim and arm the drone."""
        import cosysairsim as airsim
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("Connected!")
        time.sleep(0.5)

    def enable_recording(self, save_dir: str | None = None):
        """Enable saving raw frames to disk alongside live mapping."""
        if save_dir is None:
            base = os.path.join(os.path.dirname(__file__), "flight_recordings")
            save_dir = os.path.join(base, f"flight_{int(time.time())}")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        print(f"  Recording frames to {self.save_dir}")

    def process_once(self) -> dict | None:
        """Grab one LiDAR scan + pose from AirSim and feed to the pipeline.

        Returns None if called too soon (rate-limited) or no data available.
        """
        import cosysairsim as airsim

        now = time.time()
        if now - self._last_scan_time < self._min_interval:
            return None

        if self.client is None:
            raise RuntimeError("Call connect() before process_once()")

        # ── Sample state BEFORE LiDAR ─────────────────────────────────────
        state_before = self.client.getMultirotorState()

        # ── Get sensor data ───────────────────────────────────────────────
        lidar_data = self.client.getLidarData()

        # ── Sample state AFTER LiDAR ──────────────────────────────────────
        state_after = self.client.getMultirotorState()

        if len(lidar_data.point_cloud) < 9:  # at least 3 points
            return None

        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape((-1, 3))

        # Interpolate pose to LiDAR timestamp
        pos, ori = self._interpolate_pose(state_before, state_after, lidar_data)

        # LiDAR mount offset
        lpos = lidar_data.pose.position
        lori = lidar_data.pose.orientation
        lidar_position = np.array([lpos.x_val, lpos.y_val, lpos.z_val])
        lidar_orientation = np.array([lori.w_val, lori.x_val, lori.y_val, lori.z_val])

        # GPS
        gps = None
        try:
            gps_data = self.client.getGpsData()
            gp = gps_data.gnss.geo_point
            gps = np.array([gp.latitude, gp.longitude, gp.altitude])
        except Exception:
            pass

        # ── Optional: save frame to disk ──────────────────────────────────
        if self.save_dir is not None:
            self._save_frame(points, pos, ori, lidar_position, lidar_orientation, gps)

        # ── Feed to pipeline ──────────────────────────────────────────────
        self.pipeline.set_drone_pos(pos)
        result = self.pipeline.process_frame(
            points, pos, ori,
            gps=gps,
            lidar_position=lidar_position,
            lidar_orientation=lidar_orientation,
            frame_label=self._frame_count,
        )

        self._frame_count += 1
        self._last_scan_time = now
        return result

    def run(self, save_dir: str | None = None, takeoff: bool = True):
        """Full blocking loop: connect, optionally takeoff, map until Ctrl+C.

        Parameters
        ----------
        save_dir : str, optional — also record raw frames to disk.
        takeoff : bool — call takeoffAsync() before starting the loop.
        """
        self.connect()

        if save_dir is not None:
            self.enable_recording(save_dir)

        if self.cfg.enable_viewer:
            self.pipeline.start_viewer()

        if takeoff:
            print("Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(1)

        print(f"Live mapping started  (registration={self.cfg.registration}, "
              f"max_hz={self.cfg.live_max_hz})")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                self.process_once()
                time.sleep(0.001)  # yield to OS
        except KeyboardInterrupt:
            print("\nStopping live mapping...")

        self.finish()

    def finish(self, output_dir: str | None = None):
        """Finalise: correct map, save OctoMap, print summary, stop viewer."""
        self.pipeline.get_corrected_map_points()

        out = output_dir or self.save_dir
        if out:
            bt_path = os.path.join(out, "map.bt")
            try:
                self.pipeline.save_octomap(bt_path)
            except Exception as e:
                print(f"  (could not save OctoMap: {e})")

        self.pipeline.print_summary()
        self.pipeline.stop_viewer()

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_pose(state):
        ts = float(state.timestamp)
        p = state.kinematics_estimated.position
        o = state.kinematics_estimated.orientation
        pos  = np.array([p.x_val, p.y_val, p.z_val])
        quat = np.array([o.w_val, o.x_val, o.y_val, o.z_val])  # w,x,y,z
        return ts, pos, quat

    @staticmethod
    def _interpolate_pose(state_before, state_after, lidar_data):
        """Interpolate vehicle pose to the LiDAR timestamp."""
        from scipy.spatial.transform import Slerp

        ts_b, pos_b, ori_b = LiveSLAM._extract_pose(state_before)
        ts_a, pos_a, ori_a = LiveSLAM._extract_pose(state_after)
        lidar_ts = float(lidar_data.time_stamp)

        dt = ts_a - ts_b
        if dt <= 0:
            return pos_b, ori_b

        t = np.clip((lidar_ts - ts_b) / dt, 0.0, 1.0)

        # Linear position
        pos = (1 - t) * pos_b + t * pos_a

        # SLERP orientation (scipy wants x, y, z, w)
        rots = Rotation.from_quat([
            [ori_b[1], ori_b[2], ori_b[3], ori_b[0]],
            [ori_a[1], ori_a[2], ori_a[3], ori_a[0]],
        ])
        slerp = Slerp([0.0, 1.0], rots)
        q_scipy = slerp([t])[0].as_quat()   # x, y, z, w
        quat = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # w, x, y, z

        return pos, quat

    def _save_frame(self, points, pos, ori, lidar_pos, lidar_ori, gps):
        path = os.path.join(self.save_dir, f"frame_{self._frame_count:05d}.npz")
        kw = dict(
            points=points,
            position=pos,
            orientation=ori,
            lidar_position=lidar_pos,
            lidar_orientation=lidar_ori,
        )
        if gps is not None:
            kw["gps"] = gps
        np.savez(path, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# Quality plot helper (used by ReplaySLAM; can also be used standalone)
# ══════════════════════════════════════════════════════════════════════════════

class _QualityPlot:
    """Live matplotlib registration quality dashboard."""

    def __init__(self, method_name: str = ""):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
        except Exception:
            self._ok = False
            return

        self._ok = True
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        self.fig.suptitle(f"Registration Quality  ({method_name})", fontsize=13)
        self.ax_fit, self.ax_rmse, self.ax_dt, self.ax_dr = axes.flat
        self.axes = axes

        self.line_fit,  = self.ax_fit.plot([], [], 'b-', lw=1)
        self.ax_fit.set_ylabel('Fitness'); self.ax_fit.set_xlabel('Frame')
        self.ax_fit.set_title('Fitness (higher = better overlap)')
        self.ax_fit.axhline(0.5, color='g', ls='--', lw=0.7, label='good')
        self.ax_fit.axhline(0.2, color='r', ls='--', lw=0.7, label='poor')
        self.ax_fit.legend(fontsize=8)

        self.line_rmse, = self.ax_rmse.plot([], [], 'r-', lw=1, label='Full RMSE')
        self.line_inlier_rmse, = self.ax_rmse.plot([], [], 'b-', lw=1,
                                                    alpha=0.6, label='Inlier RMSE')
        self.ax_rmse.set_ylabel('RMSE (m)'); self.ax_rmse.set_xlabel('Frame')
        self.ax_rmse.set_title('RMSE (lower = tighter fit)')
        self.ax_rmse.legend(fontsize=8, loc='upper left')

        self.line_dt,   = self.ax_dt.plot([], [], 'm-', lw=1)
        self.ax_dt.set_ylabel('\u0394t (m)'); self.ax_dt.set_xlabel('Frame')
        self.ax_dt.set_title('Translation correction')
        self.ax_dt2 = self.ax_dt.twinx()
        self.line_p_dt, = self.ax_dt2.plot([], [], color='orange', ls='--',
                                            lw=0.8, alpha=0.7, label='p-value')
        self.ax_dt2.set_ylabel('p-value', color='orange')
        self.ax_dt2.tick_params(axis='y', labelcolor='orange')
        self.ax_dt2.set_ylim(-0.05, 1.05)

        self.line_dr,   = self.ax_dr.plot([], [], 'c-', lw=1)
        self.ax_dr.set_ylabel('\u0394r (\u00b0)'); self.ax_dr.set_xlabel('Frame')
        self.ax_dr.set_title('Rotation correction')
        self.ax_dr2 = self.ax_dr.twinx()
        self.line_p_dr, = self.ax_dr2.plot([], [], color='orange', ls='--',
                                            lw=0.8, alpha=0.7, label='p-value')
        self.ax_dr2.set_ylabel('p-value', color='orange')
        self.ax_dr2.tick_params(axis='y', labelcolor='orange')
        self.ax_dr2.set_ylim(-0.05, 1.05)

        self.scat_rej_dt   = self.ax_dt.scatter([], [], c='red', marker='x',
                                                  s=40, zorder=5, linewidths=1.5)
        self.scat_rej_dr   = self.ax_dr.scatter([], [], c='red', marker='x',
                                                  s=40, zorder=5, linewidths=1.5)

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, pipeline: SLAMPipeline):
        if not self._ok:
            return
        try:
            p = pipeline
            self.line_fit.set_data(p.q_frames, p.q_fitness)
            self.line_rmse.set_data(p.q_frames, p.q_rmse)
            self.line_inlier_rmse.set_data(p.q_frames, p.q_inlier_rmse)
            self.line_dt.set_data(p.q_frames, p.q_dt)
            self.line_p_dt.set_data(p.q_frames, p.q_p_dt)
            self.line_dr.set_data(p.q_frames, p.q_dr)
            self.line_p_dr.set_data(p.q_frames, p.q_p_dr)

            if p.q_rejected:
                rej_xy_dt = np.column_stack(
                    ([p.q_frames[j] for j in p.q_rejected],
                     [p.q_dt[j]     for j in p.q_rejected]))
                rej_xy_dr = np.column_stack(
                    ([p.q_frames[j] for j in p.q_rejected],
                     [p.q_dr[j]     for j in p.q_rejected]))
                self.scat_rej_dt.set_offsets(rej_xy_dt)
                self.scat_rej_dr.set_offsets(rej_xy_dr)

            for ax in self.axes.flat:
                ax.relim(); ax.autoscale_view()
            self.ax_dt2.set_ylim(-0.05, 1.05)
            self.ax_dr2.set_ylim(-0.05, 1.05)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def save(self, path: str, pipeline: SLAMPipeline):
        if not self._ok:
            return
        p = pipeline
        # Mark low-fitness frames
        bad_idx = [j for j, f in enumerate(p.q_fitness) if 0 < f < 0.3]
        if bad_idx:
            self.ax_fit.scatter([p.q_frames[j] for j in bad_idx],
                                [p.q_fitness[j] for j in bad_idx],
                                c='red', s=25, zorder=5, label='fit < 0.3')
            self.ax_fit.legend(fontsize=8)

        if p.q_rejected:
            rej_f = [p.q_frames[j] for j in p.q_rejected]
            self.ax_dt.scatter(rej_f, [p.q_dt[j] for j in p.q_rejected],
                               c='red', marker='x', s=40, zorder=5,
                               linewidths=1.5, label='rejected')
            self.ax_dt.legend(fontsize=8, loc='upper left')
            self.ax_dr.scatter(rej_f, [p.q_dr[j] for j in p.q_rejected],
                               c='red', marker='x', s=40, zorder=5,
                               linewidths=1.5, label='rejected')
            self.ax_dr.legend(fontsize=8, loc='upper left')

        for ax in self.axes.flat:
            ax.relim(); ax.autoscale_view()
        self.ax_dt2.set_ylim(-0.05, 1.05)
        self.ax_dr2.set_ylim(-0.05, 1.05)
        self.fig.savefig(path, dpi=150, bbox_inches='tight')


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point (backward-compatible)
# ══════════════════════════════════════════════════════════════════════════════

def run_replay(recording_dir: str = ""):
    """Legacy entry point — wraps ``ReplaySLAM`` for backward compatibility."""
    cfg = SLAMConfig(
        registration="vgicp",
        recording_dir=recording_dir,
    )
    replay = ReplaySLAM(cfg)
    replay.run(recording_dir)
    replay.wait_for_exit()


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_replay(sys.argv[1] if len(sys.argv) > 1 else "")
    print(f"\nWall-clock: {time.perf_counter()-t0:.1f}s")
