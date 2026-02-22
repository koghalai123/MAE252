#!/usr/bin/env python3
"""Final SLAM pipeline — GTSAM pose-graph + OctoMap + swappable registration.

Combines:
  - Frame-to-map registration (default: FPFH+RANSAC, swappable via REGISTRATION)
  - GTSAM iSAM2 pose-graph optimisation with between-factors + GPS priors
  - Loop closure via scan-context place recognition + ICP refinement
  - OctoMap voxel storage + Viewer3D live visualisation

To switch registration algorithm, change the REGISTRATION variable below.
Valid options:  "fpfh_ransac"  "icp"  "gicp"  "ndt"  "state_only"

Bayesian occupancy updates (OCTO_BAYESIAN = True):
    Uses insertPointCloud() with raycasting from the sensor origin.
    Rays trace free space; endpoints increment occupied log-odds.
    Clamping thresholds prevent over-confident beliefs.
    Set OCTO_BAYESIAN = False to revert to the legacy updateNodes() path.

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

# Import registration functions + utilities from RegistrationComparison
from RegistrationComparison import (
    get_register_fn, resolve_recording_dir, filter_valid, xform_pts,
    apply_T, _to_pcd, _local, ScanContext, fit_plane,
    ICP_VOXEL, ICP_CORR, ICP_ITER, ICP_FIT_TOL, ICP_RMSE_TOL,
    NORM_R, NORM_NN, LOCAL_R, MIN_VOXELS,
    SC_RINGS, SC_SECTORS, SC_MAX_RANGE,
)

# ══════════════════════════════════════════════════════════════════════════════
# ★★★  CHANGE THIS TO SWITCH REGISTRATION ALGORITHM  ★★★
# ══════════════════════════════════════════════════════════════════════════════
#   "fpfh_ransac"  — FPFH feature matching + RANSAC global + ICP refine (default)
#   "icp"          — Point-to-plane ICP
#   "gicp"         — Generalized ICP (plane-to-plane covariances)
#   "ndt"          — Normal Distributions Transform
#   "state_only"   — No registration (raw vehicle pose baseline)
# ──────────────────────────────────────────────────────────────────────────────
REGISTRATION = "gicp"
# ══════════════════════════════════════════════════════════════════════════════

# ── Recording ─────────────────────────────────────────────────────────────────
RECORDING_DIR = ""           # empty → latest in flight_recordings/
MAX_FRAMES    = 0            # 0 = all
FRAME_SKIP    = 3            # process every Nth frame

# ── OctoMap ───────────────────────────────────────────────────────────────────
OCTO_RESOLUTION = 0.10       # leaf voxel size (m)
OCTO_BAYESIAN   = False       # True = raycasting (insertPointCloud), False = old updateNodes
OCTO_PROB_HIT   = 0.70       # P(occupied | hit)  — log-odds increment per occupied obs
OCTO_PROB_MISS  = 0.35       # P(occupied | miss) — log-odds decrement per free-space ray
OCTO_CLAMP_MIN  = 0.12       # min occupancy prob  (prevents over-confident free)
OCTO_CLAMP_MAX  = 0.97       # max occupancy prob  (prevents over-confident occupied)
OCTO_MAX_RANGE  = 80.0       # max raycasting range in metres  (-1 = unlimited)
VIS_SYNC_EVERY  = 20         # re-extract occupied leaves for vis every N frames

# ── GTSAM / Loop Closure ─────────────────────────────────────────────────────
GPS_SIGMA           = 3.0    # GPS position noise (m)
ICP_NOISE_SCALE     = 1.0    # multiplier on ICP-derived noise model
LC_SEARCH_EVERY     = 100     # try loop closure every N frames
LC_MIN_INDEX_SEP    = 15     # ignore candidates within this many frames
LC_FITNESS_THRESH   = 0.20   # min ICP fitness for a loop closure
LC_MAX_SPATIAL_DIST = 50.0   # metres — skip candidates farther than this
LC_SCAN_CONTEXT     = True   # use scan-context place recognition
SC_DIST_THRESH      = 0.3    # scan-context similarity threshold
SC_ICP_CORR         = 5.0    # ICP corr dist for loop closure refinement


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
    T[:3, 3] = np.array(p.translation()).reshape(3)
    return T


def T_to_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(T)


def _icp_noise(rmse):
    rot_s = max(0.05, rmse * 0.5) * ICP_NOISE_SCALE
    tra_s = max(0.10, rmse * 1.5) * ICP_NOISE_SCALE
    sigma = float(max(rot_s, tra_s))
    return gtsam.noiseModel.Isotropic.Sigma(6, sigma)


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
    print(f"  registration={REGISTRATION}  octo_res={OCTO_RESOLUTION}m"
          f"  bayesian={OCTO_BAYESIAN}")
    print(f"  lc_every={LC_SEARCH_EVERY}  lc_min_sep={LC_MIN_INDEX_SEP}  "
          f"lc_fit={LC_FITNESS_THRESH}  scan_context={LC_SCAN_CONTEXT}")

    # ── GTSAM iSAM2 ──────────────────────────────────────────────────────
    isam = gtsam.ISAM2()
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

    # Per-scan storage
    scan_pcds  = []      # downsampled pcds in BODY frame for loop-closure ICP
    scan_T_raw = []      # raw 4×4 world poses
    scan_T_est = []      # estimated 4×4 world poses (registration-corrected)
    scan_sensor_origins_body = []   # sensor origin in body frame per scan (for Bayesian rebuild)
    pose_count = 0
    loop_closures = 0

    # ── OctoMap + viewer ──────────────────────────────────────────────────
    tree   = pyoctomap.OcTree(OCTO_RESOLUTION)
    if OCTO_BAYESIAN:
        tree.setProbHit(OCTO_PROB_HIT)
        tree.setProbMiss(OCTO_PROB_MISS)
        tree.setClampingThresMin(OCTO_CLAMP_MIN)
        tree.setClampingThresMax(OCTO_CLAMP_MAX)
        print(f"  Bayesian OctoMap: hit={OCTO_PROB_HIT} miss={OCTO_PROB_MISS} "
              f"clamp=[{OCTO_CLAMP_MIN}, {OCTO_CLAMP_MAX}]  max_range={OCTO_MAX_RANGE}m")
    viewer = Viewer3D()

    inv_res  = 1.0 / OCTO_RESOLUTION
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

    def _sync_vis_from_tree():
        """Re-extract occupied leaf centres from the octree into vis_buf.

        Bayesian updates can mark previously-occupied cells as free, so the
        monotonically-growing seen_keys set drifts out of date.  This function
        replaces vis_buf with the *actual* occupied leaves, keeping the viewer
        consistent with the probabilistic map.
        """
        nonlocal vis_buf, vis_len, seen_keys
        occ_pts = tree.extractPointCloud()          # Nx3 float64 occupied centres
        if occ_pts is None or len(occ_pts) == 0:
            return
        occ_pts = np.asarray(occ_pts, dtype=np.float32)
        n = len(occ_pts)
        if n > len(vis_buf):
            vis_buf = np.zeros((n + VIS_BUF, 3), dtype=np.float32)
        vis_buf[:n] = occ_pts
        vis_len = n
        # Rebuild seen_keys so incremental appends stay consistent
        ijk = np.floor(occ_pts * inv_res).astype(np.int32)
        seen_keys = set(map(tuple, ijk))

    def _rebuild_octree_from_optimized():
        nonlocal tree, seen_keys, vis_buf, vis_len
        values = isam.calculateEstimate()
        tree = pyoctomap.OcTree(OCTO_RESOLUTION)
        if OCTO_BAYESIAN:
            tree.setProbHit(OCTO_PROB_HIT)
            tree.setProbMiss(OCTO_PROB_MISS)
            tree.setClampingThresMin(OCTO_CLAMP_MIN)
            tree.setClampingThresMax(OCTO_CLAMP_MAX)
        seen_keys = set()
        vis_buf = np.zeros((VIS_BUF, 3), dtype=np.float32)
        vis_len = 0
        for j in range(pose_count):
            try:
                opt_pose = values.atPose3(symbol('x', j))
            except Exception:
                continue
            T_opt = pose3_to_T(opt_pose)
            body_pts = np.asarray(scan_pcds[j].points)   # body frame
            world_pts = apply_T(body_pts, T_opt).astype(np.float64)
            if OCTO_BAYESIAN:
                origin_body = scan_sensor_origins_body[j]
                origin_world = apply_T(origin_body.reshape(1, 3), T_opt).astype(np.float64)[0]
                tree.insertPointCloud(world_pts, origin_world, OCTO_MAX_RANGE)
            else:
                tree.updateNodes(world_pts, True)
            _voxelise_and_append(world_pts)
        # After full rebuild, sync vis from actual occupied leaves
        if OCTO_BAYESIAN:
            _sync_vis_from_tree()

    # ── Scan-context for place recognition ────────────────────────────────
    sc = ScanContext() if LC_SCAN_CONTEXT else None
    sc_descs = []

    # ── Timing ────────────────────────────────────────────────────────────
    tkeys = ["load", "transform", "register", "gtsam", "octo_insert",
             "vox_track", "vis", "loop_closure", "total"]
    timings = {k: [] for k in tkeys}
    raw_total = 0

    # ── Live quality plot ─────────────────────────────────────────────────
    q_frames   = []   # frame indices
    q_fitness  = []
    q_rmse     = []
    q_dt       = []   # translation correction magnitude (m)
    q_dr       = []   # rotation correction magnitude (deg)

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

    line_dt,   = ax_dt.plot([], [], 'm-', lw=1)
    ax_dt.set_ylabel('Δt (m)'); ax_dt.set_xlabel('Frame')
    ax_dt.set_title('Translation correction')

    line_dr,   = ax_dr.plot([], [], 'c-', lw=1)
    ax_dr.set_ylabel('Δr (°)'); ax_dr.set_xlabel('Frame')
    ax_dr.set_title('Rotation correction')

    for ax in axes_q.flat:
        ax.grid(True, alpha=0.3)
    fig_q.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    PLOT_UPDATE_EVERY = 1   # update plot every N frames

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
        # Compute world-frame sensor origin for Bayesian raycasting
        sensor_origin_body = np.asarray(lp, dtype=np.float64)
        sensor_origin_world = xform_pts(
            sensor_origin_body.reshape(1, 3), pos, ori)[0]
        timings["transform"].append(time.perf_counter() - t0)
        raw_total += len(world_init)

        # Raw world-pose 4×4
        T_raw = np.eye(4)
        R_raw = Rotation.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_matrix()
        T_raw[:3, :3] = R_raw
        T_raw[:3, 3] = pos

        # ── Registration against voxel map ────────────────────────────────
        t0 = time.perf_counter()
        if vis_len >= MIN_VOXELS and pose_count > 0 and REGISTRATION != "state_only":
            target_pts = _get_vis()
            T_reg, fitness, rmse, _ = register_fn(
                world_init.astype(np.float64),
                target_pts.astype(np.float64))
            world_pts = apply_T(world_init, T_reg).astype(np.float32)

            ct = T_reg[:3, 3]
            ce = Rotation.from_matrix(T_reg[:3, :3]).as_euler("xyz", degrees=True)
            dt_mag = float(np.linalg.norm(ct))
            dr_mag = float(np.linalg.norm(ce))
            print(f"  REG {i:03d}: fit={fitness:.4f} rmse={rmse:.4f} "
                  f"Δt={dt_mag:.4f}m "
                  f"Δr=({ce[0]:+.2f},{ce[1]:+.2f},{ce[2]:+.2f})°")
        else:
            world_pts = world_init.astype(np.float32)
            T_reg = np.eye(4)
            fitness = 0.0
            rmse = 0.0
            dt_mag = 0.0
            dr_mag = 0.0
            print(f"  frame {i:03d}: {'baseline' if REGISTRATION == 'state_only' else f'too few voxels ({vis_len})'}, state pose only")
        timings["register"].append(time.perf_counter() - t0)

        # Record quality metrics
        q_frames.append(i)
        q_fitness.append(fitness)
        q_rmse.append(rmse)
        q_dt.append(dt_mag)
        q_dr.append(dr_mag)

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

        # Store body-frame scan for loop closure & octree rebuild
        pcd_body = _to_pcd(body)
        if ICP_VOXEL > 0:
            pcd_body = pcd_body.voxel_down_sample(ICP_VOXEL)
        pcd_body.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(NORM_R, NORM_NN))
        scan_pcds.append(pcd_body)
        scan_T_raw.append(T_raw.copy())
        scan_sensor_origins_body.append(sensor_origin_body.copy())
        # Estimated world pose (registration-corrected if available, else raw)
        T_est_curr = (T_reg @ T_raw) if (
            np.any(T_reg[:3, :3] != np.eye(3))
            or np.any(T_reg[:3, 3] != 0)) else T_raw
        scan_T_est.append(T_est_curr.copy())
        pose_count += 1

        # ── Loop closure ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        did_loop_close = False

        if pose_count > LC_MIN_INDEX_SEP and (pose_count - 1) % LC_SEARCH_EVERY == 0:
            # Scan-context filtering (if enabled)
            sc_candidates = None
            if sc is not None:
                desc = sc.descriptor(world_pts, center=pos)
                sc_descs.append(desc)
                sc_candidates = []
                for j in range(max(0, len(sc_descs) - LC_MIN_INDEX_SEP)):
                    if abs(pose_count - 1 - j) < LC_MIN_INDEX_SEP:
                        continue
                    d = sc.distance(desc, sc_descs[j])
                    if d < SC_DIST_THRESH:
                        sc_candidates.append(j)
            else:
                if len(sc_descs) == 0:
                    # still need to track descriptors count for indexing
                    pass

            # Build candidate list
            if sc_candidates is not None:
                candidates = sc_candidates
            else:
                candidates = list(range(max(0, pose_count - LC_MIN_INDEX_SEP - 1)))

            print(f"  LC search: {len(candidates)} candidates "
                  f"({'scan-context' if sc_candidates is not None else 'temporal'})")
            for j in candidates:
                if abs(pose_count - 1 - j) < LC_MIN_INDEX_SEP:
                    continue
                # Spatial distance check using optimised poses
                try:
                    values_lc = isam.calculateEstimate()
                    T_j_opt = pose3_to_T(
                        values_lc.atPose3(symbol('x', j)))
                    T_curr_opt = pose3_to_T(
                        values_lc.atPose3(symbol('x', pose_count - 1)))
                    dist = np.linalg.norm(
                        T_j_opt[:3, 3] - T_curr_opt[:3, 3])
                    if dist > LC_MAX_SPATIAL_DIST:
                        continue
                except Exception:
                    continue

                # ICP on body-frame clouds with relative-pose init guess
                curr_body = scan_pcds[pose_count - 1]  # body frame + normals
                cand_body = scan_pcds[j]
                init_guess = np.linalg.inv(T_j_opt) @ T_curr_opt
                lc_res = o3d.pipelines.registration.registration_icp(
                    curr_body, cand_body, SC_ICP_CORR, init_guess,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=ICP_FIT_TOL,
                        relative_rmse=ICP_RMSE_TOL,
                        max_iteration=80))

                if (lc_res.fitness > LC_FITNESS_THRESH
                        and lc_res.inlier_rmse < 1.0):
                    # result.transformation is the refined relative pose j→curr
                    rel_pose_lc = T_to_pose3(lc_res.transformation)
                    noise_lc = _icp_noise(lc_res.inlier_rmse)
                    lc_graph = gtsam.NonlinearFactorGraph()
                    lc_graph.add(gtsam.BetweenFactorPose3(
                        symbol('x', j), key_curr, rel_pose_lc, noise_lc))
                    isam.update(lc_graph, gtsam.Values())
                    loop_closures += 1
                    did_loop_close = True
                    print(f"  >> LOOP CLOSURE x{j}<->x{pose_count-1}  "
                          f"fit={lc_res.fitness:.3f} rmse={lc_res.inlier_rmse:.3f}")
                    break
        else:
            # Still record scan-context descriptor even when not searching
            if sc is not None:
                desc = sc.descriptor(world_pts, center=pos)
                sc_descs.append(desc)

        timings["loop_closure"].append(time.perf_counter() - t0)

        # Rebuild octree after loop closure
        if did_loop_close:
            t0 = time.perf_counter()
            _rebuild_octree_from_optimized()
            print(f"  octree rebuilt ({vis_len:,} voxels, "
                  f"{time.perf_counter()-t0:.2f}s)")
        else:
            # ── Insert into octree (Bayesian raycasting or legacy) ─────
            t0 = time.perf_counter()
            wf64 = world_pts.astype(np.float64)
            if OCTO_BAYESIAN:
                # Correct sensor origin if registration was applied
                if np.any(T_reg[:3, :3] != np.eye(3)) or np.any(T_reg[:3, 3] != 0):
                    origin_ins = apply_T(
                        sensor_origin_world.reshape(1, 3), T_reg
                    ).astype(np.float64)[0]
                else:
                    origin_ins = sensor_origin_world.astype(np.float64)
                tree.insertPointCloud(wf64, origin_ins, OCTO_MAX_RANGE)
            else:
                tree.updateNodes(wf64, True)
            timings["octo_insert"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            new_vox = _voxelise_and_append(wf64)
            timings["vox_track"].append(time.perf_counter() - t0)

            # Periodically sync vis_buf from actual occupied tree leaves
            if OCTO_BAYESIAN and pose_count % VIS_SYNC_EVERY == 0:
                t0_sync = time.perf_counter()
                _sync_vis_from_tree()
                print(f"  vis sync: {vis_len:,} occupied voxels "
                      f"({(time.perf_counter()-t0_sync)*1e3:.1f}ms)")

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
            try:
                line_fit.set_data(q_frames, q_fitness)
                line_rmse.set_data(q_frames, q_rmse)
                line_dt.set_data(q_frames, q_dt)
                line_dr.set_data(q_frames, q_dr)
                for ax in axes_q.flat:
                    ax.relim(); ax.autoscale_view()
                fig_q.canvas.draw_idle()
                fig_q.canvas.flush_events()
            except Exception:
                pass   # window closed — keep running

        # ── Per-frame log ─────────────────────────────────────────────────
        new_v = timings["vox_track"][-1] * 1e3 if timings["vox_track"] else 0
        print(f"  {i+1:3d}/{len(frames)} | raw={len(pts):6,} "
              f"voxels={vis_len:8,} mem={tree.memoryUsage()/1e6:.1f}MB "
              f"lc={loop_closures} | "
              f"tot={timings['total'][-1]*1e3:.0f}ms\n", flush=True)

    # ── Save final quality plot ────────────────────────────────────────────
    try:
        # Highlight bad frames on fitness plot
        bad_idx = [j for j, f in enumerate(q_fitness) if 0 < f < 0.3]
        if bad_idx:
            ax_fit.scatter([q_frames[j] for j in bad_idx],
                           [q_fitness[j] for j in bad_idx],
                           c='red', s=25, zorder=5, label='fit < 0.3')
            ax_fit.legend(fontsize=8)
        for ax in axes_q.flat:
            ax.relim(); ax.autoscale_view()
        png_path = os.path.join(recording_dir, "reg_quality.png")
        fig_q.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"  Quality plot saved to {png_path}")
    except Exception as e:
        print(f"  (could not save quality plot: {e})")

    # Save the OctoMap binary
    try:
        bt_path = os.path.join(recording_dir, "map.bt")
        tree.writeBinary(bt_path)
        print(f"  OctoMap saved to {bt_path}")
    except Exception as e:
        print(f"  (could not save OctoMap: {e})")

    # ── Final ─────────────────────────────────────────────────────────────
    plane = fit_plane(_get_vis())
    ratio = vis_len / raw_total * 100 if raw_total else 0

    print(f"\n{'='*65}")
    print(f"Final Mapping Summary  (registration={REGISTRATION})")
    print(f"{'='*65}")
    # Final vis sync so summary stats reflect actual occupancy
    if OCTO_BAYESIAN:
        _sync_vis_from_tree()

    print(f"  Octo resolution:  {OCTO_RESOLUTION} m")
    print(f"  Bayesian update:  {OCTO_BAYESIAN}")
    if OCTO_BAYESIAN:
        print(f"    P(hit)={OCTO_PROB_HIT}  P(miss)={OCTO_PROB_MISS}  "
              f"clamp=[{OCTO_CLAMP_MIN}, {OCTO_CLAMP_MAX}]  "
              f"max_range={OCTO_MAX_RANGE}m")
    print(f"  Raw pts total:    {raw_total:,}")
    print(f"  Occupied voxels:  {vis_len:,}  "
          f"({ratio:.1f}% -> {raw_total/max(vis_len,1):.1f}x)")
    print(f"  Tree nodes:       {tree.size():,}  "
          f"mem={tree.memoryUsage()/1e6:.1f}MB")
    print(f"  Free leaves:      {tree.size() - vis_len:,}")
    print(f"  Pose graph nodes: {pose_count}")
    print(f"  Loop closures:    {loop_closures}")
    if plane:
        n, sx, sy, res = plane
        print(f"  Plane residual:   {res:.4f} m")
        print(f"  Plane slope:      x={sx:+.4f}  y={sy:+.4f}")

    gt = sum(timings["total"])
    print(f"\n  {'step':<14} {'total':>7} {'mean':>7} {'max':>7} {'%':>5}")
    print(f"  {'-'*44}")
    for k in tkeys:
        v = timings[k]
        if not v:
            continue
        s = sum(v)
        pct = 100 * s / gt if gt else 0
        print(f"  {k:<14} {s:>7.3f} {s/len(v):>7.4f} {max(v):>7.4f} "
              f"{pct:>4.1f}%")
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
