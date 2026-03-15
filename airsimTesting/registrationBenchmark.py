#!/usr/bin/env python3
"""Benchmark every registration method on the N most recent flight recordings.

Imports ``ReplayRunner`` from ``exploration.py`` and runs it once per
(recording × registration method × noise level), saving each resulting
map for later comparison with ``mapAnalysis.py``.

Usage
-----
    python registrationBenchmark.py              # N_RECORDINGS most recent
    python registrationBenchmark.py 5            # last 5 recordings
    python registrationBenchmark.py path/to/dir  # single explicit directory

Configuration
-------------
Edit the CONFIGURATION section below to set the recording directory,
exploration bounds, registration methods to run, etc.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from exploration import ReplayRunner, EXPLORE_BOUNDS, PLANNER_RES, FRAME_SKIP
from RegistrationComparison import REGISTRATION_METHODS, resolve_recording_dir


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Recording base directory (contains exploration_* / flight_* sub-folders).
RECORDING_DIR = "flight_recordings"

# How many of the most-recent recordings to benchmark (0 = all available).
# Ignored when a single explicit directory is given on the command line.
N_RECORDINGS = 5

# Voxel / OctoMap resolution (same as exploration.py defaults).
OCTO_RESOLUTION = 0.15

# Registration methods to benchmark.
# Set to None to run ALL available methods automatically.
# Otherwise provide a list of method keywords, e.g.:
#METHODS = ["state_only", "icp", "gicp", "ndt", "fpfh_ransac", "small_gicp", "vgicp", "kiss_icp"]
METHODS = ["vgicp"]


#METHODS = ["state_only","fpfh_ransac"]


# Show the Open3D viewer while processing each method.
ENABLE_VIEWER = True

# Run the exploration planner during replay to generate candidate
# waypoint selection logs (candidate_sources.csv).
ENABLE_PLANNER = True

# Save quality plot for each method alongside the map.
SAVE_QUALITY_PLOT = True

# ── Pose noise injection ──────────────────────────────────────────────────────
# Add normally-distributed noise to the drone's reported position / orientation
# before it enters the SLAM pipeline.  Set both to 0.0 to disable.
#
# Each entry is (position_std_m, orientation_std_deg).
# Every method will be run once *clean* plus once per noise level listed here.
# All noisy runs share the same random seed so the noise sequence is identical
# across registration methods (and across noise levels with the same index).
'''POSE_NOISE_LEVELS: list[tuple[float, float]] = [
    (0.0125, 0.05),  # 1.25 cm, 0.05°
    (0.125, 0.5),  
    (0.25, 1.0),     # 2× base
    (0.5, 2.0),      # 4× base
    (1.0, 4.0),      # 8× base
]'''
POSE_NOISE_LEVELS: list[tuple[float, float]] = [
]
# Random seed for reproducible noise (None → non-deterministic).
POSE_NOISE_SEED: int | None = 42
# Also run each method with zero noise for a clean baseline.
RUN_CLEAN_BASELINE = True

# Output directory for saved maps. Each method gets its own sub-folder.
SAVED_MAPS_DIR = os.path.join(_SCRIPT_DIR, "savedMaps")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_n_recordings(base_dir: str, n: int) -> list[str]:
    """Return the *n* most-recent recording directories under *base_dir*.

    Each returned path is a directory that contains ``frame_*.npz`` files.
    Sorted newest-first.  If *n* ≤ 0, return all found directories.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Recording base directory not found: {base_dir}")

    # If the directory itself contains frames, treat it as a single recording
    if glob.glob(os.path.join(base_dir, "frame_*.npz")):
        return [base_dir]

    subdirs = sorted(
        glob.glob(os.path.join(base_dir, "flight_*"))
        + glob.glob(os.path.join(base_dir, "exploration_*")))

    # Filter to those that actually contain frame files
    valid = [d for d in subdirs
             if os.path.isdir(d) and glob.glob(os.path.join(d, "frame_*.npz"))]
    if not valid:
        raise FileNotFoundError(
            f"No recording directories with frame_*.npz under {base_dir}")

    # newest first (lexicographic sort puts newest timestamp last)
    valid = list(reversed(valid))
    if n > 0:
        valid = valid[:n]
    return valid


def _timestamp_from_dir(rec_dir: str) -> int:
    """Extract the numeric timestamp from a recording folder name."""
    m = re.search(r'(\d{10,})', os.path.basename(rec_dir))
    return int(m.group(1)) if m else int(time.time())


def _deduplicate_methods(methods: list[str]) -> list[str]:
    """Remove method aliases that map to the same underlying function."""
    seen_fns: dict[int, str] = {}
    deduped: list[str] = []
    for m in methods:
        fn = REGISTRATION_METHODS[m]
        if id(fn) not in seen_fns:
            seen_fns[id(fn)] = m
            deduped.append(m)
    return deduped


def _save_timing_csv(pipeline, method: str, out_dir: str,
                     wall_time: float) -> None:
    """Save a CSV with per-step timing breakdown next to the map.

    Rows: one per timing category (transform, register, gtsam, …).
    Columns: total_s, mean_s, max_s, min_s, std_s, pct, count.
    An extra "wall_clock" row records the overall elapsed time.
    """
    try:
        summary = pipeline.get_summary()
        timings = summary.get("timings", {})
        grand_total = sum(timings.get("total", [0.0]))

        csv_path = os.path.join(out_dir, "timing_breakdown.csv")
        with open(csv_path, "w") as f:
            f.write("step,total_s,mean_s,max_s,min_s,std_s,pct,count\n")
            for key in ["load", "transform", "register", "gtsam",
                        "octo_insert", "vox_track", "vis", "total"]:
                vals = timings.get(key, [])
                if not vals:
                    continue
                arr = np.array(vals)
                s_sum = float(arr.sum())
                pct = 100.0 * s_sum / grand_total if grand_total > 0 else 0.0
                f.write(f"{key},{s_sum:.6f},{arr.mean():.6f},{arr.max():.6f},"
                        f"{arr.min():.6f},{arr.std():.6f},{pct:.2f},{len(vals)}\n")
            # Wall-clock row
            f.write(f"wall_clock,{wall_time:.6f},,,,,,\n")
        print(f"  Timing CSV saved: {csv_path}")
    except Exception as e:
        print(f"  (could not save timing CSV: {e})")


def _save_quality_plot(pipeline, method: str, out_dir: str):
    """Save a 2×2 registration-quality figure next to the map."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle(f"Registration Quality  ({method})", fontsize=13)
        ax_fit, ax_rmse, ax_dt, ax_dr = axes.flat

        ax_fit.plot(pipeline.q_frames, pipeline.q_fitness, "b-", lw=1)
        ax_fit.set_ylabel("Fitness"); ax_fit.set_xlabel("Frame")
        ax_fit.set_title("Fitness (higher = better overlap)")
        ax_fit.axhline(0.5, color="g", ls="--", lw=0.7)
        ax_fit.axhline(0.2, color="r", ls="--", lw=0.7)
        ax_fit.grid(True, alpha=0.3)

        ax_rmse.plot(pipeline.q_frames, pipeline.q_rmse, "r-", lw=1, label="Full RMSE")
        ax_rmse.plot(pipeline.q_frames, pipeline.q_inlier_rmse, "b-", lw=1,
                     alpha=0.6, label="Inlier RMSE")
        ax_rmse.set_ylabel("RMSE (m)"); ax_rmse.set_xlabel("Frame")
        ax_rmse.set_title("RMSE")
        ax_rmse.legend(fontsize=8)
        ax_rmse.grid(True, alpha=0.3)

        ax_dt.plot(pipeline.q_frames, pipeline.q_dt, "g-", lw=1)
        ax_dt.set_ylabel("Δt (m)"); ax_dt.set_xlabel("Frame")
        ax_dt.set_title("Translation Correction")
        ax_dt.grid(True, alpha=0.3)

        ax_dr.plot(pipeline.q_frames, pipeline.q_dr, "m-", lw=1)
        ax_dr.set_ylabel("Δr (°)"); ax_dr.set_xlabel("Frame")
        ax_dr.set_title("Rotation Correction")
        ax_dr.grid(True, alpha=0.3)

        for ax in axes.flat:
            for ri in pipeline.q_rejected:
                if ri < len(pipeline.q_frames):
                    ax.axvline(pipeline.q_frames[ri], color="red", alpha=0.3, lw=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        qp_path = os.path.join(out_dir, "quality.png")
        fig.savefig(qp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Quality plot saved: {qp_path}")
    except Exception as e:
        print(f"  (could not save quality plot: {e})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Resolve recording(s) ──────────────────────────────────────────
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # If the CLI arg is a plain integer, treat it as "last N recordings".
    if cli_arg is not None and cli_arg.isdigit():
        n_requested = int(cli_arg)
        base = RECORDING_DIR
        if base and not os.path.isabs(base):
            base = os.path.join(_SCRIPT_DIR, base)
        recordings = _resolve_n_recordings(base, n_requested)
    elif cli_arg is not None:
        # Explicit path to one directory
        p = cli_arg if os.path.isabs(cli_arg) else os.path.join(_SCRIPT_DIR, cli_arg)
        recordings = [resolve_recording_dir(p)]
    else:
        # Default: use N_RECORDINGS from config
        base = RECORDING_DIR
        if base and not os.path.isabs(base):
            base = os.path.join(_SCRIPT_DIR, base)
        recordings = _resolve_n_recordings(base, N_RECORDINGS)

    # ── Determine methods to run ──────────────────────────────────────
    methods = (METHODS if METHODS is not None
               else _deduplicate_methods(list(REGISTRATION_METHODS.keys())))

    # ── Build per-recording noise runs ────────────────────────────────
    noise_runs: list[tuple[str, str, float, float]] = []
    for method in methods:
        if RUN_CLEAN_BASELINE:
            noise_runs.append((method, "clean", 0.0, 0.0))
        for pos_std, rot_std in POSE_NOISE_LEVELS:
            tag = f"noise_{pos_std*100:.3f}cm_{rot_std:.3f}deg"
            noise_runs.append((method, tag, pos_std, rot_std))

    total_runs = len(recordings) * len(noise_runs)

    print(f"{'='*70}")
    print(f"  Registration Benchmark")
    print(f"{'='*70}")
    print(f"  Recordings: {len(recordings)}")
    for i, rec in enumerate(recordings):
        print(f"       [{i+1}]  {os.path.basename(rec)}")
    print(f"  Bounds:     {EXPLORE_BOUNDS}")
    print(f"  Resolution: {OCTO_RESOLUTION} m")
    print(f"  Methods:    {', '.join(methods)}")
    if POSE_NOISE_LEVELS:
        print(f"  Noise lvls: {len(POSE_NOISE_LEVELS)}")
        for j, (ps, rs) in enumerate(POSE_NOISE_LEVELS):
            print(f"       [{j}]  pos σ={ps*100:.2f} cm,  rot σ={rs:.3f}°")
        print(f"  Seed:       {POSE_NOISE_SEED}")
        if RUN_CLEAN_BASELINE:
            print(f"  Mode:       clean baseline + {len(POSE_NOISE_LEVELS)} noise level(s)")
    else:
        print(f"  Pose noise: disabled (clean only)")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*70}\n")

    # ── Run each recording × method × noise ───────────────────────────
    results_summary: list[dict] = []
    run_counter = 0

    for rec_idx, resolved_rec in enumerate(recordings):
        rec_name = os.path.basename(resolved_rec)
        timestamp = _timestamp_from_dir(resolved_rec)

        print(f"\n{'▓'*70}")
        print(f"  Recording [{rec_idx+1}/{len(recordings)}]: {rec_name}")
        print(f"{'▓'*70}")

        for method, noise_tag, noise_pos, noise_rot in noise_runs:
            run_counter += 1
            print(f"\n{'━'*70}")
            print(f"  [{run_counter}/{total_runs}]  {method}  [{noise_tag}]  on {rec_name}")
            print(f"{'━'*70}")

            runner = ReplayRunner(
                resolved_rec,
                registration=method,
                octo_resolution=OCTO_RESOLUTION,
                bounds=EXPLORE_BOUNDS,
                planner_res=PLANNER_RES,
                frame_skip=FRAME_SKIP,
                enable_viewer=ENABLE_VIEWER,
                enable_planner=ENABLE_PLANNER,
                pose_noise_pos_std=noise_pos,
                pose_noise_rot_std_deg=noise_rot,
                pose_noise_seed=POSE_NOISE_SEED,
            )

            t0 = time.perf_counter()
            try:
                pipeline = runner.run()
            except Exception as e:
                print(f"  ERROR running {method} [{noise_tag}] on {rec_name}: {e}")
                results_summary.append({
                    "recording": rec_name, "timestamp": timestamp,
                    "method": method, "noise_tag": noise_tag,
                    "noise_pos": noise_pos, "noise_rot": noise_rot,
                    "status": "FAILED", "error": str(e)})
                continue
            elapsed = time.perf_counter() - t0

            pipeline.print_summary()

            # ── Save the map ──────────────────────────────────────────
            suffix = f"{method}_{noise_tag}"
            map_out_dir = os.path.join(SAVED_MAPS_DIR,
                                       f"slam_map_{timestamp}_{suffix}")
            npz_path = runner.save_map(
                out_dir=map_out_dir,
                source=f"benchmark_{suffix}",
                extra_metadata={
                    "recording": rec_name,
                    "registration_method": method,
                    "noise_tag": noise_tag,
                    "pose_noise_pos_std": noise_pos,
                    "pose_noise_rot_std_deg": noise_rot,
                },
            )

            # ── Quality plot ──────────────────────────────────────────
            if SAVE_QUALITY_PLOT:
                _save_quality_plot(pipeline, f"{method} [{noise_tag}]", map_out_dir)

            # ── Timing breakdown CSV ──────────────────────────────────
            _save_timing_csv(pipeline, method, map_out_dir, elapsed)

            # ── Candidate source CSV ──────────────────────────────────
            if ENABLE_PLANNER:
                runner.save_candidate_log(map_out_dir)

            # ── Collect summary ───────────────────────────────────────
            summary = pipeline.get_summary()
            results_summary.append({
                "recording":      rec_name,
                "timestamp":      timestamp,
                "method":         method,
                "noise_tag":      noise_tag,
                "noise_pos":      noise_pos,
                "noise_rot":      noise_rot,
                "status":         "OK",
                "voxel_count":    summary["voxel_count"],
                "pose_count":     summary["pose_count"],
                "submap_count":   summary["submap_count"],
                "rejected_count": summary["rejected_count"],
                "plane_residual": (summary["plane"][3]
                                   if summary["plane"] else float("inf")),
                "total_time":     elapsed,
                "map_path":       npz_path,
            })

            runner.stop_viewer()

    # ── Summary table ─────────────────────────────────────────────────
    ok = [r for r in results_summary if r["status"] == "OK"]
    failed = [r for r in results_summary if r["status"] != "OK"]

    print(f"\n\n{'='*110}")
    print(f"  REGISTRATION BENCHMARK RESULTS")
    print(f"  {len(ok)}/{len(results_summary)} runs succeeded  "
          f"({len(recordings)} recording(s) × {len(noise_runs)} run(s) each)")
    print(f"{'='*110}")

    if ok:
        hdr = (f"  {'Recording':<28} {'Method':<16} {'Noise':<28} "
               f"{'Voxels':>10} {'Poses':>7} {'Submaps':>8} "
               f"{'Rejected':>9} {'Plane Res':>10} {'Time(s)':>9}")
        print(hdr)
        print(f"  {'─'*138}")
        for r in ok:
            print(f"  {r['recording']:<28} {r['method']:<16} "
                  f"{r['noise_tag']:<28} {r['voxel_count']:>10,} "
                  f"{r['pose_count']:>7} {r['submap_count']:>8} "
                  f"{r['rejected_count']:>9} {r['plane_residual']:>10.4f} "
                  f"{r['total_time']:>9.1f}")
    if failed:
        print(f"\n  Failed runs:")
        for r in failed:
            print(f"    {r.get('recording','?')} / {r['method']}: "
                  f"{r.get('error', 'unknown')}")

    print(f"{'='*110}")

    # ── Save CSV summary ──────────────────────────────────────────────
    bench_ts = int(time.time())
    csv_path = os.path.join(SAVED_MAPS_DIR, f"benchmark_{bench_ts}.csv")
    with open(csv_path, "w") as f:
        f.write("recording,timestamp,method,noise_tag,noise_pos_std_m,"
                "noise_rot_std_deg,status,voxel_count,pose_count,"
                "submap_count,rejected_count,plane_residual,"
                "total_time,map_path\n")
        for r in results_summary:
            f.write(f"{r.get('recording','')},{r.get('timestamp','')},"
                    f"{r['method']},{r.get('noise_tag','clean')},"
                    f"{r.get('noise_pos','')},{r.get('noise_rot','')},"
                    f"{r['status']},"
                    f"{r.get('voxel_count','')},{r.get('pose_count','')},"
                    f"{r.get('submap_count','')},{r.get('rejected_count','')},"
                    f"{r.get('plane_residual','')},{r.get('total_time','')},"
                    f"{r.get('map_path','')}\n")
    print(f"\n  CSV summary saved to {csv_path}")
    print(f"  All maps saved under {SAVED_MAPS_DIR}/")
    print(f"  Run mapAnalysis.py to compare against ground truth.")
    print("Done.")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nTotal wall-clock: {time.perf_counter() - t0:.1f}s")
