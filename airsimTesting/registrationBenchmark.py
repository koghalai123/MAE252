#!/usr/bin/env python3
"""Benchmark every registration method on one flight recording.

Imports ``ReplayRunner`` from ``exploration.py`` and runs it once per
registration method, saving each resulting map for later comparison
with ``mapAnalysis.py``.

Usage
-----
    python registrationBenchmark.py                          # latest recording
    python registrationBenchmark.py  flight_recordings/exploration_1772938325

Configuration
-------------
Edit the CONFIGURATION section below to set the recording directory,
exploration bounds, registration methods to run, etc.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from exploration import ReplayRunner, EXPLORE_BOUNDS, PLANNER_RES, FRAME_SKIP
from RegistrationComparison import REGISTRATION_METHODS


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Recording directory (empty string → auto-detect latest exploration_* or flight_*).
RECORDING_DIR = "flight_recordings"

# Voxel / OctoMap resolution (same as exploration.py defaults).
OCTO_RESOLUTION = 0.15

# Registration methods to benchmark.
# Set to None to run ALL available methods automatically.
# Otherwise provide a list of method keywords, e.g.:
#   ["state_only", "icp", "gicp", "ndt", "fpfh_ransac", "small_gicp", "vgicp", "kiss_icp"]
METHODS = None

# Show the Open3D viewer while processing each method.
ENABLE_VIEWER = True

# Save quality plot for each method alongside the map.
SAVE_QUALITY_PLOT = True

# Output directory for saved maps. Each method gets its own sub-folder.
SAVED_MAPS_DIR = os.path.join(_SCRIPT_DIR, "savedMaps")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

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
    # ── Resolve recording directory ───────────────────────────────────
    rec_arg = sys.argv[1] if len(sys.argv) > 1 else RECORDING_DIR
    if rec_arg and not os.path.isabs(rec_arg):
        rec_arg = os.path.join(_SCRIPT_DIR, rec_arg)

    # ── Determine methods to run ──────────────────────────────────────
    methods = (METHODS if METHODS is not None
               else _deduplicate_methods(list(REGISTRATION_METHODS.keys())))

    print(f"{'='*70}")
    print(f"  Registration Benchmark")
    print(f"{'='*70}")
    print(f"  Recording:  {rec_arg}")
    print(f"  Bounds:     {EXPLORE_BOUNDS}")
    print(f"  Resolution: {OCTO_RESOLUTION} m")
    print(f"  Methods:    {', '.join(methods)}")
    print(f"{'='*70}\n")

    # ── Run each method ───────────────────────────────────────────────
    results_summary: list[dict] = []
    timestamp = int(time.time())

    for method in methods:
        print(f"\n{'━'*70}")
        print(f"  Running: {method}")
        print(f"{'━'*70}")

        runner = ReplayRunner(
            rec_arg,
            registration=method,
            octo_resolution=OCTO_RESOLUTION,
            bounds=EXPLORE_BOUNDS,
            planner_res=PLANNER_RES,
            frame_skip=FRAME_SKIP,
            enable_viewer=ENABLE_VIEWER,
            enable_planner=False,
        )

        t0 = time.perf_counter()
        try:
            pipeline = runner.run()
        except Exception as e:
            print(f"  ERROR running {method}: {e}")
            results_summary.append({"method": method, "status": "FAILED",
                                    "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0

        pipeline.print_summary()

        # ── Save the map ──────────────────────────────────────────────
        map_out_dir = os.path.join(SAVED_MAPS_DIR,
                                   f"slam_map_{timestamp}_{method}")
        npz_path = runner.save_map(
            out_dir=map_out_dir,
            source=f"benchmark_{method}",
            extra_metadata={
                "recording": Path(runner.recording_dir).name,
                "registration_method": method,
            },
        )

        # ── Quality plot ──────────────────────────────────────────────
        if SAVE_QUALITY_PLOT:
            _save_quality_plot(pipeline, method, map_out_dir)

        # ── Collect summary ───────────────────────────────────────────
        summary = pipeline.get_summary()
        results_summary.append({
            "method":         method,
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

    print(f"\n\n{'='*90}")
    print(f"  REGISTRATION BENCHMARK RESULTS")
    print(f"  {len(ok)}/{len(results_summary)} methods succeeded")
    print(f"{'='*90}")

    if ok:
        hdr = (f"  {'Method':<16} {'Voxels':>10} {'Poses':>7} {'Submaps':>8} "
               f"{'Rejected':>9} {'Plane Res':>10} {'Time(s)':>9}")
        print(hdr)
        print(f"  {'─'*82}")
        for r in ok:
            print(f"  {r['method']:<16} {r['voxel_count']:>10,} "
                  f"{r['pose_count']:>7} {r['submap_count']:>8} "
                  f"{r['rejected_count']:>9} {r['plane_residual']:>10.4f} "
                  f"{r['total_time']:>9.1f}")
    if failed:
        print(f"\n  Failed methods:")
        for r in failed:
            print(f"    {r['method']}: {r.get('error', 'unknown')}")

    print(f"{'='*90}")

    # ── Save CSV summary ──────────────────────────────────────────────
    csv_path = os.path.join(SAVED_MAPS_DIR, f"benchmark_{timestamp}.csv")
    with open(csv_path, "w") as f:
        f.write("method,status,voxel_count,pose_count,submap_count,"
                "rejected_count,plane_residual,total_time,map_path\n")
        for r in results_summary:
            f.write(f"{r['method']},{r['status']},"
                    f"{r.get('voxel_count','')},{r.get('pose_count','')},"
                    f"{r.get('submap_count','')},{r.get('rejected_count','')},"
                    f"{r.get('plane_residual','')},{r.get('total_time','')},"
                    f"{r.get('map_path','')}\n")
    print(f"\n  CSV summary saved to {csv_path}")
    print(f"  All maps saved under {SAVED_MAPS_DIR}/slam_map_{timestamp}_*/")
    print(f"  Run mapAnalysis.py to compare against ground truth.")
    print("Done.")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nTotal wall-clock: {time.perf_counter() - t0:.1f}s")
