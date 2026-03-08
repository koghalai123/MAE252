#!/usr/bin/env python3
"""Compare a SLAM map against a ground-truth voxel model.

Configure the paths and parameters in the CONFIGURATION section below,
then run:  python mapAnalysis.py

Only points inside the exploration bounding box (stored in the SLAM .npz as
``bounds``) are used for comparison.  If no bounds are present the full
point clouds are used.

Metrics
-------
1. **RMSE** — Root Mean Square Error of nearest-neighbour distances
   (SLAM → GT and GT → SLAM).
2. **Chamfer Distance** — symmetric mean of NN distances
   (average of mean(d_s→g) and mean(d_g→s)).
3. **Hausdorff Distance** — max of the two directed Hausdorff distances
   (worst-case surface deviation).
4. **Precision / Recall / F-score** at a configurable distance threshold τ.
   - Precision: fraction of SLAM points within τ of *some* GT point.
   - Recall:    fraction of GT points within τ of *some* SLAM point.
   - F-score:   harmonic mean of Precision and Recall.
5. **Completeness** — percentage of GT points within τ of a SLAM point
   (same as Recall, reported separately for clarity).
6. **Accuracy** — mean NN distance from SLAM → GT
   (lower is better, like mean surface error).
7. **Median NN distance** (SLAM → GT and GT → SLAM).

All distances are in metres.

Output
------
- Prints a comparison table to stdout.
- Saves a ``map_analysis.png`` figure with:
    (a) histogram of NN distances,
    (b) top-down XY overlay of both point clouds,
    (c) cumulative distribution of NN distances.
- When multiple SLAM maps are given, produces a grouped bar chart and
  CSV summary.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these variables instead of using command-line arguments
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing saved SLAM maps (each sub-folder holds a slam_map.npz).
# Every entry inside this folder is automatically added to the comparison.
SLAM_MAP_DIR = os.path.join(_SCRIPT_DIR, "savedMaps")
SLAM_PATHS = sorted(
    os.path.join(SLAM_MAP_DIR, d)
    for d in os.listdir(SLAM_MAP_DIR)
    if os.path.isdir(os.path.join(SLAM_MAP_DIR, d))
) if os.path.isdir(SLAM_MAP_DIR) else []

# Folder containing the ground-truth map (auto-finds the single sub-folder).
GT_MAP_DIR = os.path.join(_SCRIPT_DIR, "groundTruthMap")
_gt_entries = [
    os.path.join(GT_MAP_DIR, d) for d in os.listdir(GT_MAP_DIR)
] if os.path.isdir(GT_MAP_DIR) else []
GT_PATH = _gt_entries[0] if len(_gt_entries) == 1 else GT_MAP_DIR

# Distance threshold (m) for precision / recall / F-score.
TAU = 0.5

# Output directory for figures / CSV.  None = beside first SLAM file.
OUT_DIR = None

# Set True to skip matplotlib plots (text + CSV output only).
NO_PLOT = False

# Set True to open separate Open3D windows for GT and each SLAM map
# with coordinate axes displayed, useful for diagnosing orientation issues.
SHOW_INDIVIDUAL_MAPS = False


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_npz(path: str) -> dict:
    """Load an .npz and return a plain dict of arrays."""
    path = str(path)
    if os.path.isdir(path):
        candidates = list(Path(path).glob("*.npz"))
        if len(candidates) == 1:
            path = str(candidates[0])
        else:
            # Prefer slam_map.npz or ground_truth.npz
            for name in ("slam_map.npz", "ground_truth.npz"):
                p = Path(path) / name
                if p.exists():
                    path = str(p)
                    break
            else:
                raise FileNotFoundError(
                    f"Multiple/no .npz files in {path}: {candidates}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def clip_to_bounds(pts: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Keep only points inside the axis-aligned box ``bounds``.

    Parameters
    ----------
    pts : (N, 3)
    bounds : (6,) — (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    mask = ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax))
    return pts[mask]


# ══════════════════════════════════════════════════════════════════════════════
# Core metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    slam_pts: np.ndarray,
    gt_pts: np.ndarray,
    tau: float = 0.5,
) -> dict:
    """Compute all comparison metrics between *slam_pts* and *gt_pts*.

    Parameters
    ----------
    slam_pts : (M, 3) float
    gt_pts   : (N, 3) float
    tau      : float
        Distance threshold (m) for precision / recall / F-score.

    Returns
    -------
    dict with keys:
        rmse_s2g, rmse_g2s, rmse_sym,
        chamfer,
        hausdorff_s2g, hausdorff_g2s, hausdorff,
        precision, recall, fscore,
        completeness, accuracy,
        median_s2g, median_g2s,
        mean_s2g, mean_g2s,
        n_slam, n_gt, tau
    """
    tree_gt = cKDTree(gt_pts)
    tree_sl = cKDTree(slam_pts)

    # ── Nearest-neighbour distances ───────────────────────────────────
    d_s2g, _ = tree_gt.query(slam_pts)    # SLAM → GT
    d_g2s, _ = tree_sl.query(gt_pts)      # GT → SLAM

    # ── RMSE ──────────────────────────────────────────────────────────
    rmse_s2g = float(np.sqrt(np.mean(d_s2g ** 2)))
    rmse_g2s = float(np.sqrt(np.mean(d_g2s ** 2)))
    rmse_sym = float(np.sqrt(0.5 * (np.mean(d_s2g ** 2) + np.mean(d_g2s ** 2))))

    # ── Chamfer distance ──────────────────────────────────────────────
    chamfer = float(0.5 * (np.mean(d_s2g) + np.mean(d_g2s)))

    # ── Hausdorff distance ────────────────────────────────────────────
    hausdorff_s2g = float(np.max(d_s2g))
    hausdorff_g2s = float(np.max(d_g2s))
    hausdorff = max(hausdorff_s2g, hausdorff_g2s)

    # ── Precision / Recall / F-score at threshold τ ───────────────────
    precision = float(np.mean(d_s2g <= tau))
    recall    = float(np.mean(d_g2s <= tau))
    fscore    = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ── Completeness & Accuracy ───────────────────────────────────────
    completeness = recall  # same definition, different name
    accuracy = float(np.mean(d_s2g))

    return dict(
        rmse_s2g=rmse_s2g, rmse_g2s=rmse_g2s, rmse_sym=rmse_sym,
        chamfer=chamfer,
        hausdorff_s2g=hausdorff_s2g, hausdorff_g2s=hausdorff_g2s,
        hausdorff=hausdorff,
        precision=precision, recall=recall, fscore=fscore,
        completeness=completeness, accuracy=accuracy,
        mean_s2g=float(np.mean(d_s2g)), mean_g2s=float(np.mean(d_g2s)),
        median_s2g=float(np.median(d_s2g)), median_g2s=float(np.median(d_g2s)),
        n_slam=len(slam_pts), n_gt=len(gt_pts), tau=tau,
        # Keep raw distances for plotting
        _d_s2g=d_s2g, _d_g2s=d_g2s,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(m: dict, label: str = "") -> None:
    header = f"  Map Quality Metrics{f'  ({label})' if label else ''}"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  SLAM points (in bounds):  {m['n_slam']:,}")
    print(f"  GT   points (in bounds):  {m['n_gt']:,}")
    print(f"  Threshold τ:              {m['tau']:.2f} m")
    print(f"  {'-'*56}")
    print(f"  RMSE (SLAM→GT):           {m['rmse_s2g']:.4f} m")
    print(f"  RMSE (GT→SLAM):           {m['rmse_g2s']:.4f} m")
    print(f"  RMSE (symmetric):         {m['rmse_sym']:.4f} m")
    print(f"  {'-'*56}")
    print(f"  Chamfer distance:         {m['chamfer']:.4f} m")
    print(f"  Hausdorff (SLAM→GT):      {m['hausdorff_s2g']:.4f} m")
    print(f"  Hausdorff (GT→SLAM):      {m['hausdorff_g2s']:.4f} m")
    print(f"  Hausdorff (symmetric):    {m['hausdorff']:.4f} m")
    print(f"  {'-'*56}")
    print(f"  Accuracy  (mean S→G):     {m['accuracy']:.4f} m")
    print(f"  Median NN (SLAM→GT):      {m['median_s2g']:.4f} m")
    print(f"  Median NN (GT→SLAM):      {m['median_g2s']:.4f} m")
    print(f"  {'-'*56}")
    print(f"  Precision (τ={m['tau']:.2f}m):      {m['precision']*100:.1f}%")
    print(f"  Recall    (τ={m['tau']:.2f}m):      {m['recall']*100:.1f}%")
    print(f"  F-score   (τ={m['tau']:.2f}m):      {m['fscore']*100:.1f}%")
    print(f"  Completeness:             {m['completeness']*100:.1f}%")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — single map
# ══════════════════════════════════════════════════════════════════════════════

def plot_single(m: dict, slam_pts: np.ndarray, gt_pts: np.ndarray,
                label: str = "", save_path: str | None = None,
                slam_pts_full: np.ndarray | None = None,
                gt_pts_full: np.ndarray | None = None) -> None:
    """Three-panel figure for a single SLAM-vs-GT comparison.

    If *slam_pts_full* / *gt_pts_full* are provided they are used for
    the top-down overlay so the viewer can see the full maps for spatial
    orientation.  Metrics are still computed on the clipped clouds.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    d_s2g = m["_d_s2g"]
    d_g2s = m["_d_g2s"]

    # Use the full clouds for display if available
    slam_display = slam_pts_full if slam_pts_full is not None else slam_pts
    gt_display   = gt_pts_full   if gt_pts_full   is not None else gt_pts

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Map Quality Analysis{f'  —  {label}' if label else ''}",
                 fontsize=14)

    # ── (a) NN distance histograms ────────────────────────────────────
    ax = axes[0]
    clip_val = np.percentile(np.concatenate([d_s2g, d_g2s]), 99)
    bins = np.linspace(0, clip_val, 80)
    ax.hist(d_s2g, bins=bins, alpha=0.6, label="SLAM→GT", color="#1f77b4")
    ax.hist(d_g2s, bins=bins, alpha=0.6, label="GT→SLAM", color="#ff7f0e")
    ax.axvline(m["tau"], color="r", ls="--", lw=1, label=f"τ = {m['tau']:.2f}m")
    ax.set_xlabel("NN distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-Neighbour Distance Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (b) Top-down XY overlay (full maps for orientation) ───────────
    ax = axes[1]
    # Subsample for plotting speed
    max_draw = 50_000
    if len(gt_display) > max_draw:
        idx = np.random.choice(len(gt_display), max_draw, replace=False)
        gt_draw = gt_display[idx]
    else:
        gt_draw = gt_display
    if len(slam_display) > max_draw:
        idx = np.random.choice(len(slam_display), max_draw, replace=False)
        sl_draw = slam_display[idx]
    else:
        sl_draw = slam_display

    ax.scatter(gt_draw[:, 1], gt_draw[:, 0], c="#c0c0c0", s=0.3,
               alpha=0.3, rasterized=True, label="Ground truth")
    ax.scatter(sl_draw[:, 1], sl_draw[:, 0], c="#1f77b4", s=0.3,
               alpha=0.3, rasterized=True, label="SLAM map")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title("Top-Down Overlay (X–Y) — Full Maps")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9, markerscale=10)
    ax.grid(True, alpha=0.3)

    # ── (c) Cumulative distribution of NN distances ───────────────────
    ax = axes[2]
    for d, lbl, clr in [(d_s2g, "SLAM→GT", "#1f77b4"),
                         (d_g2s, "GT→SLAM", "#ff7f0e")]:
        sorted_d = np.sort(d)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        # Downsample for speed
        if len(sorted_d) > 5000:
            step = len(sorted_d) // 5000
            sorted_d = sorted_d[::step]
            cdf = cdf[::step]
        ax.plot(sorted_d, cdf * 100, label=lbl, color=clr, lw=1.5)
    ax.axvline(m["tau"], color="r", ls="--", lw=1, label=f"τ = {m['tau']:.2f}m")
    ax.set_xlabel("NN distance (m)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of NN Distances")
    ax.set_xlim(0, clip_val)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(0.01)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — multi-map comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: list[tuple[str, dict]],
                    save_path: str | None = None) -> None:
    """Grouped bar chart comparing multiple SLAM maps on key metrics."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in results]
    metrics_to_plot = [
        ("rmse_sym",      "RMSE (sym)\n[m]"),
        ("chamfer",       "Chamfer\n[m]"),
        ("accuracy",      "Accuracy\n[m]"),
        ("hausdorff",     "Hausdorff\n[m]"),
        ("precision",     "Precision\n[%]"),
        ("recall",        "Recall\n[%]"),
        ("fscore",        "F-score\n[%]"),
    ]

    n_metrics = len(metrics_to_plot)
    n_maps = len(results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 5))
    fig.suptitle("Multi-Map Comparison", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_maps, 1)))

    for ax, (key, ylabel) in zip(axes, metrics_to_plot):
        vals = []
        for _, m in results:
            v = m[key]
            # Scale precision/recall/fscore to percentage
            if key in ("precision", "recall", "fscore", "completeness"):
                v *= 100
            vals.append(v)
        x = np.arange(n_maps)
        bars = ax.bar(x, vals, color=colors[:n_maps], edgecolor="k",
                       linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        # Show value on each bar
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(0.01)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3-D overlay (Open3D — same library as Viewer3D in sensorFeed.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_overlay(slam_pts: np.ndarray, gt_pts: np.ndarray,
                    label: str = "",
                    max_draw: int = 500_000,
                    slam_pts_full: np.ndarray | None = None,
                    gt_pts_full: np.ndarray | None = None) -> None:
    """Open an interactive Open3D window showing both clouds together.

    If *slam_pts_full* / *gt_pts_full* are provided they are displayed
    instead of the clipped clouds so the viewer can use the full maps
    for spatial orientation.

    * Ground truth  — grey  (0.70, 0.70, 0.70)
    * SLAM map      — blue  (0.12, 0.47, 0.71)

    Close the window to continue execution.
    """
    import open3d as o3d

    # Use full clouds for display when available
    gt_src   = gt_pts_full   if gt_pts_full   is not None else gt_pts
    slam_src = slam_pts_full if slam_pts_full is not None else slam_pts

    # Subsample if needed so the viewer stays responsive
    if len(gt_src) > max_draw:
        idx = np.random.default_rng(0).choice(len(gt_src), max_draw, replace=False)
        gt_draw = gt_src[idx]
    else:
        gt_draw = gt_src

    if len(slam_src) > max_draw:
        idx = np.random.default_rng(1).choice(len(slam_src), max_draw, replace=False)
        sl_draw = slam_src[idx]
    else:
        sl_draw = slam_src

    # Ground truth point cloud (grey)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_draw.astype(np.float64))
    gt_pcd.paint_uniform_color([0.70, 0.70, 0.70])

    # SLAM point cloud (blue)
    sl_pcd = o3d.geometry.PointCloud()
    sl_pcd.points = o3d.utility.Vector3dVector(sl_draw.astype(np.float64))
    sl_pcd.paint_uniform_color([0.12, 0.47, 0.71])

    # Bounding-box wireframe so the exploration region is visible
    all_pts = np.concatenate([gt_draw, sl_draw], axis=0)
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max)
    bbox.color = (0.3, 0.3, 0.3)

    title = f"3-D Overlay — GT (grey) vs SLAM (blue)"
    if label:
        title += f"  [{label}]"

    print(f"  Opening 3-D viewer: {len(gt_draw):,} GT + {len(sl_draw):,} SLAM points")
    print(f"  Close the Open3D window to continue.")

    o3d.visualization.draw_geometries(
        [gt_pcd, sl_pcd, bbox],
        window_name=title,
        width=1280, height=720,
        point_show_normal=False,
    )


def plot_individual_map(pts: np.ndarray, title: str,
                        color: list[float],
                        axis_size: float | None = None) -> None:
    """Open an Open3D window showing a single point cloud with a coordinate frame.

    All points are displayed (no subsampling or bounding-box clipping)
    so the full extent of the map is visible.

    Parameters
    ----------
    pts : (N, 3) float
        Point cloud to display.
    title : str
        Window title.
    color : list[float]
        RGB colour in [0, 1] for the point cloud.
    axis_size : float or None
        Length of the coordinate-frame axes.  *None* → auto (10 % of the
        point-cloud diagonal).
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.paint_uniform_color(color)

    # Coordinate frame at the origin
    if axis_size is None:
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        axis_size = max(diag * 0.10, 1.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size, origin=[0.0, 0.0, 0.0])

    print(f"  [{title}] Displaying {len(pts):,} points  "
          f"(axis size = {axis_size:.1f} m)")
    print(f"  Close the window to continue.")

    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name=title,
        width=1280, height=720,
        point_show_normal=False,
    )


def save_csv(results: list[tuple[str, dict]], path: str) -> None:
    """Write a CSV summary of all maps and their metrics."""
    keys = [
        "n_slam", "n_gt", "tau",
        "rmse_s2g", "rmse_g2s", "rmse_sym",
        "chamfer", "hausdorff_s2g", "hausdorff_g2s", "hausdorff",
        "accuracy", "median_s2g", "median_g2s", "mean_s2g", "mean_g2s",
        "precision", "recall", "fscore", "completeness",
    ]
    with open(path, "w") as f:
        f.write("map," + ",".join(keys) + "\n")
        for label, m in results:
            vals = ",".join(str(m[k]) for k in keys)
            f.write(f"{label},{vals}\n")
    print(f"  CSV saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    slam_paths = SLAM_PATHS
    gt_path    = GT_PATH
    tau        = TAU
    out        = OUT_DIR
    no_plot    = NO_PLOT

    # ── Load ground truth ─────────────────────────────────────────────
    print(f"Loading ground truth from {gt_path} ...")
    gt_data = load_npz(gt_path)
    gt_pts_raw = gt_data["points"].astype(np.float64)
    print(f"  {len(gt_pts_raw):,} raw GT points")

    # ── Process each SLAM map ─────────────────────────────────────────
    results: list[tuple[str, dict]] = []

    for slam_path in slam_paths:
        label = Path(slam_path).stem
        if os.path.isdir(slam_path):
            label = Path(slam_path).name
        print(f"\n{'─'*60}")
        print(f"Loading SLAM map: {slam_path}")

        slam_data = load_npz(slam_path)
        slam_pts_raw = slam_data["points"].astype(np.float64)
        print(f"  {len(slam_pts_raw):,} raw SLAM points")

        # Determine bounding box
        bounds = slam_data.get("bounds", None)
        if bounds is not None:
            bounds = bounds.astype(np.float64)
            print(f"  Bounds from SLAM file: "
                  f"x=[{bounds[0]:.1f}, {bounds[1]:.1f}], "
                  f"y=[{bounds[2]:.1f}, {bounds[3]:.1f}], "
                  f"z=[{bounds[4]:.1f}, {bounds[5]:.1f}]")
            slam_pts = clip_to_bounds(slam_pts_raw, bounds)
            gt_pts   = clip_to_bounds(gt_pts_raw, bounds)
            print(f"  After clipping: {len(slam_pts):,} SLAM, "
                  f"{len(gt_pts):,} GT points")
        else:
            print("  No bounds found — using full point clouds")
            slam_pts = slam_pts_raw
            gt_pts   = gt_pts_raw

        if len(slam_pts) == 0:
            print("  WARNING: No SLAM points in bounds — skipping")
            continue
        if len(gt_pts) == 0:
            print("  WARNING: No GT points in bounds — skipping")
            continue

        # Compute metrics
        t0 = time.time()
        m = compute_metrics(slam_pts, gt_pts, tau=tau)
        print(f"  Metrics computed in {time.time()-t0:.2f}s")
        print_metrics(m, label=label)
        results.append((label, m))

        # Per-map plot (full maps shown for spatial orientation)
        if not no_plot:
            out_dir = out or str(Path(slam_path).parent)
            os.makedirs(out_dir, exist_ok=True)
            fig_path = os.path.join(out_dir, f"map_analysis_{label}.png")
            plot_single(m, slam_pts, gt_pts, label=label, save_path=fig_path,
                        slam_pts_full=slam_pts_raw, gt_pts_full=gt_pts_raw)

        # 3-D overlay so you can visually verify alignment
        if not no_plot:
            plot_3d_overlay(slam_pts, gt_pts, label=label,
                            slam_pts_full=slam_pts_raw, gt_pts_full=gt_pts_raw)

        # Individual windows with coordinate axes for orientation diagnosis
        if SHOW_INDIVIDUAL_MAPS:
            plot_individual_map(gt_pts_raw,
                                title=f"Ground Truth  [{label}]",
                                color=[0.70, 0.70, 0.70])
            plot_individual_map(slam_pts_raw,
                                title=f"SLAM Map  [{label}]",
                                color=[0.12, 0.47, 0.71])

    # ── Multi-map comparison ──────────────────────────────────────────
    if len(results) > 1:
        out_dir = out or str(Path(slam_paths[0]).parent)
        os.makedirs(out_dir, exist_ok=True)
        if not no_plot:
            plot_comparison(results,
                            save_path=os.path.join(out_dir, "map_comparison.png"))
        save_csv(results, os.path.join(out_dir, "map_comparison.csv"))
    elif len(results) == 1:
        # Single map — still save CSV
        out_dir = out or str(Path(slam_paths[0]).parent)
        os.makedirs(out_dir, exist_ok=True)
        save_csv(results, os.path.join(out_dir, "map_analysis.csv"))

    if not no_plot and results:
        print("\n  Plots are open — close the windows or press Enter to exit.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
