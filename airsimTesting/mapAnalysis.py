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
TAU = 0.31

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


def extract_method_name(folder_name: str) -> str:
    """Extract the registration method from a folder name.

    Examples
    --------
    >>> extract_method_name("slam_map_1773015800_vgicp")
    'vgicp'
    >>> extract_method_name("slam_map_1773015800_fpfh_ransac")
    'fpfh_ransac'
    >>> extract_method_name("slam_map_1773015800_state_only")
    'state_only'
    """
    import re
    # Match prefix + digits + underscore, keep everything after
    m = re.match(r'^[A-Za-z_]*\d+_(.*)', folder_name)
    return m.group(1) if m else folder_name


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

    _FONT = {"family": "Arial", "size": 12}
    plt.rc("font", **_FONT)

    d_s2g = m["_d_s2g"]
    d_g2s = m["_d_g2s"]

    # Use the full clouds for display if available
    slam_display = slam_pts_full if slam_pts_full is not None else slam_pts
    gt_display   = gt_pts_full   if gt_pts_full   is not None else gt_pts

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Map Quality Analysis{f'  —  {label}' if label else ''}")

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
    ax.legend()
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
    ax.legend(markerscale=10)
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
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(0.01)
    return fig


def plot_combined(
    results: list[tuple[str, dict]],
    all_slam_pts_full: list[np.ndarray],
    gt_pts_full: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Three-panel figure overlaying ALL registration methods on shared axes.

    Panels
    ------
    (a) Histogram of SLAM→GT NN distances for every method (overlaid).
    (b) Top-down XY overlay showing GT (grey) + every SLAM map in a
        distinct colour.
    (c) CDF of SLAM→GT NN distances for every method.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    _FONT = {"family": "Arial", "size": 12}
    plt.rc("font", **_FONT)

    n = len(results)
    cmap = plt.cm.tab10
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Registration Method Comparison")

    # Compute a shared clip value for the histograms / CDF x-axis
    all_d = np.concatenate([m["_d_s2g"] for _, m in results])
    clip_val = float(np.percentile(all_d, 99))
    bins = np.linspace(0, clip_val, 80)
    tau = results[0][1]["tau"]

    # ── (a) SLAM→GT NN-distance histograms ────────────────────────────
    ax = axes[0]
    for (label, m), clr in zip(results, colors):
        ax.hist(m["_d_s2g"], bins=bins, alpha=0.45, label=label, color=clr,
                edgecolor=clr, linewidth=0.5)
    ax.axvline(tau, color="r", ls="--", lw=1, label=f"τ = {tau:.2f}m")
    ax.set_xlabel("NN distance SLAM→GT (m)")
    ax.set_ylabel("Count")
    ax.set_title("NN Distance Distribution (SLAM→GT)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (b) Top-down XY overlay — GT + all SLAM maps ─────────────────
    ax = axes[1]
    max_draw = 50_000
    # Draw GT first (grey, behind everything)
    gt_disp = gt_pts_full
    if len(gt_disp) > max_draw:
        idx = np.random.choice(len(gt_disp), max_draw, replace=False)
        gt_disp = gt_disp[idx]
    ax.scatter(gt_disp[:, 1], gt_disp[:, 0], c="#c0c0c0", s=0.3,
               alpha=0.25, rasterized=True, label="Ground truth")
    # Each SLAM map
    for (label, _m), clr, slam_full in zip(results, colors, all_slam_pts_full):
        sl_disp = slam_full
        if len(sl_disp) > max_draw:
            idx = np.random.choice(len(sl_disp), max_draw, replace=False)
            sl_disp = sl_disp[idx]
        ax.scatter(sl_disp[:, 1], sl_disp[:, 0], c=[clr], s=0.3,
                   alpha=0.3, rasterized=True, label=label)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title("Top-Down Overlay (X–Y) — All Methods")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(markerscale=10)
    ax.grid(True, alpha=0.3)

    # ── (c) CDF of SLAM→GT NN distances ──────────────────────────────
    ax = axes[2]
    for (label, m), clr in zip(results, colors):
        d = m["_d_s2g"]
        sorted_d = np.sort(d)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        if len(sorted_d) > 5000:
            step = len(sorted_d) // 5000
            sorted_d = sorted_d[::step]
            cdf = cdf[::step]
        ax.plot(sorted_d, cdf * 100, label=label, color=clr, lw=1.5)
    ax.axvline(tau, color="r", ls="--", lw=1, label=f"τ = {tau:.2f}m")
    ax.set_xlabel("NN distance SLAM→GT (m)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF of NN Distances (SLAM→GT)")
    ax.set_xlim(0, clip_val)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Combined figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(0.01)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — multi-map comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: list[tuple[str, dict]],
                    save_path: str | None = None) -> None:
    """Single grouped bar chart: metrics on the x-axis, one bar per method."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    _FONT = {"family": "Arial", "size": 12}
    plt.rc("font", **_FONT)

    labels = [r[0] for r in results]
    metrics_to_plot = [
        ("rmse_sym",  "RMSE (sym) [m] ↓"),
        ("chamfer",   "Chamfer [m] ↓"),
        ("accuracy",  "Accuracy [m] ↓"),
        ("hausdorff", "Hausdorff [10m] ↓"),
        ("precision", "Precision ↑"),
        ("recall",    "Recall ↑"),
        ("fscore",    "F-score ↑"),
    ]

    n_metrics = len(metrics_to_plot)
    n_maps = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_maps, 1)))

    # Bar geometry
    bar_width = 0.8 / max(n_maps, 1)
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * n_metrics), 6))
    fig.suptitle("Registration Method Comparison")

    for i, (map_label, m) in enumerate(results):
        vals = []
        for key, _ in metrics_to_plot:
            v = m[key]
            if key == "hausdorff":
                v /= 10.0          # display in units of 10 m
            # precision/recall/fscore stay in [0, 1]
            vals.append(v)
        offset = (i - (n_maps - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=map_label,
                      color=colors[i], edgecolor="k", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{v:.2f}",
                    ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics_to_plot])
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

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


def plot_3d_overlay_combined(
    results: list[tuple[str, dict]],
    all_slam_pts_full: list[np.ndarray],
    gt_pts_full: np.ndarray,
    max_draw: int = 500_000,
) -> None:
    """Open a single Open3D window showing GT + every SLAM map together.

    Each registration method gets a distinct colour so they can be
    compared visually.  Ground truth is drawn in grey.
    """
    import open3d as o3d

    # Distinct colours for each SLAM map (tab10 palette)
    _TAB10 = [
        [0.12, 0.47, 0.71],  # blue
        [1.00, 0.50, 0.05],  # orange
        [0.17, 0.63, 0.17],  # green
        [0.84, 0.15, 0.16],  # red
        [0.58, 0.40, 0.74],  # purple
        [0.55, 0.34, 0.29],  # brown
        [0.89, 0.47, 0.76],  # pink
        [0.50, 0.50, 0.50],  # grey
        [0.74, 0.74, 0.13],  # olive
        [0.09, 0.75, 0.81],  # cyan
    ]

    rng = np.random.default_rng(0)
    geometries: list = []

    # GT cloud (grey)
    gt_draw = gt_pts_full
    if len(gt_draw) > max_draw:
        gt_draw = gt_draw[rng.choice(len(gt_draw), max_draw, replace=False)]
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_draw.astype(np.float64))
    gt_pcd.paint_uniform_color([0.70, 0.70, 0.70])
    geometries.append(gt_pcd)

    legend_parts = ["GT (grey)"]
    for i, ((label, _m), slam_full) in enumerate(
            zip(results, all_slam_pts_full)):
        sl_draw = slam_full
        per_map = max_draw // max(len(results), 1)
        if len(sl_draw) > per_map:
            sl_draw = sl_draw[rng.choice(len(sl_draw), per_map, replace=False)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sl_draw.astype(np.float64))
        clr = _TAB10[i % len(_TAB10)]
        pcd.paint_uniform_color(clr)
        geometries.append(pcd)
        legend_parts.append(f"{label} (rgb {clr[0]:.2f},{clr[1]:.2f},{clr[2]:.2f})")

    # Bounding box
    all_pts = np.concatenate(
        [np.asarray(g.points) for g in geometries], axis=0)
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max)
    bbox.color = (0.3, 0.3, 0.3)
    geometries.append(bbox)

    title = "3-D Overlay — All Registration Methods"
    print(f"  Opening combined 3-D viewer …")
    for part in legend_parts:
        print(f"    • {part}")
    print(f"  Close the Open3D window to continue.")

    o3d.visualization.draw_geometries(
        geometries,
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


# ══════════════════════════════════════════════════════════════════════════════
# Timing analysis
# ══════════════════════════════════════════════════════════════════════════════

def load_timing_csv(slam_path: str) -> dict | None:
    """Load ``timing_breakdown.csv`` from a SLAM map folder.

    Returns a dict mapping step name to its row dict
    (total_s, mean_s, max_s, min_s, std_s, pct, count), or *None*
    if the file doesn't exist.
    """
    import csv
    p = Path(slam_path) / "timing_breakdown.csv"
    if not p.exists():
        return None
    rows: dict[str, dict] = {}
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row["step"]
            parsed: dict = {}
            for k in ("total_s", "mean_s", "max_s", "min_s", "std_s", "pct", "count"):
                val = row.get(k, "")
                parsed[k] = float(val) if val else 0.0
            rows[step] = parsed
    return rows


def plot_timing_comparison(
    timing_data: list[tuple[str, dict]],
    save_path: str | None = None,
) -> None:
    """Two-panel timing figure comparing registration methods.

    Left:  stacked bar chart showing total time broken into pipeline steps.
    Right: grouped bars of mean per-frame time for each step.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    _FONT = {"family": "Arial", "size": 12}
    plt.rc("font", **_FONT)

    # Steps to show (exclude 'total' and 'wall_clock' from the breakdown)
    steps = ["transform", "register", "gtsam", "octo_insert", "vox_track", "vis"]
    step_labels = {
        "transform":  "Transform",
        "register":   "Register",
        "gtsam":      "GTSAM",
        "octo_insert": "Octo Insert",
        "vox_track":  "Vox Track",
        "vis":        "Viewer",
    }
    step_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52",
                   "#8172b3", "#937860"]

    methods = [label for label, _ in timing_data]
    n = len(methods)
    x = np.arange(n)

    fig, (ax_stack, ax_mean) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Pipeline Timing Breakdown")

    # ── Left: stacked bar of total seconds per step ───────────────────
    bottoms = np.zeros(n)
    for step, clr in zip(steps, step_colors):
        vals = []
        for _, td in timing_data:
            vals.append(td.get(step, {}).get("total_s", 0.0))
        vals_arr = np.array(vals)
        ax_stack.bar(x, vals_arr, bottom=bottoms, color=clr,
                     edgecolor="k", linewidth=0.4,
                     label=step_labels.get(step, step))
        bottoms += vals_arr

    # Add wall-clock markers
    for i, (_, td) in enumerate(timing_data):
        wc = td.get("wall_clock", {}).get("total_s", 0.0)
        if wc > 0:
            ax_stack.plot(i, wc, "kv", ms=8, zorder=5)
    # Invisible point for legend
    ax_stack.plot([], [], "kv", ms=8, label="Wall clock")

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(methods)
    ax_stack.set_ylabel("Total time (s)")
    ax_stack.set_title("Total Time per Step")
    ax_stack.legend(loc="upper left")
    ax_stack.grid(True, axis="y", alpha=0.3)

    # ── Right: grouped bars of mean per-frame time ────────────────────
    n_steps = len(steps)
    bar_w = 0.8 / max(n, 1)
    sx = np.arange(n_steps)
    colors_methods = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    for i, (label, td) in enumerate(timing_data):
        means = [td.get(s, {}).get("mean_s", 0.0) * 1000 for s in steps]  # ms
        offset = (i - (n - 1) / 2) * bar_w
        bars = ax_mean.bar(sx + offset, means, bar_w, label=label,
                           color=colors_methods[i], edgecolor="k", linewidth=0.4)
        for bar, v in zip(bars, means):
            if v > 0:
                ax_mean.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height(), f"{v:.1f}",
                             ha="center", va="bottom", fontsize=8)

    ax_mean.set_xticks(sx)
    ax_mean.set_xticklabels([step_labels.get(s, s) for s in steps],
                            rotation=25, ha="right")
    ax_mean.set_ylabel("Mean per-frame time (ms)")
    ax_mean.set_title("Mean Per-Frame Time")
    ax_mean.legend()
    ax_mean.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Timing figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(0.01)
    return fig


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
    all_slam_pts_full: list[np.ndarray] = []   # raw clouds for combined plots
    timing_data: list[tuple[str, dict]] = []   # (label, timing_dict) pairs

    for slam_path in slam_paths:
        raw_name = Path(slam_path).stem
        if os.path.isdir(slam_path):
            raw_name = Path(slam_path).name
        label = extract_method_name(raw_name)
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
        all_slam_pts_full.append(slam_pts_raw)

        # Load timing breakdown if available
        td = load_timing_csv(slam_path)
        if td is not None:
            timing_data.append((label, td))

        # Individual windows with coordinate axes for orientation diagnosis
        if SHOW_INDIVIDUAL_MAPS:
            plot_individual_map(gt_pts_raw,
                                title=f"Ground Truth  [{label}]",
                                color=[0.70, 0.70, 0.70])
            plot_individual_map(slam_pts_raw,
                                title=f"SLAM Map  [{label}]",
                                color=[0.12, 0.47, 0.71])

    # ── Combined visualisation (all methods on the same axes) ─────────
    if results:
        out_dir = out or str(Path(slam_paths[0]).parent)
        os.makedirs(out_dir, exist_ok=True)

        if not no_plot:
            # Combined 2-D figure: histograms, top-down overlay, CDF
            plot_combined(
                results,
                all_slam_pts_full,
                gt_pts_raw,
                save_path=os.path.join(out_dir, "map_comparison_combined.png"),
            )

            # Combined 3-D overlay (all SLAM maps + GT in one viewer)
            plot_3d_overlay_combined(
                results, all_slam_pts_full, gt_pts_raw)

        # Grouped bar chart for key metrics
        if len(results) > 1 and not no_plot:
            plot_comparison(
                results,
                save_path=os.path.join(out_dir, "map_comparison.png"))

        # Timing comparison figure
        if timing_data and not no_plot:
            plot_timing_comparison(
                timing_data,
                save_path=os.path.join(out_dir, "timing_comparison.png"))

        # CSV summary
        csv_name = "map_comparison.csv" if len(results) > 1 else "map_analysis.csv"
        save_csv(results, os.path.join(out_dir, csv_name))

    if not no_plot and results:
        print("\n  Plots are open — close the windows or press Enter to exit.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
