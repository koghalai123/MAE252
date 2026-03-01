#!/usr/bin/env python3
"""Fetch ground-truth voxel grid from CoSys-AirSim and display with Viewer3D."""

import cosysairsim as airsim
import numpy as np
import os
import time
from sensorFeed import Viewer3D


def read_binvox(path, center=np.zeros(3)):
    """Parse a .binvox file and return occupied voxel centres in **NED** frame.

    CosysAirSim's ``simCreateVoxelGrid`` writes a binvox file whose
    three dimensions map to NED axes as:

    - **dim 0** (slowest-varying) → NED Y
    - **dim 1** (middle)          → UE Z-up (negated for NED Z-down)
    - **dim 2** (fastest-varying) → NED X

    The ``translate`` vector in the header is in **NED (X, Y, Z) order**,
    *not* in binvox dim order.  It represents the offset from ``center``
    to the grid's minimum corner.

    Parameters
    ----------
    path : str
        Path to the ``.binvox`` file.
    center : array-like, shape (3,)
        NED world-frame centre of the voxel grid (as passed to
        ``simCreateVoxelGrid``).  Added to the local translate to produce
        global NED coordinates.

    Returns
    -------
    ndarray, shape (N, 3)
        Occupied voxel centres in NED world coordinates.
    """
    center = np.asarray(center, dtype=np.float64)
    t0 = time.time()
    with open(path, "rb") as f:
        # --- ASCII header ---------------------------------------------------
        line = f.readline().strip()
        assert line.startswith(b"#binvox"), f"Not a binvox file: {line}"

        dims = None
        translate = np.zeros(3)
        scale = 1.0

        while True:
            line = f.readline().strip()
            if line.startswith(b"dim"):
                dims = list(map(int, line.split()[1:4]))
            elif line.startswith(b"translate"):
                translate = np.array(list(map(float, line.split()[1:4])))
            elif line.startswith(b"scale"):
                scale = float(line.split()[1])
            elif line.startswith(b"data"):
                break

        assert dims is not None, "Missing dim in binvox header"
        d0, d1, d2 = dims  # (X_NED, Z_local, Y_NED)
        total = d0 * d1 * d2
        max_d = max(dims)
        print(f"  [parse] Header: dims={dims}, translate={translate}, scale={scale}")
        print(f"  [parse] Total voxels in grid: {total:,}  ({total*1e-6:.1f}M)")
        print(f"  [parse] Header read: {time.time()-t0:.3f}s")

        # --- RLE binary data ------------------------------------------------
        t1 = time.time()
        raw = f.read()
        fsize = len(raw)
        print(f"  [parse] File binary payload: {fsize:,} bytes — read: {time.time()-t1:.3f}s")

    t2 = time.time()
    data = np.frombuffer(raw, dtype=np.uint8)
    values = data[0::2]
    counts = data[1::2]
    print(f"  [parse] RLE pairs: {len(values):,}  — slice: {time.time()-t2:.3f}s")

    t3 = time.time()
    grid_flat = np.repeat(values, counts).astype(bool)
    if len(grid_flat) < total:
        grid_flat = np.append(grid_flat, np.zeros(total - len(grid_flat), dtype=bool))

    # CosysAirSim binvox: dims are stored as (d0, d1, d2) where d1 is the
    # vertical (UE Z-up) axis and d0/d2 are the horizontal axes.
    # Standard binvox linearisation: d0-slowest, d1-middle, d2-fastest.
    # Keep the natural reshape so argwhere indices match the storage order.
    grid = grid_flat[:total].reshape((d0, d1, d2))
    print(f"  [parse] RLE decode + reshape: {time.time()-t3:.3f}s")

    # --- Convert occupied voxel indices to NED world coordinates ---------
    # Empirically determined axis mapping (verified against simGetCollisionInfo):
    #   NED X  ←  occupied[:,2]  (binvox dim 2, fastest-varying)
    #   NED Y  ←  occupied[:,0]  (binvox dim 0, slowest-varying)
    #   NED Z  ← -occupied[:,1]  (binvox dim 1, vertical / UE-Z-up, negated for NED-Z-down)
    # translate is in NED (X, Y, Z) order, NOT in binvox dim order.
    t4 = time.time()
    occupied = np.argwhere(grid)  # (N, 3)
    print(f"  [parse] argwhere ({len(occupied):,} occupied): {time.time()-t4:.3f}s")

    t5 = time.time()
    points = np.empty((len(occupied), 3), dtype=np.float64)
    points[:, 0] = occupied[:, 2] / max_d * scale + translate[0] + center[0]    # NED X
    points[:, 1] = occupied[:, 0] / max_d * scale + translate[1] + center[1]    # NED Y
    points[:, 2] = -(occupied[:, 1] / max_d * scale + translate[2] + center[2]) # NED Z
    print(f"  [parse] Coordinate transform → NED: {time.time()-t5:.3f}s")
    print(f"  [parse] TOTAL parse time: {time.time()-t0:.3f}s")
    print(f"  [parse] NED X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  [parse] NED Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  [parse] NED Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    return points


# ── Configuration ────────────────────────────────────────────────────────────
SAVE_DIR = "/home/koghalai/MAE252/airsimTesting/flight_recordings/"
CENTER   = airsim.Vector3r(0, -15, 0)   # centre of the voxel grid
X_SIZE   = 60                          # metres
Y_SIZE   = 60
Z_SIZE   = 25
RES      = 0.15                          # voxel resolution in metres
BINVOX   = "/tmp/airsim_ground_truth.binvox"

# ── Main ─────────────────────────────────────────────────────────────────────
t_start = time.time()

print("[1/6] Connecting to AirSim …")
t = time.time()
client = airsim.MultirotorClient()
client.confirmConnection()
print(f"       Done — {time.time()-t:.3f}s")

print(f"[2/6] Creating voxel grid ({X_SIZE}×{Y_SIZE}×{Z_SIZE} m, res={RES} m) …")
t = time.time()
ok = client.simCreateVoxelGrid(CENTER, X_SIZE, Y_SIZE, Z_SIZE, RES, BINVOX)
print(f"       Done — {time.time()-t:.3f}s")
if not ok:
    print("simCreateVoxelGrid failed — is the simulation running?")
    exit(1)

print(f"[3/6] Parsing {BINVOX} …")
t = time.time()
center_np = np.array([CENTER.x_val, CENTER.y_val, CENTER.z_val])
points = read_binvox(BINVOX, center=center_np)
print(f"       Occupied voxels: {len(points):,} — {time.time()-t:.3f}s")

print("[4/6] Cleaning up temp file …")
os.remove(BINVOX)

print("[5/6] Saving ground truth to .npz …")
t = time.time()
out_dir = os.path.join(SAVE_DIR, f"ground_truth_{int(time.time())}")
os.makedirs(out_dir, exist_ok=True)
np.savez(
    os.path.join(out_dir, "ground_truth.npz"),
    points=points.astype(np.float32),
    center=np.array([CENTER.x_val, CENTER.y_val, CENTER.z_val]),
    grid_size=np.array([X_SIZE, Y_SIZE, Z_SIZE]),
    resolution=np.array(RES),
    timestamp=np.array(time.time()),
)
print(f"       Saved to {out_dir}/ground_truth.npz — {time.time()-t:.3f}s")

print(f"[6/6] Launching Viewer3D with {len(points):,} points …")
t = time.time()
viewer = Viewer3D()
viewer.start(initial_points=points)
print(f"       Viewer started — {time.time()-t:.3f}s")

print(f"\nTotal pipeline: {time.time()-t_start:.3f}s")
input("Press Enter to exit …")
viewer.stop()