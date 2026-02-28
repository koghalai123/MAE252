#!/usr/bin/env python3
"""Fetch ground-truth voxel grid from CoSys-AirSim and display with Viewer3D."""

import cosysairsim as airsim
import numpy as np
import os
import time
from sensorFeed import Viewer3D


def read_binvox(path):
    """Parse a .binvox file and return occupied voxel centres as (N, 3) array.

    The binvox format has an ASCII header followed by run-length-encoded
    binary data (value, count) pairs.  We decode the occupancy grid, find
    occupied voxels, and convert their indices back to world coordinates
    using the translate/scale from the header.
    """
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
        nx, ny, nz = dims
        total = nx * ny * nz
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
    # pairs: (value, count)
    values = data[0::2]
    counts = data[1::2]
    print(f"  [parse] RLE pairs: {len(values):,}  — slice: {time.time()-t2:.3f}s")

    t3 = time.time()
    grid = np.repeat(values, counts).astype(bool)
    if len(grid) < total:
        grid = np.append(grid, np.zeros(total - len(grid), dtype=bool))
    grid = grid[:total].reshape((nx, ny, nz))  # binvox uses x-major order
    print(f"  [parse] RLE decode + reshape: {time.time()-t3:.3f}s")

    # --- Convert occupied voxel indices to world coordinates -------------
    t4 = time.time()
    occupied = np.argwhere(grid)  # (N, 3) indices
    print(f"  [parse] argwhere ({len(occupied):,} occupied): {time.time()-t4:.3f}s")

    t5 = time.time()
    # Normalise to [0, 1] then apply scale and translate
    points = occupied.astype(np.float64) / max(dims)
    points = points * scale + translate
    print(f"  [parse] Coordinate transform: {time.time()-t5:.3f}s")
    print(f"  [parse] TOTAL parse time: {time.time()-t0:.3f}s")

    return points


# ── Configuration ────────────────────────────────────────────────────────────
CENTER   = airsim.Vector3r(0, 0, 0)   # centre of the voxel grid
X_SIZE   = 100                          # metres
Y_SIZE   = 100
Z_SIZE   = 20
RES      = 0.15                          # voxel resolution in metres
BINVOX   = "/tmp/airsim_ground_truth.binvox"

# ── Main ─────────────────────────────────────────────────────────────────────
t_start = time.time()

print("[1/5] Connecting to AirSim …")
t = time.time()
client = airsim.MultirotorClient()
client.confirmConnection()
print(f"       Done — {time.time()-t:.3f}s")

print(f"[2/5] Creating voxel grid ({X_SIZE}×{Y_SIZE}×{Z_SIZE} m, res={RES} m) …")
t = time.time()
ok = client.simCreateVoxelGrid(CENTER, X_SIZE, Y_SIZE, Z_SIZE, RES, BINVOX)
print(f"       Done — {time.time()-t:.3f}s")
if not ok:
    print("simCreateVoxelGrid failed — is the simulation running?")
    exit(1)

print(f"[3/5] Parsing {BINVOX} …")
t = time.time()
points = read_binvox(BINVOX)
print(f"       Occupied voxels: {len(points):,} — {time.time()-t:.3f}s")

print("[4/5] Cleaning up temp file …")
os.remove(BINVOX)

print(f"[5/5] Launching Viewer3D with {len(points):,} points …")
t = time.time()
viewer = Viewer3D()
viewer.start(initial_points=points)
print(f"       Viewer started — {time.time()-t:.3f}s")

print(f"\nTotal pipeline: {time.time()-t_start:.3f}s")
input("Press Enter to exit …")
viewer.stop()