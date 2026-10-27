#!/usr/bin/env python3
"""
view_map.py
───────────
Opens the SLAM map point cloud (results/test_run/map.ply) in a 3D viewer.
Runs statistical outlier removal first so the cloud isn't a noisy mess.

Usage:
    python3 src_pc/utils/view_map.py

Controls (Open3D viewer):
    Left mouse    rotate
    Middle mouse  pan
    Scroll        zoom
    R             reset view
    Q / Esc       quit
"""

import os
import sys
from pathlib import Path

# force X11 - Open3D's GL renderer just doesn't work on Wayland
os.environ.setdefault("XDG_SESSION_TYPE", "x11")
os.environ.setdefault("DISPLAY", ":0")

PC_ROOT  = Path(__file__).resolve().parents[2]
PLY_FILE = PC_ROOT / "results" / "gradslam" / "map_icp.ply"

if not PLY_FILE.is_file():
    sys.exit(f"ERROR: map.ply not found at {PLY_FILE}\n"
             "       Run first_run_pyslam.py first.")

print(f"Loading : {PLY_FILE}")
print(f"Size    : {PLY_FILE.stat().st_size // 1024} KB")

# Outlier removal (numpy, works with or without Open3D) ------------------------------------

import numpy as np


def _load_ply_ascii(path: Path):
    """Minimal ASCII PLY reader → (xyz, rgb) arrays."""
    xyz, rgb = [], []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "end_header":
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) >= 3:
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
                rgb.append([int(parts[3])/255, int(parts[4])/255, int(parts[5])/255]
                           if len(parts) >= 6 else [1.0, 0.3, 0.0])
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32)


def remove_outliers(xyz: np.ndarray, rgb: np.ndarray,
                    nb_neighbors: int = 20,
                    std_ratio: float = 2.0):
    """Statistical outlier removal: drop points whose mean distance to their
    nb_neighbors nearest neighbours is more than std_ratio std devs above
    the global mean. Lower std_ratio removes more (2.0 gentle, 1.0 aggressive)."""
    if len(xyz) < nb_neighbors + 1:
        return xyz, rgb

    # batched to avoid an N×N distance matrix blowing up RAM
    BATCH = 2000
    mean_dists = np.empty(len(xyz), dtype=np.float32)

    for start in range(0, len(xyz), BATCH):
        end   = min(start + BATCH, len(xyz))
        batch = xyz[start:end]                          # (B, 3)
        diff  = xyz[np.newaxis, :, :] - batch[:, np.newaxis, :]  # (B, N, 3)
        sq    = (diff ** 2).sum(axis=2)                           # (B, N)
        sq.sort(axis=1)  # nb_neighbors closest per point (skip index 0 = itself)
        mean_dists[start:end] = np.sqrt(sq[:, 1:nb_neighbors+1]).mean(axis=1)

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask      = mean_dists <= threshold

    removed = int((~mask).sum())
    print(f"Outlier removal: kept {mask.sum():,} / {len(xyz):,} points "
          f"(removed {removed:,}, threshold={threshold:.3f} m)")

    return xyz[mask], rgb[mask]


# Load and clean ---------------------------------------------------------------------------

xyz, rgb = _load_ply_ascii(PLY_FILE)
print(f"Raw     : {len(xyz):,} points")

xyz, rgb = remove_outliers(xyz, rgb, nb_neighbors=50, std_ratio=1.5)

# Try Open3D -------------------------------------------------------------------------------
try:
    import open3d as o3d

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    print(f"\nControls: left-drag=rotate  middle-drag=pan  scroll=zoom  R=reset  Q=quit\n")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="PC map — orb2_dbow3 test0",
        width=1280,
        height=720,
    )
    sys.exit(0)

except ImportError:
    print("Open3D not installed — falling back to matplotlib.")
    print("For a much better viewer:  pip install open3d\n")
except Exception as e:
    print(f"Open3D failed ({e}) — falling back to matplotlib.\n")

# Matplotlib fallback ----------------------------------------------------------------------
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

MAX_POINTS = 40_000
if len(xyz) > MAX_POINTS:
    idx      = np.random.choice(len(xyz), MAX_POINTS, replace=False)
    xyz, rgb = xyz[idx], rgb[idx]
    print(f"Downsampled to {MAX_POINTS:,} for matplotlib (install open3d for full view)")

fig = plt.figure(figsize=(14, 9))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=0.4, linewidths=0)
ax.set_facecolor("#111111")
fig.patch.set_facecolor("#111111")
ax.set_title("ASP-SLAM map — orb2_dbow3 test0", color="white")
plt.tight_layout()
plt.show()