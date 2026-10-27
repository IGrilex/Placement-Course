"""
printing_helper_functions.py
─────────────────────────────
Grab-bag of output helpers: calibration readers, text-file writers, and matplotlib
plots. Anything that turns data into a file or a picture lives here.

Roughly in order:
  1. Calibration readers          (used by prep_project.py)
  2. Dataset text writers         (used by prep_project.py)
  3. Dataset plots                (used by prep_project.py)
  4. Run text writers             (used by slam_runner.py + first_run_pyslam.py)
  5. Dashboard writer             (used by slam_runner.py)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# 1.  Calibration readers ------------------------------------------------------------------

def read_image_resolution(perspective_file: Path) -> tuple[int, int]:
    """Read (width, height) from S_rect_00 in perspective.txt."""
    try:
        with open(perspective_file) as f:
            for line in f:
                if line.startswith("S_rect_00"):
                    parts = line.split(":")[1].split()
                    return int(float(parts[0])), int(float(parts[1]))
    except FileNotFoundError:
        pass
    return None, None


def compute_stereo_baseline(perspective_file: Path) -> float | None:
    """Compute baseline (metres) from P_rect_01 tx / fx."""
    try:
        with open(perspective_file) as f:
            for line in f:
                if line.startswith("P_rect_01"):
                    P = np.array(line.split(":")[1].split(), dtype=float).reshape(3, 4)
                    return -P[0, 3] / P[0, 0]
    except FileNotFoundError:
        pass
    return None


# 2.  Dataset text writers -----------------------------------------------------------------

def write_sequence_stats(
    output_file: Path,
    test_name: str,
    drive: str,
    frames: int,
    frame_min: int,
    frame_max: int,
    fps: int = 10,
) -> None:
    seconds = frames / fps
    with open(output_file, "w") as f:
        f.write(f"Sequence : {test_name} / {drive}\n")
        f.write(f"Frames   : {frames}\n")
        f.write(f"Range    : {frame_min}  →  {frame_max}\n")
        f.write(f"FPS      : {fps}\n")
        f.write(f"Duration : {seconds:.2f} s  ({seconds/60:.2f} min)\n")


def write_dataset_summary(
    output_file: Path,
    resolution: tuple[int, int],
    baseline: float | None,
    sequences: list[dict],
) -> None:
    with open(output_file, "w") as f:
        f.write("Dataset  : KITTI-360 test-SLAM subset\n\n")
        if resolution[0]:
            f.write(f"Resolution : {resolution[0]} × {resolution[1]} px\n")
        if baseline:
            f.write(f"Baseline   : {baseline:.4f} m\n")
        f.write("Frame rate : 10 Hz\n")
        f.write("\n" + "─" * 48 + "\n\n")
        for seq in sequences:
            f.write(f"  {seq['test']}  /  {seq['drive']}\n")
            f.write(f"    frames   : {seq['frames']}\n")
            f.write(f"    range    : {seq['min']}  →  {seq['max']}\n")
            f.write(f"    duration : {seq['frames']/10:.1f} s\n\n")


# 3.  Dataset plots ------------------------------------------------------------------------

def _load_kitti_poses(kitti_file: Path) -> np.ndarray | None:
    """
    Load KITTI-format poses and keep just the translations → (N,3) of [x, y, z].
    Each line is 12 floats = a 3×4 matrix, row-major, so translation is cols 3, 7, 11.
    """
    try:
        data = np.loadtxt(kitti_file)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.size == 0 or data.shape[1] < 12:
            return None
        return data[:, [3, 7, 11]]
    except Exception:
        return None


def plot_trajectory(
    kitti_poses_file: Path,
    output_file: Path,
    title: str = "Trajectory",
) -> None:
    """
    Top-down X-Y trajectory, coloured by how far along the path each point is.
    Axes start at -PAD so the path never sits right on the border.
    """
    PAD = 15

    xyz = _load_kitti_poses(kitti_poses_file)
    if xyz is None:
        print(f"  [!] Could not plot — {kitti_poses_file.name} empty or missing.")
        return

    x = xyz[:, 0] - xyz[:, 0].min()
    y = xyz[:, 1] - xyz[:, 1].min()

    dx   = np.diff(x, prepend=x[0])
    dy   = np.diff(y, prepend=y[0])
    dist = np.cumsum(np.sqrt(dx**2 + dy**2))

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm     = Normalize(vmin=0, vmax=dist[-1])
    lc       = LineCollection(segments, cmap="plasma", norm=norm,
                               linewidth=2, alpha=0.85)
    lc.set_array(dist[:-1])
    ax.add_collection(lc)

    fig.colorbar(ScalarMappable(norm=norm, cmap="plasma"), ax=ax,
                 shrink=0.7).set_label("Distance from start (m)", fontsize=9)

    ax.scatter(x[0],  y[0],  color="lime", s=80, zorder=5,
               label="Start", edgecolors="k", linewidths=0.5)
    ax.scatter(x[-1], y[-1], color="red",  s=80, zorder=5,
               label="End",   edgecolors="k", linewidths=0.5)

    ax.set_xlim(-PAD, x.max() + PAD)
    ax.set_ylim(-PAD, y.max() + PAD)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title(f"{len(x)} frames  |  {dist[-1]:.0f} m", fontsize=9)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  - Trajectory plot  →  {output_file.name}")


def plot_dataset_overview(sequences: list[dict], output_file: Path) -> None:
    """Bar chart of each sequence's duration in seconds, labelled test0/test1/..."""
    if not sequences:
        return

    labels    = [s["test"] for s in sequences]
    durations = [s["frames"] / 10 for s in sequences]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Dataset overview", fontweight="bold")
    ax.bar(labels, durations, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Duration per sequence")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(0, max(durations) * 1.15)
    for i, v in enumerate(durations):
        ax.text(i, v + max(durations)*0.02, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  - Dataset overview  →  {output_file.name}")


def plot_run_trajectory(
    gt_traj,                 # Trajectory (t, xyz, quat) — ground truth
    pred_traj,               # Trajectory (t, xyz, quat) — estimate
    output_file: Path,
    title: str = "Trajectory",
    ate=None,                # ATEResult | None — used for alignment + RMSE label
) -> None:
    """
    Top-down plot of ground truth (grey) vs prediction (green), aligned with the
    same Umeyama transform used for ATE so the picture matches the reported RMSE.
    Plots the two highest-variance axes of the GT path.
    """
    # lazy import so the util modules don't get tangled up in each other's import order.
    try:
        from src_pc.utils.metrics_helper_functions import _umeyama, _associate
    except Exception:
        _umeyama = _associate = None

    try:
        gt_xyz   = np.asarray(gt_traj.xyz,   dtype=float)
        pred_xyz = np.asarray(pred_traj.xyz, dtype=float)
    except Exception:
        return
    if gt_xyz.size == 0 or pred_xyz.size == 0:
        print("  [!] Trajectory plot skipped — empty trajectory.")
        return

    # matched index pairs (GT idx, pred idx)
    ir = iq = None
    if ate is not None and getattr(ate, "idx_ref", None) is not None \
            and getattr(ate, "idx_query", None) is not None:
        ir = np.asarray(ate.idx_ref)
        iq = np.asarray(ate.idx_query)
    elif _associate is not None:
        try:
            ir, iq = _associate(np.asarray(gt_traj.t, dtype=float),
                                np.asarray(pred_traj.t, dtype=float), 0.05)
        except Exception:
            ir = iq = None

    # align prediction onto GT (Umeyama, with scale) using just the matched subset
    pred_aligned = pred_xyz
    if _umeyama is not None and ir is not None and iq is not None and len(ir) >= 3:
        try:
            R, t, s = _umeyama(gt_xyz[ir], pred_xyz[iq], True)
            pred_aligned = (s * (R @ pred_xyz.T)).T + t
        except Exception:
            pred_aligned = pred_xyz

    # pick the two axes to plot = the two the GT path varies along the most
    var  = gt_xyz.var(0)
    a, b = sorted(int(i) for i in np.argsort(var)[::-1][:2])
    axis_name = {0: "X", 1: "Y", 2: "Z"}

    GT_COLOR   = "0.55"      # medium grey
    PRED_COLOR = "#2CA02C"   # green

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    ax.plot(gt_xyz[:, a], gt_xyz[:, b], color=GT_COLOR,
            linewidth=2.2, alpha=0.9, label="ground truth", zorder=2)
    ax.plot(pred_aligned[:, a], pred_aligned[:, b], color=PRED_COLOR,
            linewidth=1.4, alpha=0.95, label="prediction", zorder=3)
    ax.scatter(gt_xyz[0, a], gt_xyz[0, b], s=55, facecolor="white",
               edgecolor="black", linewidths=1.2, zorder=4, label="start")

    ax.set_xlabel(f"{axis_name[a]} (m)")
    ax.set_ylabel(f"{axis_name[b]} (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="best")
    if ate is not None and getattr(ate, "rmse", None) is not None:
        ax.set_title(f"ATE RMSE = {ate.rmse:.3f} m   |   scale = {ate.scale:.3f}",
                     fontsize=9)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectory   →  {output_file.name}")


# 4.  Run text writers  (slam_runner.py + first_run_pyslam.py) -----------------------------
 
def write_run_summary(
    output_file: Path,
    *,
    sequence:          str,
    variant:           str,
    config_name:       str,
    total_frames:      int,
    processed:         int,
    wall_time_s:       float,
    avg_fps:           float,
    map_points:        Optional[int] = None,
    tracking_quality:  Optional[dict] = None,
) -> None:
    """Dump a plain-text run summary. tracking_quality keys used: pre_init_frames,
    tracked_frames, lost_frames, first_lost_frame, reloc_attempts, reloc_successes,
    longest_lost_streak, map_points_count, num_keyframes, map_ply_saved."""
    tq = tracking_quality or {}
    lost_pct = tq.get("lost_frames", 0) / max(total_frames, 1) * 100
 
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"timestamp          : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"sequence           : {sequence}\n")
        f.write(f"variant            : {variant}\n")
        f.write(f"config             : {config_name}\n")
        f.write(f"total_frames       : {total_frames}\n")
        f.write(f"processed          : {processed}\n")
        f.write(f"pre_init_frames    : {tq.get('pre_init_frames', 'N/A')}\n")
        f.write(f"tracked_frames     : {tq.get('tracked_frames', 'N/A')}\n")
        f.write(f"lost_frames        : {tq.get('lost_frames', 0)}  ({lost_pct:.1f}%)\n")
        f.write(f"first_lost_frame   : {tq.get('first_lost_frame', 'none')}\n")
        f.write(f"reloc_attempts     : {tq.get('reloc_attempts', 0)}\n")
        f.write(f"reloc_successes    : {tq.get('reloc_successes', 0)}\n")
        f.write(f"longest_lost_streak: {tq.get('longest_lost_streak', 0)}\n")
        f.write(f"map_points_count   : {tq.get('map_points_count', 'N/A')}\n")
        f.write(f"num_keyframes      : {tq.get('num_keyframes', 'N/A')}\n")
        f.write(f"map_ply_saved      : {tq.get('map_ply_saved', False)}\n")
        if map_points is not None:
            f.write(f"map_ply_points     : {map_points}\n")
        f.write(f"wall_time_s        : {wall_time_s:.2f}\n")
        f.write(f"avg_fps            : {avg_fps:.2f}\n")
    print(f"  Run summary  →  {output_file.name}")

def write_system_usage(
    output_file: Path,
    sys_summary: dict,
) -> None:
    """Dump the CPU/RAM/GPU avg+max numbers to a plain-text file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("System usage during SLAM run\n")
        f.write("─" * 32 + "\n")
        for key, val in sys_summary.items():
            f.write(f"  {key:<16}: {val:.1f}\n")
    print(f"  System usage →  {output_file}")

# 5. Machine usage writers and plots -------------------------------------------------------

def write_system_usage_per_frame(
    output_file: Path,
    frame_timestamps: list[float],          # one entry per tracked frame
    monitor_samples,                        # list[SystemSample] from SystemMonitor
) -> None:
    """Match each frame timestamp to its nearest SystemMonitor sample; writes
    frame_id  timestamp  cpu_%  ram_%  gpu_% per row."""
    if not monitor_samples or not frame_timestamps:
        return
 
    sample_t = np.array([s.t for s in monitor_samples], dtype=float)

    # align frame ts (relative seconds) to monitor ts (wall-clock) via first-sample offset
    t0_monitor = sample_t[0]
    t0_frames  = frame_timestamps[0]
    offset     = t0_monitor - t0_frames   # add this to a frame ts to get wall-clock time
 
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("# frame_id  timestamp_s  cpu_%  ram_%  gpu_%\n")
 
        for fid, fts in enumerate(frame_timestamps):
            wall_t = fts + offset
            idx = int(np.argmin(np.abs(sample_t - wall_t)))
            s   = monitor_samples[idx]
            gpu = f"{s.gpu_percent:.1f}" if s.gpu_percent is not None else "N/A"
            f.write(
                f"{fid:6d}  {fts:.6f}  "
                f"{s.cpu_percent:.1f}  {s.ram_percent:.1f}  {gpu}\n"
            )
 
    print(f"  Usage/frame  →  {output_file.name}")
 
 
# 2. Machine usage plot (CPU / RAM / GPU over time) ----------------------------------------
 
def plot_system_usage(
    monitor_samples,          # list[SystemSample] from SystemMonitor
    output_file: Path,
    title: str = "System usage during SLAM run",
) -> None:
    """Line plot of CPU/RAM/GPU usage (%) over elapsed time, all on one axis."""
    if not monitor_samples:
        return
 
    t0  = monitor_samples[0].t
    ts  = np.array([s.t - t0          for s in monitor_samples], dtype=float)
    cpu = np.array([s.cpu_percent      for s in monitor_samples], dtype=float)
    ram = np.array([s.ram_percent      for s in monitor_samples], dtype=float)
    gpu = np.array(
        [s.gpu_percent if s.gpu_percent is not None else np.nan
         for s in monitor_samples],
        dtype=float,
    )
 
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")
 
    ax.plot(ts, cpu, color="steelblue",  linewidth=1.2, label="CPU %",  alpha=0.9)
    ax.plot(ts, ram, color="darkorange", linewidth=1.2, label="RAM %",  alpha=0.9)
 
    if not np.all(np.isnan(gpu)):
        ax.plot(ts, gpu, color="mediumseagreen", linewidth=1.2,
                label="GPU %", alpha=0.9)
 
    ax.set_xlabel("Elapsed time (s)", fontsize=10)
    ax.set_ylabel("Utilisation (%)",  fontsize=10)
    ax.set_xlim(0, ts[-1])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
 
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Usage plot   →  {output_file.name}")

# 6.  Dashboard writer ---------------------------------------------------------------------


def write_dashboard(
    output_file: Path,
    *,
    sequence:         str,
    variant:          str,
    total_frames:     int,
    processed:        int,
    wall_time_s:      float,
    avg_fps:          float,
    map_points,                  # int | None  (PLY point count if saved)
    sys_summary:      dict,
    ate,                         # ATEResult | None
    rpe_short,                   # RPEResult | None   delta_t = 0.5 s
    rpe_long,                    # RPEResult | None   delta_t = 2.0 s
    tracking_quality: Optional[dict] = None,
    abort_reason:     str = "completed",
) -> None:
    """
    Write the all-in-one dashboard text file.
    Key names here are exactly what parse_dashboard() greps for -- rename one
    here and update the matching regex in aggregate_results.py too.
    """
 
    def _f(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "N/A"
 
    tq       = tracking_quality or {}
    W        = 52
    lost_pct = tq.get("lost_frames", 0) / max(total_frames, 1) * 100
 
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
 
        # Header ---------------------------------------------------------------------------
        f.write("=" * W + "\n")
        f.write(f" {variant}  --  DASHBOARD\n")
        f.write("=" * W + "\n")
        f.write(f" timestamp         : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f" sequence          : {sequence}\n")
        f.write("\n")
 
        # Run info -------------------------------------------------------------------------
        f.write("-- Run info " + "-" * (W - 11) + "\n")
        f.write(f" abort_reason      : {abort_reason.upper()}\n")
        f.write(f" total_frames      : {total_frames}\n")
        f.write(f" processed         : {processed}\n")
        f.write(f" wall_time         : {wall_time_s:.2f} s\n")
        f.write(f" avg_fps           : {avg_fps:.2f}\n")
        f.write("\n")
 
        # Tracking quality -----------------------------------------------------------------
        tracked      = tq.get("tracked_frames", None)
        lost         = tq.get("lost_frames", 0)
        first_loss   = tq.get("first_lost_frame", None)
        reloc_att    = tq.get("reloc_attempts", 0)
        reloc_suc    = tq.get("reloc_successes", 0)
        longest      = tq.get("longest_lost_streak", 0)
        pre_init     = tq.get("pre_init_frames", None)
        mp_count     = tq.get("map_points_count", None)
        n_kf         = tq.get("num_keyframes", None)
        ply_saved    = tq.get("map_ply_saved", False)
 
        f.write("-- Tracking quality " + "-" * (W - 19) + "\n")
        # high pre_init with an otherwise-good ATE = slow extractor init, not drift
        if pre_init is not None:
            pct_pre = pre_init / max(total_frames, 1) * 100
            f.write(f" pre_init_frames   : {pre_init}"
                    f"  ({pct_pre:.1f}% of sequence before init)\n")
        else:
            f.write(f" pre_init_frames   : N/A\n")
        f.write(f" tracked_frames    : {tracked if tracked is not None else 'N/A'}"
                f"  /  {total_frames}\n")
        f.write(f" lost_frames       : {lost}  ({lost_pct:.1f}%)\n")
 
        if first_loss is not None:
            pct_through = first_loss / max(total_frames, 1) * 100
            f.write(f" first_loss_at     : frame {first_loss}"
                    f"  ({pct_through:.1f}% through sequence)\n")
        else:
            f.write(f" first_loss_at     : none\n")
 
        f.write(f" reloc_attempts    : {reloc_att}\n")
        f.write(f" reloc_successes   : {reloc_suc}"
                f"{'  (never recovered)' if reloc_att > 0 and reloc_suc == 0 else ''}\n")
        f.write(f" longest_streak    : {longest} frames\n")
        f.write("\n")
 
        # Map ------------------------------------------------------------------------------
        f.write("-- Map " + "-" * (W - 6) + "\n")
        if n_kf is not None:
            f.write(f" num_keyframes     : {n_kf:,}\n")
        else:
            f.write(f" num_keyframes     : N/A\n")
        if mp_count is not None:
            f.write(f" points_before_ba  : {mp_count:,}  (before final bundle adjust)\n")
        if ply_saved and map_points is not None:
            f.write(f" points_in_ply     : {map_points:,}\n")
            f.write(f" map_ply           : map/map.ply\n")
        else:
            reason = "skip_map_save variant" if not ply_saved else "--no-save-map"
            f.write(f" map_ply           : not saved  ({reason})\n")
        f.write("\n")
 
        # ATE ------------------------------------------------------------------------------
        f.write("-- ATE (metres, Umeyama alignment) " + "-" * (W - 35) + "\n")
        if ate is not None:
            f.write(f" rmse              : {_f(ate.rmse)}\n")
            f.write(f" mean              : {_f(ate.mean)}\n")
            f.write(f" median            : {_f(ate.median)}\n")
            f.write(f" max               : {_f(ate.max)}\n")
            f.write(f" frames_ate        : {ate.num_frames}\n")
            drift = "  (drift)" if abs(ate.scale - 1.0) > 0.05 else ""
            f.write(f" scale             : {_f(ate.scale)}{drift}\n")
        else:
            f.write(" N/A\n")
        f.write("\n")
 
        # RPE ------------------------------------------------------------------------------
        for rpe, label in [(rpe_short, "RPE  delta_t=0.5 s"),
                           (rpe_long,  "RPE  delta_t=2.0 s")]:
            f.write(f"-- {label} " + "-" * max(1, W - len(label) - 4) + "\n")
            if rpe is not None:
                f.write(f" trans rmse        : {_f(rpe.trans_rmse)} m\n")
                f.write(f" trans mean        : {_f(rpe.trans_mean)} m\n")
                f.write(f" trans median      : {_f(rpe.trans_median)} m\n")
                f.write(f" rot   rmse        : {_f(rpe.rot_rmse_deg)} deg\n")
                f.write(f" rot   mean        : {_f(rpe.rot_mean_deg)} deg\n")
                f.write(f" rot   median      : {_f(rpe.rot_median_deg)} deg\n")
                f.write(f" pairs             : {rpe.num_pairs}\n")
            else:
                f.write(" N/A\n")
            f.write("\n")
 
        # System usage ---------------------------------------------------------------------
        f.write("-- System usage " + "-" * (W - 16) + "\n")
        if sys_summary:
            for key, val in sys_summary.items():
                f.write(f" {key:<18}: {val:.1f}\n")
        else:
            f.write(" N/A\n")
        f.write("=" * W + "\n")
 
    print(f"  Dashboard    →  {output_file.name}")

def write_ate_detailed(
    output_file: Path,
    ate,          # ATEResult
    gt_traj,      # Trajectory
) -> None:
    """
    Write the per-frame ATE errors - handy for seeing where the drift creeps in.
    Columns: timestamp  gt_x  gt_y  gt_z  error_m
    """
    if ate is None or ate.errors is None:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("# timestamp  gt_x  gt_y  gt_z  ate_error_m\n")
        for k, (ir, err) in enumerate(zip(ate.idx_ref, ate.errors)):
            ts = gt_traj.t[ir]
            x, y, z = gt_traj.xyz[ir]
            f.write(f"{ts:.6f}  {x:.4f}  {y:.4f}  {z:.4f}  {err:.6f}\n")
    print(f"  ATE detailed →  {output_file}")


def write_rpe_detailed(
    output_file: Path,
    rpe,          # RPEResult
) -> None:
    """
    Write the per-pair RPE errors.
    Columns: t0  t1  trans_error_m  rot_error_deg
    """
    if rpe is None or rpe.trans_errors is None:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("# t0  t1  trans_error_m  rot_error_deg\n")
        for t0, t1, te, re in zip(rpe.t0, rpe.t1, rpe.trans_errors, rpe.rot_errors_deg):
            f.write(f"{t0:.6f}  {t1:.6f}  {te:.6f}  {re:.6f}\n")
    print(f"  RPE detailed →  {output_file}")