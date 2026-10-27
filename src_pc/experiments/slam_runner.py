#!/usr/bin/env python3
"""
src_pc/experiments/slam_runner.py

Runs a single SLAM/VO experiment for one (variant, test) pair: sets up pyslam,
runs the tracking loop, and writes metrics/plots/dashboard.

Usage:
  python3 src_pc/experiments/slam_runner.py --variant baseline --test test0
  python3 src_pc/experiments/slam_runner.py --variant fe_superpoint --test test0
  python3 src_pc/experiments/slam_runner.py --variant baseline --test test0 --view
  python3 src_pc/experiments/slam_runner.py --variant baseline --test test0 --no-loop-closure
  python3 src_pc/experiments/slam_runner.py --list-variants

Sensor modes:
  stereo      - depth from stereo matching (default)
  mono+depth  - left image + neural depth predictor (variant sets depth_estimator)

Wall-time budget:
  MAX_RUN_WALL_TIME_S starts at the first processed frame and is only checked
  between frames (best-effort) - run_experiments.py's master timer is the real
  safety net and SIGKILLs the whole process group if this hangs.

Output (results/{SLAM|VO}/{test_id}/{exp_type}/{short_name}/{timestamp}/):
  dashboard.txt, map/map.ply
  files/  run_summary.txt, system_usage.txt, progress.csv (per-frame, survives OOM),
          trajectory_online.txt, trajectory_final.txt, ate/rpe error files
  plot/   system_usage.png, progress.png, trajectory.png
"""

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import psutil
from datetime import datetime
from pathlib import Path

# shut ALSA/Qt up before anything else gets imported
os.environ["ALSA_PCM_CARD"] = "null"

PC_ROOT     = Path(__file__).resolve().parents[2]
PYSLAM_ROOT = PC_ROOT.parent / "pyslam"
sys.path.insert(0, str(PYSLAM_ROOT))
sys.path.insert(0, str(PC_ROOT))

if not PYSLAM_ROOT.is_dir():
    sys.exit(f"ERROR: pySLAM not found at {PYSLAM_ROOT}")

from pyslam.config import Config
from pyslam.config_parameters import Parameters
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.trajectory_writer import TrajectoryWriter
from pyslam.slam import USE_CPP, PinholeCamera
from pyslam.slam.slam import Slam, SlamState
from pyslam.utilities.evaluation import eval_ate
from pyslam.utilities.geom_trajectory import find_poses_associations
from pyslam.utilities.img_management import ImgWriter
from pyslam.utilities.logging import LoggerQueue, Printer
from pyslam.utilities.system import force_kill_all_and_exit
from pyslam.utilities.timer import TimerFps
from pyslam.viz.cvimage_thread import CvImageViewer
from pyslam.viz.viewer3D import Viewer3D

# Optional: neural-depth shadow-point filter (mono+depth path only)
try:
    from pyslam.utilities.depth import filter_shadow_points
except Exception:
    filter_shadow_points = None

from src_pc.configs.project_settings import (
    ATE_MAX_DT,
    RPE_DELTA_LONG, RPE_DELTA_SHORT,
    TRACKING_ABORT_STREAK, MAX_RUN_WALL_TIME_S,
    FILTER_SHADOW_POINTS,
    SAVE_MAP_DEFAULT,
    TEST_FRAMES,
)
from src_pc.experiments.variants import get_variant, is_baseline
from src_pc.utils.metrics_helper_functions import (
    SystemMonitor, compute_ate, compute_rpe, load_tum_trajectory,
)
from src_pc.utils.paths import CONFIGS_SEQUENCES, PROCESSED_POSES, RESULTS
from src_pc.utils.printing_helper_functions import (
    plot_run_trajectory, plot_system_usage, write_ate_detailed, write_dashboard,
    write_rpe_detailed,
)


# Helpers ----------------------------------------------------------------------------------

def _ram_available_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def _cleanup_ram(label: str = "") -> None:
    gc.collect(); gc.collect()
    time.sleep(2)
    gc.collect()
    tag = f"  [{label}] " if label else "  "
    print(f"{tag}RAM after cleanup: {_ram_available_gb():.1f} GB free")


def _apply_runtime_params(vcfg: dict) -> None:
    """
    Set the handful of pyslam Parameters actually read at runtime.
    ORB pyramid/threshold params go through the camera yaml instead, and map-memory
    caps are baked into the C++ core - neither is settable here, so don't add them back.
    """
    print(f"  USE_CPP_CORE : {getattr(Parameters, 'USE_CPP_CORE', 'n/a')}")

    Parameters.kOptimizationBundleAdjustUseGtsam = vcfg["use_gtsam"]

    Parameters.kStereoMaxDepth = vcfg["max_depth"]

    # front-end depth flag - only means anything for the mono+depth variants.
    if hasattr(Parameters, "kUseDepthEstimatorInFrontEnd"):
        Parameters.kUseDepthEstimatorInFrontEnd = vcfg.get("depth_estimator") is not None


def _load_depth_estimator(vcfg: dict, camera):
    """Load the neural depth estimator if the variant asks for one. None for stereo variants."""
    depth_estimator_name = vcfg.get("depth_estimator")
    if not depth_estimator_name:
        return None
    from pyslam.depth_estimation.depth_estimator_factory import (
        depth_estimator_factory, DepthEstimatorType,
    )
    from pyslam.io.dataset_types import DatasetEnvironmentType
    depth_type = DepthEstimatorType.from_string(depth_estimator_name)
    estimator = depth_estimator_factory(
        depth_estimator_type = depth_type,
        camera               = camera,
        max_depth            = vcfg["max_depth"],
        dataset_env_type     = DatasetEnvironmentType.OUTDOOR,
    )
    print(f"  Depth model  : {depth_estimator_name}")
    return estimator


def _get_map_points_xyzrgb(slam: Slam) -> list:
    xyzrgb = []
    try:
        for p in slam.map.get_points():
            if p is None: continue
            arr = np.asarray(p.pt()).ravel()
            if arr.size < 3: continue
            try:
                bgr = np.asarray(p.color).ravel()
                r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
            except Exception:
                r, g, b = 255, 0, 0
            xyzrgb.append((arr[0], arr[1], arr[2], r, g, b))
    except Exception as e:
        print(f"  [!] Map point extraction failed: {e}")
    return xyzrgb


def _save_map_ply(slam: Slam, out_path: Path) -> int | None:
    xyzrgb = _get_map_points_xyzrgb(slam)
    if not xyzrgb:
        print("  [!] No map points — map.ply not written.")
        return None
    pts = np.array([[x, y, z] for x, y, z, *_ in xyzrgb])
    c   = np.median(pts, axis=0)
    d   = np.linalg.norm(pts - c, axis=1)
    thr = np.median(d) + 4.0 * d.std()
    xyzrgb = [p for p, di in zip(xyzrgb, d) if di <= thr]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyzrgb)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in xyzrgb:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    print(f"  Map PLY      →  map/map.ply  ({len(xyzrgb):,} points)")
    return len(xyzrgb)


def _find_config(test_id: str) -> Path:
    candidates = sorted(CONFIGS_SEQUENCES.glob(f"{test_id}__*.yaml"))
    if not candidates:
        sys.exit(f"ERROR: No config for '{test_id}' — run prep_project.py first.")
    return candidates[0]


def _plot_run_progress(out_path: Path, fids, states, num_kf, num_mp, fps, title: str) -> None:
    """Draw the post-run progress plot from the per-frame data we kept in memory. Best-effort."""
    if not fids:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        state_code = {"OK": 2, "RELOCALIZE": 1, "LOST": 0}
        codes = [state_code.get(s, 2) for s in states]
        fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        fig.suptitle(title, fontsize=12, fontweight="bold")

        axs[0].step(fids, codes, where="post", color="#1A6BBF", linewidth=1.0)
        axs[0].set_yticks([0, 1, 2]); axs[0].set_yticklabels(["LOST", "RELOC", "OK"])
        axs[0].set_ylabel("state"); axs[0].set_ylim(-0.3, 2.3)
        axs[0].grid(True, linestyle="--", alpha=0.35)

        axs[1].plot(fids, num_mp, color="mediumseagreen", linewidth=1.0, label="map points")
        ax1b = axs[1].twinx()
        ax1b.plot(fids, num_kf, color="darkorange", linewidth=1.0, label="keyframes")
        axs[1].set_ylabel("map points", color="mediumseagreen")
        ax1b.set_ylabel("keyframes", color="darkorange")
        axs[1].grid(True, linestyle="--", alpha=0.35)

        axs[2].plot(fids, fps, color="steelblue", linewidth=0.8)
        axs[2].set_ylabel("fps (approx)"); axs[2].set_xlabel("frame id")
        axs[2].grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Progress plot→  {out_path.name}")
    except Exception as e:
        print(f"  [!] Progress plot failed: {e}")


# progress.csv columns ---------------------------------------------------------------------
# one row per frame, line-buffered so it survives a hard SIGKILL/OOM.

_PROGRESS_HEADER = [
    # bookkeeping
    "unix_ts", "elapsed_s",
    "frame_id", "seq_timestamp",
    # SLAM state
    "state", "num_kf", "num_map_pts",
    "lost_streak", "total_lost", "reloc_attempts", "fps_approx",
    # machine load (%)
    "ram_pct", "cpu_pct", "gpu_pct",
    # machine absolute  (cpu has no GB equivalent)
    "ram_used_gb", "gpu_mem_mb", "gpu_power_w",
]


# Main runner function ---------------------------------------------------------------------

def run_sequence(
    variant_name:     str,
    test_id:          str,
    output_dir:       Path | None = None,
    headless:         bool = False,
    save_map:         bool = True,
    no_loop_closure:  bool = False,
    test_mode:        bool = False,   # run TEST_FRAMES frames, skip all output/metrics
) -> dict:
    config_file   = _find_config(test_id)
    vcfg          = get_variant(variant_name)
    exp_type      = vcfg["experiment_type"]
    short_name    = vcfg["short_name"]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_depth     = vcfg.get("depth_estimator") is not None

    # results/VO/... (loop closure off) vs results/SLAM/... (loop closure on)
    mode = "VO" if no_loop_closure else "SLAM"

    # test mode: throwaway temp dir + frame cap. normal mode: real results tree.
    max_frames = 0
    if test_mode:
        output_dir = Path(tempfile.mkdtemp(prefix="slam_test_"))
        max_frames = TEST_FRAMES
        save_map   = False
    elif output_dir is None:
        # baseline runs once per (mode, test_id) into a shared folder so FE/LC/BE
        # batches can reuse it as their reference instead of re-running it each time.
        if is_baseline(variant_name):
            output_dir = RESULTS / mode / test_id / "baseline" / timestamp_str
        else:
            output_dir = RESULTS / mode / test_id / exp_type / short_name / timestamp_str

    map_dir   = output_dir / "map"
    plot_dir  = output_dir / "plot"
    files_dir = output_dir / "files"
    for d in (output_dir, map_dir, plot_dir, files_dir):
        d.mkdir(parents=True, exist_ok=True)

    sensor_label = f"mono+{vcfg['depth_estimator']}" if use_depth else "stereo"

    print(f"\n{'='*60}")
    print(f"  {variant_name}  —  {test_id}"
          + ("  [TEST MODE — no output]" if test_mode else ""))
    print(f"  mode     : {mode}{'  (loop closure OFF)' if no_loop_closure else '  (loop closure ON)'}")
    print(f"  exp type : {exp_type}")
    print(f"  sensor   : {sensor_label}")
    if not test_mode:
        print(f"  output   : {output_dir.relative_to(PC_ROOT)}")
    print(f"  RAM free : {_ram_available_gb():.1f} GB")
    print(f"{'='*60}\n")

    os.chdir(str(PC_ROOT))
    if headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    _apply_runtime_params(vcfg)

    print(f"  Backend    : {'GTSAM' if vcfg['use_gtsam'] else 'G2O'}")
    print(f"  Features   : {vcfg['num_features']}")
    print(f"  Max depth  : {vcfg['max_depth']} m")

    config      = Config(str(config_file))
    dataset     = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings)
    camera      = PinholeCamera(config)

    # neural-depth variants use the RGBD front-end; dataset.sensor_type stays STEREO
    # (KITTI yaml), so is_monocular is False - correct, since neural depth is metric.
    sensor_type  = SensorType.RGBD if use_depth else dataset.sensorType()
    is_monocular = (dataset.sensor_type == SensorType.MONOCULAR)

    num_total_frames = dataset.num_frames
    drive_name       = config_file.stem.split("__")[1]
    sequence_name    = f"{test_id}/{drive_name}"

    traj_online = files_dir / "trajectory_online.txt"
    traj_final  = files_dir / "trajectory_final.txt"
    fmt = config.trajectory_saving_settings.get("format_type", "tum")
    online_writer = TrajectoryWriter(format_type=fmt, filename=str(traj_online))
    final_writer  = TrajectoryWriter(format_type=fmt, filename=str(traj_final))

    print(f"  Frames   : {num_total_frames}\n")

    loop_cfg = None if no_loop_closure else vcfg["loop_cfg"]

    slam = Slam(
        camera, vcfg["tracker_cfg"], loop_cfg, None,
        sensor_type,
        environment_type=dataset.environmentType(),
        config=config,
        headless=headless,
    )
    slam.set_viewer_scale(dataset.scale_viewer_3d)

    depth_estimator = _load_depth_estimator(vcfg, camera)

    viewer = img_viewer = img_writer = timer_fps = None
    if not headless:
        viewer = Viewer3D(scale=dataset.scale_viewer_3d)
        if groundtruth:
            gt_traj3d, _, gt_ts = groundtruth.getFull6dTrajectory()
            viewer.set_gt_trajectory(gt_traj3d, gt_ts, align_with_scale=is_monocular)
        viewer.wait_for_ready()
        img_viewer = CvImageViewer()
        img_writer = ImgWriter(font_scale=0.5)
        timer_fps  = TimerFps("Main", is_verbose=False)
        timer_fps.start()

    # wall-time clock starts here; loading Config/dataset/Slam/vocab doesn't count.
    t0 = time.time()

    # prime cpu_percent() now, else the first per-frame reading comes back 0.0.
    monitor = SystemMonitor()
    psutil.cpu_percent(interval=None)

    progress_fh  = open(files_dir / "progress.csv", "w", newline="", buffering=1)
    progress_csv = csv.writer(progress_fh)
    progress_fh.write(f"# t0={t0:.6f}  variant={variant_name}  sequence={sequence_name}  sensor={sensor_label}\n")
    progress_fh.flush()
    progress_csv.writerow(_PROGRESS_HEADER)
    progress_fh.flush()

    img_id = num_frames = 0
    is_initialized = False
    first_initialized_frame = None
    pre_init_frames = 0
    num_lost = num_reloc_attempts = num_reloc_success = 0
    first_lost_frame = None
    longest_lost_streak = _cur_streak = 0
    _was_lost = False
    _fps_approx = 0.0
    abort_reason = "completed"

    # in-memory per-frame series for the progress plot (post-init only)
    p_fids, p_states, p_kf, p_mp, p_fps = [], [], [], [], []

    try:
        from tqdm import tqdm
        frame_iter = tqdm(range(num_total_frames), desc=short_name,
                          unit="frame", dynamic_ncols=True, leave=True)
    except ImportError:
        frame_iter = range(num_total_frames)

    try:
        for _ in frame_iter:
            if not dataset.is_ok:
                break

            # wall-time guard, checked between frames only (best-effort).
            if MAX_RUN_WALL_TIME_S > 0 and (time.time() - t0) > MAX_RUN_WALL_TIME_S:
                print(f"\n  [!] Wall time limit ({MAX_RUN_WALL_TIME_S}s) reached — aborting.")
                abort_reason = "walltime"
                break

            # max-frames guard - just for test mode / quick sanity checks
            if max_frames > 0 and num_frames >= max_frames:
                print(f"\n  [max-frames={max_frames}] Stopping after {num_frames} frames.")
                abort_reason = "max_frames"
                break

            img = dataset.getImageColor(img_id)
            if img is None:
                time.sleep(0.1)
                continue

            ts = dataset.getTimestamp()

            # right image: None on mono+depth variants (RGBD path uses neural depth instead)
            img_right = None
            if not use_depth:
                img_right = dataset.getImageColorRight(img_id)

            # depth: neural prediction on mono+depth, else None (stereo - pyslam works it out)
            depth = None
            if use_depth and depth_estimator is not None:
                try:
                    depth_pred, _ = depth_estimator.infer(img, img_right)
                    if FILTER_SHADOW_POINTS and filter_shadow_points is not None:
                        depth = filter_shadow_points(depth_pred)
                    else:
                        depth = depth_pred
                except Exception as e:
                    print(f"  [!] Depth estimator failed frame {img_id}: {e}")
                    depth = None

            slam.track(img, img_right, depth=depth, img_id=img_id, timestamp=ts)

            if online_writer and slam.tracking.cur_R is not None:
                online_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, ts)

            if viewer:
                viewer.draw_slam_map(slam)
                if viewer.is_closed():
                    print("  Viewer closed — exiting.")
                    abort_reason = "viewer_closed"
                    break

            if img_viewer and img is not None:
                is_radius = viewer.is_draw_features_with_radius() if viewer else False
                img_draw  = slam.map.draw_feature_trails(
                    img, with_level_radius=is_radius,
                    trail_max_length=Parameters.kMaxFeatureTrailLength)
                if timer_fps:
                    timer_fps.refresh()
                    fps_txt = f" fps:{timer_fps.get_fps():.1f}" if USE_CPP else ""
                else:
                    fps_txt = ""
                img_writer.write(img_draw, f"id:{img_id}{fps_txt}", (20, 20))
                img_viewer.draw(img_draw, "Camera")

            state   = slam.tracking.state
            is_ok   = (state == SlamState.OK)
            is_lost = state in (SlamState.LOST, SlamState.RELOCALIZE)

            if not is_initialized:
                if is_ok:
                    is_initialized = True
                    first_initialized_frame = img_id
                    print(f"\n  [init] Map initialized at frame {img_id}")
                else:
                    pre_init_frames += 1
                    img_id += 1; num_frames += 1
                    continue

            if is_lost:
                num_lost += 1
                if first_lost_frame is None:
                    first_lost_frame = img_id
                if state == SlamState.RELOCALIZE:
                    num_reloc_attempts += 1
                _cur_streak += 1
                longest_lost_streak = max(longest_lost_streak, _cur_streak)
            else:
                if _was_lost:
                    num_reloc_success += 1
                _cur_streak = 0
            _was_lost = is_lost

            _now     = time.time()
            elapsed  = _now - t0
            _fps_approx = num_frames / max(elapsed, 1e-6)

            num_kf      = slam.map.num_keyframes() if hasattr(slam.map, "num_keyframes") else 0
            num_map_pts = slam.map.num_points()    if hasattr(slam.map, "num_points")    else 0
            state_str   = ("RELOCALIZE" if state == SlamState.RELOCALIZE
                           else "LOST"  if is_lost
                           else "OK"    if is_ok
                           else str(state))

            s = monitor.sample()

            progress_csv.writerow([
                # bookkeeping
                f"{_now:.6f}", f"{elapsed:.3f}",
                img_id, f"{ts:.6f}",
                # SLAM state
                state_str, num_kf, num_map_pts,
                _cur_streak, num_lost, num_reloc_attempts,
                f"{_fps_approx:.2f}",
                # machine load (%)
                f"{s.ram_percent:.1f}",
                f"{s.cpu_percent:.1f}",
                f"{s.gpu_percent:.1f}"     if s.gpu_percent     is not None else "",
                # machine absolute
                f"{s.ram_used_gb:.2f}",
                f"{s.gpu_mem_used_mb:.0f}" if s.gpu_mem_used_mb is not None else "",
                f"{s.gpu_power_w:.1f}"     if s.gpu_power_w     is not None else "",
            ])
            progress_fh.flush()

            p_fids.append(img_id); p_states.append(state_str)
            p_kf.append(num_kf);   p_mp.append(num_map_pts); p_fps.append(_fps_approx)

            if hasattr(frame_iter, "set_postfix"):
                _vm = psutil.virtual_memory()
                frame_iter.set_postfix(
                    state="LOST" if is_lost else "OK",
                    kf=num_kf, lost=num_lost,
                    ram=f"{_vm.used/1024**3:.1f}GB/{_vm.percent:.0f}%",
                    refresh=False,
                )

            if is_lost and _cur_streak >= TRACKING_ABORT_STREAK:
                print(f"\n  [!] Tracking lost for {TRACKING_ABORT_STREAK} frames — aborting.")
                abort_reason = "lost"
                break

            img_id += 1; num_frames += 1

    except KeyboardInterrupt:
        print("\n  CTRL+C")
        try: progress_fh.close()
        except Exception: pass
        force_kill_all_and_exit(verbose=False)
        sys.exit(0)
    except Exception as e:
        print(f"\n  ERROR — {type(e).__name__}: {e}")
        abort_reason = "error"
        traceback.print_exc()
    finally:
        try: progress_fh.close()
        except Exception: pass

    if hasattr(frame_iter, "close"):
        frame_iter.close()

    wall_time  = time.time() - t0
    avg_fps    = num_frames / wall_time if wall_time > 0 else 0.0

    active_frames  = num_frames - pre_init_frames
    tracked_frames = active_frames - num_lost

    if online_writer:
        online_writer.close_file()

    map_points_count = None
    try:
        map_points_count = slam.map.num_points()
        print(f"  Map points   :  {map_points_count:,} (before final BA)")
    except Exception as e:
        print(f"  [!] Map point count failed: {e}")

    map_points = None
    if save_map:
        map_points = _save_map_ply(slam, map_dir / "map.ply")

    tracking_quality = {
        "pre_init_frames": pre_init_frames, "first_init_frame": first_initialized_frame,
        "initialized": is_initialized, "active_frames": active_frames,
        "tracked_frames": tracked_frames, "lost_frames": num_lost,
        "first_lost_frame": first_lost_frame, "reloc_attempts": num_reloc_attempts,
        "reloc_successes": num_reloc_success, "longest_lost_streak": longest_lost_streak,
        "map_points_count": map_points_count, "map_ply_saved": save_map,
        "sensor_mode": sensor_label, "abort_reason": abort_reason,
    }

    ate = rpe_short = rpe_long = gt_traj_obj = None

    if not test_mode:
        try:
            est_poses, timestamps, ids = slam.get_final_trajectory()
            if groundtruth and len(est_poses) > 0:
                _, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
                _, assoc_est, assoc_gt = find_poses_associations(
                    timestamps, est_poses, gt_timestamps, gt_poses)
                _evo_tmp = output_dir / "_evo_tmp"
                _evo_tmp.mkdir(parents=True, exist_ok=True)
                ape_stats, _ = eval_ate(
                    poses_est=assoc_est, poses_gt=assoc_gt, frame_ids=ids,
                    curr_frame_id=img_id, is_final=not dataset.is_ok,
                    is_monocular=is_monocular, save_dir=str(_evo_tmp))
                shutil.rmtree(_evo_tmp, ignore_errors=True)
                Printer.green(f"\n pySLAM ATE: {json.dumps(ape_stats, indent=4)}\n")
            if final_writer:
                final_writer.write_full_trajectory(est_poses, timestamps)
                final_writer.close_file()
        except Exception as e:
            print(f"  Trajectory extraction failed: {e}")
            traceback.print_exc()

    # teardown order copied from main_slam.py; pyslam's multiprocessing teardown can
    # raise or hang, so each step is wrapped - a teardown hiccup must never skip the
    # metrics/dashboard writing below (main() force-kills the process right after).
    try:
        slam.quit()
    except Exception as e:
        print(f"  [!] slam.quit() failed during teardown: {e}")
    time.sleep(0.5)                # let slam threads/processes die before viewers

    try:
        sys_summary = monitor.summarize()
    except Exception as e:
        print(f"  [!] monitor.summarize() failed during teardown: {e}")
        sys_summary = {}

    try:
        if img_viewer: img_viewer.quit()
        if viewer:     viewer.quit()
    except Exception as e:
        print(f"  [!] viewer quit failed during teardown: {e}")
    try:
        LoggerQueue.stop_all_instances()
    except Exception as e:
        print(f"  [!] LoggerQueue.stop_all_instances() failed during teardown: {e}")
    time.sleep(1.0)                # let loggers/viewers finish before we write files

    if not test_mode:
        gt_tum_file = PROCESSED_POSES / test_id / f"{drive_name}_tum.txt"
        if traj_final.is_file() and gt_tum_file.is_file():
            try:
                gt_traj_obj = load_tum_trajectory(gt_tum_file)
                pred_traj   = load_tum_trajectory(traj_final)
                ate         = compute_ate(gt_traj_obj, pred_traj, max_dt=ATE_MAX_DT, with_scale=True)
                rpe_short   = compute_rpe(gt_traj_obj, pred_traj, delta_t=RPE_DELTA_SHORT)
                rpe_long    = compute_rpe(gt_traj_obj, pred_traj, delta_t=RPE_DELTA_LONG)
                print(f"  Custom ATE   :  RMSE={ate.rmse:.4f} m  scale={ate.scale:.4f}")
                plot_run_trajectory(
                    gt_traj_obj, pred_traj, plot_dir / "trajectory.png",
                    title=f"{variant_name}  —  {test_id}  [{mode}]",
                    ate=ate,
                )
            except Exception as e:
                print(f"  Custom metrics failed: {e}")

        plot_system_usage(monitor.samples, plot_dir / "system_usage.png",
                          title=f"{variant_name}  —  {sequence_name}")
        _plot_run_progress(plot_dir / "progress.png", p_fids, p_states, p_kf, p_mp, p_fps,
                           title=f"{variant_name}  —  {sequence_name}  ({abort_reason})")
        if ate is not None and gt_traj_obj is not None:
            write_ate_detailed(files_dir / "ate_errors.txt", ate, gt_traj_obj)
        if rpe_short is not None:
            write_rpe_detailed(files_dir / "rpe_short_errors.txt", rpe_short)
        if rpe_long is not None:
            write_rpe_detailed(files_dir / "rpe_long_errors.txt", rpe_long)
        write_dashboard(
            output_dir / "dashboard.txt",
            sequence=sequence_name, variant=variant_name,
            total_frames=num_total_frames, processed=num_frames,
            wall_time_s=wall_time, avg_fps=avg_fps, map_points=map_points,
            sys_summary=sys_summary, ate=ate, rpe_short=rpe_short, rpe_long=rpe_long,
            tracking_quality=tracking_quality, abort_reason=abort_reason,
        )

    del slam
    if depth_estimator is not None:
        del depth_estimator
    if viewer:
        try: del viewer
        except Exception: pass
    if img_viewer:
        try: del img_viewer
        except Exception: pass
    del dataset, groundtruth, camera
    _cleanup_ram(label=short_name)

    if test_mode:
        # nuke the temp dir - test mode must never leave anything in results/
        shutil.rmtree(output_dir, ignore_errors=True)
        result_tag = "PASS" if abort_reason in ("max_frames", "completed") \
                     else f"STOPPED ({abort_reason})"
        print(f"\n  [test-mode] {result_tag}  "
              f"{num_frames}/{TEST_FRAMES} frames  {wall_time:.1f}s\n")
    else:
        print(f"\n  Done ({abort_reason}) — {output_dir.relative_to(PC_ROOT)}\n")

    lost_pct = num_lost / max(active_frames, 1) * 100
    return {
        "variant": variant_name, "short_name": short_name,
        "exp_type": exp_type, "test_id": test_id, "sequence": sequence_name,
        "sensor_mode": sensor_label, "abort_reason": abort_reason,
        "ate_rmse":       ate.rmse             if ate       else None,
        "ate_mean":       ate.mean             if ate       else None,
        "ate_scale":      ate.scale            if ate       else None,
        "rpe_short_rmse": rpe_short.trans_rmse if rpe_short else None,
        "rpe_long_rmse":  rpe_long.trans_rmse  if rpe_long  else None,
        "pre_init_frames": pre_init_frames, "first_init_frame": first_initialized_frame,
        "active_frames": active_frames, "tracked_frames": tracked_frames,
        "tracking_lost": num_lost, "lost_pct": lost_pct,
        "first_lost_frame": first_lost_frame, "reloc_attempts": num_reloc_attempts,
        "reloc_successes": num_reloc_success, "longest_lost_streak": longest_lost_streak,
        "map_points": map_points, "map_points_count": map_points_count,
        "avg_fps": avg_fps, "wall_time_s": wall_time, "out_dir": str(output_dir),
    }


# CLI entry point --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PC project SLAM runner")
    parser.add_argument("--variant",          "-v", type=str, default="baseline")
    parser.add_argument("--test",             "-t", type=str, default="test0")
    parser.add_argument("--view",             action="store_true",
                        help="Show the 3D/image viewers. Default is headless.")
    parser.add_argument("--no-save-map",      action="store_true")
    parser.add_argument("--no-loop-closure",  action="store_true",
                        help="Disable loop closure (visual odometry). "
                             "Output goes to results/VO/ instead of results/SLAM/.")
    parser.add_argument("--test-mode",        action="store_true",
                        help=f"Run {TEST_FRAMES} frames only, skip all output/metrics.")
    parser.add_argument("--list-variants",    action="store_true")
    args = parser.parse_args()

    if args.list_variants:
        from src_pc.experiments.variants import list_variants, EXPERIMENT_TYPES
        print("\nAvailable variants:")
        for et in EXPERIMENT_TYPES:
            names = list_variants(et)
            if names:
                print(f"\n  {et}:")
                for n in names:
                    print(f"    {n}")
        print()
        sys.exit(0)

    if args.test == "all":
        sys.exit("ERROR: Use batch_runner.py for batch runs.")

    headless = not args.view

    run_sequence(
        variant_name    = args.variant,
        test_id         = args.test,
        headless        = headless,
        save_map        = SAVE_MAP_DEFAULT and not args.no_save_map,
        no_loop_closure = args.no_loop_closure,
        test_mode       = args.test_mode,
    )

    if headless:
        force_kill_all_and_exit(verbose=False)


if __name__ == "__main__":
    main()