#!/usr/bin/env python3
"""
src_pc/utils/aggregate_results.py

Aggregates ONE results tree: scans it, recovers dashboard-less runs,
classifies every run, writes summary figures and overview CSVs.

Invocation: `python3 aggregate_results.py [--root PATH] [--ram-gb N]
[--threads N] [--modes VO SLAM]`. For cross-condition comparison across
several results trees, see aggregate_all.py, which imports this module.

Truth sources, in priority order:
  dashboard.txt       - authoritative when present (abort_reason + metrics).
  run_flags.csv       - primary source for runs with no dashboard; matched to
                        a run folder by (variant, test_id) + the closest
                        end-time at/after the folder's start timestamp.
  dataset_summary.txt - expected frame count per test, for completion %.

Classification:
  dashboard present:
    SUCCESS  - abort_reason in {completed, max_frames}
    LOST     - abort_reason == lost
    TIMEOUT  - abort_reason == walltime
  no dashboard (hard-killed) -> from run_flags.csv:
    TIMEOUT  - status == timeout
    OOM      - status oom/failed OR returncode -9/-15
    ERROR    - no matching row

A dashboard-less run that left partial progress data gets post-processed, so
its recovered metrics still show up in the summary next to proper runs.

Every run is rebuilt from scratch on every pass: run_tree() deletes
<root>/summary/ and every <run>/post_processing/ before rebuilding. There is
no incremental mode or --reuse flag.

Metric columns are always population-suffixed, never bare:
    <key>_succ_{mean,median,std}   over completers only
    <key>_all_{mean,median,std}    over every run that produced data
Accuracy keys (ATE, RPE, scale) are _succ_ only -- scoring a run that died
early against completers would reward failing fast. Throughput, load and
map-size keys get both pools, since an OOM-killed run's peak RAM is still a
real measurement.
"""

from __future__ import annotations
import argparse
import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_FILE   = Path(__file__).resolve()
PC_ROOT = _FILE.parents[2]
sys.path.insert(0, str(PC_ROOT))

from src_pc.experiments.variants import (VARIANTS, EXPERIMENT_TYPES,     # noqa: E402
                                          BASELINE_VARIANT)
from src_pc.utils.paths import RESULTS, PROCESSED_POSES, DOCS_DATASET    # noqa: E402

_FE, _LC, _BE, _CO, _DE = (
    "feature_extraction", "loop_closure", "backend", "combined", "depth_estimation",
)
MONO_EXPS   = [_DE]
EXP_SHORT   = {_FE: "fe", _LC: "lc", _BE: "be", _CO: "co", _DE: "de"}

# ─── metric families ────────────────────────────────────────────────────────
# ACCURACY_KEYS: completers-only (<key>_succ_*) -- pooling in early failures
# would reward dying young. POOLED_KEYS: both pools, since an OOM-killed run's
# peak RAM/load is still a real, wanted measurement.
ACCURACY_KEYS: list[str] = [
    "ate_rmse", "ate_mean", "ate_scale",
    "rpe_short_rmse", "rpe_long_rmse",
]

POOLED_KEYS: list[str] = [
    "lost_pct", "pre_init_frames",
    "avg_fps", "wall_time_s",
    "cpu_avg", "cpu_max", "ram_avg", "ram_max", "gpu_avg", "gpu_max",
    "map_points", "num_keyframes",
]

METRIC_KEYS: list[str] = ACCURACY_KEYS + POOLED_KEYS

# Absolute machine columns, derived from the percentage ones once the budget is
# known. Same pool suffixes as whatever they are derived from.
_DERIVED_FROM = {
    "ram_avg": "ram_avg_gb",
    "ram_max": "ram_max_gb",
    "cpu_avg": "cpu_busy_threads",
    "cpu_max": "cpu_busy_threads_max",
}
DERIVED_KEYS: list[str] = list(_DERIVED_FROM.values())

STAT_SUFFIXES = ("mean", "median", "std")
POOL_SUFFIXES = ("_all", "_succ")


def metric_pools(key: str) -> tuple[str, ...]:
    """Which populations a metric key is published for. Accuracy keys are
    completers-only; everything else gets both pools."""
    return ("_succ",) if key in ACCURACY_KEYS else POOL_SUFFIXES

SUCCESS_REASONS = {"completed", "max_frames"}
LOST_REASONS    = {"lost"}
TIMEOUT_REASONS = {"walltime"}

_STATUS_KEYS = ("success", "lost", "timeout", "oom", "error")

# ─── machine description ────────────────────────────────────────────────────
# RAM%/CPU% are relative to the WSL2 guest, not comparable across machines.
# Set via --ram-gb / --threads to also emit absolute GB / busy-thread columns.
RAM_BUDGET_GB: float | None = None
N_THREADS:     int   | None = None


def _status_from_flag(flag_row: dict | None) -> str:
    """Turn a run_flags.csv row into a status, for a run that left no dashboard.

      timeout                                -> TIMEOUT
      oom / failed / returncode -9 or -15    -> OOM   (RAM/SIGKILL/SIGTERM kills)
      no matching row / anything else        -> ERROR
    """
    if flag_row is None:
        return "error"
    s = (flag_row.get("status") or "").strip()
    if s == "timeout":
        return "timeout"
    try:
        rc = int(float(flag_row.get("returncode", "")))
    except (TypeError, ValueError):
        rc = None
    # -9 = kernel OOM SIGKILL, -15 = SIGTERM from an OOM daemon / external kill.
    if s in ("oom", "failed") or rc in (-9, -15):
        return "oom"
    return "error"


# Display order + colours for the per-variant line and the status plots.
_STATUS_DISPLAY = [
    ("success", "SUCCESS", "#2ca02c"),
    ("lost",    "LOST",    "#ff7f0e"),
    ("timeout", "TIMEOUT", "#1f77b4"),
    ("oom",     "OOM",     "#d62728"),
    ("error",   "ERROR",   "#7f7f7f"),
]
STATUS_COLOR = {k: c for k, _, c in _STATUS_DISPLAY}
STATUS_LABEL = {k: l for k, l, _ in _STATUS_DISPLAY}


# ─── generic helpers, shared with aggregate_all.py ──────────────────────────
# Live here (not in aggregate_all.py) so that file holds only cross-condition logic.

def rmtree_fresh(path: Path) -> Path:
    """Delete `path` if it exists and recreate it empty."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_cell(v: Any) -> Any:
    """Round floats for CSV output; pass everything else through untouched."""
    return round(v, 4) if isinstance(v, float) else v


def get_mpl():
    """Import matplotlib with the Agg backend and hand back pyplot. Deferred so
    importing this module in a headless/plot-free context stays cheap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def dedupe_columns(df):
    """Drop duplicate-named columns, keeping the LAST occurrence, so a freshly
    computed value always wins over a stale one loaded from disk (a duplicate
    name otherwise makes df[name] return a 2-col frame and crash pd.to_numeric)."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


def coerce_numeric(df, extra_prefixes: tuple[str, ...] = ("n_",)):
    """Coerce statistic/count columns to numeric dtype in place (empty CSV
    cells otherwise poison the column to object dtype)."""
    import pandas as pd
    for c in df.columns:
        if c.endswith(("_mean", "_median", "_std", "_best")) \
                or c.startswith(extra_prefixes):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def wilcoxon_note(a, b) -> str:
    """Paired Wilcoxon signed-rank -> short printable note, or "" if scipy is
    missing or there are too few pairs."""
    try:
        from scipy import stats as _st
    except ImportError:
        return ""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 6 or len(a) != len(b):
        return ""
    try:
        p = _st.wilcoxon(a, b).pvalue
    except ValueError:          # all differences zero
        return "(identical)"
    return f"(Wilcoxon p={p:.3g})"


# 0a.  dataset_summary.txt -> expected frame count per test -------------------------------
def load_expected_frames() -> dict[str, int]:
    """Parse docs/dataset/dataset_summary.txt -> {test_id: n_frames}."""
    summary = DOCS_DATASET / "dataset_summary.txt"
    out: dict[str, int] = {}
    if not summary.is_file():
        return out
    try:
        txt = summary.read_text(encoding="utf-8")
    except Exception:
        return out
    for m in re.finditer(r"(test\d+)\b.*?frames\s*:\s*(\d+)", txt, re.DOTALL):
        out[m.group(1)] = int(m.group(2))
    return out


# 0b.  run_flags.csv loader  (PRIMARY truth for no-dashboard runs) -------------------------
def _iso_epoch(s: str) -> float | None:
    try:
        return datetime.fromisoformat(s.strip()).timestamp()
    except (ValueError, AttributeError):
        return None


def _folder_epoch(name: str) -> float | None:
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def load_run_flags(results_root: Path) -> dict[tuple[str, str], list[dict]]:
    flags_path = results_root / "run_flags.csv"
    if not flags_path.is_file():
        return {}
    out: dict[tuple[str, str], list[dict]] = {}
    try:
        with open(flags_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_epoch"] = _iso_epoch(row.get("timestamp", ""))
                key = (row.get("variant", ""), row.get("test_id", ""))
                out.setdefault(key, []).append(row)
    except Exception:
        return {}
    for rows in out.values():
        rows.sort(key=lambda r: r["_epoch"] or 0.0)
    return out


# Folder name = run START time; run_flags row = run END time, so end >= start
# always holds. This slack only absorbs clock skew, it is not a search window.
_FLAG_MATCH_SLACK_S = 5.0


def match_run_flag(
    run_flags: dict[tuple[str, str], list[dict]],
    variant_full: str,
    test_id: str,
    folder_name: str,
) -> dict | None:
    """Pick the run_flags row for a folder: same (variant, test), the FIRST row
    ending at/after the folder's start. Anchored on "ends after start" rather
    than "nearest in time" -- nearest can tie on evenly-spaced runs and
    silently pick the PREVIOUS run's outcome instead."""
    rows = run_flags.get((variant_full, test_id))
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    fe = _folder_epoch(folder_name)
    if fe is None:
        return rows[-1]
    after = [r for r in rows if (r["_epoch"] or 0.0) >= fe - _FLAG_MATCH_SLACK_S]
    if after:
        return min(after, key=lambda r: r["_epoch"] or 0.0)
    # nothing ended after this folder started -> fall back to the nearest row
    return min(rows, key=lambda r: abs((r["_epoch"] or 0.0) - fe))


# 1.  Dashboard parsing --------------------------------------------------------------------
def parse_dashboard(dashboard_path: Path) -> dict[str, Any]:
    if not dashboard_path.is_file():
        return {}
    patterns = {
        "ate_rmse":        r"rmse\s*:\s*([\d.eE+-]+)",
        "ate_mean":        r"mean\s*:\s*([\d.eE+-]+)",
        "ate_scale":       r"scale\s*:\s*([\d.eE+-]+)",
        "rpe_short_rmse":  r"delta_t=0\.5.*?trans rmse\s*:\s*([\d.eE+-]+)",
        "rpe_long_rmse":   r"delta_t=2\.0.*?trans rmse\s*:\s*([\d.eE+-]+)",
        "lost_pct":        r"lost_frames\s*:.*?\(([\d.]+)%\)",
        "lost_frames":     r"lost_frames\s*:\s*(\d+)",
        "first_loss_frame": r"first_loss_at\s*:\s*frame\s*(\d+)",
        "tracked_frames":  r"tracked_frames\s*:\s*(\d+)",
        "pre_init_frames": r"pre_init_frames\s*:\s*(\d+)",
        "longest_streak":  r"longest_streak\s*:\s*(\d+)",
        "reloc_attempts":  r"reloc_attempts\s*:\s*(\d+)",
        "reloc_successes": r"reloc_successes\s*:\s*(\d+)",
        "map_points":      r"points_before_ba\s*:\s*([\d,]+)",
        # [\d,] not \d+: values are printed with thousands separators (e.g. "1,391").
        "num_keyframes":   r"(?:num_keyframes|keyframes_count|keyframes)\s*:\s*([\d,]+)",
        "processed":       r"processed\s*:\s*(\d+)",
        "total_frames":    r"total_frames\s*:\s*(\d+)",
        "avg_fps":         r"avg_fps\s*:\s*([\d.]+)",
        "wall_time_s":     r"wall_time\s*:\s*([\d.]+)",
        "cpu_avg":         r"cpu_avg\s*:\s*([\d.]+)",
        "cpu_max":         r"cpu_max\s*:\s*([\d.]+)",
        "ram_avg":         r"ram_avg\s*:\s*([\d.]+)",
        "ram_max":         r"ram_max\s*:\s*([\d.]+)",
        "gpu_avg":         r"gpu_avg\s*:\s*([\d.]+)",
        "gpu_max":         r"gpu_max\s*:\s*([\d.]+)",
    }
    try:
        content = dashboard_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    metrics: dict[str, Any] = {}
    for key, pat in patterns.items():
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                metrics[key] = float(raw)
            except ValueError:
                pass
    m = re.search(r"sensor\w*\s*:\s*(\S+)", content, re.IGNORECASE)
    if m:
        metrics["sensor_mode"] = m.group(1).strip()
    m = re.search(r"abort_reason\s*:\s*(\w+)", content, re.IGNORECASE)
    if m:
        metrics["abort_reason"] = m.group(1).strip().lower()
    return metrics


def _infer_abort_reason(metrics: dict[str, Any]) -> str:
    if "abort_reason" in metrics:
        return metrics["abort_reason"]
    return "lost" if metrics.get("lost_frames", 0) else "completed"


# 2.  Partial run recovery  (no dashboard, but incremental files survived) -----------------
#     progress.csv columns (index): unix_ts(0) elapsed_s(1) frame_id(2)
#       seq_timestamp(3) state(4) num_kf(5) num_map_pts(6) lost_streak(7)
#       total_lost(8) reloc_attempts(9) fps_approx(10) ram_pct(11) cpu_pct(12)
#       gpu_pct(13) ram_used_gb(14) gpu_mem_mb(15) gpu_power_w(16)
def read_progress_csv(progress_csv: Path):
    """Parse progress.csv -> (sequence, samples, info), or None if there's nothing.
    `samples` is a list[SystemSample]; `info` carries the frame counts, wall time,
    median fps and a tracking-quality dict already shaped the way write_dashboard() wants."""
    if not progress_csv.is_file():
        return None
    from src_pc.utils.metrics_helper_functions import SystemSample

    def _f(parts: list[str], idx: int) -> float | None:
        try:
            v = parts[idx].strip()
            return float(v) if v else None
        except (ValueError, IndexError):
            return None

    sequence = None
    samples: list = []
    fps_vals: list[float] = []
    ok_frames = lost_frames = 0
    last_fid = None        # highest frame_id seen (0-based) -> processed = last+1
    first_fid = None       # lowest  frame_id seen -> pre-init frames (see below)
    wall_time_s = None
    first_lost_frame = None
    longest_streak = 0
    reloc_attempts = 0
    map_points = None
    num_keyframes = None

    try:
        with open(progress_csv, newline="", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"):
                    m = re.search(r"sequence=(\S+)", line)
                    if m:
                        sequence = m.group(1)
                    continue
                parts = line.split(",")
                if len(parts) < 11 or parts[0] == "unix_ts":
                    continue
                state = parts[4].strip()
                fid   = _f(parts, 2)
                if fid is not None:
                    last_fid  = fid if last_fid  is None else max(last_fid, fid)
                    first_fid = fid if first_fid is None else min(first_fid, fid)
                if state == "OK":
                    ok_frames += 1
                elif state in ("LOST", "RELOCALIZE"):
                    lost_frames += 1
                    if first_lost_frame is None and fid is not None:
                        first_lost_frame = int(fid)
                streak = _f(parts, 7)
                if streak is not None:
                    longest_streak = max(longest_streak, int(streak))
                ra = _f(parts, 9)
                if ra is not None:
                    reloc_attempts = max(reloc_attempts, int(ra))
                mp = _f(parts, 6)
                if mp is not None:
                    map_points = int(mp)
                nkf = _f(parts, 5)
                if nkf is not None:
                    num_keyframes = int(nkf)
                fps = _f(parts, 10)
                if fps is not None:
                    fps_vals.append(fps)
                el = _f(parts, 1)
                if el is not None:
                    wall_time_s = el           # elapsed_s - last value wins
                ts = _f(parts, 0)
                if ts is not None:
                    samples.append(SystemSample(
                        t=ts,
                        cpu_percent=_f(parts, 12) or 0.0,
                        ram_percent=_f(parts, 11) or 0.0,
                        ram_used_gb=_f(parts, 14) or 0.0,
                        gpu_percent=_f(parts, 13),
                        gpu_mem_used_mb=_f(parts, 15),
                        gpu_power_w=_f(parts, 16),
                    ))
    except Exception:
        return None

    total = ok_frames + lost_frames
    if total == 0:
        return None

    # frame_id is 0-based and monotonic, so last+1 = frames processed (more
    # robust than counting rows, which can undercount if early rows never flushed).
    processed = int(last_fid) + 1 if last_fid is not None else total

    # slam_runner bumps img_id but `continue`s before logging while the map is
    # uninitialised, so the FIRST frame_id here = frames consumed pre-init.
    pre_init = int(first_fid) if first_fid is not None else None

    info = {
        "processed":       processed,
        "ok_frames":       ok_frames,
        "lost_frames":     lost_frames,
        "wall_time_s":     wall_time_s,
        "pre_init_frames": pre_init,
        "num_keyframes":   num_keyframes,
        "avg_fps":         float(np.median(fps_vals)) if fps_vals else None,
        "tracking_quality": {
            "tracked_frames":      ok_frames,
            "lost_frames":         lost_frames,
            "first_lost_frame":    first_lost_frame,
            "reloc_attempts":      reloc_attempts,
            "reloc_successes":     0,
            "longest_lost_streak": longest_streak,
            "map_points_count":    map_points,
            "num_keyframes":       num_keyframes,
            "pre_init_frames":     pre_init,
            "map_ply_saved":       False,
        },
    }
    return sequence, samples, info


# 2b.  progress.csv light scan  (backfill for dashboard-BACKED runs) -----------------------
def scan_progress_light(progress_csv: Path) -> dict[str, Any]:
    """Backfills fields write_dashboard() never printed: num_keyframes/map_points
    = last value in progress.csv col 5/6; pre_init_frames = first frame_id logged
    (pre-init frames leave no row). Builds no SystemSample objects, unlike
    read_progress_csv() -- this runs over every run in the tree, not just broken ones."""
    out: dict[str, Any] = {}
    if not progress_csv.is_file():
        return out
    first_fid = last_fid = None
    num_kf = map_pts = None
    try:
        with open(progress_csv, newline="", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 7 or parts[0] == "unix_ts":
                    continue
                try:
                    fid = int(float(parts[2]))
                except (ValueError, IndexError):
                    continue
                first_fid = fid if first_fid is None else min(first_fid, fid)
                last_fid  = fid if last_fid  is None else max(last_fid, fid)
                for idx, name in ((5, "kf"), (6, "mp")):
                    try:
                        v = parts[idx].strip()
                        if v:
                            if name == "kf":
                                num_kf = int(float(v))
                            else:
                                map_pts = int(float(v))
                    except (ValueError, IndexError):
                        pass
    except Exception:
        return out

    if first_fid is not None:
        out["pre_init_frames"] = float(first_fid)
    if last_fid is not None:
        out["processed"] = float(last_fid + 1)
    if num_kf is not None:
        out["num_keyframes"] = float(num_kf)
    if map_pts is not None:
        out["map_points"] = float(map_pts)
    return out


def backfill_from_progress(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """Fill gaps in a dashboard-derived metrics dict from progress.csv. Only
    fills keys that are missing/None -- never overwrites a real measured value."""
    wanted = ("pre_init_frames", "num_keyframes", "map_points", "processed")
    if all(metrics.get(k) is not None for k in wanted):
        return metrics
    scraped = scan_progress_light(run_dir / "files" / "progress.csv")
    for k in wanted:
        if metrics.get(k) is None and scraped.get(k) is not None:
            metrics[k] = scraped[k]
    return metrics


def _seq_to_test_drive(sequence: str | None, run_dir: Path) -> tuple[str | None, str | None]:
    """sequence == 'test2/2013_..._sync' -> (test_id, drive); fall back to path."""
    test_id = drive = None
    if sequence and "/" in sequence:
        test_id, drive = sequence.split("/", 1)
    if test_id is None:
        try:
            test_id = run_dir.parents[2].name   # .../<test>/<exp>/<variant>/<ts>
        except IndexError:
            pass
    return test_id, drive


def recover_run(run_dir: Path, compute_metrics: bool = False) -> dict[str, Any] | None:
    """Scrape whatever survived in <run>/files for a run with no dashboard.
    compute_metrics=False is cheap (frame counts/fps/system from progress.csv);
    True also loads trajectory + GT and computes ATE/RPE. None if nothing survived."""
    files_dir = run_dir / "files"
    parsed = read_progress_csv(files_dir / "progress.csv")
    if parsed is None:
        return None
    sequence, samples, info = parsed
    test_id, drive = _seq_to_test_drive(sequence, run_dir)

    rec: dict[str, Any] = {
        "is_partial": True,
        "sequence":   sequence,
        "test_id":    test_id,
        "drive":      drive,
        "samples":    samples,
        "ate": None, "rpe_short": None, "rpe_long": None,
        "gt": None, "pred": None, "sys_summary": None, "ate_rmse": None,
        **info,
    }
    rec["lost_pct"] = info["lost_frames"] / max(info["processed"], 1) * 100

    if not compute_metrics:
        return rec

    from src_pc.utils.metrics_helper_functions import (
        load_tum_trajectory, compute_ate, compute_rpe, summarize_system_metrics,
    )
    rec["sys_summary"] = summarize_system_metrics(samples)
    traj_online = files_dir / "trajectory_online.txt"
    if (traj_online.is_file() and traj_online.stat().st_size > 0
            and test_id and drive):
        gt_path = PROCESSED_POSES / test_id / f"{drive}_tum.txt"
        if gt_path.is_file():
            try:
                gt   = load_tum_trajectory(gt_path)
                pred = load_tum_trajectory(traj_online)
                rec["gt"], rec["pred"] = gt, pred
                rec["ate"]       = compute_ate(gt, pred, max_dt=0.05, with_scale=True)
                rec["rpe_short"] = compute_rpe(gt, pred, delta_t=0.5, max_dt=0.05)
                rec["rpe_long"]  = compute_rpe(gt, pred, delta_t=2.0, max_dt=0.05)
                rec["ate_rmse"]  = float(rec["ate"].rmse)
            except Exception:
                pass
    return rec


# 3.  Run-level classification -------------------------------------------------------------
def _attach_completion(metrics: dict[str, Any], expected: int | None) -> None:
    proc = metrics.get("processed")
    if expected and proc:
        metrics["expected_frames"] = expected
        metrics["completion_pct"]  = float(proc) / expected * 100


def classify_run(
    run_dir: Path,
    flag_row: dict | None,
    expected: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Classify one timestamped run directory -> (status, metrics_dict).

    `flag_row` is the run_flags.csv row we already matched to this folder
    (see match_run_flag), or None if nothing matched.
    """
    dashboard = run_dir / "dashboard.txt"

    # got a dashboard -> trust it, then top up the fields it never printed
    if dashboard.is_file():
        metrics = parse_dashboard(dashboard)
        if not metrics:
            return "error", {}
        reason = _infer_abort_reason(metrics)
        metrics["abort_reason"] = reason
        # pre_init_frames / num_keyframes were never written to dashboard.txt;
        # scrape them out of progress.csv without touching anything measured.
        backfill_from_progress(run_dir, metrics)
        _attach_completion(metrics, expected)
        if reason in SUCCESS_REASONS:
            return "success", metrics
        if reason in LOST_REASONS:
            return "lost", metrics
        if reason in TIMEOUT_REASONS:
            return "timeout", metrics
        return "error", metrics

    # no dashboard -> figure out why it died from run_flags.csv (primary source here)
    status = _status_from_flag(flag_row)

    # if a previous post-processing pass already built a fake dashboard, use its metrics.
    pp_dash = run_dir / "post_processing" / "dashboard.txt"
    if pp_dash.is_file():
        m = parse_dashboard(pp_dash)
        if m:
            m["is_partial"] = True
            m["recovered"]  = True
            backfill_from_progress(run_dir, m)
            _attach_completion(m, expected)
            return status, m

    # otherwise do the cheap recovery from progress.csv (frame counts / fps / system).
    rec = recover_run(run_dir, compute_metrics=False)
    if rec is not None:
        tq = rec.get("tracking_quality") or {}
        m = {
            "ate_rmse":         None,
            "avg_fps":          rec.get("avg_fps"),
            "wall_time_s":      rec.get("wall_time_s"),
            "lost_pct":         rec.get("lost_pct"),
            "processed":        rec.get("processed"),
            "first_loss_frame": tq.get("first_lost_frame"),
            "map_points":       tq.get("map_points_count"),
            "num_keyframes":    tq.get("num_keyframes"),
            "pre_init_frames":  tq.get("pre_init_frames"),
            "is_partial":       True,
            "recovered":        False,
        }
        backfill_from_progress(run_dir, m)
        _attach_completion(m, expected)
        return status, m

    return status, {}


# 4.  Aggregation --------------------------------------------------------------------------
def _run_dirs_for_variant(variant_dir: Path) -> list[Path]:
    return sorted(d for d in variant_dir.iterdir() if d.is_dir())


def _stats(vals: list[float]) -> tuple[Any, Any, Any]:
    """(mean, median, sample std). std needs >=2 points to mean anything."""
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(np.median(arr)), std


def aggregate_variant(
    variant_dir: Path,
    run_flags: dict[tuple[str, str], list[dict]],
    variant_full: str,
    test_id: str,
    expected: int | None = None,
) -> dict[str, Any]:
    """Roll up every run inside one variant folder into <key>_succ_* /
    <key>_all_* stat columns (see module docstring for the pool convention).
    Prefer the median over the mean when reading ATE: a couple of wild runs
    drag the mean around without moving the ranking."""
    run_dirs = _run_dirs_for_variant(variant_dir)
    counts: dict[str, int] = {k: 0 for k in _STATUS_KEYS}
    metric_runs: list[dict[str, Any]] = []   # any run that produced metrics
    succ_runs:   list[dict[str, Any]] = []   # completers only
    best_comp: float | None = None           # highest completion seen ...
    best_status: str | None = None           # ... and the status of that run

    for rd in run_dirs:
        flag_row = match_run_flag(run_flags, variant_full, test_id, rd.name)
        status, metrics = classify_run(rd, flag_row, expected)
        counts[status] += 1
        if metrics:
            metric_runs.append(metrics)
            if status == "success":
                succ_runs.append(metrics)
        c = metrics.get("completion_pct")
        if c is not None and (best_comp is None or c > best_comp):
            best_comp, best_status = c, status

    result: dict[str, Any] = {
        "n_total":       len(run_dirs),
        "n_metric_runs": len(metric_runs),
        "n_succ_runs":   len(succ_runs),
        "best_status":   best_status,
        **{f"n_{k}": counts[k] for k in _STATUS_KEYS},
    }

    pools = {"_all": metric_runs, "_succ": succ_runs}
    for key in METRIC_KEYS:
        for tag in metric_pools(key):
            vals = [m[key] for m in pools[tag] if m.get(key) is not None]
            mean, med, std = _stats(vals)
            result[f"{key}{tag}_mean"]   = mean
            result[f"{key}{tag}_median"] = med
            result[f"{key}{tag}_std"]    = std

    # completion keeps its plain name (no _succ_ variant -- that's 100% by construction).
    comp = [m["completion_pct"] for m in metric_runs
            if m.get("completion_pct") is not None]
    cmean, cmed, cstd = _stats(comp)
    result["completion_pct_mean"]   = cmean
    result["completion_pct_median"] = cmed
    result["completion_pct_std"]    = cstd
    result["completion_pct_best"]   = best_comp

    # ─── machine-comparable derived columns ─────────────────────────────────
    # RAM%/CPU% don't travel between machines; with the budget known, emit the
    # absolute GB/thread-count columns instead (same pool suffixes as source).
    result["ram_budget_gb"] = RAM_BUDGET_GB
    result["n_threads"]     = N_THREADS
    for src, dst in _DERIVED_FROM.items():
        scale = RAM_BUDGET_GB if dst.endswith("_gb") else N_THREADS
        if not scale:
            continue
        for tag in metric_pools(src):
            for stat in STAT_SUFFIXES:
                v = result.get(f"{src}{tag}_{stat}")
                result[f"{dst}{tag}_{stat}"] = (v / 100.0 * scale
                                                if v is not None else None)
    return result


def collect_all_data(
    results_root: Path,
    test_ids: list[str],
    exp_types: list[str],
    variants_dict: dict[str, dict] = VARIANTS,
    run_flags: dict[tuple[str, str], list[dict]] | None = None,
    expected_frames: dict[str, int] | None = None,
) -> dict:
    """Walk the filesystem -> data[test_id][exp_type] = [row_dict, ...].
    `results_root` is the mode sub-tree (results/VO or results/SLAM);
    `run_flags` is loaded once from the global results root and passed in."""
    if run_flags is None:
        run_flags = load_run_flags(results_root.parent)
    if expected_frames is None:
        expected_frames = load_expected_frames()
    data: dict = {}
    for test_id in test_ids:
        test_dir = results_root / test_id
        if not test_dir.is_dir():
            continue
        data[test_id] = {}
        expected = expected_frames.get(test_id)
        for exp_type in exp_types:
            exp_dir = test_dir / exp_type
            if not exp_dir.is_dir():
                continue
            rows = []
            for variant_dir in sorted(exp_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                short_name = variant_dir.name
                vname = next(
                    (k for k, v in variants_dict.items()
                     if v.get("short_name") == short_name
                     and v.get("experiment_type") == exp_type),
                    short_name,
                )
                sensor = "mono" if exp_type in MONO_EXPS else "stereo"
                agg = aggregate_variant(variant_dir, run_flags,
                                        vname, test_id, expected)
                if agg["n_total"] == 0:
                    continue
                rows.append({"variant": vname, "short_name": short_name,
                             "sensor": sensor, "exp_type": exp_type, **agg})
            if rows:
                data[test_id][exp_type] = rows
    return data


# 5.  Console summary ----------------------------------------------------------------------
def _status_totals(data: dict) -> dict[str, int]:
    totals = {k: 0 for k in _STATUS_KEYS}
    for exp_map in data.values():
        for rows in exp_map.values():
            for r in rows:
                for k in _STATUS_KEYS:
                    totals[k] += r.get(f"n_{k}", 0)
    return totals


def print_summary(data: dict) -> None:
    n_variants = 0
    for test_id, exp_map in data.items():
        print(f"\n -- {test_id} ---------------------------------------------")
        for exp_type, rows in exp_map.items():
            print(f"  {EXP_SHORT.get(exp_type, exp_type)}")
            for r in rows:
                n_variants += 1
                ran = r["n_success"] + r["n_lost"] + r["n_timeout"]
                tag = ("OK " if r["n_success"] == r["n_total"]
                       else " ~ " if ran > 0 else " X ")
                comp   = r.get("completion_pct_mean")
                comp_s = f" C={comp:.0f}%" if comp is not None else ""
                # median over completers is the honest accuracy figure
                ate    = r.get("ate_rmse_succ_median")
                ate_s  = f" ATE={ate:.3f}m" if ate is not None else ""
                fps    = r.get("avg_fps_all_median")
                fps_s  = f" {fps:.1f}fps" if fps is not None else ""
                kf     = r.get("num_keyframes_all_median")
                kf_s   = f" kf={kf:.0f}" if kf is not None else ""
                print(f"   {tag} {r['variant']:<36}"
                      f" s={r['n_success']} l={r['n_lost']}"
                      f" t={r['n_timeout']} o={r['n_oom']} e={r['n_error']}"
                      f"{comp_s}{ate_s}{fps_s}{kf_s}")

    totals = _status_totals(data)
    tot = sum(totals.values())
    print(f"\n {'-'*58}")
    print(f" variants:{n_variants}  runs:{tot}")
    print("  " + "  ".join(
        f"{label} {totals[k]} ({100*totals[k]/max(tot,1):.0f}%)"
        for k, label, _ in _STATUS_DISPLAY))


# 5b.  Post-processing  (always run) -------------------------------------------------------
def post_process_run(
    run_dir: Path,
    variant_full: str,
    abort_reason: str,
    expected: int | None,
) -> str | None:
    """Build <run>/post_processing for one dashboard-less run.
    Returns a short status string, or None if there was nothing worth recovering."""
    import contextlib, io

    rec = recover_run(run_dir, compute_metrics=True)
    if rec is None:
        return None

    pp        = run_dir / "post_processing"
    files_dir = pp / "files"
    plots_dir = pp / "plots"
    seq       = rec.get("sequence") or f"{rec.get('test_id')}/{rec.get('drive')}"
    proc      = rec.get("processed") or 0
    ate       = rec.get("ate")
    note      = (f"ATE={ate.rmse:.3f}m " if ate is not None else "")

    from src_pc.utils.printing_helper_functions import (
        write_dashboard, write_ate_detailed, write_rpe_detailed,
        plot_run_trajectory, plot_system_usage,
    )
    pp.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(pp / "dashboard", ignore_errors=True)  # drop legacy nesting
    for d in (files_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    tq = rec.get("tracking_quality") or {}

    # the helper writers print a lot - swallow their stdout so the summary stays readable.
    with contextlib.redirect_stdout(io.StringIO()):
        write_dashboard(
            pp / "dashboard.txt",
            sequence=seq, variant=variant_full,
            total_frames=expected or proc, processed=proc,
            wall_time_s=rec.get("wall_time_s") or 0.0,
            avg_fps=rec.get("avg_fps") or 0.0,
            map_points=tq.get("map_points_count"),
            sys_summary=rec.get("sys_summary") or {},
            ate=ate, rpe_short=rec.get("rpe_short"), rpe_long=rec.get("rpe_long"),
            tracking_quality=tq, abort_reason=abort_reason,
        )
        if ate is not None and rec.get("gt") is not None:
            write_ate_detailed(files_dir / "ate_errors.txt", ate, rec["gt"])
        if rec.get("rpe_short") is not None:
            write_rpe_detailed(files_dir / "rpe_short_errors.txt", rec["rpe_short"])
        if rec.get("rpe_long") is not None:
            write_rpe_detailed(files_dir / "rpe_long_errors.txt", rec["rpe_long"])
        try:
            if rec.get("gt") is not None and rec.get("pred") is not None:
                plot_run_trajectory(rec["gt"], rec["pred"],
                                    plots_dir / "trajectory.png",
                                    title=f"{variant_full} - {seq} ({abort_reason})",
                                    ate=ate)
            if rec.get("samples"):
                plot_system_usage(rec["samples"], plots_dir / "system_usage.png",
                                  title=f"{variant_full} - {seq}")
        except Exception as e:
            print(f"    [!] plot failed for {run_dir.name}: {e}")

    # relative_to() can raise if --root was relative; not worth crashing a log line over.
    try:
        where = pp.resolve().relative_to(PC_ROOT)
    except ValueError:
        where = pp
    return (f"{abort_reason.upper():7} {proc}/{expected or '?'}fr "
            f"{note}-> {where}")


def run_post_processing(
    scan_root: Path,
    test_ids: list[str],
    exp_types: list[str],
    run_flags: dict[tuple[str, str], list[dict]],
    expected_frames: dict[str, int],
    variants_dict: dict = VARIANTS,
) -> None:
    """Walk the tree; post-process every dashboard-less run with recoverable data.
    Every post_processing/ folder is deleted and rebuilt every pass -- no
    "already done" skip, so a stale recovered dashboard can never linger."""
    print(f"\n{'-'*58}\n Rebuilding post_processing/ for dashboard-less runs\n{'-'*58}")
    n_done = n_skip = 0
    for test_id in test_ids:
        test_dir = scan_root / test_id
        if not test_dir.is_dir():
            continue
        expected = expected_frames.get(test_id)
        for exp_type in exp_types:
            exp_dir = test_dir / exp_type
            if not exp_dir.is_dir():
                continue
            for variant_dir in sorted(exp_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                vname = next(
                    (k for k, v in variants_dict.items()
                     if v.get("short_name") == variant_dir.name
                     and v.get("experiment_type") == exp_type),
                    variant_dir.name,
                )
                for rd in _run_dirs_for_variant(variant_dir):
                    if (rd / "dashboard.txt").is_file():
                        continue   # real dashboard - nothing to recover
                    # wipe any earlier recovery so this pass is the only author
                    shutil.rmtree(rd / "post_processing", ignore_errors=True)
                    flag_row = match_run_flag(run_flags, vname, test_id, rd.name)
                    status = _status_from_flag(flag_row)
                    msg = post_process_run(rd, vname, status, expected)
                    if msg is None:
                        n_skip += 1
                    else:
                        n_done += 1
                        print(f"   {test_id}/{exp_type}/{variant_dir.name}: {msg}")
    print(f"\n rebuilt {n_done} recovery folder(s), "
          f"skipped {n_skip} (no surviving data).")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
# Architecture families. An unmatched variant is reported and shown as
# "unknown" rather than silently defaulted to "classical".
_FAMILY_COLORS = {
    "classical":   "#2a9d8f",
    "CNN":         "#1d6fb8",
    "transformer": "#e08a1e",
    "unknown":     "#9e9e9e",
}
_FAMILY_ORDER = ["classical", "CNN", "transformer", "unknown"]

_FAMILY_SETS: dict[str, dict[str, set[str]]] = {
    _FE: {
        "classical":   {"orb", "orb2", "sift", "root_sift", "brisk", "akaze",
                        "kaze", "baseline"},
        "CNN":         {"superpoint", "disk", "aliked", "keynet", "r2d2",
                        "d2net", "xfeat", "xfeat_xfeat", "liftfeat"},
        # NOTE: the short_name is xfeat_lightglue, not xfeat_lg. The old table
        # had the wrong string, so this variant was being counted as classical.
        "transformer": {"lightglue", "xfeat_lightglue", "xfeat_lg", "rdd", "dedode"},
    },
    _LC: {
        "classical":   {"dbow2", "dbow3", "vlad", "ibow", "obindex2", "sad"},
        "CNN":         {"netvlad", "cosplace", "eigenplaces", "alexnet"},
        "transformer": {"megaloc", "cricavpr", "salad", "supervlad"},
    },
    _BE: {
        "classical":   {"gtsam", "g2o", "baseline"},
        "CNN":         set(),
        "transformer": set(),
    },
    _DE: {
        # the varied component is the depth network; DepthAnything is a ViT.
        "classical":   set(),
        "CNN":         set(),
        "transformer": {"orb2_dav2", "orb2_dav3",
                        "superpoint_dav2", "superpoint_dav3",
                        "sp_dav2", "sp_dav3"},
    },
}

_UNFAMILIED: set[tuple[str, str]] = set()   # reported once at the end of a run


def _family(short_name: str, exp_type: str) -> str:
    """short_name -> "classical" / "CNN" / "transformer" / "unknown".
    A phase with its own table (fe, lc, be, de) matches directly; "combined"
    splits on "_" and matches against the FE table so it's coloured by its FE
    backbone. Anything matching nothing is flagged, never defaulted."""
    fam_map = _FAMILY_SETS.get(exp_type)
    if fam_map is not None:
        for fam, names in fam_map.items():
            if short_name in names:
                return fam
    tokens = [short_name, *short_name.split("_")]
    for et in (_FE, _LC):
        for fam, names in _FAMILY_SETS[et].items():
            if any(tok in names for tok in tokens):
                return fam
    _UNFAMILIED.add((short_name, exp_type))
    return "unknown"


def report_unfamilied() -> None:
    if _UNFAMILIED:
        print("\n [!] variants with no architecture family (shown as 'unknown'):")
        for short, exp in sorted(_UNFAMILIED):
            print(f"       {short:<24} ({exp})   <- add it to _FAMILY_SETS")


def _display_status(row: dict) -> str:
    """The single definition of a cell's colour, used by every figure: the
    BEST run's status (so a 96%-then-OOM run reads OOM, not a more frequent but
    lower-completion LOST run). Falls back to the modal status if no completion
    was recorded."""
    st = row.get("best_status")
    if st:
        return st
    best, best_n = "error", 0
    for k, _, _ in _STATUS_DISPLAY:
        n = row.get(f"n_{k}", 0)
        if n > best_n:
            best, best_n = k, n
    return best


def _tally(row: dict) -> str:
    """Compact per-cell run tally, e.g. 's2 o1'."""
    bits = [f"{k[0]}{row.get(f'n_{k}', 0)}" for k in _STATUS_KEYS
            if row.get(f"n_{k}", 0)]
    return " ".join(bits)


def _annotate_no_overlap(ax, texts):
    """adjustText when available, plain offsets otherwise."""
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55"))
    except Exception:
        pass


# 5c.  Per-mode summary directory  (results/summary/<mode>/) -------------------------------
def _per_test_status(data: dict) -> dict[str, dict[str, int]]:
    """{test_id: {status: count}} from the collected data."""
    out: dict[str, dict[str, int]] = {}
    for test_id, exp_map in data.items():
        counts = {k: 0 for k in _STATUS_KEYS}
        for rows in exp_map.values():
            for r in rows:
                for k in _STATUS_KEYS:
                    counts[k] += r.get(f"n_{k}", 0)
        out[test_id] = counts
    return out


def plot_status_matrix(data: dict, out_png: Path, mode: str,
                       scan_root: Path | None = None,
                       run_flags: dict | None = None,
                       expected_frames: dict[str, int] | None = None,
                       only_exp: str | None = None) -> None:
    """Per-algorithm outcome grid: variant x test, coloured by the best run's
    status, annotated with the run tally and best completion %. The shared
    baseline (<test>/baseline/) is shown as the first row for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    tests = sorted(data.keys())
    cells: dict[tuple, dict] = {}   # (exp, variant) -> {test: row}
    for test in tests:
        for exp, rows in data[test].items():
            for r in rows:
                cells.setdefault((exp, r["variant"]), {})[test] = r

    base_key = (BASELINE_VARIANT, BASELINE_VARIANT)
    if scan_root is not None:
        base_cells: dict[str, dict] = {}
        for test in tests:
            bdir = scan_root / test / BASELINE_VARIANT
            if not bdir.is_dir():
                continue
            agg = aggregate_variant(bdir, run_flags or {}, BASELINE_VARIANT,
                                    test, (expected_frames or {}).get(test))
            if agg["n_total"]:
                base_cells[test] = {"variant": BASELINE_VARIANT, **agg}
        if base_cells:
            cells[base_key] = base_cells

    exp_order = list(EXPERIMENT_TYPES)
    rest = sorted((k for k in cells if k != base_key
                   and (only_exp is None or k[0] == only_exp)),
                  key=lambda k: (exp_order.index(k[0]) if k[0] in exp_order
                                 else 99, k[1]))
    keys = ([base_key] if base_key in cells else []) + rest
    if not keys or (only_exp is not None and not rest):
        return
    labels = [k[1] for k in keys]
    sidx   = {k: i for i, (k, _, _) in enumerate(_STATUS_DISPLAY)}
    cmap   = ListedColormap([c for _, _, c in _STATUS_DISPLAY])
    cmap.set_bad("white")

    M = np.full((len(keys), len(tests)), np.nan)
    for ri, key in enumerate(keys):
        for ci, test in enumerate(tests):
            row = cells[key].get(test)
            if row is not None:
                M[ri, ci] = sidx[_display_status(row)]

    fig, ax = plt.subplots(figsize=(2.6 + 1.7 * len(tests), 0.32 * len(keys) + 1.9))
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(_STATUS_DISPLAY) - 0.5)

    # annotate: <tally> | <best run's completion %>
    for ri, key in enumerate(keys):
        for ci, test in enumerate(tests):
            row = cells[key].get(test)
            if row is None:
                continue
            best = row.get("completion_pct_best")
            best_s = f"{best:.0f}%" if best is not None else "-"
            ax.text(ci, ri, f"{_tally(row)} | {best_s}",
                    ha="center", va="center", fontsize=5.6, color="white")

    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.set_xticks(np.arange(-.5, len(tests), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)

    scope = f"  -  {EXP_SHORT.get(only_exp, only_exp)} vs baseline" if only_exp else ""
    ax.set_title(f"{mode} - per-algorithm outcomes{scope}\n"
                 f"colour = best run status   |   cell = run tally | best completion %",
                 pad=34)
    handles = [Patch(facecolor=c, label=l) for _, l, c in _STATUS_DISPLAY]
    ax.legend(handles=handles, ncol=len(_STATUS_DISPLAY),
              loc="lower center", bbox_to_anchor=(0.5, 1.005), frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_status_overview(data: dict, out_png: Path, mode: str) -> None:
    """Stacked bar of run outcomes per test (+ ALL): completed vs the rest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_test = _per_test_status(data)
    tests    = sorted(per_test.keys())
    totals   = {k: sum(per_test[t][k] for t in tests) for k in _STATUS_KEYS}
    cols     = tests + ["ALL"]
    counts   = [per_test[t] for t in tests] + [totals]

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 5))
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    for key, label, colour in _STATUS_DISPLAY:
        vals = np.array([c.get(key, 0) for c in counts], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=label, color=colour, width=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, int(v), ha="center", va="center",
                        fontsize=8, color="white")
        bottom += vals

    grand = int(bottom[-1]) if len(bottom) else 0
    succ  = totals["success"]
    ax.set_xticks(x); ax.set_xticklabels(cols)
    ax.set_ylabel("runs")
    ax.set_title(f"{mode} - run outcomes   "
                 f"({succ}/{grand} completed, {grand - succ} did not)")
    ax.legend(ncol=len(_STATUS_DISPLAY), loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Failure-space maps -----------------------------------------------------------------------
_MAP_BENCHES = [("feature_extraction", "fe"),
                ("loop_closure",       "lc"),
                ("combined",           "co")]


def _bench_fail_points(
    scan_root: Path, test_id: str, exp_type: str,
    run_flags: dict, expected: int | None, variants_dict: dict,
) -> list[tuple[int, str]]:
    """(fail_frame, status) for every non-completed run of a bench in a test.
    A LOST run is marked where tracking first broke; an OOM/TIMEOUT run is
    marked where it DIED (last processed frame), not at an early stumble it
    may have recovered from."""
    pts: list[tuple[int, str]] = []
    exp_dir = scan_root / test_id / exp_type
    if not exp_dir.is_dir():
        return pts
    for variant_dir in sorted(exp_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        vname = next(
            (k for k, v in variants_dict.items()
             if v.get("short_name") == variant_dir.name
             and v.get("experiment_type") == exp_type),
            variant_dir.name,
        )
        for rd in _run_dirs_for_variant(variant_dir):
            flag_row = match_run_flag(run_flags, vname, test_id, rd.name)
            status, m = classify_run(rd, flag_row, expected)
            if status == "success" or not m:
                continue
            if status == "lost":
                frame = m.get("first_loss_frame") or m.get("processed")
            else:
                frame = m.get("processed") or m.get("first_loss_frame")
            if frame:
                pts.append((int(frame), status))
    return pts


def plot_failure_maps(
    scan_root: Path, test_ids: list[str],
    run_flags: dict, expected_frames: dict[str, int], variants_dict: dict,
    summary_dir: Path,
) -> None:
    """One figure per bench (fe / lc / co), all tests side by side. Dots are
    coloured by status so lost-tracking and memory deaths stay distinguishable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src_pc.utils.metrics_helper_functions import load_tum_trajectory

    for exp, short in _MAP_BENCHES:
        active = [t for t in test_ids if (scan_root / t / exp).is_dir()]
        if not active:
            continue

        fig, axes = plt.subplots(1, len(active), squeeze=False,
                                 figsize=(5.2 * len(active), 6.0))

        for ax, test_id in zip(axes[0], active):
            expected = expected_frames.get(test_id)
            gt_files = list((PROCESSED_POSES / test_id).glob("*_tum.txt"))
            if not gt_files:
                ax.set_title(f"{test_id}\n(no GT)")
                ax.axis("off")
                continue

            xyz = load_tum_trajectory(gt_files[0]).xyz
            n   = len(xyz)
            a, b = sorted(np.argsort(xyz.var(axis=0))[::-1][:2])

            pts = _bench_fail_points(scan_root, test_id, exp,
                                     run_flags, expected, variants_dict)

            ax.plot(xyz[:, a], xyz[:, b], color="0.78", lw=1.2, zorder=1)
            ax.scatter(xyz[0, a], xyz[0, b], c="green", s=45, zorder=4)

            annotation = ""
            if pts:
                for st in ("lost", "timeout", "oom", "error"):
                    fr = [f for f, s in pts if s == st]
                    if not fr:
                        continue
                    ax.scatter([xyz[min(f, n - 1), a] for f in fr],
                               [xyz[min(f, n - 1), b] for f in fr],
                               c=STATUS_COLOR[st], s=80, alpha=0.35,
                               edgecolors="none", zorder=3)

                farr  = np.array([f for f, _ in pts], dtype=float)
                total = expected or int(farr.max()) or 1
                counts, edges = np.histogram(farr, bins=max(3, int(total / 100)),
                                             range=(0, total))
                ranked = np.argsort(counts)[::-1]
                for bi in [i for i in ranked if counts[i] >= 3][:1]:
                    center = float((edges[bi] + edges[bi + 1]) / 2)
                    pct_s  = f" ({center / total * 100:.0f}%)" if expected else ""
                    annotation = f"cluster: {int(counts[bi])} ~ fr{center:.0f}{pct_s}"
                    ci = min(int(round(center)), n - 1)
                    ax.scatter(xyz[ci, a], xyz[ci, b], c="royalblue", s=120,
                               zorder=5, edgecolors="white", linewidths=0.8)

            ax.set_title(f"{test_id}  -  {len(pts)} failures")
            ax.set_aspect("equal", "datalim")
            ax.set_xticks([]); ax.set_yticks([])
            if annotation:
                ax.set_xlabel(annotation, fontsize=8, labelpad=4)

        handles = [Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor=STATUS_COLOR[s], markeredgecolor="none",
                          alpha=0.6, label=STATUS_LABEL[s])
                   for s in ("lost", "timeout", "oom")]
        fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False)
        fig.suptitle(f"{scan_root.name}  {short.upper()}  -  failure-space map "
                     f"(dot = where the run ended)")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        bench_out = summary_dir / short
        bench_out.mkdir(parents=True, exist_ok=True)
        fig.savefig(bench_out / f"failure_map_{short}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


_ALL_BENCHES = [
    ("feature_extraction", "fe"),
    ("loop_closure",       "lc"),
    ("backend",            "be"),
    ("depth_estimation",   "de"),
    ("combined",           "co"),
]

# (data_key_in_agg_row, file_slug, y-axis_label, higher_is_better)
# Machine load is charted over ALL runs -- an OOM-killed run's load at death is real data.
_SYS_CHART_METRICS = [
    ("cpu_avg_all_mean", "cpu",      "CPU %",      False),
    ("ram_avg_all_mean", "ram",      "RAM %",      False),
    ("ram_max_all_mean", "ram_peak", "peak RAM %", False),
    ("gpu_avg_all_mean", "gpu",      "GPU %",      False),
]
# Accuracy uses the completers-only median (mean is dominated by a few wild runs).
_PERF_CHART_METRICS = [
    ("ate_rmse_succ_median",       "ate_rmse",   "ATE RMSE (m) - completers",  False),
    ("rpe_short_rmse_succ_median", "rpe_short",  "RPE short RMSE (m)",         False),
    ("rpe_long_rmse_succ_median",  "rpe_long",   "RPE long RMSE (m)",          False),
    ("avg_fps_all_median",         "fps",        "avg FPS",                    True),
    ("completion_pct_best",        "completion", "best completion %",          True),
    ("wall_time_s_succ_median",    "wall_time",  "wall time (s) - completers", False),
    ("map_points_all_median",      "map_points", "map points",                 True),
    ("num_keyframes_all_median",   "keyframes",  "keyframes",                  True),
    ("pre_init_frames_all_median", "pre_init",   "frames before init",         False),
]


def _quadrant_ylim(max_val: float) -> float:
    """Round max_val up to the next 20-point quadrant (min 20)."""
    return max(20.0, float(np.ceil(max(max_val, 1e-9) / 20.0)) * 20.0)


def _fmt_val(v: float) -> str:
    av = abs(v)
    if av >= 100:  return f"{v:.0f}"
    if av >= 10:   return f"{v:.1f}"
    if av >= 1:    return f"{v:.2f}"
    return f"{v:.3f}"


def _variant_bar_chart(
    rows: list[dict], data_key: str, axis_label: str,
    out_path: Path, title: str,
    y_lim: tuple[float, float] | None = None,
    higher_is_better: bool | None = None,
    quadrant_ylim: bool = False,
    autoscale: bool = False,
    border_ties: bool = True,
    yerr_key: str | None = None,
    csv_path: Path | None = None,
    sort_by_value: bool = True,
) -> None:
    """One PNG: variant names on X, metric value on Y, bar colour = status.
    sort_by_value orders bars best-to-worst (needs higher_is_better). yerr is
    clipped at zero, since ATE std can exceed the mean. Truncated axes (from
    `autoscale`) are labelled so a small spread isn't exaggerated silently."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pairs = [(r["variant"], r.get(data_key), _display_status(r),
              r.get("n_success", 0) > 0, r.get("completion_pct_best"),
              r.get(yerr_key) if yerr_key else None)
             for r in rows if r.get(data_key) is not None]
    if not pairs:
        return
    if sort_by_value and higher_is_better is not None:
        pairs.sort(key=lambda p: p[1], reverse=bool(higher_is_better))

    names, vals, statuses, completed, comp_best, yerrs = zip(*pairs)
    vals_arr = np.array(vals, dtype=float)
    colors = [STATUS_COLOR.get(s or "error", "#7f7f7f") for s in statuses]

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["variant", data_key, "status", "completion_pct_best"])
            for name, val, st, cb in zip(names, vals, statuses, comp_best):
                w.writerow([name, val, st or "", cb])

    fig_w = max(6.0, len(names) * 0.75 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    x = np.arange(len(names))

    yerr_arr = None
    if yerr_key:
        e = np.array([v if v is not None else 0.0 for v in yerrs], dtype=float)
        # clip the lower whisker so it never crosses zero
        yerr_arr = np.vstack([np.minimum(e, np.maximum(vals_arr, 0.0)), e])

    bars = ax.bar(x, vals_arr, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.4,
                  yerr=yerr_arr, capsize=3,
                  error_kw={"ecolor": "0.3", "elinewidth": 0.8})

    # black border on the best bar(s), restricted to variants that finished a
    # run (a lost/oom run's short wall time must never "win").
    if higher_is_better is not None and len(vals_arr):
        comp_idxs = [i for i, done in enumerate(completed) if done]
        if comp_idxs:
            comp_vals = vals_arr[comp_idxs]
            best_val  = float(np.max(comp_vals) if higher_is_better else np.min(comp_vals))
            tol       = max(abs(best_val) * 1e-6, 1e-9)
            best_idxs = [i for i in comp_idxs if abs(vals_arr[i] - best_val) <= tol]
            if border_ties or len(best_idxs) == 1:
                for bi in best_idxs:
                    bars[int(bi)].set_edgecolor("black")
                    bars[int(bi)].set_linewidth(3.0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7,
                       family="monospace")

    truncated = False
    if quadrant_ylim and len(vals_arr):
        lo, hi = 0.0, _quadrant_ylim(float(np.nanmax(vals_arr)))
    elif y_lim:
        lo, hi = y_lim
    elif autoscale and len(vals_arr):
        vmin, vmax = float(np.nanmin(vals_arr)), float(np.nanmax(vals_arr))
        span = vmax - vmin
        if vmin > 0 and 0 < span < 0.25 * vmax:      # clustered above zero -> zoom
            pad = span * 0.6
            lo, hi = max(0.0, vmin - pad), vmax + pad
            truncated = lo > 0
        else:
            lo, hi = 0.0, vmax * 1.12 if vmax > 0 else 1.0
    else:
        lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.08)

    if truncated:
        ax.text(0.005, 0.015, "y-axis truncated (does not start at 0)",
                transform=ax.transAxes, fontsize=6, style="italic", color="0.35")

    ax.set_ylabel(axis_label, fontsize=9)
    ax.set_title(title, fontsize=10)

    for xi, v in zip(x, vals_arr):
        ax.annotate(_fmt_val(float(v)), (xi, v), textcoords="offset points",
                    xytext=(0, 2), ha="center", va="bottom", fontsize=6)

    # completion % at the base of non-completed bars, since their metric isn't like-for-like.
    for xi, (done, cb) in enumerate(zip(completed, comp_best)):
        if not done and cb is not None:
            ax.annotate(f"{cb:.0f}%", (xi, lo), textcoords="offset points",
                        xytext=(0, 2), ha="center", va="bottom", fontsize=6,
                        color="white", fontweight="bold")

    present = sorted({s for s in statuses if s},
                     key=lambda k: list(_STATUS_KEYS).index(k))
    ax.legend(handles=[Patch(facecolor=STATUS_COLOR[k], label=STATUS_LABEL[k])
                       for k in present],
              loc="best", fontsize=7, framealpha=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_system_usage_charts(data: dict, test_ids: list[str], summary_dir: Path) -> None:
    """benches x tests x machine metrics -> machine_usage/ (PNG) + csv/ (CSV)."""
    metrics = list(_SYS_CHART_METRICS)
    if RAM_BUDGET_GB:   # absolute GB is the comparable quantity, so plot it too
        metrics += [("ram_max_gb_all_mean", "ram_peak_gb",
                     f"peak RAM (GB of {RAM_BUDGET_GB:g})", False)]
    if N_THREADS:
        metrics += [("cpu_busy_threads_all_mean", "cpu_threads",
                     f"busy threads (of {N_THREADS})", False)]
    for exp_full, exp_short in _ALL_BENCHES:
        img_dir = summary_dir / exp_short / "machine_usage"
        csv_dir = summary_dir / exp_short / "csv"
        for test_id in test_ids:
            rows = data.get(test_id, {}).get(exp_full, [])
            if not rows:
                continue
            for data_key, slug, label, hib in metrics:
                _variant_bar_chart(
                    rows, data_key, label,
                    img_dir / f"{test_id}_{slug}.png",
                    f"{exp_short.upper()}  -  {test_id}  -  {label}",
                    higher_is_better=hib,
                    quadrant_ylim="%" in label,
                    csv_path=csv_dir / f"{test_id}_{slug}.csv",
                )


def plot_performance_charts(data: dict, test_ids: list[str], summary_dir: Path) -> None:
    """benches x tests x perf metrics -> performance/ (PNG) + csv/ (CSV)."""
    for exp_full, exp_short in _ALL_BENCHES:
        img_dir = summary_dir / exp_short / "performance"
        csv_dir = summary_dir / exp_short / "csv"
        for test_id in test_ids:
            rows = data.get(test_id, {}).get(exp_full, [])
            if not rows:
                continue
            for data_key, slug, label, hib in _PERF_CHART_METRICS:
                is_pct = "%" in label
                _variant_bar_chart(
                    rows, data_key, label,
                    img_dir / f"{test_id}_{slug}.png",
                    f"{exp_short.upper()}  -  {test_id}  -  {label}",
                    y_lim=(0, 100) if is_pct else None,
                    autoscale=not is_pct,
                    higher_is_better=hib,
                    border_ties=(slug != "completion"),
                    yerr_key=("ate_rmse_succ_std" if slug == "ate_rmse" else None),
                    csv_path=csv_dir / f"{test_id}_{slug}.csv",
                )


# 5c-bis.  Cross-phase synthesis figures  (summary/.../trends/) ----------------------------
def _gather_variant_rows(data: dict, exp_type: str | None = None) -> dict[str, list[dict]]:
    """{variant: [row, ...]} across every test, each row carrying its test_id."""
    out: dict[str, list[dict]] = {}
    for test_id, exp_map in data.items():
        for et, rows in exp_map.items():
            if exp_type is not None and et != exp_type:
                continue
            for r in rows:
                out.setdefault(r["variant"], []).append({"test_id": test_id, **r})
    return out


def _variant_robustness(variant_rows: list[dict], n_tests: int,
                        reps_per_test: int | None = None) -> dict:
    """Two robustness numbers, since one alone is misleading:
    attempted_rate = successes / runs actually attempted (rewards skipping hard
    sequences). coverage_rate = successes / runs it would have had if it ran
    everything (skipped sequences count as failures) -- the honest number."""
    attempted = {r.get("test_id") for r in variant_rows if r.get("test_id")}
    n_total   = sum(r.get("n_total", 0) for r in variant_rows)
    n_success = sum(r.get("n_success", 0) for r in variant_rows)
    reps = reps_per_test or (max((r.get("n_total", 0) for r in variant_rows),
                                 default=0) or 1)
    denom_full = reps * n_tests
    return {
        "n_success":      n_success,
        "n_total":        n_total,
        "n_attempted":    len(attempted),
        "attempted_rate": (n_success / n_total * 100) if n_total else 0.0,
        "coverage_rate":  (n_success / denom_full * 100) if denom_full else 0.0,
        "denom_full":     denom_full,
    }


def plot_tradeoff(data: dict, exp_type: str, mode: str, out_png: Path) -> None:
    """Per-phase practicality scatter, one point per variant, tests pooled.
    Panel A: mean GPU % vs coverage-adjusted success rate. Panel B: mean GPU %
    vs median ATE among completers (log scale). Coloured by family; skippers hollow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    var_rows = _gather_variant_rows(data, exp_type)
    if not var_rows:
        return
    n_tests = len(data)
    pts: list[dict] = []
    for rows in var_rows.values():
        rb  = _variant_robustness(rows, n_tests)
        gpu = [r.get("gpu_avg_all_mean") for r in rows if r.get("gpu_avg_all_mean") is not None]
        ate = [r.get("ate_rmse_succ_median") for r in rows
               if r.get("n_success", 0) > 0
               and r.get("ate_rmse_succ_median") is not None]
        short = rows[0].get("short_name", rows[0]["variant"])
        pts.append({
            "short":   short,
            "fam":     _family(short, exp_type),
            "gpu":     float(np.mean(gpu)) if gpu else None,
            "rate":    rb["coverage_rate"],
            "ate":     float(np.median(ate)) if ate else None,
            "skipped": rb["n_attempted"] < n_tests,
        })

    short_exp = EXP_SHORT.get(exp_type, exp_type)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

    def _draw(ax, xkey, ykey):
        texts = []
        for p in pts:
            if p[xkey] is None or p[ykey] is None:
                continue
            c = _FAMILY_COLORS[p["fam"]]
            if p["skipped"]:
                ax.scatter(p[xkey], p[ykey], facecolors="none", edgecolors=c,
                           s=85, linewidths=1.7, zorder=3)
            else:
                ax.scatter(p[xkey], p[ykey], color=c, s=85,
                           edgecolors="white", linewidths=0.5, zorder=3)
            texts.append(ax.annotate(p["short"] + ("*" if p["skipped"] else ""),
                                     (p[xkey], p[ykey]), fontsize=6))
        _annotate_no_overlap(ax, texts)

    _draw(axA, "gpu", "rate")
    axA.axhspan(50, 105, xmin=0, xmax=0.35, color="#2a9d8f", alpha=0.06, zorder=0)
    axA.set_xlabel("mean GPU %")
    axA.set_ylabel("coverage-adjusted success rate (%)")
    axA.set_ylim(0, 105)
    axA.set_title("cost vs robustness  (top-left = cheap and reliable)")
    axA.grid(True, alpha=0.25)

    _draw(axB, "gpu", "ate")
    axB.set_xlabel("mean GPU %")
    axB.set_ylabel("median ATE RMSE (m) - completers only")
    axB.set_yscale("log")
    axB.set_title("cost vs accuracy  (bottom-left = cheap and accurate)")
    axB.grid(True, which="both", alpha=0.25)

    fams = [f for f in _FAMILY_ORDER if any(p["fam"] == f for p in pts)]
    handles = [Patch(facecolor=_FAMILY_COLORS[f], label=f) for f in fams]
    handles.append(Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor="none", markeredgecolor="0.3",
                          label="* skipped some sequences"))
    axA.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.75)

    fig.suptitle(f"{mode} - {short_exp.upper()} practicality "
                 "(GPU cost vs robustness / accuracy)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gradients(data: dict, mode: str, out_png: Path) -> None:
    """Left: outcome composition per sequence (stacked %), showing the
    difficulty ladder and how failures are distributed. Right: per-family mean
    GPU cost vs coverage-adjusted success rate, grouped bars. Both panels share
    one 0-100% axis (not twinned) so they can be read directly against each other."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tests = sorted(data.keys())
    if not tests:
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))

    # ── left: outcome composition per sequence ──────────────────────────────
    per_test = _per_test_status(data)
    x = np.arange(len(tests))
    bottom = np.zeros(len(tests))
    for key, label, colour in _STATUS_DISPLAY:
        vals = np.array([per_test[t].get(key, 0) for t in tests], dtype=float)
        tot  = np.array([max(sum(per_test[t].values()), 1) for t in tests], dtype=float)
        pct  = vals / tot * 100
        if pct.sum() == 0:
            continue
        axL.bar(x, pct, bottom=bottom, color=colour, width=0.62, label=label)
        for xi, (v, b) in enumerate(zip(pct, bottom)):
            if v >= 5:
                axL.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=7, color="white")
        bottom += pct
    axL.set_xticks(x); axL.set_xticklabels(tests)
    axL.set_ylim(0, 100); axL.set_ylabel("share of runs (%)")
    axL.set_title("Sequence difficulty - outcome composition")
    axL.legend(ncol=len(_STATUS_DISPLAY), fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, -0.07), frameon=False)

    # ── right: architecture gradient, one shared percentage axis ────────────
    fe_rows = _gather_variant_rows(data, _FE)
    n_tests = len(tests)
    fams, fam_gpu, fam_rate, fam_n = [], [], [], []
    for fam in _FAMILY_ORDER:
        members = [rows for rows in fe_rows.values()
                   if _family(rows[0].get("short_name", ""), _FE) == fam]
        if not members:
            continue
        grows = [r for rows in members for r in rows]
        gpu = [r.get("gpu_avg_all_mean") for r in grows if r.get("gpu_avg_all_mean") is not None]
        n_s = sum(r.get("n_success", 0) for r in grows)
        denom = sum(_variant_robustness(rows, n_tests)["denom_full"] for rows in members)
        fams.append(fam)
        fam_gpu.append(float(np.mean(gpu)) if gpu else 0.0)
        fam_rate.append(n_s / denom * 100 if denom else 0.0)
        fam_n.append(len(members))

    xf = np.arange(len(fams)); w = 0.36
    axR.bar(xf - w / 2, fam_gpu, width=w, color="#5b7fa6", label="mean GPU load")
    axR.bar(xf + w / 2, fam_rate, width=w, color="#2ca02c", alpha=0.85,
            label="coverage-adjusted success")
    for xi, (g, r) in enumerate(zip(fam_gpu, fam_rate)):
        axR.annotate(f"{g:.0f}%", (xi - w / 2, g), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=7)
        axR.annotate(f"{r:.0f}%", (xi + w / 2, r), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=7)
    axR.set_xticks(xf)
    axR.set_xticklabels([f"{f}\n({n} variants)" for f, n in zip(fams, fam_n)])
    axR.set_ylim(0, 105); axR.set_ylabel("percent")
    axR.set_title("Architecture gradient (feature extraction)")
    axR.legend(fontsize=8, loc="upper left", framealpha=0.8)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{mode} - sequence & architecture gradients "
                 "(both panels are percentages; no twin axes)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)




def plot_completion_cdf(data: dict, mode: str, out_png: Path) -> None:
    """Survival curve: share of runs still alive past x% of the sequence. Uses
    partial runs instead of discarding them, distinguishing "dies early" from
    "dies at 95%". One curve per family plus a dashed all-runs curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_fam: dict[str, list[float]] = {}
    every: list[float] = []
    for test_id, exp_map in data.items():
        for exp, rows in exp_map.items():
            for r in rows:
                c = r.get("completion_pct_mean")
                if c is None:
                    continue
                fam = _family(r.get("short_name", r["variant"]), exp)
                n   = max(int(r.get("n_total", 1)), 1)
                by_fam.setdefault(fam, []).extend([min(c, 100.0)] * n)
                every.extend([min(c, 100.0)] * n)
    if not every:
        return

    grid = np.linspace(0, 100, 201)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in _FAMILY_ORDER:
        vals = by_fam.get(fam)
        if not vals:
            continue
        arr = np.asarray(vals)
        surv = [(arr >= g).mean() * 100 for g in grid]
        ax.plot(grid, surv, color=_FAMILY_COLORS[fam], lw=2,
                label=f"{fam} (n={len(arr)})")
    arr = np.asarray(every)
    ax.plot(grid, [(arr >= g).mean() * 100 for g in grid], color="0.3",
            lw=1.4, ls="--", label=f"all runs (n={len(arr)})")

    ax.set_xlabel("sequence progress (%)")
    ax.set_ylabel("runs still tracking (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)
    ax.set_title(f"{mode} - survival by sequence progress\n"
                 "how far runs get, not just whether they finish", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_mode_summary(
    scan_root: Path, data: dict, mode: str,
    run_flags: dict, expected_frames: dict[str, int],
    variants_dict: dict = VARIANTS,
    out_dir: Path | None = None,
) -> None:
    """Rebuild the mode summary tree from scratch.

        overview/   - status_overview, status_matrix, status_summary.txt
        fe|lc|co/   - failure_map + machine_usage/ + performance/ + trends/
        be|de/      - machine_usage/ + performance/
        trends/     - gradients, completion_cdf
    """
    summary_dir  = out_dir if out_dir is not None else scan_root / "summary"
    overview_dir = rmtree_fresh(summary_dir) / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)

    per_test = _per_test_status(data)
    totals   = {k: sum(per_test[t][k] for t in per_test) for k in _STATUS_KEYS}
    grand    = sum(totals.values())

    lines = [f"{mode}  -  run outcomes", "=" * 40, ""]
    for k, label, _ in _STATUS_DISPLAY:
        lines.append(f"  {label:<8}: {totals[k]:>4}  ({100*totals[k]/max(grand,1):.0f}%)")
    lines.append(f"  {'TOTAL':<8}: {grand:>4}")
    if RAM_BUDGET_GB or N_THREADS:
        lines += ["", f"  machine: {N_THREADS or '?'} threads, "
                      f"{RAM_BUDGET_GB or '?'} GB"]
    lines += ["", "per test:"]
    for t in sorted(per_test):
        c = per_test[t]
        lines.append(f"  {t:<7} " + "  ".join(
            f"{lbl[:3]}={c[k]}" for k, lbl, _ in _STATUS_DISPLAY))
    (overview_dir / "status_summary.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")

    plot_status_overview(data, overview_dir / "status_overview.png", mode)
    plot_status_matrix(data, overview_dir / "status_matrix.png", mode,
                       scan_root=scan_root, run_flags=run_flags,
                       expected_frames=expected_frames)
    for _exp in EXPERIMENT_TYPES:
        plot_status_matrix(
            data, overview_dir / f"status_matrix_{EXP_SHORT.get(_exp, _exp)}.png",
            mode, scan_root=scan_root, run_flags=run_flags,
            expected_frames=expected_frames, only_exp=_exp)
    plot_failure_maps(scan_root, sorted(data.keys()), run_flags,
                      expected_frames, variants_dict, summary_dir)

    test_ids_sorted = sorted(data.keys())
    plot_system_usage_charts(data, test_ids_sorted, summary_dir)
    plot_performance_charts(data,  test_ids_sorted, summary_dir)

    for _exp, _short in (("feature_extraction", "fe"),
                         ("loop_closure",       "lc"),
                         ("combined",           "co")):
        plot_tradeoff(data, _exp, mode, summary_dir / _short / "trends" / "tradeoff.png")
    plot_gradients(data, mode, summary_dir / "trends" / "gradients.png")
    # robustness_leaderboard.png was removed as redundant with the success-rate/
    # completion columns; _variant_robustness() is kept, still used below.
    plot_completion_cdf(data, mode, summary_dir / "trends" / "completion_cdf.png")

    print(f"\n summary -> {summary_dir}/")
    print(f"   overview/        : status_overview, status_matrix, status_summary.txt")
    print(f"   fe|lc|co/        : failure_map_<bench>.png, trends/tradeoff.png")
    print(f"   */machine_usage/ : <test>_cpu/ram/ram_peak/gpu(.gb).png")
    print(f"   */performance/   : <test>_ate_rmse/fps/completion/...png")
    print(f"   */csv/           : <test>_<metric>.csv")
    print(f"   trends/          : gradients, completion_cdf")


# 5d.  Overview CSVs  (results/summary/overview/) ------------------------------------------
_OVERVIEW_BASE_COLS  = ["mode", "test_id", "exp_type", "sensor",
                        "variant", "short_name", "family",
                        "ram_budget_gb", "n_threads"]
# n_recovered deliberately omitted: n_oom / n_timeout measure the hard-kill rate directly.
_OVERVIEW_COUNT_COLS = [f"n_{k}" for k in
                        ("total", "metric_runs", "succ_runs", *_STATUS_KEYS)]


def _overview_columns() -> list[str]:
    """The exact column order of every overview CSV. Every metric column is
    pool-suffixed (_all / _succ); accuracy keys contribute only _succ (see
    metric_pools())."""
    cols = list(_OVERVIEW_BASE_COLS) + list(_OVERVIEW_COUNT_COLS)
    cols += ["best_status",
             "completion_pct_mean", "completion_pct_median",
             "completion_pct_std", "completion_pct_best"]
    for k in METRIC_KEYS:
        for tag in metric_pools(k):
            cols += [f"{k}{tag}_{s}" for s in STAT_SUFFIXES]
    for src, dst in _DERIVED_FROM.items():
        for tag in metric_pools(src):
            cols += [f"{dst}{tag}_{s}" for s in STAT_SUFFIXES]
    return cols


_round_cell = round_cell   # backwards-compatible alias


def _collect_baseline_rows(
    results_root: Path, mode: str, test_ids: list[str],
    run_flags: dict[tuple[str, str], list[dict]],
    expected_frames: dict[str, int],
) -> list[dict]:
    """The shared baseline lives beside the exp_type folders rather than inside
    one, so collect_all_data never sees it. This fills that gap."""
    scan_root = results_root / mode
    rows: list[dict] = []
    for test_id in test_ids:
        bdir = scan_root / test_id / BASELINE_VARIANT
        if not bdir.is_dir():
            continue
        agg = aggregate_variant(bdir, run_flags, BASELINE_VARIANT, test_id,
                                expected_frames.get(test_id))
        if agg["n_total"] == 0:
            continue
        rows.append({"mode": mode, "test_id": test_id, "exp_type": BASELINE_VARIANT,
                     "sensor": "mono" if mode == "VO" else "stereo",
                     "variant": BASELINE_VARIANT, "short_name": BASELINE_VARIANT,
                     **agg})
    return rows


def _average_rows(rows: list[dict]) -> list[dict]:
    """One AVG row per (mode, exp_type, sensor, variant) across tests.
    Counts are SUMMED; metrics are MEANed over the tests that produced a value."""
    from collections import OrderedDict
    count_cols  = set(_OVERVIEW_COUNT_COLS)
    text_cols   = set(_OVERVIEW_BASE_COLS) | {"best_status"}
    metric_cols = [c for c in _overview_columns()
                   if c not in text_cols and c not in count_cols]

    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for r in rows:
        key = (r.get("mode", ""), r.get("exp_type", ""), r.get("sensor", ""),
               r.get("variant", ""), r.get("short_name", ""))
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for (mode, exp_type, sensor, variant, short_name), grp in groups.items():
        avg: dict = {"mode": mode, "test_id": "AVG", "exp_type": exp_type,
                     "sensor": sensor, "variant": variant, "short_name": short_name,
                     "family": grp[0].get("family", ""),
                     "ram_budget_gb": grp[0].get("ram_budget_gb"),
                     "n_threads": grp[0].get("n_threads")}
        for c in count_cols:
            avg[c] = sum(int(r.get(c) or 0) for r in grp)
        for c in metric_cols:
            vals = [r[c] for r in grp if isinstance(r.get(c), (int, float))]
            avg[c] = float(np.mean(vals)) if vals else ""
        out.append(avg)
    return out


def write_overview_csvs(
    all_data: dict[str, dict],
    summary_root: Path,
    results_root: Path | None = None,
    test_ids: list[str] | None = None,
) -> None:
    """One flat CSV per test (both modes, every bench), plus all_tests.csv and
    averages.csv. These are what aggregate_all.py reads."""
    overview_dir = summary_root / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    cols = _overview_columns()

    per_test: dict[str, list[dict]] = {}
    for mode, data in all_data.items():
        for test_id, exp_map in data.items():
            for exp_type, rows in exp_map.items():
                for r in rows:
                    per_test.setdefault(test_id, []).append(
                        {"mode": mode, "test_id": test_id,
                         "family": _family(r.get("short_name", ""), exp_type), **r})

    if results_root is not None and test_ids is not None:
        run_flags       = load_run_flags(results_root)
        expected_frames = load_expected_frames()
        for mode in all_data:
            for br in _collect_baseline_rows(results_root, mode, test_ids,
                                             run_flags, expected_frames):
                br["family"] = "classical"
                per_test.setdefault(br["test_id"], []).append(br)

    def _write(path: Path, rows: list[dict]) -> None:
        rows = sorted(rows, key=lambda r: (r.get("test_id", ""), r.get("mode", ""),
                                           r.get("exp_type", ""), r.get("variant", "")))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: _round_cell(r.get(c, "")) for c in cols})

    master: list[dict] = []
    for test_id in sorted(per_test):
        _write(overview_dir / f"{test_id}.csv", per_test[test_id])
        master.extend(per_test[test_id])

    averages = _average_rows(master)
    if averages:
        _write(overview_dir / "averages.csv", averages)
    if master:
        _write(overview_dir / "all_tests.csv", master + averages)

    print(f"\n overview CSVs -> {overview_dir}/")
    print(f"   {', '.join(f'{t}.csv' for t in sorted(per_test))}"
          f"{', averages.csv' if averages else ''}"
          f"{', all_tests.csv' if master else ''}")


# 6.  Entry point --------------------------------------------------------------------------
def aggregate_all_results(
    results_root: Path,
    test_ids: list[str],
    exp_types: list[str],
    mode: str = "SLAM",
    variants_dict: dict = VARIANTS,
) -> dict:
    """Returns data[test_id][exp_type] = [row, ...] so the caller can combine
    modes (VO + SLAM) into the overview CSVs."""
    scan_root = results_root / mode
    if not scan_root.is_dir():
        print(f" [{mode}] no results tree at {scan_root} - skipping.")
        return {}

    run_flags       = load_run_flags(results_root)
    expected_frames = load_expected_frames()

    print(f"\n{'='*64}\n [{mode}]  Scanning {scan_root}\n{'='*64}")

    run_post_processing(scan_root, test_ids, exp_types, run_flags,
                        expected_frames, variants_dict)

    data = collect_all_data(scan_root, test_ids, exp_types,
                            variants_dict=variants_dict, run_flags=run_flags,
                            expected_frames=expected_frames)
    if not data:
        print(f" [{mode}] No results found.")
        return {}

    print_summary(data)
    mode_summary = results_root / "summary" / mode
    write_mode_summary(scan_root, data, mode, run_flags, expected_frames,
                       variants_dict, out_dir=mode_summary)
    return data


def all_test_ids() -> list[str]:
    config_dir = PC_ROOT / "src_pc" / "configs" / "sequences"
    ids = sorted({c.name.split("__")[0] for c in config_dir.glob("test*__*.yaml")})
    if not ids:
        sys.exit("No test configs found - run prep_project.py first.")
    return ids


_all_test_ids = all_test_ids   # backwards-compatible alias


def run_tree(results_root: Path = RESULTS,
             modes: list[str] | None = None,
             ram_gb: float | None = None,
             threads: int | None = None) -> dict[str, dict]:
    """Aggregate ONE results tree end to end. Importable by aggregate_all.py."""
    global RAM_BUDGET_GB, N_THREADS
    if ram_gb is not None:
        RAM_BUDGET_GB = ram_gb
    if threads is not None:
        N_THREADS = threads

    test_ids  = all_test_ids()
    exp_types = list(EXPERIMENT_TYPES)
    modes     = modes or ["VO", "SLAM"]

    print(f"\n{'='*64}")
    print(f" aggregate_results   (full rebuild - nothing is reused)")
    print(f" Root      : {results_root}")
    print(f" Machine   : {N_THREADS or '?'} threads, {RAM_BUDGET_GB or '?'} GB")
    print(f" Modes     : {modes}")
    print(f" Tests     : {test_ids}")
    print(f"{'='*64}")

    # Blow away the whole summary tree up front (write_mode_summary only clears
    # its own sub-tree, which would leave stale overview/ CSVs and other modes behind).
    summary_root = results_root / "summary"
    if summary_root.exists():
        print(f" clearing {summary_root}")
    rmtree_fresh(summary_root)

    all_data: dict[str, dict] = {}
    for mode in modes:
        all_data[mode] = aggregate_all_results(
            results_root=results_root, test_ids=test_ids,
            exp_types=exp_types, mode=mode)

    if any(all_data.values()):
        write_overview_csvs(all_data, results_root / "summary",
                            results_root=results_root, test_ids=test_ids)
    report_unfamilied()
    return all_data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate one results tree: recover, classify, plot, export.")
    ap.add_argument("--root", type=Path, default=RESULTS,
                    help="results tree to scan (default: paths.RESULTS)")
    ap.add_argument("--ram-gb", type=float, default=None,
                    help="WSL2 memory allocation in GB. Turns the RAM%% columns "
                         "into absolute GB, which is what makes them comparable "
                         "between machines.")
    ap.add_argument("--threads", type=int, default=None,
                    help="logical processors given to the guest. Turns the CPU%% "
                         "columns into busy-thread counts.")
    ap.add_argument("--modes", nargs="+", default=["VO", "SLAM"])
    args = ap.parse_args()
    run_tree(args.root, modes=args.modes,
             ram_gb=args.ram_gb, threads=args.threads)


if __name__ == "__main__":
    main()