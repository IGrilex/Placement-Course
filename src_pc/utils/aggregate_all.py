#!/usr/bin/env python3
"""
src_pc/utils/aggregate_all.py

Cross-CONDITION aggregation: runs aggregate_results.py on every results tree,
then compares them. This file holds only the cross-condition logic (condition
discovery, the ladder, the rung report, comparison figures/tables); generic
helpers live in aggregate_results.py and are imported from there.

No --reuse flag: every run wipes and rebuilds each tree's summary/ and
post_processing/, then results_comparison/, from the raw run folders.

all_conditions.csv is long-form; every row carries a `level`:
    "run"        one individual execution -- provenance, repetition spread.
    "cell"       one (condition, mode, test, exp, variant), metrics pooled
                 over repetitions -- what every figure/table reads.
    "condition"  one hardware condition -- headline accounting on common
                 non-probe cells only.
Filter on `level` first -- summing unfiltered counts every run twice.

Provenance (level="run" only) -- two independent records per run, a
run_flags.csv row and a run directory, which can disagree:
    measured    both present. Normal.
    recovered   directory only. Run happened; metrics are real.
    orphaned    flag row only, no directory -- a ghost with no metrics, kept
                so run counts reconcile but never aggregated.

Repetition balance (level="cell", `rep_status`), norm = modal n_total per
(condition, mode):
    probe   n=1 where norm >=3 -- an exploratory run behind a skip decision,
            not a method measurement. Excluded from every aggregate.
    under/over   fewer/more reps than the norm. Kept, n visible.
    full    the norm.

Every headline metric gets a rel_<column> twin: ratio to the ORB2 baseline on
the SAME (condition, mode, sequence). Ratio of medians, never median of
ratios (unequal repetition counts make the latter far noisier). Accuracy
ratios are null, not zero, when the baseline itself didn't complete.

No un-suffixed metric columns: `<key>_succ_*` is completers only (the only
form accuracy keys take), `<key>_all_*` is every run with data. Machine-load
comparisons read _all_ (an OOM-killed run still reports what it was
holding); accuracy comparisons read _succ_.

Conditions are discovered, not hardcoded: any results* dir under PC_ROOT
with a SLAM/ or VO/ subfolder. CPU/threads/RAM are parsed from the folder
name (e.g. results_R9_12T_14RAM); unparseable ones go in MANUAL_OVERRIDES.
Sorted into a ladder (i7 first, then RAM, then threads); each rung's label
is inferred from whichever attribute changed between neighbours.

Produces (in <out>/): all_conditions.csv, all_conditions_DICTIONARY.md,
rung_report.txt + transitions_<rung>.csv, tables/*.tex (+.csv twins), and
figures/ including failure_map_<test>_<bench>.png -- one panel per CONDITION,
separating budget-caused failure (deaths scatter, vanish as budget grows)
from algorithm-caused failure (deaths cluster and persist).

USAGE
  python -m src_pc.utils.aggregate_all                    # everything, SLAM
  python -m src_pc.utils.aggregate_all --mode VO          # everything, VO
  python -m src_pc.utils.aggregate_all --root /some/dir   # scan elsewhere
  python -m src_pc.utils.aggregate_all --no-run-level     # skip per-run scan
"""

from __future__ import annotations
import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:                                          # pragma: no cover
    sys.exit("aggregate_all needs pandas:  pip install pandas")

try:
    from scipy import stats as _scipy_stats
except ImportError:                                          # pragma: no cover
    _scipy_stats = None   # only the Friedman test needs it directly

_FILE   = Path(__file__).resolve()
PC_ROOT = _FILE.parents[2]
sys.path.insert(0, str(PC_ROOT))

import src_pc.utils.aggregate_results as agg                 # noqa: E402
from src_pc.utils.aggregate_results import (                 # noqa: E402
    _STATUS_KEYS, _STATUS_DISPLAY, STATUS_COLOR, STATUS_LABEL,
    _FAMILY_COLORS, _FAMILY_ORDER, run_tree, report_unfamilied,
    # generic helpers -- single copy, see module docstring
    rmtree_fresh, round_cell, get_mpl, dedupe_columns, coerce_numeric,
    wilcoxon_note,
    # failure-space mapping -- reuse the per-condition extractor so the two
    # maps can never disagree about what counts as a failure
    _bench_fail_points, load_run_flags, load_expected_frames, all_test_ids,
)
from src_pc.experiments.variants import VARIANTS, BASELINE_VARIANT   # noqa: E402

# ─── column names, resolved once (aggregate_results has no un-suffixed
# metric columns; naming them here makes a suffix-convention change one edit)
C_FPS        = "avg_fps_all_median"
C_CPU        = "cpu_avg_all_mean"
C_GPU        = "gpu_avg_all_mean"
C_RAM_PEAK   = "ram_max_gb_all_mean"
C_RAM_AVG    = "ram_avg_gb_all_mean"
C_THREADS    = "cpu_busy_threads_all_mean"
C_ATE        = "ate_rmse_succ_median"       # completers only, always
C_ATE_MEAN   = "ate_rmse_succ_mean"
C_RPE_SHORT  = "rpe_short_rmse_succ_median"
C_RPE_LONG   = "rpe_long_rmse_succ_median"
C_SCALE      = "ate_scale_succ_median"
C_WALL       = "wall_time_s_succ_median"
C_KEYFRAMES  = "num_keyframes_all_median"
C_MAPPOINTS  = "map_points_all_median"
C_COMPLETION = "completion_pct_mean"

# ─── which columns get a rel_<col> twin, and what kind of quantity they are ─
# "acc"  -> undefined unless the BASELINE completed (no ATE without a finish)
# "load" -> defined whenever the baseline produced data at all, because an
#           OOM-killed run still truthfully reports what it was holding
_REL_METRICS: list[tuple[str, str]] = [
    (C_ATE,        "acc"),
    (C_ATE_MEAN,   "acc"),
    (C_RPE_SHORT,  "acc"),
    (C_RPE_LONG,   "acc"),
    (C_SCALE,      "acc"),
    (C_WALL,       "acc"),
    (C_FPS,        "load"),
    (C_CPU,        "load"),
    (C_GPU,        "load"),
    (C_RAM_PEAK,   "load"),
    (C_RAM_AVG,    "load"),
    (C_THREADS,    "load"),
    (C_KEYFRAMES,  "load"),
    (C_MAPPOINTS,  "load"),
    (C_COMPLETION, "load"),
]

# Base metric keys as classify_run() reports them for a single run (same
# names aggregate_results suffixes with _succ_/_all_ when pooling a cell).
_RUN_METRIC_KEYS = [
    "ate_rmse", "ate_mean", "ate_scale",
    "rpe_short_rmse", "rpe_long_rmse",
    "avg_fps", "wall_time_s",
    "cpu_avg", "cpu_max", "ram_avg", "ram_max", "gpu_avg", "gpu_max",
    "map_points", "num_keyframes", "pre_init_frames", "lost_pct",
]

LEVEL_RUN, LEVEL_CELL, LEVEL_COND = "run", "cell", "condition"


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    """Real boolean Series for `col`, whether or not it exists. Needed because
    a bool column holding NaN (from concatenating run/cell/condition rows)
    goes object-dtype, and `~` on an object Series silently does integer
    negation instead of logical negation."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)

# Column order at the front of the CSV; everything else follows alphabetically.
_FRONT_COLS = [
    "level", "condition", "rung", "cond_label", "cpu", "threads", "ram_budget",
    "mode", "test_id", "exp_type", "sensor", "variant", "short_name", "family",
    "provenance", "run_num", "run_dir", "timestamp",
    "is_probe", "rep_status", "n_expected",
    "n_total", "n_metric_runs", "status", "best_status",
]


# ════════════════════════════════════════════════════════════════════════════
#  CONDITION DISCOVERY
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Condition:
    key:      str          # short id used in every table and figure
    folder:   str          # directory name under PC_ROOT
    label:    str          # human label for axes
    cpu:      str
    threads:  int          # .wslconfig processors, 0 = unknown
    ram_gb:   float        # .wslconfig memory,      0 = unknown
    rung:     int          # position on the ladder (1 = first), assigned later


# Escape hatch for folder names that don't encode every attribute (cpu,
# threads, ram). Keyed by exact folder name.
MANUAL_OVERRIDES: dict[str, dict] = {
    "results_I7_14RAM": {"cpu": "i7-8700K", "threads": 12, "ram_gb": 14.0},
}

_CPU_PATTERNS = [
    (re.compile(r"i7", re.I), "i7-8700K"),
    (re.compile(r"r9", re.I), "R9 9950X"),
]


def _parse_condition(folder: Path) -> Condition | None:
    """Folder name -> Condition, or None if it isn't a results tree (no
    SLAM/ or VO/ subfolder, e.g. the comparison output directory)."""
    if not ((folder / "SLAM").is_dir() or (folder / "VO").is_dir()):
        return None

    name = folder.name
    override = MANUAL_OVERRIDES.get(name, {})

    cpu = override.get("cpu")
    if cpu is None:
        cpu = next((label for pat, label in _CPU_PATTERNS if pat.search(name)),
                   "unknown")

    threads = override.get("threads")
    if threads is None:
        # \b doesn't fire between "T" and "_" (underscore is a word char),
        # so use an explicit lookahead for underscore/digit/end instead.
        m = re.search(r"(\d+)\s*T(?:HREADS?)?(?=_|\d|$)", name, re.I)
        threads = int(m.group(1)) if m else 0

    ram_gb = override.get("ram_gb")
    if ram_gb is None:
        m = re.search(r"(\d+)\s*RAM(?=_|\d|$)", name, re.I)
        ram_gb = float(m.group(1)) if m else 0.0

    if not threads or not ram_gb:
        print(f" [!] {name}: could not fully parse threads/RAM from the name "
              f"(got threads={threads or '?'}, ram={ram_gb or '?'} GB). "
              f"Add an entry to MANUAL_OVERRIDES in aggregate_all.py.")

    cpu_short = "i7" if "i7" in cpu.lower() else ("R9" if "r9" in cpu.lower() else "cpu")
    key   = f"{cpu_short}_{threads or '?'}t_{ram_gb:g}"
    label = f"{cpu} · {threads or '?'}t · {ram_gb:g} GB"
    return Condition(key=key, folder=name, label=label, cpu=cpu,
                     threads=threads, ram_gb=ram_gb, rung=0)


def discover_conditions(root: Path, exclude: set[str]) -> list[Condition]:
    """Every results* directory under `root` that looks like a results tree,
    sorted into a ladder: i7 first, then ascending RAM, then ascending
    threads. A genuinely different hardware set still sorts consistently but
    may not isolate one factor per rung -- check rung_report.txt."""
    found: list[Condition] = []
    seen_keys: set[str] = set()
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in exclude:
            continue
        if not d.name.lower().startswith("results"):
            continue
        c = _parse_condition(d)
        if c is None:
            continue
        if c.key in seen_keys:
            c = replace(c, key=f"{c.key}_{d.name}")
        seen_keys.add(c.key)
        found.append(c)

    if not found:
        sys.exit(f"No results* trees found under {root} "
                 f"(looked for subfolders containing SLAM/ or VO/).")

    found.sort(key=lambda c: (0 if "i7" in c.cpu.lower() else 1, c.ram_gb, c.threads))
    return [replace(c, rung=i) for i, c in enumerate(found, start=1)]


def build_rungs(conds: list[Condition]) -> list[tuple[str, str, str]]:
    """What each consecutive pair on the ladder isolates, inferred from which
    attribute actually differs between them."""
    rungs = []
    for a, b in zip(conds, conds[1:]):
        if a.cpu != b.cpu:
            name = "platform"
        elif a.ram_gb != b.ram_gb:
            name = "capacity"
        elif a.threads != b.threads:
            name = "core count"
        else:
            name = "unspecified"
        rungs.append((a.key, b.key, name))
    return rungs


# Populated by main() from discover_conditions()/build_rungs(); module-level
# so plotting/reporting functions don't need it threaded through as a param.
CONDITIONS: list[Condition] = []
RUNGS: list[tuple[str, str, str]] = []

OUT_DIR = PC_ROOT / "results_aggregated" / "csv"


# ════════════════════════════════════════════════════════════════════════════
#  1.  PER-RUN SCAN AND PROVENANCE RECONCILIATION
# ════════════════════════════════════════════════════════════════════════════
def _flags_csv(tree: Path) -> Path | None:
    """The batch runner's per-run log for one results tree. Looked up by glob
    (run_flags*.csv), since campaigns were archived under a few different
    filenames and hard-coding one would mislabel every run as 'recovered'."""
    for pat in ("run_flags.csv", "run_flags*.csv", "*/run_flags*.csv"):
        hits = sorted(tree.glob(pat))
        if hits:
            return hits[0]
    return None


def _mode_of_command(cmd: str) -> str:
    """VO runs are launched with --no-loop-closure. Read off the recorded
    command (not the directory), since an orphan row has no directory."""
    return "VO" if isinstance(cmd, str) and "--no-loop-closure" in cmd else "SLAM"


def _exp_type_of(variant: str) -> str:
    v = VARIANTS.get(variant)
    return v.get("experiment_type", "unknown") if v else "unknown"


def _short_name_of(variant: str) -> str:
    v = VARIANTS.get(variant)
    return v.get("short_name", variant) if v else variant


def load_flag_rows(cond: Condition) -> pd.DataFrame:
    """Every row of one tree's run_flags.csv, normalised. Returns an empty
    frame (not None) when the tree has no flag log, so every run in that
    tree simply comes out as 'recovered'."""
    tree = PC_ROOT / cond.folder
    path = _flags_csv(tree)
    if path is None or not path.is_file():
        print(f"   [!] {cond.key}: no run_flags*.csv found - every run in this "
              f"tree will be labelled 'recovered'")
        return pd.DataFrame()
    try:
        f = pd.read_csv(path)
    except Exception as e:
        print(f"   [!] {cond.key}: could not read {path.name}: {e}")
        return pd.DataFrame()

    f.columns = [c.strip() for c in f.columns]
    need = {"variant", "test_id"}
    if not need.issubset(f.columns):
        print(f"   [!] {cond.key}: {path.name} lacks {need - set(f.columns)}")
        return pd.DataFrame()

    f["mode"] = f.get("command", pd.Series([""] * len(f))).map(_mode_of_command)
    f["exp_type"] = f["variant"].map(_exp_type_of)
    for c in ("timestamp", "status", "returncode", "elapsed_s",
              "killed_signal", "wall_limit_s", "hard_limit_s", "run_num"):
        if c not in f.columns:
            f[c] = np.nan
    f["_ts"] = pd.to_datetime(f["timestamp"], errors="coerce")
    return f


def _pick(m: dict, *names):
    """First present, non-null value among `names` (so a rename upstream
    costs an empty column, not a KeyError)."""
    for n in names:
        if n in m and m[n] is not None and not (
                isinstance(m[n], float) and np.isnan(m[n])):
            return m[n]
    return None


def scan_run_dirs(cond: Condition, mode: str,
                  expected_frames: dict) -> pd.DataFrame:
    """Walk one tree's run directories and classify each run individually.
    Re-reads the raw folders rather than the (already pooled) overview CSV,
    since the point of the run level is to see the spread pooling hides."""
    tree = PC_ROOT / cond.folder
    scan_root = tree / mode
    if not scan_root.is_dir():
        return pd.DataFrame()

    try:
        run_flags = load_run_flags(tree)
    except Exception:
        run_flags = {}

    rows: list[dict] = []
    for test_dir in sorted(p for p in scan_root.iterdir() if p.is_dir()):
        test_id  = test_dir.name
        expected = expected_frames.get(test_id)
        for exp_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            exp_type = exp_dir.name
            for var_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                variant = next(
                    (k for k, v in VARIANTS.items()
                     if v.get("short_name") == var_dir.name
                     and v.get("experiment_type") == exp_type),
                    var_dir.name)
                try:
                    run_dirs = list(agg._run_dirs_for_variant(var_dir))
                except Exception:
                    run_dirs = [p for p in sorted(var_dir.iterdir()) if p.is_dir()]

                for rd in sorted(run_dirs, key=lambda p: p.name):
                    try:
                        flag_row = agg.match_run_flag(run_flags, variant,
                                                      test_id, rd.name)
                    except Exception:
                        flag_row = None
                    try:
                        status, m = agg.classify_run(rd, flag_row, expected)
                    except Exception as e:
                        print(f"   [!] {cond.key} {rd}: classify failed ({e})")
                        continue
                    m = m or {}

                    processed = _pick(m, "processed", "n_frames",
                                      "processed_frames")
                    compl = _pick(m, "completion_pct")
                    if compl is None and processed and expected:
                        compl = 100.0 * float(processed) / float(expected)

                    row = {
                        "level":      LEVEL_RUN,
                        "mode":       mode,
                        "test_id":    test_id,
                        "exp_type":   exp_type,
                        "variant":    variant,
                        "short_name": var_dir.name,
                        "run_dir":    rd.name,
                        "status":     status,
                        "processed_frames": processed,
                        "expected_frames":  expected,
                        "completion_pct":   compl,
                        "first_loss_frame": _pick(m, "first_loss_frame"),
                        "_has_dir":   True,
                    }
                    for k in _RUN_METRIC_KEYS:
                        row[k] = _pick(m, k)

                    # Recomputed from this condition's own budget rather than
                    # read out of the metrics dict: aggregate_results' globals
                    # hold whichever tree ran last, which would silently
                    # convert one condition's percentages using another's budget.
                    if cond.ram_gb and row.get("ram_max") is not None:
                        row["ram_max_gb"] = float(row["ram_max"]) / 100.0 * cond.ram_gb
                    if cond.ram_gb and row.get("ram_avg") is not None:
                        row["ram_avg_gb"] = float(row["ram_avg"]) / 100.0 * cond.ram_gb
                    if cond.threads and row.get("cpu_max") is not None:
                        row["cpu_busy_threads_max"] = float(row["cpu_max"]) / 100.0 * cond.threads
                    if cond.threads and row.get("cpu_avg") is not None:
                        row["cpu_busy_threads"] = float(row["cpu_avg"]) / 100.0 * cond.threads

                    rows.append(row)
    return pd.DataFrame(rows)


def reconcile_runs(cond: Condition, mode: str,
                   expected_frames: dict) -> pd.DataFrame:
    """Per-run rows with provenance, one tree, one mode.

    Directories and flag rows are matched WITHIN a cell, ordinally (by
    directory name / recorded timestamp, both already execution order) rather
    than by exact per-run match -- weaker, but avoids the fragile timestamp
    arithmetic that exact matching needs across archived campaigns, and all
    this needs is the count reconciliation (how many runs have both records,
    which side is missing)."""
    disk  = scan_run_dirs(cond, mode, expected_frames)
    flags = load_flag_rows(cond)
    if not flags.empty:
        flags = flags[flags["mode"] == mode].copy()

    key = ["mode", "test_id", "exp_type", "variant"]
    out: list[pd.DataFrame] = []

    cells = set()
    if not disk.empty:
        cells |= set(map(tuple, disk[key].values))
    if not flags.empty:
        cells |= set(map(tuple, flags[key].values))

    for cell in sorted(cells):
        d = disk[np.logical_and.reduce([disk[k] == v for k, v in zip(key, cell)])] \
            if not disk.empty else pd.DataFrame()
        f = flags[np.logical_and.reduce([flags[k] == v for k, v in zip(key, cell)])] \
            if not flags.empty else pd.DataFrame()
        d = d.sort_values("run_dir").reset_index(drop=True) if len(d) else d
        f = f.sort_values("_ts").reset_index(drop=True) if len(f) else f

        n_pair = min(len(d), len(f))
        recs: list[dict] = []

        for i in range(max(len(d), len(f))):
            if i < len(d) and i < len(f):
                r = d.iloc[i].to_dict()
                fr = f.iloc[i]
                r["provenance"]    = "measured"
                r["flag_status"]   = fr.get("status")
                r["returncode"]    = fr.get("returncode")
                r["elapsed_s"]     = fr.get("elapsed_s")
                r["killed_signal"] = fr.get("killed_signal")
                r["wall_limit_s"]  = fr.get("wall_limit_s")
                r["hard_limit_s"]  = fr.get("hard_limit_s")
                r["run_num"]       = fr.get("run_num")
                r["timestamp"]     = fr.get("timestamp")
            elif i < len(d):
                r = d.iloc[i].to_dict()
                r["provenance"]  = "recovered"
                r["flag_status"] = None
                r["run_num"]     = i + 1
            else:
                fr = f.iloc[i]
                # ORPHAN: no directory, so no metrics are invented -- the row
                # exists only so run counts reconcile and the ghost is visible.
                r = {
                    "level":      LEVEL_RUN,
                    "mode":       cell[0], "test_id": cell[1],
                    "exp_type":   cell[2], "variant": cell[3],
                    "short_name": _short_name_of(cell[3]),
                    "run_dir":    None,
                    "status":     fr.get("status"),
                    "flag_status": fr.get("status"),
                    "provenance": "orphaned",
                    "returncode": fr.get("returncode"),
                    "elapsed_s":  fr.get("elapsed_s"),
                    "killed_signal": fr.get("killed_signal"),
                    "wall_limit_s":  fr.get("wall_limit_s"),
                    "hard_limit_s":  fr.get("hard_limit_s"),
                    "run_num":    fr.get("run_num"),
                    "timestamp":  fr.get("timestamp"),
                    "expected_frames": expected_frames.get(cell[1]),
                    "_has_dir":   False,
                }
                r["level"] = LEVEL_RUN
            recs.append(r)

        if recs:
            out.append(pd.DataFrame(recs))

    if not out:
        return pd.DataFrame()

    runs = pd.concat(out, ignore_index=True)
    runs["level"]      = LEVEL_RUN
    runs["condition"]  = cond.key
    runs["rung"]       = cond.rung
    runs["cond_label"] = cond.label
    runs["cpu"]        = cond.cpu
    runs["threads"]    = cond.threads
    runs["ram_budget"] = cond.ram_gb
    # _family() needs (short_name, exp_type): the same short name means
    # different families in different phases.
    def _fam(sn, et):
        try:
            return agg._family(str(sn), str(et))
        except Exception:
            return None
    runs["family"] = [_fam(sn, et) for sn, et
                      in zip(runs["short_name"], runs["exp_type"])]
    runs["sensor"] = runs["variant"].map(
        lambda v: "mono" if (VARIANTS.get(v, {}) or {}).get("depth_estimator")
        else "stereo")
    return runs


# ════════════════════════════════════════════════════════════════════════════
#  2.  REPETITION BALANCE  (probe / under / full / over)
# ════════════════════════════════════════════════════════════════════════════
def annotate_reps(cells: pd.DataFrame) -> pd.DataFrame:
    """Label every cell by rep_status (probe/under/full/over), using the
    MODAL n_total within each (condition, mode) as the norm rather than a
    hard-coded constant, so a campaign at a different repetition count is
    still labelled correctly. Probes (n=1 where norm>=3) are flagged and
    excluded from aggregates, never deleted."""
    if cells.empty:
        return cells
    d = cells.copy()
    if "n_total" not in d.columns:
        d["n_total"] = np.nan

    norms: dict[tuple, float] = {}
    for (cond, mode), grp in d.groupby(["condition", "mode"], dropna=False):
        vals = grp["n_total"].dropna()
        norms[(cond, mode)] = float(vals.mode().iloc[0]) if len(vals) else np.nan

    d["n_expected"] = [norms.get((c, m), np.nan)
                       for c, m in zip(d["condition"], d["mode"])]

    def _status(n, exp):
        if pd.isna(n) or pd.isna(exp):
            return "unknown"
        if n == exp:
            return "full"
        if n == 1 and exp >= 3:
            return "probe"
        return "under" if n < exp else "over"

    d["rep_status"] = [_status(n, e)
                       for n, e in zip(d["n_total"], d["n_expected"])]
    d["is_probe"] = d["rep_status"] == "probe"
    return d


# ════════════════════════════════════════════════════════════════════════════
#  3.  RELATIVE-TO-BASELINE COLUMNS
# ════════════════════════════════════════════════════════════════════════════
def add_relative(cells: pd.DataFrame) -> pd.DataFrame:
    """Attach a rel_<col> twin to every headline metric: each cell divided by
    the ORB2 baseline on the SAME (condition, mode, sequence) -- per sequence,
    not pooled, since sequences differ from each other more than methods do.
    Ratio of medians, not median of ratios (both columns are already per-cell
    medians, dividing them is the first form). Accuracy ratios stay null when
    the baseline didn't complete on that sequence (no ATE to divide by)."""
    if cells.empty:
        return cells
    d = cells.copy()
    keys = ["condition", "mode", "test_id"]

    base = d[d["variant"] == BASELINE_VARIANT]
    if base.empty:
        print(f"   [!] no '{BASELINE_VARIANT}' rows found - rel_* columns "
              f"will be empty. Every relative number in the thesis depends on "
              f"this variant existing in every (condition, mode, sequence).")
        for col, _ in _REL_METRICS:
            d[f"rel_{col}"] = np.nan
        d["baseline_completed"] = np.nan
        return d

    base = base.drop_duplicates(subset=keys, keep="first").set_index(keys)
    base_ok = (base.get("n_success", pd.Series(0, index=base.index))
               .fillna(0) > 0)
    idx = pd.MultiIndex.from_arrays([d[k] for k in keys])

    d["baseline_completed"] = base_ok.reindex(idx).values

    for col, kind in _REL_METRICS:
        out = np.full(len(d), np.nan, dtype=float)
        if col in d.columns and col in base.columns:
            denom = pd.to_numeric(base[col], errors="coerce").reindex(idx).values
            num   = pd.to_numeric(d[col], errors="coerce").values
            ok = np.isfinite(denom) & (np.abs(denom) > 1e-12) & np.isfinite(num)
            if kind == "acc":
                ok &= np.asarray(d["baseline_completed"].fillna(False),
                                 dtype=bool)
            out[ok] = num[ok] / denom[ok]
        d[f"rel_{col}"] = out

    return d


def baseline_health(cells: pd.DataFrame) -> pd.DataFrame:
    """The baseline's own absolute numbers, per (condition, sequence). Every
    rel_* column divides by these, so a drifted baseline shows up here as one
    number out of line with its neighbours -- the cheapest available check."""
    b = cells[(cells["variant"] == BASELINE_VARIANT)
              & (cells["test_id"] != "AVG")]
    if b.empty:
        return pd.DataFrame()
    cols = [c for c in (C_ATE, C_FPS, C_RAM_PEAK, C_GPU, C_COMPLETION)
            if c in b.columns]
    out = b[["condition", "cond_label", "rung", "mode", "test_id",
             "n_total", "n_success"] + cols].copy()
    return out.sort_values(["mode", "test_id", "rung"])


# ════════════════════════════════════════════════════════════════════════════
#  4.  Load every condition into one long DataFrame
# ════════════════════════════════════════════════════════════════════════════
def _overview_csv(cond: Condition) -> Path:
    return PC_ROOT / cond.folder / "summary" / "overview" / "all_tests.csv"


def build_condition(cond: Condition) -> pd.DataFrame | None:
    """Re-aggregate one results tree from its raw run folders (run_tree()
    wipes and rebuilds summary/ and post_processing/ -- no reuse path), then
    load the overview CSV it just wrote -> long DataFrame tagged with the
    condition."""
    root = PC_ROOT / cond.folder
    if not root.is_dir():
        print(f" [!] {cond.key}: {root} does not exist - skipping")
        return None

    # Real budget in -> RAM% becomes GB and CPU% becomes busy threads.
    run_tree(root, ram_gb=cond.ram_gb or None, threads=cond.threads or None)

    csv_path = _overview_csv(cond)
    if not csv_path.is_file():
        print(f" [!] {cond.key}: no overview CSV produced")
        return None

    raw = pd.read_csv(csv_path)

    # Built as one dict and concat'd once -- repeated df[...] = assignment
    # fragments a table this wide badly enough to warn on every write.
    tag = {
        "level":      LEVEL_CELL,
        "condition":  cond.key,
        "rung":       cond.rung,
        "cond_label": cond.label,
        "cpu":        cond.cpu,
        "threads":    cond.threads,
        "ram_budget": cond.ram_gb,
    }
    df = pd.concat(
        [pd.DataFrame({k: [v] * len(raw) for k, v in tag.items()}, index=raw.index),
         raw], axis=1)
    return dedupe_columns(df)


def condition_rows(cells: pd.DataFrame) -> pd.DataFrame:
    """One roll-up row per condition, computed on common, non-probe,
    real-sequence cells -- the same subset every figure uses -- so filtering
    level=="condition" and recomputing from level=="cell" agree."""
    if cells.empty:
        return pd.DataFrame()
    rows = []
    for mode in sorted(cells["mode"].dropna().unique()):
        d0 = cells[(cells["mode"] == mode) & (cells["test_id"] != "AVG")]
        d0 = d0[~_bool(d0, "is_probe")]
        for c in _order():
            d = d0[d0["condition"] == c.key]
            if d.empty:
                continue
            tot = max(float(d["n_total"].sum()), 1.0)
            ate = d.loc[d["n_success"] > 0, C_ATE].dropna() \
                if C_ATE in d.columns else pd.Series(dtype=float)
            # Algorithm-attributable denominator (success + lost, budget
            # deaths removed) -- not the same quantity as success_pct.
            alg = float(d["n_success"].sum() + d["n_lost"].sum())
            rows.append({
                "level": LEVEL_COND, "condition": c.key, "cond_label": c.label,
                "rung": c.rung, "cpu": c.cpu, "threads": c.threads,
                "ram_budget": c.ram_gb, "mode": mode,
                "test_id": "ALL", "exp_type": "ALL", "variant": "ALL",
                "n_cells": int(len(d)),
                "n_total": int(d["n_total"].sum()),
                "n_success": int(d["n_success"].sum()),
                "n_lost": int(d["n_lost"].sum()),
                "n_timeout": int(d["n_timeout"].sum()),
                "n_oom": int(d["n_oom"].sum()),
                "n_error": int(d["n_error"].sum()) if "n_error" in d else 0,
                "success_pct": round(d["n_success"].sum() / tot * 100, 2),
                "lost_pct":    round(d["n_lost"].sum() / tot * 100, 2),
                "timeout_pct": round(d["n_timeout"].sum() / tot * 100, 2),
                "oom_pct":     round(d["n_oom"].sum() / tot * 100, 2),
                "budget_failures": int(d["n_timeout"].sum() + d["n_oom"].sum()),
                "completion_rate_algorithmic":
                    round(d["n_success"].sum() / alg * 100, 2) if alg else None,
                C_FPS: round(float(d[C_FPS].median()), 3)
                       if C_FPS in d and d[C_FPS].notna().any() else None,
                C_ATE: round(float(ate.median()), 4) if len(ate) else None,
                C_RAM_PEAK: round(float(d[C_RAM_PEAK].mean()), 3)
                       if C_RAM_PEAK in d and d[C_RAM_PEAK].notna().any() else None,
                C_THREADS: round(float(d[C_THREADS].mean()), 3)
                       if C_THREADS in d and d[C_THREADS].notna().any() else None,
                C_COMPLETION: round(float(d[C_COMPLETION].mean()), 2)
                       if C_COMPLETION in d and d[C_COMPLETION].notna().any() else None,
            })
    return pd.DataFrame(rows)


def load_all(with_runs: bool = True, mode: str = "SLAM") -> pd.DataFrame:
    """The whole story, one frame: run rows, cell rows, condition rows.
    Cells are built first (repetition norms/baseline ratios are cell-level),
    then run rows, then the condition roll-up so it sees cells' probe flags."""
    frames = [f for f in (build_condition(c) for c in CONDITIONS) if f is not None]
    if not frames:
        sys.exit("No conditions could be loaded.")
    cells = dedupe_columns(pd.concat(frames, ignore_index=True))
    cells = coerce_numeric(cells)
    cells = annotate_reps(cells)
    cells = add_relative(cells)

    parts = [cells]

    conds = condition_rows(cells)
    if not conds.empty:
        parts.append(conds)

    if with_runs:
        print("\n per-run scan (provenance reconciliation):")
        try:
            expected_frames = load_expected_frames()
        except Exception:
            expected_frames = {}
        for c in CONDITIONS:
            for m in ("SLAM", "VO"):
                try:
                    r = reconcile_runs(c, m, expected_frames)
                except Exception as e:
                    print(f"   [!] {c.key} {m}: run scan failed ({e})")
                    continue
                if r.empty:
                    continue
                counts = r["provenance"].value_counts().to_dict()
                print(f"   {c.key:<14} {m:<5} {len(r):4d} runs   "
                      + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
                parts.append(r)

    df = dedupe_columns(pd.concat(parts, ignore_index=True))
    df = coerce_numeric(df)

    front = [c for c in _FRONT_COLS if c in df.columns]
    rest  = sorted(c for c in df.columns if c not in front and not c.startswith("_"))
    return df[front + rest]


# ════════════════════════════════════════════════════════════════════════════
#  5.  Cell bookkeeping
# ════════════════════════════════════════════════════════════════════════════
CELL = ["mode", "exp_type", "variant", "test_id"]


def real_rows(df: pd.DataFrame, mode: str = "SLAM",
              drop_probes: bool = True) -> pd.DataFrame:
    """Cell-level rows for one mode: no AVG roll-ups, no run rows, no probes.
    The level filter matters -- without it, grouping counts every run twice
    since run rows and condition roll-ups share the file."""
    d = df[(df.get("level", LEVEL_CELL) == LEVEL_CELL)
           & (df["mode"] == mode)
           & (df["test_id"] != "AVG")].copy()
    if drop_probes and "is_probe" in d.columns:
        n_before = len(d)
        d = d[~_bool(d, "is_probe")]
        if len(d) < n_before:
            print(f"   excluded {n_before - len(d)} probe cells "
                  f"(single exploratory runs; see the probe table)")
    return d


def cell_status(row) -> str:
    """Modal outcome of a cell. Ties break toward the earlier status in
    _STATUS_KEYS (success first), which matches the per-tree figures."""
    best, best_n = "error", -1
    for k in _STATUS_KEYS:
        n = row.get(f"n_{k}", 0) or 0
        if n > best_n:
            best, best_n = k, n
    return best


def common_cells(df: pd.DataFrame, mode: str = "SLAM") -> pd.DataFrame:
    """Restrict to (variant, test) cells present in EVERY condition, so a
    condition that ran fewer variants doesn't look better/worse purely for
    what it attempted. Cells are NOT dropped for unequal repetition counts
    (thin, not invalid) -- n_total travels with the cell instead."""
    d = real_rows(df, mode)
    conds = d["condition"].unique()
    sets = [set(map(tuple, d[d.condition == c][["exp_type", "variant", "test_id"]].values))
            for c in conds]
    keep = set.intersection(*sets) if sets else set()
    mask = d.apply(lambda r: (r.exp_type, r.variant, r.test_id) in keep, axis=1)
    out = d[mask].copy()
    out["cell_status"] = out.apply(cell_status, axis=1)

    print(f"\n common cells across {len(conds)} conditions ({mode}): "
          f"{len(keep)}  ->  {len(out)} rows")
    if "rep_status" in out.columns:
        odd = out[out["rep_status"].isin(["under", "over"])]
        if len(odd):
            print(f"   {len(odd)} of them have a non-standard repetition count "
                  f"(kept, with n visible):")
            for (v, t), g in odd.groupby(["variant", "test_id"]):
                ns = sorted(set(g["n_total"].dropna().astype(int)))
                exp = int(g["n_expected"].dropna().iloc[0]) \
                    if g["n_expected"].notna().any() else "?"
                print(f"     {v:<24} {t:<7} n={ns}  (norm {exp})")
    return out


def provenance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Run counts by provenance and repetition balance, per condition -- the
    data-hygiene table (recovered/orphaned/probe counts, made quotable)."""
    runs = df[df.get("level") == LEVEL_RUN]
    cells = df[df.get("level") == LEVEL_CELL]
    rows = []
    for c in _order():
        for mode in ("SLAM", "VO"):
            r = runs[(runs["condition"] == c.key) & (runs["mode"] == mode)]
            k = cells[(cells["condition"] == c.key) & (cells["mode"] == mode)
                      & (cells["test_id"] != "AVG")]
            if r.empty and k.empty:
                continue
            pv = r["provenance"].value_counts() if not r.empty else pd.Series(dtype=int)
            rs = k["rep_status"].value_counts() if "rep_status" in k else pd.Series(dtype=int)
            rows.append({
                "condition": c.key, "cond_label": c.label, "rung": c.rung,
                "mode": mode,
                "runs_total":  int(len(r)),
                "measured":    int(pv.get("measured", 0)),
                "recovered":   int(pv.get("recovered", 0)),
                "orphaned":    int(pv.get("orphaned", 0)),
                "cells_total": int(len(k)),
                "cells_full":  int(rs.get("full", 0)),
                "cells_under": int(rs.get("under", 0)),
                "cells_over":  int(rs.get("over", 0)),
                "cells_probe": int(rs.get("probe", 0)),
                "n_expected":  int(k["n_expected"].dropna().mode().iloc[0])
                               if "n_expected" in k and k["n_expected"].notna().any()
                               else None,
            })
    return pd.DataFrame(rows)


def probe_cells(df: pd.DataFrame) -> pd.DataFrame:
    """The single-run exploratory cells, listed rather than averaged in --
    these runs are the evidence behind the skip list."""
    cells = df[(df.get("level") == LEVEL_CELL) & _bool(df, "is_probe")]
    if cells.empty:
        return pd.DataFrame()
    cols = [c for c in ("condition", "cond_label", "mode", "test_id", "exp_type",
                        "variant", "family", "n_total", "best_status",
                        C_COMPLETION, C_FPS, C_RAM_PEAK, C_ATE)
            if c in cells.columns]
    return cells[cols].sort_values(["condition", "mode", "variant", "test_id"])


# ════════════════════════════════════════════════════════════════════════════
#  6.  The rung report
# ════════════════════════════════════════════════════════════════════════════
def rung_report(cc: pd.DataFrame, full: pd.DataFrame, out_dir: Path) -> list[str]:
    """Per-rung outcome shift, throughput ratio and memory headroom. Opens
    with data-hygiene numbers (everything after inherits them), and writes
    transitions_<rung>.csv -- every cell whose modal outcome changed (e.g.
    LOST -> SUCCESS on the capacity rung means memory, not algorithm)."""
    lines: list[str] = []
    lab = {c.key: c.label for c in CONDITIONS}

    def w(s: str = "") -> None:
        lines.append(s)
        print(s)

    # ── data hygiene, first, because everything below inherits it ───────────
    w("=" * 74)
    w(" DATA PROVENANCE AND REPETITION BALANCE")
    w("=" * 74)
    prov = provenance_summary(full)
    if prov.empty:
        w("   (no run-level scan in this build -- re-run without --no-run-level)")
    else:
        w(f"   {'condition':<14}{'mode':<6}{'runs':>6}{'meas':>7}{'recov':>7}"
          f"{'ORPHAN':>8}{'cells':>7}{'full':>6}{'under':>7}{'over':>6}{'probe':>7}")
        for _, r in prov.iterrows():
            w(f"   {r.condition:<14}{r['mode']:<6}{r.runs_total:>6}"
              f"{r.measured:>7}{r.recovered:>7}{r.orphaned:>8}"
              f"{r.cells_total:>7}{r.cells_full:>6}{r.cells_under:>7}"
              f"{r.cells_over:>6}{r.cells_probe:>7}")
        n_orph = int(prov["orphaned"].sum())
        n_rec  = int(prov["recovered"].sum())
        n_prb  = int(prov["cells_probe"].sum())
        w("")
        if n_orph:
            w(f"   {n_orph} ORPHANED run(s): a flag row exists but the run "
              f"directory does not.")
            w(f"     They carry a status and an elapsed time and nothing else. "
              f"Excluded from")
            w(f"     every metric; kept in the CSV so the run counts still "
              f"reconcile.")
        else:
            w("   no orphaned runs: every flag row has its directory on disk.")
        if n_rec:
            w(f"   {n_rec} RECOVERED run(s): directory on disk, no flag row. "
              f"Metrics are real.")
        if n_prb:
            w(f"   {n_prb} PROBE cell(s): a single exploratory run where the "
              f"norm is three or more.")
            w(f"     Excluded from every aggregate below; listed in "
              f"probe_cells_table.csv.")

    # ── baseline health: everything rel_* is divided by these ───────────────
    w("")
    w("=" * 74)
    w(" BASELINE HEALTH   (every rel_* column is divided by these numbers)")
    w("=" * 74)
    bh = baseline_health(full[full.get("level") == LEVEL_CELL])
    if bh.empty:
        w(f"   [!] no '{BASELINE_VARIANT}' cells found - the relative columns "
          f"are empty")
    else:
        for mode in sorted(bh["mode"].dropna().unique()):
            w(f"   {mode}")
            sub = bh[bh["mode"] == mode]
            w(f"     {'seq':<7}{'condition':<14}{'compl':>7}{'ATE m':>9}"
              f"{'fps':>8}{'peakGB':>8}")
            for _, r in sub.iterrows():
                def _n(v, d=2):
                    return f"{v:.{d}f}" if pd.notna(v) else "  --"
                w(f"     {r.test_id:<7}{r.condition:<14}"
                  f"{int(r.n_success)}/{int(r.n_total):<4}"
                  f"{_n(r.get(C_ATE), 3):>9}{_n(r.get(C_FPS)):>8}"
                  f"{_n(r.get(C_RAM_PEAK)):>8}")
        w("")
        w("   A baseline out of line with its three neighbours rescales every "
          "ratio on that")
        w("   sequence without anything downstream looking wrong. Check this "
          "block first.")

    w("")
    w("=" * 74)
    w(" RUNG REPORT   (common cells only; each rung changes ONE factor)")
    w("=" * 74)

    for src, dst, name in RUNGS:
        A = cc[cc.condition == src].set_index(["exp_type", "variant", "test_id"])
        B = cc[cc.condition == dst].set_index(["exp_type", "variant", "test_id"])
        if A.empty or B.empty:
            continue
        idx = A.index.intersection(B.index)
        A, B = A.loc[idx], B.loc[idx]

        w("")
        w(f" RUNG: {name.upper()}   {lab.get(src, src)}  ->  {lab.get(dst, dst)}")
        w(" " + "-" * 72)

        nA, nB = A.n_total.sum(), B.n_total.sum()
        w(f"   cells {len(idx)}   runs {nA:.0f} -> {nB:.0f}")
        for k in _STATUS_KEYS:
            a, b = A[f"n_{k}"].sum(), B[f"n_{k}"].sum()
            if a == 0 and b == 0:
                continue
            w(f"   {STATUS_LABEL[k]:<8} {a:5.0f} -> {b:5.0f}"
              f"   ({a/nA*100:5.1f}% -> {b/nB*100:5.1f}%)   delta {b-a:+.0f}")

        # budget-attributable vs algorithm-attributable, split explicitly
        budA = A["n_timeout"].sum() + A["n_oom"].sum()
        budB = B["n_timeout"].sum() + B["n_oom"].sum()
        algA = A["n_success"].sum() + A["n_lost"].sum()
        algB = B["n_success"].sum() + B["n_lost"].sum()
        w(f"   budget failures (oom+timeout)  {budA:5.0f} -> {budB:5.0f}"
          f"   delta {budB-budA:+.0f}")
        if algA and algB:
            w(f"   completion on the algorithm-attributable subset  "
              f"{A['n_success'].sum()/algA*100:5.1f}% -> "
              f"{B['n_success'].sum()/algB*100:5.1f}%"
              f"   (n {algA:.0f} -> {algB:.0f})")

        # throughput
        pair = pd.DataFrame({"a": A[C_FPS], "b": B[C_FPS]}).dropna()
        if len(pair):
            ratio = (pair.b / pair.a).replace([np.inf, -np.inf], np.nan).dropna()
            w(f"   fps      {pair.a.mean():5.2f} -> {pair.b.mean():5.2f}"
              f"   ratio median {ratio.median():.3f}"
              f"   slower in {(ratio < 1).sum()}/{len(ratio)} cells"
              f"   {wilcoxon_note(pair.a.values, pair.b.values)}")

        # completion + memory headroom
        w(f"   completion {A[C_COMPLETION].mean():5.1f}% -> "
          f"{B[C_COMPLETION].mean():5.1f}%")
        if C_RAM_PEAK in A.columns:
            w(f"   peak RAM   {A[C_RAM_PEAK].mean():5.2f} GB -> "
              f"{B[C_RAM_PEAK].mean():5.2f} GB")
        if C_THREADS in A.columns:
            w(f"   busy thr   {A[C_THREADS].mean():5.2f} -> "
              f"{B[C_THREADS].mean():5.2f}")

        # ── status transitions ──────────────────────────────────────────────
        tr = pd.DataFrame({"from": A.cell_status, "to": B.cell_status})
        changed = tr[tr["from"] != tr["to"]]
        if len(changed):
            w(f"   cells whose outcome changed: {len(changed)}/{len(tr)}")
            counts = changed.groupby(["from", "to"]).size().sort_values(ascending=False)
            for (f_, t_), n in counts.items():
                flag = "  <<< " if (f_ != "success" and t_ == "success") else "      "
                w(f"{flag}{STATUS_LABEL[f_]:>8} -> {STATUS_LABEL[t_]:<8} {n:3d}")
            det = changed.join(A[["n_total", C_COMPLETION]]
                               .rename(columns={C_COMPLETION: "compl_from"}))
            det = det.join(B[[C_COMPLETION]]
                           .rename(columns={C_COMPLETION: "compl_to"}))
            if C_RAM_PEAK in A.columns:
                det["ram_gb_from"] = A[C_RAM_PEAK]
                det["ram_gb_to"]   = B[C_RAM_PEAK]
            det.sort_values(["from", "to"]).to_csv(
                out_dir / f"transitions_{name.replace(' ', '_')}.csv")

    # ── accuracy invariance across ALL conditions ───────────────────────────
    w("")
    w("=" * 74)
    w(" ACCURACY INVARIANCE  (cells completing >=98% under EVERY condition)")
    w("=" * 74)
    piv = cc.pivot_table(index=["exp_type", "variant", "test_id"],
                         columns="condition",
                         values=[C_COMPLETION, C_ATE,
                                 C_RPE_SHORT])
    conds = [c.key for c in CONDITIONS if c.key in cc.condition.unique()]
    try:
        ok = (piv[C_COMPLETION][conds] >= 98).all(axis=1)
        ate = piv[C_ATE][conds][ok].dropna()
    except KeyError:
        ate = pd.DataFrame()
    w(f"   n = {len(ate)} cells")
    if len(ate):
        for c in conds:
            w(f"   {lab.get(c, c):<20} ATE median {ate[c].median():7.3f} m"
              f"   mean {ate[c].mean():8.3f} m")

        # Friedman needs >=3 matched groups; fall back to paired Wilcoxon with 2.
        if _scipy_stats is not None and len(ate) >= 5:
            if len(conds) >= 3:
                p = _scipy_stats.friedmanchisquare(*[ate[c].values for c in conds]).pvalue
                w(f"   Friedman across {len(conds)} conditions: p = {p:.3f}"
                  f"   -> {'NO detectable difference' if p > 0.05 else 'difference detected'}")
            elif len(conds) == 2:
                p = _scipy_stats.wilcoxon(ate[conds[0]].values, ate[conds[1]].values).pvalue
                w(f"   only 2 conditions so far -> paired Wilcoxon: p = {p:.3f}"
                  f"   -> {'NO detectable difference' if p > 0.05 else 'difference detected'}"
                  f"   (re-run once a 3rd condition exists for the full Friedman test)")
        else:
            w("   (need >=5 cells and scipy installed for a significance test)")
        w("   NOTE: quote the median. The mean is driven by a handful of "
          "high-variance variants and moves without the ranking moving.")
    (out_dir / "rung_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


# ════════════════════════════════════════════════════════════════════════════
#  7.  Figures
# ════════════════════════════════════════════════════════════════════════════
def _order() -> list[Condition]:
    return sorted(CONDITIONS, key=lambda c: c.rung)


_RELIEF_KIND_STYLE = {
    # key             colour     panel-B marker  group heading
    "rescued":       ("#1e8449", "o", "RESCUED — failed at the small budget, completed at the large one"),
    "worse":         ("#c0392b", "X", "REGRESSED — completed at the small budget, failed at the large one"),
    "still_failing": ("#b45309", "s", "STILL FAILING — memory was not the binding constraint"),
    "control":       ("#9e9e9e", "o", "CONTROL — completed at both budgets"),
}
_RELIEF_ORDER = ["rescued", "worse", "still_failing", "control"]


# ════════════════════════════════════════════════════════════════════════════
#  9.  LaTeX tables
# ════════════════════════════════════════════════════════════════════════════
def _tab_dir(out_dir: Path) -> Path:
    """LaTeX tables go in a sibling tables/ dir, not inside out_dir (csv/):
    keeps the thesis's table-include path separate from the data files.
    The .csv twin of each table still lives beside the data."""
    d = out_dir.parent / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tex(s: str) -> str:
    """Underscores are everywhere in variant names and nowhere legal in LaTeX
    text mode."""
    return str(s).replace("_", r"\_")


def write_provenance_table(full: pd.DataFrame, out_dir: Path) -> None:
    """The data-hygiene table: runs by provenance, cells by repetition
    balance. Belongs in the appendix, but belongs somewhere."""
    prov = provenance_summary(full)
    if prov.empty:
        return
    prov.to_csv(out_dir / "provenance_table.csv", index=False)

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Data provenance and repetition balance per condition. "
        r"\emph{Measured} runs have both a runner log entry and a run "
        r"directory; \emph{recovered} runs have only the directory, and their "
        r"metrics are real; \emph{orphaned} runs have only the log entry, "
        r"carry no metrics, and are excluded from every aggregate. "
        r"\emph{Probe} cells are single exploratory runs where the norm is "
        r"three, and are excluded from every aggregate.}",
        r"\label{tab:provenance}",
        r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
        r" & & \multicolumn{4}{c}{Runs} & \multicolumn{5}{c}{Cells} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-11}",
        r"Condition & Mode & Total & Meas. & Recov. & Orph. & "
        r"Total & Full & Under & Over & Probe \\", r"\midrule",
    ]
    for _, r in prov.iterrows():
        lines.append(
            f"{r['cond_label']} & {r['mode']} & {r['runs_total']} & "
            f"{r['measured']} & {r['recovered']} & {r['orphaned']} & "
            f"{r['cells_total']} & {r['cells_full']} & {r['cells_under']} & "
            f"{r['cells_over']} & {r['cells_probe']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (_tab_dir(out_dir) / "provenance_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_health_table(full: pd.DataFrame, out_dir: Path) -> None:
    """The ORB2 baseline's own absolute numbers, per condition and sequence --
    every rel_* column divides by one of these, so this is the audit table."""
    bh = baseline_health(full[full.get("level") == LEVEL_CELL])
    if bh.empty:
        return
    bh.to_csv(out_dir / "baseline_health_table.csv", index=False)
    slam = bh[bh["mode"] == "SLAM"]
    if slam.empty:
        slam = bh

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{The ORB2 baseline's own absolute numbers on every "
        r"(condition, sequence). Every relative figure elsewhere in this work "
        r"is divided by the value on its own row, so a baseline out of line "
        r"with its neighbours would rescale a whole sequence of comparisons.}",
        r"\label{tab:baseline-health}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Seq. & Condition & Compl. & ATE (m) & fps & Peak RAM (GB) \\",
        r"\midrule",
    ]
    cur = None
    for _, r in slam.iterrows():
        def _f(v, d=2):
            return f"{v:.{d}f}" if pd.notna(v) else "--"
        seq = r["test_id"] if r["test_id"] != cur else ""
        cur = r["test_id"]
        lines.append(
            f"\\texttt{{{seq}}} & {r['cond_label']} & "
            f"{int(r['n_success'])}/{int(r['n_total'])} & "
            f"{_f(r.get(C_ATE), 3)} & {_f(r.get(C_FPS))} & "
            f"{_f(r.get(C_RAM_PEAK))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (_tab_dir(out_dir) / "baseline_health_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_probe_table(full: pd.DataFrame, out_dir: Path) -> None:
    """The single-run exploratory cells, listed rather than pooled into a
    mean they'd distort -- these are the evidence behind the skip list."""
    df = probe_cells(full)
    if df.empty:
        return
    df.to_csv(out_dir / "probe_cells_table.csv", index=False)

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{The {len(df)} probe cells: single exploratory runs, used "
        r"to decide whether a configuration was worth a full set of "
        r"repetitions. They are the evidence behind the skip list and are "
        r"excluded from every aggregate in this work.}",
        r"\label{tab:probe-cells}",
        r"\begin{tabular}{lllllrr}", r"\toprule",
        r"Condition & Mode & Seq. & Variant & Outcome & Compl.\ (\%) & fps \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        def _f(v, d=1):
            return f"{v:.{d}f}" if pd.notna(v) else "--"
        lines.append(
            f"{r.get('condition', '')} & {r.get('mode', '')} & "
            f"{r.get('test_id', '')} & \\texttt{{{_tex(r.get('variant', ''))}}} & "
            f"{r.get('best_status', '')} & {_f(r.get(C_COMPLETION))} & "
            f"{_f(r.get(C_FPS), 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (_tab_dir(out_dir) / "probe_cells_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
#  10.  Data dictionary for all_conditions.csv
# ════════════════════════════════════════════════════════════════════════════
def write_data_dictionary(df: pd.DataFrame, out_dir: Path) -> None:
    """A README that travels with the CSV, explaining its own shape -- in
    particular that three aggregation levels share the file, so a naive sum
    over any count column double counts unless filtered by `level`."""
    n_run  = int((df.get("level") == LEVEL_RUN).sum())
    n_cell = int((df.get("level") == LEVEL_CELL).sum())
    n_cond = int((df.get("level") == LEVEL_COND).sum())

    txt = f"""# all_conditions.csv

One file, the whole campaign. {len(df)} rows: {n_run} run-level,
{n_cell} cell-level, {n_cond} condition-level.

## Read this first

Every row carries a `level`. **Filter on it before doing anything else.**
The three levels describe the same runs at three degrees of aggregation, so
summing a count column across the unfiltered file counts every run at least
twice.

| level | one row is | use it for |
|---|---|---|
| `run` | one individual execution | repetition spread, provenance, per-run failure detail |
| `cell` | one (condition, mode, sequence, variant) | every figure and table in the thesis |
| `condition` | one hardware condition | the headline outcome accounting |

## Keys

- `condition`, `rung`, `cond_label`, `cpu`, `threads`, `ram_budget` -- the
  hardware. `rung` is position on the ladder, 1 = first.
- `mode` -- `SLAM` (loop closure on) or `VO` (loop closure off).
- `test_id` -- sequence. `AVG` rows are roll-ups across sequences and should be
  excluded from anything that also reads the individual sequences.
- `exp_type`, `variant`, `short_name`, `family`, `sensor` -- what was run.

## Provenance (level=run only)

- `measured` -- runner log entry AND run directory both present.
- `recovered` -- directory only. The run happened; the log missed it. Metrics
  are real.
- `orphaned` -- log entry only, directory gone. **Carries no metrics.** Present
  so run counts reconcile and the gap is visible. Never aggregate these.

## Repetition balance (level=cell only)

- `n_total` -- repetitions actually run. Print it beside every median.
- `n_expected` -- the modal repetition count for that (condition, mode).
- `rep_status` -- `full` / `under` / `over` / `probe`.
- `is_probe` -- single exploratory run where the norm is three or more. These
  are evidence for the skip list, not measurements of the method, and are
  excluded from every aggregate. `under` and `over` cells ARE included; their
  `n_total` travels with them.

## Metric suffixes (level=cell)

- `_succ_*` -- over completed runs only. All accuracy keys take this form only,
  because a run that died at 30 % has its ATE scored on the easy opening
  stretch, so pooling it with completers makes a variant look better the
  earlier it fails.
- `_all_*` -- over every run that produced data. Machine load and throughput
  use this: an OOM-killed run still truthfully reports what it was holding.

## Relative columns

`rel_<column>` is that column divided by the ORB2 baseline's value on the
**same (condition, mode, sequence)**. Absolute figures from this bench are
partly a property of the bench; the ratios are the part that should survive a
native reimplementation.

- Ratio of medians, not median of ratios.
- Accuracy ratios are **null**, not zero, where the baseline itself did not
  complete on that sequence. `baseline_completed` says which rows those are.
- Normalised per sequence, because the sequences differ from each other by more
  than the methods differ from each other.

## Outcome labels

`success`, `lost`, `timeout`, `oom`, `error`. They are not interchangeable:

- `lost` is **algorithm-attributable** -- the tracker ran out of usable
  correspondences.
- `timeout` and `oom` are **budget-attributable** -- the same run on a larger
  machine would have continued.

Component claims should be scored on the algorithm-attributable subset
(`n_success / (n_success + n_lost)`); hardware claims on the budget-attributable
failures (`n_timeout + n_oom`). At the largest budget the two coincide, because
nothing dies of memory there.
"""
    (out_dir / "all_conditions_DICTIONARY.md").write_text(txt, encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
#  11.  FAILURE-SPACE MAP ACROSS THE LADDER   (ported from aggregate_results.py)
# ════════════════════════════════════════════════════════════════════════════
# One panel per CONDITION instead of per test: deaths that SCATTER and vanish
# as budget grows are budget-caused; deaths that CLUSTER and persist are
# algorithm-caused. MARK_AT_DEATH forces every dot to the last processed
# frame (not first_loss_frame, as the per-condition map uses for LOST) so
# panels stay comparable.
MARK_AT_DEATH = True

_LADDER_MAP_BENCHES = [
    ("feature_extraction", "fe"),
    ("loop_closure",       "lc"),
    ("combined",           "co"),
]


# ════════════════════════════════════════════════════════════════════════════
#  MAP GROWTH  (Chapter 4: ds_map_growth.png)
# ════════════════════════════════════════════════════════════════════════════
# Keyframes/map points accumulated against distance driven. Lives here rather
# than in make_results_assets.py because that script only reads
# all_conditions.csv (one final number per run), while the growth curve only
# exists in each run's own progress.csv -- a raw-directory read like the
# failure map above already does.
#
# Shares an assumption with fig_failure_map_ladder above: frame_id indexes
# directly into the GT pose array (clamped to n-1) as "distance so far". Fix
# in both places if that convention is ever wrong.

# progress.csv columns: frame_id(2) num_kf(5) num_map_pts(6). Unlike
# read_progress_csv (last value only), this keeps the whole series.
_KF_GROWTH_COLS = {"frame_id": 2, "num_kf": 5, "num_map_pts": 6}


def _read_kf_growth(progress_csv: Path) -> np.ndarray | None:
    """(frame_id, num_kf, num_map_pts) for every logged row, or None. Returns
    an (N, 3) float array in ascending frame_id order; rows missing frame_id
    are dropped, missing num_kf/num_map_pts are forward-filled (both are
    monotonic non-decreasing within a run)."""
    if not progress_csv.is_file():
        return None
    fid_i, kf_i, mp_i = (_KF_GROWTH_COLS[k] for k in ("frame_id", "num_kf",
                                                       "num_map_pts"))
    rows = []
    last_kf, last_mp = 0.0, 0.0
    try:
        with open(progress_csv, newline="", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) <= max(fid_i, kf_i, mp_i) or parts[0] == "unix_ts":
                    continue
                try:
                    fid = float(parts[fid_i])
                except ValueError:
                    continue
                try:
                    last_kf = float(parts[kf_i]) if parts[kf_i].strip() else last_kf
                except (ValueError, IndexError):
                    pass
                try:
                    last_mp = float(parts[mp_i]) if parts[mp_i].strip() else last_mp
                except (ValueError, IndexError):
                    pass
                rows.append((fid, last_kf, last_mp))
    except Exception:
        return None
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    return arr[np.argsort(arr[:, 0])]


def _pick_growth_run(scan_root: Path, test_id: str, run_flags: dict,
                     expected: int | None) -> Path | None:
    """One representative run directory for the baseline on one sequence.
    Prefers a completed run (a truncated curve looks identical to a short
    sequence unless disclosed); falls back to the furthest-progressed
    repetition otherwise, and the caller should label that case."""
    # variants.py runs the baseline once per (mode, test_id) into
    # results/{mode}/{test_id}/baseline/, without the extra
    # <exp_type>/<short_name> nesting other phases have. Try both.
    var_dir = next((d for d in (scan_root / test_id / "baseline" / "baseline",
                                scan_root / test_id / "baseline")
                    if d.is_dir()), None)
    if var_dir is None:
        return None
    best, best_processed, best_status = None, -1, None
    for rd in agg._run_dirs_for_variant(var_dir):
        flag_row = agg.match_run_flag(run_flags, "baseline", test_id, rd.name)
        try:
            status, m = agg.classify_run(rd, flag_row, expected)
        except Exception:
            continue
        if status == "success":
            return rd
        processed = (m or {}).get("processed") or 0
        if processed > best_processed:
            best, best_processed, best_status = rd, processed, status
    if best is not None:
        print(f"   [!] ds_map_growth {test_id}: no successful baseline run, "
              f"using a {best_status} run ({best_processed} frames)")
    return best


# ════════════════════════════════════════════════════════════════════════════
#  12.  main
# ════════════════════════════════════════════════════════════════════════════

def write_map_growth_csv(cond, mode: str, out_dir: Path) -> None:
    """map_growth.csv -- keyframes and map points against distance driven,
    one row per logged frame, for the baseline run on each sequence at ONE
    condition. The only per-frame series this pipeline exports, since
    all_conditions.csv only holds a final num_keyframes per run. `cond`
    should be the largest-memory condition, so the curve isn't cut short by
    an OOM unrelated to the road (see Section 8.2)."""
    tree = PC_ROOT / cond.folder
    scan_root = tree / mode
    if not scan_root.is_dir():
        print(f" [!] map_growth: {scan_root} not found")
        return

    from src_pc.utils.metrics_helper_functions import load_tum_trajectory

    run_flags = load_run_flags(tree)
    expected_frames = load_expected_frames()
    try:
        test_ids = all_test_ids()
    except SystemExit:
        return

    rows = []
    for test_id in test_ids:
        expected = expected_frames.get(test_id)
        rd = _pick_growth_run(scan_root, test_id, run_flags, expected)
        if rd is None:
            print(f" [!] map_growth {test_id}: no baseline run found")
            continue
        growth = _read_kf_growth(rd / "files" / "progress.csv")
        if growth is None:
            print(f" [!] map_growth {test_id}: no progress.csv under {rd}")
            continue

        gt_files = sorted((agg.PROCESSED_POSES / test_id).glob("*_tum.txt"))
        if not gt_files:
            print(f" [!] map_growth {test_id}: no ground-truth poses")
            continue
        xyz = load_tum_trajectory(gt_files[0]).xyz
        n = len(xyz)
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumdist = np.concatenate([[0.0], np.cumsum(step)])

        fid = np.clip(growth[:, 0].astype(int), 0, n - 1)
        for f, kf, mp, d in zip(fid, growth[:, 1], growth[:, 2],
                                cumdist[fid]):
            rows.append({"condition": cond.key, "mode": mode,
                         "test_id": test_id, "frame_id": int(f),
                         "distance_m": float(d),
                         "num_keyframes": float(kf),
                         "map_points": float(mp)})

    if not rows:
        print(" [!] map_growth: nothing to write")
        return
    path = out_dir / "map_growth.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f" wrote {path}  ({len(rows)} frame rows, "
          f"{len({r['test_id'] for r in rows})} sequences)")


def main() -> None:
    global CONDITIONS, RUNGS

    ap = argparse.ArgumentParser(
        description="Aggregate and compare every results tree on the ladder. "
                    "Always rebuilds everything from the raw run folders.")
    ap.add_argument("--root", type=Path, default=PC_ROOT,
                    help="directory to scan for results* trees "
                         "(default: repo root)")
    ap.add_argument("--out", type=Path, default=OUT_DIR,
                    help="CSV output directory. DELETED AND REBUILT "
                         "on every run.")
    ap.add_argument("--report", action="store_true",
                    help="also write rung_report.txt and transitions_*.csv")
    ap.add_argument("--mode", default="SLAM", choices=["SLAM", "VO"])
    ap.add_argument("--no-run-level", action="store_true",
                    help="skip the per-run scan. Much faster, but "
                         "all_conditions.csv then holds no run rows and the "
                         "provenance report cannot distinguish a recovered "
                         "run from an orphaned one.")
    args = ap.parse_args()

    print("=" * 74)
    print(" aggregate_all   -   CSV builder")
    print(" full rebuild: every tree is re-aggregated, the CSV is recreated")
    print(" figures and tables are make_results_assets.py's job, not this one")
    print("=" * 74)

    # exclude the comparison output dir itself, in case it starts with "results"
    CONDITIONS = discover_conditions(args.root, exclude={args.out.name})
    RUNGS = build_rungs(CONDITIONS)

    print(f" found {len(CONDITIONS)} condition(s) under {args.root}:")
    for c in CONDITIONS:
        print(f"   rung {c.rung}  {c.key:<14} {c.label:<24} <- {c.folder}")
    if RUNGS:
        print(" ladder:")
        for src, dst, name in RUNGS:
            print(f"   {src}  ->  {dst}   isolates: {name}")
    if len(CONDITIONS) < 2:
        sys.exit("Need at least 2 conditions to compare - only 1 found.")

    # Re-aggregate every tree first, so the CSVs read are never leftovers.
    df = load_all(with_runs=not args.no_run_level, mode=args.mode)

    # Only now clear the comparison dir, so a crash mid-aggregation leaves
    # the previous comparison intact instead of destroying it for nothing.
    out_dir = rmtree_fresh(args.out)
    print(f"\n rebuilt {out_dir}/")

    df.to_csv(out_dir / "all_conditions.csv", index=False)
    lv = df["level"].value_counts().to_dict() if "level" in df else {}
    print(f" wrote {out_dir / 'all_conditions.csv'}  ({len(df)} rows: "
          + ", ".join(f"{v} {k}" for k, v in sorted(lv.items())) + ")")
    write_data_dictionary(df, out_dir)
    print(f" wrote {out_dir / 'all_conditions_DICTIONARY.md'}")

    # Per-frame map growth, exported at the most capable condition.
    try:
        write_map_growth_csv(CONDITIONS[-1], args.mode, out_dir)
    except Exception as e:
        print(f" [!] map_growth.csv failed: {e}")

    # Data-hygiene tables: describe the collection, not the results.
    for name, fn in (("provenance_table",      write_provenance_table),
                     ("baseline_health_table", write_baseline_health_table),
                     ("probe_cells_table",     write_probe_table)):
        try:
            fn(df, out_dir)
            print(f" wrote {_tab_dir(out_dir) / (name + '.tex')}")
        except Exception as e:
            print(f" [!] {name} failed: {e}")

    # Prose plus a few transition CSVs, not the data file itself -- opt-in.
    if args.report:
        cc = common_cells(df, mode=args.mode)
        if cc.empty:
            sys.exit("No cells are common to every condition - "
                     "check the folder names.")
        rung_report(cc, df, out_dir)

    report_unfamilied()
    print(f"\n done -> {out_dir}/")
    print(" next:  python -m src_pc.utils.make_assets_offline")


if __name__ == "__main__":
    main()