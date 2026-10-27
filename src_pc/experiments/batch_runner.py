#!/usr/bin/env python3
"""
src_pc/experiments/batch_runner.py

Runs a batch of SLAM experiments across variants and test sequences, each as its
own slam_runner.py subprocess in its own process group.

Every run gets SIGKILL'd (whole process group) in a finally: block no matter how
it exits - clean, OOM, timeout, or Ctrl+C - so pyslam workers never stick around.
kill_all_slam_processes() then does a second psutil sweep to confirm nothing survived.

Usage:
  python3 src_pc/experiments/batch_runner.py --fe  --test all
  python3 src_pc/experiments/batch_runner.py --all --test all
  python3 src_pc/experiments/batch_runner.py --variant fe_superpoint --test test0
  python3 src_pc/experiments/batch_runner.py --fe  --test test0 --dry-run
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

# Project roots ----------------------------------------------------------------------------
PC_ROOT     = Path(__file__).resolve().parents[2]
PYSLAM_ROOT = PC_ROOT.parent / "pyslam"

sys.path.insert(0, str(PC_ROOT))
sys.path.insert(0, str(PYSLAM_ROOT))

os.chdir(str(PC_ROOT))

from src_pc.configs.project_settings import (
    INTER_RUN_SLEEP, MAX_RUN_WALL_TIME_S, SETUP_GRACE_S,
)
from src_pc.experiments.variants import (
    EXPERIMENT_TYPES, list_variants, VARIANTS,
    BASELINE_VARIANT, BASELINE_PHASES,
)
from src_pc.utils.paths import RESULTS

# Timing constants (module-level so test_models.py can import them) ------------------------
HARD_TIMEOUT_S = MAX_RUN_WALL_TIME_S + SETUP_GRACE_S
SIGINT_GRACE_S = 30   # wait after SIGINT on hard timeout before SIGKILL
CTRLC_GRACE_S  = 5    # wait after SIGINT on Ctrl+C before SIGKILL (short!)

RUNNER = PC_ROOT / "src_pc" / "experiments" / "slam_runner.py"

# marks the hardest test in the final printout
HARD_TEST = "test3"

# grace period to let teardown finish once the result file exists, instead of
# waiting out the full HARD_TIMEOUT_S on a hung slam.quit().
TEARDOWN_GRACE_S = 30

# -9 = kernel OOM SIGKILL, -15 = SIGTERM from an OOM daemon - RAM pressure, not a code bug.
OOM_SIGNALS = (-9, -15)


def _signame(returncode: int | None) -> str:
    """Turn a negative (signal-killed) returncode into a readable name, else ''."""
    if returncode is None or returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except (ValueError, KeyError):
        return f"signal {-returncode}"


def _test_tag(test_id: str) -> str:
    """Append a marker to the hardest test so it stands out in the printout."""
    return f"{test_id} *" if test_id == HARD_TEST else test_id


# Shared subprocess-management functions  (importable) -------------------------------------

def killpg(pgid: int, sig: int) -> None:
    """Send signal to a process group, silently ignoring 'no such process'."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    except Exception as e:
        print(f"  [!] killpg({pgid}, {sig}): {e}")


def kill_all_slam_processes() -> int:
    """
    SIGKILL any slam_runner / pyslam worker process that's still alive, then wait
    on psutil.wait_procs until they're actually confirmed dead.
    Returns how many processes we killed.
    """
    venv_marker = str(PYSLAM_ROOT)
    targets = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if (venv_marker in cmdline or "slam_runner" in cmdline) \
                    and proc.pid != os.getpid():
                targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not targets:
        return 0

    for proc in targets:
        try:
            proc.send_signal(signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Block until confirmed dead (up to 10 s)
    _, alive = psutil.wait_procs(targets, timeout=10)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return len(targets)


def run_one(
    cmd: list[str],
    timeout_s: int | None = None,
    result_dir: Path | None = None,
    teardown_grace_s: int | None = None,
) -> tuple[int | None, float, bool, str]:
    """
    Launch cmd in its own process group with a hard timeout; the group is ALWAYS
    SIGKILL'd in finally so pyslam workers never linger and hog RAM.

    If result_dir is given, once the run's result appears on disk a hung teardown
    is only given `teardown_grace_s` (not the full timeout) - returned as
    killed_signal="teardown" so the caller classifies by the dashboard, not by -9.

    Returns: (returncode, elapsed_s, timed_out, killed_signal)
    """
    timeout = timeout_s if timeout_s is not None else HARD_TIMEOUT_S
    grace   = teardown_grace_s if teardown_grace_s is not None else TEARDOWN_GRACE_S
    t0   = time.time()
    proc = subprocess.Popen(cmd, cwd=str(PC_ROOT), start_new_session=True)
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    timed_out         = False
    killed_signal     = ""
    teardown_deadline = None   # set once the result is detected on disk

    try:
        while True:
            try:
                proc.wait(timeout=5)
                break                       # process exited on its own
            except subprocess.TimeoutExpired:
                now = time.time()

                # Result already on disk - heavy work is done; only teardown remains.
                if result_dir is not None and _result_written(result_dir, t0):
                    if teardown_deadline is None:
                        print(f"\n  [i] Result written — allowing {grace}s for clean exit ...")
                        teardown_deadline = now + grace
                    elif now >= teardown_deadline:
                        print(f"  [!] Teardown hung after result written — "
                              f"killing group {pgid}")
                        killpg(pgid, signal.SIGINT)
                        killed_signal = "teardown"
                        try:
                            proc.wait(timeout=SIGINT_GRACE_S)
                        except subprocess.TimeoutExpired:
                            killpg(pgid, signal.SIGKILL)
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                pass
                        break

                if now - t0 >= timeout:
                    timed_out = True
                    print(f"\n  [!] HARD TIMEOUT ({timeout}s) — SIGINT to group {pgid}")
                    killpg(pgid, signal.SIGINT)
                    killed_signal = "SIGINT"
                    try:
                        proc.wait(timeout=SIGINT_GRACE_S)
                    except subprocess.TimeoutExpired:
                        print(f"  [!] Still alive — SIGKILL to group {pgid}")
                        killpg(pgid, signal.SIGKILL)
                        killed_signal = "SIGKILL"
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                    break

    except KeyboardInterrupt:
        print(f"\n  [!] Ctrl+C — killing run (group {pgid}), stopping ...")
        killpg(pgid, signal.SIGINT)
        killed_signal = "SIGINT"
        try:
            proc.wait(timeout=CTRLC_GRACE_S)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            killpg(pgid, signal.SIGKILL)
            killed_signal = "SIGKILL"
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                pass
        raise

    finally:
        # always kill the group so pyslam workers can't leak RAM
        killpg(pgid, signal.SIGKILL)

    elapsed = time.time() - t0
    return proc.returncode, elapsed, timed_out, killed_signal


# Batch-runner helpers  (internal to batch mode) -------------------------------------------

FLAG_CSV = RESULTS / "run_flags.csv"
FLAG_FIELDS = [
    "timestamp", "variant", "test_id", "run_num", "status", "returncode",
    "elapsed_s", "wall_limit_s", "hard_limit_s", "killed_signal", "command",
]


# in VO mode loop closure is off, so the loop_closure phase just collapses back to
# the baseline, and backend/combined don't make sense - only FE and depth run.
VO_EXP_TYPES = ["feature_extraction", "depth_estimation"]


def _completed_baseline_runs(mode: str, test_id: str) -> int:
    """
    How many finished baseline runs already exist for this (mode, test_id).
    Only a run with a dashboard.txt counts - a crashed/partial one doesn't, so it
    gets picked up and retried on the next batch.
    """
    base = RESULTS / mode / test_id / "baseline"
    if not base.is_dir():
        return 0
    return sum(1 for d in base.iterdir()
               if d.is_dir() and (d / "dashboard.txt").is_file())


def _ram_free_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _ram_pct() -> float:
    return psutil.virtual_memory().percent


def _log_flag(**row) -> None:
    new_file = not FLAG_CSV.exists()
    FLAG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAG_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


_DATASET_SUMMARY = PC_ROOT / "docs" / "dataset" / "dataset_summary.txt"
_TEST_FRAMES: dict[str, str] = {}   # populated once on first call

def _test_frame_counts() -> dict[str, str]:
    """Return {test_id: total_frames_str} parsed from dataset_summary.txt."""
    if _TEST_FRAMES:
        return _TEST_FRAMES
    try:
        text = _DATASET_SUMMARY.read_text(encoding="utf-8")
        current = None
        for line in text.splitlines():
            stripped = line.strip()
            # "  test1  /  ..."
            if stripped.startswith("test") and "/" in stripped:
                current = stripped.split("/")[0].strip()  # e.g. "test1"
            elif current and stripped.startswith("frames"):
                _, _, val = stripped.partition(":")
                _TEST_FRAMES[current] = val.strip()
                current = None
    except OSError:
        pass
    return _TEST_FRAMES


def _last_csv_line(csv_path: Path) -> str | None:
    """Grab the last non-empty line of csv_path by only reading the tail, not the whole file."""
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            f.seek(-min(4096, size), 2)
            chunk = f.read()
        for line in reversed(chunk.decode("utf-8", errors="replace").splitlines()):
            if line.strip():
                return line.strip()
    except OSError:
        pass
    return None


def _variant_result_dir(mode: str, vname: str, test_id: str) -> Path:
    """Folder that holds the timestamped run dirs for (mode, vname, test_id)."""
    v = VARIANTS.get(vname, {})
    exp_type = v.get("experiment_type", "")
    short    = v.get("short_name", vname)
    if exp_type == "baseline":
        return RESULTS / mode / test_id / "baseline"
    if exp_type:
        return RESULTS / mode / test_id / exp_type / short
    return RESULTS / mode / test_id / vname   # fallback (unknown variant)


def _result_written(result_dir: Path, since: float) -> bool:
    """True if a run dir created at/after `since` has its final dashboard.txt.
    Only dashboard.txt counts (not trajectory_final.txt, which TrajectoryWriter
    creates empty at startup - using it tripped the teardown killer mid-run)."""
    if not result_dir.is_dir():
        return False
    try:
        for d in result_dir.iterdir():
            if not d.is_dir() or d.stat().st_mtime < since - 5:
                continue
            if (d / "dashboard.txt").is_file():
                return True
    except OSError:
        pass
    return False


def _read_latest_dashboard(mode: str, vname: str, test_id: str) -> dict | None:
    """Pull key fields from the newest run dir for (mode, vname, test_id).
    Prefers dashboard.txt; falls back to files/progress.csv (survives a SIGKILL)
    if the process was hard-killed before writing a dashboard."""
    run_dir = _variant_result_dir(mode, vname, test_id)
    if not run_dir.is_dir():
        return None

    # Newest timestamp dir first (by mtime)
    all_dirs = sorted(
        (d for d in run_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not all_dirs:
        return None

    newest = all_dirs[0]

    # first choice: the dashboard.txt
    dash_file = newest / "dashboard.txt"
    if dash_file.is_file():
        _KEYS = {"abort_reason", "processed", "total_frames", "first_loss_at"}
        info = {}
        for line in dash_file.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lstrip()
            if key in _KEYS:
                info[key] = val.strip()
        return info or None

    # fallback: last row of progress.csv (frame_id there is 0-based)
    csv_path = newest / "files" / "progress.csv"
    if csv_path.is_file():
        last = _last_csv_line(csv_path)
        if last:
            parts = last.split(",")
            if len(parts) > 2:
                try:
                    info = {"processed": str(int(parts[2].strip()) + 1)}
                    total = _test_frame_counts().get(test_id)
                    if total:
                        info["total_frames"] = total
                    return info
                except ValueError:
                    pass

    return None


def _frame_info(dash: dict | None) -> str:
    """Return compact progress string, e.g. '1187 / 1547 frames (77%)'."""
    if not dash:
        return "? / ? frames"
    processed = dash.get("processed", "?")
    total     = dash.get("total_frames", "?")
    try:
        pct = f" ({int(processed) / int(total) * 100:.0f}%)"
    except (ValueError, ZeroDivisionError, TypeError):
        pct = ""
    return f"{processed} / {total} frames{pct}"


# CLI entry point --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SLAM experiment runner")

    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--fe",  action="store_true")
    phase.add_argument("--lc",  action="store_true")
    phase.add_argument("--be",  action="store_true")
    phase.add_argument("--co",  action="store_true")
    phase.add_argument("--de",  action="store_true")
    phase.add_argument("--all", action="store_true")

    phase.add_argument("--variant", "-v", type=str, metavar="VARIANT_NAME")

    parser.add_argument("--test",        "-t", type=str, default="test0")
    parser.add_argument("--runs",        "-r", type=int, default=1)
    parser.add_argument("--vo",          action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--no-save-map", action="store_true")

    args = parser.parse_args()

    if args.all:
        exp_types = EXPERIMENT_TYPES
    elif args.fe:
        exp_types = ["feature_extraction"]
    elif args.lc:
        exp_types = ["loop_closure"]
    elif args.be:
        exp_types = ["backend"]
    elif args.co:
        exp_types = ["combined"]
    elif args.de:
        exp_types = ["depth_estimation"]
    elif args.variant:
        exp_types = []  # not used

    mode = "VO" if args.vo else "SLAM"

    # VO has no loop closure, so only feature_extraction + depth_estimation apply.
    if args.vo and not args.variant:
        dropped = [e for e in exp_types if e not in VO_EXP_TYPES]
        if dropped:
            print(f"  [VO] Skipping non-VO phases: {dropped}")
        exp_types = [e for e in exp_types if e in VO_EXP_TYPES]
        if not exp_types:
            sys.exit("VO mode runs only feature_extraction + depth_estimation — "
                     "nothing to run for the selected phase(s).")

    if args.test == "all":
        config_dir = PC_ROOT / "src_pc" / "configs" / "sequences"
        test_ids = sorted({c.name.split("__")[0]
                           for c in config_dir.glob("test*__*.yaml")})
    else:
        test_ids = [args.test]

    if not test_ids:
        sys.exit("No test sequences found — run prep_project.py first.")

    # the baseline is the shared reference for FE/LC/BE, scheduled once per
    # (mode, test_id) and topped up to --runs by reusing earlier finished runs.
    plan = []
    skipped_entries = []

    def _should_skip(vname: str, test_id: str) -> bool:
        """True if this test_id is in the variant's skip_tests list."""
        return test_id in VARIANTS.get(vname, {}).get("skip_tests", [])

    if args.variant:
        if args.variant not in VARIANTS:
            sys.exit(f"Unknown variant '{args.variant}'")
        selection_has_baseline = (args.variant == BASELINE_VARIANT)
        if not selection_has_baseline:
            for test_id in test_ids:
                for run_num in range(1, args.runs + 1):
                    if _should_skip(args.variant, test_id):
                        skipped_entries.append((args.variant, test_id, run_num))
                    else:
                        plan.append((args.variant, test_id, run_num))
    else:
        # any FE/LC/BE phase needs the baseline to compare against.
        selection_has_baseline = any(e in BASELINE_PHASES for e in exp_types)
        for exp_type in exp_types:
            for vname in list_variants(exp_type):
                for test_id in test_ids:
                    for run_num in range(1, args.runs + 1):
                        if _should_skip(vname, test_id):
                            skipped_entries.append((vname, test_id, run_num))
                        else:
                            plan.append((vname, test_id, run_num))

    # explicit baseline (--all / --variant baseline) runs the full --runs count;
    # a phase run only tops the shared baseline up to --runs.
    explicit_baseline = bool(args.all) or (args.variant == BASELINE_VARIANT)

    if selection_has_baseline:
        for test_id in test_ids:
            start = 1 if explicit_baseline else _completed_baseline_runs(mode, test_id) + 1
            for run_num in range(start, args.runs + 1):
                plan.append((BASELINE_VARIANT, test_id, run_num))

    total = len(plan)

    print(f"\n{'='*60}")
    print(f"  Batch experiment plan")
    print(f"  Phases        : {exp_types}")
    print(f"  Tests         : {test_ids}")
    print(f"  Runs/variant  : {args.runs}")
    print(f"  Total runs    : {total}")
    print(f"  Skipped       : {len(skipped_entries)}")
    print(f"  VO mode       : {'yes' if args.vo else 'no'}")
    print(f"  Frame budget  : {MAX_RUN_WALL_TIME_S}s")
    print(f"  Hard timeout  : {HARD_TIMEOUT_S}s")
    print(f"  Inter-run gap : {INTER_RUN_SLEEP}s")
    print(f"{'='*60}\n")

    for i, (vname, test_id, run_num) in enumerate(plan, 1):
        print(f"  [{i:3d}/{total}]  {vname:<35}  {_test_tag(test_id):<7}  run {run_num}")
    if skipped_entries:
        print(f"\n  Skipped (skip_tests flag):")
        for vname, test_id, run_num in skipped_entries:
            print(f"    [SKIP]  {vname:<35}  {_test_tag(test_id):<7}  run {run_num}")
    print()

    if args.dry_run:
        print("Dry run — nothing executed.\n")
        sys.exit(0)

    print(f"  Flag log → {FLAG_CSV.relative_to(PC_ROOT)}\n")

    results_ok  = []   # completed ok                     — (vname, test_id, run_num, elapsed_s)
    lost_       = []   # exit 0, abort_reason LOST        — (vname, test_id, run_num)
    oom_        = []   # exit -9 (OOM kill)               — (vname, test_id, run_num)
    timed_out_  = []   # hard/soft timeout                 — (vname, test_id, run_num)
    errors_     = []   # everything else                   — (vname, test_id, run_num, reason)
    _interrupted = False
    t_batch_start = time.time()

    try:
        for i, (vname, test_id, run_num) in enumerate(plan, 1):
            print(f"\n{'─'*60}")
            print(f"  [{i}/{total}]  {vname}  |  {_test_tag(test_id)}  |  run {run_num}")
            print(f"  RAM: {_ram_pct():.0f}% used  ({_ram_free_gb():.1f} GB free)")
            print(f"{'─'*60}")

            cmd = [sys.executable, str(RUNNER), "--variant", vname, "--test", test_id]
            # slam_runner runs headless by default now, so we only forward the flags.
            if args.no_save_map: cmd.append("")
            if args.vo:          cmd.append("--no-loop-closure")

            result_dir = _variant_result_dir(mode, vname, test_id)
            returncode, elapsed, was_timeout, killed_signal = run_one(
                cmd, result_dir=result_dir)

            n = kill_all_slam_processes()
            if n:
                print(f"  [cleanup] Killed {n} lingering process(es)")

            errors_path = RESULTS / mode / test_id / "errors.txt"

            if was_timeout:
                # hard timeout - slam_runner got killed, so a dashboard may or may not exist.
                dash = _read_latest_dashboard(mode, vname, test_id)
                status = "timeout"
                timed_out_.append((vname, test_id, run_num))
                print(f"\n  [!] TIMED OUT after {elapsed:.0f}s  ({killed_signal})"
                      f"  — {_frame_info(dash)}")
                errors_path.parent.mkdir(parents=True, exist_ok=True)
                with open(errors_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n{'='*80}\n"
                        f"TIMEOUT — {datetime.now().isoformat(timespec='seconds')}\n"
                        f"Variant  : {vname}\n"
                        f"Test     : {test_id}  run {run_num}\n"
                        f"Elapsed  : {elapsed:.0f}s  (hard limit {HARD_TIMEOUT_S}s)\n"
                        f"Frames   : {_frame_info(dash)}\n"
                        f"Killed   : {killed_signal}\n"
                        f"Command  : {' '.join(cmd)}\n"
                        f"{'='*80}\n"
                    )
            elif returncode in OOM_SIGNALS and killed_signal != "teardown":
                status = "oom"
                oom_.append((vname, test_id, run_num))
                print(f"\n  [!] OOM KILL ({_signame(returncode)} / exit {returncode}) "
                      f"after {elapsed:.0f}s")
                errors_path.parent.mkdir(parents=True, exist_ok=True)
                with open(errors_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n{'='*80}\n"
                        f"OOM KILL — {datetime.now().isoformat(timespec='seconds')}\n"
                        f"Variant  : {vname}\n"
                        f"Test     : {test_id}  run {run_num}\n"
                        f"Signal   : {_signame(returncode)} (exit {returncode})\n"
                        f"Elapsed  : {elapsed:.0f}s\n"
                        f"Command  : {' '.join(cmd)}\n"
                        f"{'='*80}\n"
                    )
            elif returncode not in (0, None) and killed_signal != "teardown":
                status = "error"
                sigtag = _signame(returncode)
                reason = f"exit {returncode}" + (f" ({sigtag})" if sigtag else "")
                errors_.append((vname, test_id, run_num, reason))
                print(f"\n  [!] ERROR ({reason}) after {elapsed:.0f}s")
                errors_path.parent.mkdir(parents=True, exist_ok=True)
                with open(errors_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n{'='*80}\n"
                        f"ERROR — {datetime.now().isoformat(timespec='seconds')}\n"
                        f"Variant  : {vname}\n"
                        f"Test     : {test_id}  run {run_num}\n"
                        f"Exit     : {reason}\n"
                        f"Command  : {' '.join(cmd)}\n"
                        f"{'='*80}\n"
                    )
            else:
                # exit 0 or a teardown-grace kill - let the dashboard say what happened.
                if killed_signal == "teardown":
                    print(f"  [i] Killed hung teardown after result written "
                          f"({elapsed:.0f}s) — classifying by dashboard.")
                dash = _read_latest_dashboard(mode, vname, test_id)
                abort_reason_dash = (dash.get("abort_reason", "") if dash else "").upper()
                if abort_reason_dash in ("", "COMPLETED", "MAX_FRAMES"):
                    status = "ok"
                    results_ok.append((vname, test_id, run_num, elapsed))
                    print(f"\n  Completed in {elapsed:.0f}s  "
                          f"(RAM: {_ram_pct():.0f}% / {_ram_free_gb():.1f} GB free)")
                elif abort_reason_dash == "LOST":
                    status = "lost"
                    lost_.append((vname, test_id, run_num))
                    print(f"\n  [!] LOST after {elapsed:.0f}s"
                          f"  — {_frame_info(dash)}")
                elif abort_reason_dash == "WALLTIME":
                    # hit slam_runner's own soft time limit - count it as a timeout
                    status = "timeout"
                    timed_out_.append((vname, test_id, run_num))
                    print(f"\n  [!] WALLTIME after {elapsed:.0f}s"
                          f"  — {_frame_info(dash)}")
                else:
                    # VIEWER_CLOSED, ERROR, or something we don't recognise
                    status = "error"
                    errors_.append((vname, test_id, run_num, abort_reason_dash))
                    print(f"\n  [!] STOPPED ({abort_reason_dash}) after {elapsed:.0f}s")

            _log_flag(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                variant=vname, test_id=test_id, run_num=run_num,
                status=status, returncode="" if returncode is None else returncode,
                elapsed_s=f"{elapsed:.1f}",
                wall_limit_s=MAX_RUN_WALL_TIME_S, hard_limit_s=HARD_TIMEOUT_S,
                killed_signal=killed_signal, command=" ".join(cmd),
            )

            print(f"  Cooldown: {INTER_RUN_SLEEP}s ...")
            time.sleep(INTER_RUN_SLEEP)

    except KeyboardInterrupt:
        _interrupted = True
        print(f"\n{'='*60}")
        print("  [!] Batch stopped by Ctrl+C — cleaning up ...")
        print(f"{'='*60}")
        n = kill_all_slam_processes()
        if n:
            print(f"  [cleanup] Killed {n} lingering process(es)")

    total_time = time.time() - t_batch_start

    print(f"\n{'='*60}")
    print(f"  Batch {'INTERRUPTED' if _interrupted else 'complete'} — {total_time/60:.1f} min")

    print(f"  Complete  : {len(results_ok)}")
    for vname, test_id, run_num, elapsed_s in results_ok:
        print(f"    {vname:<35}  {_test_tag(test_id):<7}  run {run_num}  ({elapsed_s:.0f}s)")

    print(f"  Lost      : {len(lost_)}")
    for vname, test_id, run_num in lost_:
        dash = _read_latest_dashboard(mode, vname, test_id)
        print(f"    {vname:<35}  {_test_tag(test_id):<7}  run {run_num}  —  {_frame_info(dash)}")

    print(f"  OOM kill  : {len(oom_)}")
    for vname, test_id, run_num in oom_:
        print(f"    {vname:<35}  {_test_tag(test_id):<7}  run {run_num}")

    print(f"  Timeout   : {len(timed_out_)}")
    for vname, test_id, run_num in timed_out_:
        dash   = _read_latest_dashboard(mode, vname, test_id)
        detail = _frame_info(dash)
        print(f"    {vname:<35}  {_test_tag(test_id):<7}  run {run_num}  —  {detail}")

    print(f"  Errors    : {len(errors_)}")
    for vname, test_id, run_num, reason in errors_:
        print(f"    {vname:<35}  {_test_tag(test_id):<7}  run {run_num}  ({reason})")

    print("\n  ( * = hardest sequence )")

    if lost_ or oom_ or timed_out_ or errors_:
        print("\n  Details → results/*/errors.txt   and   results/run_flags.csv")
    print(f"{'='*60}\n")
    print("  python3 src_pc/utils/aggregate_results.py\n")


if __name__ == "__main__":
    main()