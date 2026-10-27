"""
src_pc/experiments/test_models.py
──────────────
Sanity-checks every variant in variants.py by running slam_runner on each one for
just TEST_FRAMES frames, reusing the same subprocess handling as batch_runner.

Two things get checked at once:
  1. does the model actually load and run without crashing?
  2. does batch_runner clean up properly after each run? (RAM should drop back to
     idle between variants - if it doesn't, something is leaking)

Usage:
    python3 src_pc/experiments/test_models.py

Output:
    results/test_run/model_test.csv
    columns: variant, category, status, exit_code, elapsed_s, frames, notes
    status:  PASS | FAIL | OOM | TIMEOUT

Config (edit at top of this file):
    TEST_FRAMES     = N     frames to run (set in project_settings.py)
    TEST_ID         = "test0"
    TEST_TIMEOUT_S  = 180   3 minutes per variant (set in project_settings.py)
    INTER_TEST_SLEEP= 45    seconds between variants (set in project_settings.py)
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Project roots ----------------------------------------------------------------------------
PC_ROOT     = Path(__file__).resolve().parents[2]
PYSLAM_ROOT = PC_ROOT.parent / "pyslam"
sys.path.insert(0, str(PC_ROOT))
sys.path.insert(0, str(PYSLAM_ROOT))
os.chdir(str(PC_ROOT))

# reuse batch_runner's subprocess handling instead of duplicating it -----------------------
from src_pc.experiments.batch_runner import (
    run_one,
    kill_all_slam_processes,
    RUNNER,
)
from src_pc.experiments.variants import VARIANTS
from src_pc.configs.project_settings import TEST_FRAMES, TEST_TIMEOUT_S, INTER_TEST_SLEEP
from src_pc.utils.paths import RESULTS

# Test config ------------------------------------------------------------------------------
# TEST_FRAMES lives in project_settings.py (shared with slam_runner's --test-mode)
TEST_ID          = "test0"

OUT_DIR = RESULTS / "test"
OUT_CSV = OUT_DIR / "model_test.csv"

CSV_FIELDS = [
    "variant", "category", "status",
    "exit_code", "elapsed_s", "frames_requested", "notes",
]

# Category labels --------------------------------------------------------------------------
_EXP_LABELS = {
    "feature_extraction": "FE",
    "loop_closure":       "LC",
    "backend":            "BE",
    "combined":           "CO",
    "depth_estimation":   "DE",
}


def _status_from_result(returncode, timed_out: bool) -> tuple[str, str]:
    """Return (status_label, notes_string)."""
    if timed_out:
        return "TIMEOUT", f"killed after {TEST_TIMEOUT_S}s — model too slow or hung"
    if returncode == 0:
        return "PASS", ""
    if returncode == -9:
        return "OOM", "SIGKILL from kernel OOM killer — map too large for available RAM"
    if returncode == -11:
        return "FAIL", "SIGSEGV — C extension segfault"
    return "FAIL", f"exit code {returncode}"


def _write_csv(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _print_summary(results: list[dict]) -> None:
    by_status: dict[str, list[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["variant"])

    total   = len(results)
    passed  = len(by_status.get("PASS", []))
    failed  = len(by_status.get("FAIL", []))
    oom     = len(by_status.get("OOM", []))
    timeout = len(by_status.get("TIMEOUT", []))

    print(f"\n{'='*60}")
    print(f"  PASS   : {passed}")
    print(f"  FAIL   : {failed}")
    print(f"  OOM    : {oom}   (map too large — expected for dense models on long seqs)")
    print(f"  TIMEOUT: {timeout}")
    print(f"  Total  : {total}")

    for status in ("FAIL", "OOM", "TIMEOUT"):
        variants = by_status.get(status, [])
        if variants:
            print(f"\n  {status}:")
            for v in variants:
                r = next(x for x in results if x["variant"] == v)
                print(f"    {v:<35}  {r.get('notes', '')[:60]}")

    print(f"\n  CSV → {OUT_CSV.relative_to(PC_ROOT)}")
    print(f"{'='*60}\n")


def main() -> None:
    # sort by category, then name, so the output is grouped in a sensible order
    ordered = sorted(
        VARIANTS.items(),
        key=lambda kv: (kv[1]["experiment_type"], kv[0])
    )
    total = len(ordered)

    print(f"\n{'='*60}")
    print(f"  test_models.py  —  {total} variants × {TEST_FRAMES} frames")
    print(f"  Test sequence   : {TEST_ID}")
    print(f"  Timeout/variant : {TEST_TIMEOUT_S}s")
    print(f"  Inter-test gap  : {INTER_TEST_SLEEP}s")
    print(f"  Output          : {OUT_CSV.relative_to(PC_ROOT)}")
    print(f"{'='*60}\n")

    results = []
    _interrupted = False

    try:
        for i, (vname, vcfg) in enumerate(ordered, 1):
            category = _EXP_LABELS.get(vcfg["experiment_type"], vcfg["experiment_type"])
            depth    = f"  [{vcfg['depth_estimator']}]" if vcfg.get("depth_estimator") else ""
            print(f"\n[{i}/{total}]  {category}  {vname}{depth}", flush=True)
            print("─" * 60, flush=True)

            cmd = [
                sys.executable, str(RUNNER),
                "--variant",   vname,
                "--test",      TEST_ID,
                "--test-mode",
            ]

            returncode, elapsed, timed_out, killed_signal = run_one(
                cmd, timeout_s=TEST_TIMEOUT_S
            )

            # clean up right away - this cleanup is half the point of the test
            n = kill_all_slam_processes()
            if n:
                print(f"  [cleanup] {n} process(es) killed", flush=True)

            status, notes = _status_from_result(returncode, timed_out)

            sym = {"PASS": "✓", "FAIL": "✗", "OOM": "💥", "TIMEOUT": "⏱"}.get(status, "?")
            print(f"  {sym}  {status:<8}  {elapsed:.1f}s", flush=True)
            if notes:
                print(f"     → {notes}", flush=True)

            results.append({
                "variant":           vname,
                "category":          category,
                "status":            status,
                "exit_code":         "" if returncode is None else returncode,
                "elapsed_s":         round(elapsed, 1),
                "frames_requested":  TEST_FRAMES,
                "notes":             notes,
            })

            # write as we go, so a Ctrl+C mid-run still leaves us the results so far
            _write_csv(results)

            if i < total:
                print(f"  Cooldown: {INTER_TEST_SLEEP}s ...", flush=True)
                time.sleep(INTER_TEST_SLEEP)

    except KeyboardInterrupt:
        _interrupted = True
        print(f"\n{'='*60}")
        print("  [!] Interrupted — cleaning up ...")
        n = kill_all_slam_processes()
        if n:
            print(f"  [cleanup] {n} process(es) killed")
        print(f"{'='*60}")

    _write_csv(results)

    if _interrupted:
        print(f"\n  Partial results saved → {OUT_CSV.relative_to(PC_ROOT)}")
    else:
        _print_summary(results)


if __name__ == "__main__":
    main()