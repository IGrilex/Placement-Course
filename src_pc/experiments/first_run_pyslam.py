"""
src_pc/experiments/first_run_pyslam.py
------------------------
To test if the project is running.

it dose ORB2 baseline on test0 with the GUI on. 
Output lands in results/test_run/ - always overwritten, and skipped by aggregation.

Usage:
    python3 src_pc/experiments/first_run_pyslam.py
"""

import sys
import shutil
from pathlib import Path

PC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PC_ROOT))
sys.path.insert(0, str(PC_ROOT.parent / "pyslam"))

from src_pc.experiments.slam_runner import run_sequence
from src_pc.utils.paths import ROOT

OUTPUT_DIR = ROOT / "results" / "test" / "first_run"

if OUTPUT_DIR.exists():
    if any(OUTPUT_DIR.iterdir()):
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True)
else:
    OUTPUT_DIR.mkdir(parents=True)

print(f"\n{'='*60}")
print(f"  first_run_pyslam.py  —  sanity check")
print(f"  variant  : baseline")
print(f"  sequence : test0.")
print(f"  output   : results/test_run/")
print(f"{'='*60}\n")

run_sequence(
    variant_name = "baseline",
    test_id      = "test0",
    output_dir   = OUTPUT_DIR,
    headless     = False,
    save_map     = True,
)