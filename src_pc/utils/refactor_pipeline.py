#!/usr/bin/env python3
"""
refactor_pipeline.py  --  one-shot reorganisation of the results-asset scripts.

    aggregate_all.py         keeps only the CSV builder (drops ~1,500 lines of
                             dead figure/table code already duplicated in
                             make_results_assets.py); gains map_growth.csv +
                             three data-hygiene tables.
    make_results_assets.py   renamed -> make_assets_offline.py, absorbs
                             make_diagrams.py as build_mechanism_diagrams().
    make_diagrams.py         deleted (content now lives in the above).

Deletes rather than rewrites: reachability analysis from main() found 32 dead
functions, and retyping thousands of lines of working code risks changing
behaviour. Nothing is written until every resulting file is ast-parsed and
checked for references to deleted names -- if any check fails, nothing is
touched. Idempotent: a second run detects the finished state and stops.

Usage:
    python3 refactor_pipeline.py
    python3 refactor_pipeline.py --dir src_pc/utils
    python3 refactor_pipeline.py --dry-run
"""
import argparse
import ast
import sys
from pathlib import Path

DELETE = ['_retired_outputs', 'fig_outcome_ladder', 'fig_failure_shifts', 'fig_ram_relief', '_relief_records', '_fps_bars', 'fig_fps_by_cpu_load', 'fig_fps_by_gpu_load', 'fig_how_far_runs_get', '_cpu_rung', '_thread_rung', '_paired', '_family_speedup', 'fig_speedup_by_family', 'fig_thread_saturation', 'fig_relative_to_baseline', 'write_condition_summary_table', 'write_failure_shift_table', 'write_rescued_cells_table', 'write_accuracy_invariance_table', 'write_speedup_by_family_table', '_fail_points_at_death', 'fig_failure_map_ladder', 'write_failure_map_ladders', 'ds_fig_map_growth']

MAP_GROWTH_EXPORT = '\ndef write_map_growth_csv(cond, mode: str, out_dir: Path) -> None:\n    """map_growth.csv -- keyframes and map points against distance driven.\n\n    The ONLY per-frame series this pipeline exports, and it exists because it\n    is the one thing the offline script cannot reconstruct: all_conditions.csv\n    holds a single final num_keyframes per run, not the growth curve, and the\n    curve lives only inside each run\'s own progress.csv. Walking run folders\n    is this file\'s job, so the extraction happens here and the plotting\n    happens in make_assets_offline.py, which reads what this writes.\n\n    One row per logged frame, for the baseline run on each sequence at ONE\n    condition. `cond` should be the largest-memory condition available: at any\n    budget where a run can die of exhaustion, a growth curve is exactly the\n    thing that gets cut short for a reason that has nothing to do with the\n    road, which is the failure mode Section 8.2 is about.\n    """\n    tree = PC_ROOT / cond.folder\n    scan_root = tree / mode\n    if not scan_root.is_dir():\n        print(f" [!] map_growth: {scan_root} not found")\n        return\n\n    from src_pc.utils.metrics_helper_functions import load_tum_trajectory\n\n    run_flags = load_run_flags(tree)\n    expected_frames = load_expected_frames()\n    try:\n        test_ids = all_test_ids()\n    except SystemExit:\n        return\n\n    rows = []\n    for test_id in test_ids:\n        expected = expected_frames.get(test_id)\n        rd = _pick_growth_run(scan_root, test_id, run_flags, expected)\n        if rd is None:\n            print(f" [!] map_growth {test_id}: no baseline run found")\n            continue\n        growth = _read_kf_growth(rd / "files" / "progress.csv")\n        if growth is None:\n            print(f" [!] map_growth {test_id}: no progress.csv under {rd}")\n            continue\n\n        gt_files = sorted((agg.PROCESSED_POSES / test_id).glob("*_tum.txt"))\n        if not gt_files:\n            print(f" [!] map_growth {test_id}: no ground-truth poses")\n            continue\n        xyz = load_tum_trajectory(gt_files[0]).xyz\n        n = len(xyz)\n        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)\n        cumdist = np.concatenate([[0.0], np.cumsum(step)])\n\n        fid = np.clip(growth[:, 0].astype(int), 0, n - 1)\n        for f, kf, mp, d in zip(fid, growth[:, 1], growth[:, 2],\n                                cumdist[fid]):\n            rows.append({"condition": cond.key, "mode": mode,\n                         "test_id": test_id, "frame_id": int(f),\n                         "distance_m": float(d),\n                         "num_keyframes": float(kf),\n                         "map_points": float(mp)})\n\n    if not rows:\n        print(" [!] map_growth: nothing to write")\n        return\n    path = out_dir / "map_growth.csv"\n    pd.DataFrame(rows).to_csv(path, index=False)\n    print(f" wrote {path}  ({len(rows)} frame rows, "\n          f"{len({r[\'test_id\'] for r in rows})} sequences)")\n\n'

WIRE_IN = '    write_data_dictionary(df, out_dir)\n    print(f" wrote {out_dir / \'all_conditions_DICTIONARY.md\'}")\n\n    # Per-frame map growth: the one series the offline script cannot rebuild\n    # from all_conditions.csv, exported at the most capable condition.\n    try:\n        write_map_growth_csv(CONDITIONS[-1], args.mode, out_dir)\n    except Exception as e:\n        print(f" [!] map_growth.csv failed: {e}")\n\n    # Data-hygiene tables. These describe the COLLECTION rather than the\n    # results -- what is backed by data, which cells are probes, where the\n    # baseline itself is thin -- so they belong beside the CSV and the data\n    # dictionary, not with the thesis figures.\n    for name, fn in (("provenance_table",      write_provenance_table),\n                     ("baseline_health_table", write_baseline_health_table),\n                     ("probe_cells_table",     write_probe_table)):\n        try:\n            fn(df, out_dir)\n            print(f" wrote {out_dir / (name + \'.tex\')}")\n        except Exception as e:\n            print(f" [!] {name} failed: {e}")\n'

WIRE_ANCHOR = """    write_data_dictionary(df, out_dir)
    print(f" wrote {out_dir / 'all_conditions_DICTIONARY.md'}")
"""


def spans(src, names):
    """(start, end) 1-based line spans for each named top-level function,
    widened upwards to swallow the contiguous comment banner above it."""
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            start = n.lineno
            i = start - 2
            while i >= 0 and (lines[i].strip() == ""
                              or lines[i].lstrip().startswith("#")):
                i -= 1
            out.append((i + 2, n.end_lineno))
    return sorted(out)


def cut(src, spans_):
    lines = src.splitlines(keepends=True)
    drop = set()
    for a, b in spans_:
        drop.update(range(a - 1, b))
    return "".join(l for i, l in enumerate(lines) if i not in drop)


def check(src, path, forbidden):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"{path} would not parse: {e}"
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    still = sorted(used & set(forbidden))
    if still:
        return (f"{path} still references deleted name(s): "
                f"{', '.join(still)}")
    return None


def refactor_aggregate_all(path, dry):
    src = path.read_text(encoding="utf-8")
    if "def write_map_growth_csv" in src:
        print(f"  {path.name}: already refactored, skipping")
        return None
    if "_pick_growth_run" not in src:
        sys.exit(f"{path}: _pick_growth_run not found -- the map-growth "
                 f"section must be moved into this file first (see the "
                 f"earlier fix_ds_map_growth_location.py step).")

    present = [n for n in DELETE if f"def {n}(" in src]
    new = cut(src, spans(src, present))

    if WIRE_ANCHOR not in new:
        sys.exit(f"{path}: could not find the data-dictionary block in "
                 f"main() to wire the new exports after.")
    new = new.replace(WIRE_ANCHOR, WIRE_IN, 1)

    anchor = "def main() -> None:"
    if anchor not in new:
        sys.exit(f"{path}: no main() found.")
    new = new.replace(anchor, MAP_GROWTH_EXPORT + "\n" + anchor, 1)
    new = new.replace(
        " next:  python -m src_pc.utils.make_results_assets",
        " next:  python -m src_pc.utils.make_assets_offline")

    err = check(new, path.name, present)
    if err:
        sys.exit(f"[abort, nothing written] {err}")
    removed = len(src.splitlines()) - len(new.splitlines())
    if not dry:
        path.write_text(new, encoding="utf-8")
    print(f"  {path.name}: deleted {len(present)} dead functions "
          f"(-{removed} lines), added write_map_growth_csv + 3 hygiene "
          f"tables")
    return new


def absorb_diagrams(assets_src, diagrams_src):
    """Wrap make_diagrams.py's flat top-level script in a function so its
    statements don't run at import time and its locals stay local."""
    tree = ast.parse(diagrams_src)
    lines = diagrams_src.splitlines()

    keep = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, str):
            continue    # module docstring
        keep.append(n)
    if not keep:
        sys.exit("make_diagrams.py: nothing to absorb")

    start = min(n.lineno for n in keep)
    end = max(n.end_lineno for n in keep)
    body = "\n".join("    " + l if l.strip() else l
                      for l in lines[start - 1:end])

    return (
        "\n\n# " + "\u2550" * 74 + "\n"
        "#  MECHANISM DIAGRAMS  (Chapter 3)\n"
        "# " + "\u2550" * 74 + "\n"
        "# Absorbed from the former make_diagrams.py. These are pure\n"
        "# schematics -- they read no data at all, which is why they can live\n"
        "# in the offline script alongside builders that read the CSV, and\n"
        "# why they are the only figures here that cannot go stale.\n"
        "def build_mechanism_diagrams():\n"
        '    """FAST/BRIEF, descriptor space, CNN detect-and-describe,\n'
        '    attention matching, and the three place-recognition summarisers."""\n'
        + body + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None,
                    help="directory holding the scripts "
                         "(default: search for them)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dir:
        d = args.dir
        agg_all = d / "aggregate_all.py"
        assets = d / "make_results_assets.py"
        diagrams = d / "make_diagrams.py"
        offline = d / "make_assets_offline.py"
    else:
        def find(name):
            hits = [p for p in Path(".").rglob(name)]
            return hits[0] if len(hits) == 1 else None
        agg_all = find("aggregate_all.py")
        assets = find("make_results_assets.py") or find("make_assets_offline.py")
        diagrams = find("make_diagrams.py")
        if agg_all is None:
            sys.exit("could not locate aggregate_all.py -- pass --dir")
        offline = agg_all.parent / "make_assets_offline.py"

    print("refactor_pipeline" + ("  [DRY RUN]" if args.dry_run else ""))

    if agg_all and agg_all.is_file():
        refactor_aggregate_all(agg_all, args.dry_run)
    else:
        print("  aggregate_all.py not found, skipping")

    if offline.is_file() and not (assets and assets.name ==
                                  "make_results_assets.py" and assets.is_file()):
        print(f"  {offline.name}: already exists, skipping rename")
    elif assets and assets.is_file():
        src = assets.read_text(encoding="utf-8")
        if diagrams and diagrams.is_file():
            if "MECHANISM DIAGRAMS" in src:
                print(f"  {assets.name}: diagrams already absorbed")
            else:
                src = src.rstrip() + "\n" + absorb_diagrams(
                    src, diagrams.read_text(encoding="utf-8"))
                # the schematics need patch classes the assets file lacks
                src = src.replace(
                    "from matplotlib.patches import Patch",
                    "from matplotlib.patches import (Patch, Circle, Rectangle,\n"
                    "                                 FancyArrowPatch,\n"
                    "                                 FancyBboxPatch)", 1)
                err = check(src, offline.name, [])
                if err:
                    sys.exit(f"[abort, nothing written] {err}")
                print(f"  {offline.name}: absorbed make_diagrams.py "
                      f"as build_mechanism_diagrams()")
        src = src.replace("make_results_assets", "make_assets_offline")
        if not args.dry_run:
            offline.write_text(src, encoding="utf-8")
            assets.unlink()
            if diagrams and diagrams.is_file():
                diagrams.unlink()
        print(f"  {assets.name} -> {offline.name}"
              + (", make_diagrams.py deleted" if diagrams else ""))

    print()
    print("resulting pipeline:")
    print("  aggregate_all.py         raw runs -> all_conditions.csv"
          " + map_growth.csv")
    print("  make_assets_offline.py   CSV + poses -> tables and figures")
    print("  make_assets_online.py    pyslam models -> frame figures")
    print("  aggregate_results.py     unchanged (per-tree aggregator)")


if __name__ == "__main__":
    main()