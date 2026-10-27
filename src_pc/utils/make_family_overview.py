#!/usr/bin/env python3
"""
src_pc/utils/make_family_overview.py

Collapses every feature-extraction and loop-closure variant into its feature
family (classical / CNN / transformer) and compares them: mean accuracy,
completion/speed, and how many runs each figure rests on.

SCOPE: only exp_type in {baseline, feature_extraction, loop_closure}. backend
and depth_estimation don't have a classical/CNN/transformer split worth
pooling, and combined would double-count a family (it pairs one FE with one
LC). baseline (ORB2) stays in as the classical reference row.

All three figures are pinned to one hardware condition (CONDITION_CODE) and
to FULL_COVERAGE_SEQS = ["test0", "test1"] -- the only sequences where every
family got its full line-up (test2/test3 are pilot-era skips that would
unfairly shrink CNN/Trf numbers).

Reads results_aggregated/csv/all_conditions.csv (same source as
make_assets_offline.py): cell-level rows only, one mode, probes excluded.

Outputs three PNGs to results_aggregated/figures/:
  family_avg_ate.png    mean ATE (over completed runs) per family
  family_avg_fps.png    median fps per family
  family_run_volume.png successful runs per family vs. max attempted

Also prints a plain-text summary of the same numbers.

USAGE
-----
    python -m src_pc.utils.make_family_overview
    python -m src_pc.utils.make_family_overview --in results_aggregated --mode SLAM
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAM_COLOR = {"classical": "#1b9e77", "CNN": "#1f77b4", "transformer": "#e6820e"}
FAM_LABEL = {"classical": "Cls (classical)", "CNN": "CNN", "transformer": "Trf (transformer)"}
FAM_ORDER = ["classical", "CNN", "transformer"]

PHASES = ["baseline", "feature_extraction", "loop_closure"]
FULL_COVERAGE_SEQS = ["test0", "test1"]
CONDITION_CODE = "R9_12t_28"
CONDITION_LABEL = "R9 9950X · 12t · 28 GB"

C_ATE = "ate_rmse_succ_mean"
C_FPS = "avg_fps_all_median"


def load_cells(csv_path: Path, mode: str) -> pd.DataFrame:
    """Cell-level rows for one mode, one condition, two sequences; probes and
    the AVG roll-up excluded. Mirrors make_assets_offline.load_all()'s
    filtering (see module docstring for why condition/seqs are pinned)."""
    if not csv_path.is_file():
        sys.exit(f"cannot find {csv_path} -- run aggregate_all first")
    d = pd.read_csv(csv_path, low_memory=False)
    d = d[(d["level"] == "cell") & (d["mode"] == mode)
          & (d["test_id"] != "AVG")].copy()
    if "is_probe" in d.columns:
        probe = d["is_probe"].where(d["is_probe"].notna(), False).astype(bool)
        d = d[~probe]
    d = d[d["exp_type"].isin(PHASES)]
    d = d[d["family"].isin(FAM_ORDER)]
    d = d[d["condition"] == CONDITION_CODE]
    d = d[d["test_id"].isin(FULL_COVERAGE_SEQS)]
    if d.empty:
        sys.exit(f"no {PHASES} cell rows for mode={mode}, condition="
                 f"{CONDITION_CODE}, test_id in {FULL_COVERAGE_SEQS}")
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """One row per family: completion, mean ATE over completed runs, median fps."""
    rows = []
    for fam in FAM_ORDER:
        g = d[d["family"] == fam]
        if g.empty:
            continue
        n_total = int(g["n_total"].sum())
        n_success = int(g["n_success"].sum())
        ate = g.loc[g["n_success"] > 0, C_ATE].dropna()
        rows.append({
            "family": fam,
            "n_total": n_total,
            "n_success": n_success,
            "completion_pct": 100 * n_success / n_total if n_total else np.nan,
            "ate_mean": ate.mean() if len(ate) else np.nan,
            "fps_median": g[C_FPS].median(),
            "n_variants": g["variant"].nunique(),
        })
    return pd.DataFrame(rows)


def fig_avg_ate(s: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(s))
    colors = [FAM_COLOR[f] for f in s["family"]]
    bars = ax.bar(x, s["ate_mean"], color=colors, width=.6, edgecolor="white")
    for xi, r in zip(x, s.itertuples()):
        if pd.notna(r.ate_mean):
            ax.text(xi, r.ate_mean, f"{r.ate_mean:.2f} m", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in s["family"]], fontsize=10)
    ax.set_ylabel("mean ATE over completed runs (m)")
    ax.set_title("Average accuracy by feature family\n"
                 f"{CONDITION_LABEL}, {' + '.join(FULL_COVERAGE_SEQS)}",
                 fontsize=11)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "family_avg_ate.png", dpi=160, bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    print(f"   figures/family_avg_ate.png")


def fig_avg_fps(s: pd.DataFrame, out: Path) -> None:
    """Speed only -- completion now has its own plot, family_run_volume.png."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(s))
    colors = [FAM_COLOR[f] for f in s["family"]]
    labels = [FAM_LABEL[f] for f in s["family"]]

    ax.bar(x, s["fps_median"], color=colors, width=.6, edgecolor="white")
    for xi, r in zip(x, s.itertuples()):
        if pd.notna(r.fps_median):
            ax.text(xi, r.fps_median, f"{r.fps_median:.1f}", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("median fps")
    ax.set_title("Average speed by feature family\n"
                 f"{CONDITION_LABEL}, {' + '.join(FULL_COVERAGE_SEQS)}",
                 fontsize=11)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "family_avg_fps.png", dpi=160, bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    print(f"   figures/family_avg_fps.png")


def run_volume(d: pd.DataFrame) -> pd.DataFrame:
    """Successful and attempted runs per family (d already pinned by load_cells)."""
    rows = []
    for fam in FAM_ORDER:
        g = d[d["family"] == fam]
        if g.empty:
            continue
        rows.append({"family": fam,
                     "n_total": int(g["n_total"].sum()),
                     "n_success": int(g["n_success"].sum())})
    return pd.DataFrame(rows)


def fig_run_volume(v: pd.DataFrame, out: Path) -> None:
    """Successful runs per family, with a dashed line at the max attempted."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(v))
    colors = [FAM_COLOR[f] for f in v["family"]]
    max_attempts = int(v["n_total"].max())

    ax.bar(x, v["n_success"], color=colors, width=.6, edgecolor="white")
    for xi, r in zip(x, v.itertuples()):
        ax.text(xi, r.n_success, f"{r.n_success}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(max_attempts, color="#444", ls="--", lw=1.3, zorder=0)
    ax.annotate(f"max attempted: {max_attempts}",
                (len(v) - 1, max_attempts), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=9,
                color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in v["family"]], fontsize=10)
    ax.set_ylabel("successful runs")
    ax.set_ylim(0, max_attempts * 1.12)
    ax.set_title("How many runs the averages rest on\n"
                 f"{CONDITION_LABEL}, {' + '.join(FULL_COVERAGE_SEQS)} -- the "
                 "sequences where every family got its full line-up",
                 fontsize=11)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "family_run_volume.png", dpi=160, bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    print(f"   figures/family_run_volume.png")


def print_summary(s: pd.DataFrame, v: pd.DataFrame) -> None:
    print(f"\nfamily overview -- {CONDITION_LABEL}, "
          f"{' + '.join(FULL_COVERAGE_SEQS)} only")
    print("(feature extraction + loop closure + baseline, "
          "combined/backend/depth excluded)")
    print("-" * 70)
    for r in s.itertuples():
        print(f"{FAM_LABEL[r.family]:<22} "
              f"{r.n_variants} variants   "
              f"completed {r.n_success}/{r.n_total} runs "
              f"({r.completion_pct:.1f}%)   "
              f"mean ATE {r.ate_mean:.2f} m   "
              f"median fps {r.fps_median:.1f}")
    print("-" * 70)
    max_attempts = int(v["n_total"].max())
    print(f"run volume (max attempted by any family: {max_attempts}):")
    for r in v.itertuples():
        print(f"{FAM_LABEL[r.family]:<22} {r.n_success} successful "
              f"(of {r.n_total} attempted)")
    print("-" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="indir", type=Path,
                    default=Path("results_aggregated"),
                    help="directory holding csv/all_conditions.csv "
                         "(default: results_aggregated)")
    ap.add_argument("--mode", default="SLAM", choices=["SLAM", "VO"])
    args = ap.parse_args()

    csv_path = args.indir / "csv" / "all_conditions.csv"
    fig_dir = args.indir / "figures"

    d = load_cells(csv_path, args.mode)
    s = summarize(d)
    v = run_volume(d)

    fig_avg_ate(s, fig_dir)
    fig_avg_fps(s, fig_dir)
    fig_run_volume(v, fig_dir)
    print_summary(s, v)


if __name__ == "__main__":
    main()
