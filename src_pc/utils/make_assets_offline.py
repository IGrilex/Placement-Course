#!/usr/bin/env python3
"""
src_pc/utils/make_assets_offline.py

Every figure and table in the results chapter, built from
results_comparison/all_conditions.csv and nothing else. No re-aggregation, no
walking run directories, so it runs in seconds and can be re-run any time a
number changes.

    aggregate_all.py   turns raw run folders into all_conditions.csv
    THIS FILE          turns all_conditions.csv into the chapter's assets

Three sections, three questions:
  software/   which ALGORITHM is more accurate/robust. Reference condition
              only, so starved (OOM-prone) conditions don't rank methods by
              memory footprint instead of capability.
  hardware/   what the MACHINE did, across all four conditions.
  general/    findings needing both modes (VO/SLAM) at both memory budgets.

REFERENCE = R9_12t_28 (cleanest failure profile of the four conditions).
CHECK = R9_32t_28 confirms no software conclusion depends on thread count.

Accuracy is reported as two separate columns (mean ATE over COMPLETED runs,
plus completion %), never blended, and never as a ratio to baseline: a metre
is a metre regardless of platform, unlike throughput/load which are partly
properties of this machine. rel_* accuracy columns exist in the CSV for
anyone who still wants the ratio.

Cost is always given both ways: absolute, with the ratio to the ORB2
baseline in brackets, e.g. "5.3 (0.80x)".

Marker: † means the baseline itself did not complete every run on that
sequence, so anything measured against it there rests on a thin reference.
Dashes mean a sequence was never attempted (pilot-era skip); such a row's
completion is computed only over what it ran.

Usage:
  python -m src_pc.utils.make_assets_offline
  python -m src_pc.utils.make_assets_offline --in results_comparison
  python -m src_pc.utils.make_assets_offline --only software
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:                                          # pragma: no cover
    sys.exit("make_assets_offline needs pandas:  pip install pandas")

try:
    from scipy import stats as _st
except ImportError:                                          # pragma: no cover
    _st = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import (Patch, Circle, Rectangle,
                                 FancyArrowPatch,
                                 FancyBboxPatch)                         # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402


REFERENCE = "R9_12t_28"
CHECK     = "R9_32t_28"
BASELINE  = "baseline"

LADDER = ["i7_12t_14", "R9_12t_14", "R9_12t_28", "R9_32t_28"]
COND_LABEL = {
    "i7_12t_14": "i7-8700K, 12t, 14 GB",
    "R9_12t_14": "R9 9950X, 12t, 14 GB",
    "R9_12t_28": "R9 9950X, 12t, 28 GB",
    "R9_32t_28": "R9 9950X, 32t, 28 GB",
}
COND_SHORT = {"i7_12t_14": "i7/14", "R9_12t_14": "R9/14",
              "R9_12t_28": "R9/28", "R9_32t_28": "R9/28/32t"}
RUNGS = [("i7_12t_14", "R9_12t_14", "platform"),
         ("R9_12t_14", "R9_12t_28", "capacity"),
         ("R9_12t_28", "R9_32t_28", "core count")]

SEQS = ["test0", "test1", "test2", "test3"]

DISPLAY = {
    "baseline": "ORB2 (baseline)", "orb": "ORB", "akaze": "AKAZE",
    "brisk": "BRISK", "sift": "SIFT", "root_sift": "RootSIFT", "kaze": "KAZE",
    "superpoint": "SuperPoint", "aliked": "ALIKED", "disk": "DISK",
    "keynet": "Key.Net", "r2d2": "R2D2", "d2net": "D2-Net", "xfeat": "XFeat",
    "xfeat_xfeat": "XFeat\u00d7XFeat", "liftfeat": "LiftFeat",
    "lightglue": "LightGlue", "xfeat_lg": "XFeat+LightGlue", "rdd": "RDD",
    "xfeat_lightglue": "XFeat + LightGlue",
    "dedode": "DeDoDe",
    "dbow2": "DBoW2", "obindex2": "OBIndex2", "ibow": "iBoW", "vlad": "VLAD",
    "sad": "SAD", "alexnet": "AlexNet", "netvlad": "NetVLAD",
    "cosplace": "CosPlace", "eigenplaces": "EigenPlaces", "megaloc": "MegaLoc",
    "cricavpr": "CricaVPR", "salad": "SALAD", "supervlad": "SuperVLAD",
    "gtsam": "GTSAM",
    "akaze_netvlad": "AKAZE + NetVLAD", "akaze_salad": "AKAZE + SALAD",
    "akaze_megaloc": "AKAZE + MegaLoc", "akaze_obindex2": "AKAZE + OBIndex2",
    "lightglue_netvlad": "LightGlue + NetVLAD",
    "lightglue_salad": "LightGlue + SALAD",
    "lightglue_megaloc": "LightGlue + MegaLoc",
    "lightglue_obindex2": "LightGlue + OBIndex2",
    "orb2_dav2": "ORB2 + DA V2", "orb2_dav3": "ORB2 + DA V3",
    "sp_dav2": "SuperPoint + DA V2", "sp_dav3": "SuperPoint + DA V3",
}

FAM_COLOR = {"classical": "#1b9e77", "CNN": "#1f77b4", "transformer": "#e6820e"}
FAM_SHORT = {"classical": "Cls", "CNN": "CNN", "transformer": "Trf"}
FAM_LONG  = {"classical": "Cls (classical)", "CNN": "CNN",
             "transformer": "Trf (transformer)"}
FAM_ORDER = ["classical", "CNN", "transformer"]
FAM_RANK  = {"classical": 1, "CNN": 2, "transformer": 3}


def order_rows(g, within: str = "compl", ascending: bool = False):
    if g.empty:
        return g
    g = g.copy()
    base = g["is_base"] if "is_base" in g else False
    ref = g["is_ref"] if "is_ref" in g else False
    g["_rank0"] = np.where(base, 0, np.where(ref, 9, 1))
    g["_rank1"] = g["family"].map(FAM_RANK).fillna(8)
    g = g.sort_values(["_rank0", "_rank1", within],
                      ascending=[True, True, ascending])
    return g.drop(columns=["_rank0", "_rank1"])


def fam_cell(fam) -> str:
    return FAM_SHORT.get(fam, "--")


def best_mask(series, lower_is_better: bool = True):
    """Index of the best value in a column, or None if there is nothing to mark."""
    s = series.dropna()
    if s.empty:
        return None
    return (s.idxmin() if lower_is_better else s.idxmax())

STATUS_COLOR = {"success": "#2ca02c", "lost": "#e67e22",
                "timeout": "#1f77b4", "oom": "#c0392b", "error": "#9e9e9e"}

PHASES = [("baseline", "Baseline"),
          ("feature_extraction", "Feature extraction"),
          ("loop_closure", "Loop closure"),
          ("backend", "Back end"),
          ("combined", "Combined"),
          ("depth_estimation", "Depth")]

C_ATE_M   = "ate_rmse_succ_mean"
C_ATE_SD  = "ate_rmse_succ_std"
C_ATE_MED = "ate_rmse_succ_median"
C_FPS     = "avg_fps_all_median"
C_GPU     = "gpu_avg_all_mean"
C_CPU     = "cpu_avg_all_mean"
C_RAM     = "ram_max_gb_all_mean"
C_THREADS = "cpu_busy_threads_all_mean"
C_SCALE   = "ate_scale_succ_median"
C_COMPL   = "completion_pct_mean"
R_GPU     = "rel_gpu_avg_all_mean"
R_CPU     = "rel_cpu_avg_all_mean"
R_FPS     = "rel_avg_fps_all_median"
R_ATE     = "rel_ate_rmse_succ_median"

KEY = ["exp_type", "variant", "test_id"]


def disp(s: str) -> str:
    return DISPLAY.get(s, s)


def _n(v, d=1, suffix=""):
    return f"{v:.{d}f}{suffix}" if pd.notna(v) else "--"


def load_all(csv_path: Path, mode: str = "SLAM") -> pd.DataFrame:
    """Cell-level rows for EVERY condition, probes excluded.

    Filters out per-run rows, the AVG roll-up rows, and single-sample probe
    runs, so grouping downstream doesn't double-count or mix sample sizes."""
    if not csv_path.is_file():
        sys.exit(f"cannot find {csv_path} -- run aggregate_all first")
    d = pd.read_csv(csv_path, low_memory=False)
    if "level" not in d.columns:
        sys.exit(f"{csv_path.name} has no 'level' column: it predates the "
                 f"three-level CSV. Re-run aggregate_all.")
    d = d[(d["level"] == "cell") & (d["mode"] == mode)
          & (d["test_id"] != "AVG")].copy()
    if "is_probe" in d.columns:
        d = d[~d["is_probe"].fillna(False).astype(bool)]
    if d.empty:
        sys.exit(f"no cell rows for mode={mode}")
    return d


def one(d: pd.DataFrame, condition: str) -> pd.DataFrame:
    return d[d["condition"] == condition].copy()


def common(d: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """Cells present under every condition, so the ladder compares like-for-like
    rather than being skewed by what each condition happened to attempt."""
    conditions = conditions or [c for c in LADDER if c in set(d.condition)]
    sets = [set(map(tuple, d[d.condition == c][KEY].values)) for c in conditions]
    keep = set.intersection(*sets) if sets else set()
    m = np.array([tuple(r) in keep for r in d[KEY].values])
    return d[m & d.condition.isin(conditions).values].copy()


def pivot(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """(cell x condition) matrix for one metric. The paired form every
    condition comparison needs: subtracting or dividing two conditions is
    only meaningful row by row on the same cell."""
    return d.pivot_table(index=KEY, columns="condition", values=col)


def baseline_flags(d: pd.DataFrame) -> dict:
    """Sequences where the baseline itself did not complete every run --
    these get a dagger since relative cost figures divide by it."""
    b = d[d["variant"] == BASELINE]
    return {r.test_id: bool(r.n_success < r.n_total) for r in b.itertuples()}


def phase_rows(d: pd.DataFrame, exp_types: list[str],
               extra: list[str] | None = None) -> pd.DataFrame:
    """One row per variant for one phase, per-sequence detail folded into
    columns. `extra` carries in variants from other phases as reference rows
    (used by the combined phase to show the text's baseline comparison)."""
    keep = list(exp_types) + [BASELINE]
    x = d[d["exp_type"].isin(keep) | d["variant"].isin(extra or [])].copy()
    if x.empty:
        return x

    g = x.groupby(["variant", "short_name", "family", "exp_type"],
                  dropna=False).agg(
        n_total=("n_total", "sum"), n_success=("n_success", "sum"),
        n_lost=("n_lost", "sum"), n_timeout=("n_timeout", "sum"),
        n_oom=("n_oom", "sum"), n_seqs=("test_id", "nunique"),
        gpu=(C_GPU, "mean"), cpu=(C_CPU, "mean"),
        fps=(C_FPS, "median"), ram=(C_RAM, "max"),
        rel_gpu=(R_GPU, "mean"), rel_cpu=(R_CPU, "mean"),
        rel_fps=(R_FPS, "median"), scale=(C_SCALE, "median"),
    ).reset_index()

    g["cpu_per_frame"] = g["cpu"] / g["fps"].replace(0, np.nan)
    g["compl"] = g["n_success"] / g["n_total"].replace(0, np.nan) * 100
    g["skipped"] = g["n_seqs"] < len(SEQS)
    g["is_base"] = g["variant"] == BASELINE
    g["is_ref"] = g["variant"].isin(extra or [])

    for seq in SEQS:
        s = x[x["test_id"] == seq].set_index("variant")
        for tag, col in (("ate", C_ATE_M), ("sd", C_ATE_SD), ("rel", R_ATE),
                         ("succ", "n_success"), ("tot", "n_total"),
                         ("cmp", C_COMPL)):
            g[f"{tag}_{seq}"] = g["variant"].map(s[col])

    g["fam_rank"] = g["family"].map(
        {f: i for i, f in enumerate(FAM_ORDER)}).fillna(9)
    return g.sort_values("compl", ascending=False)


def fmt_ate(row, seq: str, bflag: dict) -> str:
    """`2.63 (3/3)`, a dash, or how far it got. Std dev omitted (too wide for
    the table; it's in the .csv twin instead) but the completion count stays."""
    a = row.get(f"ate_{seq}")
    s_, t = row.get(f"succ_{seq}"), row.get(f"tot_{seq}")
    if pd.isna(t) or t == 0:
        return "--"
    if pd.isna(a) or pd.isna(s_) or s_ == 0:
        c = row.get(f"cmp_{seq}")
        return f"lost @{c:.0f}\\%" if pd.notna(c) else "lost"
    dag = "$^\\dagger$" if bflag.get(seq) else ""
    return f"{a:.2f} ({int(s_)}/{int(t)}){dag}"


FIG_DIR: Path | None = None
TAB_DIR: Path | None = None
TABLE_CSV = False


def _save(fig, out: Path, name: str) -> None:
    d = FIG_DIR or out
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"   figures/{name}.png")


def _tex(out: Path, name: str, lines: list[str],
         df: pd.DataFrame | None = None) -> None:
    d = TAB_DIR or out
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if df is not None and TABLE_CSV:
        c = d.parent / "csv" / "tables"
        c.mkdir(parents=True, exist_ok=True)
        df.to_csv(c / f"{name}.csv", index=False)
    print(f"   tables/{name}.tex")


ADDED_BY_ME = {"liftfeat", "rdd", "dedode", "salad", "cricavpr", "supervlad"}

FE_GLANCE = [
    ("baseline", "classical", "binary",
     "ORB2 as shipped, the reference every other row is measured against",
     "murartal2017orbslam2"),
    ("orb", "classical", "binary",
     "the original oriented FAST keypoint with a rotated BRIEF descriptor",
     "rublee2011orb"),
    ("brisk", "classical", "binary",
     "scale-space FAST detector, descriptor sampled on concentric rings",
     "leutenegger2011brisk"),
    ("akaze", "classical", "binary",
     "nonlinear diffusion keeps edges sharp across scales", "alcantarilla2013akaze"),
    ("sift", "classical", "float",
     "the gradient-histogram reference, matched in Euclidean space", "lowe2004sift"),
    ("root_sift", "classical", "float",
     "SIFT under a Hellinger metric, a one-line change and usually better",
     "arandjelovic2012rootsift"),
    ("kaze", "classical", "float",
     "the float-descriptor ancestor of AKAZE, kept for the timing comparison",
     "alcantarilla2012kaze"),
    ("superpoint", "CNN", "joint",
     "the well-rounded reference: one network detects and describes",
     "detone2018superpoint"),
    ("xfeat", "CNN", "joint",
     "built for raw speed on a small backbone", "potje2024xfeat"),
    ("xfeat_xfeat", "CNN", "semi-dense",
     "XFeat paired with its own semi-dense matcher instead of nearest neighbour",
     "potje2024xfeat"),
    ("aliked", "CNN", "joint",
     "a deformable descriptor head keeps it light without losing accuracy",
     "zhao2023aliked"),
    ("liftfeat", "CNN", "joint",
     "3D-geometry cues added to a lightweight backbone", "liu2025liftfeat"),
    ("disk", "CNN", "dense",
     "policy-gradient training over dense features, heavy to run",
     "tyszkiewicz2020disk"),
    ("r2d2", "CNN", "dense",
     "predicts repeatability and reliability as separate maps", "revaud2019r2d2"),
    ("d2net", "CNN", "dense",
     "detection deferred until after description, on a dense feature map",
     "dusmanu2019d2net"),
    ("keynet", "CNN", "detector",
     "detector-centric and paired with a separate descriptor, unlike its neighbours",
     "barrosolaguna2019keynet"),
    ("lightglue", "transformer", "matcher",
     "convolutional features, attention matching over both images jointly",
     "lindenberger2023lightglue"),
    ("xfeat_lg", "transformer", "matcher",
     "the same attention matcher on XFeat's cheaper front end",
     "lindenberger2023lightglue"),
    ("rdd", "transformer", "feature",
     "attention pushed into the feature stage via a deformable transformer",
     "chen2025rdd"),
    ("dedode", "transformer", "feature",
     "a Vision-Transformer descriptor, detection and description trained apart",
     "edstedt2024dedode"),
]

LC_GLANCE = [
    ("baseline", "classical", "BoW, offline",
     "DBoW3 as shipped, reusing the tracker's own ORB descriptors",
     "galvezlopez2012dbow2", "DBoW3 (baseline)"),
    ("dbow2", "classical", "BoW, offline",
     "the 2012 bag of binary words, vocabulary trained ahead of the run",
     "galvezlopez2012dbow2"),
    ("ibow", "classical", "BoW, online",
     "builds its vocabulary incrementally as the drive proceeds",
     "garciafidalgo2018ibow"),
    ("obindex2", "classical", "BoW, online",
     "the incremental index behind iBoW, used directly", "garciafidalgo2018ibow"),
    ("vlad", "classical", "aggregation",
     "local descriptors pooled into one vector, no vocabulary lookup",
     "jegou2010vlad"),
    ("sad", "classical", "whole image",
     "down-sampled images compared directly, no descriptor stage at all", None),
    ("alexnet", "CNN", "global desc.",
     "the deliberately weak older baseline for the learned descriptors",
     "krizhevsky2012alexnet"),
    ("netvlad", "CNN", "global desc.",
     "the classic learned aggregation layer over convolutional features",
     "arandjelovic2016netvlad"),
    ("cosplace", "CNN", "global desc.",
     "trained for place recognition as a classification problem",
     "berton2022cosplace"),
    ("eigenplaces", "CNN", "global desc.",
     "trained on viewpoint-diverse views of the same place",
     "berton2023eigenplaces"),
    ("salad", "transformer", "DINOv2",
     "optimal-transport aggregation over the frozen backbone",
     "izquierdo2024salad"),
    ("cricavpr", "transformer", "DINOv2",
     "adds cross-image correlation between candidate keyframes", "lu2024cricavpr"),
    ("supervlad", "transformer", "DINOv2",
     "a VLAD-style descriptor compressed to a compact vector", "lu2024supervlad"),
    ("megaloc", "transformer", "DINOv2",
     "the recent all-rounder, trained across many place-recognition datasets",
     "berton2025megaloc"),
]


def glance_table(spec, out, name, label, caption, head="Method"):
    """One line per method: what it is, not how it did. Row order matches the
    results tables (baseline, Cls, CNN, Trf); a star marks a module
    integrated for this study."""
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}llp{1.9cm}p{6.5cm}@{}}", r"\toprule",
             rf"\textbf{{{head}}} & \textbf{{fam.}} & \textbf{{type}} & "
             r"\textbf{in one line} \\", r"\midrule"]
    rank = {f: i for i, f in enumerate(FAM_ORDER)}
    ordered = sorted(enumerate(spec),
                     key=lambda e: (0 if e[1][0] == BASELINE else 1,
                                    rank.get(e[1][1], 8), e[0]))
    seen = None
    for _, entry in ordered:
        key, fam, kind, blurb, cite = entry[:5]
        nm = entry[5] if len(entry) > 5 else disp(key)
        if key == BASELINE:
            nm = rf"\textbf{{{nm}}}"
        elif fam != seen and seen is not None:
            lines.append(r"\addlinespace[2pt]")
        seen = fam
        if key in ADDED_BY_ME:
            nm += r"$^\star$"
        if cite:
            nm += rf" \cite{{{cite}}}"
        lines.append(f"{nm} & {fam_cell(fam)} & {kind} & {blurb} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, name, lines)

def sw_accuracy_table(g, bflag, out, name, label, caption, head="Method",
                      show_fam=True):
    """Accuracy per sequence and completion. No cost columns -- those answer
    the hardware section's question and are emitted by hw_cost_table()."""
    if g.empty:
        return
    best = {}
    for seq in SEQS:
        ok = g[(g[f"succ_{seq}"].fillna(0) > 0) & ~g["is_ref"]]
        best[seq] = ok[f"ate_{seq}"].min() if not ok.empty else None

    fam_col = r" & \textbf{fam.}" if show_fam else ""
    tab_spec = "llcrrrr" if show_fam else "lcrrrr"
    n_cols = 7 if show_fam else 6
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             rf"\begin{{tabular}}{{@{{}}{tab_spec}@{{}}}}", r"\toprule",
             rf"\textbf{{{head}}}{fam_col} & \textbf{{compl.}} & "
             r"\texttt{test0} & \texttt{test1} & \texttt{test2} & "
             r"\texttt{test3} \\", r"\midrule"]

    body = order_rows(g[~g["is_ref"]])
    refs = order_rows(g[g["is_ref"]])
    for block, sep in ((body, False), (refs, True)):
        if block.empty:
            continue
        if sep:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\itshape "
                         r"for reference, from other phases} \\")
        for r in block.itertuples():
            rd = r._asdict()
            nm = disp(r.short_name)
            if r.is_base:
                nm = rf"\textbf{{{nm}}}"
            cells = []
            for seq in SEQS:
                c = fmt_ate(rd, seq, bflag)
                a, s_ = rd.get(f"ate_{seq}"), rd.get(f"succ_{seq}")
                if (best[seq] is not None and pd.notna(a) and (s_ or 0) > 0
                        and a == best[seq] and not r.is_ref):
                    c = rf"\textbf{{{c}}}"
                cells.append(c)
            fam_cl = f" & {fam_cell(r.family)}" if show_fam else ""
            lines.append(f"{nm}{fam_cl} & "
                         f"{int(r.n_success)}/{int(r.n_total)} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    keep = (["short_name", "family", "exp_type", "n_success", "n_total",
             "compl", "skipped"]
            + [f"{q}_{s}" for s in SEQS
               for q in ("ate", "sd", "succ", "tot", "rel")])
    _tex(out, name, lines, g[[c for c in keep if c in g.columns]].round(3))


def sw_depth_table(g, bflag, out):
    """Depth keeps the recovered scale (an accuracy diagnostic, not a cost
    one) which no other phase has; throughput/GPU columns go to hardware."""
    if g.empty:
        return
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Monocular plus neural depth at the reference "
             r"condition, against the stereo baseline. ``DA'' is "
             r"DepthAnything. ``Scale'' is the similarity scale recovered by "
             r"Umeyama alignment: 1.0 means the reconstruction is metrically "
             r"correct, and a value below~1 means the map is too small by that "
             r"factor before the alignment absorbs it.}",
             r"\label{tab:sw-depth}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lccrrrr@{}}", r"\toprule",
             r"\textbf{Configuration} & \textbf{compl.} & \textbf{scale} & "
             r"\texttt{test0} & \texttt{test1} & \texttt{test2} & "
             r"\texttt{test3} \\", r"\midrule"]
    g = order_rows(g)
    best = {s: (g.loc[g[f"succ_{s}"].fillna(0) > 0, f"ate_{s}"].min()
                if (g[f"succ_{s}"].fillna(0) > 0).any() else None)
            for s in SEQS}
    for r in g.itertuples():
        nm = disp(r.short_name)
        if r.is_base:
            nm = rf"\textbf{{{nm}}}"
        rd = r._asdict()
        cells = []
        for s in SEQS:
            c = fmt_ate(rd, s, bflag)
            a, sc = rd.get(f"ate_{s}"), rd.get(f"succ_{s}")
            if best[s] is not None and pd.notna(a) and (sc or 0) > 0 \
                    and a == best[s]:
                c = rf"\textbf{{{c}}}"
            cells.append(c)
        lines.append(f"{nm} & {int(r.n_success)}/{int(r.n_total)} & "
                     f"{_n(r.scale, 2)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "sw_depth_table", lines)


def sw_backend_table(d_all, out):
    """The solver comparison at both 28 GB conditions -- two conditions
    because the finding is a null, which needs confirming on more than one
    run."""
    rows = []
    for cond in (REFERENCE, CHECK):
        dd = one(d_all, cond)
        if dd.empty:
            continue
        g = phase_rows(dd, ["backend"])
        a = g[g.variant == BASELINE]
        b = g[g.variant == "be_gtsam"]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        for seq in SEQS:
            rows.append({"condition": COND_SHORT.get(cond, cond), "seq": seq,
                         "g2o": a.get(f"ate_{seq}"), "g2o_sd": a.get(f"sd_{seq}"),
                         "g2o_n": f"{int(a.get(f'succ_{seq}') or 0)}/{int(a.get(f'tot_{seq}') or 0)}",
                         "gtsam": b.get(f"ate_{seq}"), "gtsam_sd": b.get(f"sd_{seq}"),
                         "gtsam_n": f"{int(b.get(f'succ_{seq}') or 0)}/{int(b.get(f'tot_{seq}') or 0)}"})
    if not rows:
        return
    df = pd.DataFrame(rows)
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Back-end solver comparison on an otherwise identical "
             r"ORB2 pipeline, at both 28\,GB conditions. ATE is the mean "
             r"$\pm$ standard deviation over completed runs, with completed "
             r"over attempted beside it. The standard deviations are kept here "
             r"because the whole claim of this phase is that the two solvers "
             r"do not separate, and that cannot be judged without them.}",
             r"\label{tab:sw-be}", r"\small",
             r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"Condition & Seq. & g2o (baseline) & GTSAM \\", r"\midrule"]
    cur = None
    for r in df.itertuples():
        def f(v, sd, n):
            if pd.isna(v):
                return f"-- ({n})"
            return f"{v:.2f}" + (f"$\\pm${sd:.2f}" if pd.notna(sd) else "") + f" ({n})"
        c = r.condition if r.condition != cur else ""
        if c and cur is not None:
            lines.append(r"\midrule")
        cur = r.condition
        lines.append(f"{c} & \\texttt{{{r.seq}}} & {f(r.g2o, r.g2o_sd, r.g2o_n)}"
                     f" & {f(r.gtsam, r.gtsam_sd, r.gtsam_n)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "sw_be_table", lines, df.round(3))


def sw_fig_backend(d_all, out):
    """g2o against GTSAM, per sequence, at both 28 GB conditions. Error bars
    matter here since the finding is that the two solvers don't separate;
    paired points (not bars from zero) so the eye reads the gap between them."""
    conds = [c for c in (REFERENCE, CHECK) if not one(d_all, c).empty]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(5.6 * len(conds), 4.6),
                             sharey=True, squeeze=False)
    drew = False
    for ax, cond in zip(axes[0], conds):
        g = phase_rows(one(d_all, cond), ["backend"])
        a = g[g.variant == BASELINE]
        b = g[g.variant == "be_gtsam"]
        if a.empty or b.empty:
            ax.set_visible(False)
            continue
        a, b = a.iloc[0], b.iloc[0]
        x = np.arange(len(SEQS))
        for i, seq in enumerate(SEQS):
            va, vb = a.get(f"ate_{seq}"), b.get(f"ate_{seq}")
            if pd.notna(va) and pd.notna(vb):
                ax.plot([i, i], [va, vb], color="#bbb", lw=1.2, zorder=1)
        for r, lbl, col, dx in ((a, "g2o (baseline)", "#1b9e77", -.07),
                                (b, "GTSAM", "#7570b3", .07)):
            v = np.array([r.get(f"ate_{s}") for s in SEQS], dtype=float)
            e = np.array([r.get(f"sd_{s}") for s in SEQS], dtype=float)
            ax.errorbar(x + dx, v, yerr=np.nan_to_num(e), fmt="o", ms=7,
                        color=col, ecolor=col, elinewidth=1.4, capsize=4,
                        label=lbl, zorder=3)
            for i, s in enumerate(SEQS):
                n = f"{int(r.get(f'succ_{s}') or 0)}/{int(r.get(f'tot_{s}') or 0)}"
                if pd.notna(v[i]):
                    ax.annotate(n, (i + dx, v[i]), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=6.5,
                                color=col)
        drew = True
        ax.set_xticks(x)
        ax.set_xticklabels(SEQS, fontsize=9)
        ax.set_yscale("log")
        ax.set_title(COND_LABEL.get(cond, cond), fontsize=10.5)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    if not drew:
        plt.close(fig)
        return
    axes[0][0].set_ylabel("mean ATE over completed runs (m), log scale")
    axes[0][0].legend(fontsize=9, loc="upper left", frameon=False)
    fig.suptitle("Back end: the two solvers on an otherwise identical "
                 "pipeline\nerror bars are the run-to-run spread, labels are "
                 "completed/attempted", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _save(fig, out, "sw_backend")

def sw_fig_robustness(g, out, name, title):
    """Completion split by outcome (LOST = algorithm's own failure, OOM/TIMEOUT
    = the machine's). At the reference condition the budget bands should be
    nearly invisible, which is what licenses reading the rest as algorithmic."""
    g = order_rows(g[~g["is_ref"]])[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.36 * len(g) + 2.4))
    y = np.arange(len(g)); left = np.zeros(len(g))
    for key in ("success", "lost", "timeout", "oom"):
        v = (g[f"n_{key}"] / g["n_total"] * 100).fillna(0).values
        if v.sum() == 0:
            continue
        ax.barh(y, v, left=left, height=.66, color=STATUS_COLOR[key],
                label=key.upper())
        for yi, (a, l) in enumerate(zip(v, left)):
            if a >= 9:
                ax.text(l + a / 2, yi, f"{a:.0f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{disp(r.short_name)}  ({int(r.n_success)}/{int(r.n_total)})"
                        for r in g.itertuples()], fontsize=8)
    for lbl, b in zip(ax.get_yticklabels(), g["is_base"]):
        if b:
            lbl.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of runs (%)")
    ax.set_title(f"{title}\nshares are over the sequences each method attempted",
                 fontsize=10.5)

    outcomes = ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.09),
                         frameon=False, fontsize=8.5, title="bar: run outcome")
    outcomes.get_title().set_fontsize(8.5)
    ax.grid(axis="x", alpha=.22); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, name)


def sw_fig_lc_by_sequence(g, out):
    """Loop-closure accuracy per sequence, one panel each -- the finding is
    that the ranking CHANGES with the sequence, which a pooled number would
    hide. 2x2 grid rather than 1x4 to keep method labels readable."""
    g = g[~g["is_ref"]]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.4))
    for ax, seq in zip(axes.ravel(), SEQS):
        s = order_rows(g[g[f"succ_{seq}"].fillna(0) > 0],
                       within=f"ate_{seq}", ascending=True)
        if s.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(s))
        ax.barh(y, s[f"ate_{seq}"], height=.66, edgecolor="white",
                color=[FAM_COLOR.get(f, "#999") for f in s.family])
        ax.errorbar(s[f"ate_{seq}"], y, xerr=s[f"sd_{seq}"].fillna(0),
                    fmt="none", ecolor="#444", elinewidth=1.0, capsize=2.5)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{disp(r.short_name)} ({int(r._asdict()[f'succ_{seq}'])}/"
             f"{int(r._asdict()[f'tot_{seq}'])})" for r in s.itertuples()],
            fontsize=7.5)
        for lbl, b in zip(ax.get_yticklabels(), s["is_base"]):
            if b:
                lbl.set_fontweight("bold")
        ax.invert_yaxis(); ax.set_xlabel("ATE (m)")
        ax.set_title(seq, fontsize=10)
        ax.grid(axis="x", alpha=.22); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
                        for f in FAM_ORDER],
               ncol=3, loc="lower center", frameon=False, fontsize=10)
    fig.suptitle("Loop closure: mean ATE over completed runs, per sequence\n"
                 "bars carry the run-to-run spread and are labelled with "
                 "completed/attempted", fontsize=12)
    fig.tight_layout(rect=(0, .035, 1, 1))
    _save(fig, out, "sw_lc_by_sequence")


def sw_fig_combined(g, out):
    """Combined pairs: completion left, drift right -- two panels because
    here they disagree most violently (some pairs complete every run but
    post the largest errors in the study, so no single score ranks fairly)."""
    g = order_rows(g[(g.exp_type == "combined") | g["is_base"]])
    if g.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 0.44 * len(g) + 3.0),
                             sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.5]})
    y = np.arange(len(g))
    axes[0].barh(y, g.compl, height=.66, edgecolor="white",
                 color=[FAM_COLOR.get(f, "#999") for f in g.family])
    for yi, r in zip(y, g.itertuples()):
        axes[0].text(2, yi, f"{int(r.n_success)}/{int(r.n_total)}", va="center",
                     fontsize=7.5, color="white", fontweight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("completed runs (%)")
    axes[0].set_title("Did it finish?", fontsize=10.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([disp(s) for s in g.short_name], fontsize=8.5)
    for lbl, b in zip(axes[0].get_yticklabels(), g["is_base"]):
        if b:
            lbl.set_fontweight("bold")

    w = .2
    for i, seq in enumerate(SEQS):
        axes[1].barh(y + (i - 1.5) * w, g[f"ate_{seq}"].values, height=w,
                     label=seq, color=plt.cm.viridis(i / 3.4),
                     edgecolor="white", linewidth=.4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean ATE over completed runs (m), log scale")
    axes[1].set_title("Was the map any good?", fontsize=10.5)
    axes[1].legend(fontsize=8, ncol=4, loc="lower right")
    axes[1].grid(axis="x", alpha=.22); axes[1].set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Combined configurations: completion and accuracy are not the "
                 "same ranking\na long bar on the left beside a long bar on "
                 "the right is a finished drive with a broken map", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _save(fig, out, "sw_combined")


def sw_fig_depth_scale(g, out):
    """Depth: error against recovered similarity scale. Stereo aligns at 1.0
    (measured baseline); a monocular scale of 0.75 means the reconstructed
    world is a quarter too small, absorbed by alignment before ATE is
    computed."""
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.axvspan(.97, 1.03, color="#2ca02c", alpha=.10, zorder=0)
    ax.axvline(1.0, color="#2ca02c", ls="--", lw=1.3, zorder=1)
    ax.annotate("metrically correct", (1.0, .93),
                xycoords=("data", "axes fraction"), fontsize=8.5,
                color="#2ca02c", ha="center", fontweight="bold")
    for _, r in g.iterrows():
        if pd.isna(r.scale):
            continue
        for seq in SEQS:
            v = r.get(f"ate_{seq}")
            if pd.isna(v) or (r.get(f"succ_{seq}") or 0) == 0:
                continue
            base = r.variant == BASELINE
            ax.scatter(r.scale, v, s=150 if base else 95,
                       marker="*" if base else "o", zorder=4, alpha=.9,
                       color=FAM_COLOR.get(r.family, "#999"),
                       edgecolor="white", linewidth=1.0)
            ax.annotate(f"{disp(r.short_name)}  {seq}", (r.scale, v),
                        textcoords="offset points", xytext=(8, 2),
                        fontsize=7, color="#333")
    ax.set_yscale("log")
    ax.set_xlabel("recovered similarity scale (1.0 = metrically correct)")
    ax.set_ylabel("mean ATE over completed runs (m), log scale")
    ax.set_title("Depth: the error follows the scale\n"
                 "stereo sits in the green band because its scale is a "
                 "measured baseline; a predicted depth map is an inference",
                 fontsize=10.5)
    ax.grid(alpha=.22, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "sw_depth_scale")


def hw_ladder_table(cc, out):
    """The headline table: one row per condition, on the common cells. ``Succ.''
    is the raw completion share; ``Alg.'' divides successes by successes plus
    tracking losses (removing budget deaths) -- the fairer per-component
    score. The two converge as the budget grows, justifying the software
    section's single-condition analysis at 28 GB."""
    rows = []
    for c in [k for k in LADDER if k in set(cc.condition)]:
        d = cc[cc.condition == c]
        tot = max(d.n_total.sum(), 1)
        alg = d.n_success.sum() + d.n_lost.sum()
        ate = d.loc[d.n_success > 0, C_ATE_MED].dropna()
        rows.append({
            "condition": COND_LABEL.get(c, c), "runs": int(d.n_total.sum()),
            "succ": 100 * d.n_success.sum() / tot,
            "lost": 100 * d.n_lost.sum() / tot,
            "timeout": int(d.n_timeout.sum()), "oom": int(d.n_oom.sum()),
            "alg": 100 * d.n_success.sum() / alg if alg else np.nan,
            "fps": d[C_FPS].median(), "ram": d[C_RAM].median(),
            "cpu": d[C_CPU].median(), "gpu": d[C_GPU].median(),
            "thr": d[C_THREADS].median(),
            "ate": ate.median() if len(ate) else np.nan})
    if not rows:
        return
    df = pd.DataFrame(rows)
    n_cells = len(cc) // max(cc.condition.nunique(), 1)
    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{The hardware ladder, on the {n_cells} cells common "
             r"to all four conditions. The runs column is not identical across "
             r"rows, because the baseline carries extra repetitions at the "
             r"reference condition. ``Succ.'' is the raw share of completed "
             r"runs; "
             r"``alg.'' is the share over the algorithm-attributable subset "
             r"(successes plus tracking losses, with memory and timeout deaths "
             r"removed). The two converge as the budget grows and are "
             r"identical once no run dies of memory.}",
             r"\label{tab:hw-ladder}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
             r"Condition & runs & succ.\ \% & alg.\ \% & lost \% & OOM & "
             r"t/out & fps & peak GB & ATE m \\", r"\midrule"]
    for r in df.itertuples():
        lines.append(f"{r.condition} & {r.runs} & {_n(r.succ)} & {_n(r.alg)} & "
                     f"{_n(r.lost)} & {r.oom} & {r.timeout} & {_n(r.fps, 2)} & "
                     f"{_n(r.ram, 1)} & {_n(r.ate, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "hw_ladder_table", lines, df.round(3))


def hw_outcome_table(cc, out):
    """Completed runs over attempts, every configuration at every condition --
    a row read left to right shows one configuration's response to more
    memory and more cores."""
    conds = [k for k in LADDER if k in set(cc.condition)]
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Completed runs over attempts for every configuration "
             r"at every hardware condition, on the common cells. Read a row "
             r"left to right to see what that configuration did with more "
             r"memory and more threads.}",
             r"\label{tab:hw-outcome}", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}ll" + "r" * len(conds) + r"@{}}", r"\toprule",
             r"Configuration & fam. & "
             + " & ".join(COND_SHORT.get(c, c) for c in conds) + r" \\",
             r"\midrule"]
    first = True
    for exp, title in PHASES:
        sub = cc[cc.exp_type == exp]
        if sub.empty:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(rf"\multicolumn{{{2 + len(conds)}}}{{@{{}}l}}"
                     rf"{{\itshape {title}}} \\")
        g = (sub.groupby(["short_name", "family", "variant"])
             .agg(s=("n_success", "sum"), n=("n_total", "sum")).reset_index())
        g["rate"] = g.s / g.n
        g["is_base"] = g["variant"] == BASELINE
        g["is_ref"] = False
        for r in order_rows(g, within="rate").itertuples():
            cells = []
            for c in conds:
                q = sub[(sub.short_name == r.short_name) & (sub.condition == c)]
                cells.append(f"{int(q.n_success.sum())}/{int(q.n_total.sum())}"
                             if not q.empty else "--")
            nm = disp(r.short_name)
            if r.is_base:
                nm = rf"\textbf{{{nm}}}"
            lines.append(f"{nm} & {fam_cell(r.family)} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "hw_outcome_table", lines)


def hw_cost_table(g, out, name, label, caption, head="Method", show_fam=True):
    """What the machine paid: absolute, plus the ratio to the baseline (the
    part that should survive on different hardware). Raw CPU% FALLS as
    methods get deeper -- that's the GPU-bound pipeline idling the CPU, not
    less work; CPU/frame (dividing by throughput) is the column that shows
    the true, reversed ordering."""
    if g.empty:
        return
    g = order_rows(g[~g["is_ref"]], within="rel_gpu", ascending=True)
    win = {"gpu": best_mask(g["gpu"]), "cpu": best_mask(g["cpu"]),
           "fps": best_mask(g["fps"], lower_is_better=False),
           "ram": best_mask(g["ram"]),
           "cpu_per_frame": best_mask(g["cpu_per_frame"])}
    fam_col = r" & \textbf{fam.}" if show_fam else ""
    tab_spec = "llrrrrr" if show_fam else "lrrrrr"
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             rf"\begin{{tabular}}{{@{{}}{tab_spec}@{{}}}}", r"\toprule",
             rf"\textbf{{{head}}}{fam_col} & \textbf{{GPU\,\% (rel)}} "
             r"& \textbf{CPU\,\% (rel)} & \textbf{fps (rel)} & "
             r"\textbf{peak GB} & \textbf{CPU/frame} \\", r"\midrule"]
    for r in g.itertuples():
        nm = disp(r.short_name)
        if r.is_base:
            nm = rf"\textbf{{{nm}}}"

        def pair(v, rel, key, d=1):
            if pd.isna(v):
                return "--"
            s = f"{v:.{d}f}" + (rf"\,({rel:.2f}$\times$)" if pd.notna(rel) else "")
            return rf"\textbf{{{s}}}" if win.get(key) == r.Index else s

        def solo(v, key, d=1):
            s = _n(v, d)
            return rf"\textbf{{{s}}}" if win.get(key) == r.Index else s
        fam_cl = f" & {fam_cell(r.family)}" if show_fam else ""
        lines.append(f"{nm}{fam_cl} & "
                     f"{pair(r.gpu, r.rel_gpu, 'gpu')} & "
                     f"{pair(r.cpu, r.rel_cpu, 'cpu')} & "
                     f"{pair(r.fps, r.rel_fps, 'fps')} & "
                     f"{solo(r.ram, 'ram', 1)} & "
                     f"{solo(r.cpu_per_frame, 'cpu_per_frame', 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    keep = ["short_name", "family", "gpu", "rel_gpu", "cpu", "rel_cpu",
            "fps", "rel_fps", "ram", "cpu_per_frame"]
    _tex(out, name, lines, g[[c for c in keep if c in g.columns]].round(3))


def hw_fig_outcome_ladder(cc, out):
    """Outcome share, one bar per condition, down the ladder -- the headline
    figure. Annotations between bars name the one factor each rung changes."""
    conds = [k for k in LADDER if k in set(cc.condition)]
    fig, ax = plt.subplots(figsize=(11, 1.05 * len(conds) + 2.4))
    y = np.arange(len(conds)); left = np.zeros(len(conds))
    for key in ("success", "lost", "timeout", "oom"):
        v = np.array([cc[cc.condition == c][f"n_{key}"].sum() for c in conds],
                     dtype=float)
        tot = np.array([max(cc[cc.condition == c].n_total.sum(), 1)
                        for c in conds], dtype=float)
        pct = v / tot * 100
        if pct.sum() == 0:
            continue
        ax.barh(y, pct, left=left, color=STATUS_COLOR[key], height=.62,
                label=key.upper())
        for yi, (p, l, n) in enumerate(zip(pct, left, v)):
            if p >= 4:
                ax.text(l + p / 2, yi, f"{p:.0f}%\n({n:.0f})", ha="center",
                        va="center", fontsize=7.5, color="white")
        left += pct
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}.  {COND_LABEL.get(c, c)}"
                        for i, c in enumerate(conds)], fontsize=9,
                       family="monospace")
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("share of runs (%)")
    ax.set_title(f"Run outcomes down the hardware ladder\n"
                 f"common cells only ({len(cc[cc.condition == conds[0]])} per "
                 f"condition); each condition changes exactly one factor",
                 fontsize=11)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.16),
              frameon=False)
    idx = {c: i for i, c in enumerate(conds)}
    for a, b, nm in RUNGS:
        if a in idx and b in idx:
            ax.annotate(f"changes {nm}", xy=(101, (idx[a] + idx[b]) / 2),
                        fontsize=7.5, style="italic", color="0.35",
                        va="center", annotation_clip=False)
    fig.tight_layout()
    _save(fig, out, "hw_outcome_ladder")


def hw_fig_speedup(cc, out):
    """Who benefits from the faster platform, by family and by phase. "Twice
    as fast" isn't uniform: CPU-bound classical extractors get the full
    generational win, GPU-bound deep variants (GPU unchanged) get pulled
    toward 1x."""
    fps = pivot(cc, C_FPS).dropna()
    src, dst = RUNGS[0][0], RUNGS[0][1]
    if src not in fps or dst not in fps:
        return
    sp = (fps[dst] / fps[src]).replace([np.inf, -np.inf], np.nan).dropna()
    fam = pd.Series(cc.drop_duplicates(KEY).set_index(KEY)["family"])
    fam = fam.reindex(sp.index)
    phase = pd.Series(sp.index.get_level_values("exp_type"), index=sp.index)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, grp, order, title in (
            (axes[0], fam, [f for f in FAM_ORDER if f in set(fam.dropna())],
             "by architecture family"),
            (axes[1], phase, [p for p, _ in PHASES if p in set(phase)],
             "by pipeline block")):
        vals = [sp[grp == k].values for k in order]
        for i, v in enumerate(vals):
            if len(v) == 0:
                continue
            col = FAM_COLOR.get(order[i], "#6a8caf")
            ax.bar(i, np.median(v), width=.62, color=col, edgecolor="white")
            j = (np.random.default_rng(0).random(len(v)) - .5) * .32
            ax.scatter(i + j, v, s=13, color="#222", alpha=.55, zorder=3,
                       linewidths=0)
            ax.text(i, np.median(v), f"  {np.median(v):.2f}x (n={len(v)})",
                    ha="center", va="bottom", fontsize=8.5, zorder=4)
        ax.axhline(1.0, color="#888", ls="--", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([FAM_SHORT.get(o, o).replace("_", " ")
                            for o in order], fontsize=8.5, rotation=15,
                           ha="right")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", ls="--", alpha=.35); ax.set_axisbelow(True)
    axes[0].set_ylabel("throughput ratio (new platform / pilot)")
    fig.suptitle(f"Who actually benefits from the faster platform\n"
                 f"{COND_LABEL.get(src, src)}  $\\rightarrow$  "
                 f"{COND_LABEL.get(dst, dst)}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _save(fig, out, "hw_speedup")


def hw_fig_ram_relief(cc, out):
    """The capacity condition, cell by cell: peak RAM before and after.
    Dumbbell plot (not bars from zero) since the quantity of interest is the
    CHANGE. Right panel uses %-of-own-budget, since 13.9 GB absolute means
    suffocating at 14 GB but comfortable at 28 GB -- only utilisation of each
    machine's own ceiling is comparable across the two."""
    src, dst = RUNGS[1][0], RUNGS[1][1]
    ram = pivot(cc, C_RAM)
    st, tt = pivot(cc, "n_success"), pivot(cc, "n_total")
    if src not in ram or dst not in ram:
        return
    budget = {src: 14.0, dst: 28.0}
    recs = []
    for k in ram.dropna(subset=[src, dst]).index:
        was_ok = st.loc[k, src] == tt.loc[k, src]
        now_ok = st.loc[k, dst] == tt.loc[k, dst]
        kind = ("rescued" if now_ok and not was_ok else
                "control" if now_ok and was_ok else
                "regressed" if was_ok else "still failing")
        recs.append({"label": f"{disp(k[1].split('_', 1)[-1])} [{k[2]}]",
                     "a": ram.loc[k, src], "b": ram.loc[k, dst], "kind": kind})
    if not recs:
        return
    style = {"rescued": "#1e8449", "regressed": "#c0392b",
             "still failing": "#b45309", "control": "#9e9e9e"}
    order = ["rescued", "regressed", "still failing", "control"]
    rows = [r for k in order for r in sorted(
        (x for x in recs if x["kind"] == k), key=lambda z: -z["a"])][:30]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, .38 * len(rows) + 3.2),
                                   sharey=True,
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        c = style[r["kind"]]
        axL.plot([r["a"], r["b"]], [i, i], color=c, lw=2.4, alpha=.55, zorder=2)
        axL.scatter(r["a"], i, s=52, facecolor="white", edgecolor=c,
                    linewidths=1.7, zorder=3)
        axL.scatter(r["b"], i, s=52, color=c, zorder=3)
        ua, ub = r["a"] / budget[src] * 100, r["b"] / budget[dst] * 100
        axR.plot([ua, ub], [i, i], color=c, lw=2.4, alpha=.55, zorder=2)
        axR.scatter(ua, i, s=52, facecolor="white", edgecolor=c,
                    linewidths=1.7, zorder=3)
        axR.scatter(ub, i, s=52, color=c, zorder=3)
    axL.axvline(budget[src], color="#c0392b", ls="--", lw=1.4)
    axL.set_xlabel("peak RAM (GB)")
    axL.set_title("Peak memory used\nhollow = 14 GB budget, filled = 28 GB",
                  fontsize=10)
    axR.axvline(100, color="#333", lw=1.8)
    axR.axvspan(90, 105, color="#c0392b", alpha=.07, zorder=0)
    axR.set_xlim(0, 106)
    axR.set_xlabel("peak RAM as \\% of that machine's own budget")
    axR.set_title("How close each run came to ITS ceiling\n"
                  "the only view in which the two budgets compare", fontsize=10)
    axL.set_yticks(y)
    axL.set_yticklabels([r["label"] for r in rows], fontsize=7.5,
                        family="monospace")
    axL.invert_yaxis()
    for ax in (axL, axR):
        ax.grid(axis="x", alpha=.22); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=style[k], label=k) for k in order],
               ncol=4, loc="lower center", frameon=False, fontsize=9)
    n_r = sum(1 for r in recs if r["kind"] == "rescued")
    fig.suptitle(f"What lifting the memory ceiling did, cell by cell\n"
                 f"{n_r} cells went from failing to completing every run",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, .04, 1, 1))
    _save(fig, out, "hw_ram_relief")


def hw_fig_threads(cc, out):
    """Cores given, cores used, speed gained, across the whole ladder. If busy
    threads climb while throughput doesn't, the parallel section has
    saturated and the rest is scheduling overhead."""
    conds = [k for k in LADDER if k in set(cc.condition)]
    avail = [int(k.split("_")[1].rstrip("t")) for k in conds]
    busy = [cc[cc.condition == k][C_THREADS].mean() for k in conds]
    fps = [cc[cc.condition == k][C_FPS].median() for k in conds]

    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    ax2 = ax.twinx()
    ax.plot(x, avail, marker="s", ls=":", color="#bbb", lw=1.4,
            label="threads available")
    ax.plot(x, busy, marker="o", color="#1d6fb8", lw=2.0,
            label="threads actually busy")
    ax2.plot(x, fps, marker="^", color="#2a9d8f", lw=2.0, label="median fps")
    for i, (b, f) in enumerate(zip(busy, fps)):
        ax.annotate(f"{b:.1f}", (i, b), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#1d6fb8")
        ax2.annotate(f"{f:.1f}", (i, f), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=8, color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds], fontsize=8,
                       rotation=12, ha="right")
    ax.set_ylabel("threads", color="#1d6fb8")
    ax2.set_ylabel("median fps", color="#2a9d8f")
    ax.set_ylim(0, max(avail) * 1.18)
    ax2.set_ylim(0, max(fps) * 1.35)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left", frameon=False)
    ax.set_title("Thread saturation: cores given, cores used, speed gained",
                 fontsize=11)
    ax.grid(axis="y", ls="--", alpha=.3); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "hw_thread_saturation")


def hw_accuracy_invariance(cc, out):
    """ATE on the cells that complete under every condition -- the result
    that justifies the software section using one machine: accuracy does
    not move with the hardware."""
    conds = [k for k in LADDER if k in set(cc.condition)]
    ate, comp = pivot(cc, C_ATE_MED), pivot(cc, C_COMPL)
    ok = (comp[conds] >= 98).all(axis=1) & ate[conds].notna().all(axis=1)
    sub = ate[conds][ok]
    if sub.empty:
        return
    p = None
    if _st is not None and len(sub) >= 5 and len(conds) >= 3:
        p = _st.friedmanchisquare(*[sub[c].values for c in conds]).pvalue

    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{ATE on the {len(sub)} cells that completed at least "
             r"98\% of their sequence under \emph{every} condition. If "
             r"accuracy moved with the hardware, no single-condition "
             r"comparison would generalise."
             + (rf" A Friedman test across the four conditions gives "
                rf"$p={p:.2f}$." if p is not None else "") + r"}",
             r"\label{tab:hw-invariance}", r"\small",
             r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Condition & median ATE (m) & mean ATE (m) \\", r"\midrule"]
    for c in conds:
        lines.append(f"{COND_LABEL.get(c, c)} & {sub[c].median():.3f} & "
                     f"{sub[c].mean():.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "hw_invariance_table", lines, sub.reset_index().round(4))


def gen_fig_vo_vs_slam(csv, out):
    """The same front ends, with and without loop closure, at BOTH budgets.
    One panel per budget: at 14 GB loop closure costs completion (bars move
    DOWN), at 28 GB it mostly helps (bars move UP) -- the reversal is the
    point and needs both panels to see."""
    panels = []
    for cond in ("R9_12t_14", REFERENCE):
        try:
            vo = phase_rows(one(load_all(csv, "VO"), cond),
                            ["feature_extraction"])
            sl = phase_rows(one(load_all(csv, "SLAM"), cond),
                            ["feature_extraction"])
        except SystemExit:
            continue
        if vo.empty or sl.empty:
            continue
        m = vo[["short_name", "family", "compl"]].merge(
            sl[["short_name", "compl"]], on="short_name",
            suffixes=("_vo", "_slam"))
        if not m.empty:
            panels.append((cond, m))
    if not panels:
        return

    _first = panels[0][1].copy()
    _first["is_base"] = _first["short_name"] == BASELINE
    _first["is_ref"] = False
    order = list(order_rows(_first, within="compl_vo").short_name)
    panels = [(c, m.set_index("short_name").reindex(order).dropna(how="all")
               .reset_index()) for c, m in panels]

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(0.46 * max(len(m) for _, m in panels) + 3.6,
                                      4.6 * len(panels)), squeeze=False)
    for ax, (cond, m) in zip(axes[:, 0], panels):
        x = np.arange(len(m)); w = .38
        ax.bar(x - w / 2, m.compl_vo, width=w, color="#8fb8d6",
               edgecolor="white", label="VO (front end alone)")
        ax.bar(x + w / 2, m.compl_slam, width=w, edgecolor="white",
               color=[FAM_COLOR.get(f, "#999") for f in m.family],
               label="full SLAM")
        down = 0
        for xi, (a, b) in enumerate(zip(m.compl_vo, m.compl_slam)):
            if pd.notna(a) and pd.notna(b) and b < a - 1:
                down += 1
                ax.annotate("", xy=(xi + w / 2, b + 2), xytext=(xi - w / 2, a + 2),
                            arrowprops=dict(arrowstyle="->", color="#c0392b",
                                            lw=1.2, alpha=.75))
        ax.set_xticks(x)
        ax.set_xticklabels([disp(s) for s in m.short_name], rotation=40,
                           ha="right", fontsize=8)
        for lbl, s_ in zip(ax.get_xticklabels(), m.short_name):
            if s_ == BASELINE:
                lbl.set_fontweight("bold")
        ax.set_ylabel("completed runs (%)")
        ax.set_ylim(0, 112)
        ax.set_title(f"{COND_LABEL.get(cond, cond)}   "
                     f"({down} of {len(m)} front ends lose completion when "
                     f"loop closure is switched on)", fontsize=10.5)
        ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor="#8fb8d6",
                              label="VO (front end alone)"),
                        Patch(facecolor="0.55", label="full SLAM "
                              "(bar colour = family)")]
               + [Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
                  for f in FAM_ORDER],
               ncol=5, loc="lower center", frameon=False, fontsize=9)
    fig.suptitle("Loop closure is a liability when starved and an asset when "
                 "fed\nred arrows mark a front end that the surrounding "
                 "pipeline costs completion", fontsize=11.5)
    fig.tight_layout(rect=(0, .05, 1, 1))
    _save(fig, out, "gen_vo_vs_slam")


def gen_rescued_table(cc, out):
    """Cells that had zero successes at 14 GB and at least one success at 28 GB.
    The most important table in the study: a cell recorded LOST at 14 GB that
    completes cleanly at 28 GB was never an algorithmic failure -- memory
    starvation was mislabelled as tracking loss."""
    src, dst = RUNGS[1][0], RUNGS[1][1]
    st, tt = pivot(cc, "n_success"), pivot(cc, "n_total")
    lo, om, to = pivot(cc, "n_lost"), pivot(cc, "n_oom"), pivot(cc, "n_timeout")
    ram, comp = pivot(cc, C_RAM), pivot(cc, C_COMPL)
    if src not in st or dst not in st:
        return
    rows = []
    for k in st.index:
        if pd.isna(st.loc[k, src]) or pd.isna(st.loc[k, dst]):
            continue
        if st.loc[k, src] == 0 and st.loc[k, dst] >= 1:
            l = int(lo.loc[k, src]) if pd.notna(lo.loc[k, src]) else 0
            o = int(om.loc[k, src]) if pd.notna(om.loc[k, src]) else 0
            t = int(to.loc[k, src]) if pd.notna(to.loc[k, src]) else 0
            label = max((("lost", l), ("oom", o), ("timeout", t)),
                        key=lambda p: p[1])[0]
            rows.append({"phase": k[0], "variant": k[1], "seq": k[2],
                         "was": label, "n_lost": l, "n_oom": o, "n_to": t,
                         "before": f"{int(st.loc[k, src])}/{int(tt.loc[k, src])}",
                         "after": f"{int(st.loc[k, dst])}/{int(tt.loc[k, dst])}",
                         "compl_before": comp.loc[k, src],
                         "ram_before": ram.loc[k, src],
                         "ram_after": ram.loc[k, dst]})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["was", "variant", "seq"])
    n_runs = 0
    n_full_recover = 0
    for r in rows:
        a, b = r["before"].split("/")[0], r["after"].split("/")[0]
        n_runs += int(b) - int(a)
        if r["after"].split("/")[0] == r["after"].split("/")[1]:
            n_full_recover += 1
    run_lost = int(df["n_lost"].sum())
    run_oom = int(df["n_oom"].sum())
    run_to = int(df["n_to"].sum())
    run_total = run_lost + run_oom + run_to
    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{The {len(df)} cells that went from zero successful "
             rf"runs at 14\,GB to at least one at 28\,GB, accounting for "
             rf"{n_runs} recovered runs; {n_full_recover} recovered every "
             r"repetition. The \emph{was} column gives the modal outcome "
             r"label across the three failed runs at 14\,GB. At run level, "
             rf"{run_lost} of the {run_total} failures here were recorded as "
             rf"\textsc{{lost}}, {run_oom} as \textsc{{oom}} and {run_to} as "
             r"\textsc{timeout}: roughly nine in ten carried an algorithmic "
             r"label for what the 28\,GB column shows to be memory pressure. "
             r"Not one line of code differs between the two columns.}",
             r"\label{tab:gen-rescued}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
             r"Variant & Seq. & was & 14\,GB & 28\,GB & peak GB @14 & "
             r"peak GB @28 \\", r"\midrule"]
    for r in df.itertuples():
        nm = disp(r.variant.split("_", 1)[-1] if "_" in r.variant else r.variant)
        lines.append(f"{nm} & {r.seq} & \\textsc{{{r.was}}} & {r.before} & "
                     f"{r.after} & {_n(r.ram_before, 1)} & "
                     f"{_n(r.ram_after, 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "gen_rescued_table", lines, df.round(3))


def gen_mode_summary(csv, out):
    """VO against SLAM at both budgets, as four numbers. VO barely notices the
    memory budget (little history accumulated); full SLAM notices it a lot --
    the mechanism behind the rescued cells, at campaign level."""
    rows = []
    for mode in ("VO", "SLAM"):
        try:
            d = load_all(csv, mode)
        except SystemExit:
            continue
        for cond in ("R9_12t_14", REFERENCE):
            x = one(d, cond)
            if x.empty:
                continue
            tot = max(x.n_total.sum(), 1)
            rows.append({"mode": mode, "condition": COND_LABEL.get(cond, cond),
                         "runs": int(x.n_total.sum()),
                         "succ": 100 * x.n_success.sum() / tot,
                         "lost": int(x.n_lost.sum()), "oom": int(x.n_oom.sum())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Completion with and without loop closure, at both "
             r"memory budgets. Visual odometry accumulates almost no history "
             r"and barely notices the extra memory; full SLAM does almost "
             r"nothing else.}",
             r"\label{tab:gen-modes}", r"\small",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Mode & Condition & runs & completed \% & lost & OOM \\",
             r"\midrule"]
    cur = None
    for r in df.itertuples():
        m = r.mode if r.mode != cur else ""
        if m and cur is not None:
            lines.append(r"\midrule")
        cur = r.mode
        lines.append(f"{m} & {r.condition} & {r.runs} & {r.succ:.1f} & "
                     f"{r.lost} & {r.oom} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "gen_mode_table", lines, df.round(2))


CAMERA_HZ = 10.0

BUDGET_GB = {"i7_12t_14": 14.0, "R9_12t_14": 14.0,
             "R9_12t_28": 28.0, "R9_32t_28": 28.0}

PHASE_MARKER = {"feature_extraction": "o", "loop_closure": "s",
                "backend": "P", "combined": "D", "depth_estimation": "^",
                "baseline": "*"}
PHASE_LABEL = dict(PHASES)


def per_config(d: pd.DataFrame, condition: str | None = None) -> pd.DataFrame:
    """One row per configuration, sequences folded together (unlike
    phase_rows(), which keeps per-sequence columns for the software tables).
    ATE is taken only over cells that completed something -- never invented
    as zero or max for cells that didn't."""
    x = d if condition is None else d[d["condition"] == condition]
    if x.empty:
        return x.copy()
    g = x.groupby(["variant", "short_name", "family", "exp_type"],
                  dropna=False).agg(
        n_total=("n_total", "sum"), n_success=("n_success", "sum"),
        n_lost=("n_lost", "sum"), n_timeout=("n_timeout", "sum"),
        n_oom=("n_oom", "sum"),
        fps=(C_FPS, "median"), cpu=(C_CPU, "mean"), gpu=(C_GPU, "mean"),
        ram=(C_RAM, "max"),
    ).reset_index()
    fin = x[x["n_success"] > 0]
    g["ate"] = g["variant"].map(
        fin.groupby("variant")[C_ATE_MED].median() if not fin.empty
        else pd.Series(dtype=float))
    g["compl"] = g["n_success"] / g["n_total"].replace(0, np.nan) * 100
    g["is_base"] = g["variant"] == BASELINE
    g["is_ref"] = False
    return g


def hw_fig_fps_by_condition(cc, out):
    """Throughput per configuration at every condition, against the camera.
    Absolute fps (not a speedup ratio) with the sensor's 10 Hz line drawn on
    every panel: doubling 4 fps to 8 fps is still a pipeline falling behind,
    which a ratio would hide as a win. One panel per condition, shared row
    order (from the reference condition) so a configuration tracks across
    the ladder."""
    conds = [k for k in LADDER if k in set(cc.condition)]
    if not conds:
        return
    anchor = REFERENCE if REFERENCE in conds else conds[-1]
    ordered = order_rows(per_config(cc, anchor), within="fps", ascending=False)
    if ordered.empty:
        return
    order = list(ordered["variant"])
    fam = dict(zip(ordered["variant"], ordered["family"]))
    name = dict(zip(ordered["variant"], ordered["short_name"]))
    base = set(ordered.loc[ordered["is_base"], "variant"])

    fig, axes = plt.subplots(1, len(conds), sharey=True, sharex=True,
                             squeeze=False,
                             figsize=(3.3 * len(conds) + 1.8,
                                      0.25 * len(order) + 3.2))
    y = np.arange(len(order))
    xmax = CAMERA_HZ * 1.2
    for ax, cond in zip(axes[0], conds):
        g = per_config(cc, cond).set_index("variant").reindex(order)
        v = pd.to_numeric(g["fps"], errors="coerce").values.astype(float)
        if np.isfinite(v).any():
            xmax = max(xmax, float(np.nanmax(v)))
        ax.barh(y, np.nan_to_num(v), height=.68, edgecolor="white",
                linewidth=.5,
                color=[FAM_COLOR.get(fam.get(k), "#999") for k in order])
        ax.axvline(CAMERA_HZ, color="#c0392b", ls="--", lw=1.5, zorder=5)
        ok = int(np.nansum(np.where(np.isfinite(v), v, 0) >= CAMERA_HZ))
        have = int(np.isfinite(v).sum())
        ax.set_title(f"{COND_LABEL.get(cond, cond)}\n"
                     f"{ok} of {have} clear {CAMERA_HZ:.0f}\u2009Hz",
                     fontsize=9.5)
        ax.set_xlabel("throughput (fps)")
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)

    ax0 = axes[0][0]
    ax0.set_yticks(y)
    ax0.set_yticklabels([disp(name[k]) for k in order], fontsize=7)
    for lbl, k in zip(ax0.get_yticklabels(), order):
        if k in base:
            lbl.set_fontweight("bold")
    ax0.invert_yaxis()
    ax0.set_xlim(0, xmax * 1.10)

    handles = [Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
               for f in FAM_ORDER if f in set(ordered.family.dropna())]
    handles.append(Line2D([0], [0], color="#c0392b", ls="--", lw=1.5,
                          label=f"camera rate ({CAMERA_HZ:.0f}\u2009Hz)"))
    fig.legend(handles=handles, ncol=len(handles), loc="lower center",
               frameon=False, fontsize=9)
    fig.suptitle("Throughput against the camera\n"
                 "same row order in every panel; the dashed line is the rate "
                 "the sensor delivers frames at", fontsize=11.5)
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, .45 / h, 1, 1))
    _save(fig, out, "hw_fps_by_condition")


def hw_fig_machine_profile(ref, out):
    """Where each configuration sits in the CPU/GPU plane, one point each.
    Unlike the cost tables' single expensiveness scale, plotting both
    processors as axes shows families occupying regions rather than a line.
    Colour = family, shape = pipeline block, so combined pairs (loading both
    processors) stand out without a second figure."""
    g = order_rows(per_config(ref))
    g = g[g["cpu"].notna() & g["gpu"].notna()]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(10.6, 7.4))

    ax.axvline(g["cpu"].median(), color="#ccc", ls=":", lw=1.2, zorder=0)
    ax.axhline(g["gpu"].median(), color="#ccc", ls=":", lw=1.2, zorder=0)
    ax.annotate("medians", (g["cpu"].median(), 1.0),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(4, -11), fontsize=7.5, color="#bbb")

    for i, r in enumerate(g.itertuples()):
        combined = r.exp_type == "combined"
        mk = "*" if r.is_base else PHASE_MARKER.get(r.exp_type, "o")
        col = FAM_COLOR.get(r.family, "#999")
        ax.scatter(r.cpu, r.gpu, marker=mk,
                   s=250 if r.is_base else (105 if combined else 72),
                   color=col, zorder=5 if r.is_base else 4, alpha=.92,
                   edgecolor="#222" if combined else "white",
                   linewidth=1.2 if combined else .9)
        ax.annotate(disp(r.short_name), (r.cpu, r.gpu),
                    textcoords="offset points",
                    xytext=(7, 3) if i % 2 == 0 else (7, -9), fontsize=6.2,
                    color="#333", alpha=.9)

    ax.set_xlabel("mean CPU load (%)")
    ax.set_ylabel("mean GPU load (%)")
    ax.margins(.12)
    for tx, ty, t, ha, va in ((.99, .02, "processor-bound", "right", "bottom"),
                              (.01, .98, "GPU-bound", "left", "top"),
                              (.99, .98, "both at once", "right", "top")):
        ax.text(tx, ty, t, transform=ax.transAxes, ha=ha, va=va, fontsize=9,
                color="0.6", style="italic")

    fams = [f for f in FAM_ORDER if f in set(g.family.dropna())]
    leg1 = fig.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
                               for f in fams]
                      + [Line2D([0], [0], marker="*", ls="", ms=13,
                                color=FAM_COLOR["classical"],
                                label="ORB2 (baseline)")],
                      loc="lower center", bbox_to_anchor=(.27, -.01),
                      frameon=False, fontsize=8.5, ncol=2,
                      title="colour: architecture family")
    leg1.get_title().set_fontsize(8.5)
    blocks = [p for p, _ in PHASES if p in set(g.exp_type) and p != "baseline"]
    if blocks:
        leg2 = fig.legend(handles=[Line2D([0], [0],
                                          marker=PHASE_MARKER.get(p, "o"),
                                          ls="", ms=7, color="#555",
                                          markeredgecolor="#222"
                                          if p == "combined" else "white",
                                          label=PHASE_LABEL.get(p, p))
                                   for p in blocks],
                          loc="lower center", bbox_to_anchor=(.73, -.01),
                          frameon=False, fontsize=8.5, ncol=3,
                          title="shape: pipeline block")
        leg2.get_title().set_fontsize(8.5)
    ax.grid(alpha=.20, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("How each family uses the machine\n"
                 "one point per configuration at the reference condition; the "
                 "families occupy regions, not positions on a line",
                 fontsize=11)
    fig.tight_layout(rect=(0, .10, 1, 1))
    _save(fig, out, "hw_machine_profile")


def gen_fig_accuracy_vs_strain(cc, out):
    """Strain falls, accuracy does not: one row per configuration (avoids the
    overplotting of a scatter of forty tracks). Left panel: peak RAM as % of
    each condition's own budget, hollow-to-filled marker showing the drop.
    Right panel: ATE ratio on a log axis, where 1.0 means accuracy didn't
    move -- rows sorted by strain drop so the right panel reads as a
    vertical line at 1.0, the whole finding."""
    conds = [k for k in LADDER if k in set(cc.condition)]
    if len(conds) < 2:
        return
    lo, hi = conds[0], conds[-1]
    per_lo = per_config(cc, lo).set_index("variant")
    per_hi = per_config(cc, hi).set_index("variant")
    b_lo, b_hi = BUDGET_GB.get(lo), BUDGET_GB.get(hi)
    if not b_lo or not b_hi:
        return

    rows = []
    for v in per_hi.index:
        if v not in per_lo.index:
            continue
        a, b = per_lo.loc[v], per_hi.loc[v]
        if pd.isna(a["ram"]) or pd.isna(b["ram"]):
            continue
        s_lo, s_hi = a["ram"] / b_lo * 100, b["ram"] / b_hi * 100
        ratio = (b["ate"] / a["ate"]
                 if pd.notna(a["ate"]) and pd.notna(b["ate"]) and a["ate"] > 0
                 else np.nan)
        rows.append({"variant": v, "short": b["short_name"],
                     "family": b["family"], "is_base": bool(b["is_base"]),
                     "s_lo": s_lo, "s_hi": s_hi, "drop": s_lo - s_hi,
                     "ratio": ratio,
                     "rescued": pd.isna(a["ate"]) and pd.notna(b["ate"])})
    if not rows:
        return
    _RANK = {"classical": 1, "CNN": 2, "transformer": 3}
    df = pd.DataFrame(rows)
    df["rank"] = [0 if b else _RANK.get(f, 9)
                  for b, f in zip(df["is_base"], df["family"])]
    df = df.sort_values(["rank", "drop"], ascending=[False, True])
    n = len(df)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.8, max(5.0, .26 * n + 2.0)),
        gridspec_kw={"width_ratios": [2.5, 1]}, sharey=True)
    y = np.arange(n)

    for yi, r in zip(y, df.itertuples()):
        col = FAM_COLOR.get(r.family, "#999")
        axL.plot([r.s_hi, r.s_lo], [yi, yi], color=col,
                 lw=2.4 if r.is_base else 1.4, alpha=.9 if r.is_base else .55,
                 zorder=2, solid_capstyle="round")
        axL.scatter(r.s_lo, yi, s=34, facecolor="white", edgecolor=col,
                    linewidth=1.3, zorder=4)
        axL.scatter(r.s_hi, yi, s=34, color=col, zorder=4)

    axL.axvline(100, color="#7a2318", lw=1.2, ls="--", zorder=1)
    axL.text(100, -.85, " budget ceiling", color="#7a2318", fontsize=8.5,
             va="bottom", ha="left", style="italic", zorder=5)
    axL.set_yticks(y)
    axL.set_yticklabels([disp(s) + ("  *" if r else "")
                         for s, r in zip(df["short"], df["rescued"])],
                        fontsize=7.4)
    for lbl, b in zip(axL.get_yticklabels(), df["is_base"]):
        if b:
            lbl.set_fontweight("bold")
    axL.set_ylim(-1, n)
    axL.set_xlabel("peak RAM as % of that condition's own budget")
    axL.set_title(f"strain falls:  {COND_LABEL.get(lo, lo)} (hollow)  ->  "
                  f"{COND_LABEL.get(hi, hi)} (filled)", fontsize=10)
    axL.grid(axis="x", alpha=.20)
    axL.set_axisbelow(True)

    got = df.dropna(subset=["ratio"])
    for yi, r in zip(y, df.itertuples()):
        if pd.isna(r.ratio):
            continue
        col = FAM_COLOR.get(r.family, "#999")
        axR.scatter(r.ratio, yi, s=30, color=col, zorder=4,
                    alpha=.95 if r.is_base else .75)
    axR.axvline(1.0, color="#333", lw=1.4, zorder=2)
    axR.set_xscale("log")
    axR.set_xlabel("ATE ratio,  reference / starved")
    axR.set_title("accuracy does not", fontsize=10)
    axR.grid(axis="x", alpha=.20)
    axR.set_axisbelow(True)
    if not got.empty:
        med = float(got["ratio"].median())
        axR.text(.5, .015,
                 f"median {med:.2f}x  ({len(got)} of {n} comparable)",
                 transform=axR.transAxes, ha="center", va="bottom",
                 fontsize=8.5, color="0.35",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor="none", alpha=.85))

    fams = [f for f in FAM_ORDER if f in set(df.family.dropna())]
    fig.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
                        for f in fams],
               ncol=3, loc="lower center", frameon=False, fontsize=9)

    fig.suptitle("What a bigger machine changed, and what it did not\n"
                 "rows grouped baseline / classical / CNN / transformer;  *  "
                 "configuration that produced no answer at all until the "
                 "budget was raised", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    _save(fig, out, "gen_accuracy_vs_strain")


def gen_fig_label_flip(cc, out):
    """What the 14 GB labels said, against what the 28 GB machine proves.
    Companion to Table~ref{tab:gen-rescued}: left bar is outcomes as recorded,
    right bar is the same runs re-labelled by whether they complete once the
    budget is raised. Re-attribution charges recovered runs to OOM cells
    first, deliberately understating rather than inflating the mislabelling."""
    src, dst = RUNGS[1][0], RUNGS[1][1]
    st, tt = pivot(cc, "n_success"), pivot(cc, "n_total")
    lo, to, om = pivot(cc, "n_lost"), pivot(cc, "n_timeout"), pivot(cc, "n_oom")
    if src not in st.columns or dst not in st.columns:
        return

    def _i(frame, key, cond):
        v = frame.loc[key, cond] if cond in frame.columns else np.nan
        return 0 if pd.isna(v) else int(v)

    rec = {"success": 0, "lost": 0, "timeout": 0, "oom": 0}
    prv = {"success": 0, "lost": 0, "lost_budget": 0,
           "timeout": 0, "timeout_budget": 0, "oom": 0}
    for k in st.index:
        if pd.isna(st.loc[k, src]) or pd.isna(st.loc[k, dst]):
            continue
        s0 = _i(st, k, src)
        l0, t0, o0 = _i(lo, k, src), _i(to, k, src), _i(om, k, src)
        rec["success"] += s0; rec["lost"] += l0
        rec["timeout"] += t0; rec["oom"] += o0

        recovered = max(0, _i(st, k, dst) - s0)
        r_o = min(recovered, o0); rem = recovered - r_o
        r_l = min(rem, l0);       rem -= r_l
        r_t = min(rem, t0)
        prv["success"] += s0
        prv["oom"] += o0
        prv["lost"] += l0 - r_l
        prv["lost_budget"] += r_l
        prv["timeout"] += t0 - r_t
        prv["timeout_budget"] += r_t

    total = max(sum(rec.values()), 1)
    HATCH = "//"
    left = [("success", rec["success"], STATUS_COLOR["success"], None,
             "SUCCESS"),
            ("lost", rec["lost"], STATUS_COLOR["lost"], None, "LOST"),
            ("timeout", rec["timeout"], STATUS_COLOR["timeout"], None,
             "TIMEOUT"),
            ("oom", rec["oom"], STATUS_COLOR["oom"], None, "OOM")]
    right = [("success", prv["success"], STATUS_COLOR["success"], None, ""),
             ("lost", prv["lost"], STATUS_COLOR["lost"], None, ""),
             ("lost_budget", prv["lost_budget"], STATUS_COLOR["oom"], HATCH,
              "LOST, but the budget"),
             ("timeout", prv["timeout"], STATUS_COLOR["timeout"], None, ""),
             ("timeout_budget", prv["timeout_budget"], STATUS_COLOR["oom"],
              HATCH, ""),
             ("oom", prv["oom"], STATUS_COLOR["oom"], None, "")]

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    mid = {}
    for x, stack in ((0, left), (1, right)):
        bottom = 0.0
        for key, n, col, hatch, _lbl in stack:
            pct = n / total * 100
            if pct <= 0:
                continue
            ax.bar(x, pct, bottom=bottom, width=.52, color=col, hatch=hatch,
                   edgecolor="white", linewidth=1.0, zorder=3)
            if pct >= 3.2:
                ax.text(x, bottom + pct / 2, f"{pct:.0f}%  ({n})",
                        ha="center", va="center", fontsize=8.5, color="white",
                        fontweight="bold", zorder=4)
            mid[(x, key)] = bottom + pct / 2
            bottom += pct

    if (0, "lost") in mid and (1, "lost_budget") in mid:
        ax.annotate("", xy=(1 - .27, mid[(1, "lost_budget")]),
                    xytext=(0 + .27, mid[(0, "lost")]),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6,
                                    alpha=.85), zorder=5)
    ratio = (prv["lost_budget"] / rec["oom"]) if rec["oom"] else np.nan
    note = (f"{prv['lost_budget']} runs filed under the algorithm's name "
            f"complete\nunchanged at 28 GB. {rec['oom']} died of memory "
            f"honestly."
            + (f"\nA ratio of {ratio:.1f} to 1." if pd.notna(ratio) else ""))
    ax.text(.5, 104, note, ha="center", va="bottom", fontsize=9,
            color="#7b241c")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["as the 14 GB machine\nrecorded it",
                        "as the 28 GB machine\nproves it was"], fontsize=10)
    ax.set_xlim(-.55, 1.55)
    ax.set_ylim(0, 124)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("share of the 14 GB runs (%)")
    handles = [Patch(facecolor=STATUS_COLOR[k], label=k.upper())
               for k in ("success", "lost", "timeout", "oom") if rec[k] > 0]
    if prv["lost_budget"] or prv["timeout_budget"]:
        handles.append(Patch(facecolor=STATUS_COLOR["oom"], hatch=HATCH,
                             edgecolor="white",
                             label="recorded LOST, proven to be the budget"))
    ax.legend(handles=handles, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.10), frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.set_title("What the labels said against what they meant\n"
                 f"the same {total} runs, judged twice", fontsize=11.5)
    fig.tight_layout(rect=(0, .04, 1, 1))
    _save(fig, out, "gen_label_flip")


DS_TURN_DEG_S = 20.0
DS_REVISIT_M  = 15.0
DS_REVISIT_S  = 20.0
DS_CONDS      = (REFERENCE, CHECK)

_GT_CACHE: dict = {}
_GT_WARNED = [False]


def _gt(test_id: str):
    """Ground-truth Trajectory for one sequence, cached, or None."""
    if test_id in _GT_CACHE:
        return _GT_CACHE[test_id]
    traj = None
    try:
        import sys
        script_dir = Path(__file__).resolve().parent
        root = script_dir.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src_pc.utils.paths import PROCESSED_POSES
        from src_pc.utils.metrics_helper_functions import load_tum_trajectory
        files = sorted(Path(PROCESSED_POSES / test_id).glob("*_tum.txt"))
        if files:
            traj = load_tum_trajectory(files[0])
    except Exception as e:
        if not _GT_WARNED[0]:
            print(f"   [!] ground-truth poses unreachable ({e}); "
                  f"dataset figures skipped")
            _GT_WARNED[0] = True
    _GT_CACHE[test_id] = traj
    return traj


def _have_gt() -> list[str]:
    return [s for s in SEQS if _gt(s) is not None]


def _quat_R(q):
    """[qx,qy,qz,qw] -> 3x3. Local copy so this file does not depend on a
    private name in metrics_helper_functions."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _plane_axes(xyz):
    """(a, b, vertical) column indices. Vertical = axis of smallest variance
    (found, not hard-coded, since camera convention varies by dataset)."""
    v = xyz.var(axis=0)
    vert = int(np.argmin(v))
    a, b = [i for i in range(3) if i != vert]
    return a, b, vert


def _route(test_id: str) -> dict | None:
    """Everything Chapter 4 asks of a route, computed once per sequence.
    yaw_rate comes from the rotation, not consecutive positions -- a vehicle
    reversing through a two-point turn keeps its heading while its
    displacement flips, which a position-based heading would get wrong."""
    traj = _gt(test_id)
    if traj is None:
        return None
    xyz, quat, t = traj.xyz, traj.quat, traj.t
    n = len(xyz)
    a, b, vert = _plane_axes(xyz)
    e_vert = np.zeros(3); e_vert[vert] = 1.0

    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.1
    speed = np.concatenate([[0.0], step / dt])

    R = np.array([_quat_R(q) for q in quat])
    yaw = np.zeros(n)
    heading = np.zeros(n)
    for i in range(n):
        fwd = R[i] @ np.array([0.0, 0.0, 1.0])
        heading[i] = np.degrees(np.arctan2(fwd[b], fwd[a]))
        if i == 0:
            continue
        Rr = R[i - 1].T @ R[i]
        cos = np.clip((np.trace(Rr) - 1.0) / 2.0, -1.0, 1.0)
        ang = np.arccos(cos)
        if ang < 1e-9:
            continue
        ax = np.array([Rr[2, 1] - Rr[1, 2],
                       Rr[0, 2] - Rr[2, 0],
                       Rr[1, 0] - Rr[0, 1]]) / (2 * np.sin(ang))
        yaw[i] = np.degrees(ang * float((R[i - 1] @ ax) @ e_vert)) / dt[i - 1]

    turns, i = [], 1
    hot = np.abs(yaw) > DS_TURN_DEG_S
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        swept = float(np.sum(np.abs(yaw[i:j]) * np.median(dt)))
        if swept > 45.0:
            turns.append({"start": i, "end": j,
                          "center": int((i + j) // 2), "swept": swept,
                          "peak": float(np.max(np.abs(yaw[i:j])))})
        i = j

    return {"test_id": test_id, "xyz": xyz, "t": t, "n": n,
            "a": a, "b": b, "yaw": yaw, "speed": speed, "heading": heading,
            "path_m": float(step.sum()), "dur_s": float(t[-1] - t[0]),
            "turns": turns}


def _revisits(rt: dict) -> list[dict]:
    """Pose pairs that are the same place at different times (greedy, coarse:
    nearest later pose within DS_REVISIT_M and DS_REVISIT_S). What matters
    downstream is the HEADING DIFFERENCE distribution, not the exact count."""
    xyz, t, hd = rt["xyz"], rt["t"], rt["heading"]
    n = rt["n"]
    out, used = [], np.zeros(n, dtype=bool)
    stride = max(1, n // 1200)
    for i in range(0, n, stride):
        if used[i]:
            continue
        far = np.where(t - t[i] > DS_REVISIT_S)[0]
        if far.size == 0:
            continue
        d = np.linalg.norm(xyz[far] - xyz[i], axis=1)
        k = int(np.argmin(d))
        if d[k] > DS_REVISIT_M:
            continue
        j = int(far[k])
        diff = abs((hd[j] - hd[i] + 180.0) % 360.0 - 180.0)
        out.append({"i": i, "j": j, "d": float(d[k]), "heading_diff": diff})
        used[max(0, i - stride):i + stride + 1] = True
        used[max(0, j - stride):j + stride + 1] = True
    return out


def _death_points(d_all: pd.DataFrame,
                  conditions=DS_CONDS) -> dict[str, list[tuple[float, str, int]]]:
    """test_id -> [(frame, status, weight)], recovered from the CSV.

    The CSV has no per-run stopping frame (only cell-level completion_pct), so
    this backs it out: for an all-failed cell the median completion IS where
    it stopped; for a mixed cell, successes are subtracted out of the mean
    (clamped -- killed-process metrics can be slightly off). Weight is the
    failed-run count behind the point, for the density map."""
    out: dict[str, list[tuple[float, str, int]]] = {s: [] for s in SEQS}
    d = d_all[d_all["condition"].isin(list(conditions))]
    for r in d.itertuples():
        rt = _route(r.test_id) if r.test_id in out else None
        if rt is None:
            continue
        tot = int(r.n_total or 0)
        suc = int(r.n_success or 0)
        fail = tot - suc
        if fail <= 0:
            continue
        nm = getattr(r, "n_metric_runs", np.nan)
        nm = tot if pd.isna(nm) or nm <= 0 else float(nm)
        if suc == 0:
            pct = getattr(r, "completion_pct_median", np.nan)
        else:
            mean = getattr(r, "completion_pct_mean", np.nan)
            pct = ((mean * nm - 100.0 * suc) / fail
                   if pd.notna(mean) and fail > 0 else np.nan)
        if pd.isna(pct):
            continue
        pct = float(np.clip(pct, 0.0, 100.0))
        counts = {"lost": int(r.n_lost or 0), "oom": int(r.n_oom or 0),
                  "timeout": int(r.n_timeout or 0)}
        status = max(counts, key=counts.get) if max(counts.values()) else "lost"
        out[r.test_id].append((pct / 100.0 * (rt["n"] - 1), status, fail))
    return out


def _clusters(points, n_frames: int, bin_frames: int = 100, top: int = 3):
    """The frames where deaths pile up. Same idea as aggregate_results'
    failure map, weighted by how many runs each point stands for."""
    if not points:
        return []
    fr = np.array([p[0] for p in points], dtype=float)
    w = np.array([p[2] for p in points], dtype=float)
    nb = max(3, int(n_frames / bin_frames))
    counts, edges = np.histogram(fr, bins=nb, range=(0, n_frames), weights=w)
    order = np.argsort(counts)[::-1]
    out = []
    for bi in order[:top]:
        if counts[bi] < 2:
            continue
        c = float((edges[bi] + edges[bi + 1]) / 2)
        out.append({"frame": c, "pct": c / max(n_frames - 1, 1) * 100,
                    "deaths": int(round(counts[bi])),
                    "lo": float(edges[bi]), "hi": float(edges[bi + 1])})
    return out


def _topdown(ax, rt, lw=1.4, color="0.78"):
    a, b, xyz = rt["a"], rt["b"], rt["xyz"]
    ax.plot(xyz[:, a], xyz[:, b], color=color, lw=lw, zorder=1, clip_on=False)
    ax.scatter(xyz[0, a], xyz[0, b], c="#2ca02c", s=48, zorder=6,
               edgecolors="white", linewidths=.8, clip_on=False)
    ax.set_aspect("equal", "datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def ds_fig_trajectories(out):
    """The four routes at a common metric scale, sharp turns marked -- shows
    the difficulty ladder from Table 4.1 as geometry."""
    seqs = _have_gt()
    if not seqs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    span = 0
    routes = {s: _route(s) for s in seqs}
    for rt in routes.values():
        xy = rt["xyz"][:, [rt["a"], rt["b"]]]
        span = max(span, float(np.ptp(xy, axis=0).max()))
    for ax, s in zip(axes.ravel(), seqs):
        rt = routes[s]
        _topdown(ax, rt)
        for tn in rt["turns"]:
            c = min(tn["center"], rt["n"] - 1)
            ax.scatter(rt["xyz"][c, rt["a"]], rt["xyz"][c, rt["b"]],
                       marker="^", c="#c0392b", s=52, zorder=5,
                       edgecolors="white", linewidths=.6, clip_on=False)
        xy = rt["xyz"][:, [rt["a"], rt["b"]]]
        mid = (xy.min(axis=0) + xy.max(axis=0)) / 2
        ax.set_xlim(mid[0] - span / 2 * 1.15, mid[0] + span / 2 * 1.15)
        ax.set_ylim(mid[1] - span / 2 * 1.15, mid[1] + span / 2 * 1.15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{s}   {rt['n']} frames, {rt['path_m']:.0f} m, "
                     f"{len(rt['turns'])} sharp turns", fontsize=10.5)
    for ax in axes.ravel()[len(seqs):]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", linestyle="none", color="#2ca02c",
                      label="start"),
               Line2D([0], [0], marker="^", linestyle="none", color="#c0392b",
                      label=f"manoeuvre sweeping >45 deg at >{DS_TURN_DEG_S:.0f} deg/s")]
    fig.legend(handles=handles, ncol=2, loc="lower center", frameon=False,
               fontsize=9)
    fig.suptitle("The four routes, common scale", fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, 1))
    _save(fig, out, "ds_trajectories")


def ds_fig_yaw_rate(out):
    """Difficulty is concentrated, not distributed: most of each drive sits
    near zero, with spikes at manoeuvres where inter-frame overlap collapses
    (the load-bearing claim in Section 4.4)."""
    seqs = _have_gt()
    if not seqs:
        return
    fig, axes = plt.subplots(len(seqs), 1, figsize=(10.0, 2.05 * len(seqs)),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, seqs):
        rt = _route(s)
        y = rt["yaw"]
        ax.axhspan(-DS_TURN_DEG_S, DS_TURN_DEG_S, color="0.92", zorder=0)
        ax.plot(np.arange(rt["n"]), y, lw=.8, color="#34495e", zorder=2)
        for tn in rt["turns"]:
            ax.axvspan(tn["start"], tn["end"], color="#c0392b", alpha=.16,
                       zorder=1)
        ax.axhline(0, color="0.6", lw=.6)
        ax.set_ylabel("deg/s", fontsize=9)
        ax.set_xlim(0, rt["n"])
        pk = np.max(np.abs(y)) if rt["n"] else 0
        ax.set_title(f"{s}   peak |yaw| {pk:.0f} deg/s, "
                     f"{len(rt['turns'])} manoeuvres", fontsize=10,
                     loc="left")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Yaw rate along each route: the spikes are where overlap "
                 "collapses", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _save(fig, out, "ds_yaw_rate")


def ds_fig_failure_map(d_all, out):
    """Where runs die, as a density along the route -- distinguishes
    accumulated drift (spread, biased to the end) from manoeuvre failures
    (clustered on the ds_yaw_rate spikes)."""
    seqs = _have_gt()
    if not seqs:
        return
    pts = _death_points(d_all)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    sm = None
    for ax, s in zip(axes.ravel(), seqs):
        rt = _route(s)
        _topdown(ax, rt, color="0.85")
        p = pts.get(s, [])
        n_deaths = sum(w for _, _, w in p)
        if p:
            nb = max(6, int(rt["n"] / 60))
            fr = np.array([x[0] for x in p]); w = np.array([x[2] for x in p],
                                                           dtype=float)
            cnt, edges = np.histogram(fr, bins=nb, range=(0, rt["n"] - 1),
                                      weights=w)
            centers = ((edges[:-1] + edges[1:]) / 2).astype(int)
            live = cnt > 0
            sm = ax.scatter(rt["xyz"][centers[live], rt["a"]],
                            rt["xyz"][centers[live], rt["b"]],
                            c=cnt[live], cmap="inferno_r",
                            s=44, zorder=4,
                            edgecolors="none", alpha=.92, clip_on=False)
        for tn in rt["turns"]:
            c = min(tn["center"], rt["n"] - 1)
            ax.scatter(rt["xyz"][c, rt["a"]], rt["xyz"][c, rt["b"]],
                       marker="^", facecolors="none", edgecolors="#2c3e50",
                       s=70, linewidths=1.0, zorder=5, clip_on=False)
        ax.set_title(f"{s}   {int(n_deaths)} runs ended here", fontsize=10.5)
    for ax in axes.ravel()[len(seqs):]:
        ax.axis("off")
    if sm is not None:
        cb = fig.colorbar(sm, ax=axes, shrink=.6, pad=.02)
        cb.set_label("runs ending in this stretch")
    fig.suptitle("Where the runs die  (both 28 GB conditions, so nothing here "
                 "is a memory kill)\nhollow triangles are the manoeuvres of "
                 "the yaw-rate figure", fontsize=11.5)
    _save(fig, out, "ds_failure_map")


def ds_hard_sections(d_all, out):
    """The top-down overview of the worst stretches, plus a table of frame
    ranges to extract images from by hand (no script picks which frames tell
    the story)."""
    seqs = _have_gt()
    if not seqs:
        return
    pts = _death_points(d_all)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    for ax, s in zip(axes.ravel(), seqs):
        rt = _route(s)
        _topdown(ax, rt, color="0.85")
        p = pts.get(s, [])
        if p:
            ax.scatter([rt["xyz"][min(int(f), rt["n"] - 1), rt["a"]]
                        for f, _, _ in p],
                       [rt["xyz"][min(int(f), rt["n"] - 1), rt["b"]]
                        for f, _, _ in p],
                       s=[16 + 12 * w for _, _, w in p], alpha=.30,
                       c="#7f8c8d", edgecolors="none", zorder=3, clip_on=False)
        for k, cl in enumerate(_clusters(p, rt["n"]), start=1):
            ci = min(int(round(cl["frame"])), rt["n"] - 1)
            lo = max(0, int(cl["lo"]) - 15); hi = min(rt["n"], int(cl["hi"]) + 15)
            peak = float(np.max(np.abs(rt["yaw"][lo:hi]))) if hi > lo else 0.0
            spd = float(np.median(rt["speed"][lo:hi])) if hi > lo else 0.0
            ax.scatter(rt["xyz"][ci, rt["a"]], rt["xyz"][ci, rt["b"]],
                       c="#c0392b", s=150, zorder=6, edgecolors="white",
                       linewidths=1.1, marker="X", clip_on=False)
            ax.annotate(f"C{k}", (rt["xyz"][ci, rt["a"]], rt["xyz"][ci, rt["b"]]),
                        textcoords="offset points", xytext=(9, 7), fontsize=10,
                        fontweight="bold", color="#7b241c", zorder=7)
            rows.append({"seq": s, "tag": f"C{k}", "frame": ci,
                         "pct": cl["pct"], "deaths": cl["deaths"],
                         "lo": int(cl["lo"]), "hi": int(cl["hi"]),
                         "peak_yaw": peak, "speed": spd})
        ax.set_title(f"{s}", fontsize=11)
    for ax in axes.ravel()[len(seqs):]:
        ax.axis("off")
    fig.suptitle("The stretches that end runs\ngrey dots are individual "
                 "endings, X marks the worst 100-frame windows", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _save(fig, out, "ds_hard_sections")

    if not rows:
        return
    df = pd.DataFrame(rows)
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{The stretches of road that end runs. Clusters are the "
         r"100-frame windows in which most runs stopped, pooled over both "
         r"28\,GB conditions so that none of these is a memory kill. "
         r"``Frames'' is the range to extract consecutive images from for "
         r"Figure~\ref{fig:ds-hard-sections}.}",
         r"\label{tab:ds-hard-clusters}", r"\small",
         r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Seq. & & Frames & At & Runs & Peak yaw & Speed \\",
         r" & & & (\% route) & ended & (deg/s) & (m/s) \\", r"\midrule"]
    last = None
    for r in df.itertuples():
        head = r.seq.replace("test", r"\texttt{test") + "}" if r.seq != last \
            else ""
        last = r.seq
        L.append(f"{head} & {r.tag} & {r.lo}--{r.hi} & {r.pct:.0f} & "
                 f"{r.deaths} & {r.peak_yaw:.0f} & {r.speed:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "ds_hard_clusters", L, df)


def ds_fig_revisit_geometry(out):
    """Revisits, coloured by how differently the vehicle was pointing. Heading
    (not ground covered) is what decides the loop-closure result: head-on
    revisits are easy for a bag of words, large-heading ones need a
    viewpoint-invariant descriptor."""
    seqs = _have_gt()
    if not seqs:
        return
    fig, axes = plt.subplots(2, len(seqs), figsize=(3.4 * len(seqs), 7.6),
                             gridspec_kw={"height_ratios": [2.1, 1]})
    axes = np.atleast_2d(axes)
    sm = None
    for k, s in enumerate(seqs):
        rt = _route(s)
        rev = _revisits(rt)
        ax = axes[0, k]
        _topdown(ax, rt, color="0.86")
        if rev:
            hd = np.array([r["heading_diff"] for r in rev])
            idx = np.array([r["i"] for r in rev])
            sm = ax.scatter(rt["xyz"][idx, rt["a"]], rt["xyz"][idx, rt["b"]],
                            c=hd, cmap="viridis", vmin=0, vmax=180, s=42,
                            zorder=4, edgecolors="none", clip_on=False)
        ax.set_title(f"{s}   {len(rev)} revisits", fontsize=10.5)
        h = axes[1, k]
        if rev:
            h.hist([r["heading_diff"] for r in rev],
                   bins=np.arange(0, 181, 22.5), color="#34495e")
        h.set_xlim(0, 180)
        h.set_xticks([0, 90, 180])
        h.set_xlabel("heading difference (deg)", fontsize=9)
        if k == 0:
            h.set_ylabel("revisits", fontsize=9)
        h.grid(axis="y", alpha=.2); h.set_axisbelow(True)
    if sm is not None:
        cb = fig.colorbar(sm, ax=axes[0, :].tolist(), shrink=.75, pad=.02)
        cb.set_label("heading difference between passes (deg)")
    fig.suptitle(f"Revisit geometry  (same place = within {DS_REVISIT_M:.0f} m "
                 f"and {DS_REVISIT_S:.0f} s apart)", fontsize=11.5)
    _save(fig, out, "ds_revisit_geometry")


def ds_seq_profile_table(d_all, out):
    """Table 4.1: route geometry from the poses, completion from the CSV at
    the reference condition."""
    seqs = _have_gt()
    if not seqs:
        return
    ref = d_all[d_all["condition"] == REFERENCE]
    rows = []
    for s in seqs:
        rt = _route(s)
        rev = _revisits(rt)
        r = ref[ref["test_id"] == s]
        tot, suc = float(r["n_total"].sum()), float(r["n_success"].sum())
        rows.append({
            "seq": s, "frames": rt["n"], "dur": rt["dur_s"],
            "path": rt["path_m"],
            "speed": rt["path_m"] / rt["dur_s"] if rt["dur_s"] else np.nan,
            "turns": len(rt["turns"]),
            "peak_yaw": float(np.max(np.abs(rt["yaw"]))) if rt["n"] else np.nan,
            "revisits": len(rev),
            "rev_wide": sum(1 for x in rev if x["heading_diff"] > 60),
            "variants": int(r["variant"].nunique()),
            "compl": suc / tot * 100 if tot else np.nan})
    df = pd.DataFrame(rows)

    def row(label, key, fmt="{:.0f}"):
        vals = " & ".join(fmt.format(v) if pd.notna(v) else "--"
                          for v in df[key])
        return f"{label} & {vals} \\\\"

    hdr = " & ".join(r"\texttt{" + s + "}" for s in df["seq"])
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{The four sequences, characterised. Route geometry is "
         r"computed from the ground-truth poses; a manoeuvre is a contiguous "
         r"stretch above " + f"{DS_TURN_DEG_S:.0f}" + r"\,deg/s sweeping more "
         r"than $45^\circ$, and a revisit is a pose pair within "
         + f"{DS_REVISIT_M:.0f}" + r"\,m and at least "
         + f"{DS_REVISIT_S:.0f}" + r"\,s apart. Completion is the share of "
         r"full-SLAM runs that finished at the reference condition "
         r"\condMem.}",
         r"\label{tab:seq-profile}", r"\small",
         r"\begin{tabular}{@{}l" + "r" * len(df) + r"@{}}", r"\toprule",
         f" & {hdr} \\\\", r"\midrule",
         row("Frames", "frames"),
         row("Duration (s)", "dur", "{:.1f}"),
         row("Path length (m)", "path"),
         row("Mean speed (m/s)", "speed", "{:.1f}"),
         row("Manoeuvres $>45^\\circ$", "turns"),
         row("Peak yaw rate (deg/s)", "peak_yaw"),
         row("Revisited segments", "revisits"),
         row("\\ \\ of which $>60^\\circ$ apart", "rev_wide"),
         r"\midrule",
         row("Variants attempted", "variants"),
         row("Completion at \\condMem\\ (\\%)", "compl", "{:.1f}"),
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex(out, "ds_seq_profile", L, df)


def ds_fig_keyframes_vs_ram(ref, out):
    """Peak memory against map size: the map is what runs out, not the model
    -- footprint tracks route length, not network size (why the rescued set
    in Section 8.2 is all light classical front ends)."""
    need = ["num_keyframes_all_median", C_RAM]
    d = ref.dropna(subset=need)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for s in SEQS:
        p = d[d["test_id"] == s]
        if p.empty:
            continue
        ax.scatter(p["num_keyframes_all_median"], p[C_RAM], s=34, alpha=.78,
                   label=s, edgecolors="none")
    x = d["num_keyframes_all_median"].to_numpy(float)
    y = d[C_RAM].to_numpy(float)
    r = float(np.corrcoef(x, y)[0, 1])
    m, c = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + c, color="0.35", lw=1.4, ls="--", zorder=1,
            label=f"fit: r = {r:.2f}, n = {len(d)}")
    ax.set_xlabel("keyframes retained (median over the cell's runs)")
    ax.set_ylabel("peak resident memory (GB)")
    ax.set_title("Peak memory follows map size, not model size\n"
                 f"reference condition {COND_LABEL[REFERENCE]}", fontsize=11.5)
    ax.grid(alpha=.22); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, out, "ds_keyframes_vs_ram")


def res_fig_rpe_flatness(ref, out):
    """Local motion estimation does not separate the families: short-window
    RPE (frame-to-frame, before global correction) is flat across families
    while ATE on the same runs spans two orders of magnitude."""
    d = ref[ref["exp_type"].isin(["feature_extraction", "baseline"])]
    cols = ["rpe_short_rmse_succ_median", "rpe_long_rmse_succ_median",
            C_ATE_MED]
    if d.empty or d[cols].isna().all().all():
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    panels = [("rpe_short_rmse_succ_median", "RPE, 0.5 s window (m)", False),
              ("rpe_long_rmse_succ_median", "RPE, 2 s window (m)", False),
              (C_ATE_MED, "ATE RMSE (m)", True)]
    w, seqs = .26, SEQS
    for ax, (col, ylab, logy) in zip(axes, panels):
        for k, fam in enumerate(FAM_ORDER):
            vals = [d[(d.test_id == s) & (d.family == fam)][col].median()
                    for s in seqs]
            ax.bar(np.arange(len(seqs)) + (k - 1) * w, vals, width=w,
                   color=FAM_COLOR[fam], label=FAM_LONG[fam], zorder=3)
        ax.set_xticks(np.arange(len(seqs)))
        ax.set_xticklabels(seqs, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9.5)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
                        for f in FAM_ORDER],
               ncol=3, loc="lower center", frameon=False, fontsize=9)
    fig.suptitle("Frame-to-frame quality is flat across the families; "
                 "global consistency is not\n"
                 "median over completed runs, feature-extraction phase, "
                 f"{COND_LABEL[REFERENCE]}", fontsize=11.5)
    fig.tight_layout(rect=(0, .07, 1, 1))
    _save(fig, out, "res_rpe_flatness")


def res_fig_preinit_frames(ref, out):
    """Time to first pose -- a curiosity on a long sequence, a spec for a
    robot that must localise before moving. Family medians are all exactly 1
    so are stated in the subtitle rather than drawn as overlapping dashed
    lines. Rows keep the house order (baseline/Cls/CNN/Trf) but sort slowest
    first within each family block."""
    col = "pre_init_frames_all_median"
    if col not in ref.columns:
        return
    d = ref[ref["exp_type"].isin(["feature_extraction", "baseline"])]
    g = (d.groupby(["short_name", "family"], dropna=False)[col]
         .median().reset_index().dropna(subset=[col]))
    if g.empty:
        return

    g["is_base"] = g["short_name"] == BASELINE
    g = order_rows(g, within=col, ascending=False)

    n_slow = int((g[col] > 1).sum())
    n_all = len(g)

    fig, ax = plt.subplots(figsize=(8.6, max(4.5, .3 * n_all + 1.8)))
    y = np.arange(n_all)
    ax.barh(y, g[col].clip(lower=1),
            color=[FAM_COLOR.get(f, "0.6") for f in g["family"]], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([disp(s) for s in g["short_name"]], fontsize=9)
    for lbl, b in zip(ax.get_yticklabels(), g["is_base"]):
        if b:
            lbl.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.85, float(g[col].max()) * 2.2)
    ax.set_xlabel("frames before the map initialises  (log scale)")

    ax.axvline(1, color="0.35", lw=1.1, zorder=2)

    if 0 < n_slow < n_all:
        ax.text(0.98, 0.02, f"{n_slow} slower than one frame, "
                f"{n_all - n_slow} initialise on the first frame",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color="0.35")

    for yi, v in zip(y, g[col]):
        ax.text(max(v, 1) * 1.12, yi, f"{v:.0f}", va="center", fontsize=8,
                color="0.25")

    fig.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=FAM_LONG[f])
                        for f in FAM_ORDER],
               ncol=3, loc="lower center", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.22); ax.set_axisbelow(True)
    ax.set_title("Time to first pose\n"
                 "every family's median is one frame; the split is within "
                 f"the families, not between them  ({COND_LABEL[REFERENCE]})",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, .05, 1, 1))
    _save(fig, out, "res_preinit_frames")

def ds_fig_map_growth(indir, out):
    path = None
    for cand in (Path(indir) / "csv" / "map_growth.csv",
                 Path(indir) / "map_growth.csv"):
        if cand.is_file():
            path = cand
            break
    if path is None:
        print("   [!] ds_map_growth: map_growth.csv not found -- "
              "re-run aggregate_all.py to export it")
        return

    g = pd.read_csv(path)
    if g.empty:
        print("   [!] ds_map_growth: map_growth.csv is empty")
        return

    fig, (ax_kf, ax_mp) = plt.subplots(2, 1, figsize=(7, 9.2))
    seqs = [s for s in SEQS if s in set(g["test_id"])]
    for i, s in enumerate(seqs):
        d = g[g["test_id"] == s].sort_values("distance_m")
        c = plt.cm.viridis(i / max(len(seqs) - 1, 1))
        ax_kf.plot(d["distance_m"], d["num_keyframes"], color=c, lw=1.7,
                   label=s)
        ax_mp.plot(d["distance_m"], d["map_points"], color=c, lw=1.7,
                   label=s)

    for ax, ylab, title in ((ax_kf, "keyframes retained",
                             "Keyframes vs. distance"),
                            (ax_mp, "map points", "Map points vs. distance")):
        ax.set_xlabel("distance driven (m)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
    ax_kf.legend(frameon=False, fontsize=9, title="sequence")

    cond = g["condition"].iloc[0] if "condition" in g else "?"
    fig.suptitle("Map growth against distance driven  --  baseline, "
                 f"{COND_LABEL.get(cond, cond)}\n"
                 "the map grows with route length, not with scene difficulty",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 1))
    _save(fig, out, "ds_map_growth")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Results figures and tables from all_conditions.csv")
    ap.add_argument("--in", dest="indir", type=Path,
                    default=Path("results_aggregated"),
                    help="root written by aggregate_all.py; the CSV is read "
                         "from <root>/csv/ and assets are written to "
                         "<root>/figures/ and <root>/tables/")
    ap.add_argument("--only",
                    choices=["software", "hardware", "general", "dataset"],
                    default=None, help="build one section only")
    ap.add_argument("--table-csv", action="store_true",
                    help="also write the numbers behind each table to "
                         "<root>/csv/tables/")
    args = ap.parse_args()

    global FIG_DIR, TAB_DIR, TABLE_CSV
    csv = args.indir / "csv" / "all_conditions.csv"
    FIG_DIR = args.indir / "figures"
    TAB_DIR = args.indir / "tables"
    TABLE_CSV = args.table_csv
    for p in (FIG_DIR, TAB_DIR):
        p.mkdir(parents=True, exist_ok=True)
    dirs = {k: args.indir
            for k in ("software", "hardware", "general", "dataset")}

    print("=" * 70)
    print(f" results assets   reference = {REFERENCE}")
    print(f" reading  {csv}")
    print(f" figures  {FIG_DIR}")
    print(f" tables   {TAB_DIR}")
    print("=" * 70)

    d_all = load_all(csv, "SLAM")
    ref = one(d_all, REFERENCE)
    if ref.empty:
        sys.exit(f"no rows for reference condition {REFERENCE}")
    bflag = baseline_flags(ref)
    bad = [s for s, v in bflag.items() if v]
    print(f" reference cells {len(ref)}   runs {int(ref.n_total.sum())}"
          + (f"   [!] thin baseline on {', '.join(bad)}" if bad else ""))

    cc = common(d_all)
    print(f" common cells across {cc.condition.nunique()} conditions: "
          f"{len(cc) // max(cc.condition.nunique(), 1)}")

    want = lambda s: args.only in (None, s)

    if want("software"):
        print("\n SOFTWARE ->", dirs["software"])
        fe = phase_rows(ref, ["feature_extraction"])
        lc = phase_rows(ref, ["loop_closure"])
        co = phase_rows(ref, ["combined"], extra=["fe_brisk"])
        de = phase_rows(ref, ["depth_estimation"])

        sw_accuracy_table(fe, bflag, dirs["software"], "sw_fe_table",
                          "tab:sw-fe",
                          r"Feature extraction at the reference condition. "
                          r"Rows run baseline, Cls, CNN, Trf. ``Compl.'' is "
                          r"completed runs over attempts. Accuracy is the mean "
                          r"ATE over completed repetitions, in metres, with "
                          r"that count beside it; best per column in bold. A "
                          r"cell that never finished reports how far it got, a "
                          r"dash was never attempted, and $^\dagger$ marks a "
                          r"sequence the baseline did not itself complete.")
        sw_accuracy_table(lc, bflag, dirs["software"], "sw_lc_table",
                          "tab:sw-lc",
                          r"Loop closure at the reference condition, every run "
                          r"sharing the ORB2 front end so only the "
                          r"place-recognition module varies. Rows and columns "
                          r"as in Table~\ref{tab:sw-fe}.")
        sw_accuracy_table(co, bflag, dirs["software"], "sw_combined_table",
                          "tab:sw-combined",
                          r"Combined front-end and loop-closer pairs at the "
                          r"reference condition, with the plain ORB2 baseline "
                          r"and BRISK, the only front end from "
                          r"Table~\ref{tab:sw-fe} to complete every sequence "
                          r"on its own, carried in for comparison. Rows and "
                          r"columns as in Table~\ref{tab:sw-fe}, apart from "
                          r"the family column, dropped here since every row "
                          r"but the reference is a front-end/loop-closer "
                          r"pair rather than a single family.",
                          head="Configuration", show_fam=False)
        sw_depth_table(de, bflag, dirs["software"])
        sw_backend_table(d_all, dirs["software"])
        try:
            sw_fig_backend(d_all, dirs["software"])
        except Exception as e:
            print(f"   [!] sw_fig_backend: {e}")
        glance_table(FE_GLANCE, dirs["software"], "fe_glance", "tab:fe-glance",
                     r"The twenty feature extractors compared in this study, "
                     r"one line each. Rows run baseline, Cls, CNN, Trf. "
                     r"$^\star$ marks a module integrated into pySLAM for this "
                     r"study rather than shipped with it.")
        glance_table(LC_GLANCE, dirs["software"], "lc_glance", "tab:lc-glance",
                     r"The fourteen loop closers, one line each. Columns and "
                     r"row order as in Table~\ref{tab:fe-glance}. The classical "
                     r"rows reuse descriptors the tracker has already computed; "
                     r"every deep row needs its own forward pass per keyframe.")
        for fn, a in ((sw_fig_lc_by_sequence, lc), (sw_fig_combined, co),
                      (sw_fig_depth_scale, de)):
            try:
                fn(a, dirs["software"])
            except Exception as e:
                print(f"   [!] {fn.__name__}: {e}")
        for g, nm, t in ((fe, "sw_fe_robustness",
                          "Feature extraction: what happened to every run"),
                         (lc, "sw_lc_robustness",
                          "Loop closure: what happened to every run")):
            try:
                sw_fig_robustness(g, dirs["software"], nm, t)
            except Exception as e:
                print(f"   [!] {nm}: {e}")

        for fn in (res_fig_rpe_flatness, res_fig_preinit_frames):
            try:
                fn(ref, dirs["software"])
            except Exception as e:
                print(f"   [!] {fn.__name__}: {e}")

    if want("hardware"):
        print("\n HARDWARE ->", dirs["hardware"])
        hw_ladder_table(cc, dirs["hardware"])
        hw_outcome_table(cc, dirs["hardware"])
        hw_accuracy_invariance(cc, dirs["hardware"])
        for fn in (hw_fig_outcome_ladder, hw_fig_speedup, hw_fig_ram_relief,
                   hw_fig_threads, hw_fig_fps_by_condition):
            try:
                fn(cc, dirs["hardware"])
            except Exception as e:
                print(f"   [!] {fn.__name__}: {e}")
        try:
            hw_fig_machine_profile(ref, dirs["hardware"])
        except Exception as e:
            print(f"   [!] hw_fig_machine_profile: {e}")
        for exps, nm, lab, cap, head, show_fam in (
                (["feature_extraction"], "hw_cost_fe", "tab:hw-cost-fe",
                 r"Machine cost of each feature extractor at the reference "
                 r"condition, absolute with the ratio to the ORB2 baseline in "
                 r"brackets. Rows run baseline, Cls, CNN, Trf; best per column "
                 r"in bold. CPU/frame is mean CPU load divided by throughput: "
                 r"a pipeline stalled on the GPU posts a low instantaneous CPU "
                 r"per cent while doing less work, and this column removes "
                 r"that illusion.", "Method", True),
                (["loop_closure"], "hw_cost_lc", "tab:hw-cost-lc",
                 r"Machine cost of each loop closer. Rows and columns as in "
                 r"Table~\ref{tab:hw-cost-fe}.", "Method", True),
                (["combined"], "hw_cost_combined", "tab:hw-cost-combined",
                 r"Machine cost of the combined pairs. Rows and columns as in "
                 r"Table~\ref{tab:hw-cost-fe}, apart from the family column, "
                 r"dropped here since every row is a front-end/loop-closer "
                 r"pair rather than a single family.", "Configuration", False),
                (["depth_estimation"], "hw_cost_depth", "tab:hw-cost-depth",
                 r"Machine cost of each depth source. The V2 generation "
                 r"saturates the GPU and runs an order of magnitude below the "
                 r"camera rate; the V3 generation does neither. Columns as in "
                 r"Table~\ref{tab:hw-cost-fe}, apart from the family column, "
                 r"dropped here since every row is the same family "
                 r"(transformer).", "Configuration", False)):
            try:
                hw_cost_table(phase_rows(ref, exps), dirs["hardware"], nm,
                              lab, cap, head, show_fam=show_fam)
            except Exception as e:
                print(f"   [!] {nm}: {e}")

    if want("general"):
        print("\n GENERAL ->", dirs["general"])
        for fn in (gen_rescued_table,):
            try:
                fn(cc, dirs["general"])
            except Exception as e:
                print(f"   [!] {fn.__name__}: {e}")
        for fn in (gen_fig_vo_vs_slam, gen_mode_summary):
            try:
                fn(csv, dirs["general"])
            except Exception as e:
                print(f"   [!] {fn.__name__}: {e}")
        for fn in (gen_fig_accuracy_vs_strain, gen_fig_label_flip):
            try:
                fn(cc, dirs["general"])
            except Exception as e:
                print(f"   [!] {fn.__name__}: {e}")

    if want("dataset"):
        print("\n DATASET ->", dirs["dataset"])
        if not _have_gt():
            print("   [!] no ground-truth poses under PROCESSED_POSES; "
                  "run prep_project.py first")
        else:
            for fn in (ds_fig_trajectories, ds_fig_yaw_rate,
                       ds_fig_revisit_geometry):
                try:
                    fn(dirs["dataset"])
                except Exception as e:
                    print(f"   [!] {fn.__name__}: {e}")
            for fn in (ds_fig_failure_map, ds_hard_sections,
                       ds_seq_profile_table):
                try:
                    fn(d_all, dirs["dataset"])
                except Exception as e:
                    print(f"   [!] {fn.__name__}: {e}")
            try:
                ds_fig_keyframes_vs_ram(ref, dirs["dataset"])
            except Exception as e:
                print(f"   [!] ds_fig_keyframes_vs_ram: {e}")
            try:
                ds_fig_map_growth(args.indir, dirs["dataset"])
            except Exception as e:
                print(f"   [!] ds_fig_map_growth: {e}")

    if want("software"):
        try:
            build_mechanism_diagrams()
        except Exception as e:
            print(f"   [!] build_mechanism_diagrams: {e}")

    print("\n done")


def build_mechanism_diagrams():
    """FAST/BRIEF, descriptor space, and attention matching."""
    import os
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.linewidth": 0.8,
    })

    out_dir = str(FIG_DIR or "results_aggregated/figures")
    os.makedirs(out_dir, exist_ok=True)

    NAVY   = "#002060"
    BRIGHT = "#f2c14e"
    DARK   = "#3a3a3a"
    NEUT   = "#d9d9d9"
    ARC    = "#c0392b"
    CENTER = "#1b6fa8"
    TEAL   = "#008080"
    PURPLE = "#6a1b9a"
    GREEN  = "#2e7d32"

    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.6))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    def square_lims(ax, pad=0.0):
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 + pad
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    def panel_title(ax, text):
        ax.set_title(text, fontsize=11)

    ax = axes[0]
    GRID = 9
    cx0, cy0 = GRID // 2, GRID // 2

    ring_offsets = [
        (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
        (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3),
    ]
    ring_xy = [(cx0 + dx, cy0 + dy) for dx, dy in ring_offsets]

    base_gray = 0.55
    img = np.full((GRID, GRID), base_gray)
    bright_run = set(range(0, 10))
    for k in bright_run:
        x, y = ring_xy[k]
        img[y, x] = 0.90
    dark_idx = [k for k in range(16) if k not in bright_run]
    for k in rng.choice(dark_idx, size=3, replace=False):
        x, y = ring_xy[k]
        img[y, x] = 0.15

    for y in range(GRID):
        for x in range(GRID):
            c = str(img[y, x])
            ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=c,
                                   edgecolor="white", linewidth=0.6, zorder=1))

    for k, (x, y) in enumerate(ring_xy):
        on_arc = k in bright_run
        edge = ARC if on_arc else "0.4"
        ax.add_patch(Circle((x, y), 0.30, facecolor="none", edgecolor=edge,
                             linewidth=2.4 if on_arc else 1.0, zorder=3))

    ax.add_patch(Circle((cx0, cy0), 0.22, facecolor=CENTER, edgecolor="white",
                         linewidth=1.0, zorder=4))
    ax.annotate("candidate\npixel $p$", (cx0, cy0), xytext=(cx0 + 2.7, cy0 - 3.6),
                fontsize=9, ha="center", color=CENTER,
                arrowprops=dict(arrowstyle="-", color=CENTER, lw=0.8))
    ax.annotate("10 contiguous pixels\nbrighter than $p$ by $t$",
                (ring_xy[4][0], ring_xy[4][1]), xytext=(cx0 + 4.6, cy0 + 3.4),
                fontsize=8.8, ha="left", color=ARC,
                arrowprops=dict(arrowstyle="-", color=ARC, lw=0.8,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlim(-1.3, GRID + 3.6)
    ax.set_ylim(-1.3, GRID + 0.7)
    square_lims(ax)
    panel_title(ax, "(a) FAST: the ring test")

    ax = axes[1]
    R = 3.2
    theta0 = np.radians(38)

    circ = Circle((0, 0), R, facecolor="#efe9dd", edgecolor=NAVY, linewidth=1.2, zorder=1)
    ax.add_patch(circ)

    n_pairs = 9
    rng2 = np.random.default_rng(3)
    pair_r = rng2.uniform(0.35, 0.92, n_pairs) * R
    pair_ang = rng2.uniform(0, 2 * np.pi, n_pairs)
    pair_len = rng2.uniform(0.9, 2.0, n_pairs)
    pair_dir = rng2.uniform(0, 2 * np.pi, n_pairs)

    highlight = 2
    pts_a, pts_b = [], []
    for i in range(n_pairs):
        a = np.array([pair_r[i] * np.cos(pair_ang[i] + theta0),
                      pair_r[i] * np.sin(pair_ang[i] + theta0)])
        d = np.array([np.cos(pair_dir[i] + theta0), np.sin(pair_dir[i] + theta0)])
        b = a + pair_len[i] * d
        if np.linalg.norm(b) > R * 0.98:
            b = a - pair_len[i] * d
        pts_a.append(a); pts_b.append(b)
        is_hi = (i == highlight)
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=ARC if is_hi else "0.35",
                linewidth=2.2 if is_hi else 1.1, zorder=3, alpha=1 if is_hi else .8)
        for p in (a, b):
            ax.add_patch(Circle(p, 0.09, facecolor=ARC if is_hi else NAVY,
                                 edgecolor="none", zorder=4))

    arr = FancyArrowPatch((0, 0), (R * 0.92 * np.cos(theta0), R * 0.92 * np.sin(theta0)),
                          arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
                          color=CENTER, zorder=5)
    ax.add_patch(arr)
    ax.annotate(r"orientation $\theta$" "\n(intensity centroid)",
                (R * 0.55 * np.cos(theta0), R * 0.55 * np.sin(theta0)),
                xytext=(-R - 1.3, R * 0.55), fontsize=8.8, color=CENTER, ha="left",
                arrowprops=dict(arrowstyle="-", color=CENTER, lw=0.8))
    ax.annotate("one sampling pair\n$\\to$ one bit",
                (pts_a[highlight] + pts_b[highlight]) / 2 + np.array([0.2, 0.3]),
                xytext=(1.6, -R - 1.0), fontsize=8.8, color=ARC, ha="center",
                arrowprops=dict(arrowstyle="-", color=ARC, lw=0.8,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_xlim(-R - 2.4, R + 1.6)
    ax.set_ylim(-R - 1.9, R + 1.3)
    square_lims(ax)
    panel_title(ax, "(b) oriented patch + sampling pairs")

    ax = axes[2]
    n_bits = 16
    bit_rng = np.random.default_rng(11)
    bits = bit_rng.integers(0, 2, n_bits)
    bits[4] = 1
    hi_bit = 4

    box_w, gap = 0.86, 0.14
    total_w = n_bits * box_w + (n_bits - 1) * gap
    x0 = -total_w / 2
    y0 = 0

    for i, b in enumerate(bits):
        x = x0 + i * (box_w + gap)
        is_hi = (i == hi_bit)
        face = ARC if is_hi else (NAVY if b else "white")
        edge = ARC if is_hi else NAVY
        txtcol = "white" if (b or is_hi) else NAVY
        ax.add_patch(Rectangle((x, y0), box_w, box_w, facecolor=face,
                               edgecolor=edge, linewidth=1.6 if is_hi else 1.0,
                               zorder=2))
        ax.text(x + box_w / 2, y0 + box_w / 2, str(int(b)), ha="center",
                va="center", fontsize=10.5, color=txtcol, zorder=3,
                fontweight="bold" if is_hi else "normal")

    ax.annotate("from the highlighted\npair in (b)",
                (x0 + hi_bit * (box_w + gap) + box_w / 2, y0),
                xytext=(x0 + hi_bit * (box_w + gap) + box_w / 2, y0 - 1.7),
                fontsize=8.8, color=ARC, ha="center",
                arrowprops=dict(arrowstyle="-", color=ARC, lw=0.8))

    ax.annotate("", xy=(x0 + hi_bit * (box_w + gap) + box_w / 2, y0 + box_w + 0.15),
                xytext=(x0 + hi_bit * (box_w + gap) + box_w / 2, y0 + box_w + 1.9))

    brace_y = y0 + box_w + 0.35
    ax.plot([x0, x0 + total_w], [brace_y, brace_y], color="0.4", lw=0.9)
    ax.plot([x0, x0], [brace_y - 0.1, brace_y + 0.1], color="0.4", lw=0.9)
    ax.plot([x0 + total_w, x0 + total_w], [brace_y - 0.1, brace_y + 0.1],
            color="0.4", lw=0.9)
    ax.text(0, brace_y + 0.35, "256-bit descriptor (16 of 256 bits shown)",
            ha="center", fontsize=9, color="0.3")

    ax.set_xlim(x0 - 0.6, x0 + total_w + 0.6)
    ax.set_ylim(y0 - 2.4, brace_y + 1.0)
    square_lims(ax)
    panel_title(ax, "(c) the bit-string")

    fig.suptitle("How a hand-crafted extractor works\n(ORB: FAST detector + BRIEF-style descriptor)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 1))

    out_path1 = os.path.join(out_dir, "comp_handcrafted_mechanism.png")
    fig.savefig(out_path1, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"written {out_path1}")


    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for ax in (ax_l, ax_r):
        ax.axis("off")

    ax_l.set_title("(a) Binary Descriptor Space\n(Hamming Distance)", fontsize=11)

    bits_a = [1, 0, 1, 1, 0, 1, 0, 0]
    bits_b = [1, 1, 1, 0, 0, 1, 1, 0]

    box_w, gap = 0.55, 0.10
    n_b = len(bits_a)
    tot_w = n_b * box_w + (n_b - 1) * gap
    x_start = -tot_w / 2

    y_a, y_b = 1.2, -0.2

    ax_l.text(x_start - 0.4, y_a + box_w/2, "Vector $\\mathbf{a}$:", ha="right", va="center", fontsize=10.5)
    ax_l.text(x_start - 0.4, y_b + box_w/2, "Vector $\\mathbf{b}$:", ha="right", va="center", fontsize=10.5)

    diff_indices = [1, 3, 6]

    for i in range(n_b):
        x = x_start + i * (box_w + gap)
        is_diff = i in diff_indices
    
        ax_l.add_patch(Rectangle((x, y_a), box_w, box_w, 
                                 facecolor=ARC if is_diff else NAVY, 
                                 edgecolor="none", zorder=2))
        ax_l.text(x + box_w/2, y_a + box_w/2, str(bits_a[i]), ha="center", va="center", 
                  color="white", fontweight="bold", fontsize=10)
    
        ax_l.add_patch(Rectangle((x, y_b), box_w, box_w, 
                                 facecolor=ARC if is_diff else NAVY, 
                                 edgecolor="none", zorder=2))
        ax_l.text(x + box_w/2, y_b + box_w/2, str(bits_b[i]), ha="center", va="center", 
                  color="white", fontweight="bold", fontsize=10)
    
        if is_diff:
            ax_l.plot([x + box_w/2, x + box_w/2], [y_b + box_w, y_a], 
                      color=ARC, linestyle="--", linewidth=1.5, zorder=1)
            ax_l.text(x + box_w/2, (y_a + y_b + box_w)/2, "≠", ha="center", va="center", 
                      color=ARC, fontweight="bold", fontsize=12,
                      bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="none"))

    ax_l.text(0, -1.2, f"Hamming Distance $d_H(\\mathbf{{a}}, \\mathbf{{b}}) = 3$ bits\n(XOR + popcount)", 
              ha="center", va="top", fontsize=10, color=ARC,
              bbox=dict(boxstyle="round,pad=0.5", fc="#fdf3f2", ec=ARC, lw=1.0))

    ax_l.set_xlim(x_start - 1.8, x_start + tot_w + 0.8)
    ax_l.set_ylim(-2.2, 2.2)

    ax_r.set_title("(b) Continuous Descriptor Space\n(Euclidean Distance)", fontsize=11)

    ax_r.set_aspect("equal")

    ax_r.annotate("", xy=(3.5, 0), xytext=(-0.5, 0), arrowprops=dict(arrowstyle="->", color="0.6", lw=1.0))
    ax_r.annotate("", xy=(0, 3.2), xytext=(0, -0.5), arrowprops=dict(arrowstyle="->", color="0.6", lw=1.0))
    ax_r.text(3.6, 0, "Dim 1", va="center", fontsize=9, color="0.4")
    ax_r.text(0, 3.3, "Dim 2", ha="center", fontsize=9, color="0.4")

    p_a = np.array([0.8, 2.4])
    p_b = np.array([2.9, 0.9])

    ax_r.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=CENTER, linestyle="--", linewidth=1.8, zorder=2)

    ax_r.scatter(*p_a, color=NAVY, s=70, zorder=4)
    ax_r.scatter(*p_b, color=NAVY, s=70, zorder=4)

    ax_r.annotate("$\\mathbf{x}_a = [0.21, -0.84, \\dots, 0.45]^T$", p_a, xytext=(p_a[0] - 0.4, p_a[1] + 0.35),
                  fontsize=9.5, color=NAVY, weight="bold")
    ax_r.annotate("$\\mathbf{x}_b = [0.15, 0.12, \\dots, -0.91]^T$", p_b, xytext=(p_b[0] - 0.6, p_b[1] - 0.45),
                  fontsize=9.5, color=NAVY, weight="bold")

    mid_p = (p_a + p_b) / 2
    ax_r.text(mid_p[0] + 0.2, mid_p[1] + 0.2, "$d_E = ||\\mathbf{x}_a - \\mathbf{x}_b||_2$", 
              color=CENTER, fontsize=10, weight="bold", ha="left")

    ax_r.text(1.5, -1.2, "Euclidean Distance in $\\mathbb{R}^d$\n$d_E(\\mathbf{x}_a, \\mathbf{x}_b) = \\sqrt{\\sum (x_{a,i} - x_{b,i})^2}$", 
              ha="center", va="top", fontsize=10, color=CENTER,
              bbox=dict(boxstyle="round,pad=0.5", fc="#eff6fb", ec=CENTER, lw=1.0))

    ax_r.set_xlim(-1.0, 4.2)
    ax_r.set_ylim(-2.2, 3.6)

    fig2.suptitle("Descriptor Distance Metrics: Binary vs. Continuous Space", fontsize=12.5)
    fig2.tight_layout(rect=(0, 0, 1, 1))

    out_path2 = os.path.join(out_dir, "comp_descriptor_space.png")
    fig2.savefig(out_path2, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig2)
    print(f"written {out_path2}")


    fig4, (ax_att, ax_mat) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax in (ax_att, ax_mat):
        ax.axis("off")

    ax_att.set_xlim(-0.5, 5.5)
    ax_att.set_ylim(-0.5, 5.0)

    ax_att.text(2.5, 4.8, "(a) Self & Cross-Attention (e.g., SuperGlue)", ha="center", va="bottom", fontsize=11)

    ax_att.text(-0.3, 3.5, "Image $A$:\nTokens", ha="right", va="center", fontsize=9.5, color=NAVY)
    pts_A = [(1.0, 3.5), (2.5, 3.5), (4.0, 3.5)]
    for idx, pt in enumerate(pts_A):
        ax_att.add_patch(Circle(pt, 0.35, facecolor="#d0e1f9", edgecolor=NAVY, lw=1.2, zorder=3))
        ax_att.text(pt[0], pt[1], f"$i_{idx+1}$", ha="center", va="center", fontsize=9, color=NAVY, weight="bold")

    ax_att.text(-0.3, 1.0, "Image $B$:\nTokens", ha="right", va="center", fontsize=9.5, color=PURPLE)
    pts_B = [(1.0, 1.0), (2.5, 1.0), (4.0, 1.0)]
    for idx, pt in enumerate(pts_B):
        ax_att.add_patch(Circle(pt, 0.35, facecolor="#f3e5f5", edgecolor=PURPLE, lw=1.2, zorder=3))
        ax_att.text(pt[0], pt[1], f"$j_{idx+1}$", ha="center", va="center", fontsize=9, color=PURPLE, weight="bold")

    ax_att.add_patch(FancyArrowPatch((1.35, 3.6), (2.15, 3.6), arrowstyle="<->", mutation_scale=10, color=CENTER, lw=1.2, connectionstyle="arc3,rad=-0.3"))
    ax_att.add_patch(FancyArrowPatch((2.85, 3.6), (3.65, 3.6), arrowstyle="<->", mutation_scale=10, color=CENTER, lw=1.2, connectionstyle="arc3,rad=-0.3"))
    ax_att.text(2.5, 4.2, "Self-Attention (Intra)", ha="center", fontsize=8.5, color=CENTER)

    ax_att.add_patch(FancyArrowPatch((1.35, 0.9), (2.15, 0.9), arrowstyle="<->", mutation_scale=10, color=PURPLE, lw=1.2, connectionstyle="arc3,rad=0.3"))
    ax_att.add_patch(FancyArrowPatch((2.85, 0.9), (3.65, 0.9), arrowstyle="<->", mutation_scale=10, color=PURPLE, lw=1.2, connectionstyle="arc3,rad=0.3"))

    ax_att.add_patch(FancyArrowPatch((1.0, 3.15), (1.0, 1.35), arrowstyle="<->", mutation_scale=10, color=ARC, lw=1.4))
    ax_att.add_patch(FancyArrowPatch((2.5, 3.15), (2.5, 1.35), arrowstyle="<->", mutation_scale=10, color=ARC, lw=1.4))
    ax_att.add_patch(FancyArrowPatch((4.0, 3.15), (2.5, 1.35), arrowstyle="<->", mutation_scale=10, color=ARC, lw=1.0, linestyle="--"))
    ax_att.text(0.4, 2.25, "Cross-Attention\n(Inter)", ha="center", fontsize=8.5, color=ARC)

    ax_mat.set_xlim(-0.5, 5.0)
    ax_mat.set_ylim(-0.5, 5.0)

    ax_mat.text(2.0, 4.8, "(b) Optimal Transport / Assignment Matrix $P$", ha="center", va="bottom", fontsize=11)

    grid_data = np.array([
        [0.85, 0.05, 0.02, 0.08],
        [0.03, 0.90, 0.01, 0.06],
        [0.01, 0.02, 0.10, 0.87],
        [0.10, 0.03, 0.87, 0.00]
    ])

    labels_y = ["$i_1$", "$i_2$", "$i_3$", "Dustbin $\\phi$"]
    labels_x = ["$j_1$", "$j_2$", "$j_3$", "Dustbin $\\phi$"]

    for i in range(4):
        ax_mat.text(-0.3, 3.2 - i*0.9, labels_y[i], ha="right", va="center", fontsize=9, color=NAVY if i<3 else DARK)
        ax_mat.text(0.5 + i*0.9, 3.9, labels_x[i], ha="center", va="bottom", fontsize=9, color=PURPLE if i<3 else DARK)

    for i in range(4):
        for j in range(4):
            val = grid_data[i, j]
            is_match = (i == j and i < 2) or (i == 2 and j == 3)
            bg_col = "#abebc6" if (i == j and i < 2) else ("#f5b7b1" if (i == 2 and j == 3) else "white")
        
            ax_mat.add_patch(Rectangle((0.1 + j*0.9, 2.8 - i*0.9), 0.8, 0.8, 
                                       facecolor=bg_col, edgecolor="0.7", lw=0.8))
            ax_mat.text(0.5 + j*0.9, 3.2 - i*0.9, f"{val:.2f}", ha="center", va="center", 
                        fontsize=8.5, weight="bold" if is_match else "normal")

    ax_mat.text(1.85, -0.2, "Matched Pairs: $(i_1, j_1), (i_2, j_2)$\nUnmatched $i_3 \\to \\phi$ (Dustbin)", 
              ha="center", va="top", fontsize=9.5, color=GREEN,
              bbox=dict(boxstyle="round,pad=0.4", fc="#e8f8f5", ec=GREEN, lw=1.0))

    fig4.suptitle("Attentional Feature Matching and Soft Assignment Pipeline", fontsize=12.5)
    fig4.tight_layout(rect=(0, 0, 1, 1))

    out_path4 = os.path.join(out_dir, "comp_attention_mechanism.png")
    fig4.savefig(out_path4, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig4)
    print(f"written {out_path4}")


if __name__ == "__main__":
    main()