#!/usr/bin/env python3
"""
src_pc/utils/make_assets_online.py

Builds figures that need pySLAM loaded (models, CUDA weights, dataset reader).
Companion to make_assets_offline.py, which reads all_conditions.csv and never
imports pyslam -- that split keeps the offline script runnable without a
GPU/pySLAM checkout even when this one breaks.

    aggregate_all.py        raw run folders  ->  all_conditions.csv
    make_assets_offline.py  all_conditions.csv + GT poses  ->  most assets
    THIS FILE                pyslam models on real frames  ->  the rest

Note: figures here only draw keypoints from a single frame (detectAndCompute),
never matches. For a DL matcher whose keypoints are computed jointly across
both images, single-image detectAndCompute is close to a no-op -- kps/idxs
would need to come from the matching_result instead.

Usage:
    python3 src_pc/utils/make_assets_online.py
    python3 src_pc/utils/make_assets_online.py --only fe
    python3 src_pc/utils/make_assets_online.py --test test0 --frame 200
    python3 src_pc/utils/make_assets_online.py --front-ends ORB2 AKAZE ALIKED
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project roots, same pattern as slam_runner.py ─────────────────────────
PC_ROOT     = Path(__file__).resolve().parents[2]
PYSLAM_ROOT = PC_ROOT.parent / "pyslam"
sys.path.insert(0, str(PC_ROOT))
sys.path.insert(0, str(PYSLAM_ROOT))

from pyslam.config import Config                                      # noqa: E402
from pyslam.io.dataset_factory import dataset_factory                  # noqa: E402
from pyslam.local_features.feature_tracker import (                    # noqa: E402
    feature_tracker_factory,
)
from pyslam.local_features.feature_tracker_configs import (            # noqa: E402
    FeatureTrackerConfigs,
)
from src_pc.utils.paths import CONFIGS_SEQUENCES                       # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════
FIG_DIR = PC_ROOT / "results_aggregated" / "figures"

# BGR: fed straight to cv2.drawKeypoints, not matplotlib.
FE_COLORS = {
    "ORB2":       (208, 32, 0),      # navy
    "SUPERPOINT": (128, 128, 0),     # teal
    "AKAZE":      (0, 128, 0),
    "BRISK":      (0, 140, 220),
    "ALIKED":     (160, 60, 200),
    "LIGHTGLUE":  (60, 60, 200),
}
DEFAULT_COLOR = (90, 90, 90)

DISPLAY = {
    "ORB2": "ORB2", "SUPERPOINT": "SuperPoint", "LIGHTGLUE": "LightGlue",
    "AKAZE": "AKAZE", "BRISK": "BRISK", "ALIKED": "ALIKED",
    "XFEAT": "XFeat", "DISK": "DISK", "R2D2": "R2D2", "D2NET": "D2-Net",
}


def disp(name: str) -> str:
    return DISPLAY.get(name, name)


def _save(fig, name: str, pad_inches: float = 0.1) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"   figures/{name}.png")


# ════════════════════════════════════════════════════════════════════════════
#  DATASET ACCESS
# ════════════════════════════════════════════════════════════════════════════
def _find_seq_yaml(test_id: str) -> Path:
    hits = sorted(CONFIGS_SEQUENCES.glob(f"{test_id}__*.yaml"))
    if not hits:
        sys.exit(f"no sequence config for {test_id} under {CONFIGS_SEQUENCES}"
                 f" -- run prep_project.py first")
    return hits[0]


def load_frames(test_id: str, frame_idx: int, n: int = 2) -> list:
    """`n` consecutive LEFT images, loaded the same way slam_runner.py does
    (same Config/dataset_factory/getImageColor). Returns BGR uint8 arrays."""
    config = Config(str(_find_seq_yaml(test_id)))
    dataset = dataset_factory(config)
    if frame_idx + n - 1 >= dataset.num_frames:
        sys.exit(f"{test_id} has {dataset.num_frames} frames; frame "
                 f"{frame_idx + n - 1} does not exist. Pick a smaller "
                 f"--frame.")
    imgs = []
    for k in range(n):
        img = dataset.getImageColor(frame_idx + k)
        if img is None:
            sys.exit(f"dataset returned no image at frame {frame_idx + k} -- "
                     f"check the sequence was processed by prep_project.py")
        imgs.append(img)
    return imgs


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (mirrors main_feature_matching.py exactly)
# ════════════════════════════════════════════════════════════════════════════
def tracker(name: str, num_features: int):
    """Builds one tracker the way variants.py does for a real run, so the
    figure matches the actual experiment settings."""
    cfg = dict(getattr(FeatureTrackerConfigs, name))
    cfg["num_features"] = num_features
    return feature_tracker_factory(**cfg)


def detect(trk, img):
    """(keypoints, descriptors) for one image."""
    return trk.detectAndCompute(img)


# ════════════════════════════════════════════════════════════════════════════
#  BUILDERS
# ════════════════════════════════════════════════════════════════════════════
def fig_fe_side_by_side(test_id: str, frame_idx: int, num_features: int,
                        front_ends: list[str]) -> None:
    """comp_fe_side_by_side.png -- raw frame plus each front end's keypoints
    overlaid. Single frame only: a two-frame match panel was dropped since
    consecutive frames here look near-identical and added nothing to compare."""
    img0 = load_frames(test_id, frame_idx, n=1)[0]

    panels = [(img0, "input frame  --  no keypoints")]
    for name in front_ends:
        print(f"  extracting {disp(name)} ...")
        kps, _ = detect(tracker(name, num_features), img0)
        drawn = cv2.drawKeypoints(
            img0, kps, None, color=FE_COLORS.get(name, DEFAULT_COLOR),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        panels.append((drawn, f"{disp(name)}  --  {len(kps)} keypoints"))

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(9.5, 3.1 * len(panels)))
    axes = np.atleast_1d(axes)
    for k, (ax, (img, title)) in enumerate(zip(axes, panels)):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"({chr(97 + k)}) {title}", fontsize=11, pad=1)
        ax.axis("off")

    fig.suptitle(f"One KITTI-360 frame through {len(front_ends)} front ends"
                 f"  ({test_id}, frame {frame_idx})", fontsize=12.5)
    fig.tight_layout(h_pad=0.05)
    _save(fig, "comp_fe_side_by_side", pad_inches=0.03)


def fig_fe_density(test_id: str, frame_idx: int, num_features: int,
                   front_ends: list[str]) -> None:
    """comp_fe_density.png -- each front end's keypoint count by image row,
    one row of small multiples per detector (not overlaid): LIGHTGLUE reuses
    SUPERPOINT's keypoints, so overlaying would hide one identical curve
    behind the other."""
    img0 = load_frames(test_id, frame_idx, n=1)[0]
    h = img0.shape[0]
    bins = np.linspace(0, h, 25)

    counts = {}
    for name in front_ends:
        print(f"  extracting {disp(name)} ...")
        kps, _ = detect(tracker(name, num_features), img0)
        counts[name] = (np.array([k.pt[1] for k in kps]) if len(kps)
                        else np.array([]))

    fig, axes = plt.subplots(len(front_ends), 1, sharex=True,
                             figsize=(8.4, 1.6 * len(front_ends) + 1.2))
    axes = np.atleast_1d(axes)
    peak = 0
    for name in front_ends:
        if counts[name].size:
            peak = max(peak, int(np.histogram(counts[name], bins=bins)[0].max()))

    for ax, name in zip(axes, front_ends):
        rows = counts[name]
        c = FE_COLORS.get(name, DEFAULT_COLOR)
        rgb = tuple(v / 255 for v in c[::-1])          # BGR -> RGB, 0..1
        if rows.size:
            ax.hist(rows, bins=bins, color=rgb, alpha=.85)
        ax.set_ylabel(f"{disp(name)}\n({rows.size})", fontsize=9,
                      rotation=0, ha="right", va="center", labelpad=12)
        ax.set_ylim(0, peak * 1.12 if peak else 1)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[-1].set_xlabel("image row (px)   --   0 is the top of the frame")
    fig.suptitle(f"Where each front end spends its keypoint budget\n"
                 f"{test_id}, frame {frame_idx}, {num_features} requested "
                 f"(actual count beside each row)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "comp_fe_density")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["fe", "density"], default=None,
                    help="build one figure only")
    ap.add_argument("--test", default="test0")
    ap.add_argument("--frame", type=int, default=200,
                    help="an ORDINARY forward-driving frame, not a hard turn:"
                         " these figures are about the mechanism, not a "
                         "failure case (default: 200)")
    ap.add_argument("--num-features", type=int, default=2048,
                    help="matches NUM_FEATURES used in the real runs, so the "
                         "keypoint counts are comparable to the thesis text")
    ap.add_argument("--front-ends", nargs="+",
                    default=["ORB2", "SUPERPOINT", "LIGHTGLUE"],
                    help="FeatureTrackerConfigs names, one keypoint panel "
                         "each (default: ORB2 SUPERPOINT LIGHTGLUE)")
    args = ap.parse_args()

    print("=" * 70)
    print(" assets ONLINE   (needs pyslam + models + dataset)")
    print(f" sequence {args.test}, frame {args.frame}")
    print(f" figures  {FIG_DIR}")
    print("=" * 70)

    want = lambda s: args.only in (None, s)

    if want("fe"):
        try:
            fig_fe_side_by_side(args.test, args.frame, args.num_features,
                                args.front_ends)
        except Exception as e:
            print(f"   [!] comp_fe_side_by_side: {e}")

    if want("density"):
        try:
            fig_fe_density(args.test, args.frame, args.num_features,
                           args.front_ends)
        except Exception as e:
            print(f"   [!] comp_fe_density: {e}")

    print("\n done")


if __name__ == "__main__":
    main()