"""
src_pc/experiments/variants.py
Every pySLAM variant to test, grouped by experiment phase. Each phase changes
one thing at a time and holds the rest fixed:
  1. Feature Extraction (FE)  - ORB2 baseline, swap the FE. Loop=DBOW3/INDEPENDENT, backend=G2O.
  2. Loop Closure (LC)        - fix ORB2 FE, swap LC. Backend=G2O.
  3. Backend (BE)             - fix ORB2 FE + DBOW3, G2O vs GTSAM.
  4. Combined (CO)            - pair best FE + best LC.
  5. Depth Estimation (DE)    - mono + neural depth vs stereo.

Loop detector per FE: ORB/ORB2 use DBOW3 (shared vocabulary already exists);
everything else uses DBOW3_INDEPENDENT (builds its own vocabulary).

Variant fields:
  tracker           FeatureTrackerConfigs attribute name
  loop              LoopDetectorConfigs attribute name
  experiment_type   feature_extraction | loop_closure | backend | combined | depth_estimation
  short_name        folder-safe name used in output paths
  use_gtsam         True = GTSAM, False = G2O
  num_features      keypoints per frame
  max_depth         stereo/depth cutoff in metres
  depth_estimator   None = stereo | "DEPTH_ANYTHING_V2" | "DEPTH_ANYTHING_V3" | "DEPTH_PRO"
"""
 
from __future__ import annotations
import sys
from pathlib import Path
 
_PC_ROOT     = Path(__file__).resolve().parents[2]
_PYSLAM_ROOT = _PC_ROOT.parent / "pyslam"
 
from src_pc.configs.project_settings import NUM_FEATURES, MAX_STEREO_DEPTH
 
_FE = "feature_extraction"
_LC = "loop_closure"
_BE = "backend"
_CO = "combined"
_DE = "depth_estimation"
_BL = "baseline"
 
_DEFAULTS = {
    "use_gtsam":       False,   # G2O by default, only be_gtsam flips this to True
    "num_features":    NUM_FEATURES,
    "max_depth":       MAX_STEREO_DEPTH,
    "depth_estimator": None,
    "skip_tests":      [],      # tests to skip for this variant (too slow / blows up RAM)
}
 
 
def _v(tracker, loop, exp_type, short, **kwargs) -> dict:
    d = {**_DEFAULTS, "tracker": tracker, "loop": loop,
         "experiment_type": exp_type, "short_name": short}
    d.update(kwargs)
    return d
 
 
VARIANTS: dict[str, dict] = {

    # BASELINE  (ORB2 + DBOW3 + G2O, stereo) -----------------------------------------------
    # shared reference for FE/LC/BE comparisons - run once, all three phases point at it.

    "baseline":           _v("ORB2",          "DBOW3_INDEPENDENT", _BL, "baseline"),

    # FEATURE EXTRACTION  (Loop = fixed.  Backend = G2O fixed.  Stereo.) -------------------

    # Classical ----------------------------------------------------------------------------
    "fe_orb":             _v("ORB",           "DBOW3_INDEPENDENT", _FE, "orb"),
    "fe_sift":            _v("SIFT",          "DBOW3_INDEPENDENT", _FE, "sift",
                            skip_tests=["test2", "test3"]),
    "fe_root_sift":       _v("ROOT_SIFT",     "DBOW3_INDEPENDENT", _FE, "root_sift",
                            skip_tests=["test2", "test3"]),
    "fe_brisk":           _v("BRISK",         "DBOW3_INDEPENDENT", _FE, "brisk"),
    "fe_akaze":           _v("AKAZE",         "DBOW3_INDEPENDENT", _FE, "akaze"),
    "fe_kaze":            _v("KAZE",          "DBOW3_INDEPENDENT", _FE, "kaze",
                            skip_tests=["test0", "test1", "test2", "test3"]),    

    # Deep learned -------------------------------------------------------------------------
    "fe_superpoint":      _v("SUPERPOINT",    "DBOW3_INDEPENDENT", _FE, "superpoint"),
    "fe_disk":            _v("DISK",          "DBOW3_INDEPENDENT", _FE, "disk",
                            skip_tests=["test2", "test3"]),
    "fe_aliked":          _v("ALIKED",        "DBOW3_INDEPENDENT", _FE, "aliked"),
    "fe_keynet":          _v("KEYNET",        "DBOW3_INDEPENDENT", _FE, "keynet",
                            skip_tests=["test2", "test3"]),
    "fe_r2d2":            _v("R2D2",          "DBOW3_INDEPENDENT", _FE, "r2d2",
                            skip_tests=["test2", "test3"]),
    "fe_d2net":           _v("D2NET",         "DBOW3_INDEPENDENT", _FE, "d2net",
                            skip_tests=["test2", "test3"]),
    "fe_xfeat":           _v("XFEAT",         "DBOW3_INDEPENDENT", _FE, "xfeat",
                            skip_tests=["test2", "test3"]),
    "fe_xfeat_xfeat":     _v("XFEAT_XFEAT",  "DBOW3_INDEPENDENT", _FE, "xfeat_xfeat",
                            skip_tests=["test2", "test3"]),
    "fe_lightglue":       _v("LIGHTGLUE",     "DBOW3_INDEPENDENT", _FE, "lightglue"),
    "fe_xfeat_lightglue": _v("XFEAT_LIGHTGLUE","DBOW3_INDEPENDENT",_FE, "xfeat_lg",
                            skip_tests=["test2", "test3"]),

    # New models (ICRA/CVPR 2025) ----------------------------------------------------------
    "fe_liftfeat":        _v("LIFTFEAT",      "DBOW3_INDEPENDENT", _FE, "liftfeat",
                            skip_tests=["test2", "test3"]),
    "fe_rdd":             _v("RDD",           "DBOW3_INDEPENDENT", _FE, "rdd",
                            skip_tests=["test2", "test3"]),
    "fe_dedode":          _v("DEDODE",        "DBOW3_INDEPENDENT", _FE, "dedode",
                            skip_tests=["test2", "test3"]),
 
    # LOOP CLOSURE  (FE = ORB2 fixed.  Backend = G2O fixed.  Stereo.) ----------------------
 
    # Classical vocabulary-based -----------------------------------------------------------
    "lc_dbow2":           _v("ORB2", "DBOW2",       _LC, "dbow2"),
    "lc_vlad":            _v("ORB2", "VLAD",        _LC, "vlad"),
    "lc_ibow":            _v("ORB2", "IBOW",        _LC, "ibow",
                            skip_tests=["test2", "test3"]),
    "lc_obindex2":        _v("ORB2", "OBINDEX2",    _LC, "obindex2"),
    "lc_sad":             _v("ORB2", "SAD",         _LC, "sad"),
 
    # Deep global descriptors --------------------------------------------------------------
    "lc_netvlad":         _v("ORB2", "NETVLAD",     _LC, "netvlad",
                            skip_tests=["test2", "test3"]),
    "lc_cosplace":        _v("ORB2", "COSPLACE",    _LC, "cosplace",
                            skip_tests=["test2", "test3"]),
    "lc_eigenplaces":     _v("ORB2", "EIGENPLACES", _LC, "eigenplaces",
                            skip_tests=["test2", "test3"]),
    "lc_alexnet":         _v("ORB2", "ALEXNET",     _LC, "alexnet",
                            skip_tests=["test1", "test2", "test3"]),
    "lc_megaloc":         _v("ORB2", "MEGALOC",     _LC, "megaloc"),
 
    # New models (ICRA/CVPR 2025) ----------------------------------------------------------
    "lc_cricavpr":  _v("ORB2", "CRICAVPR",  _LC, "cricavpr"),
    "lc_salad":     _v("ORB2", "SALAD",     _LC, "salad"),
    "lc_supervlad": _v("ORB2", "SUPERVLAD", _LC, "supervlad"),
 
    # BACKEND  (FE = ORB2 fixed.  Loop = DBOW3 fixed.  Stereo.) ----------------------------

    "be_gtsam":           _v("ORB2", "DBOW3_INDEPENDENT", _BE, "gtsam", use_gtsam=True),
 
    # COMBINED  (FE + LC pairs.  Stereo.) --------------------------------------------------
 
    "co_akaze_salad":         _v("AKAZE",     "SALAD",    _CO, "akaze_salad"), 
    "co_akaze_megaloc":       _v("AKAZE",     "MEGALOC",  _CO, "akaze_megaloc"),
    "co_akaze_obindex2":      _v("AKAZE",     "OBINDEX2", _CO, "akaze_obindex2"),
    "co_akaze_netvlad":       _v("AKAZE",     "NETVLAD",  _CO, "akaze_netvlad"),
    
    "co_lightglue_salad":     _v("LIGHTGLUE",     "SALAD",    _CO, "lightglue_salad"),
    "co_lightglue_megaloc":   _v("LIGHTGLUE",     "MEGALOC",  _CO, "lightglue_megaloc"),
    "co_lightglue_obindex2":  _v("LIGHTGLUE",     "OBINDEX2", _CO, "lightglue_obindex2"),
    "co_lightglue_netvlad":   _v("LIGHTGLUE",     "NETVLAD",  _CO, "lightglue_netvlad"),
 
    # DEPTH ESTIMATION  (Monocular + neural depth vs stereo baseline) ----------------------
    # FE varies; depth_estimator picks the neural depth model.

    # ORB2 baseline FE - fixes the FE so we only see the depth model's effect --------------
    "de_orb2_dav2":       _v("ORB2",       "DBOW3_INDEPENDENT",             _DE, "orb2_dav2",
                            depth_estimator="DEPTH_ANYTHING_V2",
                            skip_tests=["test1", "test2", "test3"]),
    "de_orb2_dav3":       _v("ORB2",       "DBOW3_INDEPENDENT",             _DE, "orb2_dav3",
                            depth_estimator="DEPTH_ANYTHING_V3"),
 
    # Learned FE + DepthAnythingV2 ---------------------------------------------------------
    "de_superpoint_dav2": _v("SUPERPOINT", "DBOW3_INDEPENDENT", _DE, "sp_dav2",
                            depth_estimator="DEPTH_ANYTHING_V2",
                            skip_tests=["test1", "test2", "test3"]),
    "de_superpoint_dav3": _v("SUPERPOINT", "DBOW3_INDEPENDENT", _DE, "sp_dav3",
                            depth_estimator="DEPTH_ANYTHING_V3"),
}
 
EXPERIMENT_TYPES = [_FE, _LC, _BE, _CO, _DE]

BASELINE_VARIANT = "baseline"
# phases whose comparison plots measure against the baseline
BASELINE_PHASES = (_FE, _LC, _BE)


def is_baseline(name: str) -> bool:
    return name == BASELINE_VARIANT


# Accessors --------------------------------------------------------------------------------
 
def get_variant(name: str) -> dict:
    """Turn a variant name into a fully filled-in config dict. Config() must already be init'd."""
    if name not in VARIANTS:
        available = "\n  ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown variant '{name}'.\nAvailable:\n  {available}")
 
    raw = VARIANTS[name]
 
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
 
    tracker_cfg = dict(getattr(FeatureTrackerConfigs, raw["tracker"]))
    tracker_cfg["num_features"] = raw["num_features"]
 
    return {
        "tracker_cfg":      tracker_cfg,
        "loop_cfg":         getattr(LoopDetectorConfigs, raw["loop"]),
        "experiment_type":  raw["experiment_type"],
        "short_name":       raw["short_name"],
        "use_gtsam":        raw["use_gtsam"],
        "num_features":     raw["num_features"],
        "max_depth":        raw["max_depth"],
        "depth_estimator":  raw["depth_estimator"],
    }
 
 
def get_experiment_variants(experiment_type: str) -> dict[str, dict]:
    return {k: v for k, v in VARIANTS.items() if v["experiment_type"] == experiment_type}
 
 
def list_variants(experiment_type: str | None = None) -> list[str]:
    if experiment_type:
        return sorted(k for k, v in VARIANTS.items() if v["experiment_type"] == experiment_type)
    return sorted(VARIANTS)
 
 
def print_variants() -> None:
    """Print every variant, grouped by experiment type - easy to eyeball."""
    for et in EXPERIMENT_TYPES:
        items = [(k, v) for k, v in VARIANTS.items() if v["experiment_type"] == et]
        if not items:
            continue
        print(f"\n  {et.upper().replace('_', ' ')} ({len(items)} variants):")
        for name, v in sorted(items):
            depth = f"  [depth: {v['depth_estimator']}]" if v.get("depth_estimator") else ""
            print(f"    {name:<40} {v['tracker']:<20} + {v['loop']}{depth}")
 
 
if __name__ == "__main__":
    print("\n=== SLAM Variants ===")
    print_variants()
    print(f"\n  Total: {len(VARIANTS)} variants")
    print()