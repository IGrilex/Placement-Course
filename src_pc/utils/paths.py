# src_pc/utils/paths.py
# One place that defines every path in the project, so nothing hard-codes paths.
#
# This file sits at PC/src_pc/utils/paths.py, so parents[2] walks back up to the
# PC root (paths.py → utils → src_pc → PC).

from pathlib import Path

# Project roots ----------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]   # .../ASP-SLAM/PC
PYSLAM_ROOT = ROOT.parent / "pyslam"                # .../ASP-SLAM/pyslam

# Raw dataset (read-only, downloaded KITTI-360) --------------------------------------------
RAW        = ROOT / "dataset" / "raw"
RAW_IMAGES = RAW / "data_2d_test_slam"   # test_0/drive/image_00/data_rect/
RAW_POSES  = RAW / "data_poses"          # drive/poses.txt
RAW_CALIB  = RAW / "calibration"

# Processed dataset ------------------------------------------------------------------------
#
#   processed/                           ← KITTI_BASE_PATH  (base_path in yaml)
#     calibration/
#     sequences/
#       test0/drive/
#         times.txt
#         image_2/*.png
#         image_3/*.png
#     poses/
#       test0/
#         drive.txt          ← pySLAM KittiGroundTruth (KITTI format)
#         drive_tum.txt      ← our custom metrics (TUM format)
#
PROCESSED        = ROOT / "dataset" / "processed"
PROCESSED_CALIB  = PROCESSED / "calibration"
PROCESSED_IMAGES = PROCESSED / "sequences"
PROCESSED_POSES  = PROCESSED / "poses"
KITTI_BASE_PATH  = PROCESSED

# Results ----------------------------------------------------------------------------------
#
#   results/
#     test_run/                          ← first_run_pyslam.py (sanity check)
#     logs/                              ← pySLAM internal logs
#     custom_experiments/                ← gradslam, ICP, etc.
#     summary/                           ← aggregate CSVs across all runs
#     test0/
#       feature_extraction/
#         orb2/  superpoint/  xfeat/ ...
#           20260315_143341/             ← run output (timestamped)
#       loop_closure/
#         dbow3/  netvlad/  cosplace/ ...
#       backend/
#         g2o/  gtsam/
#       combined/
#         superpoint_netvlad/ ...
#     test1/ test2/ test3/
#
RESULTS_ROOT       = ROOT / "results"
RESULTS            = RESULTS_ROOT          # slam_runner builds full path below this
CUSTOM_EXPERIMENTS = RESULTS_ROOT / "custom_experiments"
SUMMARY_DIR        = RESULTS_ROOT / "summary"

# Documentation ----------------------------------------------------------------------------
DOCS         = ROOT / "docs"
DOCS_DATASET = DOCS / "dataset"

# Configs ----------------------------------------------------------------------------------
CONFIGS_DIR       = ROOT / "src_pc" / "configs"
CONFIGS_SEQUENCES = CONFIGS_DIR / "sequences"
CAM_CONFIG        = CONFIGS_DIR / "cam_kitti360.yaml"
