"""
src_pc/configs/project_settings.py
────────────────────
Parameters tuner for the whole project. These are the "master" settings that control
everything downstream.

HOW TO EDIT:
  Change a value here and it propagates everywhere. I shouldn't need to touch
  prep_project.py, slam_runner.py, run_experiments.py or any yaml by hand.

IMPORTANT:
  If I change a DATASET_* / IMAGE_* / CAMERA_* value, I have to
  !! RE-RUN prep_project.py !!
  so the camera yaml and sequence yamls get regenerated.
"""


# DATASET ------------------------------------------------------------------------------------
# Colour vs grayscale.
#   False → keep RGB. Frames written to image_2/ (left) and image_3/ (right).
#           Sequence yaml gets is_color: True,  camera yaml gets Camera.RGB: 1.
#   True  → convert to grayscale. Frames written to image_0/ / image_1/.
#           Sequence yaml gets is_color: False, camera yaml gets Camera.RGB: 0.
DATASET_GRAYSCALE = False  

# Target image width in pixels. height is auto-scaled to keep the aspect ratio.
IMAGE_WIDTH = 1408
# Capture rate of the sensor (Hz). KITTI-360 is 10 Hz.
CAMERA_FPS = 10.0


# FEATURES -----------------------------------------------------------------------------------
# 2048   power of 2 babby 
NUM_FEATURES = 2048          # keypoints per frame (written into the camera yaml too)

# ORB extractor (written into the camera yaml AND applied at runtime) ----------------------
ORB_SCALE_FACTOR = 1.2       # pyramid scale factor between levels
ORB_NUM_LEVELS   = 8         # number of pyramid levels
ORB_INI_TH_FAST  = 20        # FAST initial corner threshold
ORB_MIN_TH_FAST  = 7         # FAST minimum corner threshold (fallback)

# STEREO / DEPTH -----------------------------------------------------------------------------
# for my sanity, this is the error calculation on the dataset we have 1408x376 (KITTI-360 baseline ≈ 0.594 m):
#   30 m → ~1.4 m error (~4.5%)
#   40 m → ~2.4 m error (~6%)
#   50 m → ~3.8 m error (~7.6%)
#
# NOTE: what this value actually controls depends on the mode:
#   - mono+depth (DE) variants  - handed to the neural depth estimator factory as max_depth.
#   - stereo variants           - the max depth is baked into cam_kitti360.yaml, so if I
#                                 change this I have to re-run prep_project.py to regenerate it.
MAX_STEREO_DEPTH = 50  # m

# Drop "shadow" points from a neural depth prediction (only on the mono+depth path).
# Does nothing for variants without a depth_estimator set.
FILTER_SHADOW_POINTS = True


# MAP SAVING -------------------------------------------------------------------------------
SAVE_MAP_DEFAULT = False


# TRAJECTORY -------------------------------------------------------------------------------
TRAJ_FORMAT = "tum"          # trajectory output format understood by pySLAM


# METRICS -----------------------------------------------------------------------------------
ATE_MAX_DT      = 0.05       # max timestamp gap for GT - prediction association (s)
RPE_DELTA_SHORT = 0.5        # RPE translational window - short-range accuracy (s)
RPE_DELTA_LONG  = 2.0        # RPE translational window - long-range drift     (s)


# TEST MODE --------------------------------------------------------------------------------
# only used in test mode (run_experiments.py --test). quick sanity check that the batch
# runner and slam_runner still work - runs a few frames of each variant, then bails out.
TEST_FRAMES = 10
TEST_TIMEOUT_S   = 3 * 60  # minutes
INTER_TEST_SLEEP = 60      # seconds


# BATCH RUNNER --------------------------------------------------------------------------------
# settings for the batch runner (run_experiments.py), which runs slam_runner.py in its
# own subprocess for each variant.
INTER_RUN_SLEEP       = 3 * 60   # minutes between runs
TRACKING_ABORT_STREAK = 10       # abort a run early if tracking is lost for this many


# RUNTIME LIMITS ------------------------------------------------------------------------------
MAX_RUN_WALL_TIME_S = 20 * 60  # minutes
SETUP_GRACE_S = 2 * 60         # minutes
