# ASP-SLAM PC Project

Benchmarking harness for the thesis *"How is memory capacity, single-core performance and
multithreading constraints affecting deep-learning SLAM components compared to classical
baselines?"* ([Thesis_Zincenco_Petru.pdf](Thesis_Zincenco_Petru.pdf)).

None of the results come from pySLAM's own entry points. This repo (`src_pc`) sits beside a
pySLAM checkout, imports it as a library, and owns what pySLAM leaves to the caller: which
configurations exist, how a run is bounded, what it leaves behind, and how ~1,900 runs on four
hardware conditions become one CSV.

---

## 1. Project Structure

```
~/ASP-SLAM/                                         # ← parent folder
├── pyslam/                                         # ← cloned pySLAM (sibling, not inside PC/)
└── PC/                                             # ← actual project root
    ├── src_pc/
    │   ├── configs/
    │   │   ├── sequences/                          # auto-generated yaml per test sequence
    │   │   │   ├── test0__2013_05_28_drive_0008_sync.yaml
    │   │   │   ├── test1__2013_05_28_drive_0008_sync.yaml
    │   │   │   ├── test2__2013_05_28_drive_0004_sync.yaml
    │   │   │   └── test3__2013_05_28_drive_0002_sync.yaml
    │   │   ├── cam_kitti360.yaml                   # generated: intrinsics / ORBextractor block
    │   │   └── project_settings.py                 # single source of truth for all tunable params
    │   ├── experiments/
    │   │   ├── variants.py                         # all SLAM variant definitions (FE / LC / BE / CO / DE)
    │   │   ├── slam_runner.py                      # run ONE variant on ONE sequence, in-process
    │   │   ├── batch_runner.py                     # batch launcher — spawns slam_runner as subprocesses
    │   │   ├── test_models.py                      # sanity-check every variant (TEST_FRAMES each)
    │   │   └── first_run_pyslam.py                 # one-shot integration check (ORB2 on test0, with GUI)
    │   └── utils/
    │       ├── paths.py                            # centralised path constants
    │       ├── prep_project.py                     # dataset extraction + yaml generation
    │       ├── dataset_helper_functions.py         # image / calibration helpers
    │       ├── metrics_helper_functions.py         # ATE / RPE / system-monitor
    │       ├── printing_helper_functions.py        # dashboard / plot writers
    │       ├── aggregate_results.py                # stage 1 — aggregate ONE results tree
    │       ├── aggregate_all.py                    # stage 2 — every tree → all_conditions.csv
    │       ├── make_assets_offline.py              # CSV → thesis figures + LaTeX tables (no GPU)
    │       ├── make_assets_online.py               # figures that need pySLAM models loaded
    │       ├── make_family_overview.py             # classical / CNN / transformer family figures
    │       ├── refactor_pipeline.py                # one-shot historical refactor (already applied)
    │       └── view_map.py                         # 3-D viewer for .ply map files
    ├── dataset/
    │   ├── raw/
    │   │   └── kitti360/                           # extracted KITTI-360 downloads (read-only)
    │   │       ├── calibration/
    │   │       ├── data_2d_test_slam/
    │   │       └── data_poses/
    │   └── processed/                              # created by prep_project.py
    │       ├── calibration/
    │       ├── sequences/                          # test*/<drive>/image_2/ and image_3/ PNGs + times.txt
    │       └── poses/                              # test*/<drive>.txt (KITTI) + <drive>_tum.txt (TUM)
    ├── results/                                    # ← the LIVE tree the runners always write to
    │   └── test/
    │       ├── first_run/                          # output of first_run_pyslam.py
    │       └── model_test.csv                      # output of test_models.py
    ├── results_I7_14RAM/                           # ← archived campaigns, one per hardware condition
    ├── results_R9_12T_14RAM/                       #    (see §3 — renamed by hand after each batch)
    ├── results_R9_12T_28RAM/
    ├── results_R9_32T_28RAM/
    ├── results_aggregated/                         # cross-condition output (aggregate_all + assets)
    │   ├── csv/                                    # all_conditions.csv, DICTIONARY.md, map_growth.csv, …
    │   ├── figures/                                # every figure in the results chapters
    │   └── tables/                                 # every LaTeX table in the results chapters
    ├── logs/                                       # pySLAM internal logs (gba, loop, mapping …)
    ├── docs/
    │   ├── dataset/                                # dataset_summary.txt, overview.png, per-test plots
    │   ├── memos/                                  # notes / slides
    │   └── pyslam_description.txt                  # notes on the pySLAM internals
    ├── Thesis_Zincenco_Petru.pdf
    ├── activate.sh                                 # enter pySLAM venv
    ├── .gitignore
    └── README.md
```

> `docs/dataset/dataset_summary.txt` is not optional: `batch_runner.py` and
> `aggregate_results.py` read the per-sequence frame counts back out of it, which is how a
> hard-killed run can still be reported as a percentage of its sequence. It is written by
> `prep_project.py`.

### 1.1 Layout of one results tree

Every tree — the live `results/` and each archived `results_*/` — has the same shape:

```
results[_<CONDITION>]/
├── SLAM/                                   # loop-closure-enabled runs
│   ├── test0/
│   │   ├── baseline/<timestamp>/           # ORB2 + DBOW3_INDEPENDENT + G2O shared reference
│   │   ├── feature_extraction/<short_name>/<timestamp>/
│   │   ├── loop_closure/<short_name>/<timestamp>/
│   │   ├── backend/<short_name>/<timestamp>/
│   │   ├── combined/<short_name>/<timestamp>/
│   │   ├── depth_estimation/<short_name>/<timestamp>/
│   │   └── errors.txt                      # appended by batch_runner on failures
│   └── test1/ test2/ test3/                # same layout
├── VO/                                     # same layout, loop closure disabled
├── summary/                                # written by aggregate_results.py (§6.1)
│   ├── overview/                           # test0-3.csv, all_tests.csv, averages.csv
│   ├── SLAM/
│   │   ├── overview/                       # status_overview.png, status_matrix[_<phase>].png,
│   │   │                                   #   status_summary.txt
│   │   ├── trends/                         # completion_cdf.png, gradients.png
│   │   └── fe|lc|be|co|de/
│   │       ├── failure_map_<phase>.png
│   │       ├── machine_usage/              # <test>_cpu/ram/gpu.png
│   │       ├── performance/                # <test>_ate_rmse/fps/…png
│   │       ├── trends/
│   │       └── csv/                        # <test>_<metric>.csv
│   └── VO/                                 # same layout (fe / de phases only)
└── run_flags.csv                           # per-run outcome log (batch_runner)
```

---

## 2. Experiment Phases and Variants

Variants are defined in [src_pc/experiments/variants.py](src_pc/experiments/variants.py) — 46
entries, which is the experimental design expressed as a dictionary.

| Phase | Flag | Count | What varies | Fixed |
|---|---|---|---|---|
| **Baseline** | *(implicit)* | 1 | ORB2 + DBOW3\_INDEPENDENT + G2O | — shared reference for FE / LC / BE |
| **Feature Extraction** (FE) | `--fe` | 19 | Feature extractor | Loop = DBOW3\_INDEPENDENT, Backend = G2O |
| **Loop Closure** (LC) | `--lc` | 13 | Loop detector | FE = ORB2, Backend = G2O |
| **Backend** (BE) | `--be` | 1 | G2O vs GTSAM | FE = ORB2, Loop = DBOW3\_INDEPENDENT |
| **Combined** (CO) | `--co` | 8 | FE + LC pairs | Stereo, Backend = G2O |
| **Depth Estimation** (DE) | `--de` | 4 | Neural depth model | Loop = DBOW3\_INDEPENDENT, Backend = G2O |

The CO pairs are `akaze|lightglue` × `salad|megaloc|obindex2|netvlad`.

Results land in `results/SLAM/` (loop closure on) or `results/VO/` (loop closure off, `--vo`
flag). Loop closure is switched off by passing no loop configuration to pySLAM at all, which is
also what selects the output tree.

**The loop detector per FE follows one rule.** ORB-family descriptors could use pySLAM's shipped
vocabulary, but everything in this study — baseline included — uses `DBOW3_INDEPENDENT`, which
extracts its own ORB2 descriptors per keyframe. Holding that constant is what keeps the FE
comparison fair. There is no plain-`DBOW3` variant anywhere in the table; the baseline folder is
the DBoW3 reference point.

**Baseline reuse.** Rather than run the identical ORB2/DBoW3/G2O configuration once per phase, it
runs once per `(mode, test_id)` into a shared `baseline/` folder that FE, LC and BE all point at.
A phase batch tops that folder up to the requested repetition count, counting only runs that left
a `dashboard.txt`, so a crashed baseline is retried rather than silently reused.

### 2.1 The skip list

Each variant carries a `skip_tests` list, and `batch_runner` honours it — so `--test all` does
**not** mean "every variant on every sequence". 20 of the 46 variants skip at least one sequence.
Every exclusion was made for failing at 14 GB, so the excluded set is drawn from the
worst-performing tail; see thesis §5.1.3 for the bias this introduces and why pooled completion
figures for the deep families are lower bounds.

Two cases worth knowing:

- **SIFT and LiftFeat** ignore the platform's feature cap and emit 5,000–7,000 keypoints per
  frame, so they OOM on the long sequences and are skipped there rather than allowed to burn a
  twenty-minute slot to die.
- **KAZE** has all four sequences in its skip list. It is still in the variant table and still
  appears in the design, but it never executes and contributes no data — read it as a documented
  exclusion, not a result. (This is why the design has 46 variants but `test0` only ever ran 45.)

---

## 3. The Hardware Ladder

The spine of the thesis is four hardware conditions, each differing from the previous by exactly
one factor. Each archived `results_*/` tree is one condition.

| ID | Tree | CPU | Threads | Guest RAM | Isolates |
|---|---|---|---|---|---|
| 1 | `results_I7_14RAM/` | i7-8700K | 12 | 14 GB | baseline (pilot) |
| 2 | `results_R9_12T_14RAM/` | R9 9950X | 12 | 14 GB | platform |
| 3 | `results_R9_12T_28RAM/` | R9 9950X | 12 | 28 GB | capacity |
| 4 | `results_R9_32T_28RAM/` | R9 9950X | 32 | 28 GB | core count |

Condition 3 (`R9_12t_28`) is the **reference condition**: every software comparison is made there
alone, so a method that failed for want of memory is not ranked by its memory footprint instead
of its capability. Condition 4 is the **check** that no software conclusion depends on thread
count. Both are hard-coded as `REFERENCE` / `CHECK` in
[make_assets_offline.py:69-70](src_pc/utils/make_assets_offline.py#L69-L70) and as
`CONDITION_CODE` in [make_family_overview.py:53](src_pc/utils/make_family_overview.py#L53).

**The runners always write to `results/`.** There is no condition flag. The workflow was: set the
WSL2 memory and processor count in `.wslconfig`, restart the guest, verify from inside it, run the
batch, then rename `results/` to `results_<CONDITION>/` by hand before the next condition.

**Folder naming is load-bearing.** `aggregate_all.py` discovers conditions by scanning for
directories named `results*` that contain a `SLAM/` or `VO/` subfolder, then parses the CPU
(`i7` / `r9`), thread count (`<n>T`) and memory (`<n>RAM`) out of the folder name. A tree whose
name does not parse needs an entry in `MANUAL_OVERRIDES`
([aggregate_all.py:188](src_pc/utils/aggregate_all.py#L188)) — which is exactly why
`results_I7_14RAM` has one: it carries no `12T` in its name. The ladder is then sorted i7 first,
then ascending RAM, then ascending threads, and each rung's label is inferred from whichever
attribute changed between neighbours.

Guest configuration held constant across all four conditions: 8 GB swap, the `dropcache`
memory-reclaim policy, an RTX 4060 Ti (8 GB), a Samsung 990 PRO 2 TB, and Windows 10 Pro hosting
the Ubuntu guest under WSL2.

---

## 4. Setup

### 4.1 Clone pySLAM and build it

pySLAM must sit **alongside** this folder:

```
~/ASP-SLAM/
├── pyslam/     ← clone here
└── PC/         ← this repo
```

Follow pySLAM's own build instructions, then verify with `pyenv-activate.sh`.

> Because the harness imports pySLAM from a sibling checkout rather than an installed package,
> pulling a new pySLAM version leaves the compiled C++ cores stale and the Python / pybind11
> layers desynchronised. Nothing detects this. Rebuild the cores cleanly after any update —
> the symptom is an import or signature error at start-up, or worse, a silently different code
> path.

### 4.2 Download the KITTI-360 dataset

From the KITTI-360 website download:

- **2D data & labels → Test SLAM** (~14 GB)
- **Calibrations & Poses → Calibrations** (~3 KB)
- **Calibrations & Poses → Vehicle Poses** (~8.9 MB)

Extract everything into `dataset/raw/kitti360/` with this layout:

```
dataset/raw/kitti360/
├── calibration/
├── data_2d_test_slam/
└── data_poses/
```

> The `kitti360/` level is required — [prep_project.py](src_pc/utils/prep_project.py#L48) looks
> for `dataset/raw/kitti360/` and nothing else. When copying calibration files, remove the extra
> nesting folder that the archive may add (e.g. `calibration/calibration/…` → `calibration/…`).

### 4.3 Prepare the dataset

```bash
python3 src_pc/utils/prep_project.py
```

This reads the raw download and produces a processed tree in pySLAM's KITTI layout: rectified
left and right images, a per-sequence timestamp file, and ground truth written twice — KITTI
format for pySLAM's own reader, TUM format for this harness's metrics. It also generates
`src_pc/configs/sequences/*.yaml`, `src_pc/configs/cam_kitti360.yaml` and
`docs/dataset/dataset_summary.txt`. It will take a few minutes.

The camera file is **generated, not copied**: intrinsics come from the dataset's own calibration
and are scaled to `IMAGE_WIDTH` with the height following, the stereo baseline is recovered in
metres before scaling and re-multiplied by the scaled focal length so the physical baseline
survives the resize, and the `ORBextractor.*` block is written from `project_settings.py`. That
YAML is the only route by which the feature count and pyramid settings actually reach pySLAM's
extractor — see §8.

### 4.4 Activate the environment

```bash
. ./activate.sh
```

---

## 5. Running Experiments

All commands must be run from `PC/` (the project root) with the pySLAM venv active.

### 5.1 Integration check (first time only)

Runs the ORB2 baseline on test0 with the 3-D viewer open. Output goes to
`results/test/first_run/`, which is always overwritten and skipped by aggregation.

```bash
python3 src_pc/experiments/first_run_pyslam.py
```

### 5.2 Single variant — one sequence

[slam_runner.py](src_pc/experiments/slam_runner.py) is the low-level runner. It runs one variant
on one test in-process, headless by default.

```bash
# default: baseline on test0, headless
python3 src_pc/experiments/slam_runner.py

# specific variant and test
python3 src_pc/experiments/slam_runner.py --variant fe_superpoint        --test test0
python3 src_pc/experiments/slam_runner.py --variant lc_cosplace          --test test1
python3 src_pc/experiments/slam_runner.py --variant be_gtsam             --test test0
python3 src_pc/experiments/slam_runner.py --variant co_lightglue_salad   --test test2
python3 src_pc/experiments/slam_runner.py --variant de_orb2_dav2         --test test0

# open the 3-D and camera viewers
python3 src_pc/experiments/slam_runner.py --variant baseline --test test0 --view

# visual-odometry mode (loop closure off → results/VO/ instead of results/SLAM/)
python3 src_pc/experiments/slam_runner.py --variant baseline --test test0 --no-loop-closure

# quick test: run TEST_FRAMES frames only, no file output
python3 src_pc/experiments/slam_runner.py --variant fe_superpoint --test test0 --test-mode

# skip saving the map.ply (faster, less disk)
python3 src_pc/experiments/slam_runner.py --variant baseline --test test0 --no-save-map

# list all available variants
python3 src_pc/experiments/slam_runner.py --list-variants
```

Only three of pySLAM's ~200 global parameters are set at run time: the GTSAM flag, the stereo
depth cutoff, and the front-end depth flag for the monocular variants. Everything else arrives
through the generated YAML or is fixed inside the C++ core — see §8.

Two sensor paths exist. The **stereo** path passes both images and lets pySLAM do the matching.
The **monocular-plus-depth** path (DE variants) passes the left image with a depth map from a
pySLAM depth estimator, optionally shadow-filtered, and declares the sensor type as RGBD while
leaving the dataset's own type alone: the RGBD front end is what consumes a predicted depth map,
while the dataset still reports itself as stereo so pySLAM's evaluation does not solve away a
scale the depth model has already fixed.

**Output structure** (one run):

```
results/SLAM/<test>/<exp_type>/<short_name>/<timestamp>/
    dashboard.txt               ← summary: ATE, RPE, tracking quality, system usage
    map/map.ply                 ← coloured point cloud (if --no-save-map not set)
    files/
        progress.csv            ← one row per frame, survives SIGKILL / OOM
        trajectory_online.txt   ← pose written at tracking time
        trajectory_final.txt    ← pose after final BA
        ate_errors.txt
        rpe_short_errors.txt
        rpe_long_errors.txt
    plot/
        trajectory.png          ← GT (grey) vs prediction (blue), top-down
        progress.png            ← tracking state / map growth / fps vs frame
        system_usage.png        ← CPU / RAM / GPU over time
```

`progress.csv` carries tracking state, keyframe and map-point counts, loss and relocalisation
counters, approximate frame rate, and CPU / RAM / GPU load sampled at that frame. It is
line-buffered and flushed after every row, which is the mechanism by which a run killed by the
kernel still contributes real measurements.

`dashboard.txt` is written exactly once, at the very end. It is therefore the only reliable "is
this run finished?" signal, and both the teardown killer and the outcome classifier key on it.

Accuracy is computed twice by two independent routes: pySLAM's own evaluation, and this harness's
own metrics from the final TUM trajectory and ground truth (Umeyama alignment plus RPE over a
short and a long window). **The harness's number is the one reported.** The harness always solves
for scale and reports the factor as a diagnostic rather than a correction — on a stereo run it
should come out at essentially 1.0, and a value that does not is scale drift worth looking at.
Where the two ATE numbers disagree noticeably, the association window is the first thing to check.

### 5.3 Batch runs — full phase or specific variant

[batch_runner.py](src_pc/experiments/batch_runner.py) builds a plan from the requested phases,
sequences and repetition count, then executes each entry as a separate `slam_runner.py`
subprocess.

```bash
# run a full phase across all test sequences
python3 src_pc/experiments/batch_runner.py --fe  --test all
python3 src_pc/experiments/batch_runner.py --lc  --test all
python3 src_pc/experiments/batch_runner.py --be  --test all
python3 src_pc/experiments/batch_runner.py --co  --test all
python3 src_pc/experiments/batch_runner.py --de  --test all
python3 src_pc/experiments/batch_runner.py --all --test all   # every phase

# run a single phase on a single test
python3 src_pc/experiments/batch_runner.py --fe --test test0

# run a specific variant (one test or all)
python3 src_pc/experiments/batch_runner.py --variant fe_superpoint         --test test0
python3 src_pc/experiments/batch_runner.py --variant lc_netvlad            --test all
python3 src_pc/experiments/batch_runner.py --variant co_lightglue_megaloc  --test all
python3 src_pc/experiments/batch_runner.py --variant de_orb2_dav2          --test all

# visual-odometry mode (only FE + DE phases make sense without loop closure)
python3 src_pc/experiments/batch_runner.py --fe  --test all --vo
python3 src_pc/experiments/batch_runner.py --de  --test all --vo

# dry run — print the plan without executing
python3 src_pc/experiments/batch_runner.py --fe  --test test0 --dry-run
python3 src_pc/experiments/batch_runner.py --variant fe_superpoint --test all --dry-run

# multiple runs per variant (for statistical averaging)
python3 src_pc/experiments/batch_runner.py --fe --test all --runs 3
```

> `batch_runner.py` also accepts `--no-save-map`, but **the flag is broken**: it appends an empty
> argument instead of the switch it means to pass, which the argument parser would reject. It has
> gone unnoticed because map saving is off by default and the switch is never used. Don't use it.
> ([batch_runner.py:557](src_pc/experiments/batch_runner.py#L557))

**Isolation.** pySLAM spawns processes of its own, and a leaked worker holds memory the next run
needs. Each run is launched into its own process group; however it ends, the whole group is
killed in a `finally` block, followed by a sweep of the process table for anything whose command
line still names the checkout or the runner.

**Every run is bounded three times:**

1. `slam_runner` checks the 20-minute wall-clock limit inside the frame loop, with the clock
   starting at the first frame actually processed so model loading does not count against it.
2. Because that check happens *between* frames it cannot interrupt a single pathological frame,
   so `batch_runner` adds a hard limit two minutes later (`MAX_RUN_WALL_TIME_S + SETUP_GRACE_S`),
   interrupting the process group first and killing after a 30-second grace period.
3. A third timer covers the case the other two handle badly: once `dashboard.txt` appears the
   heavy work is done and only shutdown remains, so a hung teardown is killed after 30 seconds
   rather than consuming the remaining hard timeout.

A run is additionally declared **lost** after `TRACKING_ABORT_STREAK` consecutive frames without
tracking. There is **no in-loop memory guard** — out-of-memory deaths are not predicted but
recognised afterwards from the kill signal (-9 / -15), and the per-frame log is what keeps those
runs countable.

**Outcome classification uses two independent records.** `dashboard.txt` is the truth when it
exists; when the process was killed first, the last row of `progress.csv` gives the frame reached
and the flag row gives the exit status. Every run appends one row to `results/run_flags.csv`, and
failures are additionally appended to `results/<mode>/<test>/errors.txt` with the full command,
so any of them can be reproduced by copying one line.

Each run is followed by an `INTER_RUN_SLEEP` cooldown (3 min) so the OS can reclaim memory before
the next one starts. Under WSL2 the guest releases page cache back to Windows lazily, and early
batches showed later runs starting with less effective memory than earlier ones — the cooldown
exists specifically to let reclaim catch up.

### 5.4 Model sanity check

Runs every variant for `TEST_FRAMES` frames (default 10) to verify they load and run without
crashing. Output: `results/test/model_test.csv`.

```bash
python3 src_pc/experiments/test_models.py
```

Columns: `variant, category, status (PASS|FAIL|OOM|TIMEOUT), exit_code, elapsed_s,
frames_requested, notes`. Uses its own limits — `TEST_TIMEOUT_S` (3 min) and `INTER_TEST_SLEEP`
(60 s) — not the batch runner's.

---

## 6. Aggregation and Thesis Assets

Four scripts, two stages, one mechanical split. Both aggregation stages **rebuild from scratch
every time**, because a stale figure produced under an older definition of "success" is
indistinguishable from a fresh one and will end up in a thesis.

```
raw run folders  ──(6.1 aggregate_results)──▶  <tree>/summary/ + <run>/post_processing/
                 ──(6.2 aggregate_all)─────▶  results_aggregated/csv/all_conditions.csv
                 ──(6.3 make_assets_offline)▶  results_aggregated/figures/ + tables/
                 ──(6.4 make_assets_online)─▶  figures that need pySLAM loaded
```

### 6.1 Stage 1 — one results tree

```bash
# the live results/ tree
python3 src_pc/utils/aggregate_results.py

# an archived condition, with its budget so RAM%/CPU% become absolute GB and busy-thread counts
python3 src_pc/utils/aggregate_results.py --root results_R9_12T_28RAM --ram-gb 28 --threads 12
python3 src_pc/utils/aggregate_results.py --root results_I7_14RAM --ram-gb 14 --threads 12 --modes SLAM
```

This will:

1. Recover metrics for runs that were killed before writing `dashboard.txt` (writes
   `<run>/post_processing/`)
2. Classify every run (SUCCESS / LOST / TIMEOUT / OOM / ERROR)
3. Print a per-variant status table to the terminal
4. Delete and rebuild `<root>/summary/` — see §1.1 for the full layout

`--ram-gb` / `--threads` are what make the resource columns comparable *between* machines; omit
them and you get percentages of an unstated budget. Stage 2 fills them in automatically from each
folder name, so you normally only pass them when aggregating one tree by hand.

### 6.2 Stage 2 — every tree, and the ladder between them

```bash
python3 -m src_pc.utils.aggregate_all
python3 -m src_pc.utils.aggregate_all --report          # also rung_report.txt + transitions_*.csv
python3 -m src_pc.utils.aggregate_all --mode VO
python3 -m src_pc.utils.aggregate_all --no-run-level    # much faster, loses provenance
```

Discovers every condition (§3), re-runs stage 1 over each one with its parsed RAM and thread
count, then writes `results_aggregated/csv/`:

- **`all_conditions.csv`** — long-form, one `level` column per row. Filter on `level` first;
  summing unfiltered counts every run twice.
  - `run` — one individual execution (provenance, repetition spread)
  - `cell` — one `(condition, mode, test, exp, variant)`, pooled over repetitions. **This is what
    every figure and table reads.**
  - `condition` — one hardware condition, headline accounting on common non-probe cells only
- **`all_conditions_DICTIONARY.md`** — generated column reference. Read this before using the CSV.
- **`map_growth.csv`** — the only per-frame series exported, because it is the one thing the
  offline script cannot reconstruct: the CSV holds a single final keyframe count per run, not the
  growth curve, which lives only in each run's own `progress.csv`.
- **`provenance_table`, `baseline_health_table`, `probe_cells_table`** (`.tex` in
  `results_aggregated/tables/`) — data-hygiene tables describing the *collection* rather than the
  results.

Two labelling schemes in that CSV are worth knowing before reading any number out of it:

| `provenance` (run level) | Meaning |
|---|---|
| `measured` | both a `run_flags.csv` row and a run directory. Normal. |
| `recovered` | directory only. The run happened; its metrics are real. |
| `orphaned` | flag row but no directory. Kept so run counts reconcile, never aggregated. |

| `rep_status` (cell level) | Meaning |
|---|---|
| `probe` | n=1 where the norm is ≥3 — an exploratory run behind a skip decision, not a method measurement. **Excluded from every aggregate.** |
| `under` / `over` | fewer / more repetitions than the norm. Kept, with n visible. |
| `full` | the norm. |

Every headline metric has a `rel_<column>` twin: the ratio to the ORB2 baseline on the *same*
`(condition, mode, sequence)`. Ratio of medians, never median of ratios — unequal repetition
counts make the latter far noisier. Accuracy ratios are null, not zero, when the baseline itself
did not complete.

### 6.3 Thesis figures and tables (offline, no GPU)

```bash
python3 -m src_pc.utils.make_assets_offline
python3 -m src_pc.utils.make_assets_offline --only software     # software | hardware | general | dataset
python3 -m src_pc.utils.make_assets_offline --table-csv         # also dump the numbers behind each table
python3 -m src_pc.utils.make_assets_offline --in results_aggregated
```

Reads `all_conditions.csv` and the ground-truth poses, and nothing else — so it runs in seconds,
needs no GPU and no pySLAM checkout, and can be re-run any time a number changes. Three sections,
three questions: `software/` (which algorithm is more accurate/robust — reference condition only),
`hardware/` (what the machine did, across all four conditions), `general/` (findings needing both
modes at both memory budgets).

Conventions it enforces: accuracy is reported as mean ATE over *completed* runs plus a separate
completion %, never blended and never as a ratio to baseline (a metre is a metre regardless of
platform, unlike throughput). Cost is always given both ways — absolute with the baseline ratio in
brackets, e.g. `5.3 (0.80x)`. A `†` marks a sequence where the baseline itself did not complete
every run, so anything measured against it there rests on a thin reference. A dash means a
sequence was never attempted.

Family-level figures (classical / CNN / transformer) are a separate script, pinned to the
reference condition and to `test0`+`test1` — the only sequences where every family got its full
line-up:

```bash
python3 -m src_pc.utils.make_family_overview
python3 -m src_pc.utils.make_family_overview --mode VO
```

### 6.4 Figures that need pySLAM loaded

```bash
python3 src_pc/utils/make_assets_online.py
python3 src_pc/utils/make_assets_online.py --only fe            # fe | density
python3 src_pc/utils/make_assets_online.py --test test0 --frame 200
python3 src_pc/utils/make_assets_online.py --front-ends ORB2 AKAZE ALIKED
```

The only asset script that imports pySLAM. The split is deliberate: the moment one figure in the
offline script imported pySLAM, a missing weight file would take down forty figures that never
needed a GPU.

> These figures draw keypoints from a single frame via `detectAndCompute`, never matches. For a
> DL matcher whose keypoints are computed jointly across both images that is close to a no-op —
> keypoints would have to come from the `matching_result` instead.

### 6.5 View a map

```bash
python3 src_pc/utils/view_map.py results_R9_12T_28RAM/SLAM/test0/baseline/<timestamp>/map/map.ply
```

### 6.6 `refactor_pipeline.py`

A one-shot, already-applied reorganisation of the asset scripts (it split
`make_results_assets.py` into `make_assets_offline.py`, absorbed `make_diagrams.py`, and stripped
~1,500 lines of dead figure code from `aggregate_all.py`). It is idempotent and detects the
finished state, so running it now does nothing. Kept for provenance only.

---

## 7. Key Configuration

All tunable parameters live in
[src_pc/configs/project_settings.py](src_pc/configs/project_settings.py). Edit there and they
propagate everywhere. **Parameters that affect the dataset or the camera YAML (`DATASET_*`,
`IMAGE_*`, `CAMERA_*`, `NUM_FEATURES`, `ORB_*`, `MAX_STEREO_DEPTH`) require re-running
`prep_project.py` afterwards**, or the change never reaches pySLAM.

| Parameter | Default | Description |
|---|---|---|
| `DATASET_GRAYSCALE` | `False` | Grayscale conversion; also selects `image_0/1` vs `image_2/3` and the two colour flags |
| `IMAGE_WIDTH` | 1408 | Target image width (height auto-scaled) |
| `CAMERA_FPS` | 10.0 | Sensor capture rate (KITTI-360 is 10 Hz) |
| `NUM_FEATURES` | 2048 | Keypoints per frame (also written into the camera yaml) |
| `ORB_SCALE_FACTOR` | 1.2 | ORB pyramid scale factor between levels |
| `ORB_NUM_LEVELS` | 8 | ORB pyramid levels |
| `ORB_INI_TH_FAST` / `ORB_MIN_TH_FAST` | 20 / 7 | FAST initial / fallback corner threshold |
| `MAX_STEREO_DEPTH` | 50 m | Depth cutoff — baked into the yaml for stereo, passed at runtime for DE |
| `FILTER_SHADOW_POINTS` | `True` | Drop shadow points from neural depth (mono+depth path only) |
| `SAVE_MAP_DEFAULT` | `False` | Save `map.ply` by default |
| `TRAJ_FORMAT` | `"tum"` | Trajectory output format |
| `ATE_MAX_DT` | 0.05 s | Max timestamp gap for GT↔prediction association |
| `RPE_DELTA_SHORT` / `RPE_DELTA_LONG` | 0.5 s / 2.0 s | RPE windows — short-range accuracy / long-range drift |
| `TEST_FRAMES` | 10 | Frames per variant in `test_models.py` |
| `TEST_TIMEOUT_S` | 180 s | Per-variant timeout in `test_models.py` |
| `INTER_TEST_SLEEP` | 60 s | Cooldown in `test_models.py` |
| `INTER_RUN_SLEEP` | 180 s | Cooldown between batch runs (WSL2 page-cache reclaim) |
| `TRACKING_ABORT_STREAK` | 10 | Abort early if tracking is lost this many consecutive frames |
| `MAX_RUN_WALL_TIME_S` | 1200 s | In-loop wall-time limit per run (20 min) |
| `SETUP_GRACE_S` | 120 s | Margin added to get the batch runner's hard timeout |

---

## 8. Known Issues and Traps

From thesis §B.6, for anyone extending the project.

**Assigning to a pySLAM config object in Python proves nothing.** An early version of the runner
set ORB pyramid and FAST thresholds on pySLAM's global `Parameters` object. Those attribute names
do not exist there, so the assignments created new attributes nothing read — silent no-ops for as
long as they were in the code. Same for the map-memory caps, which live in the C++ core's own
header and cannot be reached from Python while that core is in use. The extractor settings now
travel through the generated camera YAML, which pySLAM's extractor actually reads back. The only
honest check is whether a value appears in the run's own log.

**Capping map memory does not work.** A memory-capping option intended to save RAM starved the
local map in hard sections and made tracking worse. Permanently disabled. In miniature it is the
capacity finding: constraining memory does not degrade this system gracefully, it degrades it as
tracking loss, wearing the wrong label on the way out.

**Any "is it done yet?" check must key on `dashboard.txt`.** The teardown-grace timer originally
accepted the final trajectory file as evidence a run had finished. The trajectory writer creates
that file *empty* at start-up, so the evidence appeared within seconds of launch and healthy long
runs were killed 30 seconds in and recorded as complete. Only the dashboard counts now, because
it is written exactly once, at the very end.

**Extractors that ignore the feature cap.** SIFT and LiftFeat emit 5,000–7,000 keypoints per
frame regardless of `NUM_FEATURES`, which is why their long runs ended in OOM kills at 14 GB. A
property of the models, not a bug in the bench — and what the skip list is for.

**A skip list that skips everything.** KAZE never executes (§2.1).

**`batch_runner --no-save-map` is wired wrong** ([batch_runner.py:557](src_pc/experiments/batch_runner.py#L557)) — recorded rather than quietly fixed, because the campaigns ran without it.

**Grayscale saves nothing.** OpenCV promotes grayscale images back to three channels on read, and
the study runs RGB throughout. The switch stays in the settings because it also selects the
directory layout and the two colour flags, and those three must move together.

**iBoW's detector queue can overflow** when the front end outruns the loop-closing process — a
real-time mismatch reported as a finding rather than fixed.

**The DINOv2-family closers must be lazy-initialised** to avoid `torch.hub` network calls inside
the spawned subprocess. CricaVPR additionally needs the xFormers build matching the installed
PyTorch.

**Two ATE numbers per run.** pySLAM's evaluation and this harness's metrics are computed
independently and need not agree exactly, since they associate poses differently and treat the
monocular case differently. The harness's number is the one reported.

### Stale strings in the code (harmless, but misleading)

- `test_models.py` and `first_run_pyslam.py` docstrings say `results/test_run/`; the real output
  is `results/test/`.
- `aggregate_all.py`'s module docstring says it writes `results_comparison/`, and
  `make_assets_offline.py`'s usage block passes `--in results_comparison`. The actual default is
  `results_aggregated/`.
- `paths.py` still exposes `RAW_IMAGES` / `RAW_POSES` / `RAW_CALIB` pointing at `dataset/raw/`
  without the `kitti360/` level, and `CUSTOM_EXPERIMENTS`. Nothing imports any of them;
  `prep_project.py` defines its own `RAW_KITTI*` constants.
