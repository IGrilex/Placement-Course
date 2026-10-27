# all_conditions.csv

One file, the whole campaign. 3217 rows: 2136 run-level,
1073 cell-level, 8 condition-level.

## Read this first

Every row carries a `level`. **Filter on it before doing anything else.**
The three levels describe the same runs at three degrees of aggregation, so
summing a count column across the unfiltered file counts every run at least
twice.

| level | one row is | use it for |
|---|---|---|
| `run` | one individual execution | repetition spread, provenance, per-run failure detail |
| `cell` | one (condition, mode, sequence, variant) | every figure and table in the thesis |
| `condition` | one hardware condition | the headline outcome accounting |

## Keys

- `condition`, `rung`, `cond_label`, `cpu`, `threads`, `ram_budget` -- the
  hardware. `rung` is position on the ladder, 1 = first.
- `mode` -- `SLAM` (loop closure on) or `VO` (loop closure off).
- `test_id` -- sequence. `AVG` rows are roll-ups across sequences and should be
  excluded from anything that also reads the individual sequences.
- `exp_type`, `variant`, `short_name`, `family`, `sensor` -- what was run.

## Provenance (level=run only)

- `measured` -- runner log entry AND run directory both present.
- `recovered` -- directory only. The run happened; the log missed it. Metrics
  are real.
- `orphaned` -- log entry only, directory gone. **Carries no metrics.** Present
  so run counts reconcile and the gap is visible. Never aggregate these.

## Repetition balance (level=cell only)

- `n_total` -- repetitions actually run. Print it beside every median.
- `n_expected` -- the modal repetition count for that (condition, mode).
- `rep_status` -- `full` / `under` / `over` / `probe`.
- `is_probe` -- single exploratory run where the norm is three or more. These
  are evidence for the skip list, not measurements of the method, and are
  excluded from every aggregate. `under` and `over` cells ARE included; their
  `n_total` travels with them.

## Metric suffixes (level=cell)

- `_succ_*` -- over completed runs only. All accuracy keys take this form only,
  because a run that died at 30 % has its ATE scored on the easy opening
  stretch, so pooling it with completers makes a variant look better the
  earlier it fails.
- `_all_*` -- over every run that produced data. Machine load and throughput
  use this: an OOM-killed run still truthfully reports what it was holding.

## Relative columns

`rel_<column>` is that column divided by the ORB2 baseline's value on the
**same (condition, mode, sequence)**. Absolute figures from this bench are
partly a property of the bench; the ratios are the part that should survive a
native reimplementation.

- Ratio of medians, not median of ratios.
- Accuracy ratios are **null**, not zero, where the baseline itself did not
  complete on that sequence. `baseline_completed` says which rows those are.
- Normalised per sequence, because the sequences differ from each other by more
  than the methods differ from each other.

## Outcome labels

`success`, `lost`, `timeout`, `oom`, `error`. They are not interchangeable:

- `lost` is **algorithm-attributable** -- the tracker ran out of usable
  correspondences.
- `timeout` and `oom` are **budget-attributable** -- the same run on a larger
  machine would have continued.

Component claims should be scored on the algorithm-attributable subset
(`n_success / (n_success + n_lost)`); hardware claims on the budget-attributable
failures (`n_timeout + n_oom`). At the largest budget the two coincide, because
nothing dies of memory there.
