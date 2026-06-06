# Phase 1 Human Decisions

Date: 2026-06-06

These decisions extend `prompt-plan/multitask-phase1-audit.md` and should be read before starting Phase 2.

## Dataset Scope

Confirmed datasets:

- Main dataset: `BPI12w`
- Validation/secondary dataset: `Helpdesk`
- Additional training/evaluation datasets: `BPI13i`, `BPI13c`

If current checkpoints do not match the selected dataset encodings, retrain new models instead of forcing incompatible checkpoints.

## Task Scope

Use multi-task prediction with:

1. `next_activity`
2. `remaining_time`
3. `outcome_or_risk`

## Outcome/Risk Label Decision

Outcome/risk requires an explicit derived label before training.
No extra raw data is strictly required if a meaningful label can be derived from existing case traces or case attributes.

Recommended label policy for Phase 2:

- `BPI12w`: derive outcome from final event/status where meaningful; otherwise use a delay/risk proxy based on remaining/case duration quantiles.
- `Helpdesk`: derive risk from case duration/SLA-like threshold because `status` is constant in `output/helpdesk.csv`; do not use `case:variant-index` as an input if it is used to define an outcome label.
- `BPI13i` and `BPI13c`: inspect final statuses and case-level fields; use final-status outcome only if class distribution is meaningful, otherwise use duration-based risk.

Default fallback:

```text
risk = 1 if case_duration_seconds >= train_split_duration_p75 else 0
```

Important:

- Thresholds must be fitted on train split only.
- Labels must be derived after case-level split metadata is created.
- Final-only fields must not be used as prefix input features if they reveal outcome/risk.

## Checkpoint Policy

Existing checkpoints may be treated as legacy/smoke-test artifacts.
For thesis experiments, train fresh checkpoints on the confirmed datasets and shared splits when needed.

## Phase 2 Start Conditions

Next phase should:

1. Build shared case-level train/validation/test splits for `BPI12w`, `Helpdesk`, `BPI13i`, and `BPI13c`.
2. Generate prefix samples after splitting cases.
3. Add labels for next activity, remaining time, and outcome/risk.
4. Save split IDs, label metadata, risk threshold metadata, and dataset statistics under `output/experiments/`.

## Git LFS Status

The earlier `git status` failure was caused by Git LFS clean-filter needing to write a temp file under `.git/lfs/tmp`.
After running `git status --short` with local Git permission, normal `git status --short` works again.

Verification:

```text
git lfs fsck -> Git LFS fsck OK
```

Tracked LFS checkpoint files:

- `output/models/1.pt`
- `output/models/1BPI13i_prefix15_es.pt`
- `output/models/BPI12_timeedge_es.pt`
- `output/models/BPI12w_transedge_es.pt`
- `output/models/BPI12w_transedgedecay_es.pt`
