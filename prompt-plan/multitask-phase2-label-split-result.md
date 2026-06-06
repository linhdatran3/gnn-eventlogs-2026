# Phase 2 Result: Labels and Splits

Date: 2026-06-06

This result follows:

- `prompt-plan/multitask-gat-tdte-execution-plan.md`
- `prompt-plan/multitask-phase1-audit.md`
- `prompt-plan/multitask-phase1-decisions.md`

## Implemented Artifact

Added reusable builder:

- `src/training/build_multitask_prefix_dataset.py`

Command used:

```bash
python3 src/training/build_multitask_prefix_dataset.py --overwrite
```

Outputs:

```text
output/experiments/multitask_prefix/
  summary.json
  BPI12w/
    events.csv
    cases.csv
    prefixes.csv
    metadata.json
  Helpdesk/
    events.csv
    cases.csv
    prefixes.csv
    metadata.json
  BPI13i/
    events.csv
    cases.csv
    prefixes.csv
    metadata.json
  BPI13c/
    events.csv
    cases.csv
    prefixes.csv
    metadata.json
```

## Label Definitions

For each case and each prefix ending at event index `t`:

```text
prefix = events[0:t]
next_activity = events[t + 1], or EOS for terminal prefix
remaining_time_seconds = case_end_time - current_event_time
remaining_time_norm = normalized using train-prefix mean/std only
outcome_label = dataset-specific final outcome or duration risk
```

## Outcome/Risk Policy

Automatic policy:

- If final event outcome has at least two meaningful classes in train split and top class share <= 0.95, use `final_event_outcome`.
- Otherwise use binary `duration_p75_risk`, fitted on train cases only.

Result:

| Dataset | Outcome strategy |
|---|---|
| BPI12w | `final_event_outcome` |
| Helpdesk | `duration_p75_risk` |
| BPI13i | `final_event_outcome` |
| BPI13c | `duration_p75_risk` |

No extra raw data was required. A label generation step was required and is now implemented.

## Dataset Results

| Dataset | Cases | Events | Prefixes | Train cases | Val cases | Test cases |
|---|---:|---:|---:|---:|---:|---:|
| BPI12w | 9658 | 170107 | 170107 | 6760 | 1448 | 1450 |
| Helpdesk | 4580 | 21221 | 21221 | 3206 | 687 | 687 |
| BPI13i | 7553 | 65532 | 65532 | 5287 | 1132 | 1134 |
| BPI13c | 1486 | 6659 | 6659 | 1040 | 222 | 224 |

## Verification

Checks passed:

- Train/val/test case IDs have zero overlap for all datasets.
- Remaining-time normalized values have train mean approximately 0 and train std approximately 1.
- Number of terminal prefixes equals number of cases for all datasets.
- Prefix max index aligns with event count per case after normalizing case IDs to string.
- `git lfs fsck` returned `Git LFS fsck OK`.

## Leakage Controls

- Case-level split is created before prefix samples are used.
- Remaining-time normalization is fitted on train prefixes only.
- Duration-risk threshold is fitted on train cases only.
- Prefix rows store `prefix_end_index`; future events are not materialized into prefix inputs.
- If a future phase uses outcome labels derived from final event, final-only fields must not be included as prefix input features.

## Next Phase

Phase 3 should retrain/reproduce GAT-T and GAT-TDTE using these shared splits.
Existing checkpoints can remain legacy/smoke-test artifacts because current BPI12w encoding does not match the old checkpoint dimensions.
