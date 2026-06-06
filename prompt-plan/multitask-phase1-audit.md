# Phase 1 Audit: Multi-task GAT-TDTE

Date: 2026-06-06

## Scope

This audit follows `prompt-plan/multitask-gat-tdte-execution-plan.md`.
No source code was modified in Phase 1.

## Repository Structure

Core model files:

- `src/GATConv.py`: GAT-T style full-trace GAT with time-difference edge attributes.
- `src/GATConvStatusEmb.py`: transition-semantics GAT using edge type embeddings.
- `src/GATConvTimeDecay.py`: time-aware GAT with learned time-decay attention.
- `src/GATConvTimeDecayStatusEmb.py`: GAT-TDTE with time decay plus transition edge semantics.
- `src/PrefixEmbeddingGCN.py`: prefix-based GCN classifier.

Preprocessing and pipeline files:

- `src/DataEncoder.py`: event/sequence encoding, next-event labels, time differences, node times, transition edge labels.
- `src/DataProcess.ipynb`: original preprocessing notebook, especially BPI12 transformations.
- `app/core/dataset_config.py`: dataset and model registry used by the app and CLI evaluation.
- `app/core/pipeline.py`: inference-time encoding pipeline for GAT and prefix models.
- `app/core/evaluation.py`: shared Top-1, Top-K, Macro-F1, Weighted-F1 evaluation helpers.
- `src/training/evaluate.py`: CLI evaluation entrypoint for saved checkpoints.

## Dataset Inventory

### Raw datasets

| File | Rows | Cases | Key columns |
|---|---:|---:|---|
| `data/BPI_Challenge_2012.csv` | 262200 | 13087 | `case:concept:name`, `concept:name`, `lifecycle:transition`, `time:timestamp`, `org:resource`, `case:AMOUNT_REQ` |
| `data/Helpdesk.csv` | 21348 | 4580 | `case:concept:name`, `concept:name`, `Activity`, `lifecycle:transition`, `time:timestamp`, `org:resource`, `case:variant-index` |

### Preprocessed datasets

| File | Rows | Cases | Event classes | Notes |
|---|---:|---:|---:|---|
| `output/BPI12.csv` | 262200 | 13087 | 36 | Uses `event = event_label + status`; min/median/max events = 3/11/175 |
| `output/BPI12w.csv` | 170107 | 9658 | 19 | W-only subset; min/median/max events = 2/14/156 |
| `output/helpdesk.csv` | 21221 | 4580 | 14 | Status is constant; min/median/max events = 2/4/15 |
| `output/BPI13i.csv` | 65533 | 7554 | 13 | Contains cases with 1 event; configured min size is 2 |
| `output/BPI13c.csv` | 6660 | 1487 | 7 | Contains cases with 1 event; configured min size is 2 |

Standardized columns already available:

- Case id: `sequence`
- Activity/event label: `event`
- Timestamp: `time`
- Event resource/category-like fields: `ec*`
- Case-level categorical fields: `sc*`
- Case-level numerical fields: `sn*`
- Lifecycle/status: `status` where available

## Current Label Behavior

Current `encode_label_event()` creates full-trace next-event labels:

```text
input events = all events in a case
target       = shifted events + EOS
```

This is useful for single-task next activity reproduction, but Phase 2 needs a shared prefix-sample dataset:

```text
prefix [event_1 ... event_t]
next activity = event_{t+1}
remaining time = case_end_time - time_t
```

The current code does not yet provide a reusable case-level train/validation/test split for all models.

## Checkpoints

Available checkpoints:

| File | Inferred model | Feature dim | Output dim | Epoch | Saved test acc |
|---|---|---:|---:|---:|---:|
| `output/models/BPI12_timeedge_es.pt` | `DualGATModel` | 248 | 15 | 4 | 0.9244 |
| `output/models/BPI12w_transedge_es.pt` | `DualGAT2EdgesModel` | 248 | 15 | 3 | 0.9215 |
| `output/models/BPI12w_transedgedecay_es.pt` | `DualGATTimeAwareETModel` | 248 | 15 | 6 | 0.7407 |
| `output/models/1.pt` | `DualGATTimeAwareModel` | 248 | 15 | 7 | 0.7401 |
| `output/models/1BPI13i_prefix15_es.pt` | `PrefixGCNClassifier` | 60 | 20 | 9 | 0.7288 |

Important compatibility issue:

- Current `output/helpdesk.csv` encodes to 248 event features and 15 output classes.
- Current `output/BPI12w.csv` encodes to 61 event features and 20 output classes.
- Current `output/BPI12.csv` encodes to 70 event features and 37 output classes.
- Therefore, the checkpoint names `BPI12*` do not match the current BPI12/BPI12w encoding dimensions.
- Existing `week12_helpdesk_*_eval.json` files evaluate these checkpoints on Helpdesk-compatible dimensions, so they should be treated as smoke tests, not final thesis evidence.

## Phase 1 Findings

1. The repo already has strong GAT-T, GAT-TT, GAT-TD, and GAT-TDTE source implementations.
2. Training is mostly notebook-driven; standardized reusable training scripts are still needed.
3. Existing CLI evaluation covers next activity metrics but not remaining-time regression.
4. Dataset configs exist for BPI12, BPI12w, Helpdesk, BPI13i, and BPI13c.
5. The current app/inference pipeline is not enough for fair experiments because it does not enforce a shared case-level split.
6. Saved checkpoint filenames and current encoded dataset dimensions need human confirmation before reuse.
7. Outcome/risk labels are not clearly defined yet; remaining time is feasible for all datasets with timestamps.

## Recommended Phase 2 Direction

Use strong reasoning model for design and implementation.

Smallest safe next step:

1. Create a reusable prefix dataset builder with case-level split.
2. Generate labels for next activity and remaining time.
3. Normalize remaining time using train split statistics only.
4. Store split IDs and label metadata under `output/experiments/`.
5. Keep outcome/risk disabled unless the human confirms a meaningful label definition.

## Human Confirmation Required Before Phase 2

Please confirm:

1. Primary dataset: use `BPI12w` as main and `Helpdesk` as validation, or switch main to `Helpdesk` because existing checkpoints match it?
2. Additional datasets: include `BPI13i`/`BPI13c` now, or defer until the pipeline is stable?
3. Task scope: use two tasks only (`next_activity`, `remaining_time`) for now, or define an outcome/risk label?
4. Existing checkpoints: treat them as legacy/smoke-test artifacts, or attempt to reconstruct the exact old preprocessing that produced feature dim 248 and output dim 15?
