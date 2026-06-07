# Phase 3-5 Pilot Result

Date: 2026-06-07

Read this before continuing Phase 3-5.

## Implemented

Added:

- `src/training/train_phase3_5.py`
- `docs/multitask-phase3-5-pilot-report.md`

The script trains:

- `gat_t`
- `gat_tdte`
- `process_transformer`
- `multitask_gat_tdte`

Input artifacts:

- `output/experiments/multitask_prefix/**`

Output artifacts:

- `output/experiments/phase3_5_smoke/**`
- `output/experiments/phase3_5_pilot/**`

## Verification

Passed:

```bash
python3 -m py_compile src/training/train_phase3_5.py
```

Pilot command:

```bash
python3 src/training/train_phase3_5.py \
  --datasets BPI12w Helpdesk BPI13i BPI13c \
  --epochs 2 \
  --batch-size 64 \
  --hidden-dim 16 \
  --event-emb-dim 16 \
  --heads 2 \
  --transformer-layers 1 \
  --max-train-cases 300 \
  --max-val-cases 100 \
  --max-test-cases 100 \
  --output-dir output/experiments/phase3_5_pilot
```

## Pilot Metrics

| Dataset | Best next-activity pilot model | Note |
|---|---|---|
| BPI12w | ProcessTransformer, acc 0.3536 | GAT-T close at 0.3214 |
| Helpdesk | ProcessTransformer, acc 0.6970 | GAT-TDTE reached 0.5390 |
| BPI13i | GAT-T and Multi-task GAT-TDTE tied, acc 0.4203 | Multi-task has better macro F1 than GAT-T |
| BPI13c | ProcessTransformer, acc 0.4842 | Multi-task needs longer training |

Multi-task pilot additional metrics:

| Dataset | Remaining MAE Norm | Outcome Acc |
|---|---:|---:|
| BPI12w | 0.7141 | 0.2458 |
| Helpdesk | 0.6925 | 0.6710 |
| BPI13i | 0.2193 | 0.8249 |
| BPI13c | 0.6180 | 0.6538 |

## Important Caveat

This is a pilot/sanity run, not final thesis evidence.

Limits:

- 300 train cases, 100 validation cases, 100 test cases per dataset.
- 2 epochs.
- GAT-TDTE uses a runnable PyG approximation with time-decay and transition semantics as edge attributes because current `python3` lacks `torch_scatter`.

## Recommended Next Step

Run a full experiment under:

```text
output/experiments/phase3_5_full
```

Suggested command:

```bash
python3 src/training/train_phase3_5.py \
  --datasets BPI12w Helpdesk BPI13i BPI13c \
  --epochs 30 \
  --batch-size 64 \
  --hidden-dim 32 \
  --event-emb-dim 32 \
  --heads 2 \
  --transformer-layers 2 \
  --output-dir output/experiments/phase3_5_full
```

Before final claims, tune `alpha` and `beta` for Multi-task GAT-TDTE.
