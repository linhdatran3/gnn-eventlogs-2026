# Multi-task GAT-TDTE Phase 3-5 Pilot Report

Date: 2026-06-07

## Scope

This report records the first executable implementation for Phases 3-5:

- Phase 3: reproduce GAT baselines on shared splits
- Phase 4: implement ProcessTransformer baseline
- Phase 5: implement proposed Multi-task GAT-TDTE

This is a **pilot run**, not the final thesis training run. It uses limited cases and 2 epochs to verify that the full training/evaluation/checkpoint pipeline works across all confirmed datasets.

## Implemented Training Entrypoint

Added:

```text
src/training/train_phase3_5.py
```

Supported model names:

- `gat_t`
- `gat_tdte`
- `process_transformer`
- `multitask_gat_tdte`

Important environment note:

- The current `python3` environment has PyTorch and PyG.
- It does not have `torch_scatter`.
- Because the original `src/GATConvTimeDecayStatusEmb.py` imports `torch_scatter` at top level, the pilot script uses a self-contained PyG implementation rather than importing the original model file directly.

## Model Implementation Notes

### GAT-T

Single-task next-activity prediction.

Input:

- Node features from event/case attributes
- Event ID embeddings
- Chain graph over case events
- Edge attribute: scaled local time difference only

Output:

- Next activity logits per prefix/event node

### GAT-TDTE

Single-task next-activity prediction.

Input:

- Same node/event inputs as GAT-T
- Edge attributes:
  - scaled local time difference
  - time-decay value
  - transition-type one-hot semantics

Output:

- Next activity logits per prefix/event node

### ProcessTransformer

Sequence baseline for next-activity prediction.

Input:

- Event ID sequence
- Positional embeddings
- Causal Transformer mask

Output:

- Next activity logits per prefix position

### Multi-task GAT-TDTE

Proposed model.

Shared GAT-TDTE-style encoder with three heads:

- Next activity classifier
- Remaining-time regressor
- Outcome/risk classifier

Loss:

```text
CE(next_activity)
+ alpha * SmoothL1(remaining_time_norm)
+ beta * CE(outcome_or_risk)
```

Pilot values:

```text
alpha = 0.5
beta = 0.5
```

## Verification

Passed:

```bash
python3 -m py_compile src/training/train_phase3_5.py
```

Smoke command:

```bash
python3 src/training/train_phase3_5.py \
  --datasets Helpdesk \
  --epochs 1 \
  --batch-size 64 \
  --hidden-dim 16 \
  --event-emb-dim 16 \
  --heads 2 \
  --transformer-layers 1 \
  --max-train-cases 80 \
  --max-val-cases 30 \
  --max-test-cases 30 \
  --output-dir output/experiments/phase3_5_smoke
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

## Pilot Results

| Dataset | Model | Next Acc | Macro F1 | Remaining MAE Norm | Outcome Acc |
|---|---|---:|---:|---:|---:|
| BPI12w | `gat_t` | 0.3214 | 0.1521 | - | - |
| BPI12w | `gat_tdte` | 0.0595 | 0.0406 | - | - |
| BPI12w | `process_transformer` | 0.3536 | 0.1798 | - | - |
| BPI12w | `multitask_gat_tdte` | 0.2018 | 0.0871 | 0.7141 | 0.2458 |
| Helpdesk | `gat_t` | 0.3550 | 0.1603 | - | - |
| Helpdesk | `gat_tdte` | 0.5390 | 0.2286 | - | - |
| Helpdesk | `process_transformer` | 0.6970 | 0.2742 | - | - |
| Helpdesk | `multitask_gat_tdte` | 0.3810 | 0.1532 | 0.6925 | 0.6710 |
| BPI13i | `gat_t` | 0.4203 | 0.1203 | - | - |
| BPI13i | `gat_tdte` | 0.3469 | 0.0848 | - | - |
| BPI13i | `process_transformer` | 0.3864 | 0.1420 | - | - |
| BPI13i | `multitask_gat_tdte` | 0.4203 | 0.1357 | 0.2193 | 0.8249 |
| BPI13c | `gat_t` | 0.3348 | 0.1411 | - | - |
| BPI13c | `gat_tdte` | 0.2579 | 0.0819 | - | - |
| BPI13c | `process_transformer` | 0.4842 | 0.2292 | - | - |
| BPI13c | `multitask_gat_tdte` | 0.2330 | 0.0755 | 0.6180 | 0.6538 |

Raw artifacts:

```text
output/experiments/phase3_5_smoke/
output/experiments/phase3_5_pilot/
```

Each model run contains:

```text
metrics.json
best_model.pt
```

## Interpretation

Do not use this pilot as final model comparison evidence.

What the pilot does show:

- The shared split artifacts from Phase 2 are usable by all Phase 3-5 models.
- The training loop, evaluation metrics, and checkpoint saving work end to end.
- ProcessTransformer is already strong in short training, especially on Helpdesk and BPI13c.
- GAT-TDTE is promising on Helpdesk in this pilot but underperforms on BPI12w, BPI13i, and BPI13c with only 2 epochs and 300 train cases.
- Multi-task GAT-TDTE learns all three heads, but next-activity performance needs longer training and task-weight tuning.

## Risks and Follow-up

Known limitations:

- Pilot uses only 300 train cases, 100 validation cases, and 100 test cases per dataset.
- Pilot uses 2 epochs only.
- GAT-TDTE implementation is a runnable PyG approximation using time-decay and transition semantics as edge attributes, not a direct import of the original `TimeAwareETGATConv` class because `torch_scatter` is missing in the current environment.
- Multi-task task weights are fixed at `alpha=0.5`, `beta=0.5`.

Recommended next full run:

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

Recommended ablations before final thesis claims:

- Multi-task GAT-TDTE with `alpha` in `{0.1, 0.5, 1.0}`
- Multi-task GAT-TDTE with `beta` in `{0.1, 0.5, 1.0}`
- Longer single-task GAT-TDTE training before comparing against multi-task
- Optional installation of `torch_scatter` in the target training environment to enable exact original model reuse
