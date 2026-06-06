# Multi-task GAT-TDTE Phase 1-2 Report

Date: 2026-06-06

## Objective

The project extends the original single-task next-activity PBPM setting into a multi-task setting:

- Next activity prediction
- Remaining time prediction
- Outcome/risk prediction

The confirmed experiment scope is:

- Main dataset: `BPI12w`
- Secondary validation dataset: `Helpdesk`
- Additional datasets: `BPI13i`, `BPI13c`
- Existing incompatible checkpoints may be treated as legacy artifacts; retrain when needed.

## Phase 1: Repository and Data Audit

### Completed Work

- Inspected the repository structure and model files.
- Identified existing model variants:
  - `src/GATConv.py`: GAT-T
  - `src/GATConvStatusEmb.py`: GAT with transition semantics
  - `src/GATConvTimeDecay.py`: GAT with time-decay attention
  - `src/GATConvTimeDecayStatusEmb.py`: GAT-TDTE
  - `src/PrefixEmbeddingGCN.py`: prefix GCN baseline
- Inspected preprocessing and app pipeline files:
  - `src/DataEncoder.py`
  - `src/DataProcess.ipynb`
  - `app/core/dataset_config.py`
  - `app/core/pipeline.py`
  - `app/core/evaluation.py`
  - `src/training/evaluate.py`
- Audited raw and preprocessed datasets.
- Audited saved checkpoints under `output/models`.
- Fixed and verified the Git LFS clean-filter issue.

### Key Findings

- The repo already contains strong GAT-T, GAT-TT, GAT-TD, and GAT-TDTE implementations.
- Training is mostly notebook-driven, so reusable training scripts are still needed for fair experiments.
- Existing evaluation covers next-activity metrics but not remaining-time or outcome/risk metrics.
- Current pipeline does not yet enforce one shared case-level train/validation/test split across all models.
- Some checkpoint names do not match current dataset encodings. For example, BPI12/BPI12w-named checkpoints have dimensions matching Helpdesk-like encoding, so they should not be treated as final thesis evidence without retraining or reconstructing old preprocessing.

### Phase 1 Artifacts

- `prompt-plan/multitask-phase1-audit.md`
- `prompt-plan/multitask-phase1-decisions.md`

## Phase 2: Labels and Splits

### Completed Work

Implemented a reusable prefix dataset builder:

```text
src/training/build_multitask_prefix_dataset.py
```

Command run:

```bash
python3 src/training/build_multitask_prefix_dataset.py --overwrite
```

Generated outputs:

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

### Label Definitions

For each prefix ending at event index `t`:

```text
prefix = events[0:t]
next_activity = events[t + 1], or EOS for terminal prefix
remaining_time_seconds = case_end_time - current_event_time
remaining_time_norm = normalized using train-prefix mean/std only
outcome_label = final-event outcome or duration-risk label
```

Outcome/risk did not require extra raw data, but it required a derived label-generation step.

### Outcome/Risk Strategy

The builder chooses the label strategy automatically:

- Use `final_event_outcome` when final events have meaningful class diversity.
- Otherwise use binary `duration_p75_risk`, fitted from train cases only.

| Dataset | Outcome/risk strategy |
|---|---|
| `BPI12w` | `final_event_outcome` |
| `Helpdesk` | `duration_p75_risk` |
| `BPI13i` | `final_event_outcome` |
| `BPI13c` | `duration_p75_risk` |

### Dataset Results

| Dataset | Cases | Events | Prefixes | Train cases | Val cases | Test cases |
|---|---:|---:|---:|---:|---:|---:|
| `BPI12w` | 9658 | 170107 | 170107 | 6760 | 1448 | 1450 |
| `Helpdesk` | 4580 | 21221 | 21221 | 3206 | 687 | 687 |
| `BPI13i` | 7553 | 65532 | 65532 | 5287 | 1132 | 1134 |
| `BPI13c` | 1486 | 6659 | 6659 | 1040 | 222 | 224 |

### Verification

Checks passed:

- Train/validation/test case IDs have zero overlap.
- Remaining-time normalization is fitted on train prefixes only.
- Train normalized remaining-time mean is approximately `0` and std is approximately `1`.
- Number of terminal prefixes equals number of cases.
- Prefix max index aligns with event count per case after normalizing case IDs to string.
- `python3 -m py_compile src/training/build_multitask_prefix_dataset.py` passed.
- `git lfs fsck` returned `Git LFS fsck OK`.

### Phase 2 Artifacts

- `src/training/build_multitask_prefix_dataset.py`
- `output/experiments/multitask_prefix/**`
- `prompt-plan/multitask-phase2-label-split-result.md`

## Google Drive Storage Assessment

Target Drive folder provided:

```text
https://drive.google.com/drive/folders/1TD6x3yPio30f2-1fqA7SghH3SpVYRTiw
```

Local size snapshot:

| Path | Size |
|---|---:|
| `data/` | 30 MB |
| `output/*.csv` | about 27 MB |
| `output/experiments/multitask_prefix/` | 38 MB |
| `output/models/` | 41 MB |
| `output/statis/` | 56 KB |

Recommendation:

- Keep source code and lightweight metadata in GitHub.
- Keep raw/preprocessed datasets in both GitHub and Drive if the current thesis workflow depends on Colab/Drive access.
- Store trained model checkpoints on Drive, and keep only selected milestone checkpoints in Git LFS.
- Store final experiment outputs on Drive, especially:
  - checkpoints
  - metrics JSON/CSV
  - plots
  - attention visualizations
  - large generated prefix datasets
- Keep small reproducibility metadata in GitHub:
  - script configs
  - split metadata JSON
  - result summaries
  - report markdown files

Current GitHub Actions workflow uploads:

- `run_colab.ipynb`
- `data/**`
- `output/*.csv`
- `src/**/*.ipynb`
- `src/**/*.py`
- `config/**` if present
- `requirements*.txt`
- `pyproject.toml`

It does not currently upload:

- `output/models/**`
- `output/statis/**`
- `output/experiments/**`
- `docs/**`
- `prompt-plan/**`

Recommended Drive layout:

```text
gnn-eventlogs-2026/
  src/
  data/
  output/
    base_csv/
    experiments/
      multitask_prefix/
      phase3_gat_reproduction/
      phase5_multitask_gat_tdte/
    models/
      legacy/
      phase3/
      phase5/
    statis/
    figures/
  docs/
  reports/
```

Recommended workflow update before long training:

- Add `output/experiments/**/*.json` and result summaries to Drive sync.
- Add `output/statis/**` and plot folders to Drive sync.
- Do not automatically upload every checkpoint unless naming and retention rules are defined.
- For checkpoints, use a retention policy such as:
  - best validation checkpoint per dataset/model/seed
  - final checkpoint per run
  - no per-epoch checkpoint uploads unless debugging

## Next Step

Proceed to Phase 3:

- Reproduce/retrain GAT-T and GAT-TDTE with the shared splits from `output/experiments/multitask_prefix`.
- Save configs, metrics, and checkpoints under structured run directories.
- Treat existing checkpoints as legacy/smoke-test artifacts unless their original preprocessing is reconstructed.
