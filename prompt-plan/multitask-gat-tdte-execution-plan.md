# Execution Plan: Multi-task GAT-TDTE for Predictive Business Process Monitoring

## 0. Project Goal

Phat trien luan van tu paper:

- Time-Aware and Transition-Semantic Graph Neural Networks for Interpretable Predictive Business Process Monitoring
- Paper: https://arxiv.org/abs/2508.09527

Muc tieu la mo rong bai toan tu **single-task next activity prediction** sang **multi-task predictive process monitoring**, gom:

- Next activity prediction
- Remaining time prediction
- Optional: outcome/risk prediction neu label du ro

Ca model can so sanh:

- Baseline 1: ProcessTransformer
- Baseline 2: GAT-T
- Baseline 3: GAT-TDTE
- Proposed: Multi-task GAT-TDTE

Datasets du kien:

- Main: BPI12 / BPI12w
- Additional validation: Helpdesk
- Optional: BPI13i / BPI13c neu pipeline on

## 1. Research Questions

### RQ1

Graph-based representation co cai thien PBPM so voi sequence-based Transformer khong?

Compare:

```text
ProcessTransformer vs GAT-T
```

### RQ2

Time-decay va transition semantics co cai thien next activity prediction so voi GAT co ban khong?

Compare:

```text
GAT-T vs GAT-TDTE
```

### RQ3

Multi-task GAT-TDTE co cai thien kha nang du doan nhieu khia canh cua mot running case khong?

Compare:

```text
GAT-TDTE vs Multi-task GAT-TDTE
```

### RQ4

Multi-task learning co tao trade-off giua next activity accuracy va remaining time prediction khong?

Analyze:

```text
Next activity metrics
Remaining time metrics
Loss behavior
Dataset-specific behavior
```

## 2. AI Model Usage Recommendation

### Use strongest model, e.g. GPT-5.5 / best reasoning model, for:

- Reading and understanding the existing repo architecture
- Designing experiment protocol
- Refactoring notebooks into reusable scripts
- Implementing Multi-task GAT-TDTE
- Debugging tensor shape, loss masking, graph batching
- Interpreting unexpected experiment results
- Writing thesis narrative and research discussion

### Use normal/cheaper model for:

- Generating routine scripts
- Cleaning markdown/logs
- Formatting tables
- Creating plotting utilities
- Converting metrics JSON/CSV into summary tables
- Writing basic README sections
- Running repeated experiment variants once logic is stable

### Use human review for:

- Deciding final task scope
- Choosing datasets
- Confirming label definitions
- Reading result tables
- Deciding final research narrative
- Determining whether unexpected results are bugs or valid findings

## 3. Phase 1: Repository and Data Audit

### Goal

Understand current source code, datasets, preprocessing, model variants, and checkpoints.

### AI Tasks

1. Inspect repo structure.
2. Identify current model files:
   - `GATConv.py`
   - `GATConvStatusEmb.py`
   - `GATConvTimeDecay.py`
   - `GATConvTimeDecayStatusEmb.py`
   - `PrefixEmbeddingGCN.py`
3. Identify preprocessing files:
   - `DataEncoder.py`
   - `DataProcess.ipynb`
   - app pipeline files if useful
4. Inspect datasets:
   - `data/BPI_Challenge_2012.csv`
   - `data/Helpdesk.csv`
   - files under `output/`
5. Summarize available columns:
   - case id
   - activity/event
   - timestamp
   - resource
   - status/lifecycle
   - case-level attributes
6. Check available checkpoints under `output/models`.

### Human Review

Decide:

```text
Use BPI12/BPI12w as main dataset?
Use Helpdesk as validation dataset?
Include BPI13i/BPI13c now or later?
```

### Suggested AI Model

Strong model for repo understanding.

## 4. Phase 2: Define Labels and Splits

### Goal

Create consistent labels for all models.

### Required Labels

For each prefix of a case:

```text
Input prefix = [event_1, event_2, ..., event_t]
Label 1 = next activity/event
Label 2 = remaining time
Label 3 = outcome/risk optional
```

### AI Tasks

1. Generate prefix samples from each case.
2. Compute next activity label.
3. Compute remaining time:

```text
remaining_time = case_end_time - prefix_current_time
```

4. Normalize remaining time for training.
5. Decide whether outcome/risk is feasible:
   - BPI12: possible from final event/status, but must verify.
   - Helpdesk: possible from final status/variant only if meaningful.
6. Create train/validation/test split.
7. Ensure same split is used for all models.

### Human Review

Decide:

```text
If outcome/risk label is unclear:
    use only next activity + remaining time
else:
    include outcome/risk as optional third task
```

### Important Checks

- No data leakage from future events.
- Prefix must not include event after prediction point.
- Case-level attributes must not reveal final outcome if used as input.
- Split should preferably be case-level, not prefix-level, to avoid leakage.

### Suggested AI Model

Strong model for label/split design.

## 5. Phase 3: Reproduce Existing GAT Baselines

### Goal

Reproduce existing single-task GNN results before adding proposed model.

### Models

```text
Baseline 2: GAT-T
Baseline 3: GAT-TDTE
```

### AI Tasks

1. Create or refactor training script for GAT-T.
2. Create or refactor training script for GAT-TDTE.
3. Use same dataset split.
4. Log:
   - config
   - seed
   - train loss
   - validation loss
   - test metrics
5. Save results to structured files:
   - CSV or JSON
6. Compare with existing checkpoint results if available.

### Metrics

For next activity:

```text
Accuracy
Top-k Accuracy
Macro F1
Weighted F1
Optional: DL score if already implemented
```

### Human Review

Check:

```text
Are reproduced results close to expected?
Is GAT-TDTE better than GAT-T?
Are metrics computed correctly?
```

### Decision Rules

If GAT-TDTE result is unexpectedly poor:

```text
Check preprocessing.
Check label encoding.
Check train/test split.
Check checkpoint compatibility.
Check edge_time_diff and edge_type construction.
```

### Suggested AI Model

Strong model for first implementation and debugging.
Normal model for repeated runs after stable.

## 6. Phase 4: Implement ProcessTransformer Baseline

### Goal

Build a modern sequence-based baseline.

### Motivation

ProcessTransformer represents event logs as sequences and uses self-attention to learn which previous events are important for prediction.

### AI Tasks

1. Implement a simple ProcessTransformer-like model.
2. Input:
   - event/activity embedding
   - optional resource/status embedding
   - positional encoding
   - optional time feature
3. Output:
   - next activity prediction
   - optional remaining time if implemented as multi-output baseline
4. Train on same split as GAT models.
5. Evaluate with same metrics.

### Minimum Version

```text
Input: activity sequence + position
Output: next activity
```

### Better Version

```text
Input: activity + resource/status + time gap
Output: next activity + remaining time
```

### Human Review

Check fairness:

```text
Does ProcessTransformer use same information as GAT models?
Is it trained on same split?
Is it overpowered or underpowered compared to GAT?
```

### Decision Rules

If ProcessTransformer outperforms GAT-TDTE:

```text
Do not treat as failure.
Check whether dataset is mostly linear/sequential.
Check GAT hyperparameters.
Check graph construction.
Discuss that Transformer is strong for sequence-dominant logs.
Position GAT-TDTE as better for graph semantics and interpretability.
```

### Suggested AI Model

Strong model for initial implementation.
Normal model for simple training runs and table generation.

## 7. Phase 5: Implement Proposed Multi-task GAT-TDTE

### Goal

Extend GAT-TDTE from single-task next activity prediction to multi-task prediction.

### Architecture

```text
GAT-TDTE Encoder
        |
        |-- Head 1: next activity classifier
        |-- Head 2: remaining time regressor
        |-- Head 3: outcome/risk classifier optional
```

### AI Tasks

1. Refactor GAT-TDTE into:
   - encoder
   - task heads
2. Add remaining time regression head.
3. Add optional outcome/risk head.
4. Implement multi-task loss:

```text
Loss = CE(next_activity)
     + alpha * RegressionLoss(remaining_time)
     + beta * CE/BCE(outcome)
```

5. Support configurable task weights:

```text
alpha in [0.1, 0.5, 1.0]
beta in [0.1, 0.5, 1.0]
```

6. Log task-specific metrics separately.
7. Save best model by validation score.

### Metrics

Next activity:

```text
Accuracy
Top-k Accuracy
Macro F1
Weighted F1
```

Remaining time:

```text
MAE
RMSE
Normalized MAE if remaining time is normalized
```

Outcome/risk optional:

```text
Accuracy
F1
AUC if binary
```

### Human Review

Check:

```text
Does remaining time loss dominate?
Does next activity accuracy drop too much?
Do task weights need adjustment?
Should outcome/risk be removed?
```

### Decision Rules

If proposed improves both tasks:

```text
Strong positive result.
Proceed to ablation and case study.
```

If proposed improves remaining time but next activity drops slightly:

```text
Frame as trade-off.
Useful for real monitoring where multiple predictions matter.
```

If proposed performs worse on all tasks:

```text
Try task weight tuning.
Try single-task pretraining then multi-task fine-tuning.
Try freezing encoder for early epochs.
Check label scale and data leakage.
```

### Suggested AI Model

Strongest model recommended.

## 8. Phase 6: Ablation Study

### Goal

Show which components contribute to performance.

### Experiments

Minimum ablation:

```text
GAT-T
GAT-TDTE
Multi-task GAT-TDTE
```

Better ablation:

```text
GAT-T
GAT-T + time decay
GAT-T + transition semantics
GAT-TDTE
Multi-task GAT-TDTE
```

Optional:

```text
Multi-task GAT-TDTE without time decay
Multi-task GAT-TDTE without transition semantics
Multi-task GAT-TDTE with different alpha/beta
```

### AI Tasks

1. Run ablation experiments.
2. Collect all results into one table.
3. Generate plots:
   - accuracy comparison
   - MAE comparison
   - task trade-off chart
4. Summarize findings.

### Human Review

Answer:

```text
Does time-decay help?
Does transition semantics help?
Does multi-task learning help?
Which dataset benefits most?
```

### Suggested AI Model

Normal model for running variants.
Strong model for interpreting patterns.

## 9. Phase 7: Interpretability and Case Studies

### Goal

Use attention analysis to explain model behavior.

### AI Tasks

1. Extract attention weights from GAT-TDTE and Multi-task GAT-TDTE.
2. Generate:
   - attention heatmap
   - top attended events
   - top attended transitions
   - critical windows
3. Select case examples:
   - correct next activity prediction
   - wrong next activity prediction
   - high remaining time case
   - long/complex trace
4. Compare attention behavior before and after multi-task learning.

### Human Review

Assess business meaning:

```text
Does the model focus on reasonable events?
Are high-attention transitions meaningful?
Does attention help explain delay/risk?
```

### Suggested AI Model

Strong model for case-study interpretation.
Normal model for plotting.

## 10. Phase 8: Final Analysis and Thesis Narrative

### Goal

Convert experimental results into research conclusions.

### Possible Result Scenarios

#### Scenario A: Proposed is best overall

Conclusion:

```text
Multi-task GAT-TDTE learns richer process representations and improves PBPM across next activity and remaining time.
```

#### Scenario B: Proposed improves remaining time but slightly reduces next activity

Conclusion:

```text
Multi-task learning creates a trade-off but is valuable for operational monitoring where multiple predictions are needed.
```

#### Scenario C: ProcessTransformer beats GAT-TDTE on next activity

Conclusion:

```text
Sequence-based Transformer remains highly competitive for linear process logs.
GAT-TDTE is more suitable when graph structure, transition semantics, and interpretability matter.
```

#### Scenario D: GAT-TDTE only helps on some datasets

Conclusion:

```text
Graph-based PBPM is dataset-sensitive.
It works best when event logs contain rich transition semantics, meaningful timestamps, and complex process variants.
```

### Human Tasks

1. Choose final narrative.
2. Decide which tables/plots to include.
3. Write discussion around:
   - practical need
   - model comparison
   - limitations
   - future work

### AI Tasks

1. Draft result interpretation.
2. Draft limitations.
3. Draft future work.
4. Draft experiment methodology.
5. Format tables and figures.

### Suggested AI Model

Strong model for writing and argumentation.

## 11. Practical Application Narrative

Use this story in the thesis:

```text
In real operational systems, managers do not only need to know the next step of a running case.
They also need to know how long the case may take, whether it may violate SLA, and whether it may end in an undesired outcome.
Therefore, multi-task PBPM is more useful than single-task next activity prediction.
```

Suitable domains:

```text
IT Helpdesk / ticketing
Bank loan application
Insurance claim processing
E-commerce order fulfillment
Manufacturing workflow
Hospital process monitoring
```

Best domains for future pretraining/fine-tuning:

```text
IT helpdesk
customer support
loan/finance workflow
insurance claim
manufacturing
```

Reason:

```text
They produce many event logs automatically.
They have clear timestamps.
They have clear case IDs.
They often have clear final outcomes.
They often have SLA/deadline labels.
```

## 12. Risks and Mitigation

### Risk 1: Outcome label unclear

Mitigation:

```text
Use only next activity + remaining time.
Move outcome/risk to future work.
```

### Risk 2: ProcessTransformer outperforms GAT-TDTE

Mitigation:

```text
Check implementation.
Check fairness.
If valid, explain that sequence models are strong for linear logs.
Emphasize GAT-TDTE interpretability and graph-semantics advantages.
```

### Risk 3: Multi-task hurts next activity

Mitigation:

```text
Tune task weights.
Report trade-off honestly.
Use remaining time improvement as operational value.
```

### Risk 4: Dataset too limited

Mitigation:

```text
Use at least BPI12/BPI12w + Helpdesk.
Add BPI13i/BPI13c if feasible.
Avoid claiming universal generalization.
```

### Risk 5: Data leakage

Mitigation:

```text
Use case-level split.
Generate prefixes after split if needed.
Do not use future events.
Do not use final-only attributes as input.
```

## 13. Final Deliverables

### Code

```text
Preprocessing scripts
ProcessTransformer baseline
GAT-T reproduction
GAT-TDTE reproduction
Multi-task GAT-TDTE
Training/evaluation scripts
Ablation scripts
Plotting scripts
```

### Experiment Artifacts

```text
Result tables
Metric logs
Saved configs
Model checkpoints
Attention visualizations
Case study figures
```

### Thesis Materials

```text
Research questions
Methodology
Dataset description
Model architecture
Experiment setup
Results
Discussion
Limitations
Future work
```

## 14. Suggested Prompt for AI Code Agent

Use this prompt when starting implementation:

```text
You are working on a PBPM research repo based on GNN event logs.
The goal is to compare ProcessTransformer, GAT-T, GAT-TDTE, and a proposed Multi-task GAT-TDTE.

First, inspect the repository and summarize:
1. Existing datasets and columns.
2. Existing preprocessing pipeline.
3. Existing GAT-T and GAT-TDTE implementations.
4. Existing checkpoints and evaluation scripts.

Do not modify code yet.
After inspection, propose the smallest safe implementation plan to:
1. Reproduce GAT-T and GAT-TDTE.
2. Add ProcessTransformer baseline.
3. Add labels for remaining time.
4. Implement Multi-task GAT-TDTE.
5. Run fair evaluation on shared train/val/test splits.

Pay special attention to avoiding data leakage and preserving existing repo behavior.
```
