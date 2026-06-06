"""
Build shared case-level splits and prefix labels for multi-task PBPM.

Outputs per dataset:
  output/experiments/multitask_prefix/<dataset>/
    - events.csv
    - cases.csv
    - prefixes.csv
    - metadata.json

The builder creates labels for:
  - next_activity
  - remaining_time_seconds and normalized remaining time
  - outcome/risk, using final-event outcome when meaningful, otherwise
    train-split P75 duration risk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_CORE = PROJECT_ROOT / "app" / "core"
if str(APP_CORE) not in sys.path:
    sys.path.insert(0, str(APP_CORE))

from dataset_config import DATASET_CONFIGS, list_datasets


DEFAULT_DATASETS = ["BPI12w", "Helpdesk", "BPI13i", "BPI13c"]
EOS_LABEL = "EOS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-task prefix labels and case-level splits."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list_datasets())
    parser.add_argument("--output-dir", default="output/experiments/multitask_prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--risk-quantile", type=float, default=0.75)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    output_root = _resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset_name in args.datasets:
        summary = build_dataset(
            dataset_name=dataset_name,
            output_root=output_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            risk_quantile=args.risk_quantile,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(
            f"{dataset_name}: cases={summary['num_cases']} "
            f"prefixes={summary['num_prefixes']} "
            f"outcome={summary['outcome_strategy']}"
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps({"datasets": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    return 0


def build_dataset(
    dataset_name: str,
    output_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    risk_quantile: float,
    overwrite: bool,
) -> dict:
    ds_config = DATASET_CONFIGS[dataset_name]
    dataset_dir = output_root / dataset_name
    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{dataset_dir} already exists. Pass --overwrite to rebuild it."
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    event_df = _load_sorted_events(ds_config)
    case_index = ds_config["case_index"]
    event_col = ds_config["core_event"]
    time_col = ds_config["time_col"]

    case_ids = event_df[case_index].drop_duplicates().astype(str).to_numpy()
    split_map = _split_cases(
        case_ids=case_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    events_out = event_df.copy()
    events_out[case_index] = events_out[case_index].astype(str)
    events_out["event_index"] = events_out.groupby(case_index).cumcount()
    events_out["split"] = events_out[case_index].map(split_map)

    cases = _build_cases(events_out, case_index, event_col, time_col)
    cases["split"] = cases["case_id"].map(split_map)

    outcome_strategy, outcome_meta = _choose_outcome_strategy(
        cases=cases,
        train_case_mask=cases["split"] == "train",
        risk_quantile=risk_quantile,
    )
    cases, outcome_label_map = _apply_outcome_labels(
        cases=cases,
        strategy=outcome_strategy,
        outcome_meta=outcome_meta,
    )

    prefixes = _build_prefixes(events_out, cases, case_index, event_col, time_col)
    train_remaining = prefixes.loc[
        prefixes["split"] == "train", "remaining_time_seconds"
    ].to_numpy(dtype=float)
    remaining_mean = float(train_remaining.mean())
    remaining_std = float(train_remaining.std(ddof=0) or 1.0)
    prefixes["remaining_time_norm"] = (
        prefixes["remaining_time_seconds"] - remaining_mean
    ) / remaining_std

    activity_vocab = sorted(set(events_out[event_col].astype(str).tolist()) | {EOS_LABEL})
    activity_label_map = {label: idx for idx, label in enumerate(activity_vocab)}
    prefixes["next_activity_id"] = prefixes["next_activity"].map(activity_label_map)
    events_out["event_id"] = events_out[event_col].astype(str).map(activity_label_map)

    events_out.to_csv(dataset_dir / "events.csv", index=False)
    cases.to_csv(dataset_dir / "cases.csv", index=False)
    prefixes.to_csv(dataset_dir / "prefixes.csv", index=False)

    metadata = {
        "dataset": dataset_name,
        "source_file": str(_resolve_path(ds_config["file"])),
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "num_cases": int(cases.shape[0]),
        "num_events": int(events_out.shape[0]),
        "num_prefixes": int(prefixes.shape[0]),
        "split_counts_cases": _value_counts(cases["split"]),
        "split_counts_prefixes": _value_counts(prefixes["split"]),
        "activity_label_map": activity_label_map,
        "outcome_strategy": outcome_strategy,
        "outcome_metadata": outcome_meta,
        "outcome_label_map": outcome_label_map,
        "remaining_time_normalization": {
            "mean_seconds": remaining_mean,
            "std_seconds": remaining_std,
            "fit_split": "train",
        },
        "columns": {
            "case_id": case_index,
            "event": event_col,
            "timestamp": time_col,
            "event_features": ds_config.get("cat_col_event", [])
            + ds_config.get("num_col_event", []),
            "case_features": ds_config.get("cat_col_seq", [])
            + ds_config.get("num_col_seq", []),
        },
        "leakage_controls": [
            "Cases are split before prefix rows are used for training.",
            "Remaining-time normalization is fitted on train prefixes only.",
            "Duration-risk threshold is fitted on train cases only.",
            "Prefix rows store prefix_end_index and do not include future events.",
        ],
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "dataset": dataset_name,
        "num_cases": metadata["num_cases"],
        "num_events": metadata["num_events"],
        "num_prefixes": metadata["num_prefixes"],
        "outcome_strategy": outcome_strategy,
        "output_dir": str(dataset_dir),
    }


def _load_sorted_events(ds_config: dict) -> pd.DataFrame:
    path = _resolve_path(ds_config["file"])
    event_df = pd.read_csv(path)
    case_index = ds_config["case_index"]
    time_col = ds_config["time_col"]

    if ds_config.get("ec1_as_str") and "ec1" in event_df.columns:
        event_df["ec1"] = event_df["ec1"].astype(str)

    min_size = ds_config.get("min_seq_size", 2)
    event_df = event_df[
        event_df.groupby(case_index)[case_index].transform("size") >= min_size
    ].copy()
    event_df[time_col] = pd.to_datetime(event_df[time_col], errors="raise")
    event_df = event_df.sort_values([case_index, time_col]).reset_index(drop=True)
    return event_df


def _split_cases(
    case_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    split_map = {}
    for case_id in shuffled[:train_end]:
        split_map[str(case_id)] = "train"
    for case_id in shuffled[train_end:val_end]:
        split_map[str(case_id)] = "val"
    for case_id in shuffled[val_end:]:
        split_map[str(case_id)] = "test"
    return split_map


def _build_cases(
    events: pd.DataFrame,
    case_index: str,
    event_col: str,
    time_col: str,
) -> pd.DataFrame:
    rows = []
    for case_id, group in events.groupby(case_index, sort=False):
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        duration = (end_time - start_time).total_seconds()
        rows.append(
            {
                "case_id": str(case_id),
                "num_events": int(group.shape[0]),
                "start_time": start_time,
                "end_time": end_time,
                "case_duration_seconds": float(duration),
                "final_event": str(group[event_col].iloc[-1]),
                "final_status": str(group["status"].iloc[-1])
                if "status" in group.columns
                else "",
            }
        )
    return pd.DataFrame(rows)


def _choose_outcome_strategy(
    cases: pd.DataFrame,
    train_case_mask: pd.Series,
    risk_quantile: float,
) -> tuple[str, dict]:
    train_cases = cases.loc[train_case_mask].copy()
    final_counts = train_cases["final_event"].value_counts()
    top_share = float(final_counts.iloc[0] / final_counts.sum())
    meaningful_classes = int((final_counts / final_counts.sum() >= 0.01).sum())

    if meaningful_classes >= 2 and top_share <= 0.95:
        return (
            "final_event_outcome",
            {
                "fit_split": "train",
                "class_counts_train": {
                    str(k): int(v) for k, v in final_counts.to_dict().items()
                },
                "top_class_share_train": top_share,
                "meaningful_class_min_share": 0.01,
            },
        )

    threshold = float(train_cases["case_duration_seconds"].quantile(risk_quantile))
    return (
        "duration_p75_risk",
        {
            "fit_split": "train",
            "risk_quantile": risk_quantile,
            "duration_threshold_seconds": threshold,
            "reason": "Final event/status is too dominant for a meaningful outcome label.",
            "final_event_counts_train": {
                str(k): int(v) for k, v in final_counts.to_dict().items()
            },
            "top_class_share_train": top_share,
        },
    )


def _apply_outcome_labels(
    cases: pd.DataFrame,
    strategy: str,
    outcome_meta: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cases = cases.copy()
    if strategy == "final_event_outcome":
        labels = cases["final_event"].astype(str)
    elif strategy == "duration_p75_risk":
        threshold = float(outcome_meta["duration_threshold_seconds"])
        labels = np.where(cases["case_duration_seconds"] >= threshold, "risk_high", "risk_normal")
    else:
        raise ValueError(f"Unsupported outcome strategy: {strategy}")

    label_vocab = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(label_vocab)}
    cases["outcome_label"] = labels
    cases["outcome_label_id"] = cases["outcome_label"].map(label_map)
    return cases, label_map


def _build_prefixes(
    events: pd.DataFrame,
    cases: pd.DataFrame,
    case_index: str,
    event_col: str,
    time_col: str,
) -> pd.DataFrame:
    case_lookup = cases.set_index("case_id")
    rows = []
    sample_idx = 0
    for case_id, group in events.groupby(case_index, sort=False):
        case_id = str(case_id)
        case_row = case_lookup.loc[case_id]
        event_values = group[event_col].astype(str).tolist()
        time_values = group[time_col].tolist()

        for idx, current_time in enumerate(time_values):
            next_activity = event_values[idx + 1] if idx + 1 < len(event_values) else EOS_LABEL
            remaining_seconds = (case_row["end_time"] - current_time).total_seconds()
            rows.append(
                {
                    "sample_id": sample_idx,
                    "case_id": case_id,
                    "split": case_row["split"],
                    "prefix_end_index": idx,
                    "prefix_length": idx + 1,
                    "current_event": event_values[idx],
                    "next_activity": next_activity,
                    "is_terminal_prefix": bool(idx + 1 == len(event_values)),
                    "remaining_time_seconds": float(remaining_seconds),
                    "outcome_label": case_row["outcome_label"],
                    "outcome_label_id": int(case_row["outcome_label_id"]),
                }
            )
            sample_idx += 1
    return pd.DataFrame(rows)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists() or not path.parts or path.parts[0] == "output":
        return candidate
    app_relative = PROJECT_ROOT / "app" / path
    if app_relative.exists():
        return app_relative.resolve()
    return path


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive.")


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


if __name__ == "__main__":
    raise SystemExit(main())
