"""
Train Phase 3-5 PBPM models on shared multi-task prefix splits.

Supported models:
  - gat_t: single-task next-activity GAT with temporal edge feature
  - gat_tdte: single-task next-activity GAT with time-decay + transition edge semantics
  - process_transformer: sequence baseline for next-activity prediction
  - multitask_gat_tdte: GAT-TDTE encoder with next-activity, remaining-time, and outcome heads
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_CORE = PROJECT_ROOT / "app" / "core"
if str(APP_CORE) not in sys.path:
    sys.path.insert(0, str(APP_CORE))

from dataset_config import DATASET_CONFIGS


DEFAULT_DATASETS = ["BPI12w", "Helpdesk", "BPI13i", "BPI13c"]
DEFAULT_MODELS = ["gat_t", "gat_tdte", "process_transformer", "multitask_gat_tdte"]


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    metadata: dict
    train_graphs: list[Data]
    val_graphs: list[Data]
    test_graphs: list[Data]
    input_dim: int
    num_events: int
    num_edge_types: int
    num_outcomes: int


class GraphCaseDataset(Dataset):
    def __init__(self, graphs: list[Data]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def collate_graphs(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


class SequenceCaseDataset(Dataset):
    def __init__(self, graphs: list[Data]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g.event_ids, g.y_next, g.y_remaining, g.y_outcome


def collate_sequences(batch):
    event_ids, y_next, y_remaining, y_outcome = zip(*batch)
    lengths = torch.tensor([x.numel() for x in event_ids], dtype=torch.long)
    return {
        "event_ids": pad_sequence(event_ids, batch_first=True, padding_value=0),
        "y_next": pad_sequence(y_next, batch_first=True, padding_value=-1),
        "y_remaining": pad_sequence(y_remaining, batch_first=True, padding_value=0.0),
        "y_outcome": pad_sequence(y_outcome, batch_first=True, padding_value=-1),
        "lengths": lengths,
    }


class GATNextActivity(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_events: int,
        hidden_dim: int,
        event_emb_dim: int,
        heads: int,
        edge_dim: int,
    ):
        super().__init__()
        self.event_embedding = nn.Embedding(num_events, event_emb_dim)
        self.gat_event = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.gat_embed = GATConv(event_emb_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.gat_final = GATConv(hidden_dim * heads * 2, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.classifier = nn.Linear(hidden_dim * heads, num_events)

    def encode(self, graph: Batch) -> torch.Tensor:
        edge_attr = graph.edge_attr
        x_event = F.elu(self.gat_event(graph.x, graph.edge_index, edge_attr=edge_attr))
        x_embed = self.event_embedding(graph.event_ids)
        x_embed = F.elu(self.gat_embed(x_embed, graph.edge_index, edge_attr=edge_attr))
        x = torch.cat([x_event, x_embed], dim=-1)
        return F.elu(self.gat_final(x, graph.edge_index, edge_attr=edge_attr))

    def forward(self, graph: Batch) -> torch.Tensor:
        return self.classifier(self.encode(graph))


class MultiTaskGATTDTE(GATNextActivity):
    def __init__(
        self,
        input_dim: int,
        num_events: int,
        num_outcomes: int,
        hidden_dim: int,
        event_emb_dim: int,
        heads: int,
        edge_dim: int,
    ):
        super().__init__(input_dim, num_events, hidden_dim, event_emb_dim, heads, edge_dim)
        encoder_dim = hidden_dim * heads
        self.classifier = nn.Linear(encoder_dim, num_events)
        self.remaining_head = nn.Linear(encoder_dim, 1)
        self.outcome_head = nn.Linear(encoder_dim, num_outcomes)

    def forward(self, graph: Batch) -> dict[str, torch.Tensor]:
        encoded = self.encode(graph)
        return {
            "next_logits": self.classifier(encoded),
            "remaining": self.remaining_head(encoded).squeeze(-1),
            "outcome_logits": self.outcome_head(encoded),
        }


class ProcessTransformer(nn.Module):
    def __init__(
        self,
        num_events: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.event_embedding = nn.Embedding(num_events, hidden_dim)
        self.position_embedding = nn.Embedding(max_len + 1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(hidden_dim, num_events)

    def forward(self, event_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = event_ids.shape
        positions = torch.arange(seq_len, device=event_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.event_embedding(event_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=event_ids.device), diagonal=1).bool()
        padding_mask = positions >= lengths.unsqueeze(1)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.classifier(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 3-5 PBPM experiments.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--input-dir", default="output/experiments/multitask_prefix")
    parser.add_argument("--output-dir", default="output/experiments/phase3_5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--event-emb-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Remaining-time loss weight.")
    parser.add_argument("--beta", type=float, default=0.5, help="Outcome loss weight.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-cases", type=int, default=None)
    parser.add_argument("--max-val-cases", type=int, default=None)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for dataset_name in args.datasets:
        bundle = load_bundle(
            dataset_name=dataset_name,
            input_root=resolve_path(args.input_dir),
            max_train_cases=args.max_train_cases,
            max_val_cases=args.max_val_cases,
            max_test_cases=args.max_test_cases,
            seed=args.seed,
        )
        for model_name in args.models:
            print(f"\n== {dataset_name} | {model_name} ==")
            run_dir = output_root / dataset_name / model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            result = train_one_model(bundle, model_name, args, device, run_dir)
            summary.append(result)
            print(
                f"test_acc={result['test']['next_accuracy']:.4f} "
                f"test_macro_f1={result['test']['next_macro_f1']:.4f}"
            )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    return 0


def load_bundle(
    dataset_name: str,
    input_root: Path,
    max_train_cases: int | None,
    max_val_cases: int | None,
    max_test_cases: int | None,
    seed: int,
) -> DatasetBundle:
    dataset_dir = input_root / dataset_name
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    events = pd.read_csv(dataset_dir / "events.csv")
    prefixes = pd.read_csv(dataset_dir / "prefixes.csv")

    case_col = metadata["columns"]["case_id"]
    event_col = metadata["columns"]["event"]
    time_col = metadata["columns"]["timestamp"]
    feature_cols = [
        col
        for col in metadata["columns"]["event_features"] + metadata["columns"]["case_features"]
        if col in events.columns
    ]

    events[case_col] = events[case_col].astype(str)
    prefixes["case_id"] = prefixes["case_id"].astype(str)
    events[time_col] = pd.to_datetime(events[time_col])

    train_case_ids = sorted(prefixes.loc[prefixes["split"] == "train", "case_id"].unique())
    encoders = fit_feature_encoders(events, case_col, train_case_ids, feature_cols)
    edge_type_map = build_edge_type_map(events, case_col, event_col, train_case_ids)

    graphs = []
    prefix_groups = {cid: grp for cid, grp in prefixes.groupby("case_id", sort=False)}
    for case_id, group in events.groupby(case_col, sort=False):
        graph = build_graph(
            case_id=case_id,
            events=group,
            prefix_rows=prefix_groups[str(case_id)],
            metadata=metadata,
            encoders=encoders,
            edge_type_map=edge_type_map,
            feature_cols=feature_cols,
            case_col=case_col,
            event_col=event_col,
            time_col=time_col,
        )
        graphs.append(graph)

    train_graphs = sample_graphs([g for g in graphs if g.split == "train"], max_train_cases, seed)
    val_graphs = sample_graphs([g for g in graphs if g.split == "val"], max_val_cases, seed)
    test_graphs = sample_graphs([g for g in graphs if g.split == "test"], max_test_cases, seed)

    return DatasetBundle(
        dataset_name=dataset_name,
        metadata=metadata,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        input_dim=int(train_graphs[0].x.shape[-1]),
        num_events=len(metadata["activity_label_map"]),
        num_edge_types=max(edge_type_map.values(), default=0) + 1,
        num_outcomes=len(metadata["outcome_label_map"]),
    )


def fit_feature_encoders(
    events: pd.DataFrame,
    case_col: str,
    train_case_ids: list[str],
    feature_cols: list[str],
) -> dict:
    if not feature_cols:
        return {"categorical": [], "numeric": [], "onehot": None, "scaler": None}

    train_mask = events[case_col].astype(str).isin(train_case_ids)
    categorical = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(events[col])]
    numeric = [col for col in feature_cols if pd.api.types.is_numeric_dtype(events[col])]

    onehot = None
    if categorical:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        onehot.fit(events.loc[train_mask, categorical].fillna("<NA>").astype(str))

    scaler = None
    if numeric:
        scaler = StandardScaler()
        scaler.fit(events.loc[train_mask, numeric].fillna(0.0).astype(float))

    return {"categorical": categorical, "numeric": numeric, "onehot": onehot, "scaler": scaler}


def build_edge_type_map(
    events: pd.DataFrame,
    case_col: str,
    event_col: str,
    train_case_ids: list[str],
) -> dict[str, int]:
    transitions = set()
    train_events = events[events[case_col].astype(str).isin(train_case_ids)]
    for _, group in train_events.groupby(case_col, sort=False):
        values = group[event_col].astype(str).tolist()
        transitions.update(f"{a}->{b}" for a, b in zip(values[:-1], values[1:]))
    return {transition: idx + 1 for idx, transition in enumerate(sorted(transitions))}


def build_graph(
    case_id: str,
    events: pd.DataFrame,
    prefix_rows: pd.DataFrame,
    metadata: dict,
    encoders: dict,
    edge_type_map: dict[str, int],
    feature_cols: list[str],
    case_col: str,
    event_col: str,
    time_col: str,
) -> Data:
    events = events.sort_values("event_index")
    prefix_rows = prefix_rows.sort_values("prefix_end_index")
    x = encode_features(events, encoders, feature_cols)
    event_ids = torch.tensor(events["event_id"].to_numpy(), dtype=torch.long)
    num_nodes = len(events)

    if num_nodes > 1:
        edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()
        times = events[time_col].to_numpy()
        diffs = np.diff(times) / np.timedelta64(1, "s")
        diffs = np.asarray(diffs, dtype=np.float32)
        max_diff = float(diffs.max()) if diffs.size and diffs.max() > 0 else 1.0
        scaled_diff = diffs / max_diff
        decay = np.exp(-scaled_diff).astype(np.float32)
        event_values = events[event_col].astype(str).tolist()
        edge_types = [
            edge_type_map.get(f"{a}->{b}", 0)
            for a, b in zip(event_values[:-1], event_values[1:])
        ]
        edge_type_onehot = np.zeros((num_nodes - 1, max(edge_type_map.values(), default=0) + 1), dtype=np.float32)
        for row_idx, edge_type in enumerate(edge_types):
            edge_type_onehot[row_idx, edge_type] = 1.0
        edge_attr_t = scaled_diff.reshape(-1, 1).astype(np.float32)
        edge_attr_tdte = np.column_stack([scaled_diff, decay, edge_type_onehot]).astype(np.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = np.empty((0, 1), dtype=np.float32)
        edge_attr_tdte = np.empty((0, max(edge_type_map.values(), default=0) + 3), dtype=np.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr_tdte, dtype=torch.float32),
        edge_attr_t=torch.tensor(edge_attr_t, dtype=torch.float32),
        edge_attr_tdte=torch.tensor(edge_attr_tdte, dtype=torch.float32),
        event_ids=event_ids,
        y_next=torch.tensor(prefix_rows["next_activity_id"].to_numpy(), dtype=torch.long),
        y_remaining=torch.tensor(prefix_rows["remaining_time_norm"].to_numpy(), dtype=torch.float32),
        y_outcome=torch.tensor(prefix_rows["outcome_label_id"].to_numpy(), dtype=torch.long),
    )
    data.case_id = str(case_id)
    data.split = str(prefix_rows["split"].iloc[0])
    return data


def encode_features(events: pd.DataFrame, encoders: dict, feature_cols: list[str]) -> np.ndarray:
    parts = []
    if encoders["categorical"]:
        part = encoders["onehot"].transform(
            events[encoders["categorical"]].fillna("<NA>").astype(str)
        )
        parts.append(part)
    if encoders["numeric"]:
        part = encoders["scaler"].transform(
            events[encoders["numeric"]].fillna(0.0).astype(float)
        )
        parts.append(part)
    if not parts:
        return np.ones((len(events), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def sample_graphs(graphs: list[Data], limit: int | None, seed: int) -> list[Data]:
    if limit is None or limit >= len(graphs):
        return graphs
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(graphs)), limit))
    return [graphs[i] for i in indices]


def train_one_model(
    bundle: DatasetBundle,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> dict:
    if model_name == "process_transformer":
        model = ProcessTransformer(
            num_events=bundle.num_events,
            hidden_dim=args.hidden_dim * args.heads,
            heads=args.heads,
            layers=args.transformer_layers,
            dropout=args.dropout,
            max_len=max(g.num_nodes for g in bundle.train_graphs + bundle.val_graphs + bundle.test_graphs),
        )
        train_loader = DataLoader(SequenceCaseDataset(bundle.train_graphs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
        val_loader = DataLoader(SequenceCaseDataset(bundle.val_graphs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequences)
        test_loader = DataLoader(SequenceCaseDataset(bundle.test_graphs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequences)
        trainer = train_transformer_epoch
        evaluator = evaluate_transformer
    else:
        edge_dim = edge_dim_for_model(bundle, model_name)
        if model_name == "multitask_gat_tdte":
            model = MultiTaskGATTDTE(
                input_dim=bundle.input_dim,
                num_events=bundle.num_events,
                num_outcomes=bundle.num_outcomes,
                hidden_dim=args.hidden_dim,
                event_emb_dim=args.event_emb_dim,
                heads=args.heads,
                edge_dim=edge_dim,
            )
        else:
            model = GATNextActivity(
                input_dim=bundle.input_dim,
                num_events=bundle.num_events,
                hidden_dim=args.hidden_dim,
                event_emb_dim=args.event_emb_dim,
                heads=args.heads,
                edge_dim=edge_dim,
            )
        train_loader = DataLoader(GraphCaseDataset(bundle.train_graphs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
        val_loader = DataLoader(GraphCaseDataset(bundle.val_graphs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
        test_loader = DataLoader(GraphCaseDataset(bundle.test_graphs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
        trainer = train_graph_epoch
        evaluator = evaluate_graph

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer(model, train_loader, optimizer, device, model_name, args.alpha, args.beta)
        val_metrics = evaluator(model, val_loader, device, model_name, args.alpha, args.beta)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['next_accuracy']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluator(model, test_loader, device, model_name, args.alpha, args.beta)
    val_metrics = evaluator(model, val_loader, device, model_name, args.alpha, args.beta)

    config = {
        "dataset": bundle.dataset_name,
        "model": model_name,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "event_emb_dim": args.event_emb_dim,
        "heads": args.heads,
        "alpha": args.alpha,
        "beta": args.beta,
        "num_train_cases": len(bundle.train_graphs),
        "num_val_cases": len(bundle.val_graphs),
        "num_test_cases": len(bundle.test_graphs),
    }
    result = {"config": config, "history": history, "val": val_metrics, "test": test_metrics}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "val": val_metrics, "test": test_metrics}, run_dir / "best_model.pt")
    return result


def train_graph_epoch(model, loader, optimizer, device, model_name, alpha, beta) -> dict:
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for graph in loader:
        graph = graph.to(device)
        select_edge_attr(graph, model_name)
        optimizer.zero_grad()
        loss = graph_loss(model, graph, model_name, alpha, beta)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * graph.num_nodes
        total_nodes += int(graph.num_nodes)
    return {"loss": total_loss / max(total_nodes, 1)}


def evaluate_graph(model, loader, device, model_name, alpha, beta) -> dict:
    model.eval()
    losses = []
    y_true_next, y_pred_next = [], []
    y_true_outcome, y_pred_outcome = [], []
    y_true_remaining, y_pred_remaining = [], []
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(device)
            select_edge_attr(graph, model_name)
            loss = graph_loss(model, graph, model_name, alpha, beta)
            losses.append(float(loss.item()) * graph.num_nodes)
            if model_name == "multitask_gat_tdte":
                output = model(graph)
                next_logits = output["next_logits"]
                remaining = output["remaining"]
                outcome_logits = output["outcome_logits"]
                y_true_outcome.extend(graph.y_outcome.cpu().tolist())
                y_pred_outcome.extend(outcome_logits.argmax(dim=-1).cpu().tolist())
                y_true_remaining.extend(graph.y_remaining.cpu().tolist())
                y_pred_remaining.extend(remaining.cpu().tolist())
            else:
                next_logits = model(graph)
            y_true_next.extend(graph.y_next.cpu().tolist())
            y_pred_next.extend(next_logits.argmax(dim=-1).cpu().tolist())

    metrics = classification_metrics(y_true_next, y_pred_next)
    metrics["loss"] = sum(losses) / max(len(y_true_next), 1)
    if y_true_remaining:
        metrics.update(regression_metrics(y_true_remaining, y_pred_remaining))
        metrics.update(outcome_metrics(y_true_outcome, y_pred_outcome))
    return metrics


def graph_loss(model, graph, model_name, alpha, beta) -> torch.Tensor:
    if model_name == "multitask_gat_tdte":
        output = model(graph)
        next_loss = F.cross_entropy(output["next_logits"], graph.y_next)
        remaining_loss = F.smooth_l1_loss(output["remaining"], graph.y_remaining)
        outcome_loss = F.cross_entropy(output["outcome_logits"], graph.y_outcome)
        return next_loss + alpha * remaining_loss + beta * outcome_loss
    logits = model(graph)
    return F.cross_entropy(logits, graph.y_next)


def edge_dim_for_model(bundle: DatasetBundle, model_name: str) -> int:
    if model_name == "gat_t":
        return int(bundle.train_graphs[0].edge_attr_t.shape[-1])
    return int(bundle.train_graphs[0].edge_attr_tdte.shape[-1])


def select_edge_attr(graph: Batch, model_name: str) -> None:
    if model_name == "gat_t":
        graph.edge_attr = graph.edge_attr_t
    else:
        graph.edge_attr = graph.edge_attr_tdte


def train_transformer_epoch(model, loader, optimizer, device, model_name, alpha, beta) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = move_sequence_batch(batch, device)
        optimizer.zero_grad()
        logits = model(batch["event_ids"], batch["lengths"])
        loss = masked_ce(logits, batch["y_next"])
        loss.backward()
        optimizer.step()
        tokens = int((batch["y_next"] != -1).sum().item())
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    return {"loss": total_loss / max(total_tokens, 1)}


def evaluate_transformer(model, loader, device, model_name, alpha, beta) -> dict:
    model.eval()
    losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_sequence_batch(batch, device)
            logits = model(batch["event_ids"], batch["lengths"])
            loss = masked_ce(logits, batch["y_next"])
            mask = batch["y_next"] != -1
            losses.append(float(loss.item()) * int(mask.sum().item()))
            y_true.extend(batch["y_next"][mask].cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1)[mask].cpu().tolist())
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = sum(losses) / max(len(y_true), 1)
    return metrics


def masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target != -1
    return F.cross_entropy(logits[mask], target[mask])


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "next_accuracy": float(accuracy_score(y_true, y_pred)),
        "next_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "next_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def outcome_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "outcome_accuracy": float(accuracy_score(y_true, y_pred)),
        "outcome_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "outcome_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "remaining_mae_norm": float(mean_absolute_error(y_true, y_pred)),
        "remaining_rmse_norm": float(math.sqrt(mse)),
    }


def move_sequence_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(name)


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


if __name__ == "__main__":
    raise SystemExit(main())
