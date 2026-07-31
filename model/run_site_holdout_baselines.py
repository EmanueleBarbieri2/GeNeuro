#!/usr/bin/env python3
"""Train and aggregate raw-graph baselines on existing site-held-out folds."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

# PyG has a correct native fallback when the optional torch-scatter package is
# unavailable.  Its per-call advisory otherwise overwhelms long nohup logs.
warnings.filterwarnings(
    "ignore",
    message=r"The usage of `scatter\(reduce='max'\)` can be accelerated.*",
    category=UserWarning,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.downstream.downstream_classification import load_csv_labels  # noqa: E402
from model.downstream.downstream_progression import load_csv_visits  # noqa: E402
from model.downstream.downstream_updrs import load_csv_targets  # noqa: E402
from model.encoders import DTIEncoder, MRIEncoder, SPECTEncoder, fMRIEncoder  # noqa: E402


MODALITIES = ("SPECT", "MRI", "fMRI", "DTI")
CLASS_NAMES = ("Control", "PD", "Prodromal")
BINARY_CLASS_NAMES = ("Control", "PD")
PARTITIONS = ("train", "val", "test")
TARGETS = ((1, "U2_ADL"), (2, "U3_Motor"))
TASKS = (
    "classification",
    "static_u2",
    "static_u3",
    "progression_u2",
    "progression_u3",
)
BASELINES = {
    "spect": {"modalities": ("SPECT",), "model": "GCN", "modality_label": "SPECT"},
    "mri": {"modalities": ("MRI",), "model": "GCN", "modality_label": "MRI"},
    "fmri": {"modalities": ("fMRI",), "model": "GINE", "modality_label": "fMRI"},
    "dti": {"modalities": ("DTI",), "model": "GINE", "modality_label": "DTI"},
    "multimodal": {
        "modalities": MODALITIES,
        "model": "GCN\\&GINE",
        "modality_label": "Multi",
    },
}
FULL_MODEL_FILES = {
    "classification": "classifier.pt",
    "static_u2": "static_U2_ADL.pt",
    "static_u3": "static_U3_Motor.pt",
    "progression_u2": "prog_U2_ADL.pt",
    "progression_u3": "prog_U3_Motor.pt",
}
TABLE_COLUMNS = (
    ("classification", "bal_acc", "Bal. Acc."),
    ("classification", "f1_macro", "Macro F1"),
    ("classification", "auc_macro", "Macro AUC"),
    ("static_u2", "r2", "Severity Part II"),
    ("static_u3", "r2", "Severity Part III"),
    ("progression_u2", "r2", "Progression Part II"),
    ("progression_u3", "r2", "Progression Part III"),
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_seed(base_seed, *parts):
    text = "|".join([str(base_seed), *map(str, parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _patient_id(visit_id):
    return str(visit_id).split("_", 1)[0]


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return value


def _load_split(path):
    partitions = {partition: set() for partition in PARTITIONS}
    mode = None
    comments = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("#"):
                comments.append(line)
            elif line in {"train_ids:", "val_ids:", "test_ids:"}:
                mode = line.split("_", 1)[0]
            elif line:
                if mode is None:
                    raise ValueError(f"ID found before a partition header in {path}: {line}")
                partitions[mode].add(line)
    if not all(partitions.values()):
        raise ValueError(f"Site-held-out split must have train, val, and test IDs: {path}")
    patients = {
        partition: {_patient_id(visit_id) for visit_id in ids}
        for partition, ids in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[left] & patients[right]
        if overlap:
            raise ValueError(
                f"Patient leakage in {path}: {len(overlap)} patients overlap {left}/{right}."
            )
    if not any("site" in comment.lower() for comment in comments):
        raise ValueError(f"Split is not documented as site-held-out: {path}")
    visit_partition = {
        visit_id: partition
        for partition, ids in partitions.items()
        for visit_id in ids
    }
    patient_partition = {
        patient: partition
        for partition, ids in patients.items()
        for patient in ids
    }
    return partitions, patients, visit_partition, patient_partition, comments


def _load_site_metadata(csv_path):
    visit_sites = {}
    patient_sites = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient = row.get("PATNO")
            event = row.get("EVENT_ID")
            site = row.get("SITE")
            if not patient or not event or not site:
                continue
            visit_sites[f"{patient}_{event}"] = site
            patient_sites[patient].add(site)
    inconsistent = {
        patient: sites for patient, sites in patient_sites.items() if len(sites) != 1
    }
    if inconsistent:
        patient, sites = next(iter(inconsistent.items()))
        raise ValueError(
            f"Patient {patient} occurs at multiple sites {sorted(sites)}; "
            "site-held-out evaluation is not well defined."
        )
    return visit_sites


def _audit_site_disjointness(partitions, visit_sites, split_path):
    missing = [
        visit_id
        for ids in partitions.values()
        for visit_id in ids
        if visit_id not in visit_sites
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} split IDs have no SITE metadata in {split_path}; "
            f"example: {missing[0]}"
        )
    sites = {
        partition: {visit_sites[visit_id] for visit_id in ids}
        for partition, ids in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sites[left] & sites[right]
        if overlap:
            raise ValueError(
                f"Site leakage in {split_path}: {sorted(overlap)} overlap {left}/{right}."
            )
    return sites


def _find_fold_dirs(root, folds):
    output = {}
    for fold in folds:
        fold_dir = os.path.join(root, f"fold_{fold}")
        split_path = os.path.join(fold_dir, "site_grouped_split.txt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing site-held-out split for fold {fold}: {split_path}")
        output[fold] = {"fold_dir": fold_dir, "split_path": split_path}
    return output


class GraphStore:
    def __init__(self, data_root, cache=True):
        self.data_root = os.path.abspath(data_root)
        self.cache_enabled = cache
        self.cache = {}
        self.available = {}
        for modality in MODALITIES:
            modality_dir = os.path.join(self.data_root, modality)
            if not os.path.isdir(modality_dir):
                raise FileNotFoundError(f"Missing modality directory: {modality_dir}")
            self.available[modality] = {
                os.path.splitext(filename)[0]
                for filename in os.listdir(modality_dir)
                if filename.endswith(".pt")
            }

    def has(self, modality, visit_id):
        return visit_id in self.available[modality]

    def any(self, modalities, visit_id):
        return any(self.has(modality, visit_id) for modality in modalities)

    def get(self, modality, visit_id):
        key = (modality, visit_id)
        if key in self.cache:
            return self.cache[key]
        path = os.path.join(self.data_root, modality, f"{visit_id}.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            kwargs = dict(payload)
            if "num_nodes" in kwargs and "x" not in kwargs:
                kwargs["num_nodes"] = int(kwargs["num_nodes"])
            if "edge_weight" in kwargs and "edge_attr" not in kwargs:
                kwargs["edge_attr"] = kwargs.pop("edge_weight")
            payload = Data(**kwargs)
        if hasattr(payload, "edge_weight") and not hasattr(payload, "edge_attr"):
            payload.edge_attr = payload.edge_weight
        if hasattr(payload, "edge_attr") and torch.is_tensor(payload.edge_attr):
            if payload.edge_attr.ndim == 2 and payload.edge_attr.shape[1] == 1:
                payload.edge_attr = payload.edge_attr.view(-1)
        if self.cache_enabled:
            self.cache[key] = payload
        return payload


@dataclass
class VisitRecord:
    visit_id: str
    patient_id: str
    value: object


@dataclass
class ProgressionRecord:
    sample_id: str
    patient_id: str
    history_ids: tuple
    history_years: tuple
    next_delta: float
    target: float


class VisitDataset(Dataset):
    def __init__(self, records, modalities, graph_store):
        self.records = records
        self.modalities = tuple(modalities)
        self.graph_store = graph_store

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        graphs = {
            modality: self.graph_store.get(modality, record.visit_id)
            for modality in self.modalities
            if self.graph_store.has(modality, record.visit_id)
        }
        return record, graphs


class ProgressionDataset(Dataset):
    def __init__(self, records, modalities, graph_store):
        self.records = records
        self.modalities = tuple(modalities)
        self.graph_store = graph_store

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        visits = []
        for visit_id in record.history_ids:
            visits.append(
                {
                    modality: self.graph_store.get(modality, visit_id)
                    for modality in self.modalities
                    if self.graph_store.has(modality, visit_id)
                }
            )
        return record, visits


def _collate_visit(batch):
    records, graph_dicts = zip(*batch)
    modality_graphs = defaultdict(list)
    modality_positions = defaultdict(list)
    for position, graphs in enumerate(graph_dicts):
        for modality, graph in graphs.items():
            modality_graphs[modality].append(graph)
            modality_positions[modality].append(position)
    return {
        "records": records,
        "batch_size": len(records),
        "graphs": {
            modality: Batch.from_data_list(graphs)
            for modality, graphs in modality_graphs.items()
        },
        "positions": {
            modality: torch.tensor(positions, dtype=torch.long)
            for modality, positions in modality_positions.items()
        },
    }


def _collate_progression(batch):
    records, sequences = zip(*batch)
    modality_graphs = defaultdict(list)
    modality_positions = defaultdict(list)
    flat_index = 0
    lengths = []
    max_length = max(len(sequence) for sequence in sequences)
    deltas = torch.zeros(len(sequences), max_length, dtype=torch.float32)
    flat_sample = []
    flat_time = []
    for sample_index, (record, sequence) in enumerate(zip(records, sequences)):
        lengths.append(len(sequence))
        years = torch.tensor(record.history_years, dtype=torch.float32)
        if len(years) > 1:
            deltas[sample_index, 1 : len(years)] = years[1:] - years[:-1]
        for time_index, graphs in enumerate(sequence):
            flat_sample.append(sample_index)
            flat_time.append(time_index)
            for modality, graph in graphs.items():
                modality_graphs[modality].append(graph)
                modality_positions[modality].append(flat_index)
            flat_index += 1
    return {
        "records": records,
        "batch_size": len(records),
        "total_visits": flat_index,
        "max_length": max_length,
        "graphs": {
            modality: Batch.from_data_list(graphs)
            for modality, graphs in modality_graphs.items()
        },
        "positions": {
            modality: torch.tensor(positions, dtype=torch.long)
            for modality, positions in modality_positions.items()
        },
        "flat_sample": torch.tensor(flat_sample, dtype=torch.long),
        "flat_time": torch.tensor(flat_time, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "deltas": deltas,
        "next_delta": torch.tensor(
            [record.next_delta for record in records], dtype=torch.float32
        ).unsqueeze(1),
        "targets": torch.tensor(
            [record.target for record in records], dtype=torch.float32
        ).unsqueeze(1),
    }


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, inputs):
        if self.training and inputs.ndim >= 2 and inputs.shape[0] == 1:
            return F.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(inputs)


def _replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            replacement = SafeBatchNorm1d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=True,
            )
            if child.affine:
                replacement.weight.data.copy_(child.weight.data)
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child)
    return module


def _make_encoder(modality, hidden_dim, embed_dim, threshold):
    if modality == "SPECT":
        model = SPECTEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim)
    elif modality == "MRI":
        model = MRIEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim)
    elif modality == "fMRI":
        model = fMRIEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim, threshold=threshold)
    elif modality == "DTI":
        model = DTIEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim, threshold=threshold)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return _replace_batch_norm(model)


class FusionEncoder(nn.Module):
    def __init__(
        self,
        modalities,
        hidden_dim,
        embed_dim,
        threshold,
        include_missingness_mask,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.embed_dim = embed_dim
        self.include_missingness_mask = include_missingness_mask and len(modalities) > 1
        self.encoders = nn.ModuleDict(
            {
                modality: _make_encoder(modality, hidden_dim, embed_dim, threshold)
                for modality in self.modalities
            }
        )

    @property
    def output_dim(self):
        return len(self.modalities) * self.embed_dim + (
            len(self.modalities) if self.include_missingness_mask else 0
        )

    def forward(self, graph_batches, positions, total_items, device):
        chunks = []
        missingness = []
        for modality in self.modalities:
            full = torch.zeros(total_items, self.embed_dim, device=device)
            present = torch.zeros(total_items, dtype=torch.bool, device=device)
            if modality in graph_batches:
                indices = positions[modality].to(device)
                encoded = self.encoders[modality](graph_batches[modality].to(device))
                full = full.index_copy(0, indices, encoded)
                present[indices] = True
            chunks.append(full)
            missingness.append((~present).float().unsqueeze(1))
        output = torch.cat(chunks, dim=1)
        if self.include_missingness_mask:
            output = torch.cat([output, *missingness], dim=1)
        return output


class ClassificationModel(nn.Module):
    def __init__(self, fusion, dropout, num_classes):
        super().__init__()
        self.fusion = fusion
        self.head = nn.Sequential(
            nn.Linear(fusion.output_dim, 512),
            SafeBatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            SafeBatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch, device):
        features = self.fusion(
            batch["graphs"],
            batch["positions"],
            batch["batch_size"],
            device,
        )
        return self.head(features)


class StaticRegressionModel(nn.Module):
    def __init__(self, fusion, dropout):
        super().__init__()
        self.fusion = fusion
        self.head = nn.Sequential(
            nn.Linear(fusion.output_dim, 512),
            SafeBatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            SafeBatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 1),
        )

    def forward(self, batch, device):
        features = self.fusion(
            batch["graphs"],
            batch["positions"],
            batch["batch_size"],
            device,
        )
        return self.head(features)


class ProgressionModel(nn.Module):
    def __init__(self, fusion, recurrent_dim, dropout):
        super().__init__()
        self.fusion = fusion
        self.compressor = nn.Sequential(
            nn.Linear(fusion.output_dim, recurrent_dim),
            nn.LayerNorm(recurrent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_encoder = nn.Linear(1, 32)
        self.gru = nn.GRU(
            recurrent_dim + 32,
            recurrent_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(recurrent_dim + 1, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch, device):
        flat = self.fusion(
            batch["graphs"],
            batch["positions"],
            batch["total_visits"],
            device,
        )
        compressed = self.compressor(flat)
        padded = torch.zeros(
            batch["batch_size"],
            batch["max_length"],
            compressed.shape[1],
            device=device,
        )
        padded[
            batch["flat_sample"].to(device),
            batch["flat_time"].to(device),
        ] = compressed
        time_features = F.gelu(self.time_encoder(batch["deltas"].to(device).unsqueeze(-1)))
        packed = pack_padded_sequence(
            torch.cat([padded, time_features], dim=2),
            batch["lengths"].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(
            torch.cat([hidden[-1], batch["next_delta"].to(device)], dim=1)
        )


def _build_visit_records(
    values,
    visit_partition,
    graph_store,
    modalities,
    target_index=None,
    eligible_ids=None,
):
    records = {partition: [] for partition in PARTITIONS}
    for visit_id, value in values.items():
        if eligible_ids is not None and visit_id not in eligible_ids:
            continue
        partition = visit_partition.get(visit_id)
        if partition is None:
            continue
        if len(modalities) == 1:
            if not graph_store.has(modalities[0], visit_id):
                continue
        elif not graph_store.any(modalities, visit_id):
            continue
        if target_index is not None:
            numeric = float(value[target_index])
            if not np.isfinite(numeric):
                continue
            value = numeric
        records[partition].append(
            VisitRecord(
                visit_id=visit_id,
                patient_id=_patient_id(visit_id),
                value=value,
            )
        )
    for partition in PARTITIONS:
        records[partition].sort(key=lambda record: record.visit_id)
    return records


def _build_progression_records(
    visits,
    patient_partition,
    graph_store,
    modalities,
    target_index,
    eligible_ids=None,
):
    records = {partition: [] for partition in PARTITIONS}
    for patient_id, patient_visits in visits.items():
        partition = patient_partition.get(str(patient_id))
        if partition is None:
            continue
        ordered = sorted(
            (
                visit
                for visit in patient_visits
                if eligible_ids is None or visit["key"] in eligible_ids
            ),
            key=lambda visit: visit["year"],
        )
        if len(modalities) == 1:
            valid = [
                visit
                for visit in ordered
                if graph_store.has(modalities[0], visit["key"])
            ]
        else:
            valid = [
                visit
                for visit in ordered
                if graph_store.any(modalities, visit["key"])
            ]
        for index in range(len(valid) - 1):
            target = float(valid[index + 1]["targets"][target_index])
            if not np.isfinite(target):
                continue
            history = valid[: index + 1]
            records[partition].append(
                ProgressionRecord(
                    sample_id=valid[index + 1]["key"],
                    patient_id=str(patient_id),
                    history_ids=tuple(visit["key"] for visit in history),
                    history_years=tuple(float(visit["year"]) for visit in history),
                    next_delta=float(valid[index + 1]["year"] - valid[index]["year"]),
                    target=target,
                )
            )
    for partition in PARTITIONS:
        records[partition].sort(key=lambda record: (record.patient_id, record.sample_id))
    return records


def _make_loader(dataset, batch_size, shuffle, seed, collate, drop_singleton=False):
    drop_last = drop_singleton and len(dataset) % batch_size == 1
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate,
        num_workers=0,
        generator=generator,
    )


def _classification_arrays(model, loader, device, class_names):
    class_to_index = {
        name: index for index, name in enumerate(class_names)
    }
    model.eval()
    targets, predictions, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch, device)
            probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(class_to_index[record.value] for record in batch["records"])
    return (
        np.asarray(targets, dtype=int),
        np.asarray(predictions, dtype=int),
        np.concatenate(probabilities),
    )


def _classification_metrics(targets, predictions, probabilities, class_names):
    metrics = {
        "bal_acc": float(balanced_accuracy_score(targets, predictions)),
        "acc": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }
    try:
        if len(class_names) == 2:
            metrics["auc_macro"] = float(roc_auc_score(targets, probabilities[:, 1]))
        else:
            metrics["auc_macro"] = float(
                roc_auc_score(
                    targets,
                    probabilities,
                    labels=np.arange(len(class_names)),
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["auc_macro"] = math.nan
    return metrics


def _availability_only_diagnostic(split_info, labels, graph_store, eligible_ids, fold):
    class_to_index = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }
    pattern_counts = defaultdict(Counter)
    global_counts = Counter()
    test_rows = []
    for visit_id, label in labels.items():
        if visit_id not in eligible_ids:
            continue
        partition = split_info["visit_partition"].get(visit_id)
        if partition not in {"train", "test"}:
            continue
        pattern = tuple(
            int(graph_store.has(modality, visit_id)) for modality in MODALITIES
        )
        if partition == "train":
            pattern_counts[pattern][label] += 1
            global_counts[label] += 1
        else:
            test_rows.append((label, pattern))
    if not test_rows or any(global_counts[name] == 0 for name in CLASS_NAMES):
        return {
            "fold": fold,
            "status": "not_evaluable",
            "reason": "Availability-only diagnostic lacks samples or a training class.",
        }
    targets_array = []
    predictions = []
    probabilities = []
    for label, pattern in test_rows:
        counts = pattern_counts.get(pattern) or global_counts
        smoothed = np.asarray(
            [counts[name] + 1.0 for name in CLASS_NAMES],
            dtype=float,
        )
        probability = smoothed / smoothed.sum()
        targets_array.append(class_to_index[label])
        predictions.append(int(probability.argmax()))
        probabilities.append(probability)
    metrics = _classification_metrics(
        np.asarray(targets_array, dtype=int),
        np.asarray(predictions, dtype=int),
        np.asarray(probabilities, dtype=float),
        CLASS_NAMES,
    )
    return {
        "fold": fold,
        "status": "ok",
        "train_samples": sum(global_counts.values()),
        "test_samples": len(test_rows),
        "train_patterns": len(pattern_counts),
        **metrics,
        "train_pattern_counts": json.dumps(
            {
                "".join(map(str, pattern)): dict(sorted(counts.items()))
                for pattern, counts in sorted(pattern_counts.items())
            },
            sort_keys=True,
        ),
    }


def _regression_arrays(model, loader, device, progression):
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch, device).cpu().numpy().ravel() * 100.0
            predictions.append(output)
            if progression:
                targets.append(batch["targets"].numpy().ravel())
            else:
                targets.append(
                    np.asarray([record.value for record in batch["records"]], dtype=float)
                )
    return np.concatenate(targets), np.concatenate(predictions)


def _pipeline_r2(targets, predictions):
    if len(targets) < 2:
        return math.nan
    variance = float(np.var(targets, ddof=1))
    if variance <= 0:
        return math.nan
    return float(1.0 - mean_squared_error(targets, predictions) / (variance + 1e-8))


def _regression_metrics(targets, predictions):
    return {
        "r2": _pipeline_r2(targets, predictions),
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
    }


def _class_counts(records):
    return {
        partition: dict(sorted(Counter(record.value for record in rows).items()))
        for partition, rows in records.items()
    }


def _record_identity(records):
    identifiers = sorted(
        record.sample_id if isinstance(record, ProgressionRecord) else record.visit_id
        for record in records
    )
    digest = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
    return {
        "test_samples": len(identifiers),
        "test_patients": len({record.patient_id for record in records}),
        "test_ids_sha256": digest,
    }


def _not_evaluable(task, reason, counts):
    return {
        "status": "not_evaluable",
        "task": task,
        "reason": reason,
        "counts": counts,
    }


def _train_classification(records, modalities, graph_store, args, device, seed):
    counts = _class_counts(records)
    class_sets = [set(counts[partition]) for partition in PARTITIONS]
    if all(set(CLASS_NAMES).issubset(classes) for classes in class_sets):
        class_names = CLASS_NAMES
    elif all(set(BINARY_CLASS_NAMES).issubset(classes) for classes in class_sets) and all(
        "Prodromal" not in classes for classes in class_sets
    ):
        class_names = BINARY_CLASS_NAMES
    else:
        missing_by_partition = {
            partition: sorted(set(CLASS_NAMES) - set(counts[partition]))
            for partition in PARTITIONS
        }
        return _not_evaluable(
            "classification",
            (
                "Neither a consistent three-class task nor a consistent binary "
                f"Control-vs-PD task is defined: {missing_by_partition}."
            ),
            counts,
        )
    for partition in PARTITIONS:
        missing = set(class_names) - set(counts[partition])
        if missing:
            return _not_evaluable(
                "classification",
                (
                    f"{len(class_names)}-class classification is undefined because {partition} "
                    f"has no observed samples for classes {sorted(missing)}."
                ),
                counts,
            )
    train_dataset = VisitDataset(records["train"], modalities, graph_store)
    val_dataset = VisitDataset(records["val"], modalities, graph_store)
    test_dataset = VisitDataset(records["test"], modalities, graph_store)
    train_loader = _make_loader(
        train_dataset, args.batch_size, True, seed, _collate_visit, True
    )
    val_loader = _make_loader(
        val_dataset, args.batch_size, False, seed, _collate_visit
    )
    test_loader = _make_loader(
        test_dataset, args.batch_size, False, seed, _collate_visit
    )
    fusion = FusionEncoder(
        modalities,
        args.encoder_hidden_dim,
        args.embed_dim,
        args.edge_threshold,
        args.multimodal_missingness_mask,
    )
    model = ClassificationModel(fusion, args.dropout, len(class_names)).to(device)
    class_counter = Counter(record.value for record in records["train"])
    class_to_index = {
        name: index for index, name in enumerate(class_names)
    }
    weights = torch.tensor(
        [
            len(records["train"]) / (len(class_names) * class_counter[class_name])
            for class_name in class_names
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            targets = torch.tensor(
                [class_to_index[record.value] for record in batch["records"]],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch, device), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        val_true, val_pred, val_prob = _classification_arrays(
            model, val_loader, device, class_names
        )
        val_metrics = _classification_metrics(
            val_true, val_pred, val_prob, class_names
        )
        score = val_metrics["bal_acc"]
        if score > best_score + args.min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"      epoch {epoch + 1:03d}/{args.epochs} "
                f"val_bal_acc={score:.4f} best={best_score:.4f}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            break
    model.load_state_dict(best_state)
    test_true, test_pred, test_prob = _classification_arrays(
        model, test_loader, device, class_names
    )
    return {
        "status": "ok",
        "task": "classification",
        "metrics": _classification_metrics(
            test_true, test_pred, test_prob, class_names
        ),
        "class_names": list(class_names),
        "n_classes": len(class_names),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "validation_best": best_score,
        "epochs_completed": epoch + 1,
        "counts": counts,
        **_record_identity(records["test"]),
    }


def _train_static(records, modalities, graph_store, args, device, seed, task):
    counts = {
        partition: {
            "samples": len(rows),
            "patients": len({record.patient_id for record in rows}),
        }
        for partition, rows in records.items()
    }
    if any(not records[partition] for partition in PARTITIONS):
        return _not_evaluable(task, "At least one partition has no usable targets.", counts)
    train_dataset = VisitDataset(records["train"], modalities, graph_store)
    val_dataset = VisitDataset(records["val"], modalities, graph_store)
    test_dataset = VisitDataset(records["test"], modalities, graph_store)
    train_loader = _make_loader(
        train_dataset, args.batch_size, True, seed, _collate_visit, True
    )
    val_loader = _make_loader(
        val_dataset, args.batch_size, False, seed, _collate_visit
    )
    test_loader = _make_loader(
        test_dataset, args.batch_size, False, seed, _collate_visit
    )
    fusion = FusionEncoder(
        modalities,
        args.encoder_hidden_dim,
        args.embed_dim,
        args.edge_threshold,
        args.multimodal_missingness_mask,
    )
    model = StaticRegressionModel(fusion, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            targets = torch.tensor(
                [record.value for record in batch["records"]],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch, device), targets / 100.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        val_true, val_pred = _regression_arrays(model, val_loader, device, False)
        score = _pipeline_r2(val_true, val_pred)
        selection_score = score if np.isfinite(score) else -mean_squared_error(val_true, val_pred)
        if selection_score > best_score + args.min_delta:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"      epoch {epoch + 1:03d}/{args.epochs} "
                f"val_R2={score:.4f} best={best_score:.4f}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            break
    model.load_state_dict(best_state)
    test_true, test_pred = _regression_arrays(model, test_loader, device, False)
    return {
        "status": "ok",
        "task": task,
        "metrics": _regression_metrics(test_true, test_pred),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "validation_best": best_score,
        "epochs_completed": epoch + 1,
        "counts": counts,
        **_record_identity(records["test"]),
    }


def _train_progression(records, modalities, graph_store, args, device, seed, task):
    counts = {
        partition: {
            "samples": len(rows),
            "patients": len({record.patient_id for record in rows}),
        }
        for partition, rows in records.items()
    }
    if any(not records[partition] for partition in PARTITIONS):
        return _not_evaluable(
            task,
            "At least one partition has no usable longitudinal transitions.",
            counts,
        )
    train_dataset = ProgressionDataset(records["train"], modalities, graph_store)
    val_dataset = ProgressionDataset(records["val"], modalities, graph_store)
    test_dataset = ProgressionDataset(records["test"], modalities, graph_store)
    train_loader = _make_loader(
        train_dataset,
        args.progression_batch_size,
        True,
        seed,
        _collate_progression,
        True,
    )
    val_loader = _make_loader(
        val_dataset,
        args.progression_batch_size,
        False,
        seed,
        _collate_progression,
    )
    test_loader = _make_loader(
        test_dataset,
        args.progression_batch_size,
        False,
        seed,
        _collate_progression,
    )
    fusion = FusionEncoder(
        modalities,
        args.encoder_hidden_dim,
        args.embed_dim,
        args.edge_threshold,
        args.multimodal_missingness_mask,
    )
    model = ProgressionModel(fusion, args.recurrent_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch, device)
            loss = F.mse_loss(predictions, batch["targets"].to(device) / 100.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        val_true, val_pred = _regression_arrays(model, val_loader, device, True)
        score = _pipeline_r2(val_true, val_pred)
        selection_score = score if np.isfinite(score) else -mean_squared_error(val_true, val_pred)
        if selection_score > best_score + args.min_delta:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"      epoch {epoch + 1:03d}/{args.epochs} "
                f"val_R2={score:.4f} best={best_score:.4f}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            break
    model.load_state_dict(best_state)
    test_true, test_pred = _regression_arrays(model, test_loader, device, True)
    return {
        "status": "ok",
        "task": task,
        "metrics": _regression_metrics(test_true, test_pred),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "validation_best": best_score,
        "epochs_completed": epoch + 1,
        "counts": counts,
        **_record_identity(records["test"]),
    }


def _run_one(
    fold,
    baseline,
    task,
    split_info,
    labels,
    targets,
    visits,
    graph_store,
    args,
    device,
):
    configuration = BASELINES[baseline]
    modalities = configuration["modalities"]
    eligible_ids = (
        split_info.get("raw_multimodal_ids")
        if baseline == "multimodal"
        else None
    )
    seed = _stable_seed(args.seed, fold, baseline, task)
    _set_seed(seed)
    if task == "classification":
        records = _build_visit_records(
            labels,
            split_info["visit_partition"],
            graph_store,
            modalities,
            eligible_ids=eligible_ids,
        )
        return _train_classification(
            records, modalities, graph_store, args, device, seed
        )
    target_index = 1 if task.endswith("u2") else 2
    if task.startswith("static"):
        records = _build_visit_records(
            targets,
            split_info["visit_partition"],
            graph_store,
            modalities,
            target_index,
            eligible_ids=eligible_ids,
        )
        return _train_static(
            records, modalities, graph_store, args, device, seed, task
        )
    records = _build_progression_records(
        visits,
        split_info["patient_partition"],
        graph_store,
        modalities,
        target_index,
        eligible_ids=eligible_ids,
    )
    return _train_progression(
        records, modalities, graph_store, args, device, seed, task
    )


class _VisitAvailability:
    def __init__(self, visit_ids):
        self.visit_ids = set(visit_ids)

    def has(self, _modality, visit_id):
        return visit_id in self.visit_ids

    def any(self, _modalities, visit_id):
        return visit_id in self.visit_ids


def _load_reconstruction_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "A matched comparison requires the full model's reconstruction artifact: "
            f"{path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected recon_demo.pt to be keyed by visit ID: {path}")
    return set(payload)


def _expected_full_model_identity(
    task,
    split_info,
    reconstruction_ids,
    labels,
    targets,
    visits,
):
    availability = _VisitAvailability(reconstruction_ids)
    if task == "classification":
        records = _build_visit_records(
            labels,
            split_info["visit_partition"],
            availability,
            ("full_model",),
        )
    elif task.startswith("static"):
        target_index = 1 if task.endswith("u2") else 2
        records = _build_visit_records(
            targets,
            split_info["visit_partition"],
            availability,
            ("full_model",),
            target_index,
        )
    else:
        target_index = 1 if task.endswith("u2") else 2
        records = _build_progression_records(
            visits,
            split_info["patient_partition"],
            availability,
            ("full_model",),
            target_index,
        )
    return _record_identity(records["test"])


def _load_full_model_result(
    checkpoint_path,
    task,
    fold,
    expected_use_missingness_mask,
    expected_identity,
):
    if not os.path.exists(checkpoint_path):
        return {
            "status": "missing",
            "task": task,
            "reason": f"Full-model checkpoint not found: {checkpoint_path}",
        }
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("evaluation_split") != "test":
        raise ValueError(
            f"Full-model checkpoint for fold {fold} task {task} does not contain "
            f"untouched test metrics: {checkpoint_path}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"No metrics dictionary in {checkpoint_path}")
    actual_mask = payload.get("use_missingness_mask")
    if actual_mask is None:
        raise ValueError(
            f"Checkpoint does not record use_missingness_mask, so comparability cannot "
            f"be established: {checkpoint_path}"
        )
    if bool(actual_mask) != bool(expected_use_missingness_mask):
        raise ValueError(
            f"Mask-policy mismatch in {checkpoint_path}: checkpoint records "
            f"use_missingness_mask={actual_mask}, expected "
            f"{expected_use_missingness_mask}. Point --full_model_root to the "
            "corresponding full-model run."
        )
    model_state = payload.get("model_state") or {}
    return {
        "status": "ok",
        "task": task,
        "metrics": metrics,
        "source": checkpoint_path,
        "use_missingness_mask": bool(actual_mask),
        "class_names": list(payload.get("class_names", [])),
        "n_classes": payload.get("num_classes", ""),
        "trainable_parameters": sum(
            value.numel() for value in model_state.values() if torch.is_tensor(value)
        ),
        **expected_identity,
    }


def _maskless_full_heads_ready(checkpoints_dir):
    for filename in FULL_MODEL_FILES.values():
        path = os.path.join(checkpoints_dir, filename)
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("evaluation_split") != "test"
            or payload.get("use_missingness_mask") is not False
            or not isinstance(payload.get("metrics"), dict)
        ):
            return False
    return True


def _prepare_maskless_full_model_root(source_fold_dirs, target_root, args):
    pipeline_path = os.path.join(PROJECT_DIR, "model", "run_full_pipeline.py")
    print(
        "No --full_model_root supplied; preparing maskless full-model downstream "
        f"heads in {target_root}.",
        flush=True,
    )
    for fold in args.folds:
        source = source_fold_dirs[fold]
        source_checkpoints = os.path.join(source["fold_dir"], "checkpoints")
        reconstruction_path = os.path.join(source_checkpoints, "recon_demo.pt")
        if not os.path.exists(reconstruction_path):
            raise FileNotFoundError(
                f"Cannot reuse full-model representations for fold {fold}: "
                f"{reconstruction_path}"
            )
        target_fold = os.path.join(target_root, f"fold_{fold}")
        target_checkpoints = os.path.join(target_fold, "checkpoints")
        target_split = os.path.join(target_fold, "site_grouped_split.txt")
        os.makedirs(target_checkpoints, exist_ok=True)
        shutil.copy2(source["split_path"], target_split)
        if _maskless_full_heads_ready(target_checkpoints) and not args.retrain_full_heads:
            print(f"  fold {fold}: resumed completed maskless full heads", flush=True)
            continue
        print(
            f"\n{'=' * 72}\n"
            f"Preparing maskless full-model downstream heads: fold {fold}/{len(args.folds)}",
            flush=True,
        )
        command = [
            sys.executable,
            pipeline_path,
            "--data_csv",
            os.path.abspath(args.data_csv),
            "--split_path",
            target_split,
            "--checkpoints_dir",
            target_checkpoints,
            "--reuse_representations_dir",
            source_checkpoints,
            "--no_missingness_mask",
            "--device",
            args.device,
            "--seed",
            str(args.seed + fold),
            "--cls_epochs",
            str(args.full_cls_epochs),
            "--prog_epochs",
            str(args.full_prog_epochs),
            "--updrs_epochs",
            str(args.full_updrs_epochs),
            "--downstream_lr",
            str(args.full_downstream_lr),
        ]
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        if not _maskless_full_heads_ready(target_checkpoints):
            raise RuntimeError(
                f"Fold {fold} finished without a complete set of maskless test "
                f"checkpoints in {target_checkpoints}"
            )
    return target_root


def _mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _aggregate(all_results, folds):
    grouped = defaultdict(list)
    status = defaultdict(list)
    for result in all_results:
        key = (result["baseline"], result["task"])
        status[key].append(result)
        if result["status"] != "ok":
            continue
        for metric, value in result["metrics"].items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                grouped[(result["baseline"], result["task"], metric)].append(
                    (result["fold"], float(value))
                )
    rows = []
    for baseline in [*BASELINES, "full_model"]:
        for task in TASKS:
            task_results = status[(baseline, task)]
            ok_folds = sorted(result["fold"] for result in task_results if result["status"] == "ok")
            task_status = "ok" if ok_folds == sorted(folds) else "not_evaluable"
            reasons = sorted(
                {
                    result.get("reason", result["status"])
                    for result in task_results
                    if result["status"] != "ok"
                }
            )
            metrics = sorted(
                metric
                for candidate_baseline, candidate_task, metric in grouped
                if candidate_baseline == baseline and candidate_task == task
            )
            if not metrics:
                rows.append(
                    {
                        "baseline": baseline,
                        "task": task,
                        "metric": "",
                        "mean": "",
                        "sd": "",
                        "n_folds": len(ok_folds),
                        "status": task_status,
                        "reason": " | ".join(reasons),
                        "fold_values": "",
                    }
                )
                continue
            for metric in metrics:
                values = sorted(grouped[(baseline, task, metric)])
                numeric = [value for _, value in values]
                mean, sd = _mean_sd(numeric)
                rows.append(
                    {
                        "baseline": baseline,
                        "task": task,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "n_folds": len(numeric),
                        "status": task_status,
                        "reason": " | ".join(reasons),
                        "fold_values": ";".join(
                            f"{fold}:{value:.8g}" for fold, value in values
                        ),
                    }
                )
    return rows


def _summary_lookup(rows):
    return {
        (row["baseline"], row["task"], row["metric"]): row
        for row in rows
        if row["metric"]
    }


def _format_value(row, bold=False):
    if row is None or row["status"] != "ok":
        return "--"
    value = f"{float(row['mean']):.4f} $\\pm$ {float(row['sd']):.4f}"
    return f"\\textbf{{{value}}}" if bold else value


def _latex_escape(text):
    return str(text).replace("_", "\\_")


def _write_latex(path, summary_rows):
    lookup = _summary_lookup(summary_rows)
    baseline_order = [*BASELINES, "full_model"]
    best = {}
    for task, metric, _ in TABLE_COLUMNS:
        candidates = []
        for baseline in baseline_order:
            if task == "classification" and baseline in {"spect", "dti"}:
                continue
            row = lookup.get((baseline, task, metric))
            if row is not None and row["status"] == "ok":
                candidates.append((float(row["mean"]), baseline))
        best[(task, metric)] = max(candidates)[1] if candidates else None

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Site-held-out comparison of raw-graph baselines and the full model. "
            r"Values are mean $\pm$ sample standard deviation across five outer folds. "
            r"Each outer test fold contains sites absent from both training and validation. "
            r"Unimodal baselines use only genuinely observed scans. The naïve multimodal "
            r"baseline zero-fills unavailable modality slots without availability indicators, "
            r"without contrastive alignment or generative reconstruction. "
            r"SPECT and DTI classification ($^\dagger$) is binary Control versus PD because "
            r"no Prodromal acquisitions exist for those modalities; all other classification "
            r"rows are three-class and binary/three-class values are not ranked against each other. "
            r"MRI progression is unavailable because no patient has repeated observed MRI. "
            r"Dashes indicate that a valid evaluation was unavailable.}"
        ),
        r"\label{tab:sota_comparison_site_holdout}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llccccccc@{}}",
        r"\toprule",
        (
            r"& & \multicolumn{3}{c}{\textbf{Classification}} & "
            r"\multicolumn{2}{c}{\textbf{Severity ($R^2$)}} & "
            r"\multicolumn{2}{c}{\textbf{Progression ($R^2$)}} \\"
        ),
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        (
            r"\textbf{Model} & \textbf{Modality} & \textbf{Bal. Acc.} & "
            r"\textbf{Macro F1} & \textbf{Macro AUC} & \textbf{Part II} & "
            r"\textbf{Part III} & \textbf{Part II} & \textbf{Part III} \\"
        ),
        r"\midrule",
    ]
    for baseline in baseline_order:
        if baseline == "full_model":
            model_label = r"\textbf{\modelName}"
            modality_label = r"\textbf{Multi}"
        else:
            model_label = BASELINES[baseline]["model"]
            modality_label = BASELINES[baseline]["modality_label"]
            if baseline in {"spect", "dti"}:
                modality_label += r"$^\dagger$"
        cells = []
        for task, metric, _ in TABLE_COLUMNS:
            row = lookup.get((baseline, task, metric))
            cells.append(
                _format_value(row, bold=best[(task, metric)] == baseline)
            )
        if baseline == "full_model":
            lines.append(r"\midrule")
        lines.append(
            f"{model_label} & {modality_label} & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_diagnostic_report(
    path,
    summary_rows,
    availability_summary,
    comparison_audit_rows,
):
    lookup = _summary_lookup(summary_rows)
    availability_lookup = {
        row["metric"]: row for row in availability_summary
    }
    all_matched = bool(comparison_audit_rows) and all(
        row["matched"] for row in comparison_audit_rows
    )
    lines = [
        "# GCN+GINE versus full-model diagnostic",
        "",
        f"- Exact test-sample hashes matched in every audited fold/task: **{all_matched}**.",
        "- Both compared rows exclude the four explicit availability indicators.",
        "- The raw GCN+GINE encoders are optimized end-to-end for each supervised task; "
        "the full model uses representations constrained by contrastive alignment and "
        "generative reconstruction.",
        "- Zero-filled raw modality blocks still expose acquisition availability, even "
        "without explicit flags.",
        "",
        "## Mean outer-fold differences",
        "",
        "| Metric | Raw GCN+GINE | Full model | Raw − full |",
        "|---|---:|---:|---:|",
    ]
    for task, metric, label in TABLE_COLUMNS:
        raw = lookup.get(("multimodal", task, metric))
        full = lookup.get(("full_model", task, metric))
        if (
            raw is None
            or full is None
            or raw["status"] != "ok"
            or full["status"] != "ok"
        ):
            continue
        raw_mean = float(raw["mean"])
        full_mean = float(full["mean"])
        lines.append(
            f"| {label} | {raw_mean:.4f} | {full_mean:.4f} | "
            f"{raw_mean - full_mean:+.4f} |"
        )
    lines.extend(["", "## Availability-only classification", ""])
    for metric, label in (
        ("bal_acc", "Balanced accuracy"),
        ("f1_macro", "Macro F1"),
        ("auc_macro", "Macro AUC"),
    ):
        row = availability_lookup.get(metric)
        if row:
            lines.append(
                f"- {label}: {float(row['mean']):.4f} ± {float(row['sd']):.4f} "
                f"across {row['n_folds']} folds."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive raw-minus-full difference is an observed performance difference, "
            "not proof that raw fusion learned better pathology. If the availability-only "
            "scores are high, label-dependent acquisition patterns are a supported shortcut "
            "explanation. Remaining explanations include task-specific end-to-end optimization, "
            "contrastive/reconstruction objectives that trade discriminative information for "
            "alignment, and hyperparameters selected for the original rather than external-site "
            "distribution. There is no theoretical requirement that the full model outperform "
            "a supervised raw-graph model on every endpoint.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train GCN/GINE unimodal and naïve multimodal baselines on existing "
            "five-fold site-held-out splits, then aggregate a LaTeX table."
        )
    )
    parser.add_argument("--site_cv_root", required=True)
    parser.add_argument(
        "--full_model_root",
        default=None,
        help=(
            "Optional fold root containing already-trained maskless full-model "
            "downstream checkpoints. If omitted, they are trained automatically "
            "from the fixed representations under --site_cv_root."
        ),
    )
    parser.add_argument(
        "--data_root",
        default=os.path.join(PROJECT_DIR, "data"),
    )
    parser.add_argument(
        "--data_csv",
        default=os.path.join(PROJECT_DIR, "data", "PPMI_Curated_Data_Cut_Public_20251112.csv"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_DIR, "model", "results", "site_holdout_baselines"),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=tuple(BASELINES),
        default=list(BASELINES),
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--progression_batch_size", type=int, default=16)
    parser.add_argument("--encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--recurrent_dim", type=int, default=256)
    parser.add_argument("--edge_threshold", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--full_cls_epochs", type=int, default=100)
    parser.add_argument("--full_prog_epochs", type=int, default=100)
    parser.add_argument("--full_updrs_epochs", type=int, default=100)
    parser.add_argument("--full_downstream_lr", type=float, default=0.01)
    parser.add_argument(
        "--retrain_full_heads",
        action="store_true",
        help="Retrain maskless full-model downstream heads even if complete checkpoints exist.",
    )
    parser.add_argument(
        "--multimodal_missingness_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append availability indicators to the naïve multimodal zero-filled "
            "representation (disabled by default for the matched comparison)."
        ),
    )
    parser.add_argument("--no_cache_graphs", action="store_true")
    parser.add_argument("--restart", action="store_true", help="Ignore completed result JSON files.")
    parser.add_argument(
        "--include_full_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the full-model site-held-out metrics from SITE_CV_ROOT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    site_cv_root = os.path.abspath(args.site_cv_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    fold_dirs = _find_fold_dirs(site_cv_root, args.folds)
    if args.full_model_root:
        full_model_root = os.path.abspath(args.full_model_root)
    elif args.include_full_model:
        full_model_root = _prepare_maskless_full_model_root(
            fold_dirs,
            os.path.join(output_dir, "maskless_full_model"),
            args,
        )
    else:
        full_model_root = site_cv_root
    full_fold_dirs = _find_fold_dirs(full_model_root, args.folds)
    graph_store = GraphStore(args.data_root, cache=not args.no_cache_graphs)
    visit_sites = _load_site_metadata(args.data_csv)
    labels = load_csv_labels(args.data_csv, drop_prodromal=False)
    targets = load_csv_targets(args.data_csv)
    visits = load_csv_visits(args.data_csv)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    print("Starting unified site-held-out baseline experiment.", flush=True)
    print(f"  site folds: {site_cv_root}", flush=True)
    print(f"  full model: {full_model_root}", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  device: {device}", flush=True)
    print(f"  baselines: {', '.join(args.baselines)}", flush=True)
    print(f"  tasks: {', '.join(args.tasks)}", flush=True)

    split_audits = {}
    split_infos = {}
    full_model_results = []
    availability_diagnostics = []
    for fold, paths in fold_dirs.items():
        partitions, patients, visit_partition, patient_partition, comments = _load_split(
            paths["split_path"]
        )
        sites = _audit_site_disjointness(
            partitions,
            visit_sites,
            paths["split_path"],
        )
        split_infos[fold] = {
            "partitions": partitions,
            "patients": patients,
            "visit_partition": visit_partition,
            "patient_partition": patient_partition,
            "raw_multimodal_ids": {
                visit_id
                for partition in PARTITIONS
                for visit_id in partitions[partition]
                if graph_store.any(MODALITIES, visit_id)
            },
        }
        full_partitions, _, _, _, _ = _load_split(
            full_fold_dirs[fold]["split_path"]
        )
        if any(
            set(partitions[partition]) != set(full_partitions[partition])
            for partition in PARTITIONS
        ):
            raise ValueError(
                f"Fold {fold} differs between --site_cv_root and --full_model_root; "
                "the comparison would not use the same splits."
            )
        reconstruction_path = os.path.join(
            full_fold_dirs[fold]["fold_dir"], "checkpoints", "recon_demo.pt"
        )
        reconstruction_ids = _load_reconstruction_ids(reconstruction_path)
        split_infos[fold]["full_model_available_ids"] = reconstruction_ids
        availability_diagnostics.append(
            _availability_only_diagnostic(
                split_infos[fold],
                labels,
                graph_store,
                reconstruction_ids,
                fold,
            )
        )
        split_audits[str(fold)] = {
            "path": paths["split_path"],
            "comments": comments,
            "patient_overlap": 0,
            "site_overlap": 0,
            "sites": {
                partition: sorted(sites[partition]) for partition in PARTITIONS
            },
            "counts": {
                partition: {
                    "visits": len(partitions[partition]),
                    "patients": len(patients[partition]),
                }
                for partition in PARTITIONS
            },
        }
        if args.include_full_model:
            for task, filename in FULL_MODEL_FILES.items():
                identity = _expected_full_model_identity(
                    task,
                    split_infos[fold],
                    reconstruction_ids,
                    labels,
                    targets,
                    visits,
                )
                result = _load_full_model_result(
                    os.path.join(
                        full_fold_dirs[fold]["fold_dir"],
                        "checkpoints",
                        filename,
                    ),
                    task,
                    fold,
                    args.multimodal_missingness_mask,
                    identity,
                )
                result.update(
                    {
                        "fold": fold,
                        "baseline": "full_model",
                        "modalities": list(MODALITIES),
                    }
                )
                full_model_results.append(result)

    all_results = []
    total_runs = len(args.folds) * len(args.baselines) * len(args.tasks)
    run_index = 0
    started_at = time.time()
    for fold in args.folds:
        for baseline in args.baselines:
            for task in args.tasks:
                run_index += 1
                result_path = os.path.join(
                    output_dir,
                    "runs",
                    f"fold_{fold}",
                    baseline,
                    f"{task}.json",
                )
                print(
                    f"\n[{run_index}/{total_runs}] fold={fold} baseline={baseline} task={task}",
                    flush=True,
                )
                if os.path.exists(result_path) and not args.restart:
                    with open(result_path, encoding="utf-8") as handle:
                        result = json.load(handle)
                    if (
                        result.get("initialization") != "random_from_scratch"
                        or result.get("pretrained_weights_loaded") is not False
                    ):
                        raise ValueError(
                            "Refusing to resume a result without verified "
                            f"from-scratch provenance: {result_path}. Use a new "
                            "--output_dir or --restart."
                        )
                    if result.get("multimodal_missingness_mask") != args.multimodal_missingness_mask:
                        raise ValueError(
                            f"Stale result has a different missingness-mask policy: "
                            f"{result_path}. Use a new --output_dir or --restart."
                        )
                    if result.get("status") == "ok" and not result.get("test_ids_sha256"):
                        raise ValueError(
                            f"Stale result predates matched-sample auditing: {result_path}. "
                            "Use a new --output_dir or --restart."
                        )
                    print(f"    resumed: {result['status']}", flush=True)
                else:
                    run_started = time.time()
                    result = _run_one(
                        fold,
                        baseline,
                        task,
                        split_infos[fold],
                        labels,
                        targets,
                        visits,
                        graph_store,
                        args,
                        device,
                    )
                    result.update(
                        {
                            "fold": fold,
                            "baseline": baseline,
                            "modalities": list(BASELINES[baseline]["modalities"]),
                            "seed": _stable_seed(args.seed, fold, baseline, task),
                            "initialization": "random_from_scratch",
                            "pretrained_weights_loaded": False,
                            "weights_shared_with_other_baselines": False,
                            "optimizer_state_loaded": False,
                            "elapsed_seconds": time.time() - run_started,
                            "split_path": fold_dirs[fold]["split_path"],
                            "multimodal_missingness_mask": args.multimodal_missingness_mask,
                        }
                    )
                    _atomic_json(result_path, result)
                    if result["status"] == "ok":
                        print(f"    test metrics: {result['metrics']}", flush=True)
                    else:
                        print(f"    {result['status']}: {result['reason']}", flush=True)
                all_results.append(result)

    if args.include_full_model:
        all_results.extend(full_model_results)

    comparison_audit_rows = []
    if "multimodal" in args.baselines and args.include_full_model:
        indexed = {
            (result["fold"], result["baseline"], result["task"]): result
            for result in all_results
        }
        for fold in args.folds:
            for task in args.tasks:
                raw = indexed[(fold, "multimodal", task)]
                full = indexed[(fold, "full_model", task)]
                matched = (
                    raw.get("test_samples") == full.get("test_samples")
                    and raw.get("test_ids_sha256") == full.get("test_ids_sha256")
                )
                comparison_audit_rows.append(
                    {
                        "fold": fold,
                        "task": task,
                        "matched": matched,
                        "raw_test_samples": raw.get("test_samples", ""),
                        "full_test_samples": full.get("test_samples", ""),
                        "raw_test_patients": raw.get("test_patients", ""),
                        "full_test_patients": full.get("test_patients", ""),
                        "raw_test_ids_sha256": raw.get("test_ids_sha256", ""),
                        "full_test_ids_sha256": full.get("test_ids_sha256", ""),
                        "raw_trainable_parameters": raw.get("trainable_parameters", ""),
                        "full_head_parameters": full.get("trainable_parameters", ""),
                    }
                )
                if not matched:
                    raise RuntimeError(
                        f"Matched-sample audit failed for fold={fold}, task={task}. "
                        "No aggregate table was emitted."
                    )

    per_fold_rows = []
    for result in all_results:
        base = {
            "fold": result["fold"],
            "baseline": result["baseline"],
            "task": result["task"],
            "status": result["status"],
            "reason": result.get("reason", ""),
            "n_classes": result.get("n_classes", ""),
            "class_names": "|".join(result.get("class_names", [])),
            "test_samples": result.get("test_samples", ""),
            "test_patients": result.get("test_patients", ""),
            "test_ids_sha256": result.get("test_ids_sha256", ""),
            "trainable_parameters": result.get("trainable_parameters", ""),
        }
        if result["status"] == "ok":
            for metric, value in result["metrics"].items():
                per_fold_rows.append({**base, "metric": metric, "value": value})
        else:
            per_fold_rows.append({**base, "metric": "", "value": ""})
    summary_rows = _aggregate(all_results, args.folds)
    availability_summary = []
    for metric in ("bal_acc", "f1_macro", "auc_macro"):
        values = [
            float(row[metric])
            for row in availability_diagnostics
            if row.get("status") == "ok" and np.isfinite(row.get(metric, math.nan))
        ]
        if values:
            mean, sd = _mean_sd(values)
            availability_summary.append(
                {"metric": metric, "mean": mean, "sd": sd, "n_folds": len(values)}
            )
    _write_csv(os.path.join(output_dir, "per_fold_metrics.csv"), per_fold_rows)
    _write_csv(
        os.path.join(output_dir, "availability_only_diagnostics.csv"),
        availability_diagnostics,
    )
    _write_csv(
        os.path.join(output_dir, "availability_only_summary.csv"),
        availability_summary,
    )
    _write_csv(
        os.path.join(output_dir, "multimodal_full_comparability_audit.csv"),
        comparison_audit_rows,
    )
    _write_csv(os.path.join(output_dir, "site_holdout_baseline_summary.csv"), summary_rows)
    _write_latex(os.path.join(output_dir, "site_holdout_baseline_table.tex"), summary_rows)
    _write_diagnostic_report(
        os.path.join(output_dir, "gcn_gine_vs_full_diagnostic.md"),
        summary_rows,
        availability_summary,
        comparison_audit_rows,
    )
    _atomic_json(
        os.path.join(output_dir, "run_manifest.json"),
        {
            "site_cv_root": site_cv_root,
            "full_model_root": full_model_root,
            "data_root": os.path.abspath(args.data_root),
            "data_csv": os.path.abspath(args.data_csv),
            "folds": args.folds,
            "baselines": args.baselines,
            "tasks": args.tasks,
            "split_type": "five-fold site-grouped holdout",
            "split_audits": split_audits,
            "configuration": vars(args),
            "elapsed_seconds": time.time() - started_at,
            "standard_deviation": "sample SD across outer folds (ddof=1)",
            "classification_policy": (
                "Three-class metrics are used when Control, PD, and Prodromal are all "
                "represented in train, validation, and test. If Prodromal is absent "
                "from every partition, binary Control-vs-PD metrics are used and marked "
                "with a dagger in the LaTeX table."
            ),
            "multimodal_policy": (
                "Naïve modality-specific encoders; missing modality embeddings are zero-filled; "
                f"availability mask included={args.multimodal_missingness_mask}; no contrastive "
                "alignment and no generative reconstruction. Multimodal raw/full test-ID "
                "hashes must match for every task and fold."
            ),
            "baseline_initialization_policy": (
                "Every fold x baseline x downstream-task run constructs new randomly "
                "initialized encoder and head modules. No pretrained weights, optimizer "
                "state, embeddings, reconstructions, or parameters from another baseline "
                "are loaded. The best validation state restored before test evaluation "
                "belongs only to that same training run."
            ),
            "availability_only_diagnostics": availability_diagnostics,
            "availability_only_summary": availability_summary,
        },
    )
    print("\nAll requested runs completed.", flush=True)
    print(f"Summary CSV: {os.path.join(output_dir, 'site_holdout_baseline_summary.csv')}")
    print(f"LaTeX table: {os.path.join(output_dir, 'site_holdout_baseline_table.tex')}")
    print(
        "Comparability audit: "
        f"{os.path.join(output_dir, 'multimodal_full_comparability_audit.csv')}"
    )
    print(
        "Availability-only diagnostic: "
        f"{os.path.join(output_dir, 'availability_only_diagnostics.csv')}"
    )
    print(
        "Interpretation report: "
        f"{os.path.join(output_dir, 'gcn_gine_vs_full_diagnostic.md')}"
    )


if __name__ == "__main__":
    main()
