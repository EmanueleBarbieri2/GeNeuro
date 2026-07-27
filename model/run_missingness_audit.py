#!/usr/bin/env python3
"""Audit how much downstream performance is available from acquisition patterns alone."""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.downstream.downstream_classification import load_csv_labels  # noqa: E402
from model.downstream.downstream_progression import load_csv_visits  # noqa: E402
from model.downstream.downstream_updrs import load_csv_targets  # noqa: E402


MODALITIES = ("SPECT", "MRI", "fMRI", "DTI")
CLASS_NAMES = ("Control", "PD", "Prodromal")
TARGETS = ((1, "U2_ADL"), (2, "U3_Motor"))
PARTITIONS = ("train", "val", "test")
PROGRESSION_BASELINES = ("missingness_only", "time_only", "missingness_plus_time")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _patient_id(visit_id):
    return str(visit_id).split("_", 1)[0]


def _load_split(path):
    split = {partition: set() for partition in PARTITIONS}
    mode = None
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line in {"train_ids:", "val_ids:", "test_ids:"}:
                mode = {
                    "train_ids:": "train",
                    "val_ids:": "val",
                    "test_ids:": "test",
                }[line]
            elif line and not line.startswith("#"):
                if mode is None:
                    raise ValueError(f"ID encountered before a partition header in {path}: {line}")
                split[mode].add(line)

    if not split["train"] or not split["val"]:
        raise ValueError("The split must contain non-empty train_ids and val_ids partitions.")

    patient_sets = {
        partition: {_patient_id(visit_id) for visit_id in ids}
        for partition, ids in split.items()
    }
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Patient leakage between {left} and {right}: "
                    f"{len(overlap)} overlapping patients (e.g. {examples})."
                )

    visit_partition = {}
    patient_partition = {}
    for partition, ids in split.items():
        for visit_id in ids:
            visit_partition[visit_id] = partition
        for patient in patient_sets[partition]:
            patient_partition[patient] = partition
    return split, visit_partition, patient_partition


def _load_masks(data_root, recon_path=None):
    if recon_path:
        if not os.path.exists(recon_path):
            raise FileNotFoundError(f"Explicit reconstruction artifact not found: {recon_path}")
        payload = torch.load(recon_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("recon_demo.pt must contain a dictionary keyed by visit ID.")
        masks = {}
        for visit_id, entry in payload.items():
            if not isinstance(entry, dict) or "real" not in entry:
                continue
            real_modalities = entry["real"]
            masks[str(visit_id)] = np.asarray(
                [0.0 if modality in real_modalities else 1.0 for modality in MODALITIES],
                dtype=np.float32,
            )
        source = os.path.abspath(recon_path)
    else:
        availability = {}
        for modality in MODALITIES:
            modality_dir = os.path.join(data_root, modality)
            if not os.path.isdir(modality_dir):
                raise FileNotFoundError(f"Modality directory not found: {modality_dir}")
            availability[modality] = {
                os.path.splitext(filename)[0]
                for filename in os.listdir(modality_dir)
                if filename.endswith(".pt")
            }
        visit_ids = set().union(*availability.values())
        masks = {
            visit_id: np.asarray(
                [
                    0.0 if visit_id in availability[modality] else 1.0
                    for modality in MODALITIES
                ],
                dtype=np.float32,
            )
            for visit_id in visit_ids
        }
        source = os.path.abspath(data_root)
    if not masks:
        raise ValueError("No observed/reconstructed modality indicators were found.")
    return masks, source


def _evaluation_partition(split):
    return "test" if split["test"] else "val"


def _pipeline_r2(y_true, y_pred):
    """Match the repository's 1 - MSE / unbiased-variance calculation."""
    if len(y_true) < 2:
        return math.nan
    variance = float(np.var(y_true, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        return math.nan
    return float(1.0 - mean_squared_error(y_true, y_pred) / (variance + 1e-8))


def _classification_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _regression_metrics(y_true, y_pred):
    return {
        "pipeline_r2": _pipeline_r2(y_true, y_pred),
        "standard_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else math.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def _bootstrap_patient_ci(y_true, y_pred, patient_ids, metric_fn, iterations, seed):
    if iterations <= 0:
        return {}
    grouped_indices = defaultdict(list)
    for index, patient in enumerate(patient_ids):
        grouped_indices[str(patient)].append(index)
    patients = np.asarray(sorted(grouped_indices))
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(iterations):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_indices = np.concatenate(
            [np.asarray(grouped_indices[patient], dtype=int) for patient in sampled_patients]
        )
        try:
            metrics = metric_fn(y_true[sampled_indices], y_pred[sampled_indices])
        except ValueError:
            continue
        for metric, value in metrics.items():
            if np.isfinite(value):
                values[metric].append(float(value))
    intervals = {}
    for metric, samples in values.items():
        if samples:
            intervals[f"{metric}_ci_low"] = float(np.percentile(samples, 2.5))
            intervals[f"{metric}_ci_high"] = float(np.percentile(samples, 97.5))
    return intervals


def _metric_row(task, baseline, target, split_name, seed, y_true, y_pred, patient_ids, args):
    is_classification = task == "classification"
    metric_fn = _classification_metrics if is_classification else _regression_metrics
    metrics = metric_fn(y_true, y_pred)
    metrics.update(
        _bootstrap_patient_ci(
            y_true,
            y_pred,
            patient_ids,
            metric_fn,
            args.bootstrap_iterations if seed in (None, "ensemble") else 0,
            args.seed,
        )
    )
    return {
        "task": task,
        "baseline": baseline,
        "target": target,
        "evaluation_split": split_name,
        "seed": "" if seed is None else seed,
        "n_samples": len(y_true),
        "n_patients": len(set(patient_ids)),
        **metrics,
    }


def _append_predictions(
    rows,
    task,
    baseline,
    target,
    split_name,
    seed,
    ids,
    patient_ids,
    y_true,
    y_pred,
):
    for sample_id, patient, truth, prediction in zip(ids, patient_ids, y_true, y_pred):
        rows.append(
            {
                "task": task,
                "baseline": baseline,
                "target": target,
                "evaluation_split": split_name,
                "seed": "" if seed is None else seed,
                "sample_id": sample_id,
                "patient_id": patient,
                "target_value": truth,
                "prediction": prediction,
            }
        )


def _tabular_records(values_by_id, masks, visit_partition):
    records = {partition: [] for partition in PARTITIONS}
    for visit_id, value in values_by_id.items():
        partition = visit_partition.get(visit_id)
        if partition is None or visit_id not in masks:
            continue
        records[partition].append(
            {
                "id": visit_id,
                "patient_id": _patient_id(visit_id),
                "mask": masks[visit_id],
                "value": value,
            }
        )
    return records


def _arrays(records, partition, target_index=None):
    selected = records[partition]
    if target_index is not None:
        selected = [
            record
            for record in selected
            if np.isfinite(float(record["value"][target_index]))
        ]
    if not selected:
        raise ValueError(f"No usable samples in the {partition} partition.")
    x = np.stack([record["mask"] for record in selected])
    if target_index is None:
        y = np.asarray([record["value"] for record in selected])
    else:
        y = np.asarray([float(record["value"][target_index]) for record in selected])
    ids = np.asarray([record["id"] for record in selected])
    patients = np.asarray([record["patient_id"] for record in selected])
    return x, y, ids, patients


def _fit_classification(records, split_name, args, metric_rows, prediction_rows, coefficient_rows):
    x_train, y_train, _, _ = _arrays(records, "train")
    x_eval, y_eval, ids_eval, patients_eval = _arrays(records, split_name)
    print(
        f"[classification] train={len(y_train)} visits, "
        f"{split_name}={len(y_eval)} visits",
        flush=True,
    )
    missing_train_classes = set(CLASS_NAMES) - set(y_train)
    missing_eval_classes = set(CLASS_NAMES) - set(y_eval)
    if missing_train_classes:
        raise ValueError(f"Classification training data lack classes: {sorted(missing_train_classes)}")
    if missing_eval_classes:
        raise ValueError(f"Classification evaluation data lack classes: {sorted(missing_eval_classes)}")

    feature_sets = [("missingness_only_logistic", np.arange(len(MODALITIES)))]
    feature_sets.extend(
        (f"{modality}_availability_only", np.asarray([index]))
        for index, modality in enumerate(MODALITIES)
    )
    for baseline, feature_indices in feature_sets:
        print(f"[classification] fitting {baseline}...", flush=True)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=args.logistic_max_iter,
            random_state=args.seed,
        )
        model.fit(x_train[:, feature_indices], y_train)
        predictions = model.predict(x_eval[:, feature_indices])
        baseline_metrics = _classification_metrics(y_eval, predictions)
        print(
            f"[classification] {baseline}: "
            f"balanced_accuracy={baseline_metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={baseline_metrics['macro_f1']:.4f}",
            flush=True,
        )
        metric_rows.append(
            _metric_row(
                "classification",
                baseline,
                "diagnosis",
                split_name,
                None,
                y_eval,
                predictions,
                patients_eval,
                args,
            )
        )
        _append_predictions(
            prediction_rows,
            "classification",
            baseline,
            "diagnosis",
            split_name,
            None,
            ids_eval,
            patients_eval,
            y_eval,
            predictions,
        )
        selected_modalities = [MODALITIES[index] for index in feature_indices]
        for class_name, coefficients, intercept in zip(
            model.classes_,
            model.coef_,
            model.intercept_,
        ):
            for modality, coefficient in zip(selected_modalities, coefficients):
                coefficient_rows.append(
                    {
                        "task": "classification",
                        "baseline": baseline,
                        "target": class_name,
                        "feature": f"{modality}_missing",
                        "coefficient": float(coefficient),
                    }
                )
            coefficient_rows.append(
                {
                    "task": "classification",
                    "baseline": baseline,
                    "target": class_name,
                    "feature": "intercept",
                    "coefficient": float(intercept),
                }
            )

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(x_train, y_train)
    dummy_predictions = dummy.predict(x_eval)
    metric_rows.append(
        _metric_row(
            "classification",
            "class_prior",
            "diagnosis",
            split_name,
            None,
            y_eval,
            dummy_predictions,
            patients_eval,
            args,
        )
    )


def _fit_static_regression(
    records,
    target_index,
    target_name,
    split_name,
    args,
    metric_rows,
    prediction_rows,
    coefficient_rows,
):
    x_train, y_train, _, _ = _arrays(records, "train", target_index)
    x_eval, y_eval, ids_eval, patients_eval = _arrays(records, split_name, target_index)
    print(
        f"[static {target_name}] fitting missingness-only ridge: "
        f"train={len(y_train)}, {split_name}={len(y_eval)}",
        flush=True,
    )

    model = Ridge(alpha=args.ridge_alpha)
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    baseline_metrics = _regression_metrics(y_eval, predictions)
    print(
        f"[static {target_name}] R2={baseline_metrics['pipeline_r2']:.4f}, "
        f"MAE={baseline_metrics['mae']:.4f}",
        flush=True,
    )
    metric_rows.append(
        _metric_row(
            "static_severity",
            "missingness_only_ridge",
            target_name,
            split_name,
            None,
            y_eval,
            predictions,
            patients_eval,
            args,
        )
    )
    _append_predictions(
        prediction_rows,
        "static_severity",
        "missingness_only_ridge",
        target_name,
        split_name,
        None,
        ids_eval,
        patients_eval,
        y_eval,
        predictions,
    )
    for modality, coefficient in zip(MODALITIES, model.coef_):
        coefficient_rows.append(
            {
                "task": "static_severity",
                "baseline": "missingness_only_ridge",
                "target": target_name,
                "feature": f"{modality}_missing",
                "coefficient": float(coefficient),
            }
        )
    coefficient_rows.append(
        {
            "task": "static_severity",
            "baseline": "missingness_only_ridge",
            "target": target_name,
            "feature": "intercept",
            "coefficient": float(model.intercept_),
        }
    )

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, y_train)
    dummy_predictions = dummy.predict(x_eval)
    metric_rows.append(
        _metric_row(
            "static_severity",
            "training_mean",
            target_name,
            split_name,
            None,
            y_eval,
            dummy_predictions,
            patients_eval,
            args,
        )
    )


@dataclass
class ProgressionSample:
    masks: torch.Tensor
    times: torch.Tensor
    next_delta: float
    target: float
    patient_id: str
    sample_id: str


def _build_progression_samples(visits, masks, patient_partition, target_index):
    samples = {partition: [] for partition in PARTITIONS}
    for patient, patient_visits in visits.items():
        partition = patient_partition.get(str(patient))
        if partition is None:
            continue
        ordered = sorted(patient_visits, key=lambda visit: visit["year"])
        valid = [visit for visit in ordered if visit["key"] in masks]
        for index in range(len(valid) - 1):
            target = float(valid[index + 1]["targets"][target_index])
            if not np.isfinite(target):
                continue
            history = valid[: index + 1]
            samples[partition].append(
                ProgressionSample(
                    masks=torch.tensor(
                        np.stack([masks[visit["key"]] for visit in history]),
                        dtype=torch.float32,
                    ),
                    times=torch.tensor([visit["year"] for visit in history], dtype=torch.float32),
                    next_delta=float(valid[index + 1]["year"] - valid[index]["year"]),
                    target=target,
                    patient_id=str(patient),
                    sample_id=str(valid[index + 1]["key"]),
                )
            )
    return samples


class _ProgressionDataset(Dataset):
    def __init__(self, samples, baseline):
        self.samples = samples
        self.baseline = baseline

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.baseline == "time_only":
            features = torch.zeros((len(sample.masks), 1), dtype=torch.float32)
        else:
            features = sample.masks
        use_time = self.baseline in {"time_only", "missingness_plus_time"}
        times = sample.times if use_time else torch.zeros_like(sample.times)
        next_delta = sample.next_delta if use_time else 0.0
        return features, times, next_delta, sample.target


def _collate_progression(batch):
    features, times, next_deltas, targets = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in features], dtype=torch.long)
    padded_features = pad_sequence(features, batch_first=True)
    padded_deltas = torch.zeros(len(batch), padded_features.shape[1])
    for index, sequence_times in enumerate(times):
        if len(sequence_times) > 1:
            padded_deltas[index, 1 : len(sequence_times)] = (
                sequence_times[1:] - sequence_times[:-1]
            )
    return (
        padded_features,
        padded_deltas,
        lengths,
        torch.tensor(next_deltas, dtype=torch.float32).unsqueeze(1),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
    )


class _ProgressionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_encoder = nn.Linear(1, 8)
        self.gru = nn.GRU(hidden_dim + 8, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, max(8, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def forward(self, features, deltas, lengths, next_delta):
        compressed = self.compressor(features)
        time_embedding = F.gelu(self.time_encoder(deltas.unsqueeze(-1)))
        packed = pack_padded_sequence(
            torch.cat([compressed, time_embedding], dim=2),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(torch.cat([hidden[-1], next_delta], dim=1))


def _progression_predict(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for features, deltas, lengths, next_delta, target in loader:
            output = model(
                features.to(device),
                deltas.to(device),
                lengths,
                next_delta.to(device),
            )
            predictions.append(output.cpu().numpy().ravel() * 100.0)
            targets.append(target.numpy().ravel())
    return np.concatenate(targets), np.concatenate(predictions)


def _train_progression(train_samples, val_samples, baseline, seed, args, device):
    _set_seed(seed)
    train_dataset = _ProgressionDataset(train_samples, baseline)
    val_dataset = _ProgressionDataset(val_samples, baseline)
    input_dim = 1 if baseline == "time_only" else len(MODALITIES)
    model = _ProgressionGRU(input_dim, args.progression_hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.progression_lr, weight_decay=1e-2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_progression,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_progression,
    )

    best_state = None
    best_score = -float("inf")
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(args.progression_epochs):
        completed_epochs = epoch + 1
        model.train()
        for features, deltas, lengths, next_delta, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                features.to(device),
                deltas.to(device),
                lengths,
                next_delta.to(device),
            )
            loss = F.mse_loss(prediction, target.to(device) / 100.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_true, val_prediction = _progression_predict(model, val_loader, device)
        score = _pipeline_r2(val_true, val_prediction)
        if not np.isfinite(score):
            score = -mean_squared_error(val_true, val_prediction)
        if score > best_score + args.early_stopping_min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                break
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(
                f"    epoch {epoch + 1:03d}/{args.progression_epochs} "
                f"val_R2={_pipeline_r2(val_true, val_prediction):.4f} "
                f"best={best_score:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError(f"No valid progression model was trained for {baseline}, seed {seed}.")
    model.load_state_dict(best_state)
    print(
        f"    finished after {completed_epochs} epochs; best validation score={best_score:.4f}",
        flush=True,
    )
    return model


def _fit_progression(
    samples,
    target_name,
    split_name,
    args,
    device,
    metric_rows,
    prediction_rows,
):
    if not samples["train"] or not samples["val"] or not samples[split_name]:
        raise ValueError(f"Insufficient longitudinal samples for progression {target_name}.")

    evaluation_samples = samples[split_name]
    evaluation_ids = np.asarray([sample.sample_id for sample in evaluation_samples])
    evaluation_patients = np.asarray([sample.patient_id for sample in evaluation_samples])
    evaluation_targets = np.asarray([sample.target for sample in evaluation_samples])

    training_mean = float(np.mean([sample.target for sample in samples["train"]]))
    mean_predictions = np.full_like(evaluation_targets, training_mean, dtype=float)
    metric_rows.append(
        _metric_row(
            "progression",
            "training_mean",
            target_name,
            split_name,
            None,
            evaluation_targets,
            mean_predictions,
            evaluation_patients,
            args,
        )
    )

    for baseline in args.progression_baselines:
        per_seed_predictions = []
        for seed in args.seeds:
            print(
                f"[progression {target_name}] fitting {baseline}, seed={seed} "
                f"(train={len(samples['train'])}, val={len(samples['val'])}, "
                f"{split_name}={len(evaluation_samples)})",
                flush=True,
            )
            model = _train_progression(
                samples["train"],
                samples["val"],
                baseline,
                seed,
                args,
                device,
            )
            evaluation_loader = DataLoader(
                _ProgressionDataset(evaluation_samples, baseline),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=_collate_progression,
            )
            y_true, predictions = _progression_predict(model, evaluation_loader, device)
            seed_metrics = _regression_metrics(y_true, predictions)
            print(
                f"[progression {target_name}] {baseline}, seed={seed}: "
                f"R2={seed_metrics['pipeline_r2']:.4f}, "
                f"MAE={seed_metrics['mae']:.4f}",
                flush=True,
            )
            per_seed_predictions.append(predictions)
            metric_rows.append(
                _metric_row(
                    "progression",
                    baseline,
                    target_name,
                    split_name,
                    seed,
                    y_true,
                    predictions,
                    evaluation_patients,
                    args,
                )
            )
            _append_predictions(
                prediction_rows,
                "progression",
                baseline,
                target_name,
                split_name,
                seed,
                evaluation_ids,
                evaluation_patients,
                y_true,
                predictions,
            )

        ensemble_predictions = np.mean(np.stack(per_seed_predictions), axis=0)
        ensemble_metrics = _regression_metrics(evaluation_targets, ensemble_predictions)
        print(
            f"[progression {target_name}] {baseline} ensemble: "
            f"R2={ensemble_metrics['pipeline_r2']:.4f}, "
            f"MAE={ensemble_metrics['mae']:.4f}",
            flush=True,
        )
        metric_rows.append(
            _metric_row(
                "progression",
                baseline,
                target_name,
                split_name,
                "ensemble",
                evaluation_targets,
                ensemble_predictions,
                evaluation_patients,
                args,
            )
        )
        _append_predictions(
            prediction_rows,
            "progression",
            baseline,
            target_name,
            split_name,
            "ensemble",
            evaluation_ids,
            evaluation_patients,
            evaluation_targets,
            ensemble_predictions,
        )


def _availability_tables(labels, masks, visit_partition):
    counts = defaultdict(Counter)
    patterns = defaultdict(Counter)
    for visit_id, label in labels.items():
        partition = visit_partition.get(visit_id)
        if partition is None or visit_id not in masks:
            continue
        mask = masks[visit_id]
        group = (partition, label)
        counts[group]["samples"] += 1
        for modality, missing in zip(MODALITIES, mask):
            if missing == 0:
                counts[group][modality] += 1
        pattern = "".join(str(int(value)) for value in mask)
        patterns[group][pattern] += 1

    availability_rows = []
    pattern_rows = []
    for (partition, label), counter in sorted(counts.items()):
        total = counter["samples"]
        for modality in MODALITIES:
            observed = counter[modality]
            availability_rows.append(
                {
                    "partition": partition,
                    "class": label,
                    "modality": modality,
                    "samples": total,
                    "observed": observed,
                    "observed_fraction": observed / total,
                }
            )
        for pattern, count in sorted(patterns[(partition, label)].items()):
            pattern_rows.append(
                {
                    "partition": partition,
                    "class": label,
                    "pattern": pattern,
                    "pattern_order": "SPECT,MRI,fMRI,DTI; 1=missing/reconstructed",
                    "count": count,
                    "fraction_within_class": count / total,
                }
            )
    return availability_rows, pattern_rows


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


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train missingness-only baselines for diagnosis, static severity, and "
            "longitudinal progression using patient-disjoint partitions."
        )
    )
    parser.add_argument(
        "--data_csv",
        default=os.path.join(PROJECT_DIR, "data", "PPMI_Curated_Data_Cut_Public_20251112.csv"),
    )
    parser.add_argument(
        "--split_path",
        default=os.path.join(PROJECT_DIR, "data", "unified_split_fold0.txt"),
    )
    parser.add_argument(
        "--data_root",
        default=os.path.join(PROJECT_DIR, "data"),
        help="Directory containing SPECT, MRI, fMRI, and DTI subdirectories.",
    )
    parser.add_argument(
        "--recon_path",
        help="Optional recon_demo.pt override; normally unnecessary.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="Deprecated compatibility option; checkpoints are not required.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_DIR, "model", "results", "missingness_audit"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--logistic_max_iter", type=int, default=2000)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--progression_epochs", type=int, default=100)
    parser.add_argument("--progression_lr", type=float, default=1e-3)
    parser.add_argument("--progression_hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-5)
    parser.add_argument(
        "--progression_baselines",
        nargs="+",
        choices=PROGRESSION_BASELINES,
        default=list(PROGRESSION_BASELINES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _set_seed(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    print("Starting missingness-only downstream audit.", flush=True)
    print(f"  data_root: {os.path.abspath(args.data_root)}", flush=True)
    print(f"  split: {os.path.abspath(args.split_path)}", flush=True)
    print(f"  device: {device}", flush=True)
    split, visit_partition, patient_partition = _load_split(args.split_path)
    split_name = _evaluation_partition(split)
    masks, mask_source = _load_masks(args.data_root, args.recon_path)
    print(
        f"Loaded {len(masks)} modality-availability patterns from {mask_source}.",
        flush=True,
    )
    print(
        "Patient-disjoint split: "
        + ", ".join(
            f"{partition}={len(ids)} visits/"
            f"{len({_patient_id(visit_id) for visit_id in ids})} patients"
            for partition, ids in split.items()
            if ids
        ),
        flush=True,
    )
    print(f"Final evaluation partition: {split_name}", flush=True)
    labels = load_csv_labels(args.data_csv, drop_prodromal=False)
    targets = load_csv_targets(args.data_csv)
    visits = load_csv_visits(args.data_csv)

    metric_rows = []
    prediction_rows = []
    coefficient_rows = []

    classification_records = _tabular_records(labels, masks, visit_partition)
    _fit_classification(
        classification_records,
        split_name,
        args,
        metric_rows,
        prediction_rows,
        coefficient_rows,
    )

    severity_records = _tabular_records(targets, masks, visit_partition)
    for target_index, target_name in TARGETS:
        _fit_static_regression(
            severity_records,
            target_index,
            target_name,
            split_name,
            args,
            metric_rows,
            prediction_rows,
            coefficient_rows,
        )
        progression_samples = _build_progression_samples(
            visits,
            masks,
            patient_partition,
            target_index,
        )
        _fit_progression(
            progression_samples,
            target_name,
            split_name,
            args,
            device,
            metric_rows,
            prediction_rows,
        )

    availability_rows, pattern_rows = _availability_tables(
        labels,
        masks,
        visit_partition,
    )
    _write_csv(os.path.join(output_dir, "metrics.csv"), metric_rows)
    _write_csv(os.path.join(output_dir, "predictions.csv"), prediction_rows)
    _write_csv(os.path.join(output_dir, "coefficients.csv"), coefficient_rows)
    _write_csv(os.path.join(output_dir, "availability_by_class.csv"), availability_rows)
    _write_csv(os.path.join(output_dir, "missingness_patterns_by_class.csv"), pattern_rows)

    summary = {
        "data_csv": os.path.abspath(args.data_csv),
        "split_path": os.path.abspath(args.split_path),
        "mask_source": mask_source,
        "evaluation_split": split_name,
        "device": str(device),
        "mask_order": list(MODALITIES),
        "mask_semantics": {"0": "observed/real", "1": "missing/reconstructed"},
        "split_counts": {partition: len(ids) for partition, ids in split.items()},
        "split_patient_counts": {
            partition: len({_patient_id(visit_id) for visit_id in ids})
            for partition, ids in split.items()
        },
        "configuration": vars(args),
        "metrics": metric_rows,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)

    print(f"Missingness audit complete. Evaluation partition: {split_name}")
    print(f"Patient-disjoint split validation passed.")
    print(f"Results: {output_dir}")
    for row in metric_rows:
        if row["seed"] not in ("", "ensemble"):
            continue
        metric_text = (
            f"balanced_accuracy={row['balanced_accuracy']:.4f}"
            if "balanced_accuracy" in row
            else f"R2={row.get('pipeline_r2', math.nan):.4f}, MAE={row.get('mae', math.nan):.4f}"
        )
        print(
            f"  {row['task']} | {row['target']} | {row['baseline']} "
            f"| {row['seed'] or 'single'} | {metric_text}"
        )


if __name__ == "__main__":
    main()
