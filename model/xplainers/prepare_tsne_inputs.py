#!/usr/bin/env python3
"""Export latent representations from trained GeNeuro checkpoints for t-SNE."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.downstream.downstream_classification import (  # noqa: E402
    Classifier,
    SmartClassDataset,
    load_csv_labels,
)
from model.downstream.downstream_progression import (  # noqa: E402
    ForecastingGRU,
    SmartSequenceDataset,
    collate_fn,
    load_csv_visits,
)
from model.downstream.downstream_updrs import (  # noqa: E402
    Regressor,
    SmartUpdrsDataset,
    load_csv_targets,
)


MODALITIES = ("SPECT", "MRI", "fMRI", "DTI")
TARGET_FILES = {
    "U2_ADL": ("static_U2_ADL.pt", "prog_U2_ADL.pt"),
    "U3_Motor": ("static_U3_Motor.pt", "prog_U3_Motor.pt"),
}


def _torch_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_partition_ids(split_path, partition):
    if partition == "all":
        return None
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(f"--partition {partition} requires an existing --split_path.")
    ids = set()
    mode = None
    with open(split_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "train_ids:":
                mode = "train"
            elif line == "val_ids:":
                mode = "val"
            elif line == "test_ids:":
                mode = "test"
            elif line and not line.startswith("#") and mode == partition:
                ids.add(line)
    if not ids:
        raise ValueError(f"No IDs found for partition {partition!r} in {split_path}")
    return ids


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _save(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Saved {path}: {len(next(iter(arrays.values())))} samples")


def _export_real_embeddings(embeddings_path, output_dir, selected_ids):
    payload = _torch_load(embeddings_path)
    features, modalities, ids = [], [], []
    for embedding, modality, subject_id in zip(
        payload["embeddings"], payload["labels"], payload["ids"]
    ):
        if selected_ids is not None and subject_id not in selected_ids:
            continue
        features.append(_as_numpy(embedding.flatten()))
        modalities.append(str(modality))
        ids.append(str(subject_id))
    if not features:
        raise ValueError("No real embeddings remain after partition filtering.")
    _save(
        os.path.join(output_dir, "realEmb.npz"),
        features=np.stack(features),
        modality=np.asarray(modalities, dtype=str),
        ids=np.asarray(ids, dtype=str),
    )


def _export_reconstructed_embeddings(recon_path, output_dir, selected_ids, active_modalities):
    payload = _torch_load(recon_path)
    features, modalities, sources, ids = [], [], [], []
    for subject_id, entry in sorted(payload.items()):
        if selected_ids is not None and subject_id not in selected_ids:
            continue
        for modality in active_modalities:
            if modality not in entry["recon"]:
                continue
            features.append(_as_numpy(entry["recon"][modality].flatten()))
            modalities.append(modality)
            sources.append("observed" if modality in entry.get("real", {}) else "reconstructed")
            ids.append(f"{subject_id}_{modality}")
    if not features:
        raise ValueError("No reconstructed embeddings remain after partition filtering.")
    _save(
        os.path.join(output_dir, "genEmb.npz"),
        features=np.stack(features),
        modality=np.asarray(modalities, dtype=str),
        source=np.asarray(sources, dtype=str),
        ids=np.asarray(ids, dtype=str),
    )


def _batched_mlp_features(model, inputs, batch_size, device):
    hidden_rows, prediction_rows = [], []
    model.eval()
    feature_extractor = model.net[:-1]
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            hidden = feature_extractor(batch)
            prediction = model.net[-1](hidden)
            hidden_rows.append(hidden.cpu())
            prediction_rows.append(prediction.cpu())
    return torch.cat(hidden_rows), torch.cat(prediction_rows)


def _export_classification(
    data_csv,
    representation_path,
    checkpoint_path,
    output_dir,
    selected_ids,
    active_modalities,
    batch_size,
    device,
):
    checkpoint = _torch_load(checkpoint_path)
    class_names = checkpoint.get("class_names", ["Control", "PD", "Prodromal"])
    labels = load_csv_labels(data_csv, drop_prodromal=(len(class_names) == 2))
    dataset = SmartClassDataset(
        representation_path,
        labels,
        active_modalities,
        class_names=class_names,
        use_mask=True,
    )
    samples = [
        sample for sample in dataset.samples
        if selected_ids is None or sample[0] in selected_ids
    ]
    if not samples:
        raise ValueError("No classification samples remain after partition filtering.")
    ids = [sample[0] for sample in samples]
    true_indices = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    inputs = torch.stack([sample[2] for sample in samples])

    model = Classifier(
        checkpoint.get("input_dim", inputs.shape[1]),
        num_classes=checkpoint.get("num_classes", len(class_names)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    hidden, logits = _batched_mlp_features(model, inputs, batch_size, device)
    predicted_indices = logits.argmax(dim=1)
    _save(
        os.path.join(output_dir, "classEmb.npz"),
        features=_as_numpy(hidden),
        class_label=np.asarray([class_names[index] for index in true_indices.tolist()], dtype=str),
        predicted_class=np.asarray([class_names[index] for index in predicted_indices.tolist()], dtype=str),
        ids=np.asarray(ids, dtype=str),
    )


def _export_severity(
    data_csv,
    representation_path,
    checkpoint_path,
    output_dir,
    selected_ids,
    active_modalities,
    batch_size,
    device,
    output_name,
):
    checkpoint = _torch_load(checkpoint_path)
    target_index = int(checkpoint["target_idx"])
    targets = load_csv_targets(data_csv)
    dataset = SmartUpdrsDataset(
        representation_path,
        targets,
        active_modalities,
        use_mask=True,
    )
    samples = [
        sample for sample in dataset.samples
        if selected_ids is None or sample[0] in selected_ids
    ]
    if not samples:
        raise ValueError("No severity samples remain after partition filtering.")
    ids = [sample[0] for sample in samples]
    inputs = torch.stack([sample[1] for sample in samples])
    observed = torch.stack([sample[2] for sample in samples])[:, target_index]

    model = Regressor(inputs.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    hidden, predictions = _batched_mlp_features(model, inputs, batch_size, device)
    _save(
        os.path.join(output_dir, output_name),
        features=_as_numpy(hidden),
        predicted_severity=_as_numpy(predictions[:, 0] * 100.0),
        observed_severity=_as_numpy(observed),
        ids=np.asarray(ids, dtype=str),
    )


def _progression_hidden(model, x, deltas, lengths, delta_next):
    batch, sequence, dimensions = x.shape
    compressed = model.compressor(x.reshape(-1, dimensions)).view(batch, sequence, -1)
    time_embedding = F.gelu(model.time_encoder(deltas.unsqueeze(-1)))
    recurrent_input = torch.cat([compressed, time_embedding], dim=2)
    packed = pack_padded_sequence(
        recurrent_input,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    _, hidden_states = model.gru(packed)
    hidden = hidden_states[-1]
    prediction = model.head(torch.cat([hidden, delta_next], dim=1))
    return hidden, prediction


def _progression_metadata(representation_path, visits):
    """Mirror SmartSequenceDataset's rolling-window order for plot annotations."""
    representations = _torch_load(representation_path)
    patient_ids, trajectory_order, target_visit_ids = [], [], []
    for patient_id, patient_visits in visits.items():
        ordered_visits = sorted(patient_visits, key=lambda visit: visit["year"])
        valid_visits = [
            visit for visit in ordered_visits
            if visit["key"] in representations
        ]
        for index in range(len(valid_visits) - 1):
            patient_ids.append(str(patient_id))
            trajectory_order.append(index)
            target_visit_ids.append(str(valid_visits[index + 1]["key"]))
    return patient_ids, trajectory_order, target_visit_ids


def _export_progression(
    data_csv,
    representation_path,
    checkpoint_path,
    output_dir,
    selected_ids,
    active_modalities,
    batch_size,
    device,
    output_name,
):
    checkpoint = _torch_load(checkpoint_path)
    target_index = int(checkpoint["target_idx"])
    visits = load_csv_visits(data_csv)
    if selected_ids is not None:
        selected_patients = {subject_id.split("_", 1)[0] for subject_id in selected_ids}
        visits = {patient: values for patient, values in visits.items() if patient in selected_patients}
    patient_ids, trajectory_order, target_visit_ids = _progression_metadata(
        representation_path,
        visits,
    )
    dataset = SmartSequenceDataset(
        representation_path,
        visits,
        active_modalities,
        use_mask=True,
    )
    if not len(dataset):
        raise ValueError("No longitudinal sequences remain after partition filtering.")

    input_dim = int(checkpoint.get("input_dim", dataset[0][0][0].shape[0]))
    state = checkpoint["model_state"]
    hidden_dim = int(state["compressor.0.weight"].shape[0])
    model = ForecastingGRU(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    hidden_rows, prediction_rows, observed_rows = [], [], []
    with torch.no_grad():
        for x, deltas, lengths, delta_next, targets in loader:
            hidden, prediction = _progression_hidden(
                model,
                x.to(device),
                deltas.to(device),
                lengths,
                delta_next.to(device),
            )
            hidden_rows.append(hidden.cpu())
            prediction_rows.append(prediction.cpu()[:, 0] * 100.0)
            observed_rows.append(targets[:, target_index])
    hidden = torch.cat(hidden_rows)
    predictions = torch.cat(prediction_rows)
    observed = torch.cat(observed_rows)
    if len(patient_ids) != len(hidden):
        raise RuntimeError(
            "Progression metadata does not match SmartSequenceDataset ordering: "
            f"{len(patient_ids)} annotations for {len(hidden)} sequences."
        )
    _save(
        os.path.join(output_dir, output_name),
        features=_as_numpy(hidden),
        predicted_future_severity=_as_numpy(predictions),
        observed_future_severity=_as_numpy(observed),
        patient_id=np.asarray(patient_ids, dtype=str),
        trajectory_order=np.asarray(trajectory_order, dtype=np.int64),
        ids=np.asarray(target_visit_ids, dtype=str),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract the five latent-space arrays directly from trained GeNeuro checkpoints."
    )
    parser.add_argument("--checkpoints_dir", required=True)
    parser.add_argument(
        "--data_csv",
        default=os.path.join(PROJECT_DIR, "data", "PPMI_Curated_Data_Cut_Public_20251112.csv"),
    )
    parser.add_argument("--split_path")
    parser.add_argument("--partition", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_DIR, "data", "visualizations"),
    )
    target_choices = ["both", *TARGET_FILES]
    parser.add_argument("--severity_target", choices=target_choices, default="both")
    parser.add_argument("--progression_target", choices=target_choices, default="both")
    parser.add_argument("--exclude_modality", nargs="+", default=[], choices=MODALITIES)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoints_dir = os.path.abspath(args.checkpoints_dir)
    output_dir = os.path.abspath(args.output_dir)
    selected_ids = _load_partition_ids(args.split_path, args.partition)
    active_modalities = [modality for modality in MODALITIES if modality not in args.exclude_modality]
    if not active_modalities:
        raise ValueError("At least one active modality is required.")
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    embeddings_path = os.path.join(checkpoints_dir, "embeddings.pt")
    reconstruction_path = os.path.join(checkpoints_dir, "recon_demo.pt")
    classifier_path = os.path.join(checkpoints_dir, "classifier.pt")
    _export_real_embeddings(embeddings_path, output_dir, selected_ids)
    _export_reconstructed_embeddings(
        reconstruction_path,
        output_dir,
        selected_ids,
        active_modalities,
    )
    _export_classification(
        args.data_csv,
        reconstruction_path,
        classifier_path,
        output_dir,
        selected_ids,
        active_modalities,
        args.batch_size,
        device,
    )
    severity_targets = TARGET_FILES if args.severity_target == "both" else [args.severity_target]
    progression_targets = (
        TARGET_FILES if args.progression_target == "both" else [args.progression_target]
    )
    for target_name in severity_targets:
        _export_severity(
            args.data_csv,
            reconstruction_path,
            os.path.join(checkpoints_dir, TARGET_FILES[target_name][0]),
            output_dir,
            selected_ids,
            active_modalities,
            args.batch_size,
            device,
            f"sevEmb_{target_name}.npz",
        )
    for target_name in progression_targets:
        _export_progression(
            args.data_csv,
            reconstruction_path,
            os.path.join(checkpoints_dir, TARGET_FILES[target_name][1]),
            output_dir,
            selected_ids,
            active_modalities,
            args.batch_size,
            device,
            f"progEmb_{target_name}.npz",
        )
    print(f"All t-SNE input artifacts are ready in {output_dir}")


if __name__ == "__main__":
    main()
