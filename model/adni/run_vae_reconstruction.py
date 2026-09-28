#!/usr/bin/env python3
"""Leakage-safe DTI-to-fMRI variational reconstruction and classification."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adni.atlas import AAL90_REGIONS
from model.adni.data import LABEL_NAMES, load_paired_connectomes
from model.adni.models import ADNIMissingModalityVAE
from model.adni.run_classification import (
    classifier_metrics,
    encode_indices,
    save_manifest,
    set_seed,
    train_classifier,
    train_encoders,
)
from model.adni.tasks import TASKS, prepare_task_labels


CLASSIFIER_REPRESENTATIONS = (
    "dti",
    "generated_fmri_mean",
    "latent_mu",
    "real_fmri",
    "real_multimodal",
)
REPRESENTATIONS = (
    "dti",
    "generated_fmri_mean",
    "generated_fmri_mc_ensemble",
    "latent_mu",
    "real_fmri",
    "real_multimodal",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri_path", default=str(PROJECT_ROOT / "data" / "ADNI_fMRI.npy"))
    parser.add_argument("--dti_path", default=str(PROJECT_ROOT / "data" / "ADNI_DTI.npy"))
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "model" / "checkpoints" / "adni_vae_reconstruction"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=sorted(TASKS), default="ad_vs_cn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    # Connectivity encoders. These names intentionally match run_classification.
    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--contrastive_lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--encoder_dropout", type=float, default=0.15)
    parser.add_argument("--edge_drop_prob", type=float, default=0.15)
    parser.add_argument("--feature_jitter", type=float, default=0.01)

    # Conditional VAE.
    parser.add_argument("--vae_epochs", type=int, default=150)
    parser.add_argument("--vae_patience", type=int, default=20)
    parser.add_argument("--vae_lr", type=float, default=3e-4)
    parser.add_argument("--vae_kl_weight", type=float, default=1e-3)
    parser.add_argument("--vae_kl_warmup", type=int, default=20)
    parser.add_argument("--dti_self_reconstruction_weight", type=float, default=0.25)
    parser.add_argument("--generator_hidden_dim", type=int, default=128)
    parser.add_argument("--generator_heads", type=int, default=4)
    parser.add_argument("--generator_layers", type=int, default=2)
    parser.add_argument("--generator_registers", type=int, default=2)
    parser.add_argument("--generator_mlp_depth", type=int, default=2)
    parser.add_argument("--generator_dropout", type=float, default=0.10)
    parser.add_argument("--mc_samples", type=int, default=20)

    # Identical downstream classifier settings for every representation.
    parser.add_argument("--classifier_epochs", type=int, default=150)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def build_vae(args, device):
    return ADNIMissingModalityVAE(
        embed_dim=args.embed_dim,
        hidden_dim=args.generator_hidden_dim,
        num_heads=args.generator_heads,
        num_layers=args.generator_layers,
        num_registers=args.generator_registers,
        mlp_depth=args.generator_mlp_depth,
        dropout=args.generator_dropout,
    ).to(device)


def reconstruction_error(prediction, target):
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target, dim=1)
    cosine = 1.0 - F.cosine_similarity(prediction, target).mean()
    mse = F.mse_loss(prediction, target)
    return cosine + mse


def kl_divergence(mu, logvar):
    # Mean over both subjects and latent dimensions keeps beta independent of embed_dim.
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()


def train_vae(train_dti, train_fmri, val_dti, val_fmri, args, device):
    model = build_vae(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.vae_lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_dti, train_fmri),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state, best_value, stale = None, float("inf"), 0

    for epoch in range(args.vae_epochs):
        model.train()
        total = 0.0
        warmup = min(1.0, (epoch + 1) / max(args.vae_kl_warmup, 1))
        for dti_batch, fmri_batch in loader:
            dti_batch, fmri_batch = dti_batch.to(device), fmri_batch.to(device)
            generated_fmri, reconstructed_dti, mu, logvar = model.reconstruct_fmri_from_dti(
                dti_batch, sample=True
            )
            fmri_loss = reconstruction_error(generated_fmri, fmri_batch)
            dti_loss = reconstruction_error(reconstructed_dti, dti_batch)
            kl = kl_divergence(mu, logvar)
            loss = (
                fmri_loss
                + args.dti_self_reconstruction_weight * dti_loss
                + warmup * args.vae_kl_weight * kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            generated, _, _, _ = model.reconstruct_fmri_from_dti(
                val_dti.to(device), sample=False
            )
            validation_loss = reconstruction_error(generated, val_fmri.to(device)).item()
        if validation_loss < best_value - 1e-6:
            best_value = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"    vae {epoch + 1:03d}/{args.vae_epochs}: "
                f"loss={total / max(len(loader), 1):.4f} "
                f"val_dti_to_fmri={validation_loss:.4f}",
                flush=True,
            )
        if stale >= args.vae_patience:
            break

    if best_state is None:
        raise RuntimeError("VAE training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_value


def infer_vae(model, dti, fmri, args, device, include_mc=False):
    model.eval()
    with torch.no_grad():
        generated, _, mu, logvar = model.reconstruct_fmri_from_dti(
            dti.to(device), sample=False
        )
        generated = F.normalize(generated, dim=1)
        mu_features = F.normalize(mu, dim=1)
        target = F.normalize(fmri.to(device), dim=1)
        per_subject_cosine = F.cosine_similarity(generated, target)
        per_subject_mse = (generated - target).square().mean(dim=1)
        result = {
            "generated": generated.cpu(),
            "mu": mu_features.cpu(),
            "cosine": per_subject_cosine.cpu(),
            "mse": per_subject_mse.cpu(),
            "posterior_std": torch.exp(0.5 * logvar).mean(dim=1).cpu(),
        }
        if include_mc:
            samples = []
            for _ in range(args.mc_samples):
                sampled, _, _, _ = model.reconstruct_fmri_from_dti(
                    dti.to(device), sample=True
                )
                samples.append(F.normalize(sampled, dim=1).cpu())
            result["mc_generated"] = torch.stack(samples)
    return result


def representation_sets(encoded, inferred):
    fmri, dti = encoded
    return {
        "dti": dti,
        "generated_fmri_mean": inferred["generated"],
        "latent_mu": inferred["mu"],
        "real_fmri": fmri,
        "real_multimodal": torch.cat([fmri, dti], dim=1),
    }


def aggregate_metrics(rows):
    ignored = {"fold", "representation", "best_validation_macro_f1"}
    result = {}
    for representation in REPRESENTATIONS:
        subset = [row for row in rows if row["representation"] == representation]
        result[representation] = {
            name: {
                "mean": float(np.mean([row[name] for row in subset])),
                "std": float(np.std([row[name] for row in subset], ddof=1)),
            }
            for name in subset[0]
            if name not in ignored
        }
    return result


def compare_generated_with_dti(rows):
    """Report paired outer-fold gains; positive values favor reconstruction."""
    comparisons = {}
    excluded = {"fold", "representation", "best_validation_macro_f1"}
    for generated_name in ("generated_fmri_mean", "generated_fmri_mc_ensemble"):
        differences = {}
        for fold in sorted({row["fold"] for row in rows}):
            dti_row = next(
                row for row in rows
                if row["fold"] == fold and row["representation"] == "dti"
            )
            generated_row = next(
                row for row in rows
                if row["fold"] == fold and row["representation"] == generated_name
            )
            for metric in (generated_row.keys() & dti_row.keys()) - excluded:
                differences.setdefault(metric, []).append(
                    generated_row[metric] - dti_row[metric]
                )
        comparisons[generated_name] = {
            metric: {
                "mean_difference": float(np.mean(values)),
                "std_difference": float(np.std(values, ddof=1)),
                "folds_better": int(np.sum(np.asarray(values) > 0)),
                "folds_equal": int(np.sum(np.asarray(values) == 0)),
                "folds_total": len(values),
            }
            for metric, values in differences.items()
        }
    return comparisons


def main():
    args = parse_args()
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("--val_fraction must be between 0 and 0.5.")
    if args.mc_samples < 1:
        raise ValueError("--mc_samples must be positive.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmri, dti, source_labels = load_paired_connectomes(args.fmri_path, args.dti_path)
    labels, eligible_indices, task = prepare_task_labels(source_labels, args.task)
    save_manifest(output_dir / "adni_manifest.csv", source_labels)
    with (output_dir / "aal90_regions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["zero_based_index", "aal_index", "region"])
        for index, region in enumerate(AAL90_REGIONS):
            writer.writerow([index, index + 1, region])

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = requested

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics, predictions, reconstruction_rows, fold_splits = [], [], [], []
    print(
        f"ADNI VAE task={task.name}: {len(eligible_indices)} subjects | "
        f"labels={dict(Counter(labels[eligible_indices].tolist()))} | device={device}",
        flush=True,
    )

    for fold, (development_positions, test_positions) in enumerate(
        splitter.split(eligible_indices, labels[eligible_indices]), start=1
    ):
        development_indices = eligible_indices[development_positions]
        test_indices = eligible_indices[test_positions]
        train_indices, val_indices = train_test_split(
            development_indices,
            test_size=args.val_fraction,
            random_state=args.seed + fold,
            stratify=labels[development_indices],
        )
        print(
            f"\nFold {fold}/{args.folds}: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}",
            flush=True,
        )
        fold_splits.append({
            "fold": fold,
            "train_indices": train_indices.tolist(),
            "validation_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist(),
        })
        set_seed(args.seed + fold)
        encoders = train_encoders(fmri, dti, labels, train_indices, args, device)

        encoded = {}
        encoded_ids = {}
        encoded_labels = {}
        for split_name, split_indices in (
            ("train", train_indices), ("validation", val_indices), ("test", test_indices)
        ):
            ids, features, split_labels = encode_indices(
                encoders, fmri, dti, labels, split_indices, args, device
            )
            encoded_ids[split_name] = ids
            encoded_labels[split_name] = split_labels
            encoded[split_name] = (features[:, : args.embed_dim], features[:, args.embed_dim :])

        vae, best_reconstruction = train_vae(
            encoded["train"][1], encoded["train"][0],
            encoded["validation"][1], encoded["validation"][0],
            args, device,
        )
        inferred = {
            name: infer_vae(
                vae, values[1], values[0], args, device, include_mc=(name == "test")
            )
            for name, values in encoded.items()
        }
        split_representations = {
            name: representation_sets(encoded[name], inferred[name]) for name in encoded
        }

        classifiers = {}
        for representation in CLASSIFIER_REPRESENTATIONS:
            # Identical initialization and minibatch order across representations.
            set_seed(args.seed + 10_000 + fold)
            classifier, best_val_f1 = train_classifier(
                split_representations["train"][representation], encoded_labels["train"],
                split_representations["validation"][representation], encoded_labels["validation"],
                args, device, task.num_classes,
            )
            classifiers[representation] = classifier
            classifier.eval()
            with torch.no_grad():
                logits = classifier(split_representations["test"][representation].to(device)).cpu()
            metrics, predicted, probabilities = classifier_metrics(
                logits, encoded_labels["test"], task.num_classes
            )
            metrics.update({
                "fold": fold,
                "representation": representation,
                "best_validation_macro_f1": best_val_f1,
            })
            fold_metrics.append(metrics)
            print(
                f"    {representation}: "
                + " | ".join(
                    f"{key}={value:.4f}"
                    for key, value in metrics.items()
                    if key not in {"fold", "representation"}
                ),
                flush=True,
            )
            for sample_id, truth, pred, probability in zip(
                encoded_ids["test"].tolist(), encoded_labels["test"].tolist(),
                predicted.tolist(), probabilities.tolist(),
            ):
                row = {
                    "fold": fold,
                    "representation": representation,
                    "sample_index": sample_id,
                    "sample_id": f"ADNI_{sample_id:04d}",
                    "true_label": truth,
                    "true_diagnosis": task.label_names[truth],
                    "predicted_label": pred,
                    "predicted_diagnosis": task.label_names[pred],
                }
                for class_id, probability_value in enumerate(probability):
                    row[f"prob_class_{class_id}"] = probability_value
                predictions.append(row)

        # Use the mean-trained generated-fMRI classifier and average its posterior
        # predictions over stochastic VAE draws at test time.
        generated_classifier = classifiers["generated_fmri_mean"]
        with torch.no_grad():
            mc_probabilities = torch.stack([
                F.softmax(generated_classifier(sample.to(device)), dim=1).cpu()
                for sample in inferred["test"]["mc_generated"]
            ]).mean(dim=0)
        mc_metrics, mc_predicted, mc_probability_array = classifier_metrics(
            mc_probabilities.clamp_min(1e-12).log(), encoded_labels["test"], task.num_classes
        )
        mc_metrics.update({
            "fold": fold,
            "representation": "generated_fmri_mc_ensemble",
            "best_validation_macro_f1": next(
                row["best_validation_macro_f1"]
                for row in reversed(fold_metrics)
                if row["fold"] == fold and row["representation"] == "generated_fmri_mean"
            ),
        })
        fold_metrics.append(mc_metrics)
        print(
            "    generated_fmri_mc_ensemble: "
            + " | ".join(
                f"{key}={value:.4f}"
                for key, value in mc_metrics.items()
                if key not in {"fold", "representation"}
            ),
            flush=True,
        )
        for sample_id, truth, pred, probability in zip(
            encoded_ids["test"].tolist(), encoded_labels["test"].tolist(),
            mc_predicted.tolist(), mc_probability_array.tolist(),
        ):
            row = {
                "fold": fold,
                "representation": "generated_fmri_mc_ensemble",
                "sample_index": sample_id,
                "sample_id": f"ADNI_{sample_id:04d}",
                "true_label": truth,
                "true_diagnosis": task.label_names[truth],
                "predicted_label": pred,
                "predicted_diagnosis": task.label_names[pred],
            }
            for class_id, probability_value in enumerate(probability):
                row[f"prob_class_{class_id}"] = probability_value
            predictions.append(row)

        for position, sample_id in enumerate(encoded_ids["test"].tolist()):
            reconstruction_rows.append({
                "fold": fold,
                "sample_index": sample_id,
                "sample_id": f"ADNI_{sample_id:04d}",
                "cosine_similarity": float(inferred["test"]["cosine"][position]),
                "mse": float(inferred["test"]["mse"][position]),
                "posterior_mean_std": float(inferred["test"]["posterior_std"][position]),
            })

        torch.save(
            {
                "fmri_encoder": encoders["fmri"].state_dict(),
                "dti_encoder": encoders["dti"].state_dict(),
                "vae": vae.state_dict(),
                "classifiers": {name: model.state_dict() for name, model in classifiers.items()},
                "fold": fold,
                "task": task.name,
                "label_names": task.label_names,
                "best_validation_reconstruction_loss": best_reconstruction,
                "train_indices": train_indices.tolist(),
                "validation_indices": val_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "config": vars(args),
            },
            output_dir / f"fold_{fold}.pt",
        )

    aggregate = aggregate_metrics(fold_metrics)
    reconstruction_summary = {
        name: {
            "mean": float(np.mean([row[name] for row in reconstruction_rows])),
            "std": float(np.std([row[name] for row in reconstruction_rows], ddof=1)),
        }
        for name in ("cosine_similarity", "mse", "posterior_mean_std")
    }
    summary = {
        "dataset": "ADNI paired AAL90 fMRI+DTI",
        "task": task.name,
        "task_description": task.description,
        "subjects": len(eligible_indices),
        "label_names": task.label_names,
        "method": "DTI-conditioned variational reconstruction of fMRI embeddings",
        "leakage_control": (
            "Encoders, VAE, preprocessing, and classifiers are fitted independently "
            "inside each outer fold; the VAE never receives diagnosis labels."
        ),
        "fold_splits": fold_splits,
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
        "generated_vs_dti": compare_generated_with_dti(fold_metrics),
        "reconstruction": reconstruction_summary,
        "config": vars(args),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    for filename, rows in (
        ("predictions.csv", predictions),
        ("reconstruction_metrics.csv", reconstruction_rows),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("\nCross-validation summary", flush=True)
    for representation, metrics in aggregate.items():
        print(f"  {representation}", flush=True)
        for name, values in metrics.items():
            print(f"    {name}: {values['mean']:.4f} +/- {values['std']:.4f}", flush=True)
    print(f"Artifacts saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
