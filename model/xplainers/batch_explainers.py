import os
import json
import csv
import argparse
from typing import Dict, List

import torch

from model.xplainers.explain_updrs import MOD_ORDER, TARGETS as UPDRS_TARGETS, build_models as build_updrs_models, explain_subject_with_models
from model.xplainers.explain_progression import TARGETS as PROG_TARGETS, build_models as build_prog_models, explain_transition_with_models
from model.xplainers.explain_classification import explain_subject as explain_classification_subject
try:
    from model.downstream.downstream_classification import load_embeddings, load_csv_labels, ProM3E_Generator, Classifier
except ModuleNotFoundError:
    from downstream.downstream_classification import load_embeddings, load_csv_labels, ProM3E_Generator, Classifier


def list_subject_ids(data_root: str, modalities: List[str]) -> List[str]:
    subject_ids = set()
    for mod in modalities:
        mod_dir = os.path.join(data_root, mod)
        if not os.path.isdir(mod_dir):
            continue
        for fname in os.listdir(mod_dir):
            if fname.endswith(".pt"):
                subject_ids.add(fname[:-3])
    return sorted(subject_ids)

def load_valid_subject_ids_from_csv(csv_path: str) -> set:
    valid_ids = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patno = row.get("PATNO")
            event_id = row.get("EVENT_ID")
            if patno and event_id:
                valid_ids.add(f"{patno}_{event_id}")
    return valid_ids


def load_atlas_centroids(path: str) -> List[Dict[str, str]]:
    atlas = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            atlas.append(row)
    return atlas


def _try_parse_int(value: str):
    try:
        return int(value)
    except Exception:
        return None


def load_freesurfer_lut(path: str) -> Dict[int, Dict[str, object]]:
    lut = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            label = _try_parse_int(parts[0])
            if label is None:
                continue
            # Find last 4 integer tokens as RGBA
            rgba = []
            for token in reversed(parts[1:]):
                val = _try_parse_int(token)
                if val is None:
                    break
                rgba.append(val)
                if len(rgba) == 4:
                    break
            if len(rgba) != 4:
                continue
            rgba = list(reversed(rgba))
            name_tokens = parts[1:len(parts) - 4]
            if not name_tokens:
                continue
            name = " ".join(name_tokens)
            lut[label] = {
                "name": name,
                "color": {"r": rgba[0], "g": rgba[1], "b": rgba[2], "a": rgba[3]},
            }
    return lut


SPECT_NODE_NAMES = [
    "striatum_bilat",
    "striatum_L",
    "striatum_R",
    "caudate_L",
    "putamen_L",
    "caudate_R",
    "putamen_R",
]


def build_roi_info(index: int, atlas: List[Dict[str, str]], lut: Dict[int, Dict[str, object]], modality: str | None = None):
    if modality == "SPECT":
        if index < 0 or index >= len(SPECT_NODE_NAMES):
            return None
        return {
            "node_index": index,
            "label": None,
            "roi_name": SPECT_NODE_NAMES[index],
            "mni": None,
            "num_voxels": None,
            "lut_name": None,
            "color": None,
        }
    if index < 0 or index >= len(atlas):
        return None
    row = atlas[index]
    label = int(float(row.get("label", -1))) if row.get("label") is not None else -1
    roi_name = row.get("roi_name", "unknown")
    info = {
        "node_index": index,
        "label": label,
        "roi_name": roi_name,
        "mni": {
            "x": float(row.get("mni_x", 0.0)),
            "y": float(row.get("mni_y", 0.0)),
            "z": float(row.get("mni_z", 0.0)),
        },
        "num_voxels": int(float(row.get("num_voxels", 0))) if row.get("num_voxels") is not None else 0,
    }
    if label in lut:
        info["lut_name"] = lut[label]["name"]
        info["color"] = lut[label]["color"]
    else:
        info["lut_name"] = None
        info["color"] = None
    return info


def topk_nodes(node_imp, atlas, lut, topk: int, modality: str | None = None,
               node_value: torch.Tensor | None = None,
               node_grad: torch.Tensor | None = None,
               node_contrib: torch.Tensor | None = None):
    # Robust type check
    if not hasattr(node_imp, 'numel') or not hasattr(node_imp, 'shape'):
        return []
    if node_imp.numel() == 0:
        return []
    # If node_imp is a scalar, make it 1D
    if node_imp.dim() == 0:
        node_imp = node_imp.unsqueeze(0)
    k = min(topk, node_imp.numel())
    indices = torch.topk(node_imp, k).indices.tolist()
    nodes = []
    for idx in indices:
        info = build_roi_info(idx, atlas, lut, modality=modality)
        if info is None:
            continue
        try:
            info["importance"] = float(node_imp[idx].item())
        except Exception:
            info["importance"] = float(node_imp[idx]) if isinstance(node_imp[idx], (float, int)) else None
        if node_value is not None:
            try:
                info["value_mean"] = float(node_value[idx].item())
            except Exception:
                info["value_mean"] = float(node_value[idx]) if isinstance(node_value[idx], (float, int)) else None
        if node_grad is not None:
            try:
                info["grad_mean"] = float(node_grad[idx].item())
            except Exception:
                info["grad_mean"] = float(node_grad[idx]) if isinstance(node_grad[idx], (float, int)) else None
        if node_contrib is not None:
            try:
                info["contrib_mean"] = float(node_contrib[idx].item())
            except Exception:
                info["contrib_mean"] = float(node_contrib[idx]) if isinstance(node_contrib[idx], (float, int)) else None
        nodes.append(info)
    return nodes


def topk_edges(edge_imp, edge_index, atlas, lut, topk: int, modality: str | None = None,
               edge_value: torch.Tensor | None = None,
               edge_grad: torch.Tensor | None = None,
               edge_contrib: torch.Tensor | None = None):
    # Robust type check
    if edge_imp is None or edge_index is None:
        return []
    if not hasattr(edge_imp, 'numel') or not hasattr(edge_imp, 'shape'):
        return []
    if edge_imp.numel() == 0:
        return []
    k = min(topk, edge_imp.numel())
    indices = torch.topk(edge_imp, k).indices.tolist()
    edges = []
    for e_idx in indices:
        try:
            u = int(edge_index[0, e_idx].item())
            v = int(edge_index[1, e_idx].item())
        except Exception:
            u = int(edge_index[0, e_idx]) if hasattr(edge_index[0, e_idx], '__int__') else None
            v = int(edge_index[1, e_idx]) if hasattr(edge_index[1, e_idx], '__int__') else None
        info_u = build_roi_info(u, atlas, lut, modality=modality)
        info_v = build_roi_info(v, atlas, lut, modality=modality)
        if info_u is None or info_v is None:
            continue
        edge_info = {
            "edge_index": e_idx,
            "importance": float(edge_imp[e_idx].item()) if hasattr(edge_imp[e_idx], 'item') else float(edge_imp[e_idx]) if isinstance(edge_imp[e_idx], (float, int)) else None,
            "u": info_u,
            "v": info_v,
        }
        if edge_value is not None:
            try:
                edge_info["value"] = float(edge_value[e_idx].item())
            except Exception:
                edge_info["value"] = float(edge_value[e_idx]) if isinstance(edge_value[e_idx], (float, int)) else None
        if edge_grad is not None:
            try:
                edge_info["grad"] = float(edge_grad[e_idx].item())
            except Exception:
                edge_info["grad"] = float(edge_grad[e_idx]) if isinstance(edge_grad[e_idx], (float, int)) else None
        if edge_contrib is not None:
            try:
                edge_info["contrib"] = float(edge_contrib[e_idx].item())
            except Exception:
                edge_info["contrib"] = float(edge_contrib[e_idx]) if isinstance(edge_contrib[e_idx], (float, int)) else None
        edges.append(edge_info)
    return edges


def save_report(report: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)


def generate_reports_for_subject(
    subject_id, data_root, atlas, lut, out_dir, topk,
    updrs_models=None, prog_models=None, delta_t=1.0,
    do_updrs=True, do_progression=True, device="cpu"
):
    # Classification explainer
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
        embeddings_path = os.path.join(base_dir, 'embeddings.pt')
        generator_ckpt = os.path.join(base_dir, 'prom3e_generator.pt')
        classifier_ckpt = os.path.join(base_dir, 'classifier.pt')
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'PPMI_Curated_Data_Cut_Public_20251112.csv'))
        print(f"[classification] Checking required files for {subject_id}:")
        print(f"  embeddings_path: {embeddings_path}  exists: {os.path.exists(embeddings_path)}")
        print(f"  generator_ckpt: {generator_ckpt}  exists: {os.path.exists(generator_ckpt)}")
        print(f"  classifier_ckpt: {classifier_ckpt}  exists: {os.path.exists(classifier_ckpt)}")
        print(f"  csv_path: {csv_path}  exists: {os.path.exists(csv_path)}")
        required_files = [embeddings_path, generator_ckpt, classifier_ckpt, csv_path]
        if all(os.path.exists(p) for p in required_files):
            print(f"[classification] All required files found for {subject_id}")
            labels = load_csv_labels(csv_path)
            embeddings_by_id = load_embeddings(embeddings_path)
            generator = ProM3E_Generator(embed_dim=1024)
            generator.load_state_dict(torch.load(generator_ckpt, map_location='cpu')["model_state"])
            classifier = Classifier(4*1024+4)
            classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu')["model_state"])
            classifier.eval()
            generator.eval()
            label = labels.get(subject_id, None)
            result = explain_classification_subject(subject_id, embeddings_by_id, label, generator, classifier, device='cpu')
            if result:
                reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'explainer_reports', 'classification', subject_id))
                os.makedirs(reports_dir, exist_ok=True)
                out_path = os.path.join(reports_dir, 'classification.json')
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"[classification] Saved for {subject_id} at {out_path}")
            else:
                print(f"[classification] No result for {subject_id} (likely missing embeddings)")
        else:
            for p in required_files:
                if not os.path.exists(p):
                    print(f"[classification] MISSING: {p}")
            print(f"[classification] Skipped {subject_id}: missing required files")
    except Exception as e:
        print(f"[WARN] Classification explainer failed for {subject_id}: {e}")
    if do_updrs and updrs_models is not None:
        encoders, generator, regressor = updrs_models
        for target in UPDRS_TARGETS:
            target_idx = UPDRS_TARGETS.index(target)
            for m in list(encoders.values()) + [generator, regressor]:
                m.zero_grad(set_to_none=True)
            results = explain_subject_with_models(
                subject_id,
                data_root,
                encoders,
                generator,
                regressor,
                target_idx=target_idx,
                device=device,
                include_edge_index=True,
            )
            report = {
                "subject_id": subject_id,
                "task": "updrs",
                "target": target,
                "prediction": results["prediction"],
                "node_indexing": "node index corresponds to row order in atlas_centroids.csv",
                "modalities": {},
            }
            for mod in MOD_ORDER:
                mod_nodes = results["node_importance"].get(mod)
                mod_edges = results["edge_importance"].get(mod)
                node_value = results.get("node_value", {}).get(mod)
                node_grad = results.get("node_grad", {}).get(mod)
                node_contrib = results.get("node_contrib", {}).get(mod)
                edge_value = results.get("edge_value", {}).get(mod)
                edge_grad = results.get("edge_grad", {}).get(mod)
                edge_contrib = results.get("edge_contrib", {}).get(mod)
                edge_index = results.get("edge_index", {}).get(mod)
                if mod_nodes is None and mod_edges is None:
                    continue
                report["modalities"][mod] = {
                    "top_nodes": topk_nodes(
                        mod_nodes,
                        atlas,
                        lut,
                        topk,
                        modality=mod,
                        node_value=node_value,
                        node_grad=node_grad,
                        node_contrib=node_contrib,
                    ) if mod_nodes is not None else [],
                    "top_edges": topk_edges(
                        mod_edges,
                        edge_index,
                        atlas,
                        lut,
                        topk,
                        modality=mod,
                        edge_value=edge_value,
                        edge_grad=edge_grad,
                        edge_contrib=edge_contrib,
                    ) if mod_edges is not None else [],
                }
            reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'explainer_reports', 'updrs', subject_id))
            os.makedirs(reports_dir, exist_ok=True)
            out_path = os.path.join(reports_dir, f"{target}.json")
            save_report(report, out_path)
            torch.save(results, os.path.join(reports_dir, f"{target}.pt"))

    if do_progression and prog_models is not None:
        encoders, generator, regressor = prog_models
        for target in PROG_TARGETS:
            target_idx = PROG_TARGETS.index(target)
            for m in list(encoders.values()) + [generator, regressor]:
                m.zero_grad(set_to_none=True)
            results = explain_transition_with_models(
                subject_id,
                data_root,
                encoders,
                generator,
                regressor,
                target_idx=target_idx,
                delta_t=delta_t,
                device=device,
                include_edge_index=True,
            )
            report = {
                "subject_id": subject_id,
                "task": "progression",
                "target": target,
                "prediction": results["prediction"],
                "delta_t": float(delta_t),
                "node_indexing": "node index corresponds to row order in atlas_centroids.csv",
                "modalities": {},
            }
            for mod in MOD_ORDER:
                mod_nodes = results["node_importance"].get(mod)
                mod_edges = results["edge_importance"].get(mod)
                node_value = results.get("node_value", {}).get(mod)
                node_grad = results.get("node_grad", {}).get(mod)
                node_contrib = results.get("node_contrib", {}).get(mod)
                edge_value = results.get("edge_value", {}).get(mod)
                edge_grad = results.get("edge_grad", {}).get(mod)
                edge_contrib = results.get("edge_contrib", {}).get(mod)
                edge_index = results.get("edge_index", {}).get(mod)
                if mod_nodes is None and mod_edges is None:
                    continue
                report["modalities"][mod] = {
                    "top_nodes": topk_nodes(
                        mod_nodes,
                        atlas,
                        lut,
                        topk,
                        modality=mod,
                        node_value=node_value,
                        node_grad=node_grad,
                        node_contrib=node_contrib,
                    ) if mod_nodes is not None else [],
                    "top_edges": topk_edges(
                        mod_edges,
                        edge_index,
                        atlas,
                        lut,
                        topk,
                        modality=mod,
                        edge_value=edge_value,
                        edge_grad=edge_grad,
                        edge_contrib=edge_contrib,
                    ) if mod_edges is not None else [],
                }
            reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'explainer_reports', 'progression', subject_id))
            os.makedirs(reports_dir, exist_ok=True)
            out_path = os.path.join(reports_dir, f"{target}.json")
            save_report(report, out_path)
            torch.save(results, os.path.join(reports_dir, f"{target}.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--atlas", default="./atlas_centroids.csv")
    parser.add_argument("--lut", default="./FreeSurferColorLUT.txt")
    parser.add_argument("--out_dir", default="./explainer_reports")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--generator_ckpt", default="prom3e_generator.pt")
    parser.add_argument("--updrs_ckpt", default="updrs_regressor.pt")
    parser.add_argument("--progression_ckpt", default="progression_regressor.pt")
    parser.add_argument("--delta_t", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_updrs", action="store_true")
    parser.add_argument("--skip_progression", action="store_true")
    args = parser.parse_args()

    atlas = load_atlas_centroids(args.atlas)
    lut = load_freesurfer_lut(args.lut)

    # Use all subject_ids from the CSV, regardless of .pt file presence
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'PPMI_Curated_Data_Cut_Public_20251112.csv'))
    subject_ids = sorted(list(load_valid_subject_ids_from_csv(csv_path)))
    print("First 10 subject_ids from CSV:", subject_ids[:10])
    if args.limit:
        subject_ids = subject_ids[: args.limit]

    updrs_models = None
    prog_models = None
    if not args.skip_updrs:
        updrs_models = build_updrs_models(
            device=args.device,
            encoder_ckpt=args.encoder_ckpt,
            generator_ckpt=args.generator_ckpt,
            regressor_ckpt=args.updrs_ckpt,
        )
    if not args.skip_progression:
        prog_models = build_prog_models(
            device=args.device,
            encoder_ckpt=args.encoder_ckpt,
            generator_ckpt=args.generator_ckpt,
            regressor_ckpt=args.progression_ckpt,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    total = len(subject_ids)
    for i, subject_id in enumerate(subject_ids, start=1):
        try:
            print(f"[{i}/{total}] Explaining {subject_id}")
            import traceback
            try:
                generate_reports_for_subject(
                    subject_id,
                    args.data_root,
                    atlas,
                    lut,
                    args.out_dir,
                    args.topk,
                    updrs_models=updrs_models,
                    prog_models=prog_models,
                    delta_t=args.delta_t,
                    do_updrs=not args.skip_updrs,
                    do_progression=not args.skip_progression,
                    device=args.device,
                )
            except TypeError as te:
                if "'int' object is not iterable" in str(te):
                    print(f"Skipped {subject_id}: [Guarded] {te}")
                    traceback.print_exc()
                    continue
                else:
                    raise
        except Exception as e:
            print(f"Skipped {subject_id}: {e}")


if __name__ == "__main__":
    main()
