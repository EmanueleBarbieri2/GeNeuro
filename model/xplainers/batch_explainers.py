#!/usr/bin/env python3
import os
import json
import csv
import argparse
import sys
import torch
from typing import Dict, List

sys.path.append(os.getcwd())

# --- Task Engines ---
from model.xplainers.explain_updrs import MOD_ORDER, TARGETS as UPDRS_TARGETS, build_models as build_updrs_models, explain_subject_with_models
from model.xplainers.explain_progression import TARGETS as PROG_TARGETS, build_models as build_prog_models, explain_transition_with_models
from model.xplainers.explain_classification import build_models as build_cls_models, explain_classification_subject_with_models

from model.downstream.downstream_classification import load_csv_labels as load_class_labels
from model.downstream.downstream_updrs import Regressor as StaticRegressor
from model.downstream.downstream_progression import ForecastingGRU

SPECT_NODE_NAMES = ["striatum_bilat", "striatum_L", "striatum_R", "caudate_L", "putamen_L", "caudate_R", "putamen_R"]

# ==========================================
# 1. LOCAL HELPERS
# ==========================================
def load_valid_subject_ids_from_csv(csv_path: str) -> set:
    valid_ids = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            if patno and event_id: valid_ids.add(f"{patno}_{event_id}")
    return valid_ids

def load_atlas_centroids(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f: return list(csv.DictReader(f))

def load_freesurfer_lut(path: str) -> Dict[int, Dict[str, object]]:
    lut = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 6: continue
            try: label = int(parts[0])
            except: continue
            rgba = []
            for token in reversed(parts[1:]):
                try: rgba.append(int(token))
                except: break
                if len(rgba) == 4: break
            if len(rgba) == 4:
                name = " ".join(parts[1:len(parts) - 4])
                lut[label] = {"name": name, "color": list(reversed(rgba))}
    return lut

def build_roi_info(index: int, atlas: List[Dict], lut: Dict, modality: str):
    if modality == "SPECT":
        if index < 0 or index >= len(SPECT_NODE_NAMES): return None
        return {"node_index": index, "roi_name": SPECT_NODE_NAMES[index]}
    if index < 0 or index >= len(atlas): return None
    row = atlas[index]
    label = int(float(row.get("label", -1))) if row.get("label") else -1
    return {
        "node_index": index,
        "label": label,
        "roi_name": row.get("roi_name", "unknown"),
        "lut_name": lut[label]["name"] if label in lut else None
    }

def extract_topk_nodes(node_imp, node_val, node_grad, node_contrib, atlas, lut, mod, topk):
    if node_imp is None or node_imp.numel() == 0: return []
    
    results = []
    
    # SCENARIO A: 1D Tensor (e.g., SPECT which only has 1 feature, or previously summed data)
    if node_imp.dim() == 1 or (node_imp.dim() == 2 and node_imp.shape[1] == 1):
        node_imp = node_imp.view(-1)
        k = min(topk, node_imp.numel())
        indices = torch.topk(node_imp, k).indices.tolist()
        if isinstance(indices, int): indices = [indices]
        for idx in indices:
            info = build_roi_info(idx, atlas, lut, mod)
            if not info: continue
            info.update({
                "importance": float(node_imp[idx].item()),
                "value_mean": float(node_val[idx].item() if node_val.dim()==1 else node_val[idx,0].item()),
                "grad_mean": float(node_grad[idx].item() if node_grad.dim()==1 else node_grad[idx,0].item()),
                "contrib_mean": float(node_contrib[idx].item() if node_contrib.dim()==1 else node_contrib[idx,0].item())
            })
            results.append(info)
            
    # SCENARIO B: 2D Tensor (e.g., MRI with Area, Thickness, Volume)
    else:
        N, F = node_imp.shape
        flat_imp = node_imp.view(-1)
        k = min(topk, flat_imp.numel())
        flat_indices = torch.topk(flat_imp, k).indices.tolist()
        if isinstance(flat_indices, int): flat_indices = [flat_indices]

        # ---> EXACT FEATURE MAPPING APPLIED HERE <---
        feature_names = ["Surface_Area", "Thickness", "Volume"]

        for flat_idx in flat_indices:
            node_idx = flat_idx // F
            feat_idx = flat_idx % F
            
            info = build_roi_info(node_idx, atlas, lut, mod)
            if not info: continue
            
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feat_{feat_idx}"
            
            # Append the specific feature name to the brain region name
            info["roi_name"] = f"{info['roi_name']}_{feat_name}"
            
            info.update({
                "importance": float(node_imp[node_idx, feat_idx].item()),
                "value_mean": float(node_val[node_idx, feat_idx].item()),
                "grad_mean": float(node_grad[node_idx, feat_idx].item()),
                "contrib_mean": float(node_contrib[node_idx, feat_idx].item())
            })
            results.append(info)

    return results

def extract_topk_edges(edge_imp, edge_val, edge_grad, edge_contrib, edge_index, atlas, lut, mod, topk):
    if edge_imp is None or edge_imp.numel() == 0: return []
    k = min(topk, edge_imp.numel())
    indices = torch.topk(edge_imp, k).indices.tolist()
    if isinstance(indices, int): indices = [indices]
    results = []
    for ei in indices:
        u, v = int(edge_index[0, ei].item()), int(edge_index[1, ei].item())
        info_u = build_roi_info(u, atlas, lut, mod)
        info_v = build_roi_info(v, atlas, lut, mod)
        if not info_u or not info_v: continue
        results.append({
            "edge_index": ei, "u": info_u, "v": info_v,
            "importance": float(edge_imp[ei].item() if edge_imp.dim() > 0 else edge_imp.item()),
            "value": float(edge_val[ei].item() if edge_val.dim() > 0 else edge_val.item()),
            "grad": float(edge_grad[ei].item() if edge_grad.dim() > 0 else edge_grad.item()),
            "contrib": float(edge_contrib[ei].item() if edge_contrib.dim() > 0 else edge_contrib.item())
        })
    return results

# ==========================================
# 2. MASTER ORCHESTRATOR
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--csv_path", default="./data/PPMI_Curated_Data_Cut_Public_20251112.csv")
    parser.add_argument("--atlas", default="/home/emanuele/Desktop/Studi/model/model/atlas_centroids.csv")
    parser.add_argument("--lut", default="/home/emanuele/Desktop/Studi/model/model/FreeSurferColorLUT.txt")
    parser.add_argument("--out_dir", default="/home/emanuele/Desktop/Studi/model/model/xplainers/explainer_reports")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--encoder_ckpt", default="model/checkpoints_master/encoders.pt")
    parser.add_argument("--generator_ckpt", default="model/checkpoints_master/prom3e_generator.pt")
    parser.add_argument("--classifier_ckpt", default="model/checkpoints_master/classifier.pt")
    
    parser.add_argument("--updrs_u2_ckpt", default="model/checkpoints_master/static_U2_ADL.pt")
    parser.add_argument("--updrs_u3_ckpt", default="model/checkpoints_master/static_U3_Motor.pt")
    parser.add_argument("--prog_u2_ckpt", default="model/checkpoints_master/prog_U2_ADL.pt")
    parser.add_argument("--prog_u3_ckpt", default="model/checkpoints_master/prog_U3_Motor.pt")
    
    parser.add_argument("--delta_t", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_classification", action="store_true")
    parser.add_argument("--skip_updrs", action="store_true")
    parser.add_argument("--skip_progression", action="store_true")
    args = parser.parse_args()

    atlas = load_atlas_centroids(args.atlas)
    lut = load_freesurfer_lut(args.lut)
    subject_ids = sorted(list(load_valid_subject_ids_from_csv(args.csv_path)))
    if args.limit: subject_ids = subject_ids[:args.limit]

    print(f"🚀 Initializing ProM3E Explainability Pipeline on {args.device} for {len(subject_ids)} subjects...")

    cls_tools = None
    updrs_encoders, updrs_gen, updrs_heads = None, None, {}
    prog_encoders, prog_gen, prog_heads = None, None, {}
    
    # UPGRADED CLASSIFICATION LOADER
    if not args.skip_classification:
        print("🧠 Loading Full-Graph Classification Engine...")
        labels = load_class_labels(args.csv_path)
        cls_encoders, cls_gen, clf = build_cls_models(args.device, args.encoder_ckpt, args.generator_ckpt, args.classifier_ckpt)
        cls_tools = (labels, cls_encoders, cls_gen, clf)

    if not args.skip_updrs:
        print("🧠 Loading Static UPDRS Engines (U2 & U3)...")
        updrs_encoders, updrs_gen, _ = build_updrs_models(args.device, args.encoder_ckpt, args.generator_ckpt, args.updrs_u3_ckpt)
        for name, ckpt_path in [("U2_ADL", args.updrs_u2_ckpt), ("U3_Motor", args.updrs_u3_ckpt)]:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=args.device)
                state = ckpt.get("model_state", ckpt)
                if "net.0.weight" in state: in_dim = state["net.0.weight"].shape[1]
                elif "0.weight" in state: in_dim = state["0.weight"].shape[1]
                else: in_dim = ckpt.get("input_dim", 4096)
                r = StaticRegressor(in_dim).to(args.device)
                r.load_state_dict(state)
                r.input_dim = in_dim
                r.eval()
                updrs_heads[name] = r

    if not args.skip_progression:
        print("🧠 Loading Progression GRU Engines (U2 & U3)...")
        prog_encoders, prog_gen, _ = build_prog_models(args.device, args.encoder_ckpt, args.generator_ckpt, args.prog_u3_ckpt)
        for name, ckpt_path in [("U2_ADL", args.prog_u2_ckpt), ("U3_Motor", args.prog_u3_ckpt)]:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=args.device)
                state = ckpt.get("model_state", ckpt)
                if "compressor.0.weight" in state:
                    h_dim = state["compressor.0.weight"].shape[0]
                    in_dim = state["compressor.0.weight"].shape[1]
                else:
                    h_dim = 128
                    in_dim = ckpt.get("input_dim", 4097)
                r = ForecastingGRU(input_dim=in_dim, hidden_dim=h_dim).to(args.device)
                r.load_state_dict(state)
                r.input_dim = in_dim
                r.eval()
                prog_heads[name] = r

    # --- Processing Loop ---
    for i, subj in enumerate(subject_ids, 1):
        print(f"[{i}/{len(subject_ids)}] Explaining {subj}...")
        
        # 1. UPGRADED Classification Flow
        if cls_tools:
            #try:
                labels, cls_encoders, cls_gen, clf = cls_tools
                for m in list(cls_encoders.values()) + [cls_gen, clf]: m.zero_grad(set_to_none=True)
                
                res = explain_classification_subject_with_models(subj, args.data_root, labels.get(subj), cls_encoders, cls_gen, clf, args.device)
                if res:
                    report = {"subject_id": subj, "task": "classification", "true_label": res["true_label"], "pred_label": res["pred_label"], "pred_prob": res["pred_prob"], "modalities": {}}
                    for mod in MOD_ORDER:
                        if mod in res.get("node_importance", {}):
                            report["modalities"][mod] = {
                                "top_nodes": extract_topk_nodes(res["node_importance"][mod], res["node_value"][mod], res["node_grad"][mod], res["node_contrib"][mod], atlas, lut, mod, args.topk),
                                "top_edges": extract_topk_edges(res.get("edge_importance",{}).get(mod), res.get("edge_value",{}).get(mod), res.get("edge_grad",{}).get(mod), res.get("edge_contrib",{}).get(mod), res.get("edge_index",{}).get(mod), atlas, lut, mod, args.topk)
                            }
                    out_path = os.path.join(args.out_dir, "classification", f"{subj}.json")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f: json.dump(report, f, indent=2)
            #except Exception: pass

        # 2. Static UPDRS
        if updrs_heads:
            for target_name, head in updrs_heads.items():
                #try:
                    target_idx = UPDRS_TARGETS.index("updrs2_score" if "U2" in target_name else "updrs3_score")
                    for m in list(updrs_encoders.values()) + [updrs_gen, head]: m.zero_grad(set_to_none=True)
                    
                    res = explain_subject_with_models(subj, args.data_root, updrs_encoders, updrs_gen, head, target_idx=target_idx, device=args.device)
                    
                    report = {"subject_id": subj, "task": f"updrs_static_{target_name}", "prediction": res["prediction"], "modalities": {}}
                    for mod in MOD_ORDER:
                        if mod in res.get("node_importance", {}):
                            report["modalities"][mod] = {
                                "top_nodes": extract_topk_nodes(res["node_importance"][mod], res["node_value"][mod], res["node_grad"][mod], res["node_contrib"][mod], atlas, lut, mod, args.topk),
                                "top_edges": extract_topk_edges(res.get("edge_importance",{}).get(mod), res.get("edge_value",{}).get(mod), res.get("edge_grad",{}).get(mod), res.get("edge_contrib",{}).get(mod), res.get("edge_index",{}).get(mod), atlas, lut, mod, args.topk)
                            }
                    out_path = os.path.join(args.out_dir, "updrs", f"{subj}_{target_name}.json")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f: json.dump(report, f, indent=2)
                #except Exception: pass 

        # 3. Progression
        if prog_heads:
            for target_name, head in prog_heads.items():
                #try:
                    target_idx = PROG_TARGETS.index("updrs2_score" if "U2" in target_name else "updrs3_score")
                    for m in list(prog_encoders.values()) + [prog_gen, head]: m.zero_grad(set_to_none=True)
                    
                    res = explain_transition_with_models(subj, args.data_root, prog_encoders, prog_gen, head, target_idx=target_idx, delta_t=args.delta_t, device=args.device)
                    
                    report = {"subject_id": subj, "task": f"progression_forecast_{target_name}", "prediction": res["prediction"], "modalities": {}}
                    for mod in MOD_ORDER:
                        if mod in res.get("node_importance", {}):
                            report["modalities"][mod] = {
                                "top_nodes": extract_topk_nodes(res["node_importance"][mod], res["node_value"][mod], res["node_grad"][mod], res["node_contrib"][mod], atlas, lut, mod, args.topk),
                                "top_edges": extract_topk_edges(res.get("edge_importance",{}).get(mod), res.get("edge_value",{}).get(mod), res.get("edge_grad",{}).get(mod), res.get("edge_contrib",{}).get(mod), res.get("edge_index",{}).get(mod), atlas, lut, mod, args.topk)
                            }
                    out_path = os.path.join(args.out_dir, "progression", f"{subj}_{target_name}.json")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f: json.dump(report, f, indent=2)
                #except Exception: pass 

    print("\n✅ All Explanations Generated and Saved to JSON!")

if __name__ == "__main__":
    main()