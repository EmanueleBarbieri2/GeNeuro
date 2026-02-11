import argparse
import csv

import torch

from model.xplainers.explain_updrs import build_models as build_updrs_models, explain_subject_with_models, MOD_ORDER, TARGETS
from model.xplainers.explain_progression import build_models as build_prog_models, explain_transition_with_models

SPECT_NODE_NAMES = [
    "striatum_bilat",
    "striatum_L",
    "striatum_R",
    "caudate_L",
    "putamen_L",
    "caudate_R",
    "putamen_R",
]


def load_atlas(path: str):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_lut(path: str):
    lut = {}

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            label = to_int(parts[0])
            if label is None:
                continue
            rgba = []
            for token in reversed(parts[1:]):
                val = to_int(token)
                if val is None:
                    break
                rgba.append(val)
                if len(rgba) == 4:
                    break
            if len(rgba) != 4:
                continue
            name_tokens = parts[1 : len(parts) - 4]
            if not name_tokens:
                continue
            name = " ".join(name_tokens)
            lut[label] = {"name": name, "color": rgba}
    return lut


def roi_info(index, atlas, lut, mod):
    if mod == "SPECT":
        name = SPECT_NODE_NAMES[index] if 0 <= index < len(SPECT_NODE_NAMES) else f"node_{index}"
        return None, name
    row = atlas[index]
    label = int(float(row["label"]))
    name = row["roi_name"]
    lut_name = lut.get(label, {}).get("name")
    return label, (lut_name or name)


def topk_nodes(node_imp, node_val, node_grad, node_contrib, atlas, lut, mod, k=5):
    k = min(k, node_imp.numel())
    idxs = torch.topk(node_imp, k).indices.tolist()
    out = []
    for idx in idxs:
        label, name = roi_info(idx, atlas, lut, mod)
        out.append(
            (
                name,
                idx,
                label,
                float(node_imp[idx].item()),
                float(node_val[idx].item()),
                float(node_grad[idx].item()),
                float(node_contrib[idx].item()),
            )
        )
    return out


def topk_edges(edge_imp, edge_val, edge_grad, edge_contrib, edge_index, atlas, lut, mod, k=5):
    k = min(k, edge_imp.numel())
    idxs = torch.topk(edge_imp, k).indices.tolist()
    out = []
    for ei in idxs:
        u = int(edge_index[0, ei].item())
        v = int(edge_index[1, ei].item())
        label_u, name_u = roi_info(u, atlas, lut, mod)
        label_v, name_v = roi_info(v, atlas, lut, mod)
        out.append(
            (
                name_u,
                u,
                label_u,
                name_v,
                v,
                label_v,
                float(edge_imp[ei].item()),
                float(edge_val[ei].item()),
                float(edge_grad[ei].item()),
                float(edge_contrib[ei].item()),
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject visit id like 100005_BL")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--target", default="updrs3_score", choices=TARGETS)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--generator_ckpt", default="prom3e_generator.pt")
    parser.add_argument("--regressor_ckpt", default="updrs_regressor.pt")
    parser.add_argument("--task", choices=["updrs", "progression"], default="updrs")
    parser.add_argument("--delta_t", type=float, default=1.0, help="Years to next visit for progression")
    parser.add_argument("--atlas", default="./atlas_centroids.csv")
    parser.add_argument("--lut", default="./FreeSurferColorLUT.txt")
    args = parser.parse_args()

    atlas = load_atlas(args.atlas)
    lut = load_lut(args.lut)

    target_idx = TARGETS.index(args.target)
    if args.task == "updrs":
        models = build_updrs_models(
            device=args.device,
            encoder_ckpt=args.encoder_ckpt,
            generator_ckpt=args.generator_ckpt,
            regressor_ckpt=args.regressor_ckpt,
        )
        for m in list(models[0].values()) + list(models[1:]):
            m.zero_grad(set_to_none=True)
        res = explain_subject_with_models(
            args.subject,
            args.data_root,
            *models,
            target_idx=target_idx,
            device=args.device,
            include_edge_index=True,
        )
        print(f"UPDRS | {args.subject} {args.target} prediction {res['prediction']:.4f}")
    else:
        models = build_prog_models(
            device=args.device,
            encoder_ckpt=args.encoder_ckpt,
            generator_ckpt=args.generator_ckpt,
            regressor_ckpt=args.regressor_ckpt,
        )
        for m in list(models[0].values()) + list(models[1:]):
            m.zero_grad(set_to_none=True)
        res = explain_transition_with_models(
            args.subject,
            args.data_root,
            *models,
            target_idx=target_idx,
            delta_t=args.delta_t,
            device=args.device,
            include_edge_index=True,
        )
        print(f"PROGRESSION | {args.subject} {args.target} prediction {res['prediction']:.4f} | delta_t={args.delta_t}")

    for mod in MOD_ORDER:
        if mod in res.get("node_importance", {}):
            print(f"\n {mod} top nodes (importance, value_mean, grad_mean, contrib_mean):")
            nodes = topk_nodes(
                res["node_importance"][mod],
                res["node_value"][mod],
                res["node_grad"][mod],
                res["node_contrib"][mod],
                atlas,
                lut,
                mod,
                args.topk,
            )
            for name, idx, label, imp, val, grad, contrib in nodes:
                label_txt = f", label {label}" if label is not None else ""
                print(
                    f"  - {name} (idx {idx}{label_txt}) imp={imp:.6f} val={val:.6f} grad={grad:.6f} contrib={contrib:.6f}"
                )
        if mod in res.get("edge_importance", {}):
            print(f"{mod} top edges (importance, value, grad, contrib):")
            edges = topk_edges(
                res["edge_importance"][mod],
                res["edge_value"][mod],
                res["edge_grad"][mod],
                res["edge_contrib"][mod],
                res["edge_index"][mod],
                atlas,
                lut,
                mod,
                args.topk,
            )
            for name_u, u, label_u, name_v, v, label_v, imp, val, grad, contrib in edges:
                print(
                    f"  - {name_u} (idx {u}) ↔ {name_v} (idx {v}) imp={imp:.6f} val={val:.6f} grad={grad:.6f} contrib={contrib:.6f}"
                )


if __name__ == "__main__":
    main()
