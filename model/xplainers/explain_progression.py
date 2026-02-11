import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from model.encoders import SPECTEncoder, MRIEncoder, ConnectomeEncoder
from model.generator.generator import ProM3E_Generator
from model.downstream.downstream_progression import RNNRegressor, load_csv_visits, parse_year, hallucinate_missing_modalities

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]


def load_graph(data_root, modality, subject_id):
    path = os.path.join(data_root, modality, f"{subject_id}.pt")
    if not os.path.exists(path):
        return None
    graph_data = torch.load(path, weights_only=False)
    if isinstance(graph_data, dict):
        data_kwargs = dict(graph_data)
        if "edge_weight" in data_kwargs and "edge_attr" not in data_kwargs:
            data_kwargs["edge_attr"] = data_kwargs.pop("edge_weight")
        graph_data = Data(**data_kwargs)
    if hasattr(graph_data, "edge_attr") and torch.is_tensor(graph_data.edge_attr):
        if graph_data.edge_attr.dim() == 2 and graph_data.edge_attr.size(1) == 1:
            graph_data.edge_attr = graph_data.edge_attr.view(-1)
    if not hasattr(graph_data, "batch") or graph_data.batch is None:
        graph_data = Batch.from_data_list([graph_data])
    return graph_data


def build_models(device="cpu", encoder_ckpt=None, generator_ckpt=None, regressor_ckpt=None):
    encoders = {
        "SPECT": SPECTEncoder().to(device),
        "MRI": MRIEncoder().to(device),
        "fMRI": ConnectomeEncoder().to(device),
        "DTI": ConnectomeEncoder().to(device),
    }

    if encoder_ckpt and os.path.exists(encoder_ckpt):
        ckpt = torch.load(encoder_ckpt, map_location=device)
        for mod, state in ckpt.get("models", {}).items():
            if mod in encoders:
                encoders[mod].load_state_dict(state)
        print(f"Loaded encoders from {encoder_ckpt}")

    generator = ProM3E_Generator(embed_dim=1024).to(device)
    if generator_ckpt and os.path.exists(generator_ckpt):
        ckpt = torch.load(generator_ckpt, map_location=device)
        if "model_state" in ckpt:
            generator.load_state_dict(ckpt["model_state"])
            print(f"Loaded generator from {generator_ckpt}")

    regressor_ckpt = regressor_ckpt or "progression_regressor.pt"
    if not os.path.exists(regressor_ckpt):
        raise RuntimeError(f"Missing regressor checkpoint: {regressor_ckpt}")
    reg_ckpt = torch.load(regressor_ckpt, map_location=device)
    input_dim = reg_ckpt.get("input_dim", 4 * 1024 + 4 + 1)
    state = reg_ckpt.get("model_state", {})
    regressor = RNNRegressor(input_dim).to(device)
    regressor.load_state_dict(state)
    print(f"Loaded regressor from {regressor_ckpt}")

    for m in encoders.values():
        m.eval()
    generator.eval()
    regressor.eval()

    return encoders, generator, regressor


def build_visit_feature(generator, available, delta_prev, device="cpu"):
    recon = hallucinate_missing_modalities(generator, available, device=device)
    feat = []
    mask_feat = []
    for mod in MOD_ORDER:
        if mod in available:
            feat.append(available[mod])
            mask_feat.append(1.0)
        else:
            feat.append(recon[mod])
            mask_feat.append(0.0)
    x = torch.cat(feat, dim=0)
    x = torch.cat([x, torch.tensor(mask_feat, dtype=torch.float32, device=device)], dim=0)
    x = torch.cat([x, torch.tensor([delta_prev], dtype=torch.float32, device=device)], dim=0)
    return x


def explain_transition_with_models(subject_id, data_root, encoders, generator, regressor, target_idx=0,
                                   delta_t=1.0, device="cpu", include_edge_index=True,
                                   csv_path="/home/emanuele/Desktop/Studi/model/data/PPMI_Curated_Data_Cut_Public_20251112.csv"):
    patno = subject_id.split("_")[0]
    visits_by_patno = load_csv_visits(csv_path)
    if patno not in visits_by_patno:
        raise RuntimeError(f"Patient {patno} not found in CSV.")

    visits = sorted(visits_by_patno[patno], key=lambda x: x["year"])
    visit_keys = [v["key"] for v in visits]
    if subject_id not in visit_keys:
        raise RuntimeError(f"Visit {subject_id} not found for patient {patno} in CSV.")

    idx = visit_keys.index(subject_id)
    history_visits = visits[: idx + 1]

    history_feats = []
    prev_year = None
    graphs_by_mod = {m: [] for m in MOD_ORDER}

    for v in history_visits:
        available = {}
        for mod in MOD_ORDER:
            g = load_graph(data_root, mod, v["key"])
            if g is None:
                continue
            g = g.to(device)
            if hasattr(g, "x") and torch.is_tensor(g.x):
                g.x.requires_grad_(True)
            if hasattr(g, "edge_attr") and torch.is_tensor(g.edge_attr):
                g.edge_attr.requires_grad_(True)
            graphs_by_mod[mod].append(g)
            z = encoders[mod](g)
            available[mod] = z.squeeze(0)

        if len(available) == 0:
            prev_year = v["year"]
            continue

        delta_prev = 0.0 if prev_year is None else (v["year"] - prev_year)
        feat = build_visit_feature(generator, available, delta_prev, device=device)
        history_feats.append(feat)
        prev_year = v["year"]

    if len(history_feats) == 0:
        raise RuntimeError("No usable history visits with modalities for this subject_id.")

    seq = torch.stack(history_feats, dim=0).unsqueeze(0)
    lengths = torch.tensor([seq.size(1)], dtype=torch.long, device=device)
    delta_t_next = torch.tensor([[delta_t]], dtype=torch.float32, device=device)

    pred = regressor(seq, lengths, delta_t_next)
    score = pred[0, target_idx]
    score.backward()

    results = {
        "subject_id": subject_id,
        "target": TARGETS[target_idx],
        "prediction": score.detach().cpu().item(),
        "node_importance": {},
        "edge_importance": {},
        "node_value": {},
        "node_grad": {},
        "node_contrib": {},
        "edge_value": {},
        "edge_grad": {},
        "edge_contrib": {},
    }
    if include_edge_index:
        results["edge_index"] = {}

    for mod, graphs in graphs_by_mod.items():
        if not graphs:
            continue

        node_vals = []
        node_grads = []
        node_contribs = []
        edge_vals = []
        edge_grads = []
        edge_contribs = []
        edge_indices = []

        for g in graphs:
            if hasattr(g, "x") and torch.is_tensor(g.x) and g.x.grad is not None:
                node_vals.append(g.x.detach().cpu().mean(dim=1))
                node_grads.append(g.x.grad.detach().cpu().mean(dim=1))
                node_contribs.append((g.x.grad * g.x).detach().cpu().mean(dim=1))
            if hasattr(g, "edge_attr") and torch.is_tensor(g.edge_attr) and g.edge_attr.grad is not None:
                edge_val = g.edge_attr.detach().cpu()
                edge_grad = g.edge_attr.grad.detach().cpu()
                if edge_val.dim() == 2 and edge_val.size(1) == 1:
                    edge_val = edge_val.view(-1)
                if edge_grad.dim() == 2 and edge_grad.size(1) == 1:
                    edge_grad = edge_grad.view(-1)
                edge_vals.append(edge_val)
                edge_grads.append(edge_grad)
                edge_contribs.append(edge_val * edge_grad)
            if include_edge_index and hasattr(g, "edge_index") and torch.is_tensor(g.edge_index):
                edge_indices.append(g.edge_index.detach().cpu())

        if node_vals:
            node_val = torch.stack(node_vals, dim=0).mean(dim=0)
            node_grad = torch.stack(node_grads, dim=0).mean(dim=0)
            node_contrib = torch.stack(node_contribs, dim=0).mean(dim=0)
            results["node_value"][mod] = node_val
            results["node_grad"][mod] = node_grad
            results["node_contrib"][mod] = node_contrib
            node_importance = node_grad.abs().sum(dim=-1)
            if not torch.is_tensor(node_importance):
                node_importance = torch.tensor([node_importance])
            results["node_importance"][mod] = node_importance

        if edge_vals:
            edge_val = torch.stack(edge_vals, dim=0).mean(dim=0)
            edge_grad = torch.stack(edge_grads, dim=0).mean(dim=0)
            edge_contrib = torch.stack(edge_contribs, dim=0).mean(dim=0)
            results["edge_value"][mod] = edge_val
            results["edge_grad"][mod] = edge_grad
            results["edge_contrib"][mod] = edge_contrib
            edge_importance = edge_grad.abs()
            if not torch.is_tensor(edge_importance):
                edge_importance = torch.tensor([edge_importance])
            results["edge_importance"][mod] = edge_importance
        if include_edge_index and edge_indices:
            results["edge_index"][mod] = edge_indices[-1]

    return results


def explain_transition(subject_id, data_root, target_idx=0, delta_t=1.0, device="cpu",
                       encoder_ckpt=None, generator_ckpt="prom3e_generator.pt",
                       regressor_ckpt="progression_regressor.pt",
                       csv_path="/home/emanuele/Desktop/Studi/model/data/PPMI_Curated_Data_Cut_Public_20251112.csv"):
    encoders, generator, regressor = build_models(
        device=device,
        encoder_ckpt=encoder_ckpt,
        generator_ckpt=generator_ckpt,
        regressor_ckpt=regressor_ckpt,
    )

    return explain_transition_with_models(
        subject_id,
        data_root,
        encoders,
        generator,
        regressor,
        target_idx=target_idx,
        delta_t=delta_t,
        device=device,
        include_edge_index=True,
        csv_path=csv_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject visit id like 100001_BL")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target", default="updrs3_score", choices=TARGETS)
    parser.add_argument("--delta_t", type=float, default=1.0, help="Years to next visit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--regressor_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv_path", required=True)
    args = parser.parse_args()

    target_idx = TARGETS.index(args.target)
    results = explain_transition(
        args.subject,
        args.data_root,
        target_idx=target_idx,
        delta_t=args.delta_t,
        device=args.device,
        encoder_ckpt=args.encoder_ckpt,
        generator_ckpt=args.generator_ckpt,
        regressor_ckpt=args.regressor_ckpt,
        csv_path=args.csv_path,
    )

    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    out_path = os.path.join(checkpoints_dir, os.path.basename(args.out))
    torch.save(results, out_path)
    print(f"Saved saliency to {out_path}")
    print(f"Prediction for {results['target']}: {results['prediction']:.3f}")
    for mod in MOD_ORDER:
        if mod in results["node_importance"]:
            imp = results["node_importance"][mod]
            topk = torch.topk(imp, min(5, imp.numel())).indices.tolist()
            print(f"Top nodes {mod}: {topk}")
        else:
            print(f"Top nodes {mod}: <none>")
    for mod in MOD_ORDER:
        if mod in results["edge_importance"]:
            imp = results["edge_importance"][mod]
            topk = torch.topk(imp, min(5, imp.numel())).indices.tolist()
            print(f"Top edges {mod}: {topk}")
        else:
            print(f"Top edges {mod}: <none>")


if __name__ == "__main__":
    main()
