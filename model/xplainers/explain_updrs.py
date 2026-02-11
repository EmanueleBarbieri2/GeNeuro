import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from model.encoders import SPECTEncoder, MRIEncoder, ConnectomeEncoder
from model.downstream.downstream_updrs import Regressor
from model.generator.generator import ProM3E_Generator

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
    # Normalize edge_attr shape
    if hasattr(graph_data, "edge_attr") and torch.is_tensor(graph_data.edge_attr):
        if graph_data.edge_attr.dim() == 2 and graph_data.edge_attr.size(1) == 1:
            graph_data.edge_attr = graph_data.edge_attr.view(-1)
    # Ensure batch attributes exist for single-graph inference
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

    regressor_ckpt = regressor_ckpt or "updrs_regressor.pt"
    if not os.path.exists(regressor_ckpt):
        raise RuntimeError(f"Missing regressor checkpoint: {regressor_ckpt}")
    reg_ckpt = torch.load(regressor_ckpt, map_location=device)
    input_dim = reg_ckpt.get("input_dim", 4 * 1024 + 4)
    state = reg_ckpt.get("model_state", {})
    # Decide which regressor architecture to build based on state_dict keys
    if any(k.startswith("net.") for k in state.keys()):
        regressor = Regressor(input_dim).to(device)
        regressor.load_state_dict(state)
    else:
        regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4),
        ).to(device)
        regressor.load_state_dict(state)
    print(f"Loaded regressor from {regressor_ckpt}")

    for m in encoders.values():
        m.eval()
    generator.eval()
    regressor.eval()

    return encoders, generator, regressor


def explain_subject_with_models(subject_id, data_root, encoders, generator, regressor, target_idx=0,
                                device="cpu", include_edge_index=True):
    graphs = {}
    available = {}
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    z_list = []

    for i, mod in enumerate(MOD_ORDER):
        g = load_graph(data_root, mod, subject_id)
        if g is None:
            continue
        g = g.to(device)
        if hasattr(g, "x") and torch.is_tensor(g.x):
            g.x.requires_grad_(True)
        if hasattr(g, "edge_attr") and torch.is_tensor(g.edge_attr):
            g.edge_attr.requires_grad_(True)
        graphs[mod] = g

        z = encoders[mod](g)
        available[mod] = z
        mask[0, i] = False

    # Build input tensor preserving gradients for available modalities
    if len(available) == 0:
        raise RuntimeError("No modalities found for this subject_id.")
    ref_dtype = next(iter(available.values())).dtype
    for mod in MOD_ORDER:
        if mod in available:
            z_list.append(available[mod].squeeze(0))
        else:
            z_list.append(torch.zeros(1024, device=device, dtype=ref_dtype))
    input_tensor = torch.stack(z_list, dim=0).unsqueeze(0)

    # Generator fills missing modalities
    z_recon, _, _ = generator(input_tensor, mask)
    recon = z_recon[0]

    # Build final feature vector: real for present, recon for missing + mask
    feat = []
    mask_feat = []
    for i, mod in enumerate(MOD_ORDER):
        if mod in available:
            feat.append(available[mod].squeeze(0))
            mask_feat.append(1.0)
        else:
            feat.append(recon[i])
            mask_feat.append(0.0)
    x = torch.cat(feat, dim=0)
    x = torch.cat([x, torch.tensor(mask_feat, dtype=torch.float32, device=device)], dim=0)

    x = x.unsqueeze(0)
    pred = regressor(x)
    score = pred[0, target_idx]

    # Backprop to graph inputs
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

    for mod, g in graphs.items():
        if hasattr(g, "x") and torch.is_tensor(g.x) and g.x.grad is not None:
            node_imp = g.x.grad.detach().abs().sum(dim=1).cpu()
            results["node_importance"][mod] = node_imp
            node_val = g.x.detach().cpu().mean(dim=1)
            node_grad = g.x.grad.detach().cpu().mean(dim=1)
            node_contrib = (g.x.grad * g.x).detach().cpu().mean(dim=1)
            results["node_value"][mod] = node_val
            results["node_grad"][mod] = node_grad
            results["node_contrib"][mod] = node_contrib
        if hasattr(g, "edge_attr") and torch.is_tensor(g.edge_attr) and g.edge_attr.grad is not None:
            edge_grad = g.edge_attr.grad.detach().cpu()
            edge_val = g.edge_attr.detach().cpu()
            if edge_val.dim() == 2 and edge_val.size(1) == 1:
                edge_val = edge_val.view(-1)
            if edge_grad.dim() == 2 and edge_grad.size(1) == 1:
                edge_grad = edge_grad.view(-1)
            edge_imp = edge_grad.abs()
            results["edge_importance"][mod] = edge_imp
            results["edge_value"][mod] = edge_val
            results["edge_grad"][mod] = edge_grad
            results["edge_contrib"][mod] = edge_val * edge_grad
        if include_edge_index and hasattr(g, "edge_index") and torch.is_tensor(g.edge_index):
            results["edge_index"][mod] = g.edge_index.detach().cpu()

    return results


def explain_subject(subject_id, data_root, target_idx=0, device="cpu", encoder_ckpt=None,
                    generator_ckpt="prom3e_generator.pt", regressor_ckpt="updrs_regressor.pt"):
    encoders, generator, regressor = build_models(
        device=device,
        encoder_ckpt=encoder_ckpt,
        generator_ckpt=generator_ckpt,
        regressor_ckpt=regressor_ckpt,
    )

    return explain_subject_with_models(
        subject_id,
        data_root,
        encoders,
        generator,
        regressor,
        target_idx=target_idx,
        device=device,
        include_edge_index=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject visit id like 100001_BL")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--target", default="updrs3_score", choices=TARGETS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--generator_ckpt", default="prom3e_generator.pt")
    parser.add_argument("--regressor_ckpt", default="updrs_regressor.pt")
    parser.add_argument("--out", default="updrs_saliency.pt")
    args = parser.parse_args()

    target_idx = TARGETS.index(args.target)
    results = explain_subject(
        args.subject,
        args.data_root,
        target_idx=target_idx,
        device=args.device,
        encoder_ckpt=args.encoder_ckpt,
        generator_ckpt=args.generator_ckpt,
        regressor_ckpt=args.regressor_ckpt,
    )

    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    out_path = os.path.join(checkpoints_dir, os.path.basename(args.out))
    torch.save(results, out_path)
    print(f"Saved saliency to {out_path}")
    print(f"Prediction for {results['target']}: {results['prediction']:.3f}")


if __name__ == "__main__":
    main()
