import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from model.encoders import SPECTEncoder, MRIEncoder, DTIEncoder, fMRIEncoder
from model.generator.generator import ProM3E_Generator
from model.generator.run_generator_demo import hallucinate_missing_modalities
from model.downstream.downstream_progression import ForecastingGRU, load_csv_visits, parse_year 

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
    if not encoder_ckpt or not os.path.exists(encoder_ckpt):
        raise RuntimeError("Encoder checkpoint missing!")
        
    ckpt = torch.load(encoder_ckpt, map_location=device)
    spect_state = ckpt.get("models", {}).get("SPECT", {})
    
    h_dim = spect_state.get("conv1.bias", torch.zeros(64)).shape[0]
    if "projection.3.weight" in spect_state: e_dim = spect_state["projection.3.weight"].shape[0]
    else: e_dim = 1024
    
    encoders = {
        "SPECT": SPECTEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "MRI": MRIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "fMRI": fMRIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "DTI": DTIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
    }
    
    for mod, state in ckpt.get("models", {}).items():
        if mod in encoders: encoders[mod].load_state_dict(state)

    if generator_ckpt and os.path.exists(generator_ckpt):
        gen_ckpt = torch.load(generator_ckpt, map_location=device)
        gen_state = gen_ckpt.get("model_state", {})
        layer_indices = [int(k.split(".")[2]) for k in gen_state.keys() if k.startswith("transformer.layers.")]
        num_layers = max(layer_indices) + 1 if layer_indices else 6
        generator = ProM3E_Generator(
            embed_dim=e_dim,
            hidden_dim=1024,
            num_heads=8,
            num_layers=5,
            num_registers=0,
            mlp_depth=3
        ).to(device)
        generator.load_state_dict(gen_state)
    else:
        generator = ProM3E_Generator(
            embed_dim=e_dim,
            hidden_dim=1024,
            num_heads=8,
            num_layers=5,
            num_registers=0,
            mlp_depth=3
        ).to(device)

    # Load GRU Regressor Dynamically
    if not regressor_ckpt or not os.path.exists(regressor_ckpt):
        raise RuntimeError(f"Missing regressor checkpoint: {regressor_ckpt}")
        
    reg_ckpt = torch.load(regressor_ckpt, map_location=device)
    state = reg_ckpt.get("model_state", reg_ckpt)
    
    if "compressor.0.weight" in state:
        gru_h_dim = state["compressor.0.weight"].shape[0]
        input_dim = state["compressor.0.weight"].shape[1] 
    else:
        gru_h_dim = 128
        input_dim = reg_ckpt.get("input_dim", 4097)
    
    regressor = ForecastingGRU(input_dim=input_dim, hidden_dim=gru_h_dim).to(device)
    regressor.load_state_dict(state)
    regressor.input_dim = input_dim
    
    for m in encoders.values(): m.eval()
    generator.eval(); regressor.eval()
    return encoders, generator, regressor

def build_visit_feature(generator, available, delta_prev, expected_dim, device="cpu"):
    # 1. Prepare inputs for the generator, preserving gradients!
    z_list = []
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    for i, mod in enumerate(MOD_ORDER):
        if mod in available:
            z_list.append(available[mod])
            mask[0, i] = False
        else:
            z_list.append(torch.zeros(generator.embed_dim if hasattr(generator, 'embed_dim') else 1024, device=device))
            
    input_tensor = torch.stack(z_list, dim=0).unsqueeze(0)
    z_recon, _, _ = generator(input_tensor, mask)
    recon = z_recon[0]
    
    # 2. Build the feature vector
    feat, mask_feat = [], []
    for i, mod in enumerate(MOD_ORDER):
        if mod in available:
            feat.append(available[mod])
            mask_feat.append(1.0)
        else:
            feat.append(recon[i])
            mask_feat.append(0.0)
            
    x_base = torch.cat(feat, dim=0)
    
    # 3. Dynamic Mask and Time Attachment
    if expected_dim == x_base.shape[0] + 5:
        x = torch.cat([x_base, torch.tensor(mask_feat, dtype=torch.float32, device=device), torch.tensor([delta_prev], dtype=torch.float32, device=device)], dim=0)
    elif expected_dim == x_base.shape[0] + 4:
        x = torch.cat([x_base, torch.tensor(mask_feat, dtype=torch.float32, device=device)], dim=0)
    elif expected_dim == x_base.shape[0] + 1:
        x = torch.cat([x_base, torch.tensor([delta_prev], dtype=torch.float32, device=device)], dim=0)
    else:
        x = x_base
        
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
    dt_list = []  
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
            
            for module in encoders[mod].modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    if getattr(module, 'running_mean', None) is None:
                        module.running_mean = torch.zeros(module.num_features, device=device)
                    if getattr(module, 'running_var', None) is None:
                        module.running_var = torch.ones(module.num_features, device=device)
                    module.track_running_stats = True
            
            z = encoders[mod](g)
            available[mod] = z.squeeze(0)

        if len(available) == 0:
            prev_year = v["year"]
            continue

        delta_prev = 0.0 if prev_year is None else (v["year"] - prev_year)
        dt_list.append(delta_prev)  
        
        feat = build_visit_feature(generator, available, delta_prev, regressor.input_dim, device=device)
        history_feats.append(feat)
        prev_year = v["year"]

    if len(history_feats) == 0:
        raise RuntimeError("No usable history visits with modalities for this subject_id.")

    seq = torch.stack(history_feats, dim=0).unsqueeze(0)
    lengths = torch.tensor([seq.size(1)], dtype=torch.long, device=device)
    
    dt_seq = torch.tensor([dt_list], dtype=torch.float32, device=device)
    delta_t_next = torch.tensor([[delta_t]], dtype=torch.float32, device=device)

    pred = regressor(seq, dt_seq, lengths, delta_t_next) 
    
    score = pred[0, target_idx] if pred.shape[1] > 1 else pred[0, 0]
    score.backward()

    results = {
        "subject_id": subject_id,
        "target": TARGETS[target_idx],
        "prediction": score.detach().cpu().item(),
        "node_importance": {}, "edge_importance": {}, "node_value": {}, "node_grad": {},
        "node_contrib": {}, "edge_value": {}, "edge_grad": {}, "edge_contrib": {},
    }
    if include_edge_index:
        results["edge_index"] = {}

    for mod, graphs in graphs_by_mod.items():
        if not graphs: continue

        node_vals, node_grads, node_contribs = [], [], []
        edge_vals, edge_grads, edge_contribs, edge_indices = [], [], [], []

        for g in graphs:
            if hasattr(g, "x") and torch.is_tensor(g.x) and g.x.grad is not None:
                # ---> CHANGED: Removed .sum(dim=1) to keep the 2D [Nodes, Features] shape <---
                node_vals.append(g.x.detach().cpu())
                node_grads.append(g.x.grad.detach().cpu())
                node_contribs.append((g.x.grad * g.x).detach().cpu())
            if hasattr(g, "edge_attr") and torch.is_tensor(g.edge_attr) and g.edge_attr.grad is not None:
                edge_val = g.edge_attr.detach().cpu()
                edge_grad = g.edge_attr.grad.detach().cpu()
                if edge_val.dim() == 2 and edge_val.size(1) == 1: edge_val = edge_val.view(-1)
                if edge_grad.dim() == 2 and edge_grad.size(1) == 1: edge_grad = edge_grad.view(-1)
                edge_vals.append(edge_val)
                edge_grads.append(edge_grad)
                edge_contribs.append(edge_val * edge_grad)
            if include_edge_index and hasattr(g, "edge_index") and torch.is_tensor(g.edge_index):
                edge_indices.append(g.edge_index.detach().cpu())

        if node_vals:
            # Longitudinal averaging over time
            results["node_value"][mod] = torch.stack(node_vals, dim=0).mean(dim=0)
            results["node_grad"][mod] = torch.stack(node_grads, dim=0).mean(dim=0)
            results["node_contrib"][mod] = torch.stack(node_contribs, dim=0).mean(dim=0)
            
            node_importance = results["node_grad"][mod].abs()
            if not torch.is_tensor(node_importance):
                node_importance = torch.tensor([node_importance])
            results["node_importance"][mod] = node_importance

        if edge_vals:
            same_shape = all(e.shape == edge_vals[0].shape for e in edge_vals)
            
            if same_shape:
                results["edge_value"][mod] = torch.stack(edge_vals, dim=0).mean(dim=0)
                results["edge_grad"][mod] = torch.stack(edge_grads, dim=0).mean(dim=0)
                results["edge_contrib"][mod] = torch.stack(edge_contribs, dim=0).mean(dim=0)
            else:
                results["edge_value"][mod] = edge_vals[-1]
                results["edge_grad"][mod] = edge_grads[-1]
                results["edge_contrib"][mod] = edge_contribs[-1]
            
            edge_importance = results["edge_grad"][mod].abs()
            if not torch.is_tensor(edge_importance):
                edge_importance = torch.tensor([edge_importance])
            results["edge_importance"][mod] = edge_importance
            
        if include_edge_index and edge_indices:
            results["edge_index"][mod] = edge_indices[-1]

    return results

def main():
    pass

if __name__ == "__main__":
    main()