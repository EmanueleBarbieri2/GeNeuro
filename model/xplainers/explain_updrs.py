import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from model.encoders import SPECTEncoder, MRIEncoder, DTIEncoder, fMRIEncoder
from model.downstream.downstream_updrs import Regressor
from model.generator.generator import Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

def load_graph(data_root, modality, subject_id):
    path = os.path.join(data_root, modality, f"{subject_id}.pt")
    if not os.path.exists(path): return None
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
    models_dict = ckpt.get("models", {})
    
    # Safely find ANY available modality to infer dimensions
    sample_state = {}
    for mod in ["fMRI", "DTI", "SPECT", "MRI"]:
        if mod in models_dict and len(models_dict[mod]) > 0:
            sample_state = models_dict[mod]
            break
            
    # Fallbacks
    h_dim = 256 
    e_dim = 1024
    
    # Infer h_dim based on whatever model we found
    if "conv1.bias" in sample_state:  # Usually SPECT/MRI
        h_dim = sample_state["conv1.bias"].shape[0]
    elif "node_init.0.bias" in sample_state: # Usually fMRI/DTI
        h_dim = sample_state["node_init.0.bias"].shape[0]
        
    # Infer e_dim
    if "projection.3.weight" in sample_state:
        e_dim = sample_state["projection.3.weight"].shape[0]
    
    encoders = {
        "SPECT": SPECTEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "MRI": MRIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "fMRI": fMRIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "DTI": DTIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
    }
    
    for mod, state in ckpt.get("models", {}).items():
        if mod in encoders:
            encoders[mod].load_state_dict(state)

    # --- FIX 1: Safely handle disabled generator ---
    if generator_ckpt and os.path.exists(generator_ckpt):
        gen_ckpt = torch.load(generator_ckpt, map_location=device)
        gen_state = gen_ckpt.get("model_state", {})
        layer_indices = [int(k.split(".")[2]) for k in gen_state.keys() if k.startswith("transformer.layers.")]
        num_layers = max(layer_indices) + 1 if layer_indices else 3

        num_registers = int(gen_state.get("register_tokens", torch.zeros(4, 1, e_dim)).shape[0])

        # infer hidden_dim from dim_feedforward (= hidden_dim * 4)
        ff_key = "transformer.layers.0.linear1.weight"
        if ff_key in gen_state:
            hidden_dim = int(gen_state[ff_key].shape[0] // 4)
        else:
            hidden_dim = 512

        # infer mlp_depth from highest projector linear index: 0,3,6,... => depth=(idx/3)+1
        proj_indices = []
        for k in gen_state.keys():
            if k.startswith("modality_projectors.0.") and k.endswith(".weight"):
                parts = k.split(".")
                if len(parts) >= 4 and parts[2].isdigit():
                    proj_indices.append(int(parts[2]))
        mlp_depth = int(max(proj_indices) // 3 + 1) if proj_indices else 2

        generator = Generator(
            embed_dim=e_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=num_layers,
            num_registers=num_registers,
            mlp_depth=mlp_depth,
        ).to(device)
        generator.load_state_dict(gen_state)
    else:
        # Prevent spawning an untrained, random generator
        generator = None

    # Auto-Detect Regressor Input Dimensions
    if not regressor_ckpt or not os.path.exists(regressor_ckpt):
        raise RuntimeError(f"Missing regressor checkpoint: {regressor_ckpt}")
    
    reg_ckpt = torch.load(regressor_ckpt, map_location=device)
    state = reg_ckpt.get("model_state", reg_ckpt)
    
    if "net.0.weight" in state:
        input_dim = state["net.0.weight"].shape[1]
    elif "0.weight" in state:
        input_dim = state["0.weight"].shape[1]
    else:
        input_dim = 4096
    
    regressor = Regressor(input_dim).to(device)
    regressor.load_state_dict(state)
    regressor.input_dim = input_dim

    for m in encoders.values(): m.eval()
    if generator is not None: generator.eval()
    regressor.eval()
    return encoders, generator, regressor

def explain_subject_with_models(subject_id, data_root, encoders, generator, regressor, target_idx=0, device="cpu", include_edge_index=True):
    graphs, available = {}, {} 
    mask = torch.ones(1, 4, dtype=torch.bool, device=device) 
    z_list = [] 

    for i, mod in enumerate(MOD_ORDER):
        g = load_graph(data_root, mod, subject_id) 
        if g is None: continue 
        g = g.to(device) 
        if hasattr(g, "x"): g.x.requires_grad_(True) 
        if hasattr(g, "edge_attr"): g.edge_attr.requires_grad_(True) 
        graphs[mod] = g 
        
        # --- THE BATCHNORM FIX ---
        for module in encoders[mod].modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                if getattr(module, 'running_mean', None) is None:
                    module.running_mean = torch.zeros(module.num_features, device=device)
                if getattr(module, 'running_var', None) is None:
                    module.running_var = torch.ones(module.num_features, device=device)
                module.track_running_stats = True
        # -------------------------
        
        available[mod] = encoders[mod](g) 
        mask[0, i] = False 

    if not available: return None
    
    for mod in MOD_ORDER:
        if mod in available: z_list.append(available[mod].squeeze(0)) 
        else: z_list.append(torch.zeros(1024, device=device)) 
    
    input_tensor = torch.stack(z_list, dim=0).unsqueeze(0) 
    
    # --- FIX 2: Implement Zero-Imputation Fallback ---
    if generator is not None:
        z_recon, _, _ = generator(input_tensor, mask) 
        recon = z_recon[0] 
    else:
        recon = [torch.zeros(1024, device=device) for _ in MOD_ORDER]

    feat, mask_feat = [], [] 
    for i, mod in enumerate(MOD_ORDER):
        if mod in available:
            feat.append(available[mod].squeeze(0)) 
            mask_feat.append(1.0) 
        else:
            feat.append(recon[i]) 
            mask_feat.append(0.0) 
    
    # DYNAMIC MASK ATTACHMENT
    x_base = torch.cat(feat)
    if hasattr(regressor, 'input_dim') and regressor.input_dim == x_base.shape[0] + len(mask_feat):
        x = torch.cat([x_base, torch.tensor(mask_feat, dtype=torch.float32, device=device)])
    else:
        x = x_base 
        
    x = x.unsqueeze(0) 
    pred = regressor(x) 
    score = pred[0, target_idx] if pred.shape[1] > 1 else pred[0, 0] 
    
    # Backpropagate
    score.backward() 

    results = {
        "subject_id": subject_id, "prediction": score.item(), 
        "node_importance": {}, "node_value": {}, "node_grad": {}, "node_contrib": {}, 
        "edge_importance": {}, "edge_value": {}, "edge_grad": {}, "edge_contrib": {}
    } 
    
    for mod, g in graphs.items():
        if hasattr(g, "x") and g.x.grad is not None:
            results["node_importance"][mod] = g.x.grad.abs().sum(dim=1).cpu() 
            results["node_value"][mod] = g.x.detach().cpu().sum(dim=1) 
            results["node_grad"][mod] = g.x.grad.detach().cpu().sum(dim=1) 
            results["node_contrib"][mod] = (g.x.grad * g.x).detach().cpu().sum(dim=1) 
        if hasattr(g, "edge_attr") and g.edge_attr.grad is not None:
            results["edge_importance"][mod] = g.edge_attr.grad.abs().cpu() 
            results["edge_value"][mod] = g.edge_attr.detach().cpu() 
            results["edge_grad"][mod] = g.edge_attr.grad.detach().cpu() 
            results["edge_contrib"][mod] = (g.edge_attr.grad * g.edge_attr).detach().cpu() 
        if include_edge_index: results["edge_index"] = {mod: g.edge_index.cpu() for mod, g in graphs.items()} 

    return results

def main():
    pass

if __name__ == "__main__":
    main()