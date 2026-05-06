import os
import torch
from torch_geometric.data import Data, Batch

from model.encoders import SPECTEncoder, MRIEncoder, DTIEncoder, fMRIEncoder
from model.generator.generator import Generator
try:
    from model.downstream.downstream_classification import Classifier, CLASS_NAMES
except ImportError:
    from downstream.downstream_classification import Classifier, CLASS_NAMES

# 🌟 RESTORED: batch_explainers.py will dynamically shrink this list for us!
GLOBAL_MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]

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

def build_models(device="cpu", encoder_ckpt=None, generator_ckpt=None, classifier_ckpt=None):
    if not encoder_ckpt or not os.path.exists(encoder_ckpt):
        raise RuntimeError("Encoder checkpoint missing!")
        
    ckpt = torch.load(encoder_ckpt, map_location=device)
    
    sample_state = ckpt.get("models", {}).get("fMRI", {})
    if not sample_state: sample_state = ckpt.get("models", {}).get("DTI", {})
    
    h_dim = sample_state.get("node_init.0.bias", torch.zeros(256)).shape[0]
    e_dim = sample_state.get("projection.3.weight", torch.zeros(1024, 256)).shape[0]
    
    encoders = {
        "SPECT": SPECTEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "MRI": MRIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "fMRI": fMRIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
        "DTI": DTIEncoder(hidden_dim=h_dim, embed_dim=e_dim).to(device),
    }
    
    for mod, state in ckpt.get("models", {}).items():
        if mod in encoders:
            encoders[mod].load_state_dict(state)

    if generator_ckpt and os.path.exists(generator_ckpt):
        gen_ckpt = torch.load(generator_ckpt, map_location=device)
        gen_state = gen_ckpt.get("model_state", gen_ckpt)
        
        generator = Generator(
            embed_dim=e_dim,
            hidden_dim=1024,
            num_heads=8,
            num_layers=5,
            num_registers=0,
            mlp_depth=3
        ).to(device)
        generator.load_state_dict(gen_state)
    else:
        generator = None

    if not classifier_ckpt or not os.path.exists(classifier_ckpt):
        raise RuntimeError(f"Missing classifier checkpoint: {classifier_ckpt}")
    
    clf_ckpt = torch.load(classifier_ckpt, map_location=device)
    state = clf_ckpt.get("model_state", clf_ckpt)
    
    if "net.0.weight" in state: in_dim = state["net.0.weight"].shape[1]
    elif "0.weight" in state: in_dim = state["0.weight"].shape[1]
    else: in_dim = clf_ckpt.get("input_dim", 2050)
    
    if "net.8.bias" in state: num_classes = state["net.8.bias"].shape[0]
    else: num_classes = clf_ckpt.get("num_classes", 3)
    
    if "class_names" in clf_ckpt:
        global CLASS_NAMES
        CLASS_NAMES = clf_ckpt["class_names"]
        
    classifier = Classifier(in_dim, num_classes=num_classes).to(device)
    classifier.load_state_dict(state)
    classifier.input_dim = in_dim 

    for m in encoders.values(): m.eval()
    if generator is not None: generator.eval()
    classifier.eval()
    
    return encoders, generator, classifier


def explain_classification_subject_with_models(subject_id, data_root, true_label, encoders, generator, classifier, device="cpu", include_edge_index=True):
    graphs, available = {}, {} 
    
    # 🌟 Mask is safely locked to 4 slots. True means "Missing"
    mask = torch.ones(1, 4, dtype=torch.bool, device=device) 
    z_list = [] 

    # Only process the modalities that are actually active
    for mod in MOD_ORDER:
        g = load_graph(data_root, mod, subject_id) 
        if g is None: continue 
        g = g.to(device) 
        if hasattr(g, "x"): g.x.requires_grad_(True) 
        if hasattr(g, "edge_attr"): g.edge_attr.requires_grad_(True) 
        graphs[mod] = g 
        
        for module in encoders[mod].modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                if getattr(module, 'running_mean', None) is None:
                    module.running_mean = torch.zeros(module.num_features, device=device)
                if getattr(module, 'running_var', None) is None:
                    module.running_var = torch.ones(module.num_features, device=device)
                module.track_running_stats = True
        
        encoders[mod].eval()
        available[mod] = encoders[mod](g) 

    if not available: return None
    
    # 🌟 Build a perfect 4-slot tensor for the Generator
    for i, mod in enumerate(GLOBAL_MOD_ORDER):
        if mod in available: 
            z_list.append(available[mod].squeeze(0)) 
            mask[0, i] = False # Mark as present
        else: 
            z_list.append(torch.zeros(1024, device=device)) 
    
    input_tensor = torch.stack(z_list, dim=0).unsqueeze(0) 
    
    if generator is not None:
        z_recon, _, _ = generator(input_tensor, mask) 
        recon = z_recon[0]
    else:
        recon = [torch.zeros(1024, device=device) for _ in GLOBAL_MOD_ORDER]

    # 🌟 Build the dynamic tensor for the Classifier
    feat, mask_feat = [], [] 
    for mod in MOD_ORDER:
        idx = GLOBAL_MOD_ORDER.index(mod) # Find its slot in the recon list
        if mod in available:
            feat.append(available[mod].squeeze(0)) 
            mask_feat.append(1.0) 
        else:
            feat.append(recon[idx]) 
            mask_feat.append(0.0) 
    
    x_base = torch.cat(feat)
    
    if hasattr(classifier, 'input_dim') and classifier.input_dim == x_base.shape[0] + len(mask_feat):
        x = torch.cat([x_base, torch.tensor(mask_feat, dtype=torch.float32, device=device)])
    else:
        x = x_base 
        
    x = x.unsqueeze(0) 
    logits = classifier(x) 
    pred_idx = torch.argmax(logits, dim=1).item()
    prob = torch.softmax(logits, dim=1)[0, pred_idx].item()
    
    logits[0, pred_idx].backward() 

    results = {
        "subject_id": subject_id, 
        "true_label": true_label,
        "pred_label": CLASS_NAMES[pred_idx],
        "pred_prob": prob,
        "node_importance": {}, "node_value": {}, "node_grad": {}, "node_contrib": {}, 
        "edge_importance": {}, "edge_value": {}, "edge_grad": {}, "edge_contrib": {}
    } 
    
    for mod, g in graphs.items():
        if hasattr(g, "x") and g.x.grad is not None:
            # 🌟 RESTORED: .sum(dim=1) is required to flatten the feature matrix into 1D node scores
            results["node_importance"][mod] = g.x.grad.abs().sum(dim=1).cpu() 
            results["node_value"][mod] = g.x.detach().sum(dim=1).cpu() 
            results["node_grad"][mod] = g.x.grad.detach().sum(dim=1).cpu() 
            results["node_contrib"][mod] = (g.x.grad * g.x).detach().sum(dim=1).cpu() 
            
        if hasattr(g, "edge_attr") and g.edge_attr.grad is not None:
            edge_val = g.edge_attr.detach().cpu()
            edge_grad = g.edge_attr.grad.detach().cpu()
            if edge_val.dim() == 2 and edge_val.size(1) == 1: edge_val = edge_val.view(-1)
            if edge_grad.dim() == 2 and edge_grad.size(1) == 1: edge_grad = edge_grad.view(-1)
            results["edge_importance"][mod] = edge_grad.abs()
            results["edge_value"][mod] = edge_val
            results["edge_grad"][mod] = edge_grad 
            results["edge_contrib"][mod] = edge_val * edge_grad 
            
        if include_edge_index and hasattr(g, "edge_index"): 
            results["edge_index"] = results.get("edge_index", {})
            results["edge_index"][mod] = g.edge_index.cpu() 

    return results