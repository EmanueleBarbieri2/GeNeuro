import torch
import os
from collections import defaultdict
from model.generator.generator import ProM3E_Generator

def load_embeddings(path):
    """Loads stage 1 embeddings and metadata."""
    payload = torch.load(path, map_location="cpu")
    return payload["embeddings"], payload["labels"], payload["ids"]

def hallucinate_missing_modalities(model, available_embeddings, device="cpu"):
    model.eval()
    mod_order = ["SPECT", "MRI", "fMRI", "DTI"]
    
    first_key = list(available_embeddings.keys())[0]
    embed_dim = available_embeddings[first_key].shape[-1]
    
    input_tensor = torch.zeros(1, 4, embed_dim, device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i, mod in enumerate(mod_order):
            if mod in available_embeddings:
                input_tensor[0, i] = available_embeddings[mod].to(device)
                mask[0, i] = False

        z_recon, mu, _ = model(input_tensor, mask)
        reconstructed = z_recon[0] 

    result = {}
    for i, mod in enumerate(mod_order):
        result[mod] = reconstructed[i].detach().cpu()
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    parser.add_argument('--device', default='cpu')
    
    # --- NEW: Architectural Args to match checkpoint ---
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_registers', type=int, default=4)
    parser.add_argument('--mlp_depth', type=int, default=2)
    
    args, unknown = parser.parse_known_args()

    # 1. Load Real Data
    embeddings, labels, ids = load_embeddings(args.embeddings_path)
    if len(embeddings) == 0:
        raise ValueError("No embeddings found.")
    actual_embed_dim = embeddings.size(1)

    # 2. Build subject registry
    by_id = defaultdict(dict)
    for idx, (mod, sid) in enumerate(zip(labels, ids)):
        by_id[sid][mod] = idx

    # 3. Initialize Model with CORRECT architecture
    print(f"🧠 Loading Generator ({args.num_layers}L, {args.hidden_dim}H) from {args.generator_ckpt}...")
    
    model = ProM3E_Generator(
        embed_dim=actual_embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_registers=args.num_registers,
        mlp_depth=args.mlp_depth
    ).to(args.device)
    
    ckpt = torch.load(args.generator_ckpt, map_location=args.device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    # 4. Process
    out_dict = {}
    print(f"🧪 Hallucinating missing modalities for {len(by_id)} subjects...")
    
    for sid, mods in by_id.items():
        real_data = {mod: embeddings[idx] for mod, idx in mods.items()}
        raw_recon = hallucinate_missing_modalities(model, available_embeddings=real_data, device=args.device)
        
        smart_recon = {}
        for mod in ["SPECT", "MRI", "fMRI", "DTI"]:
            if mod in real_data:
                smart_recon[mod] = real_data[mod].cpu()
            else:
                smart_recon[mod] = raw_recon[mod]
        
        out_dict[sid] = {"real": real_data, "recon": smart_recon}

    checkpoints_dir = os.path.dirname(args.generator_ckpt)
    os.makedirs(checkpoints_dir, exist_ok=True)
    recon_demo_path = os.path.join(checkpoints_dir, 'recon_demo.pt')
    torch.save(out_dict, recon_demo_path)
    print(f"✅ Saved Smart Reconstructions to {recon_demo_path}")

if __name__ == "__main__":
    main()