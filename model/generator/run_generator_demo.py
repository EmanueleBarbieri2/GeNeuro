import torch
import os
from collections import defaultdict
from model.generator.generator import ProM3E_Generator


def load_embeddings(path):
    payload = torch.load(path, map_location="cpu")
    return payload["embeddings"], payload["labels"], payload["ids"]


def hallucinate_missing_modalities(model, available_embeddings, device="cpu"):
    """
    available_embeddings: Dict {'MRI': tensor, 'SPECT': tensor...}
    Returns: Dict with ALL 4 modalities filled in.
    """
    model.eval()

    mod_order = ["SPECT", "MRI", "fMRI", "DTI"]
    input_tensor = torch.zeros(1, 4, 1024, device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i, mod in enumerate(mod_order):
            if mod in available_embeddings:
                input_tensor[0, i] = available_embeddings[mod].to(device)
                mask[0, i] = False

        z_recon, mu, _ = model(input_tensor, mask)
        reconstructed = z_recon[0]  # shape [4, 1024]

    result = {}
    for i, mod in enumerate(mod_order):
        result[mod] = reconstructed[i].detach().cpu()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--device', default='cpu')
    args, unknown = parser.parse_known_args()
    embeddings, labels, ids = load_embeddings(args.embeddings_path)
    # Build subject -> modality -> idx
    by_id = defaultdict(dict)
    for idx, (mod, sid) in enumerate(zip(labels, ids)):
        by_id[sid][mod] = idx
    model = ProM3E_Generator(embed_dim=1024).to(args.device)
    ckpt_path = args.generator_ckpt
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded checkpoint from {ckpt_path}")
    out_dict = {}
    for sid, mods in by_id.items():
        available = {mod: embeddings[idx] for mod, idx in mods.items()}
        real = {mod: embeddings[idx] for mod, idx in mods.items()}
        recon = hallucinate_missing_modalities(model, available_embeddings=available, device=args.device)
        out_dict[sid] = {"real": real, "recon": recon}
    # Always save recon_demo.pt to model/model/checkpoints (never model/model/model/checkpoints)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    recon_demo_path = os.path.join(checkpoints_dir, 'recon_demo.pt')
    torch.save(out_dict, recon_demo_path)
    print(f"Saved recon_demo.pt to {recon_demo_path}")
    print(f"Total subjects processed: {len(out_dict)}")

if __name__ == "__main__":
    main()
