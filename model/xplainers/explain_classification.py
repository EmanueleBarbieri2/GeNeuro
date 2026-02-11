import argparse
import os
import torch
import json
try:
    from model.downstream.downstream_classification import Classifier, CLASS_NAMES, load_embeddings, load_csv_labels, hallucinate_missing_modalities, MOD_ORDER
except ModuleNotFoundError:
    from downstream.downstream_classification import Classifier, CLASS_NAMES, load_embeddings, load_csv_labels, hallucinate_missing_modalities, MOD_ORDER
from model.generator.generator import ProM3E_Generator

def explain_subject(subject_id, embeddings_by_id, label, generator, classifier, device="cpu"):
    mods = embeddings_by_id.get(subject_id)
    if not mods:
        return None
    available = {m: mods[m] for m in mods.keys()}
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
    x = torch.cat([x, torch.tensor(mask_feat, dtype=torch.float32)], dim=0)
    x = x.unsqueeze(0).to(device)
    x.requires_grad_(True)
    logits = classifier(x)
    pred = torch.argmax(logits, dim=1).item()
    prob = torch.softmax(logits, dim=1)[0, pred].item()
    logits[0, pred].backward()
    grad = x.grad.detach().cpu().numpy().tolist()
    return {
        "subject_id": subject_id,
        "true_label": label,
        "pred_label": CLASS_NAMES[pred],
        "pred_prob": prob,
        "feature_grad": grad,
        "feature_names": [f"{mod}_{i}" for mod in MOD_ORDER for i in range(1024)] + [f"mask_{i}" for i in range(4)]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--classifier_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    labels = load_csv_labels(args.csv_path)
    embeddings_by_id = load_embeddings(args.embeddings_path)
    generator = ProM3E_Generator(embed_dim=1024)
    if os.path.exists(args.generator_ckpt):
        generator.load_state_dict(torch.load(args.generator_ckpt, map_location=args.device)["model_state"])
    classifier = Classifier(4*1024+4)
    if os.path.exists(args.classifier_ckpt):
        classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=args.device)["model_state"])
    classifier.eval()
    generator.eval()
    label = labels.get(args.subject, None)
    result = explain_subject(args.subject, embeddings_by_id, label, generator, classifier, device=args.device)
    # Save output to checkpoints directory
    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    out_path = os.path.join(checkpoints_dir, os.path.basename(args.out))
    if result:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved classification explainer for {args.subject} to {out_path}")
    else:
        print(f"No embeddings for {args.subject}")

if __name__ == "__main__":
    main()
