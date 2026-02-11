import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, allowed_ids=None):
        payload = torch.load(embeddings_path, map_location="cpu")
        embeddings = payload["embeddings"]
        labels = payload["labels"]
        ids = payload["ids"]

        by_id = defaultdict(dict)
        for idx, (mod, sid) in enumerate(zip(labels, ids)):
            if mod not in by_id[sid]:
                by_id[sid][mod] = embeddings[idx]

        self.samples = []
        for sid, mods in by_id.items():
            if allowed_ids is not None and sid not in allowed_ids:
                continue
            x = torch.zeros(4, embeddings.size(1))
            missing = torch.ones(4, dtype=torch.bool)
            for i, mod in enumerate(MOD_ORDER):
                if mod in mods:
                    x[i] = mods[mod]
                    missing[i] = False
            self.samples.append((sid, x, missing))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, x, missing = self.samples[idx]
        return sid, x, missing


def train_prom3e_step(model, batch_embeddings, real_missing_mask, optimizer, mask_prob=0.5, temp=0.07):
    optimizer.zero_grad()

    B, S, D = batch_embeddings.shape
    device = batch_embeddings.device

    available_mask = ~real_missing_mask
    training_mask = (torch.rand(B, S, device=device) < mask_prob) & available_mask

    model_input_mask = training_mask | real_missing_mask

    z_recon, mu, logvar = model(batch_embeddings, model_input_mask)

    # Contrastive reconstruction loss (InfoNCE over embeddings)
    recon_loss = torch.tensor(0.0, device=device)
    recon_terms = 0
    if training_mask.any():
        B, S, D = batch_embeddings.shape
        for s in range(S):
            mask_s = training_mask[:, s]
            if mask_s.sum() < 2:
                continue
            z_pred = z_recon[:, s, :][mask_s]
            z_true = batch_embeddings[:, s, :][mask_s]
            # Normalize
            z_pred = F.normalize(z_pred, p=2, dim=1)
            z_true = F.normalize(z_true, p=2, dim=1)
            logits = torch.matmul(z_pred, z_true.T) / temp
            labels = torch.arange(logits.size(0), device=device)
            recon_loss = recon_loss + F.cross_entropy(logits, labels)
            recon_terms += 1
        if recon_terms > 0:
            recon_loss = recon_loss / recon_terms

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    total_loss = recon_loss + 0.001 * kl_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), recon_loss.item(), kl_loss.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', required=True, help='Absolute path to embeddings.pt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--split_path', default='data/unified_split.txt')
    args, unknown = parser.parse_known_args()

    embeddings_path = os.path.abspath(args.embeddings_path)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load unified split
    def load_split(split_path):
        train_ids, val_ids = set(), set()
        mode = None
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if line == 'train_ids:':
                    mode = 'train'
                elif line == 'val_ids:':
                    mode = 'val'
                elif line:
                    if mode == 'train':
                        train_ids.add(line)
                    elif mode == 'val':
                        val_ids.add(line)
        return train_ids, val_ids

    train_ids, val_ids = load_split(args.split_path)

    dataset = EmbeddingDataset(embeddings_path, allowed_ids=train_ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ProM3E_Generator(embed_dim=1024).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Training ProM3E generator on {args.device} for {args.epochs} epochs...")
    print(f"Samples: {len(dataset)} | Batches: {len(loader)}")

    for epoch in range(args.epochs):
        total = 0.0
        recon_total = 0.0
        kl_total = 0.0
        for _, x, missing in loader:
            x = x.to(args.device)
            missing = missing.to(args.device)
            loss, recon, kl = train_prom3e_step(model, x, missing, optimizer)
            total += loss
            recon_total += recon
            kl_total += kl
        avg = total / max(len(loader), 1)
        avg_recon = recon_total / max(len(loader), 1)
        avg_kl = kl_total / max(len(loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs} | total_loss={avg:.4f} | recon_loss={avg_recon:.4f} | kl_loss={avg_kl:.4f}")

    # os already imported at top
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, args.out)
    print(f"Saved generator checkpoint to {args.out}")


if __name__ == "__main__":
    main()