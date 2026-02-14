import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import time

# Core components
from dataset import MultiModalDataset, BrainGraphAugmentor, multimodal_collate
from model.encoders import SPECTEncoder, MRIEncoder, fMRIEncoder, DTIEncoder

class NeuroTrainer:
    def __init__(self, data_root, device='cpu', batch_size=32, lr=1e-4, 
                 alpha=-2.0, beta=2.0, use_aug=True, hub_name='fMRI',
                 aug_mask=0.15, aug_jitter=0.01, clip_val=1.0,
                 hidden_dim=128, embed_dim=1024, threshold=0.80):
        
        self.data_root = data_root
        self.device = device
        self.batch_size = batch_size  # Now set via argument
        self.hub_name = hub_name
        self.clip_val = clip_val
        
        # 1. Initialize Specialized Encoders
        self.models = torch.nn.ModuleDict({
            'SPECT': SPECTEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim).to(device),
            'MRI': MRIEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim).to(device),
            'fMRI': fMRIEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim, threshold=threshold).to(device),
            'DTI': DTIEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim, threshold=threshold).to(device)
        })
        
        # 2. Setup Augmentor
        if use_aug:
            print(f"✨ Augmentation: ENABLED (Mask: {aug_mask}, Jitter: {aug_jitter})")
            self.augmentor = BrainGraphAugmentor(edge_mask_prob=aug_mask, jitter_std=aug_jitter)
        else:
            print("🚫 Augmentation: DISABLED")
            self.augmentor = None
        
        self.alpha = alpha 
        self.beta = beta   
        self.optimizer = torch.optim.AdamW(self.models.parameters(), lr=lr)

    def prom3e_loss(self, z_spoke, z_hub):
        z_spoke = F.normalize(z_spoke, p=2, dim=1)
        z_hub = F.normalize(z_hub, p=2, dim=1)
        
        dists = torch.cdist(z_spoke, z_hub, p=2)
        logits = self.alpha * dists + self.beta
        labels = torch.arange(logits.size(0)).to(self.device)
        
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def run_training(self, epochs=100, train_ids=None):
        spokes = [m for m in self.models.keys() if m != self.hub_name]
        
        # Use self.batch_size here
        loaders = []
        for spoke in spokes:
            ds = MultiModalDataset(self.data_root, [spoke, self.hub_name], 
                                    allowed_ids=train_ids, transform=self.augmentor)
            if len(ds) > 0:
                loaders.append((spoke, DataLoader(ds, batch_size=self.batch_size, 
                                                 shuffle=True, collate_fn=multimodal_collate)))

        print(f"🚀 ProM3E Stage 1 | Hub: {self.hub_name} | Batch: {self.batch_size} | Threshold={self.models['fMRI'].threshold}")
        
        for epoch in range(epochs):
            self.models.train()
            epoch_pair_losses = {}
            total_loss = 0
            
            for spoke_name, loader in loaders:
                pair_batch_loss = 0
                for batch in loader:
                    z_spoke = self.models[spoke_name](batch[spoke_name].to(self.device))
                    z_hub = self.models[self.hub_name](batch[self.hub_name].to(self.device))
                    
                    loss = self.prom3e_loss(z_spoke, z_hub)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.models.parameters(), self.clip_val)
                    self.optimizer.step()
                    
                    pair_batch_loss += loss.item()
                
                avg_pair_loss = pair_batch_loss / len(loader)
                epoch_pair_losses[spoke_name] = avg_pair_loss
                total_loss += avg_pair_loss
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log_str = f"Epoch {epoch+1:03d}/{epochs} | Avg: {total_loss/len(loaders):.4f} | "
                log_str += " | ".join([f"{s}➔{self.hub_name[0]}: {l:.4f}" for s, l in epoch_pair_losses.items()])
                print(log_str)

    def export_embeddings(self, output_path):
        self.models.eval()
        results = {"embeddings": [], "labels": [], "ids": []}
        with torch.no_grad():
            for mod in self.models.keys():
                ds = MultiModalDataset(self.data_root, [mod], transform=None)
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=multimodal_collate)
                for batch in loader:
                    z = self.models[mod](batch[mod].to(self.device)).cpu()
                    results["embeddings"].append(z)
                    results["labels"].extend([mod] * z.size(0))
                    results["ids"].extend(batch['id'])
        results["embeddings"] = torch.cat(results["embeddings"], 0)
        torch.save(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--encoders_path', required=True)
    parser.add_argument('--split_path', required=True)
    
    # Sweepable Batch Size Added
    parser.add_argument('--batch_size', type=int, default=32, help="Input batch size for training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=-2.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--hub_name', default='fMRI', choices=['fMRI', 'MRI', 'SPECT', 'DTI'])
    parser.add_argument('--aug_mask', type=float, default=0.15)
    parser.add_argument('--aug_jitter', type=float, default=0.01)
    parser.add_argument('--clip_val', type=float, default=1.0)
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.80)
    
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--device', default='cpu')
    
    args, _ = parser.parse_known_args()

    # (Logic for loading train_ids remains same)
    train_ids = []
    if os.path.exists(args.split_path):
        with open(args.split_path, 'r') as f:
            mode = None
            for line in f:
                line = line.strip()
                if line == "train_ids:": mode = "train"
                elif line and mode == "train" and not line.startswith("#"):
                    train_ids.append(line)

    trainer = NeuroTrainer(
        args.data_root, args.device, args.batch_size, args.lr,
        alpha=args.alpha, beta=args.beta,
        use_aug=not args.no_aug, hub_name=args.hub_name,
        aug_mask=args.aug_mask, aug_jitter=args.aug_jitter, clip_val=args.clip_val,
        hidden_dim=args.hidden_dim, embed_dim=args.embed_dim, threshold=args.threshold
    )
    
    trainer.run_training(epochs=args.epochs, train_ids=train_ids)
    trainer.export_embeddings(args.embeddings_path)
    
    os.makedirs(os.path.dirname(args.encoders_path), exist_ok=True)
    torch.save({"models": {k: v.state_dict() for k, v in trainer.models.items()}}, args.encoders_path)
    print(f"✅ Stage 1 complete. Weights saved to {args.encoders_path}")