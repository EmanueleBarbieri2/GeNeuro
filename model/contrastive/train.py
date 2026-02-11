import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os

from model.contrastive.dataset import MultiModalDataset, multimodal_collate
from model.encoders import SPECTEncoder, MRIEncoder, ConnectomeEncoder 

class NeuroTrainer:
    def __init__(self, data_root, device='cpu', batch_size=8, num_workers=0):
        self.data_root = data_root
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.models = {
            'SPECT': SPECTEncoder().to(device),
            'MRI': MRIEncoder().to(device),
            'fMRI': ConnectomeEncoder().to(device),
            'DTI': ConnectomeEncoder().to(device)
        }
        
        # ProM3E Hyperparameters from Table 7 [cite: 715]
        self.alpha = -5.0 # Base scale parameter
        self.beta = 5.0   # Base shift parameter
        
        all_params = []
        for model in self.models.values():
            all_params += list(model.parameters())
            
        self.optimizer = torch.optim.AdamW(all_params, lr=1e-4)

    def prom3e_contrastive_loss(self, z_spoke, z_hub):
        """
        Implementation of ProM3E Equation 4[cite: 162].
        Uses Euclidean distances for a contrastive objective to learn intra-modal distributions.
        """
        # 1. Normalize embeddings as ProM3E uses global normalized embeddings [cite: 108]
        z_spoke = F.normalize(z_spoke, p=2, dim=1)
        z_hub = F.normalize(z_hub, p=2, dim=1)
        
        # 2. Calculate Pairwise Euclidean Distance Matrix (N x N) [cite: 158]
        # dists[i, j] = ||z_spoke[i] - z_hub[j]||^2
        dists = torch.cdist(z_spoke, z_hub, p=2)
        
        # 3. Apply Scaling and Shifting (Equation 4) [cite: 162, 164]
        # We use a negative alpha because smaller distance should result in higher logit
        logits = self.alpha * dists + self.beta
        
        # 4. Positive pairs are on the diagonal (Patient A Spoke <-> Patient A Hub)
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # 5. Symmetric InfoNCE on distances [cite: 162]
        loss_spoke_to_hub = F.cross_entropy(logits, labels)
        loss_hub_to_spoke = F.cross_entropy(logits.T, labels)
        
        return (loss_spoke_to_hub + loss_hub_to_spoke) / 2

    def train_pair_step(self, loader, spoke, hub, log_interval=20):
        model_spoke, model_hub = self.models[spoke], self.models[hub]
        model_spoke.train(); model_hub.train()
        
        total_loss, start_t = 0, time.perf_counter()
        num_batches = len(loader)
        
        for step, batch in enumerate(loader, start=1):
            data_spoke = batch[spoke].to(self.device)
            data_hub = batch[hub].to(self.device)
            
            self.optimizer.zero_grad()
            z_spoke, z_hub = model_spoke(data_spoke), model_hub(data_hub)
            
            # Apply the ProM3E Distance-based Loss [cite: 162]
            loss = self.prom3e_contrastive_loss(z_spoke, z_hub)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
            if step % log_interval == 0 or step == num_batches:
                avg_time = (time.perf_counter() - start_t) / step
                print(f"    [{spoke}->{hub}] Batch {step}/{num_batches} | loss={loss.item():.4f}")
            
        return total_loss / num_batches if num_batches > 0 else 0

    def run_training(self, epochs=1, train_ids=None):
        hub = 'MRI'
        spokes = ['SPECT', 'fMRI', 'DTI']
        pairs = [(spoke, hub) for spoke in spokes]
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            epoch_loss, pairs_trained = 0, 0
            for spoke, current_hub in pairs:
                dataset = MultiModalDataset(self.root, [spoke, current_hub], allowed_ids=train_ids)
                if len(dataset) == 0: continue
                    
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=multimodal_collate)
                print(f"  Pair [{spoke}-{current_hub}] | Samples: {len(dataset)}")
                loss = self.train_pair_step(loader, spoke, current_hub)
                epoch_loss += loss; pairs_trained += 1
                
            print(f"  > Average Epoch Loss: {epoch_loss/max(pairs_trained,1):.4f}")

    def export_embeddings(self, output_path="embeddings.pt"):
        all_embeddings, all_labels, all_ids = [], [], []
        for mod in self.models.keys():
            model = self.models[mod]
            model.eval()
            dataset = MultiModalDataset(self.data_root, modalities=[mod])
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=multimodal_collate)
            with torch.no_grad():
                for batch in loader:
                    z = model(batch[mod].to(self.device)).detach().cpu()
                    all_embeddings.append(z)
                    all_labels.extend([mod] * z.size(0))
                    all_ids.extend(batch['id'])
            print(f"  [export] {mod}: {len(dataset)} samples")
        torch.save({"embeddings": torch.cat(all_embeddings, dim=0), "labels": all_labels, "ids": all_ids}, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--encoders_path', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()

    trainer = NeuroTrainer(args.data_root, args.device, args.batch_size)
    trainer.root = args.data_root # Patch for run_training
    trainer.run_training(epochs=args.epochs)
    trainer.export_embeddings(args.embeddings_path)
    torch.save({"models": {k: v.state_dict() for k, v in trainer.models.items()}}, args.encoders_path)