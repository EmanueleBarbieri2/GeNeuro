import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import combinations
import time
from model.contrastive.dataset import MultiModalDataset, multimodal_collate
import os

# Import your specific Graph Encoders
from model.encoders import SPECTEncoder, MRIEncoder, ConnectomeEncoder 

class NeuroTrainer:
    def __init__(self, data_root, device='cpu', batch_size=8, num_workers=0):
        self.data_root = data_root
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Instantiate all encoders (trainable)
        self.models = {
            'SPECT': SPECTEncoder().to(device),
            'MRI': MRIEncoder().to(device),
            'fMRI': ConnectomeEncoder().to(device),
            'DTI': ConnectomeEncoder().to(device)
        }
        
        # Create a single optimizer for all parameters
        # This allows gradients to flow through all encoders simultaneously
        all_params = []
        for model in self.models.values():
            all_params += list(model.parameters())
            
        self.optimizer = torch.optim.AdamW(all_params, lr=1e-4)

    '''def contrastive_loss(self, z1, z2, temp=0.07):
        """Symmetric InfoNCE Loss"""
        # Cosine similarity
        logits = torch.matmul(z1, z2.T) / temp
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # Loss in both directions (z1->z2 and z2->z1)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2'''

    
    def contrastive_loss(self, z1, z2, temp=0.07):
        """Symmetric InfoNCE Loss"""
        # 1. L2 Normalize the embeddings first! (Crucial for preventing collapse)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 2. Compute Cosine Similarity (which is now just the dot product of normalized vectors)
        logits = torch.matmul(z1, z2.T) / temp
        
        # 3. Create labels (the diagonal represents the positive pairs: patient_a == patient_a)
        labels = torch.arange(logits.size(0)).to(self.device)
        
        # 4. Calculate bidirectional cross entropy
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        
        return (loss_a + loss_b) / 2

    def train_pair_step(self, loader, mod_a, mod_b, log_interval=20):
        """Trains one epoch for a specific pair of modalities."""
        model_a = self.models[mod_a]
        model_b = self.models[mod_b]
        model_a.train()
        model_b.train()
        
        total_loss = 0
        start_t = time.perf_counter()
        num_batches = len(loader)
        
        for step, batch in enumerate(loader, start=1):
            # Move data to GPU
            data_a = batch[mod_a].to(self.device)
            data_b = batch[mod_b].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            z_a = model_a(data_a)
            z_b = model_b(data_b)
            
            # Calculate Symmetric Loss
            loss = self.contrastive_loss(z_a, z_b)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if step % log_interval == 0 or step == num_batches:
                elapsed = time.perf_counter() - start_t
                avg_time = elapsed / step
                print(
                    f"    [{mod_a}->{mod_b}] Batch {step}/{num_batches} | "
                    f"contrastive_loss={loss.item():.4f} | avg_time={avg_time:.2f}s/batch"
                )
            
        return total_loss / num_batches if num_batches > 0 else 0

    def run_training(self, epochs=50):
        all_mods = list(self.models.keys())
        
        # Generate all unique pairs: 
        #
        pairs = list(combinations(all_mods, 2))
        
        print(f"Starting All-Pairs Training on {self.device} (batch_size={self.batch_size})...")
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            epoch_loss = 0
            pairs_trained = 0
            
            for mod_a, mod_b in pairs:
                # 1. Create dataset dynamically for this pair
                dataset = MultiModalDataset(self.data_root, modalities=[mod_a, mod_b])
                
                # Skip if no intersection exists (e.g., fMRI-DTI might be 0)
                if len(dataset) == 0:
                    continue
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=multimodal_collate,
                    num_workers=self.num_workers
                )
                
                print(
                    f"  Pair [{mod_a}-{mod_b}] | Samples: {len(dataset)} | "
                    f"Batches: {len(loader)}"
                )
                # 2. Train on this pair
                loss = self.train_pair_step(loader, mod_a, mod_b)
                print(f"  Pair [{mod_a}-{mod_b}] | Epoch Loss: {loss:.4f}")
                
                epoch_loss += loss
                pairs_trained += 1
            
            avg_loss = epoch_loss / pairs_trained if pairs_trained > 0 else 0
            print(f"  > Average Epoch Loss: {avg_loss:.4f}")

        print("Training Complete. All encoders are aligned.")

    def export_embeddings(self, output_path="embeddings.pt", modalities=None, batch_size=None):
        """
        Export embeddings for each modality into a single file for visualization.
        Saves a dict with keys: embeddings (NxD), labels (N), ids (N).
        """
        if modalities is None:
            modalities = list(self.models.keys())
        if batch_size is None:
            batch_size = self.batch_size

        all_embeddings = []
        all_labels = []
        all_ids = []

        for mod in modalities:
            model = self.models[mod]
            model.eval()

            dataset = MultiModalDataset(self.data_root, modalities=[mod])
            if len(dataset) == 0:
                print(f"  [export] {mod}: 0 samples, skipped")
                continue

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=multimodal_collate,
                num_workers=self.num_workers
            )

            with torch.no_grad():
                for batch in loader:
                    data = batch[mod].to(self.device)
                    ids = batch['id']
                    z = model(data).detach().cpu()
                    all_embeddings.append(z)
                    all_labels.extend([mod] * z.size(0))
                    all_ids.extend(ids)

            print(f"  [export] {mod}: {len(dataset)} samples")

        if not all_embeddings:
            raise RuntimeError("No embeddings were exported. Check your datasets.")

        embeddings = torch.cat(all_embeddings, dim=0)
        payload = {
            "embeddings": embeddings,
            "labels": all_labels,
            "ids": all_ids
        }
        torch.save(payload, output_path)
        print(f"Saved embeddings to {output_path} (N={embeddings.size(0)})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--encoders_path', required=True)
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu', help="Device to use: 'cpu' or 'cuda'")

    # Legacy/Redundant args (added to prevent 'Unknown Argument' errors from pipeline)
    parser.add_argument('--out_encoders', help="Legacy flag")
    parser.add_argument('--out_embeddings', help="Legacy flag")

    args, unknown = parser.parse_known_args()

    # 1. Load unified split
    def load_split(split_path):
        train_ids, val_ids = set(), set()
        mode = None
        if not os.path.exists(split_path):
            print(f"⚠️ Warning: Split file not found at {split_path}. Training on all available data.")
            return None, None
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

    # 2. Initialize Trainer with CLI Arguments
    trainer = NeuroTrainer(
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 3. Apply the custom Learning Rate from CLI
    for g in trainer.optimizer.param_groups:
        g['lr'] = args.lr

    # 4. Patch run_training to use train_ids
    def run_training_with_split(self, epochs=50):
        all_mods = list(self.models.keys())
        pairs = list(combinations(all_mods, 2))
        print(f"Starting All-Pairs Training on {self.device} (batch_size={self.batch_size}, lr={args.lr})...")
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            epoch_loss = 0
            pairs_trained = 0
            for mod_a, mod_b in pairs:
                dataset = MultiModalDataset(self.data_root, modalities=[mod_a, mod_b], allowed_ids=train_ids)
                if len(dataset) == 0:
                    continue
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=multimodal_collate
                )
                print(f"  Pair [{mod_a}-{mod_b}] | Samples: {len(dataset)} | Batches: {len(loader)}")
                loss = self.train_pair_step(loader, mod_a, mod_b)
                print(f"  Pair [{mod_a}-{mod_b}] | Epoch Loss: {loss:.4f}")
                epoch_loss += loss
                pairs_trained += 1
            avg_loss = epoch_loss / pairs_trained if pairs_trained > 0 else 0
            print(f"  > Average Epoch Loss: {avg_loss:.4f}")

    trainer.run_training = run_training_with_split.__get__(trainer)

    # 5. Start Training and Export
    trainer.run_training(epochs=args.epochs)
    trainer.export_embeddings(output_path=args.embeddings_path)
    torch.save({"models": {k: v.state_dict() for k, v in trainer.models.items()}}, args.encoders_path)
