import os
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

class BrainGraphAugmentor:
    """
    Stochastic augmentations to prevent 'shortcut learning' in contrastive tasks.
    """
    def __init__(self, edge_mask_prob=0.15, jitter_std=0.01):
        self.edge_mask_prob = edge_mask_prob
        self.jitter_std = jitter_std

    def __call__(self, data):
        data = data.clone()
        
        # 1. Edge Masking: Hide random connections to force robustness
        if self.edge_mask_prob > 0 and hasattr(data, 'edge_index'):
            num_edges = data.edge_index.size(1)
            mask = torch.rand(num_edges) > self.edge_mask_prob
            data.edge_index = data.edge_index[:, mask]
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]

        # 2. Node Jitter: Tiny noise on XYZ coordinates 
        if self.jitter_std > 0 and hasattr(data, 'x') and data.x is not None:
            data.x = data.x + torch.randn_like(data.x) * self.jitter_std
            
        return data

class MultiModalDataset(Dataset):
    def __init__(self, root_dir, modalities, allowed_ids=None, transform=None):
        """
        Args:
            root_dir (str): Path to 'data' folder.
            modalities (list): List of strings.
            allowed_ids (set or list): Filter for specific subject IDs.
            transform (callable, optional): Augmentation logic for contrastive learning.
        """
        self.root = root_dir
        self.modalities = modalities
        self.transform = transform 
        
        all_ids = self._find_intersection()
        if allowed_ids is not None:
            allowed_ids = set(allowed_ids)
            self.file_list = [fid for fid in all_ids if fid in allowed_ids]
        else:
            self.file_list = all_ids

    def _find_intersection(self):
        """Finds common IDs across all requested modality folders."""
        if not self.modalities: return []

        first_mod_path = os.path.join(self.root, self.modalities[0])
        common_ids = set([f.replace('.pt', '') for f in os.listdir(first_mod_path) if f.endswith('.pt')])

        for mod in self.modalities[1:]:
            mod_path = os.path.join(self.root, mod)
            current_ids = set([f.replace('.pt', '') for f in os.listdir(mod_path) if f.endswith('.pt')])
            common_ids = common_ids.intersection(current_ids)

        return sorted(list(common_ids))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        sample = {'id': file_id}

        for mod in self.modalities:
            file_path = os.path.join(self.root, mod, f"{file_id}.pt")
            graph_data = torch.load(file_path, weights_only=False)
            
            # --- YOUR ESSENTIAL STANDARDIZATION LOGIC ---
            if isinstance(graph_data, dict):
                data_kwargs = dict(graph_data)
                if "num_nodes" in data_kwargs and "x" not in data_kwargs:
                    data_kwargs["num_nodes"] = int(data_kwargs["num_nodes"])
                if "edge_weight" in data_kwargs and "edge_attr" not in data_kwargs:
                    data_kwargs["edge_attr"] = data_kwargs.pop("edge_weight")
                graph_data = Data(**data_kwargs)

            if hasattr(graph_data, "edge_weight") and not hasattr(graph_data, "edge_attr"):
                graph_data.edge_attr = graph_data.edge_weight
            
            if hasattr(graph_data, "edge_attr") and torch.is_tensor(graph_data.edge_attr):
                if graph_data.edge_attr.dim() == 2 and graph_data.edge_attr.size(1) == 1:
                    graph_data.edge_attr = graph_data.edge_attr.view(-1)
            
            # --- NEW: APPLY AUGMENTATION ---
            # This is critical for Contrastive Learning to prevent overfitting
            if self.transform is not None:
                graph_data = self.transform(graph_data)

            sample[mod] = graph_data
            
        return sample

def multimodal_collate(batch_list):
    """Custom collate function to batch dictionaries of graphs."""
    batch_output = {'id': [item['id'] for item in batch_list]}
    modalities = [k for k in batch_list[0].keys() if k != 'id']
    
    for mod in modalities:
        graphs = [item[mod] for item in batch_list]
        batch_output[mod] = Batch.from_data_list(graphs)
        
    return batch_output