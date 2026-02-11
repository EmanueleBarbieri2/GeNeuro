import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

class MultiModalDataset(Dataset):
    def __init__(self, root_dir, modalities, allowed_ids=None):
        """
        Args:
            root_dir (str): Path to 'data' folder containing modality subfolders.
            modalities (list): List of strings, e.g.,.
            allowed_ids (set or list): Only include samples with these IDs.
        """
        self.root = root_dir
        self.modalities = modalities
        # Automatically find the common patient IDs for this specific combination
        all_ids = self._find_intersection()
        if allowed_ids is not None:
            allowed_ids = set(allowed_ids)
            self.file_list = [fid for fid in all_ids if fid in allowed_ids]
        else:
            self.file_list = all_ids

    def _find_intersection(self):
        """Finds the intersection of filenames (IDs) across all requested modality folders."""
        if not self.modalities:
            return []

        # Get set of IDs (filenames without extension) for the first modality
        first_mod_path = os.path.join(self.root, self.modalities[0])
        common_ids = set([f.replace('.pt', '') for f in os.listdir(first_mod_path) if f.endswith('.pt')])

        # Intersect with all other requested modalities
        for mod in self.modalities[1:]:
            mod_path = os.path.join(self.root, mod)
            current_ids = set([f.replace('.pt', '') for f in os.listdir(mod_path) if f.endswith('.pt')])
            common_ids = common_ids.intersection(current_ids)

        common_ids = sorted(list(common_ids))
        # Optional: Print stats to verify intersections are being found
        # print(f"  > Found {len(common_ids)} paired samples for {self.modalities}")
        return common_ids

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        sample = {'id': file_id}

        for mod in self.modalities:
            file_path = os.path.join(self.root, mod, f"{file_id}.pt")
            # Load the PyG Data object
            graph_data = torch.load(file_path, weights_only=False)
            if isinstance(graph_data, dict):
                # Convert stored dicts to PyG Data
                data_kwargs = dict(graph_data)
                if "num_nodes" in data_kwargs and "x" not in data_kwargs:
                    data_kwargs["num_nodes"] = int(data_kwargs["num_nodes"])
                # If edge_weight is provided, map to edge_attr for consistency
                if "edge_weight" in data_kwargs and "edge_attr" not in data_kwargs:
                    data_kwargs["edge_attr"] = data_kwargs.pop("edge_weight")
                graph_data = Data(**data_kwargs)
            # Normalize Data fields across modalities
            if hasattr(graph_data, "edge_weight") and not hasattr(graph_data, "edge_attr"):
                graph_data.edge_attr = graph_data.edge_weight
            if hasattr(graph_data, "edge_attr") and torch.is_tensor(graph_data.edge_attr):
                if graph_data.edge_attr.dim() == 2 and graph_data.edge_attr.size(1) == 1:
                    graph_data.edge_attr = graph_data.edge_attr.view(-1)
            sample[mod] = graph_data
            
        return sample

def multimodal_collate(batch_list):
    """
    Custom collate function to batch dictionaries of graphs.
    """
    batch_output = {'id': [item['id'] for item in batch_list]}
    
    # Identify which modalities are present in this batch
    modalities = [k for k in batch_list[0].keys() if k != 'id']
    
    for mod in modalities:
        # Extract list of graphs for this modality
        graphs = [item[mod] for item in batch_list]
        # Batch them into a single super-graph (PyG standard)
        batch_output[mod] = Batch.from_data_list(graphs)
        
    return batch_output