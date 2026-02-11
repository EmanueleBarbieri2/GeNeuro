import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GCNConv, global_mean_pool

class SPECTEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, embed_dim=1024):
        super().__init__()
        # Light 3-layer GCN for the 7-node graph
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        
        # Project to latent space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim) # Output: 1024
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Pool node features into a single graph vector
        x = global_mean_pool(x, batch) 
        
        # Project to NeuroBind space
        embedding = self.projection(x)
        
        # CRITICAL: Normalize to hypersphere for Contrastive Learning
        return F.normalize(embedding, p=2, dim=1)
    


from torch_geometric.nn import GATv2Conv

class MRIEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, embed_dim=1024):
        super().__init__()
        # GATv2 allows the model to learn edge importance dynamically
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim * 2, heads=4, concat=True, edge_dim=1)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 8, 1024), # 8 = hidden * 2 * 4 heads/layer adjustment
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        
        # Pass edge_attr (1 - d/d_max) to guide attention
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        
        x = global_mean_pool(x, batch)
        embedding = self.projection(x)
        return F.normalize(embedding, p=2, dim=1)
    
class ConnectomeEncoder(nn.Module):
    def __init__(self, num_nodes=113, hidden_dim=128, embed_dim=1024):
        super().__init__()
        # Learnable embedding for each of the 113 brain regions
        # This replaces the missing node features
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.01)
        
        self.conv1 = GraphConv(hidden_dim, hidden_dim * 2)
        self.conv2 = GraphConv(hidden_dim * 2, hidden_dim * 4)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, data):
        # Expand learnable embeddings to match batch size
        batch_size = getattr(data, 'num_graphs', 1)
        if not isinstance(batch_size, int) or batch_size < 1:
            batch_size = 1
        x = self.node_embedding.repeat(batch_size, 1)

        edge_weight = data.edge_attr
        if edge_weight is not None:
            if edge_weight.dim() > 1:
                edge_weight = edge_weight.view(-1)
            # Allow negative edge weights for GraphConv

        x = F.relu(self.conv1(x, data.edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, data.edge_index, edge_weight=edge_weight))

        x = global_mean_pool(x, data.batch)
        embedding = self.projection(x)
        return F.normalize(embedding, p=2, dim=1)