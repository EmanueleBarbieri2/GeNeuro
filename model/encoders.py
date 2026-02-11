import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GCNConv, global_max_pool

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Using Kaiming Normal to keep variance high in deep layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MRIEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, embed_dim=1024, num_nodes=113):
        super().__init__()
        # 1. Feature Scaling per ROI (Highly effective for isolating baseline differences)
        self.roi_scaler = nn.Parameter(torch.ones(num_nodes, input_dim))
        
        # 2. Distance-Weighted Graph Convolution
        # GCNConv natively applies the (1 - d/dmax) edge weights during message passing
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 3. Projection Head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, affine=True, track_running_stats=False), # Force variance
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        # We now extract edge_index and edge_attr from the PyG data object
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = batch.max().item() + 1
        
        # 1. Apply ROI specific scaling
        scaler = self.roi_scaler.repeat(num_graphs, 1)
        x = x * scaler
        
        # 2. Extract 1D edge weights for GCNConv
        edge_weight = None
        if edge_attr is not None:
            # Squeeze guarantees shape is [num_edges] instead of [num_edges, 1]
            edge_weight = edge_attr.squeeze() if edge_attr.dim() > 1 else edge_attr

        # 3. Graph Message Passing
        # Information flows strongly between nearby ROIs, weakly between distant ones
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x, 0.2)
        
        # 4. Global Max Pooling (Isolate the highest activation / most severe localized atrophy)
        z = global_max_pool(x, batch)
        
        # 5. Project and L2 Normalize
        return F.normalize(self.projection(z), p=2, dim=1)

class ConnectomeEncoder(nn.Module):
    def __init__(self, hidden_dim=128, embed_dim=1024): # <-- Increased capacity to 128
        super().__init__()
        # Use LeakyReLU to maintain variance and prevent dead neurons
        self.node_init = nn.Sequential(nn.Linear(1, hidden_dim), nn.LeakyReLU(0.2))
        self.edge_init = nn.Sequential(nn.Linear(1, hidden_dim), nn.LeakyReLU(0.2))
        
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(gin_mlp, edge_dim=hidden_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2), # <-- Swapped to LeakyReLU
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        
        # 1. Less Aggressive Sparsification (Top 5% instead of Top 1%)
        # This allows the unique patient "fingerprint" edges to survive
        threshold = torch.quantile(edge_attr, 0.99)
        mask = edge_attr >= threshold
        
        # Calculate degree based on ALL edges, providing a rich initial node identity
        w_deg = torch.zeros((data.num_nodes, 1), device=edge_index.device)
        w_deg.index_add_(0, edge_index[1], edge_attr.view(-1, 1) if edge_attr.dim()==1 else edge_attr)
        x = self.node_init(w_deg)

        edge_emb = self.edge_init(edge_attr[mask].view(-1, 1))
        
        # 2. Message passing (Single hop to avoid over-smoothing)
        x = x + self.conv(x, edge_index[:, mask], edge_attr=edge_emb)
        
        z = global_max_pool(x, batch)
        return F.normalize(self.projection(z), p=2, dim=1)

class SPECTEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, embed_dim=1024, num_nodes=7): # <-- FIXED: 7 nodes for SPECT
        super().__init__()
        # 1. Feature Scaling (Amplifies small clinical differences in striatal regions)
        self.roi_scaler = nn.Parameter(torch.ones(num_nodes, input_dim))
        
        # 2. Increased Capacity Graph Convolutions
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 3. High-Variance Projection Head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2), # Swapped to LeakyReLU to prevent dead neurons
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_graphs = batch.max().item() + 1
        
        # 1. Apply ROI specific scaling
        scaler = self.roi_scaler.repeat(num_graphs, 1)
        x = x * scaler
        
        # 2. Deeper Message Passing
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        
        # 3. Global Max Pooling to isolate the most severe dopaminergic deficits
        z = global_max_pool(x, batch)
        
        return F.normalize(self.projection(z), p=2, dim=1)