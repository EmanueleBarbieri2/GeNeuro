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
        # 1. Feature Scaling per ROI
        self.roi_scaler = nn.Parameter(torch.ones(num_nodes, input_dim))
        
        # 2. Independent Node Processing
        self.node_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim), # GroupNorm works better than LayerNorm for high similarity
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, affine=True, track_running_stats=False), # Force variance
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        x, batch = data.x, data.batch
        num_graphs = batch.max().item() + 1
        
        # 1. Scaling to amplify node-level differences
        scaler = self.roi_scaler.repeat(num_graphs, 1)
        x = x * scaler
        
        # 2. Process nodes without any graph averaging
        x = self.node_net(x)
        
        # 3. Use Max Pooling to isolate specific atrophy signatures
        z = global_max_pool(x, batch)
        return F.normalize(self.projection(z), p=2, dim=1)

class ConnectomeEncoder(nn.Module):
    def __init__(self, hidden_dim=64, embed_dim=1024):
        super().__init__()
        self.node_init = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.edge_init = nn.Linear(1, hidden_dim)
        
        gin_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv = GINEConv(gin_mlp, edge_dim=hidden_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        
        # 1. Aggressive Sparsification (Top 1% strongest edges)
        # This isolates the unique functional "fingerprint"
        threshold = torch.quantile(edge_attr, 0.99)
        mask = edge_attr >= threshold
        
        w_deg = torch.zeros((data.num_nodes, 1), device=edge_index.device)
        w_deg.index_add_(0, edge_index[1], edge_attr.view(-1, 1) if edge_attr.dim()==1 else edge_attr)
        x = self.node_init(w_deg)

        edge_emb = self.edge_init(edge_attr[mask].view(-1, 1))
        # 2. Message passing with limited iterations to avoid over-smoothing
        x = x + self.conv(x, edge_index[:, mask], edge_attr=edge_emb)
        
        z = global_max_pool(x, batch)
        return F.normalize(self.projection(z), p=2, dim=1)

class SPECTEncoder(nn.Module):
    # SPECT works well, keeping simple
    def __init__(self, input_dim=1, hidden_dim=32, embed_dim=1024):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        z = global_max_pool(x, batch)
        return F.normalize(self.projection(z), p=2, dim=1)