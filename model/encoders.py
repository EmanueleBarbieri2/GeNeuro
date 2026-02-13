import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, GCNConv, GlobalAttention

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class BaseBrainEncoder(nn.Module):
    def __init__(self, input_dim, num_nodes=113, hidden_dim=128, embed_dim=1024):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Normalize Scale: Balances XYZ (~30.0) vs Weights (~0.05)
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # 2. Identity & Location Encoder
        self.node_init = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 3. Multiplicative Region Identity
        self.roi_scaler = nn.Parameter(torch.ones(num_nodes, hidden_dim))

        # 4. Edge Standardizer
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # 5. GINE Backbone
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(gin_mlp, train_eps=True)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2),
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward_brain(self, x_combined, edge_index, edge_attr, batch, mask):
        # Normalize inputs so edges aren't ignored
        x = self.input_norm(x_combined)
        x = self.node_init(x)
        
        # Apply learned regional weights
        batch_size = batch.max().item() + 1
        x = x * self.roi_scaler.repeat(batch_size, 1)

        # Message Passing on Sparsified Graph
        edge_emb = self.edge_encoder(edge_attr[mask].view(-1, 1))
        x = x + self.conv(x, edge_index[:, mask], edge_attr=edge_emb)
        
        # Aggregate and Project
        z = global_max_pool(x, batch)
        return F.normalize(self.projection(z), p=2, dim=1)

class fMRIEncoder(BaseBrainEncoder):
    def __init__(self, hidden_dim=128, embed_dim=1024):
        # We increase input_dim to 6: [X, Y, Z, PosStrength, NegStrength, NetBalance]
        super().__init__(input_dim=6, hidden_dim=hidden_dim, embed_dim=embed_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. Separate the Signals
        pos_edges = edge_attr.clamp(min=0)
        neg_edges = edge_attr.clamp(max=0).abs() # Magnitude of inhibition
        
        w_pos = torch.zeros((data.num_nodes, 1), device=x.device)
        w_neg = torch.zeros((data.num_nodes, 1), device=x.device)
        
        # Calculate Nodal "Excitation" and "Inhibition"
        w_pos.index_add_(0, edge_index[1], pos_edges.view(-1, 1))
        w_neg.index_add_(0, edge_index[1], neg_edges.view(-1, 1))
        
        # 2. Calculate Network Balance (EI Balance)
        # This tells the model if a region is currently "integrated" or "segregated"
        ei_balance = (w_pos - w_neg) / (w_pos + w_neg + 1e-6)
        
        # Fuse: [X, Y, Z, Pos, Neg, Balance] -> 6 dims
        x_combined = torch.cat([
            x, 
            torch.log1p(w_pos), 
            torch.log1p(w_neg), 
            ei_balance
        ], dim=1)
        
        # 3. Sparsification (Balanced)
        # We keep the strongest 20% of edges, preserving the "skeleton" of the network
        threshold = torch.quantile(edge_attr.abs(), 0.80) 
        mask = edge_attr.abs() >= threshold
        
        return self.forward_brain(x_combined, edge_index, edge_attr, batch, mask)

class DTIEncoder(BaseBrainEncoder):
    def __init__(self, hidden_dim=128, embed_dim=1024):
        super().__init__(input_dim=4, hidden_dim=hidden_dim, embed_dim=embed_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Engineer Features: Strength + XYZ
        w_deg = torch.zeros((data.num_nodes, 1), device=x.device)
        w_deg.index_add_(0, edge_index[1], edge_attr.view(-1, 1))
        
        x_combined = torch.cat([x, w_deg], dim=1)
        
        # Sparsification: Relaxed to 80% to maintain Connectivity
        threshold = torch.quantile(edge_attr, 0.80)
        mask = edge_attr >= threshold
        
        return self.forward_brain(x_combined, edge_index, edge_attr, batch, mask)
    
    
class MRIEncoder(nn.Module):
    def __init__(self, input_dim=3, num_nodes=113, hidden_dim=128, embed_dim=1024):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Region Identity (Multiplicative)
        # Learns to weight features per ROI (e.g., "Hippocampus Volume matters 2x")
        self.roi_scaler = nn.Parameter(torch.ones(num_nodes, input_dim))
        
        # 2. Backbone
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 3. UPGRADE: Attention Pooling
        # Learns which ROIs drive the PD diagnosis
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.pool = GlobalAttention(gate_nn=self.pool_gate)
        
        # 4. Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2),
            nn.Linear(512, embed_dim)
        )
        self.apply(init_weights)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_graphs = batch.max().item() + 1
        
        # --- 1. Apply Identity (ROI Scaling) ---
        # Reshape scaler to match batch: [Batch*Nodes, Features]
        # We repeat the scaler for each graph in the batch
        scaler = self.roi_scaler.repeat(num_graphs, 1)
        x = x * scaler
        
        # --- 2. Message Passing ---
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        
        # --- 3. Attention Pooling ---
        z = self.pool(x, batch)
        
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