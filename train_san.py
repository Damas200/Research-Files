import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import random
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ==== Set random seed for reproducibility ====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== Load data ====
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

z1y = load_pkl("data.pkl")
z1_list = z1y['train']
z1_raw = torch.cat([t[0] for t in z1_list], dim=0)

y_list = z1y['val']
y_raw = torch.cat([t[0] for t in y_list], dim=0)

B1 = load_pkl("incidence.pkl")['B1']
B2 = load_pkl("incidence.pkl")['B2']
num_nodes = B1.shape[0]
num_edges = B1.shape[1]
num_triangles = B2.shape[1]

# Align shapes
y = y_raw[:num_edges].view(-1).long().clamp(min=0)
z1 = z1_raw[:num_edges].view(num_edges, 1)

# Initialize z0 and z2 as zero features
z0 = torch.zeros((num_nodes, 1), dtype=torch.float)
z2 = torch.zeros((num_triangles, 1), dtype=torch.float)

# Compute full Laplacians
B1 = torch.tensor(B1, dtype=torch.float)
B2 = torch.tensor(B2, dtype=torch.float)
L0 = B1 @ B1.t()  # Node Laplacian
L1_d = B1.t() @ B1  # Edge lower Laplacian
L1_u = B2 @ B2.t()  # Edge upper Laplacian
L2 = B2.t() @ B2   # Triangle Laplacian

# Apply label filter with relaxed threshold
valid_classes = torch.unique(y, return_counts=True)
valid_labels = valid_classes[0][valid_classes[1] >= 2]
mask = torch.isin(y, valid_labels)
filtered_indices = torch.nonzero(mask, as_tuple=True)[0]
filtered_indices = filtered_indices[filtered_indices < L1_d.shape[0]]
z1 = z1[filtered_indices]
y = y[filtered_indices]
L1_d = L1_d[filtered_indices][:, filtered_indices]
L1_u = L1_u[filtered_indices][:, filtered_indices]
z0 = z0  # Keep full z0 for node interactions
z2 = z2  # Keep full z2 for triangle interactions

print("Label counts:", torch.bincount(y))

# ==== GSAN Layer ====
class GSANLayer(nn.Module):
    def __init__(self, in_channels, out_channels, J=3, J_h=5, dropout=0.5):
        super().__init__()
        self.J = J
        self.J_h = J_h
        self.out_channels = out_channels
        self.W_d = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.W_u = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.W_h = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W_b1 = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.W_b2 = nn.Parameter(torch.randn(J, in_channels, out_channels))
        self.dropout = dropout
        self.attn_d = nn.Linear(in_channels * 2, 1)
        self.attn_u = nn.Linear(in_channels * 2, 1)

    def harmonic_filter(self, L, x):
        I = torch.eye(L.shape[0], device=x.device)
        eigenvalues = torch.linalg.eigvalsh(L)
        lambda_max = eigenvalues[-1].item()
        epsilon = 0.5 * (2 / lambda_max) if lambda_max > 0 else 0.1
        P = I - epsilon * L
        for _ in range(self.J_h - 1):
            P = P @ P
        return P @ x @ self.W_h

    def compute_attention(self, x, L, attn_fn):
        idx = torch.nonzero(L, as_tuple=True)
        src, dst = idx[0], idx[1]
        x_src = x[src]
        x_dst = x[dst]
        attn_input = torch.cat([x_src, x_dst], dim=-1)
        attn_scores = attn_fn(attn_input).sigmoid()
        L_attn = torch.zeros_like(L)
        L_attn[src, dst] = attn_scores.squeeze()
        L_attn[dst, src] = attn_scores.squeeze()
        return L_attn

    def forward(self, z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2):
        # Attention-weighted Laplacians
        L1_d_attn = self.compute_attention(z1, L1_d, self.attn_d)
        L1_u_attn = self.compute_attention(z1, L1_u, self.attn_u)

        # Node signal (Z_0)
        z0_out = torch.zeros(z0.size(0), self.out_channels, device=z0.device)
        for j in range(self.J):
            z0_out += (L0 @ z0) @ self.W_d[j]
            z0_out += (B1 @ z1) @ self.W_b1[j]
        z0_out += self.harmonic_filter(L0, z0)
        z0_out = F.dropout(F.relu(z0_out), p=self.dropout, training=self.training)

        # Edge signal (Z_1)
        z1_out = torch.zeros(z1.size(0), self.out_channels, device=z1.device)
        for j in range(self.J):
            z1_out += (L1_d_attn @ z1) @ self.W_d[j]
            z1_out += (L1_u_attn @ z1) @ self.W_u[j]
            z1_out += (B1.t() @ z0) @ self.W_b1[j]
            z1_out += (B2 @ z2) @ self.W_b2[j]
        z1_out += self.harmonic_filter(L1_d + L1_u, z1)
        z1_out = F.dropout(F.relu(z1_out), p=self.dropout, training=self.training)

        # Triangle signal (Z_2)
        z2_out = torch.zeros(z2.size(0), self.out_channels, device=z2.device)
        for j in range(self.J):
            z2_out += (L2 @ z2) @ self.W_u[j]
            z2_out += (B2.t() @ z1) @ self.W_b2[j]
        z2_out += self.harmonic_filter(L2, z2)
        z2_out = F.dropout(F.relu(z2_out), p=self.dropout, training=self.training)

        return z0_out, z1_out, z2_out

# ==== Classifier ====
class GSANClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.gsan = GSANLayer(in_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2):
        _, z1_out, _ = self.gsan(z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2)
        return self.mlp(z1_out)

# ==== Dataset Split ====
def prepare_dataset(z0, z1, z2, y):
    x = z1.float()
    y = y.squeeze().long()
    unique, counts = torch.unique(y, return_counts=True)
    if torch.any(counts < 2):
        raise ValueError(f"Not enough samples per class: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # First split: 70% train, 30% (validation + test)
    train_idx_temp, temp_idx = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)
    # Second split: 10% validation (1/3 of 30%), 20% test (2/3 of 30%)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.6667, stratify=y[temp_idx], random_state=42)
    train_idx = train_idx_temp
    
    return z0, z1, z2, y, train_idx, val_idx, test_idx

z0, z1, z2, y, train_idx, val_idx, test_idx = prepare_dataset(z0, z1, z2, y)

# ==== Train ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GSANClassifier(in_channels=1, hidden_channels=32, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

z0, z1, z2, y = z0.to(device), z1.to(device), z2.to(device), y.to(device)
L0, L1_d, L1_u, L2 = L0.to(device), L1_d.to(device), L1_u.to(device), L2.to(device)
B1, B2 = B1.to(device), B2.to(device)

best_train_auc = 0
best_train_loss = float('inf')
best_val_auc = 0
best_val_loss = float('inf')
best_test_auc = 0
best_test_loss = float('inf')
patience, trigger = 80, 0

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(z0, z1, z2, L0, L1_d, L1_u, L2, B1, B2)
    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        # Training metrics
        train_out = out[train_idx][:, 1].sigmoid().cpu().numpy()  # Probability of positive class
        train_loss = criterion(out[train_idx], y[train_idx]).item()
        train_auc = roc_auc_score(y[train_idx].cpu().numpy(), train_out)

        # Validation metrics
        val_out = out[val_idx][:, 1].sigmoid().cpu().numpy()  # Probability of positive class
        val_loss = criterion(out[val_idx], y[val_idx]).item()
        val_auc = roc_auc_score(y[val_idx].cpu().numpy(), val_out)

        # Test metrics
        test_out = out[test_idx][:, 1].sigmoid().cpu().numpy()  # Probability of positive class
        test_loss = criterion(out[test_idx], y[test_idx]).item()
        test_auc = roc_auc_score(y[test_idx].cpu().numpy(), test_out)

        # Update best metrics
        if train_auc > best_train_auc:
            best_train_auc = train_auc
            best_train_loss = train_loss

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_loss = test_loss
            trigger = 0
        else:
            trigger += 1

        print(f"Epoch {epoch}, Training AUC: {train_auc:.3f}, Loss: {train_loss:.3f}, "
              f"Validation AUC: {val_auc:.3f}, Loss: {val_loss:.3f}, "
              f"Test AUC: {test_auc:.3f}, Loss: {test_loss:.3f}")
        
        if trigger >= patience:
            print("Early stopping.")
            break

# Print best metrics
print(f"Best Test AUC: {best_test_auc:.3f}, Loss: {best_test_loss:.3f}")
print(f"Best Training AUC: {best_train_auc:.3f}, Loss: {best_train_loss:.3f}")
print(f"Best Validation AUC: {best_val_auc:.3f}, Loss: {best_val_loss:.3f}")