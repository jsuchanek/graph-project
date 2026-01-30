import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score

# =========================
# Dataset
# =========================

class ALPRParquetDataset(InMemoryDataset):
    def __init__(self, parquet_files):
        super().__init__()
        data_list = []
        for path in parquet_files:
            data = self.load_county(path)
            if data is not None:
                data_list.append(data)
        self.data, self.slices = self.collate(data_list)

    def load_county(self, path):
        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            return None

        if len(gdf) == 0:
            return None

        # --- Graph ---
        G = nx.from_pandas_edgelist(gdf, "u", "v")
        node_map = {n: i for i, n in enumerate(G.nodes())}

        # --- Node features (degree only) ---
        deg_u = gdf["deg_u"].values
        deg_v = gdf["deg_v"].values

        num_nodes = len(node_map)
        x = torch.zeros((num_nodes, 1), dtype=torch.float)

        for u, du in zip(gdf.u, deg_u):
            x[node_map[u], 0] = float(du)


        # --- Edge index ---
        edge_index = torch.tensor(
            [[node_map[u], node_map[v]] for u, v in zip(gdf.u, gdf.v)],
            dtype=torch.long
        ).t()

        # --- Edge features ---
        highway_cols = [c for c in gdf.columns if c.startswith("highway_")]
        edge_attr = torch.tensor(
            gdf[["log_length", "edge_bc"] + highway_cols].values,
            dtype=torch.float
        )

        # --- Labels ---
        y = torch.tensor(gdf["has_camera"].values, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

# =========================
# Model
# =========================

class EdgeClassifier(nn.Module):
    def __init__(self, in_node, in_edge, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_node, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + in_edge, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        row, col = edge_index
        return self.edge_mlp(
            torch.cat([x[row], x[col], edge_attr], dim=1)
        ).squeeze()

# =========================
# Training
# =========================

def compute_pos_weight(dataset):
    pos = sum(d.y.sum().item() for d in dataset)
    neg = sum((d.y == 0).sum().item() for d in dataset)
    return neg / max(pos, 1)

def precision_at_k(probs, labels, k):
    idx = np.argsort(-probs)[:k]
    return labels[idx].mean()

def train_model(train_loader, test_loader, in_node, in_edge, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeClassifier(in_node, in_edge).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    pos_weight = compute_pos_weight(train_loader.dataset)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    print(f"Using pos_weight = {pos_weight:.2f}")

    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(data.x, data.edge_index, data.edge_attr),
                data.y
            )
            loss.backward()
            optimizer.step()

        # --- Eval ---
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                logits.append(
                    model(data.x, data.edge_index, data.edge_attr).cpu()
                )
                labels.append(data.y.cpu())

        logits = torch.cat(logits)
        labels = torch.cat(labels)

        probs = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()

        auroc = roc_auc_score(labels_np, probs)
        k = max(1, int(0.01 * len(probs)))
        p_at_k = precision_at_k(probs, labels_np, k)

        print(
            f"Epoch {epoch:02d} | "
            f"AUROC {auroc:.4f} | "
            f"Precision@1% {p_at_k:.4f}"
        )

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

    return model

# =========================
# Load precomputed data
# =========================

BASE_DIR = "precomputed"
all_parquets = []

for state in os.listdir(BASE_DIR):
    state_dir = os.path.join(BASE_DIR, state)
    if not os.path.isdir(state_dir):
        continue
    for f in os.listdir(state_dir):
        if f.endswith(".parquet"):
            all_parquets.append(os.path.join(state_dir, f))

random.shuffle(all_parquets)

split = int(0.7 * len(all_parquets))
train_files = all_parquets[:split]
test_files = all_parquets[split:]

print("\n=== TRAIN FILES ===")
for f in train_files:
    print("  ", f)

print("\n=== TEST FILES ===")
for f in test_files:
    print("  ", f)

print(f"\nTrain counties: {len(train_files)}")
print(f"Test counties:  {len(test_files)}\n")


train_dataset = ALPRParquetDataset(train_files)
test_dataset = ALPRParquetDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

in_node_feats = train_dataset[0].x.shape[1]
in_edge_feats = train_dataset[0].edge_attr.shape[1]

# =========================
# Train
# =========================

model = train_model(
    train_loader,
    test_loader,
    in_node_feats,
    in_edge_feats,
    epochs=15
)
