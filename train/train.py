# ============================================================
# Edge-Only Training on Precomputed ALPR Parquet Files
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# ============================================================
# Config
# ============================================================

PRECOMPUTED_ROOT = "precomputed"
SEED = 67
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 1   # each Data = one county
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.txt")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Feature columns (MUST match parquet exactly)
# ============================================================

HIGHWAY_TYPES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "service", "unclassified", "road", "construction", "other",
]

EDGE_FEATURES = (
    ["log_length", "edge_bc", "deg_u", "deg_v"] +
    [f"highway_{h}" for h in HIGHWAY_TYPES]
)

EDGE_FEAT_DIM = len(EDGE_FEATURES)

# ============================================================
# Dataset (edge-only, no graph)
# ============================================================

class ALPREdgeDataset(InMemoryDataset):
    def __init__(self, root, states):
        super().__init__()
        self.data_list = []
        self.load_states(root, states)
        self.data, self.slices = self.collate(self.data_list)

    def load_states(self, root, states):
        for state in states:
            print(f"Loading state: {state}")
            state_dir = os.path.join(root, state)
            if not os.path.isdir(state_dir):
                continue

            for fname in os.listdir(state_dir):
                if not fname.endswith(".parquet"):
                    continue

                df = pd.read_parquet(os.path.join(state_dir, fname))

                edge_attr = torch.tensor(
                    df[EDGE_FEATURES].values,
                    dtype=torch.float,
                )

                y = torch.tensor(
                    df["has_camera"].values,
                    dtype=torch.float,
                )

                self.data_list.append(
                    Data(edge_attr=edge_attr, y=y)
                )

# ============================================================
# Train / Test split (by state)
# ============================================================

all_states = sorted([
    d for d in os.listdir(PRECOMPUTED_ROOT)
    if os.path.isdir(os.path.join(PRECOMPUTED_ROOT, d))
])

random.shuffle(all_states)
split_idx = int(0.7 * len(all_states))

train_states = all_states[:split_idx]
test_states = all_states[split_idx:]

print("Train states:", train_states)
print("Test states:", test_states)

train_dataset = ALPREdgeDataset(PRECOMPUTED_ROOT, train_states)
test_dataset = ALPREdgeDataset(PRECOMPUTED_ROOT, test_states)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# Model (Edge MLP)
# ============================================================

class EdgeMLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, edge_attr):
        return self.net(edge_attr).squeeze()

# ============================================================
# Metrics
# ============================================================

def precision_at_k(probs, labels, k_frac):
    k = max(1, int(k_frac * len(probs)))
    idx = np.argsort(-probs)[:k]
    return labels[idx].mean()

def prf_at_threshold(probs, labels, thresh=0.5):
    preds = (probs >= thresh).astype(np.int32)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return p, r, f

def compute_pos_weight(dataset):
    pos = sum(d.y.sum().item() for d in dataset)
    neg = sum((d.y == 0).sum().item() for d in dataset)
    return neg / max(pos, 1)

# ============================================================
# Training Loop
# ============================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = EdgeMLP(EDGE_FEAT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pos_weight = compute_pos_weight(train_dataset)
    print(f"Using pos_weight = {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    # Open metrics file
    with open(METRICS_FILE, "w") as f_metrics:

        f_metrics.write("epoch,auroc,p@1%,p@5%,precision,recall,f1\n")

        for epoch in range(1, EPOCHS + 1):
            # -------- Train --------
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                logits = model(data.edge_attr)
                loss = criterion(logits, data.y)
                loss.backward()
                optimizer.step()

            # -------- Eval --------
            model.eval()
            logits_all, labels_all = [], []

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    logits_all.append(model(data.edge_attr).cpu())
                    labels_all.append(data.y.cpu())

            logits_all = torch.cat(logits_all)
            labels_all = torch.cat(labels_all)

            probs = torch.sigmoid(logits_all).numpy()
            labels = labels_all.numpy()

            auroc = roc_auc_score(labels, probs)
            p1 = precision_at_k(probs, labels, 0.01)
            p5 = precision_at_k(probs, labels, 0.05)
            p, r, f = prf_at_threshold(probs, labels)

            # -------- Print --------
            log_str = (
                f"Epoch {epoch:02d} | "
                f"AUROC {auroc:.4f} | "
                f"P@1% {p1:.3f} | P@5% {p5:.3f} | "
                f"P {p:.3f} R {r:.3f} F1 {f:.3f}"
            )
            print(log_str)

            # -------- Write to file --------
            f_metrics.write(f"{epoch},{auroc:.4f},{p1:.3f},{p5:.3f},{p:.3f},{r:.3f},{f:.3f}\n")
            f_metrics.flush()

            # -------- Save model --------
            model_file = os.path.join(OUTPUT_DIR, f"edge_mlp_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_file)

    print(f"Training complete. Metrics saved to {METRICS_FILE}")
    return model

# ============================================================
# Run
# ============================================================

model = train()
