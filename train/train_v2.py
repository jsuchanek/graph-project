# ============================================================
# Edge-Level GNN Training on Precomputed ALPR Parquet Files
# Models: EdgeGCN, GraphSAGE, GAT
# ============================================================

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import atexit

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# ============================================================
# Config
# ============================================================

PRECOMPUTED_ROOT = "precomputed"
OUTPUT_DIR = "results_gnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 67
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 1    # one county per batch
HIDDEN_DIM = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Open a line-buffered log file immediately so early messages are captured
LOG_PATH = os.path.join(OUTPUT_DIR, f"train_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
_log_fh = open(LOG_PATH, "a", buffering=1)

def _close_log():
    try:
        _log_fh.flush()
        _log_fh.close()
    except Exception:
        pass

atexit.register(_close_log)

def log_msg(msg, flush=True):
    # print to stdout + append to log file (kept simple and robust)
    try:
        print(msg, flush=flush)
    except Exception:
        pass
    try:
        _log_fh.write(str(msg) + "\n")
        if flush:
            _log_fh.flush()
    except Exception:
        pass

log_msg(f"Using device: {DEVICE}")

# If CUDA is available, touch the device early so GPU-util monitors see activity.
if DEVICE.type == "cuda":
    try:
        torch.zeros(1, device=DEVICE)
        log_msg("Touched CUDA device successfully (warmup).")
    except Exception as e:
        log_msg(f"Warning: failed to touch CUDA device: {e}")

# ============================================================
# Feature columns (must match parquet)
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

EDGE_DIM = len(EDGE_FEATURES)
NODE_DIM = 1   # degree proxy only

# ============================================================
# Dataset loading
# ============================================================
from multiprocessing import Pool, cpu_count

def load_graphs_parallel(paths, workers=None):
    workers = workers or min(cpu_count(), 16)
    log_msg(f"Loading {len(paths)} graphs using {workers} workers")
    with Pool(workers) as pool:
        graphs = list(pool.imap(load_county_graph, paths, chunksize=1))
    return graphs



def load_county_graph(path):
    log_msg(f"Loading graph from: {path}")
    df = pd.read_parquet(path)

    node_ids = pd.Index(pd.concat([df.u, df.v]).unique())
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    src = df.u.map(node_map).to_numpy()
    dst = df.v.map(node_map).to_numpy()

    # stack into a single ndarray first to avoid slow tensor creation from a
    # list of numpy arrays (avoids PyTorch user warning)
    #edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_index = torch.from_numpy(
        np.stack([src, dst], axis=0)
    ).long()
    edge_attr = torch.tensor(df[EDGE_FEATURES].values, dtype=torch.float)
    y = torch.tensor(df.has_camera.values, dtype=torch.float)

    x = torch.zeros((len(node_ids), NODE_DIM), dtype=torch.float)
    u_idx = df.u.map(node_map).to_numpy()
    v_idx = df.v.map(node_map).to_numpy()

    x[u_idx, 0] = torch.tensor(df.deg_u.values)
    x[v_idx, 0] = torch.tensor(df.deg_v.values)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

def load_all_graphs():
    graphs = []
    for p in glob.glob(f"{PRECOMPUTED_ROOT}/**/*.parquet", recursive=True):
        graphs.append(load_county_graph(p))
    log_msg(f"Loaded {len(graphs)} county graphs")
    return graphs


def load_graphs_by_state():
    state_graphs = {}
    for state in sorted(os.listdir(PRECOMPUTED_ROOT)):
        log_msg(f"Loading state: {state}")
        state_dir = os.path.join(PRECOMPUTED_ROOT, state)
        if not os.path.isdir(state_dir):
            continue

        files = glob.glob(os.path.join(state_dir, "*.parquet"))
        if not files:
            continue

        if not files:
            continue

        graphs = load_graphs_parallel(files)
        state_graphs[state] = graphs

    log_msg(f"Loaded graphs for {len(state_graphs)} states")
    return state_graphs

# ============================================================
# Models
# ============================================================

class EdgeGCN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.msg = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.upd = nn.Linear(node_dim + hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # add self-loops to edge_index and pad edge_attr with zeros for those loops
        orig_e = edge_attr.size(0) if edge_attr is not None else 0
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        new_e = edge_index.size(1)
        if edge_attr is None:
            edge_attr = torch.zeros((new_e, self.edge_dim), dtype=x.dtype, device=x.device)
        elif new_e > orig_e:
            pad = torch.zeros((new_e - orig_e, edge_attr.size(1)), dtype=edge_attr.dtype, device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, pad], dim=0)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.msg(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.upd(torch.cat([x, aggr_out], dim=1))

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z, edge_index, edge_attr):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst], edge_attr], dim=1)
        return self.mlp(h).squeeze()

# ============================================================
# Metrics
# ============================================================

def precision_at_k(probs, labels, k_frac):
    k = max(1, int(k_frac * len(probs)))
    idx = np.argsort(-probs)[:k]
    return labels[idx].mean()

def recall_at_k(probs, labels, k_frac):
    k = max(1, int(k_frac * len(probs)))
    idx = np.argsort(-probs)[:k]
    return labels[idx].sum() / max(labels.sum(), 1)

def lift_at_k(probs, labels, k_frac):
    base = labels.mean()
    return 0.0 if base == 0 else precision_at_k(probs, labels, k_frac) / base

def threshold_metrics(probs, labels, t=0.5):
    preds = (probs >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    bal = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return p, r, f, bal, mcc

def compute_pos_weight(graphs):
    pos = sum(g.y.sum().item() for g in graphs)
    neg = sum((g.y == 0).sum().item() for g in graphs)
    return neg / max(pos, 1)

# ============================================================
# Training
# ============================================================

def train_model(name, gnn, train_graphs, eval_graphs):
    log_msg(f"\n=== Training {name} ===")

    gnn = gnn.to(DEVICE)
    decoder = EdgeDecoder(HIDDEN_DIM, EDGE_DIM).to(DEVICE)

    opt = optim.Adam(
        list(gnn.parameters()) + list(decoder.parameters()),
        lr=LR
    )

    pos_weight = compute_pos_weight(train_graphs)
    log_msg(f"Computed pos_weight: {pos_weight:.4f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )

    metrics_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(
            "epoch,roc_auc,pr_auc,"
            "p@1%,r@1%,lift@1%,"
            "p@5%,r@5%,lift@5%,"
            "precision,recall,f1,bal_acc,mcc\n"
        )

        for epoch in range(1, EPOCHS + 1):
            gnn.train()
            decoder.train()

            for data in train_graphs:
                data = data.to(DEVICE)
                opt.zero_grad()
                # call gnn with edge_attr when supported (EdgeGCN needs it)
                try:
                    z = gnn(data.x, data.edge_index, data.edge_attr)
                except TypeError:
                    z = gnn(data.x, data.edge_index)
                logits = decoder(z, data.edge_index, data.edge_attr)
                loss = criterion(logits, data.y)
                loss.backward()
                opt.step()

            # ---- Eval ----
            gnn.eval()
            decoder.eval()

            logits_all, labels_all = [], []
            with torch.no_grad():
                for data in eval_graphs:
                    data = data.to(DEVICE)
                    try:
                        z = gnn(data.x, data.edge_index, data.edge_attr)
                    except TypeError:
                        z = gnn(data.x, data.edge_index)
                    logits_all.append(decoder(z, data.edge_index, data.edge_attr).cpu())
                    labels_all.append(data.y.cpu())

            if len(logits_all) == 0:
                log_msg("No evaluation graphs available; skipping eval.")
                continue

            logits_all = torch.cat(logits_all)
            labels_all = torch.cat(labels_all)

            probs = torch.sigmoid(logits_all).numpy()
            labels = labels_all.numpy()

            roc_auc = roc_auc_score(labels, probs)
            pr_auc = average_precision_score(labels, probs)

            p1, r1, l1 = (
                precision_at_k(probs, labels, 0.01),
                recall_at_k(probs, labels, 0.01),
                lift_at_k(probs, labels, 0.01),
            )

            p5, r5, l5 = (
                precision_at_k(probs, labels, 0.05),
                recall_at_k(probs, labels, 0.05),
                lift_at_k(probs, labels, 0.05),
            )

            p, r, f1, bal, mcc = threshold_metrics(probs, labels)

            log = (
                f"[{name}] Epoch {epoch:02d} | "
                f"ROC {roc_auc:.4f} PR {pr_auc:.4f} | "
                f"P@1% {p1:.3f} R@1% {r1:.3f} L@1% {l1:.2f} | "
                f"F1 {f1:.3f} MCC {mcc:.3f}"
            )
            log_msg(log)

            f.write(
                f"{epoch},{roc_auc:.4f},{pr_auc:.4f},"
                f"{p1:.3f},{r1:.3f},{l1:.2f},"
                f"{p5:.3f},{r5:.3f},{l5:.2f},"
                f"{p:.3f},{r:.3f},{f1:.3f},{bal:.3f},{mcc:.3f}\n"
            )
            f.flush()

            torch.save(
                {
                    "gnn": gnn.state_dict(),
                    "decoder": decoder.state_dict(),
                },
                os.path.join(OUTPUT_DIR, f"{name}_epoch_{epoch}.pt")
            )

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Load graphs grouped by state, then split states 70/30 for train/eval
    state_graphs = load_graphs_by_state()

    all_states = sorted(state_graphs.keys())
    random.shuffle(all_states)
    split_idx = int(0.7 * len(all_states))

    train_states = all_states[:split_idx]
    test_states = all_states[split_idx:]

    log_msg(f"Train states: {train_states}")
    log_msg(f"Test states: {test_states}")

    train_graphs = []
    for s in train_states:
        train_graphs.extend(state_graphs[s])

    test_graphs = []
    for s in test_states:
        test_graphs.extend(state_graphs[s])

    # train_model(
    #     "edgegcn",
    #     EdgeGCN(NODE_DIM, EDGE_DIM, HIDDEN_DIM),
    #     train_graphs,
    #     test_graphs,
    # )

    # train_model(
    #     "graphsage",
    #     GraphSAGE(NODE_DIM, HIDDEN_DIM),
    #     train_graphs,
    #     test_graphs,
    # )

    train_model(
        "gat",
        GAT(NODE_DIM, HIDDEN_DIM),
        train_graphs,
        test_graphs,
    )

    log_msg("\nAll GNN models trained and evaluated.")
