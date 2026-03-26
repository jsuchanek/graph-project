import os
import glob
import random
import argparse
import datetime
import atexit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops
from multiprocessing import Pool, cpu_count
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_recall_fscore_support, 
    matthews_corrcoef
)

# ============================================================
# Config & Logging
# ============================================================
PRECOMPUTED_ROOT = "precomputed"
OUTPUT_DIR = "results_gnn_5layer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 67
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 1  # County-level batches
HIDDEN_DIM = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = os.path.join(OUTPUT_DIR, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_fh = open(LOG_PATH, "a", buffering=1)

def log_msg(msg):
    print(msg, flush=True)
    _log_fh.write(str(msg) + "\n")

atexit.register(lambda: _log_fh.close())

HIGHWAY_TYPES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "service", "unclassified", "road", "construction", "other",
]
EDGE_FEATURES = ["log_length", "edge_bc", "deg_u", "deg_v"] + [f"highway_{h}" for h in HIGHWAY_TYPES]
EDGE_DIM, NODE_DIM = len(EDGE_FEATURES), 1

# ============================================================
# Models (5 Layers)
# ============================================================

class EdgeGCNLayer(MessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        orig_e = edge_attr.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_index.size(1) > orig_e:
            pad = torch.zeros((edge_index.size(1) - orig_e, edge_attr.size(1)), device=x.device)
            edge_attr = torch.cat([edge_attr, pad], dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.msg(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.upd(torch.cat([x, aggr_out], dim=1))

class EdgeGCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([EdgeGCNLayer(node_dim if i==0 else hidden_dim, edge_dim, hidden_dim) for i in range(5)])

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < 4: x = F.relu(x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim if i==0 else hidden_dim, hidden_dim) for i in range(5)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < 4: x = F.relu(x)
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.mids = nn.ModuleList([GATConv(hidden_dim*heads, hidden_dim, heads=heads) for _ in range(3)])
        self.conv_final = GATConv(hidden_dim*heads, hidden_dim, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.mids: x = F.elu(conv(x, edge_index))
        return self.conv_final(x, edge_index)

class EdgeDecoder(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, z, edge_index, edge_attr):
        src, dst = edge_index
        return self.mlp(torch.cat([z[src], z[dst], edge_attr], dim=1)).squeeze()

# ============================================================
# Logic
# ============================================================

def get_metrics(probs, labels, k_frac=0.01):
    roc = roc_auc_score(labels, probs)
    pr = average_precision_score(labels, probs)
    k = max(1, int(k_frac * len(probs)))
    idx = np.argsort(-probs)[:k]
    p_k = labels[idx].mean()
    r_k = labels[idx].sum() / max(labels.sum(), 1)
    lift = 0.0 if labels.mean() == 0 else p_k / labels.mean()
    preds = (probs >= 0.5).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    return roc, pr, p_k, r_k, lift, f1, mcc

def evaluate_set(gnn, decoder, graphs):
    gnn.eval(); decoder.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for data in graphs:
            data = data.to(DEVICE)
            try: z = gnn(data.x, data.edge_index, data.edge_attr)
            except TypeError: z = gnn(data.x, data.edge_index)
            all_logits.append(decoder(z, data.edge_index, data.edge_attr).cpu())
            all_labels.append(data.y.cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = torch.cat(all_labels).numpy()
    return get_metrics(probs, labels)

def load_county_graph(path):
    df = pd.read_parquet(path)
    node_ids = pd.Index(pd.concat([df.u, df.v]).unique())
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    src, dst = df.u.map(node_map).to_numpy(), df.v.map(node_map).to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_attr = torch.tensor(df[EDGE_FEATURES].values, dtype=torch.float)
    y = torch.tensor(df.has_camera.values, dtype=torch.float)
    x = torch.zeros((len(node_ids), NODE_DIM), dtype=torch.float)
    x[df.u.map(node_map).to_numpy(), 0] = torch.tensor(df.deg_u.values)
    x[df.v.map(node_map).to_numpy(), 0] = torch.tensor(df.deg_v.values)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def train_model(name, gnn, train_graphs, test_graphs):
    log_msg(f"\n=== Training {name} ===")
    gnn, decoder = gnn.to(DEVICE), EdgeDecoder(HIDDEN_DIM, EDGE_DIM).to(DEVICE)
    opt = optim.Adam(list(gnn.parameters()) + list(decoder.parameters()), lr=LR)
    
    pos_ct = sum(g.y.sum().item() for g in train_graphs)
    neg_ct = sum((g.y == 0).sum().item() for g in train_graphs)
    pos_weight = neg_ct / max(pos_ct, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

    for epoch in range(1, EPOCHS + 1):
        gnn.train(); decoder.train()
        for data in train_graphs:
            data = data.to(DEVICE)
            opt.zero_grad()
            try: z = gnn(data.x, data.edge_index, data.edge_attr)
            except TypeError: z = gnn(data.x, data.edge_index)
            loss = criterion(decoder(z, data.edge_index, data.edge_attr), data.y)
            loss.backward(); opt.step()

        tr = evaluate_set(gnn, decoder, train_graphs)
        ts = evaluate_set(gnn, decoder, test_graphs)
        log_msg(f"[{name}] Ep {epoch:02d} | TRAIN: ROC {tr[0]:.4f} PR {tr[1]:.4f} | TEST: ROC {ts[0]:.4f} PR {ts[1]:.4f} MCC {ts[6]:.3f}")
        torch.save({'gnn': gnn.state_dict(), 'decoder': decoder.state_dict()}, os.path.join(OUTPUT_DIR, f"{name}_epoch_{epoch}.pt"))

# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run 5-layer GNN models on ALPR data.")
    parser.add_argument('--models', nargs='+', choices=['edgegcn', 'graphsage', 'gat', 'all'], 
                        default='all', help="List of models to run (e.g., --models gat edgegcn)")
    parser.add_argument('--workers', type=int, default=16, help="Multiprocessing workers for data loading")
    args = parser.parse_args()

    log_msg(f"Using device: {DEVICE}")

    state_paths = [(s, os.path.join(PRECOMPUTED_ROOT, s, f)) 
                   for s in sorted(os.listdir(PRECOMPUTED_ROOT)) if os.path.isdir(os.path.join(PRECOMPUTED_ROOT, s))
                   for f in os.listdir(os.path.join(PRECOMPUTED_ROOT, s)) if f.endswith('.parquet')]
    
    state_to_files = {}
    for s, p in state_paths: state_to_files.setdefault(s, []).append(p)
    all_states = sorted(state_to_files.keys())
    random.shuffle(all_states)
    split = int(0.7 * len(all_states))
    
    log_msg(f"Loading graphs with {args.workers} workers...")
    with Pool(args.workers) as pool:
        train_graphs = pool.map(load_county_graph, [f for s in all_states[:split] for f in state_to_files[s]])
        test_graphs = pool.map(load_county_graph, [f for s in all_states[split:] for f in state_to_files[s]])

    log_msg(f"Loaded {len(train_graphs)} train counties and {len(test_graphs)} test counties.")

    models_to_run = ['edgegcn', 'graphsage', 'gat'] if 'all' in args.models else args.models

    if 'edgegcn' in models_to_run:
        train_model("edgegcn", EdgeGCN(NODE_DIM, EDGE_DIM, HIDDEN_DIM), train_graphs, test_graphs)
    
    if 'graphsage' in models_to_run:
        train_model("graphsage", GraphSAGE(NODE_DIM, HIDDEN_DIM), train_graphs, test_graphs)
    
    if 'gat' in models_to_run:
        train_model("gat", GAT(NODE_DIM, HIDDEN_DIM), train_graphs, test_graphs)

if __name__ == "__main__":
    main()