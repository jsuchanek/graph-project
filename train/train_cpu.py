"""
CPU-optimized training script.

This reuses most logic from train_v2.py but forces CPU device and
uses torch_geometric DataLoader with multiple worker processes to
parallelize data loading and preprocessing. This provides substantial
speedups on CPU-bound runs where IO and parsing dominate.

Usage:
  python train/train_cpu.py --workers 8
"""
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

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, balanced_accuracy_score, matthews_corrcoef

# Config (keep in sync with parquet schema)
PRECOMPUTED_ROOT = "precomputed"
OUTPUT_DIR = "results_gnn_cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 67
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 1
HIDDEN_DIM = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Force CPU for this script
DEVICE = torch.device("cpu")

# Feature definitions
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
NODE_DIM = 1


class EdgeGCN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.msg = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.upd = nn.Linear(node_dim + hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
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
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, z, edge_index, edge_attr):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst], edge_attr], dim=1)
        return self.mlp(h).squeeze()


def load_all_graphs_flat(root):
    graphs = []
    paths = []
    for p in glob.glob(f"{root}/**/*.parquet", recursive=True):
        paths.append(p)
    paths = sorted(paths)
    for p in paths:
        df = pd.read_parquet(p)
        node_ids = pd.Index(pd.concat([df.u, df.v]).unique())
        node_map = {nid: i for i, nid in enumerate(node_ids)}

        src = df.u.map(node_map).to_numpy()
        dst = df.v.map(node_map).to_numpy()

        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        edge_attr = torch.tensor(df[EDGE_FEATURES].values, dtype=torch.float)
        y = torch.tensor(df.has_camera.values, dtype=torch.float)

        x = torch.zeros((len(node_ids), NODE_DIM), dtype=torch.float)
        u_idx = df.u.map(node_map).to_numpy()
        v_idx = df.v.map(node_map).to_numpy()
        x[u_idx, 0] = torch.tensor(df.deg_u.values)
        x[v_idx, 0] = torch.tensor(df.deg_v.values)

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs


def load_county_graph(path):
    df = pd.read_parquet(path)

    node_ids = pd.Index(pd.concat([df.u, df.v]).unique())
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    src = df.u.map(node_map).to_numpy()
    dst = df.v.map(node_map).to_numpy()

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_attr = torch.tensor(df[EDGE_FEATURES].values, dtype=torch.float)
    y = torch.tensor(df.has_camera.values, dtype=torch.float)

    x = torch.zeros((len(node_ids), NODE_DIM), dtype=torch.float)
    u_idx = df.u.map(node_map).to_numpy()
    v_idx = df.v.map(node_map).to_numpy()

    x[u_idx, 0] = torch.tensor(df.deg_u.values)
    x[v_idx, 0] = torch.tensor(df.deg_v.values)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _load_county_pair(pair):
    """Module-level helper for multiprocessing: takes (state, path) and returns (state, Data or None)."""
    state, path = pair
    try:
        g = load_county_graph(path)
        return state, g
    except Exception:
        return state, None


def load_graphs_by_state_parallel(workers=None):
    workers = workers or min(cpu_count(), 16)
    state_paths = []  # list of (state, path)
    for state in sorted(os.listdir(PRECOMPUTED_ROOT)):
        state_dir = os.path.join(PRECOMPUTED_ROOT, state)
        if not os.path.isdir(state_dir):
            continue
        for p in glob.glob(os.path.join(state_dir, "*.parquet")):
            state_paths.append((state, p))

    # load all graphs in parallel and group by state
    state_graphs = {}
    if not state_paths:
        return state_graphs

    # Use a module-level helper so it can be pickled by multiprocessing
    def _iter_results(pairs):
        with Pool(workers) as pool:
            for state, g in pool.imap(_load_county_pair, pairs, chunksize=1):
                if g is None:
                    continue
                state_graphs.setdefault(state, []).append(g)

    _iter_results(state_paths)

    return state_graphs


def precision_at_k(probs, labels, k_frac):
    k = max(1, int(k_frac * len(probs)))
    idx = np.argsort(-probs)[:k]
    return labels[idx].mean()


def compute_pos_weight(graphs):
    pos = sum(g.y.sum().item() for g in graphs)
    neg = sum((g.y == 0).sum().item() for g in graphs)
    return neg / max(pos, 1)


def train(args):
    # control intra-op threads to avoid oversubscription
    torch.set_num_threads(max(1, args.workers))

    # Load graphs grouped by state in parallel, then split states 70/30
    state_graphs = load_graphs_by_state_parallel(workers=args.workers)
    all_states = sorted(state_graphs.keys())
    random.shuffle(all_states)
    split_idx = int(0.7 * len(all_states))
    train_states = all_states[:split_idx]
    test_states = all_states[split_idx:]

    train_graphs = []
    for s in train_states:
        train_graphs.extend(state_graphs[s])

    test_graphs = []
    for s in test_states:
        test_graphs.extend(state_graphs[s])

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)

    model = GAT(NODE_DIM, HIDDEN_DIM) if args.model == 'gat' else GraphSAGE(NODE_DIM, HIDDEN_DIM)
    decoder = EdgeDecoder(HIDDEN_DIM, EDGE_DIM)
    model.to(DEVICE); decoder.to(DEVICE)

    opt = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([compute_pos_weight(train_graphs)], device=DEVICE))

    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{args.model}.csv")
    with open(metrics_path, 'w') as mf:
        mf.write('epoch,roc_auc,pr_auc\n')
        for epoch in range(1, EPOCHS + 1):
            model.train(); decoder.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                opt.zero_grad()
                try:
                    z = model(batch.x, batch.edge_index, batch.edge_attr)
                except TypeError:
                    z = model(batch.x, batch.edge_index)
                logits = decoder(z, batch.edge_index, batch.edge_attr)
                loss = criterion(logits, batch.y)
                loss.backward()
                opt.step()

            # eval
            model.eval(); decoder.eval()
            logits_all, labels_all = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(DEVICE)
                    try:
                        z = model(batch.x, batch.edge_index, batch.edge_attr)
                    except TypeError:
                        z = model(batch.x, batch.edge_index)
                    logits_all.append(decoder(z, batch.edge_index, batch.edge_attr).cpu())
                    labels_all.append(batch.y.cpu())

            if not logits_all:
                continue
            logits_all = torch.cat(logits_all)
            labels_all = torch.cat(labels_all)
            probs = torch.sigmoid(logits_all).numpy()
            labels = labels_all.numpy()

            roc = roc_auc_score(labels, probs)
            pr = average_precision_score(labels, probs)
            mf.write(f"{epoch},{roc:.4f},{pr:.4f}\n")
            mf.flush()
            torch.save({'gnn': model.state_dict(), 'decoder': decoder.state_dict()}, os.path.join(OUTPUT_DIR, f"{args.model}_epoch_{epoch}.pt"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--model', choices=['gat', 'graphsage'], default='gat')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
