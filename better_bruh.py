import os
import re
import random
import math
import numpy as np
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from shapely.geometry import Polygon
from pyrosm import OSM

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
)

# =========================
# Constants / Lookups
# =========================

R = 6378137

STATE_NAME_TO_FIPS = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
    "california": "06", "colorado": "08", "connecticut": "09",
    "delaware": "10", "district-of-columbia": "11", "florida": "12",
    "georgia": "13", "hawaii": "15", "idaho": "16", "illinois": "17",
    "indiana": "18", "iowa": "19", "kansas": "20", "kentucky": "21",
    "louisiana": "22", "maine": "23", "maryland": "24",
    "massachusetts": "25", "michigan": "26", "minnesota": "27",
    "mississippi": "28", "missouri": "29", "montana": "30",
    "nebraska": "31", "nevada": "32", "new-hampshire": "33",
    "new-jersey": "34", "new-mexico": "35", "new-york": "36",
    "north-carolina": "37", "north-dakota": "38", "ohio": "39",
    "oklahoma": "40", "oregon": "41", "pennsylvania": "42",
    "rhode-island": "44", "south-carolina": "45", "south-dakota": "46",
    "tennessee": "47", "texas": "48", "utah": "49", "vermont": "50",
    "virginia": "51", "washington": "53", "west-virginia": "54",
    "wisconsin": "55", "wyoming": "56",
}

HIGHWAY_TYPES = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "living_street",
    "service", "unclassified",
    "road", "construction",
    "other",
]

HIGHWAY_TO_IDX = {h: i for i, h in enumerate(HIGHWAY_TYPES)}
NUM_HIGHWAY_TYPES = len(HIGHWAY_TYPES)
EDGE_FEAT_DIM = NUM_HIGHWAY_TYPES + 1  # + log(length)

# =========================
# Helpers
# =========================

def encode_highway(highway):
    vec = np.zeros(NUM_HIGHWAY_TYPES, dtype=np.float32)
    if isinstance(highway, list):
        highway = highway[0]
    idx = HIGHWAY_TO_IDX.get(highway, HIGHWAY_TO_IDX["other"])
    vec[idx] = 1.0
    return vec


def get_state_counties(pbf_file, county_gdf):
    match = re.match(r"([a-z\-]+)-\d+\.osm\.pbf", pbf_file)
    state_fips = STATE_NAME_TO_FIPS[match.group(1)]
    counties = county_gdf[county_gdf["STATE"] == state_fips]
    print(f"  → {len(counties)} counties found")
    return counties


def build_camera_wedge(lat, lon, bearing, fov=45, dist=20):
    def step(b):
        b = math.radians(b)
        lat2 = math.asin(
            math.sin(math.radians(lat)) * math.cos(dist / R)
            + math.cos(math.radians(lat)) * math.sin(dist / R) * math.cos(b)
        )
        lon2 = math.radians(lon) + math.atan2(
            math.sin(b) * math.sin(dist / R) * math.cos(math.radians(lat)),
            math.cos(dist / R) - math.sin(math.radians(lat)) * math.sin(lat2),
        )
        return math.degrees(lat2), math.degrees(lon2)

    l = step(bearing - fov / 2)
    r = step(bearing + fov / 2)
    return Polygon([(lon, lat), (l[1], l[0]), (r[1], r[0])])


def precision_at_k(probs, labels, k):
    idx = np.argsort(-probs)[:k]
    return labels[idx].mean()


# =========================
# Dataset
# =========================

class ALPRDataset(InMemoryDataset):
    def __init__(self, pbf_files, county_gdf, alpr_gdf):
        super().__init__()
        self.data_list = []
        for pbf in pbf_files:
            self.process_pbf(pbf, county_gdf, alpr_gdf)
        self.data, self.slices = self.collate(self.data_list)

    def process_pbf(self, pbf_file, county_gdf, alpr_gdf):
        print(f"\nProcessing {pbf_file}")
        counties = get_state_counties(pbf_file, county_gdf)

        for _, county in counties.iterrows():
            try:
                osm = OSM(f"osm_pbf/{pbf_file}", bounding_box=county.geometry)
                nodes, edges = osm.get_network("driving", nodes=True)
            except Exception:
                continue

            if edges is None or len(edges) == 0:
                continue

            print(f"  County {county['NAME']} - {len(edges)} edges")

            # ---------- Edge features ----------
            edge_feats = []
            for _, r in edges.iterrows():
                hwy = encode_highway(r.get("highway"))
                length = r.get("length", 0.0)
                length = math.log1p(float(length)) if length else 0.0
                edge_feats.append(np.concatenate([hwy, [length]]))

            edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float)

            # ---------- Camera labels ----------
            edges["has_camera"] = 0
            cams = alpr_gdf[alpr_gdf.within(county.geometry)]
            sindex = edges.sindex

            for _, cam in cams.iterrows():
                if cam.get("direction") in [None, "N/A"]:
                    continue

                raw_dir = str(cam["direction"])
                first_part = re.split(r"[;-]", raw_dir)[0]

                try:
                    b = float(first_part)
                except ValueError:
                    continue

                wedge = build_camera_wedge(cam.geometry.y, cam.geometry.x, b)

                candidate_idx = list(sindex.intersection(wedge.bounds))
                if not candidate_idx:
                    continue

                hits = edges.iloc[candidate_idx].geometry.intersects(wedge)
                edges.loc[edges.iloc[candidate_idx][hits].index, "has_camera"] = 1


            y = torch.tensor(edges["has_camera"].values, dtype=torch.float)

            # ---------- Graph ----------
            G = nx.from_pandas_edgelist(edges, "u", "v")
            node_map = {n: i for i, n in enumerate(G.nodes())}
            degrees = np.array([G.degree[n] for n in G.nodes()], dtype=np.float32)
            degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
            x = torch.tensor(degrees[:, None], dtype=torch.float)

            edge_index = torch.tensor(
                [[node_map[u], node_map[v]] for u, v in zip(edges.u, edges.v)],
                dtype=torch.long,
            ).t()

            self.data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )


# =========================
# Model
# =========================

class EdgeClassifier(nn.Module):
    def __init__(self, in_node, in_edge, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_node, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + in_edge, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        row, col = edge_index
        return self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=1)).squeeze()


# =========================
# Training
# =========================

def compute_pos_weight(dataset):
    pos = sum(d.y.sum().item() for d in dataset)
    neg = sum((d.y == 0).sum().item() for d in dataset)
    return neg / max(pos, 1)


def train_model(train_loader, test_loader, in_node, in_edge, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeClassifier(in_node, in_edge).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    pos_weight = compute_pos_weight(train_loader.dataset)
    print(f"Using pos_weight = {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data.x, data.edge_index, data.edge_attr), data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                logits.append(model(data.x, data.edge_index, data.edge_attr).cpu())
                labels.append(data.y.cpu())

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        probs = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()

        auroc = roc_auc_score(labels_np, probs)
        k = max(1, int(0.01 * len(probs)))
        p_at_k = precision_at_k(probs, labels_np, k)

        print(
            f"Epoch {epoch:02d} | AUROC {auroc:.4f} | Precision@1% {p_at_k:.4f}"
        )
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

    return model



gdf = gpd.read_file("county/gz_2010_us_050_00_5m.json", encoding="latin1").to_crs(epsg=4326)
alpr = gpd.read_file("alpr/export.geojson").to_crs(epsg=4326)

PBF_DIR = "osm_pbf"
pbf_files = [f for f in os.listdir(PBF_DIR) if f.endswith(".osm.pbf")]

pbf_sizes = {f: os.path.getsize(os.path.join(PBF_DIR, f)) / (1024 * 1024) for f in pbf_files}
smallest_files = sorted(pbf_sizes, key=pbf_sizes.get)[:3]

random.seed(42)
random.shuffle(smallest_files)

total_size = sum(pbf_sizes[f] for f in smallest_files)
train_files, test_files, acc = [], [], 0

for f in smallest_files:
    if acc / total_size < 0.4:
        train_files.append(f)
        acc += pbf_sizes[f]
    else:
        test_files.append(f)

print("Train:", train_files)
print("Test:", test_files)

train_dataset = ALPRDataset(train_files, gdf, alpr)
test_dataset = ALPRDataset(test_files, gdf, alpr)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Node features come from data
in_node_feats = train_dataset[0].x.shape[1]

# Edge features are FIXED by design
in_edge_feats = EDGE_FEAT_DIM

model = train_model(
    train_loader,
    test_loader,
    in_node_feats,
    in_edge_feats,
    epochs=10,
)
