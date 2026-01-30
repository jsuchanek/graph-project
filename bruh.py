import os
import re
import random
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

import math
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
    "residential",
    "living_street",
    "service",
    "unclassified",
    "road",
    "construction",
    "other",
]

HIGHWAY_TO_IDX = {h: i for i, h in enumerate(HIGHWAY_TYPES)}
NUM_HIGHWAY_TYPES = len(HIGHWAY_TYPES)
EDGE_FEAT_DIM = NUM_HIGHWAY_TYPES + 1  # +1 for length

import numpy as np

def encode_highway(highway_value):
    vec = np.zeros(NUM_HIGHWAY_TYPES, dtype=np.float32)

    if isinstance(highway_value, list):
        highway_value = highway_value[0]

    idx = HIGHWAY_TO_IDX.get(highway_value, HIGHWAY_TO_IDX["other"])
    vec[idx] = 1.0
    return vec


def get_state_counties(pbf_file, county_gdf):
    match = re.match(r"([a-z\-]+)-\d+\.osm\.pbf", pbf_file)
    if not match:
        raise ValueError(f"Cannot parse state from {pbf_file}")

    state_slug = match.group(1)
    state_fips = STATE_NAME_TO_FIPS[state_slug]

    state_counties = county_gdf[county_gdf["STATE"] == state_fips]

    print(f"  → {len(state_counties)} counties found for {state_slug}")
    return state_counties


def build_camera_wedge(lat, lon, bearing, fov=45, dist=20):
    def step(b):
        b = math.radians(b)
        lat2 = math.asin(math.sin(math.radians(lat)) * math.cos(dist / R) +
                            math.cos(math.radians(lat)) * math.sin(dist / R) * math.cos(b))
        lon2 = math.radians(lon) + math.atan2(
            math.sin(b) * math.sin(dist / R) * math.cos(math.radians(lat)),
            math.cos(dist / R) - math.sin(math.radians(lat)) * math.sin(lat2)
        )
        return math.degrees(lat2), math.degrees(lon2)

    l = step(bearing - fov / 2)
    r = step(bearing + fov / 2)
    return Polygon([(lon, lat), (l[1], l[0]), (r[1], r[0])])


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

class ALPRDataset(InMemoryDataset):
    def __init__(self, pbf_files, county_gdf, alpr_gdf, node_emb_dim=16):
        super().__init__()
        self.data_list = []

        for pbf_file in pbf_files:
            self.process_pbf(pbf_file, county_gdf, alpr_gdf)

        self.data, self.slices = self.collate(self.data_list)

    def process_pbf(self, pbf_file, county_gdf, alpr_gdf):
        print(f"\nProcessing {pbf_file}")
        counties = get_state_counties(pbf_file, county_gdf)

        for _, county in counties.iterrows():
            if county.geometry is None or county.geometry.is_empty:
                continue

            try:
                osm = OSM(f"{PBF_DIR}/{pbf_file}", bounding_box=county.geometry)
                nodes, edges = osm.get_network(network_type="driving", nodes=True)
                print(f"  County {county['NAME']} - {len(edges)} edges")
            except Exception:
                continue

            if edges is None or len(edges) == 0:
                continue
            
            # road_types = edges["highway"].astype(str).fillna("unclassified")
            # uniq = road_types.unique()
            # type_map = {t: i for i, t in enumerate(uniq)}

            # edge_attr = torch.eye(len(uniq))[road_types.map(type_map).values]
            edge_features = []
            for _, row in edges.iterrows():
                highway_vec = encode_highway(row.get("highway"))  # fixed size

                length = row.get("length", 0.0)
                length = float(length) if length else 0.0

                edge_feat = np.concatenate([
                    highway_vec,   # one-hot highway
                    [length],      # scalar (can normalize later)
                ])

                edge_features.append(edge_feat)

            edge_attr = torch.tensor(edge_features, dtype=torch.float)




            edges["has_camera"] = 0
            cams = alpr_gdf[alpr_gdf.within(county.geometry)]

            print(f"    → {len(cams)} cameras in county")
            # Build spatial index once per county
            edge_sindex = edges.sindex

            for _, cam in cams.iterrows():
                if cam.get("direction") in [None, "N/A"]:
                    continue

                try:
                    bearings = list(map(float, str(cam["direction"]).split(";")))
                except Exception:
                    continue

                for b in bearings:
                    wedge = build_camera_wedge(cam.geometry.y, cam.geometry.x, b)

                    # Fast bbox prefilter
                    candidate_idx = list(edge_sindex.intersection(wedge.bounds))
                    if not candidate_idx:
                        continue

                    # Precise geometry check
                    hits = edges.iloc[candidate_idx].geometry.intersects(wedge)

                    # Mark only true intersections
                    edges.loc[edges.iloc[candidate_idx][hits].index, "has_camera"] = 1

            # for _, cam in cams.iterrows():
            #     if cam.get("direction") in [None, "N/A"]:
            #         continue

            #     for b in map(float, str(cam["direction"]).split(";")):
            #         wedge = build_camera_wedge(cam.geometry.y, cam.geometry.x, b)
            #         edges.loc[edges.geometry.intersects(wedge), "has_camera"] = 1

            y = torch.tensor(edges["has_camera"].values, dtype=torch.float)

            G = nx.from_pandas_edgelist(edges, "u", "v")
            node_map = {n: i for i, n in enumerate(G.nodes())}

            x = torch.randn(len(node_map), 16)
            edge_index = torch.tensor(
                [[node_map[u], node_map[v]] for u, v in zip(edges.u, edges.v)],
                dtype=torch.long
            ).t()

            self.data_list.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )






import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, EdgeConv, NNConv

class EdgeClassifier(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden=32):
        super().__init__()
        # Node encoder
        self.conv1 = GCNConv(in_node_feats, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden + in_edge_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Node embedding
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Edge embeddings: concat node embeddings of endpoints + edge features
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col], edge_attr], dim=1)
        return self.edge_mlp(edge_emb).squeeze()


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def compute_pos_weight(dataset):
    total_pos = 0
    total_neg = 0
    for data in dataset:
        total_pos += data.y.sum().item()
        total_neg += (data.y == 0).sum().item()
    return total_neg / max(total_pos, 1)

from sklearn.metrics import precision_recall_curve



def train_model(
    train_loader,
    test_loader,
    in_node_feats,
    in_edge_feats,
    epochs=10,
    log_file="log.txt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EdgeClassifier(in_node_feats, in_edge_feats).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---- class imbalance handling ----
    pos_weight = compute_pos_weight(train_loader.dataset)
    print(f"Using pos_weight = {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device)
    )

    with open(log_file, "w") as f:
        for epoch in range(1, epochs + 1):

            # ======================
            # Train
            # ======================
            model.train()
            train_loss = 0.0

            for data in train_loader:
                data = data.to(device)

                optimizer.zero_grad()
                logits = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(logits, data.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * data.num_edges

            train_loss /= len(train_loader.dataset)

            # ======================
            # Evaluate
            # ======================
            model.eval()
            all_logits = []
            all_labels = []

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    logits = model(data.x, data.edge_index, data.edge_attr)

                    all_logits.append(logits.cpu())
                    all_labels.append(data.y.cpu())

            logits = torch.cat(all_logits)
            labels = torch.cat(all_labels)
            probs = torch.sigmoid(logits)

            #not 0.5 since class imbalance
            preds = (probs > 0.05).float()

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels.numpy(),
                preds.numpy(),
                average="binary",
                zero_division=0,
            )


            try:
                auroc = roc_auc_score(labels.numpy(), probs.numpy())
            except ValueError:
                auroc = float("nan")

            # ======================
            # Logging
            # ======================
            log_line = (
                f"Epoch {epoch:02d} | "
                f"Train Loss {train_loss:.6f} | "
                f"Precision {precision:.4f} | "
                f"Recall {recall:.4f} | "
                f"F1 {f1:.4f} | "
                f"AUROC {auroc:.4f}\n"
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            #labels = data.y.cpu().numpy()

            precision, recall, thresholds = precision_recall_curve(labels, probs)

            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = f1.argmax()

            best_thresh = thresholds[best_idx]
            best_f1 = f1[best_idx]

            print(f"Best threshold {best_thresh:.6f} | Best F1 {best_f1:.4f}")

            f.write(log_line)
            f.flush()

            torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
            print(log_line.strip())

    return model



train_dataset = ALPRDataset(train_files, gdf, alpr)
test_dataset = ALPRDataset(test_files, gdf, alpr)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

in_node_feats = train_dataset[0].x.shape[1]
in_edge_feats = train_dataset[0].edge_attr.shape[1]

# model = train_model(train_loader, test_loader, in_node_feats, in_edge_feats, epochs=20)
in_edge_feats = EDGE_FEAT_DIM
model = train_model(train_loader, test_loader, in_node_feats, in_edge_feats)
