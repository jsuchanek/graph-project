import os
import math
import argparse
import datetime
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import folium
from folium.plugins import MarkerCluster

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops

# NOTE: Keep EDGE_FEATURES and HIGHWAY_TYPES in sync with training code
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
HIDDEN_DIM = 64


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
            nn.Linear(128, 1),
        )

    def forward(self, z, edge_index, edge_attr):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst], edge_attr], dim=1)
        return self.mlp(h).squeeze()


def load_county_graph_gdf(path):
    # read with geopandas to preserve geometry for mapping
    gdf = gpd.read_parquet(path)

    node_ids = pd.Index(pd.concat([gdf.u, gdf.v]).unique())
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    src = gdf.u.map(node_map).to_numpy()
    dst = gdf.v.map(node_map).to_numpy()

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_attr = torch.tensor(gdf[EDGE_FEATURES].values, dtype=torch.float)
    y = torch.tensor(gdf.has_camera.values, dtype=torch.float)

    x = torch.zeros((len(node_ids), NODE_DIM), dtype=torch.float)
    u_idx = gdf.u.map(node_map).to_numpy()
    v_idx = gdf.v.map(node_map).to_numpy()

    x[u_idx, 0] = torch.tensor(gdf.deg_u.values)
    x[v_idx, 0] = torch.tensor(gdf.deg_v.values)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return gdf, data


def get_direction_line(lat, lon, bearing_deg, length_meters=15):
    R = 6378137
    bearing = math.radians(bearing_deg)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    end_lat = math.asin(math.sin(lat_rad) * math.cos(length_meters/R) +
                        math.cos(lat_rad) * math.sin(length_meters/R) * math.cos(bearing))
    end_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(length_meters/R) * math.cos(lat_rad),
                                   math.cos(length_meters/R) - math.sin(lat_rad) * math.sin(end_lat))
    return [(lat, lon), (math.degrees(end_lat), math.degrees(end_lon))]


def generate_folium_map(df, cameras, title, top_k_percent=0.0005):
    minx, miny, maxx, maxy = df.total_bounds
    mean_lon = (minx + maxx) / 2
    mean_lat = (miny + maxy) / 2
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="cartodbpositron")

    watched = df[df['has_camera'] == 1]
    for _, row in watched.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="red", weight=5, opacity=0.9, tooltip="Watched Edge").add_to(m)

    k = max(1, int(top_k_percent * len(df)))
    top_preds = df.nlargest(k, "pred_score")
    for _, row in top_preds.iterrows():
        if row['has_camera'] == 0:
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="purple", weight=4, opacity=0.7,
                            tooltip=f"High Priority Prediction: {row['pred_score']:.4f}").add_to(m)

    marker_cluster = MarkerCluster(name="ALPR Hardware").add_to(m)
    for _, cam in cameras.iterrows():
        cam_lat, cam_lon = cam.geometry.y, cam.geometry.x
        folium.Marker(
            location=[cam_lat, cam_lon],
            icon=folium.Icon(color="darkred", icon="video", prefix="fa"),
            tooltip=f"Camera ID: {cam.get('id', 'Unknown')}"
        ).add_to(marker_cluster)

        dir_val = cam.get('direction')
        if dir_val is not None and not pd.isna(dir_val):
            try:
                needle_coords = get_direction_line(cam_lat, cam_lon, float(dir_val))
                folium.PolyLine(needle_coords, color="black", weight=3, opacity=1.0,
                                tooltip=f"Bearing: {dir_val}°").add_to(m)
            except ValueError:
                pass

    folium.LayerControl().add_to(m)
    return m


def run(args):
    device = torch.device(args.device)

    # instantiate models
    if args.model_type == "gat":
        gnn = GAT(NODE_DIM, HIDDEN_DIM)
    elif args.model_type == "graphsage":
        gnn = GraphSAGE(NODE_DIM, HIDDEN_DIM)
    elif args.model_type == "edgegcn":
        gnn = EdgeGCN(NODE_DIM, EDGE_DIM, HIDDEN_DIM)
    else:
        raise ValueError("Unknown model_type")

    decoder = EdgeDecoder(HIDDEN_DIM, EDGE_DIM)
    gnn.to(device)
    decoder.to(device)

    # load checkpoint (expects dict with keys 'gnn' and 'decoder')
    ckpt = torch.load(args.model_file, map_location=device)
    if 'gnn' in ckpt and 'decoder' in ckpt:
        gnn.load_state_dict(ckpt['gnn'])
        decoder.load_state_dict(ckpt['decoder'])
    else:
        # fallback: assume single model saved (edge-only) -- not expected
        try:
            gnn.load_state_dict(ckpt)
        except Exception as e:
            raise RuntimeError(f"Unable to load checkpoint: {e}")

    gnn.eval(); decoder.eval()

    all_cameras = gpd.read_file(args.alpr_geojson).to_crs(epsg=4326)

    states = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    global_dfs = []

    for state in states:
        print(f"Processing State: {state}...")
        state_input_path = os.path.join(args.data_root, state)
        state_output_path = os.path.join(args.output_root, state)
        os.makedirs(state_output_path, exist_ok=True)

        state_dfs = []
        for file in os.listdir(state_input_path):
            if not file.endswith('.parquet'):
                continue
            county_path = os.path.join(state_input_path, file)
            gdf, data = load_county_graph_gdf(county_path)
            data = data.to(device)

            with torch.no_grad():
                try:
                    z = gnn(data.x, data.edge_index, data.edge_attr)
                except TypeError:
                    z = gnn(data.x, data.edge_index)
                logits = decoder(z, data.edge_index, data.edge_attr)
                probs = torch.sigmoid(logits).cpu().numpy()

            gdf['pred_score'] = probs

            # find cameras in county bounds
            bounds = gdf.total_bounds
            cams_in_county = all_cameras.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

            county_name = file.replace('.parquet', '')
            c_map = generate_folium_map(gdf, cams_in_county, f"{county_name}, {state}", args.top_k_percent)
            c_map.save(os.path.join(state_output_path, f"{county_name}.html"))
            state_dfs.append(gdf)

        if len(state_dfs) == 0:
            continue

        full_state_df = pd.concat(state_dfs)
        s_bounds = full_state_df.total_bounds
        cams_in_state = all_cameras.cx[s_bounds[0]:s_bounds[2], s_bounds[1]:s_bounds[3]]
        s_map = generate_folium_map(full_state_df, cams_in_state, f"{state} Full Map", args.top_k_percent)
        s_map.save(os.path.join(state_output_path, f"{state}_Full_Map.html"))
        global_dfs.append(full_state_df)

    if len(global_dfs) > 0:
        usa_df = pd.concat(global_dfs)
        usa_map = generate_folium_map(usa_df, all_cameras, "USA ALPR Coverage", args.top_k_percent)
        os.makedirs(args.output_root, exist_ok=True)
        usa_map.save(os.path.join(args.output_root, f"USA_Combined_Map_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-file', default='results_gnn/gat_epoch_10.pt')
    p.add_argument('--model-type', default='gat', choices=['gat', 'graphsage', 'edgegcn'])
    p.add_argument('--data-root', default='precomputed')
    p.add_argument('--output-root', default='html')
    p.add_argument('--alpr-geojson', default='alpr/export.geojson')
    p.add_argument('--device', default='cpu')
    p.add_argument('--top-k-percent', type=float, default=0.0005)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
