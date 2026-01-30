import os
import re
import math
import numpy as np
import geopandas as gpd
import torch
import networkx as nx
import folium
from shapely.geometry import Polygon, Point, LineString
from pyrosm import OSM
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from folium.plugins import MarkerCluster

# -------------------------------
# Constants
# -------------------------------
R = 6378137
HIGHWAY_TYPES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "service", "unclassified", "road", "construction", "other"
]
HIGHWAY_TO_IDX = {h: i for i, h in enumerate(HIGHWAY_TYPES)}
NUM_HIGHWAY_TYPES = len(HIGHWAY_TYPES)
EDGE_FEAT_DIM = NUM_HIGHWAY_TYPES + 1  # + log(length)

# -------------------------------
# Helper functions
# -------------------------------
def encode_highway(highway):
    vec = np.zeros(NUM_HIGHWAY_TYPES, dtype=np.float32)
    if isinstance(highway, list):
        highway = highway[0]
    idx = HIGHWAY_TO_IDX.get(highway, HIGHWAY_TO_IDX["other"])
    vec[idx] = 1.0
    return vec

def build_camera_wedge(lat, lon, bearing, fov=45, dist=20):
    def step(b):
        b = math.radians(b)
        lat2 = math.asin(
            math.sin(math.radians(lat)) * math.cos(dist / R) +
            math.cos(math.radians(lat)) * math.sin(dist / R) * math.cos(b)
        )
        lon2 = math.radians(lon) + math.atan2(
            math.sin(b) * math.sin(dist / R) * math.cos(math.radians(lat)),
            math.cos(dist / R) - math.sin(math.radians(lat)) * math.sin(lat2)
        )
        return math.degrees(lat2), math.degrees(lon2)
    l = step(bearing - fov / 2)
    r = step(bearing + fov / 2)
    return Polygon([(lon, lat), (l[1], l[0]), (r[1], r[0])])

def prepare_edge_features(edges):
    feats = []
    for _, r in edges.iterrows():
        hwy = encode_highway(r.get("highway"))
        length = r.get("length", 0.0)
        length = math.log1p(float(length)) if length else 0.0
        feats.append(np.concatenate([hwy, [length]]))
    return torch.tensor(np.array(feats), dtype=torch.float)

def find_best_edge_in_wedge(wedge, edges, sindex, scores):
    """Return the edge inside the wedge with the highest score"""
    candidate_idx = list(sindex.intersection(wedge.bounds))
    if not candidate_idx:
        return None, None, None
    hits = edges.iloc[candidate_idx].geometry.intersects(wedge)
    candidate_idx = edges.iloc[candidate_idx][hits].index
    if len(candidate_idx) == 0:
        return None, None, None
    candidate_scores = scores[candidate_idx]
    best_idx = candidate_idx[np.argmax(candidate_scores)]
    best_score = candidate_scores[np.argmax(candidate_scores)]
    return best_idx, edges.loc[best_idx], best_score

# -------------------------------
# Model definition
# -------------------------------
class EdgeClassifier(torch.nn.Module):
    def __init__(self, in_node, in_edge, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_node, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*hidden + in_edge, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        row, col = edge_index
        return self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=1)).squeeze()

# -------------------------------
# Load data
# -------------------------------
gdf = gpd.read_file("county/gz_2010_us_050_00_5m.json", encoding="latin1").to_crs(epsg=4326)
alpr = gpd.read_file("alpr/export.geojson").to_crs(epsg=4326)
hawaii_gdf = gdf[gdf["STATE"]=="15"]  # Hawaii FIPS

pbf_file = "hawaii-260116.osm.pbf"
osm = OSM(f"osm_pbf/{pbf_file}")
nodes, edges = osm.get_network("driving", nodes=True)

# Prepare edge features
edge_attr = prepare_edge_features(edges)

# Build graph
G = nx.from_pandas_edgelist(edges, "u", "v")
node_map = {n:i for i,n in enumerate(G.nodes())}
degrees = np.array([G.degree[n] for n in G.nodes()], dtype=np.float32)
degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
x = torch.tensor(degrees[:, None], dtype=torch.float)
edge_index = torch.tensor([[node_map[u], node_map[v]] for u,v in zip(edges.u, edges.v)], dtype=torch.long).t()

# -------------------------------
# Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_node = x.shape[1]
in_edge = edge_attr.shape[1]
model = EdgeClassifier(in_node, in_edge).to(device)
model.load_state_dict(torch.load("model_epoch_10.pt", map_location=device))
model.eval()

# -------------------------------
# Run inference
# -------------------------------
edges["has_camera"] = 0
edges["camera_id"] = None
sindex = edges.sindex

# Precompute all edge scores
with torch.no_grad():
    scores = torch.sigmoid(model(x.to(device), edge_index.to(device), edge_attr.to(device))).cpu().numpy()
edges["score"] = scores
top_k = max(1, int(0.001 * len(edges)))  # 1% of all edges
top_edges = edges.nlargest(top_k, "score")


# Assign cameras to edges
for _, county in hawaii_gdf.iterrows():
    alpr_in_county = alpr[alpr.geometry.within(county.geometry)]
    for cam_idx, cam in alpr_in_county.iterrows():
        if cam.get("direction") in [None, "N/A"]:
            continue
        first_dir = float(re.split(r"[;-]", str(cam["direction"]))[0])
        wedge = build_camera_wedge(cam.geometry.y, cam.geometry.x, first_dir)
        edge_idx, _, _ = find_best_edge_in_wedge(wedge, edges, sindex, scores)
        if edge_idx is not None:
            edges.at[edge_idx, "has_camera"] = 1
            edges.at[edge_idx, "camera_id"] = cam_idx

watched_edges = edges[edges["has_camera"]==1]
print("Watched edges:", len(watched_edges))

# -------------------------------
# Build Folium map
# -------------------------------
m = folium.Map(location=[20.8, -156.3], zoom_start=7, tiles="cartodbpositron")

# Plot edges with cameras
for idx, row in watched_edges.iterrows():
    if isinstance(row.geometry, LineString):
        coords = [(y,x) for x,y in row.geometry.coords]
        folium.PolyLine(coords, color="blue", weight=3, opacity=0.7).add_to(m)
for idx, row in top_edges.iterrows():
    if isinstance(row.geometry, LineString):
        coords = [(y, x) for x, y in row.geometry.coords]  # Folium expects (lat, lon)
        folium.PolyLine(
            coords,
            color="purple",
            weight=3,
            opacity=0.7,
            tooltip=f"Predicted score: {row['score']:.4f}"
        ).add_to(m)


# Plot ALPR cameras
alpr_cluster = MarkerCluster(name="ALPR Cameras").add_to(m)
for idx, row in alpr.iterrows():
    if row.geometry.within(hawaii_gdf.unary_union):
        folium.Marker(
            [row.geometry.y, row.geometry.x],
            icon=folium.Icon(color="red", icon="camera", prefix="fa")
        ).add_to(alpr_cluster)

# Add county boundaries
for _, row in hawaii_gdf.iterrows():
    folium.GeoJson(row.geometry, style_function=lambda x: {"color":"black","fill":False,"weight":2}).add_to(m)

# Layer control and save
folium.LayerControl().add_to(m)
m.save("hawaii_alpr_map.html")
print("Map saved to hawaii_alpr_map.html")
