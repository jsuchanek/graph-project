import os
import math
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import folium
from folium.plugins import MarkerCluster

# ============================================================
# Model Definition
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
# Configuration & Constants
# ============================================================
DATA_ROOT = "precomputed"
MODEL_PATH = "results/edge_mlp_epoch_10.pt"
ALPR_GEOJSON = "alpr/export.geojson"
OUTPUT_ROOT = "html"

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

DEVICE = torch.device("cpu")
TOP_K_PERCENT = 0.0005 

# ============================================================
# Helper: Calculate Direction Line
# ============================================================
def get_direction_line(lat, lon, bearing_deg, length_meters=15):
    """Calculates a short line segment in the direction of the bearing."""
    # Earth's radius in meters
    R = 6378137 
    bearing = math.radians(bearing_deg)
    
    # Calculate end point coordinates
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    end_lat = math.asin(math.sin(lat_rad) * math.cos(length_meters/R) +
                        math.cos(lat_rad) * math.sin(length_meters/R) * math.cos(bearing))
    
    end_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(length_meters/R) * math.cos(lat_rad),
                                   math.cos(length_meters/R) - math.sin(lat_rad) * math.sin(end_lat))
    
    return [(lat, lon), (math.degrees(end_lat), math.degrees(end_lon))]

def state_is_finished(state, output_root):
    final_html = os.path.join(
        output_root,
        state,
        f"{state}_Full_Map.html"
    )
    return os.path.exists(final_html)


# ============================================================
# Mapping Logic
# ============================================================
def generate_folium_map(df, cameras, title):
    minx, miny, maxx, maxy = df.total_bounds
    mean_lon = (minx + maxx) / 2
    mean_lat = (miny + maxy) / 2
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles="cartodbpositron")

    # 1. Plot Ground Truth (Watched Edges) - RED
    watched = df[df['has_camera'] == 1]
    for _, row in watched.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="red", weight=5, opacity=0.9, tooltip="Watched Edge").add_to(m)

    # 2. Plot Predicted Top K (Purple)
    k = max(1, int(TOP_K_PERCENT * len(df)))
    top_preds = df.nlargest(k, "pred_score")
    for _, row in top_preds.iterrows():
        if row['has_camera'] == 0:
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(coords, color="purple", weight=4, opacity=0.7, 
                            tooltip=f"High Priority Prediction: {row['pred_score']:.4f}").add_to(m)

    # 3. Plot Camera Icons and Direction Needles
    marker_cluster = MarkerCluster(name="ALPR Hardware").add_to(m)
    for _, cam in cameras.iterrows():
        cam_lat, cam_lon = cam.geometry.y, cam.geometry.x
        
        # Add Marker to Cluster
        folium.Marker(
            location=[cam_lat, cam_lon],
            icon=folium.Icon(color="darkred", icon="video", prefix="fa"),
            tooltip=f"Camera ID: {cam.get('id', 'Unknown')}"
        ).add_to(marker_cluster)

        # Add Direction Needle (Directly to map so it doesn't disappear in clusters)
        # Assuming direction column is numeric degrees
        dir_val = cam.get('direction')
        if dir_val is not None and not pd.isna(dir_val):
            try:
                needle_coords = get_direction_line(cam_lat, cam_lon, float(dir_val))
                folium.PolyLine(
                    needle_coords, 
                    color="black", 
                    weight=3, 
                    opacity=1.0,
                    tooltip=f"Bearing: {dir_val}°"
                ).add_to(m)
            except ValueError:
                pass

    folium.LayerControl().add_to(m)
    return m

# ============================================================
# Main Inference Loop (remains largely the same)
# ============================================================
def run_pipeline():
    model = EdgeMLP(len(EDGE_FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_cameras = gpd.read_file(ALPR_GEOJSON).to_crs(epsg=4326)
    states = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    global_dfs = []

    for state in states:
        print(f"Processing State: {state}...")
        state_input_path = os.path.join(DATA_ROOT, state)
        state_output_path = os.path.join(OUTPUT_ROOT, state)
        os.makedirs(state_output_path, exist_ok=True)
        
        state_dfs = []
        for file in os.listdir(state_input_path):
            if not file.endswith(".parquet"): continue
            df = gpd.read_parquet(os.path.join(state_input_path, file))
            
            # Inference
            edge_attr = torch.tensor(df[EDGE_FEATURES].values, dtype=torch.float).to(DEVICE)
            with torch.no_grad():
                scores = torch.sigmoid(model(edge_attr)).numpy()
            df['pred_score'] = scores
            
            # Filter Cameras to this County
            bounds = df.total_bounds
            cams_in_county = all_cameras.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            
            # Save County Map
            county_name = file.replace(".parquet", "")
            c_map = generate_folium_map(df, cams_in_county, f"{county_name}, {state}")
            c_map.save(os.path.join(state_output_path, f"{county_name}.html"))
            state_dfs.append(df)
        
        full_state_df = pd.concat(state_dfs)
        s_bounds = full_state_df.total_bounds
        cams_in_state = all_cameras.cx[s_bounds[0]:s_bounds[2], s_bounds[1]:s_bounds[3]]
        s_map = generate_folium_map(full_state_df, cams_in_state, f"{state} Full Map")
        s_map.save(os.path.join(state_output_path, f"{state}_Full_Map.html"))
        global_dfs.append(full_state_df)

    usa_df = pd.concat(global_dfs)
    usa_map = generate_folium_map(usa_df, all_cameras, "USA ALPR Coverage")
    usa_map.save(os.path.join(OUTPUT_ROOT, "USA_Combined_Map.html"))

if __name__ == "__main__":
    run_pipeline()