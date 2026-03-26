import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyrosm import OSM
import cudf
import cugraph
import torch
import cupy
import rmm

# =========================
# Constants
# =========================
R = 6378137 

STATE_NAME_TO_FIPS = {
    "texas": "48",
    "california": "06",
}

# =========================
# Helper functions
# =========================

def verify_gpu():
    print("--- GPU Status Report ---")
    
    # 1. Check PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"PyTorch Device: {torch.cuda.get_device_name(0)}")
    
    # 2. Check cuDF (RAPIDS)
    try:
        # Create a tiny Series to test actual GPU memory allocation
        test_df = cudf.Series([1, 2, 3])
        print(f"cuDF GPU Access: SUCCESS (Found {cudf.get_current_device()})")
    except Exception as e:
        print(f"cuDF GPU Access: FAILED ({e})")

    # 3. Check cuGraph
    # Verify the G.degree() fix we just discussed
    try:
        G_test = cugraph.Graph()
        # Just checking if the object initializes correctly on GPU
        print("cuGraph Initialized: SUCCESS")
    except Exception as e:
        print(f"cuGraph Initialized: FAILED ({e})")
    
    print("-------------------------\n")

def compute_endpoint(lat, lon, bearing_deg, length_m=20):
    bearing = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(length_m / R) +
        math.cos(lat1) * math.sin(length_m / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(length_m / R) * math.cos(lat1),
        math.cos(length_m / R) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)

def build_camera_wedge(lat, lon, bearing_deg, fov_deg=45, length_m=20):
    half_fov = fov_deg / 2
    left_lat, left_lon = compute_endpoint(lat, lon, bearing_deg - half_fov, length_m)
    right_lat, right_lon = compute_endpoint(lat, lon, bearing_deg + half_fov, length_m)
    return Polygon([(lon, lat), (left_lon, left_lat), (right_lon, right_lat)])

def parallel_angle_diff(a, b):
    diff = abs(a - b) % 360
    diff = min(diff, 360 - diff)
    return min(diff, abs(diff - 180))

def edge_score(dist_m, angle_diff_deg, w_dist=1.0, w_angle=1.0):
    return w_dist * dist_m + w_angle * angle_diff_deg

def label_edges(edges_gdf, alpr_gdf, max_dist=20, w_dist=1.0, w_angle=1.0):
    edges_gdf = edges_gdf.copy()
    edges_gdf["has_camera"] = 0
    edges_gdf["camera_id"] = None
    edges_sindex = edges_gdf.sindex

    for cam_idx, cam in alpr_gdf.iterrows():
        cam_point = cam.geometry
        direction = cam.get("direction")
        bearings = []
        if direction not in [None, "N/A", ""]:
            try:
                bearings = [float(d) for d in str(direction).split(";")]
            except: bearings = []

        if not bearings:
            nearest_idx = edges_sindex.nearest(cam_point, return_distance=False).flatten()[0]
            edges_gdf.at[nearest_idx, "has_camera"] = 1
            edges_gdf.at[nearest_idx, "camera_id"] = cam_idx
            continue

        for bearing in bearings:
            wedge = build_camera_wedge(cam_point.y, cam_point.x, bearing)
            candidate_idxs = list(edges_sindex.intersection(wedge.bounds))
            if not candidate_idxs:
                candidate_idxs = edges_sindex.nearest(cam_point, return_distance=False).flatten()

            best_score, best_idx = float("inf"), None
            for idx in candidate_idxs:
                edge = edges_gdf.iloc[idx]
                geom, edge_bearing = edge.geometry, edge.get("bearing")
                if geom is None or edge_bearing is None: continue

                dist_m = cam_point.distance(geom)
                if dist_m > max_dist: continue

                angle_diff = parallel_angle_diff(bearing, edge_bearing)
                score = edge_score(dist_m, angle_diff, w_dist, w_angle)

                if score < best_score:
                    best_score, best_idx = score, idx

            if best_idx is not None:
                edges_gdf.at[best_idx, "has_camera"] = 1
                edges_gdf.at[best_idx, "camera_id"] = cam_idx
    return edges_gdf

def compute_graph_metrics(edges_cudf):
    """GPU-accelerated metrics using cuGraph."""
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edges_cudf, source="u", destination="v", edge_attr="length")

    # Centrality & Degree
    print("Computing degree...")
    deg_df = G.degree().rename(columns={"vertex": "node_id", "degree": "degree"})
    deg_df["degree"] = (deg_df["degree"] - deg_df["degree"].mean()) / (deg_df["degree"].std() + 1e-6)
    print("Computing betweenness centrality...")
    bc_df = cugraph.betweenness_centrality(G, k=50).rename(columns={"vertex": "node_id", "betweenness": "betweenness"})
    bc_df["betweenness"] = (bc_df["betweenness"] - bc_df["betweenness"].mean()) / (bc_df["betweenness"].std() + 1e-6)

    # Node2Vec (GPU Native)
    # Note: cugraph.node2vec returns (embeddings_df, nodes_series)
    print("Computing Node2Vec embeddings...")
    n2v_embeddings, nodes = cugraph.experimental.node2vec(
        G, embedding_size=32, walk_length=100, context_size=5, walks_per_node=20, p=1.0, q=0.5
    )
    n2v_df = cudf.DataFrame(n2v_embeddings)
    n2v_df = (n2v_df - n2v_df.mean()) / (n2v_df.std() + 1e-6)
    
    n2v_df["node_id"] = nodes

    # Merge all GPU dataframes
    return deg_df.merge(bc_df, on="node_id").merge(n2v_df, on="node_id")

def process_county(osm_file, county_row, alpr_gdf, out_dir):
    print(f"Loading OSM data...", flush=True)
    osm = OSM(osm_file, bounding_box=county_row.geometry)
    nodes, edges = osm.get_network("driving", nodes=True)

    if edges is None or len(edges) == 0: return

    # --- SINGLE PASS PREPROCESSING ---
    # 1. Fill missing lengths with median
    median_len = edges["length"].median()
    edges["length"] = edges["length"].fillna(median_len)
    
    # 2. Precompute bearings
    print("Labeling edges...", flush=True)
    edges["bearing"] = edges.geometry.apply(
        lambda g: math.degrees(math.atan2((g.coords[-1][0]-g.coords[0][0]),
                                         (g.coords[-1][1]-g.coords[0][1]))) % 360
        if g is not None else None
    )

    # 3. Labeling (Spatial join/index logic stays on CPU due to Shapely)
    edges = label_edges(edges, alpr_gdf)

    # 4. Move to GPU ONCE
    edges_cudf = cudf.from_pandas(edges[["u", "v", "length", "has_camera"]])
    edges_cudf["inv_length"] = 1.0 / (edges_cudf["length"] + 1e-6)
    edges_cudf["watched"] = edges_cudf["has_camera"].astype(bool)

    # Compute metrics
    node_metrics_cudf = compute_graph_metrics(edges_cudf)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    fips = county_row.GEO_ID
    node_metrics_cudf.to_parquet(os.path.join(out_dir, f"{fips}_nodes.parquet"))
    edges_cudf[["u", "v", "length", "inv_length", "watched"]].to_parquet(os.path.join(out_dir, f"{fips}_edges.parquet"))

def run():
    # Initialize RMM Pool
    # On a 40GB A100, let's reserve 30GB for RAPIDS to breathe.
    # This prevents the 'std::bad_alloc' during heavy merge/shuffle operations.
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=30 * 1024 * 1024 * 1024, # 30 GB
        maximum_pool_size=38 * 1024 * 1024 * 1024  # 38 GB
    )
    verify_gpu()
    county_gdf = gpd.read_file("county/gz_2010_us_050_00_5m.json", encoding="latin1").to_crs(epsg=4326)
    alpr_gdf = gpd.read_file("alpr/export.geojson").to_crs(epsg=4326)

    for state, fips in STATE_NAME_TO_FIPS.items():
        print(f"\n=== Processing {state.upper()} ===", flush=True)
        osm_path = f"osm_pbf/{state}-260116.osm.pbf"
        counties = county_gdf[county_gdf["STATE"] == fips]
        for _, county in counties.iterrows():
            print(f"Processing county {county.NAME} ({county.GEO_ID})...")
            process_county(osm_path, county, alpr_gdf, os.path.join("precomputed_v2", state))

if __name__ == "__main__":
    run()