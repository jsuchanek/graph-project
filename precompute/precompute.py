import os
import math
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Polygon
from pyrosm import OSM
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# Constants
# =========================

R = 6378137
MAX_WORKERS = 24  # Utilizing your 24 CPUs

STATE_NAME_TO_FIPS = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
    "california": "06", "colorado": "08", "connecticut": "09", "delaware": "10",
    "district-of-columbia": "11", "florida": "12", "georgia": "13", "hawaii": "15",
    "idaho": "16", "illinois": "17", "indiana": "18", "iowa": "19",
    "kansas": "20", "kentucky": "21", "louisiana": "22", "maine": "23",
    "maryland": "24", "massachusetts": "25", "michigan": "26", "minnesota": "27",
    "mississippi": "28", "missouri": "29", "montana": "30", "nebraska": "31",
    "nevada": "32", "new-hampshire": "33", "new-jersey": "34", "new-mexico": "35",
    "new-york": "36", "north-carolina": "37", "north-dakota": "38", "ohio": "39",
    "oklahoma": "40", "oregon": "41", "pennsylvania": "42", "rhode-island": "44",
    "south-carolina": "45", "south-dakota": "46", "tennessee": "47", "texas": "48",
    "utah": "49", "vermont": "50", "virginia": "51", "washington": "53",
    "west-virginia": "54", "wisconsin": "55", "wyoming": "56"
}

# =========================
# Helpers
# =========================

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

# =========================
# Core Processing Logic
# =========================

def process_single_county(osm_file, county_geo_id, county_geometry, alpr_gdf, out_dir):
    """Worker function to process a single county."""
    try:
        osm = OSM(osm_file, bounding_box=county_geometry)
        nodes, edges = osm.get_network("driving", nodes=True)
        
        if nodes is None or edges is None or len(edges) == 0:
            return f"Skipped {county_geo_id}: No data"

        nodes = nodes.set_index("id")
        G = nx.from_pandas_edgelist(edges, "u", "v", edge_attr="length")
        
        existing_node_ids = list(G.nodes())
        nodes = nodes.loc[nodes.index.intersection(existing_node_ids)].copy()

        # Node degree
        deg = dict(G.degree())
        nodes["degree_norm"] = nodes.index.map(deg).fillna(0)
        nodes["degree_norm"] = (nodes["degree_norm"] - nodes["degree_norm"].mean()) / (nodes["degree_norm"].std() + 1e-6)

        # Node Betweenness (k=50 sampling)
        node_bc = nx.betweenness_centrality(G, k=50, weight="length", normalized=True)
        nodes["node_bc_norm"] = nodes.index.map(node_bc).fillna(0.0)
        nodes["node_bc_norm"] = (nodes["node_bc_norm"] - nodes["node_bc_norm"].mean()) / (nodes["node_bc_norm"].std() + 1e-6)

        # Camera Presence
        nodes["has_camera"] = 0
        county_cams = alpr_gdf[alpr_gdf.within(county_geometry)]
        node_sindex = nodes.sindex

        for _, cam in county_cams.iterrows():
            if cam.get("direction") in [None, "N/A"]: continue
            try:
                bearing = float(str(cam["direction"]).split(";")[0])
                wedge = build_camera_wedge(cam.geometry.y, cam.geometry.x, bearing)
                possible_ids = list(node_sindex.intersection(wedge.bounds))
                if possible_ids:
                    subset = nodes.iloc[possible_ids]
                    hits = subset.geometry.within(wedge)
                    nodes.loc[subset[hits].index, "has_camera"] = 1
            except (ValueError, Exception): continue

        # Save Files
        os.makedirs(out_dir, exist_ok=True)
        nodes[["geometry", "degree_norm", "node_bc_norm", "has_camera"]].to_parquet(
            os.path.join(out_dir, f"{county_geo_id}_nodes.parquet")
        )
        edges[["u", "v", "length"]].to_parquet(
            os.path.join(out_dir, f"{county_geo_id}_edges.parquet")
        )
        
        return f"Success: {county_geo_id}"
    
    except Exception as e:
        return f"Error in {county_geo_id}: {str(e)}"

# =========================
# Parallel Driver
# =========================

def run():
    county_gdf = gpd.read_file("county/gz_2010_us_050_00_5m.json", encoding="latin1").to_crs(epsg=4326)
    alpr_gdf = gpd.read_file("alpr/export.geojson").to_crs(epsg=4326)

    tasks = []
    
    # 1. Collect all tasks across all states
    for state, fips in STATE_NAME_TO_FIPS.items():
        osm_file = f"osm_pbf/{state}-260116.osm.pbf"
        if not os.path.exists(osm_file):
            print(f"File missing for {state}, skipping...")
            continue
            
        counties = county_gdf[county_gdf["STATE"] == fips]
        out_dir = f"precomputed_nodes/{state}"
        
        for _, county in counties.iterrows():
            # We pass only necessary data to minimize serialization overhead
            tasks.append((
                osm_file, 
                county.GEO_ID, 
                county.geometry, 
                alpr_gdf, 
                out_dir
            ))

    print(f"Starting parallel processing for {len(tasks)} counties using {MAX_WORKERS} workers...")

    # 2. Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_county, *t) for t in tasks]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if "Error" in result:
                print(result)
            if i % 10 == 0:
                print(f"Progress: {i}/{len(tasks)} counties complete.")

if __name__ == "__main__":
    run()