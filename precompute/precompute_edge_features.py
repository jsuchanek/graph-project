import os
import re
import math
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, Polygon
from pyrosm import OSM


# =========================
# Constants
# =========================

R = 6378137

HIGHWAY_TYPES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "service", "unclassified", "road", "construction", "other"
]
HIGHWAY_TO_IDX = {h: i for i, h in enumerate(HIGHWAY_TYPES)}

STATE_NAME_TO_FIPS = {
    "ohio": "39",
    # "california": "06",
    # "texas": "48",
    # "new-york": "36",
    # add more as needed
}

# =========================
# Helpers
# =========================

def encode_highway(highway):
    vec = np.zeros(len(HIGHWAY_TYPES), dtype=np.float32)
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

# =========================
# Main processing
# =========================

def process_county(osm_file, county_row, alpr_gdf, out_dir):
    print(f"Processing county {county_row.GEO_ID}...")
    try:
        osm = OSM(osm_file, bounding_box=county_row.geometry)
        nodes, edges = osm.get_network("driving", nodes=True)
    except Exception:
        return

    if edges is None or len(edges) == 0:
        return

    edges = edges.reset_index(drop=True)

    # =========================
    # Graph
    # =========================
    G = nx.from_pandas_edgelist(edges, "u", "v")

    print("Computing degree...")
    # Node degree (per county)
    deg = dict(G.degree())
    deg_vals = np.array(list(deg.values()), dtype=np.float32)
    deg_vals = (deg_vals - deg_vals.mean()) / (deg_vals.std() + 1e-6)
    deg_norm = dict(zip(deg.keys(), deg_vals))

    print("Computing betweenness centrality...")
    # Edge betweenness (per county)
    edge_bc = nx.edge_betweenness_centrality(G, k=50, weight="length", normalized=True)
    bc_vals = []
    for u, v in zip(edges.u, edges.v):
        bc_vals.append(edge_bc.get((u, v), edge_bc.get((v, u), 0.0)))

    bc_vals = np.array(bc_vals, dtype=np.float32)
    bc_vals = (bc_vals - bc_vals.mean()) / (bc_vals.std() + 1e-6)

    print("Lenght and highway type")
    # =========================
    # Edge features
    # =========================
    highway_feats = np.vstack(edges["highway"].apply(encode_highway).values)
    length = edges.get("length", 0.0).fillna(0.0).astype(float)
    log_length = np.log1p(length)

    edges["log_length"] = log_length
    edges["edge_bc"] = bc_vals
    edges["deg_u"] = edges.u.map(deg_norm)
    edges["deg_v"] = edges.v.map(deg_norm)

    for i, h in enumerate(HIGHWAY_TYPES):
        edges[f"highway_{h}"] = highway_feats[:, i]

    print("Processing camera presence...")
    # =========================
    # Labels (camera presence)
    # =========================
    edges["has_camera"] = 0
    cams = alpr_gdf[alpr_gdf.within(county_row.geometry)]
    sindex = edges.sindex

    for _, cam in cams.iterrows():
        if cam.get("direction") in [None, "N/A"]:
            continue
        try:
            bearing = float(str(cam["direction"]).split(";")[0])
        except ValueError:
            continue

        wedge = build_camera_wedge(cam.geometry.y, cam.geometry.x, bearing)
        idxs = list(sindex.intersection(wedge.bounds))
        if not idxs:
            continue

        hits = edges.iloc[idxs].geometry.intersects(wedge)
        edges.loc[edges.iloc[idxs][hits].index, "has_camera"] = 1

    # =========================
    # Save
    # =========================
    os.makedirs(out_dir, exist_ok=True)
    county_fips = county_row.GEO_ID
    out_path = os.path.join(out_dir, f"{county_fips}.parquet")

    edges[
        ["u", "v", "geometry", "log_length", "edge_bc", "deg_u", "deg_v"] + [f"highway_{h}" for h in HIGHWAY_TYPES] + ["has_camera"]
    ].to_parquet(out_path)

    print(f"Saved {out_path}")

# =========================
# Driver
# =========================

def run():
    county_gdf = gpd.read_file(
        "county/gz_2010_us_050_00_5m.json", encoding="latin1"
    ).to_crs(epsg=4326)

    alpr_gdf = gpd.read_file("alpr/export.geojson").to_crs(epsg=4326)

    for state, fips in STATE_NAME_TO_FIPS.items():
        print(f"\n=== Processing {state.upper()} ===")
        osm_file = f"osm_pbf/{state}-260116.osm.pbf"
        counties = county_gdf[county_gdf["STATE"] == fips]

        out_dir = f"precomputed/{state}"

        for _, county in counties.iterrows():
            process_county(osm_file, county, alpr_gdf, out_dir)

if __name__ == "__main__":
    run()
