import geopandas as gpd
import fiona
import json
import gzip
import pandas as pd
import folium
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from folium.plugins import MarkerCluster
import osmnx as ox
from pyrosm import OSM

# Use fiona with latin1 encoding
gdf = gpd.read_file(
    "county/gz_2010_us_050_00_5m.json",
    encoding="latin1"
)

print(gdf.head())


# Check columns
print(gdf.columns)
# ['GEO_ID', 'STATE', 'COUNTY', 'NAME', 'LSAD', 'CENSUSAREA', 'geometry']


# Filter for Virginia (STATE = "51") and Hanover County
county_polygon = gdf[(gdf["STATE"] == "50") & (gdf["NAME"] == "Bennington")]

# Check what we got
print(county_polygon)

# Extract the polygon (geometry)
county_polygon = county_polygon.geometry.iloc[0]
print(county_polygon)


center_lat = county_polygon.centroid.y
center_lon = county_polygon.centroid.x

alpr = gpd.read_file("alpr/export.geojson")
alpr = alpr.set_crs(epsg=4326)

alpr_in_county = alpr[alpr.within(county_polygon)].copy()

print("ALPR cameras in county:", len(alpr_in_county))

def compute_endpoint(lat, lon, bearing_deg, length_m=40):
    R = 6378137
    bearing = math.radians(bearing_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(length_m / R) +
        math.cos(lat1) * math.sin(length_m / R) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(length_m / R) * math.cos(lat1),
        math.cos(length_m / R) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

def build_camera_wedge(lat, lon, bearing_deg, fov_deg=45, length_m=30):
    """
    Returns a Shapely Polygon representing the camera field-of-view wedge
    """
    half_fov = fov_deg / 2

    left_bearing = bearing_deg - half_fov
    right_bearing = bearing_deg + half_fov

    left_lat, left_lon = compute_endpoint(lat, lon, left_bearing, length_m)
    right_lat, right_lon = compute_endpoint(lat, lon, right_bearing, length_m)

    origin = (lon, lat)
    left_pt = (left_lon, left_lat)
    right_pt = (right_lon, right_lat)

    wedge = Polygon([origin, left_pt, right_pt])

    return wedge


def find_best_edge_in_wedge(
    cam_point,
    cam_bearing,
    wedge_poly,
    edges,
    edges_sindex,
    max_dist=20
):
    best_score = float("inf")
    best_edge_idx = None
    best_edge_row = None

    # Spatial index filter
    candidate_idxs = list(
        edges_sindex.intersection(wedge_poly.bounds)
    )

    for idx in candidate_idxs:
        edge = edges.iloc[idx]
        geom = edge.geometry

        if geom is None:
            continue

        if not geom.intersects(wedge_poly):
            continue

        dist = cam_point.distance(geom)
        if dist > max_dist:
            continue

        edge_bearing = edge.bearing
        if edge_bearing is None:
            continue

        angle_diff = parallel_angle_diff(cam_bearing, edge_bearing)

        score = edge_score(dist, angle_diff)

        if score < best_score:
            best_score = score
            best_edge_idx = idx
            best_edge_row = edge

    return best_edge_idx, best_edge_row, best_score

def line_bearing(linestring):
    """
    Bearing of a LineString in degrees [0, 360)
    """
    x1, y1 = linestring.coords[0]
    x2, y2 = linestring.coords[-1]

    dx = x2 - x1
    dy = y2 - y1

    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360) % 360

def parallel_angle_diff(a, b):
    """
    Returns smallest angle difference considering 180° symmetry
    """
    diff = abs(a - b) % 360
    diff = min(diff, 360 - diff)
    return min(diff, abs(diff - 180))

def edge_score(dist_m, angle_diff_deg,
               w_dist=1.0, w_angle=2.0):
    """
    Lower score = better
    """
    return w_dist * dist_m + w_angle * angle_diff_deg

print("Loading OSM data...")
osm = OSM(
    "osm_pbf/vermont-260116.osm.pbf",
    bounding_box=county_polygon
)

print("Extracting driving network...")
nodes, edges = osm.get_network(
    network_type="driving",
    nodes=True
)
edges["bearing"] = edges.geometry.apply(
    lambda g: line_bearing(g) if g is not None else None
)
edges.head()

edges = edges.reset_index(drop=True)
print(edges.head())
edges_sindex = edges.sindex

edges["has_camera"] = 0
edges["camera_id"] = None

for cam_idx, cam in alpr_in_county.iterrows():
    lat = cam.geometry.y
    lon = cam.geometry.x
    direction = cam.get("direction")

    if direction in [None, "N/A"]:
        continue

    try:
        bearings = [float(d) for d in str(direction).split(";")]
    except:
        continue

    cam_point = Point(lon, lat)

    for bearing in bearings:
        wedge = build_camera_wedge(lat, lon, bearing)

        edge_idx, edge_row, score = find_best_edge_in_wedge(
            cam_point,
            bearing,
            wedge,
            edges,
            edges_sindex
        )

        if edge_idx is not None:
            edges.at[edge_idx, "has_camera"] = 1
            edges.at[edge_idx, "camera_id"] = cam_idx

watched_edges = edges[edges["has_camera"] == 1]
print("Watched edges: ", watched_edges)

m = folium.Map(zoom_start=12, tiles="cartodbpositron")

# Plot edges with cameras
for idx, row in watched_edges.iterrows():
    if isinstance(row.geometry, LineString):
        coords = [(y, x) for x, y in row.geometry.coords]  # Folium uses (lat, lon)
        folium.PolyLine(coords, color="blue", weight=3, opacity=0.7, tooltip=f"Edge ID: {row['id']}").add_to(m)

        # Add markers at endpoints with the edge ID
        u_lat, u_lon = coords[0]
        v_lat, v_lon = coords[-1]
        folium.Marker([u_lat, u_lon], icon=folium.DivIcon(html=f"<div style='font-size:10px;color:blue'>{row['id']} (u)</div>")).add_to(m)
        folium.Marker([v_lat, v_lon], icon=folium.DivIcon(html=f"<div style='font-size:10px;color:green'>{row['id']} (v)</div>")).add_to(m)

# Plot ALPR cameras as red points
alpr_cluster = MarkerCluster(name="ALPR Cameras").add_to(m)
for idx, row in alpr_in_county.iterrows():
    folium.Marker(
        [row.geometry.y, row.geometry.x],
        icon=folium.Icon(color="red", icon="camera", prefix="fa"),
        tooltip=f"ALPR Camera {idx}"
    ).add_to(alpr_cluster)

# Optional: add county boundary
folium.GeoJson(county_polygon, name="County Boundary", style_function=lambda x: {
    "color": "black", "fill": False, "weight": 2
}).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
m.save("alpr_cameras_in_county.html")

