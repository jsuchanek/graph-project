import os
import geopandas as gpd

# Load county polygons (GeoJSON of all US counties)
gdf = gpd.read_file("/county/gz_2010_us_050_00_5m.json", encoding="latin1")

# Load ALPR cameras
alpr = gpd.read_file("alpr/export.geojson").set_crs(epsg=4326)
PBF_DIR = "osm_pbf"
pbf_files = [f for f in os.listdir(PBF_DIR) if f.endswith(".osm.pbf")]

# Get sizes
pbf_sizes = {f: os.path.getsize(os.path.join(PBF_DIR, f)) / (1024*1024) for f in pbf_files}

# Smallest 21
smallest_files = sorted(pbf_sizes.items(), key=lambda x: x[1])[:21]
smallest_files = [f for f, s in smallest_files]

# Train/test split (~75% train by cumulative size)
total_size = sum([pbf_sizes[f] for f in smallest_files])
train_files, test_files = [], []
train_size = 0

for f in smallest_files:
    if train_size / total_size < 0.67:
        train_files.append(f)
        train_size += pbf_sizes[f]
    else:
        test_files.append(f)

print("Train:", train_files)
print("Test:", test_files)
