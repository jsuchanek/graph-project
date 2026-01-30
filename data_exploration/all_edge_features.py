import os
from pyrosm import OSM

PBF_DIR = "osm_pbf"   # change if needed

highway_types = set()

pbf_files = [f for f in os.listdir(PBF_DIR) if f.endswith(".osm.pbf")]

print(f"Found {len(pbf_files)} PBF files")

for pbf in pbf_files:
    path = os.path.join(PBF_DIR, pbf)
    print(f"Processing {pbf} ...")

    osm = OSM(path)

    # Driving network only
    _, edges = osm.get_network(
        network_type="driving",
        nodes=False
    )

    if edges is None or len(edges) == 0:
        continue

    # Collect highway tags
    if "highway" in edges.columns:
        vals = edges["highway"].dropna().astype(str).unique()
        highway_types.update(vals)

print("\n===== UNIQUE HIGHWAY TYPES =====")
for h in sorted(highway_types):
    print(h)

# Save for later reuse
with open("highway_types.txt", "w") as f:
    for h in sorted(highway_types):
        f.write(h + "\n")

print(f"\nSaved {len(highway_types)} highway types to highway_types.txt")
