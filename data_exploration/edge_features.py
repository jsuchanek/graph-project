import os
import random
import pandas as pd
from collections import Counter
from multiprocessing import Pool, cpu_count
from pyrosm import OSM

# -----------------------------
# Config
# -----------------------------
PBF_DIR = "osm_pbf"
N_STATES = 50
NETWORK_TYPE = "driving"

# -----------------------------
# Worker
# -----------------------------
def process_state(pbf_file):
    state_name = pbf_file.replace(".osm.pbf", "")
    print(f"\n🚦 Starting {state_name}...", flush=True)

    osm = OSM(os.path.join(PBF_DIR, pbf_file))
    edges = osm.get_network(network_type=NETWORK_TYPE, nodes=False)

    highways = edges["highway"].dropna()
    counts = highways.value_counts()
    total = counts.sum()

    print(f"✅ Finished {state_name}", flush=True)
    for hwy, cnt in counts.head(5).items():
        print(f"  {hwy:15s}: {100 * cnt / total:6.2f}%", flush=True)

    return {
        "state": state_name,
        "counts": counts.to_dict(),
        "total": total
    }

# -----------------------------
# Main
# -----------------------------
def main():
    # List all PBF files with size
    pbf_files = [
        (f, os.path.getsize(os.path.join(PBF_DIR, f)))
        for f in os.listdir(PBF_DIR) if f.endswith(".osm.pbf")
    ]

    # Sort by size (smallest first)
    pbf_files.sort(key=lambda x: x[1])

    # Pick the 5 smallest
    sampled_pbfs = [f for f, s in pbf_files[:N_STATES]]

    print("🗺️ Sampled states (smallest files):")
    for f in sampled_pbfs:
        print(" ", f)

    n_workers = min(cpu_count(), N_STATES)

    with Pool(n_workers) as pool:
        results = pool.map(process_state, sampled_pbfs)

    # Aggregate
    global_counter = Counter()
    rows = []

    for res in results:
        global_counter.update(res["counts"])
        for hwy, cnt in res["counts"].items():
            rows.append({
                "state": res["state"],
                "highway": hwy,
                "percentage": 100 * cnt / res["total"]
            })

    state_df = pd.DataFrame(rows)

    overall_total = sum(global_counter.values())
    overall_df = (
        pd.DataFrame.from_dict(global_counter, orient="index", columns=["count"])
        .assign(percentage=lambda x: 100 * x["count"] / overall_total)
        .reset_index()
        .rename(columns={"index": "highway"})
        .sort_values("percentage", ascending=False)
    )

    print("\n🌎 Overall highway distribution:")
    print(overall_df.head(10))


# -----------------------------
# Entry point (REQUIRED)
# -----------------------------
if __name__ == "__main__":
    main()
