import os

PBF_DIR = "osm_pbf"
pbf_files = [f for f in os.listdir(PBF_DIR) if f.endswith(".osm.pbf")]

# Dictionary: filename -> size in MB
pbf_sizes = {f: os.path.getsize(os.path.join(PBF_DIR, f)) / (1024*1024) for f in pbf_files}

# Sort descending by size (optional, can help greedy split)
pbf_sizes = dict(sorted(pbf_sizes.items(), key=lambda x: x[1], reverse=True))



import random

# Shuffle states
files = list(pbf_sizes.keys())
random.seed(42)
random.shuffle(files)

total_size = sum(pbf_sizes.values())
train_size = 0
train_files = []
test_files = []

for f in files:
    if train_size / total_size < 0.75:
        train_files.append(f)
        train_size += pbf_sizes[f]
    else:
        test_files.append(f)

print(f"Training states ({len(train_files)} files, {train_size:.1f} MB):")
print(train_files)
print(f"\nTest states ({len(test_files)} files, {total_size - train_size:.1f} MB):")
print(test_files)