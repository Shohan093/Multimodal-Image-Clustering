import os
import json
import shutil
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image

# Config
exp_name = "exp_image_only"
result_type = "kmeans"
n_samples_per_cluster = 5
top_n_clusters = 10

# Paths (relative to your project root)
master_csv     = "data/master_images.csv"
captions_file  = "data/captions.jsonl"
exp_dir        = f"experiments/{exp_name}"
labels_path    = os.path.join(exp_dir, f"labels_{result_type}.npy")

# Output folder
output_root = os.path.join(exp_dir, f"cluster_results_{result_type}")
os.makedirs(output_root, exist_ok=True)

print("Loading data...")

# Load master data
df = pd.read_csv(master_csv)

# Load captions
captions = {}
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line.strip())
            captions[d['image_path']] = d.get('caption', '[no caption]')

df['caption'] = df['image_path'].map(captions).fillna('[no caption]')

# Load cluster labels
labels = np.load(labels_path)
df['cluster'] = labels

print(f"Loaded {len(df)} images | {len(set(labels))} clusters")

# Save full results CSV
results_csv = os.path.join(output_root, "all_images_with_cluster.csv")
df.to_csv(results_csv, index=False)
print(f"Full results saved → {results_csv}")

# Find largest clusters
cluster_sizes = Counter(df['cluster'])
largest = cluster_sizes.most_common(top_n_clusters)

print(f"\nProcessing top {top_n_clusters} clusters...\n")

for rank, (cluster_id, size) in enumerate(largest, 1):
    cluster_folder = os.path.join(output_root, f"cluster_{cluster_id:03d}")
    os.makedirs(cluster_folder, exist_ok=True)

    cluster_df = df[df['cluster'] == cluster_id]
    samples = cluster_df.sample(n=min(n_samples_per_cluster, len(cluster_df)), random_state=42)

    print(f"[{rank:2d}] Cluster {cluster_id:3d} → {size:5d} images | Saving {len(samples)} samples")

    metadata = []

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        src_path = row['full_path']
        dest_name = f"sample_{i:02d}_{os.path.basename(src_path)}"
        dest_path = os.path.join(cluster_folder, dest_name)

        try:
            shutil.copy2(src_path, dest_path)
            metadata.append({
                "sample_id": i,
                "image_path": row['image_path'],
                "caption": row['caption'],
                "filename": dest_name
            })
        except Exception as e:
            print(f"    Failed to copy {src_path}: {e}")

    # Save metadata for this cluster
    with open(os.path.join(cluster_folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\nAll done!")
print(f"Everything saved in: {output_root}")
print(f"   Full CSV: all_images_with_cluster.csv")
print(f"   Per-cluster folders with images + metadata.json")
