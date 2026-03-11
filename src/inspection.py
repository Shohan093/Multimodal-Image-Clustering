import os
import json
import shutil
import argparse
import logging
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="exp_image_only")
parser.add_argument("--result_type", default="kmeans")
parser.add_argument("--top_n_clusters", type=int, default=10)
parser.add_argument("--n_samples_per_cluster", type=int, default=5)
parser.add_argument("--embed_dir", default="embeddings")  # path to embeddings if centroid sampling
args = parser.parse_args()

# Logging
log_dir = os.path.join("experiments", args.exp_name)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inspection_{args.result_type}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # also print to console

# Config
master_csv    = "data/master_images.csv"
captions_file = "data/captions.jsonl"
exp_dir       = os.path.join("experiments", args.exp_name)
labels_path   = os.path.join(exp_dir, f"labels_{args.result_type}.npy")
output_root   = os.path.join(exp_dir, f"cluster_results_{args.result_type}")
os.makedirs(output_root, exist_ok=True)

logger.info(f"Experiment: {args.exp_name} | Result type: {args.result_type}")
logger.info(f"Top {args.top_n_clusters} clusters | {args.n_samples_per_cluster} samples per cluster")

# Load Data
logger.info("Loading master CSV...")
df = pd.read_csv(master_csv)

logger.info("Loading captions...")
captions = {}
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line.strip())
            captions[d['image_path']] = d.get('caption', '[no caption]')
df['caption'] = df['image_path'].map(captions).fillna('[no caption]')

logger.info("Loading cluster labels...")
labels = np.load(labels_path)
df['cluster'] = labels
n_clusters = len(set(labels))
logger.info(f"Loaded {len(df)} images | {n_clusters} clusters")

# Load embeddings (for centroid-based sampling)
use_centroid_sampling = False
fused_path = os.path.join(args.embed_dir, "fused.npy")
if os.path.exists(fused_path):
    try:
        fused_emb = np.load(fused_path)
        fused_emb = normalize(fused_emb)
        use_centroid_sampling = True
        logger.info("Loaded fused embeddings for centroid-based sampling")
    except Exception as e:
        logger.warning(f"Failed to load fused embeddings: {e}")

# Save full CSV with clusters
results_csv = os.path.join(output_root, "all_images_with_cluster.csv")
df.to_csv(results_csv, index=False)
logger.info(f"Full CSV saved → {results_csv}")

# Find largest clusters
cluster_sizes = Counter(df['cluster'])
largest = cluster_sizes.most_common(args.top_n_clusters)
logger.info(f"Processing top {args.top_n_clusters} clusters...")

for rank, (cluster_id, size) in enumerate(largest, 1):
    cluster_folder = os.path.join(output_root, f"cluster_{cluster_id:03d}")
    os.makedirs(cluster_folder, exist_ok=True)

    cluster_df = df[df['cluster'] == cluster_id]

    # Centroid-based sampling
    if use_centroid_sampling:
        indices = cluster_df.index.to_list()
        cluster_emb = fused_emb[indices]
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        distances = cosine_distances(cluster_emb, centroid).flatten()
        # pick n_samples closest to centroid
        selected_idx = np.argsort(distances)[:min(args.n_samples_per_cluster, len(cluster_df))]
        samples = cluster_df.iloc[selected_idx]
    else:
        # fallback to random sampling
        samples = cluster_df.sample(
            n=min(args.n_samples_per_cluster, len(cluster_df)), 
            random_state=42
        )

    logger.info(f"[{rank:2d}] Cluster {cluster_id:3d} → {size:5d} images | Saving {len(samples)} samples")

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
                "filename": dest_name,
                "cluster_id": cluster_id,
                "cluster_size": size,
                "cluster_rank": rank
            })
        except Exception as e:
            logger.warning(f"Failed to copy {src_path}: {e}")

    # Save metadata
    meta_path = os.path.join(cluster_folder, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved → {meta_path}")

logger.info("All clusters processed successfully!")
logger.info(f"Results saved in: {output_root}")
logger.info(f"   Full CSV: all_images_with_cluster.csv")
logger.info(f"   Per-cluster folders with images + metadata.json")
