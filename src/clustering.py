import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm 

import umap
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import matplotlib.pyplot as plt
import seaborn as sns

# Logging
log_file = "../logs/clustering.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Config
embed_dir       = "../embeddings"
master_csv      = "../data/master_images.csv"
output_base     = "../experiments"
os.makedirs(output_base, exist_ok=True)

experiments = [
    {
        'name': 'exp_image_only',
        'use_caption': False,
        'use_ocr': False
    },
    {
        'name': 'exp2_image_caption',
        'use_caption': True,
        'use_ocr': False
    },
    {
        'name': 'exp3_image_ocr',
        'use_caption': False,
        'use_ocr': True
    },
    {
        'name': 'exp4_all',
        'use_caption': True,
        'use_ocr': True
    }
]

"""same parameter for all experiments"""
# UMAP
umap_n_components = 50 # for clustering
umap_n_neighbors = 30
umap_min_dist = 0.0
umap_metric = 'cosine'

# HDBSCAN 
hdb_min_cluster_size = 100
hdb_min_sample = 20

# k-Means
kmeans_k_range = range(20, 121, 10) # multiple values for testing

# Load embeddings
img_emb  = np.load(os.path.join(embed_dir, "embeddings_image.npy"))
cap_emb  = np.load(os.path.join(embed_dir, "embeddings_caption.npy"))
ocr_emb  = np.load(os.path.join(embed_dir, "embeddings_ocr.npy"))

df_master = pd.read_csv(master_csv)
n_samples = len(df_master)
logger.info(f"Loaded {n_samples} samples")

# Function to fuse embeddings
def fuse_embeddings(use_caption, use_ocr):
    vectors = [img_emb]
    if use_caption:
        vectors.append(cap_emb)
    if use_ocr:
        vectors.append(ocr_emb)
    
    fused = np.hstack(vectors)
    """scale if dimensions differ too much (rarely needed after L2 norm)"""
    # fused = StandardScaler().fit_transform(fused)
    return fused

# UMAP reduction
def reduce_with_umap(X):
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=42,
        verbose=True
    )
    X_red = reducer.fit_transform(X)
    # Also compute 2D for visualization
    reducer_2d = umap.UMAP(n_components=2, random_state=42, metric=umap_metric)
    X_2d = reducer_2d.fit_transform(X)
    return X_red, X_2d

# Clustering and metric
def run_hdbscan(X_red):
    clusterer = HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        min_samples=hdb_min_sample,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X_red)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    
    metrics = {
        "n_clusters": n_clusters,
        "noise_pct": noise_pct,
        "dbcv": clusterer.relative_validity_ if hasattr(clusterer, 'relative_validity_') else None
    }
    return labels, metrics

def run_kmeans(X_red):
    best_k = None
    best_sil = -1
    best_labels = None
    
    for k in kmeans_k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_red)
        sil = silhouette_score(X_red, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
    
    metrics = {
        "n_clusters": best_k,
        "silhouette": best_sil,
        "davies_bouldin": davies_bouldin_score(X_red, best_labels),
        "calinski_harabasz": calinski_harabasz_score(X_red, best_labels)
    }
    return best_labels, metrics

# Visualization
def save_umap_plot(X_2d, labels, exp_name, algo_name):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette='tab20', s=10, legend=False)
    plt.title(f"{exp_name} - {algo_name} clusters (UMAP 2D)")
    plt.tight_layout()
    out_path = os.path.join(output_base, exp_name, f"umap_{algo_name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot: {out_path}")

# Experiment loop
for exp in experiments:
    exp_name = exp["name"]
    logger.info(f"\n=== Starting {exp_name} ===")
    
    exp_dir = os.path.join(output_base, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Fuse
    fused = fuse_embeddings(exp["use_caption"], exp["use_ocr"])
    logger.info(f"Fused shape: {fused.shape}")
    
    # UMAP
    X_red, X_2d = reduce_with_umap(fused)
    np.save(os.path.join(exp_dir, "umap_reduced.npy"), X_red)
    np.save(os.path.join(exp_dir, "umap_2d.npy"), X_2d)
    
    # HDBSCAN
    labels_hdb, metrics_hdb = run_hdbscan(X_red)
    np.save(os.path.join(exp_dir, "labels_hdbscan.npy"), labels_hdb)
    with open(os.path.join(exp_dir, "metrics_hdbscan.json"), 'w') as f:
        json.dump(metrics_hdb, f, indent=2)
    save_umap_plot(X_2d, labels_hdb, exp_name, "hdbscan")
    
    # KMeans
    labels_km, metrics_km = run_kmeans(X_red)
    np.save(os.path.join(exp_dir, "labels_kmeans.npy"), labels_km)
    with open(os.path.join(exp_dir, "metrics_kmeans.json"), 'w') as f:
        json.dump(metrics_km, f, indent=2)
    save_umap_plot(X_2d, labels_km, exp_name, "kmeans")
    
    logger.info(f"{exp_name} finished. Metrics saved.")

logger.info("All experiments completed.")