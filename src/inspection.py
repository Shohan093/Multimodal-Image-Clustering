import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
from PIL import Image

# Config
master_csv = "../data/master_images.csv"
captions_file = "../data/captions.jsonl"
exp_dir = "../experiments/exp2_image_caption"
label_path = os.path.join(exp_dir, "labels_kmeans.npy")

# Load data
df = pd.read_csv(master_csv)
captions = {}
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line.strip())
            captions[d['image_path']] = d.get('caption', '[no caption]')

df['caption'] = df['image_path'].map(captions).fillna('[no caption]')

# Load Kmeans label
labels = np.load(label_path)
print(f"Loaded {len(labels)} labels. Unique clusters: {len(set(labels))}")

df['cluster'] = labels

# Find largest clusters
cluster_size = Counter(df['cluster'])
largest_clusters = cluster_size.most_common(10) # Top 10 clusters
print("\nTop 10 largest clusters:")
for cluster_id, size in largest_clusters:
    print(f"Cluster {cluster_id:3d}: {size:5d} memes")

# Function to display samples from clusters
def show_cluster_samples(cluster_id, n_samples=5):
    cluster_df = df[df['cluster'] == cluster_id]
    if len(cluster_df) == 0:
        print(f"Cluster {cluster_id} is empty.")
        return
    
    # Randomly sample n images
    samples = cluster_df.sample(min(n_samples, len(cluster_df)))
    
    print(f"\nCluster {cluster_id} ({len(cluster_df)} items):")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4))
    if len(samples) == 1:
        axes = [axes]
    
    for i, (_, row) in enumerate(samples.iterrows()):
        try:
            img = Image.open(row['full_path'])
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Image {i+1}", fontsize=10)
            
            caption = row['caption'][:100] + "..." if len(row['caption']) > 100 else row['caption']
            print(f"  Sample {i+1}: {caption}")
        except Exception as e:
            print(f"  Could not load image: {row['full_path']} → {e}")
    
    plt.tight_layout()
    plt.show()

# Cluster inspection loop
for cluster_id, _ in largest_clusters[:10]:
    show_cluster_samples(cluster_id, n_samples=4)