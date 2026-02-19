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