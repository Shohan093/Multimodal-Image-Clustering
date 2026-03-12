# Multimodal Data Mining Framework for Semantic Image Clustering

## Overview

This project presents a **multimodal data mining framework** for clustering internet memes using both visual and textual modalities. The goal is to discover **semantic structure in large meme datasets** by combining image features, caption text, and OCR-extracted text.

Memes are inherently multimodal: their meaning often emerges from the interaction between an image template and textual content. This project explores how combining multiple modalities can improve unsupervised clustering performance.

The framework processes approximately **23,000 memes** collected from several public datasets and evaluates clustering quality under different modality configurations.

---

## Dataset Sources

The dataset used in this study was compiled from multiple public meme repositories:

1. Reddit Memes Dataset  
https://www.kaggle.com/datasets/sayangoswami/reddit-memes-dataset

2. Memotion Dataset 7K  
https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k

3. Memes Classified and Labelled  
https://www.kaggle.com/datasets/gmorinan/memes-classified-and-labelled

4. Meme Dataset  
https://www.kaggle.com/datasets/nikitricky/memes

All datasets were merged, cleaned, and deduplicated to produce a unified corpus for analysis.

---

## Project Pipeline

The framework follows a structured pipeline.

### 1. Image Cleaning and Dataset Preparation

Script: `create_master_list.py`

- Scans dataset folders
- Removes corrupted images
- Detects duplicates using MD5 hashing
- Generates a master CSV containing valid image paths

---

### 2. OCR Text Extraction

Script: `ocr_extraction.py`

Many memes contain important semantic information embedded directly in the image as text. To capture this information, Optical Character Recognition (OCR) is applied to each image.

OCR is used to extract visible text from the meme images.

The extracted text is stored in JSON format for later processing.

Example output structure:

```json
{
"image_path": "image_001.jpg",
"text": "When the code works on the first try"
}
```

After extraction, the OCR text undergoes **basic cleaning** to remove noisy characters, repeated symbols, and very short fragments that do not contain meaningful information.

This cleaned OCR text is later converted into semantic embeddings.

---

### 3. Image Caption Generation

Script: `generate_captions.py` (or the script used in your pipeline)

Since many memes rely on both **visual context and textual humor**, captions are generated for each image to capture its semantic meaning.

An image captioning model `nlpconnect/vit-gpt2-image-captioning` analyzes the visual content of the meme and generates a natural language description of the image.

Example caption output:

```jsonl
{
"image_path": "image_001.jpg", "caption": "A distracted boyfriend meme where the man looks at another woman while his girlfriend looks shocked."
}
```

### 4. Multimodal Embedding Extraction

Script: `embeddings.py`

Three types of embeddings are extracted.

**Image embeddings**
- Extracted using the CLIP vision encoder

**Caption embeddings**
- Generated from image captions using a SentenceTransformer model

**OCR embeddings**
- Text extracted from meme images using OCR and encoded using SentenceTransformer

---

### 5. Multimodal Feature Fusion

Embeddings from different modalities are combined using **weighted feature fusion**.

Example:
`fused = [0.6 × image, 0.25 × caption, 0.15 × OCR]`


This weighting prioritizes visual information while still incorporating textual context.

---

### 6. Dimensionality Reduction

High-dimensional embeddings are reduced using **UMAP** before clustering.

UMAP configuration:
- `n_components = 50`
- `n_neighbors = 30`
- `min_dist = 0.0`
- `metric = cosine`

A separate **2D UMAP projection** is generated for visualization.

---

### 7. Clustering Algorithms

Two clustering approaches are evaluated.

**K-Means**
- Cluster numbers searched between 20 and 120
- Best value selected using silhouette score

**HDBSCAN**
- Density-based clustering
- Automatically determines cluster count
- Identifies noise points

Evaluation metrics include:
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score

---

## Experiments

Four experiments were conducted.

| Experiment | Modalities Used |
|------------|----------------|
| exp_image_only | Image embeddings |
| exp2_image_caption | Image + Caption |
| exp3_image_ocr | Image + OCR |
| exp4_all | Image + Caption + OCR |

---

### Key Observations

- Caption text significantly improves semantic clustering
- OCR text introduces noise due to recognition errors
- Multimodal fusion improves performance when modalities are informative

---

## Cluster Inspection

Script: `inspection.py`

This script:

- Identifies the largest clusters
- Selects representative memes using centroid-based sampling
- Exports sample images and metadata

---

The pipeline is logically:  
```
Dataset collection  
      ↓
Image cleaning  
      ↓
OCR extraction
      ↓
Caption generation
      ↓
Embedding extraction
      ↓
Multimodal fusion
      ↓
UMAP reduction
      ↓
Clustering
      ↓
Cluster inspection (Optional)
```
---
