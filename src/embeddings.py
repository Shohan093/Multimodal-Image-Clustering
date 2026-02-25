import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm 
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Logging
log_file = "../logs/embeddings.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Config
master_csv = "../data/master_images.csv"
ocr_json = "../data/ocr_results.json"
captions_jsonl = "../data/captions.jsonl"
output_dir = "../embeddings"
os.makedirs(output_dir, exist_ok=True)


# Models
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer('all-MiniLM-L6-v2', device='GPU')
logger.info("SentenceTransformer forced to CPU to avoid CUDA OOM")
batch_size_img = 8
batch_size_text = 64  

# Load & align data
df_master = pd.read_csv(master_csv)
with open(ocr_json, 'r', encoding='utf-8') as f:
    ocr_list = json.load(f)
ocr_df = pd.DataFrame(ocr_list).rename(columns={'text': 'ocr_text'})

captions = {}
with open(captions_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line.strip())
        captions[d['image_path']] = d['caption']

df = df_master.merge(ocr_df, on='image_path', how='left')
df['caption'] = df['image_path'].map(captions).fillna("")
logger.info(f"Total rows after merge: {len(df)}")

# Simple OCR clean (remove extreme garbage)
def clean_ocr(t):
    if not isinstance(t, str) or len(t) < 5:
        return ""
    # Remove repeated chars >3 times, strip junk
    import re
    t = re.sub(r'(.)\1{3,}', r'\1\1', t)
    t = re.sub(r'[^a-zA-Z0-9\s\.,!?\'\"-]', '', t).strip()
    return t if len(t) > 10 else ""

df['ocr_clean'] = df['ocr_text'].apply(clean_ocr)
logger.info(f"OCR cleaning done. { (df['ocr_clean'] == '[no_text]').sum() } rows have no usable text")

# Progress / resume
progress_file = os.path.join(output_dir, "progress.txt")
start_idx = 0
if os.path.exists(progress_file):
    with open(progress_file, 'r') as pf:
        start_idx = int(pf.read().strip()) + 1
    logger.info(f"Resuming from {start_idx}")

# Prepare arrays (or append mode if resume)
n = len(df)
img_emb_path = os.path.join(output_dir, "embeddings_image.npy")
cap_emb_path = os.path.join(output_dir, "embeddings_caption.npy")
ocr_emb_path = os.path.join(output_dir, "embeddings_ocr.npy")

# Dimensions
IMG_DIM = 512
TEXT_DIM = 384

# If files exist, load partial; else create zeros
if os.path.exists(img_emb_path):
    img_emb = np.load(img_emb_path)
else:
    img_emb = np.zeros((n, 512), dtype=np.float32)

if os.path.exists(cap_emb_path):
    cap_emb = np.load(cap_emb_path)
    logger.info(f"Loaded existing caption embeddings: {cap_emb.shape}")
else:
    cap_emb = np.zeros((n, TEXT_DIM), dtype=np.float32)
    logger.info("Created new caption embeddings array")

if os.path.exists(ocr_emb_path):
    ocr_emb = np.load(ocr_emb_path)
    logger.info(f"Loaded existing OCR embeddings: {ocr_emb.shape}")
else:
    ocr_emb = np.zeros((n, TEXT_DIM), dtype=np.float32)
    logger.info("Created new OCR embeddings array")

# Resume logic
start_idx = 0
if os.path.exists(progress_file):
    try:
        with open(progress_file, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                start_idx = int(content) + 1
                logger.info(f"Resuming from index {start_idx}")
    except:
        logger.warning("Invalid progress file → starting from 0")

# Image embeddings
for i in tqdm(range(start_idx, n, batch_size_img), desc="Image Embeddings", initial=start_idx, total=n):
    end = min(i + batch_size_img, n)
    batch_df = df.iloc[i:end]
    paths = batch_df['full_path'].to_list()

    images = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            images.append(img)
        except Exception as e:
            logger.warning(f"Cannot open image {p}: {e}")
            images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

    try:
        inputs = clip_processor(images=images, return_tensors='pt').to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats.cpu().numpy()
            
            # L2 normalization
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = np.divide(feats, norms, where=norms != 0)
        img_emb[i:end] = feats
        np.save(img_emb_path, img_emb)
    except Exception as e:
        logger.error(f"Batch failed at {i}-{end}: {e}")

    # Update progress
    with open(progress_file, 'w') as f:
        f.write(str(end - 1))

logger.info("Image embeddings completed / resumed")

# Captions embeddings
torch.cuda.empty_cache()
for i in tqdm(range(0, n, batch_size_text), desc="Caption embeddings"):
    end = min(i + batch_size_text, n)
    texts = df['caption'].iloc[i:end].tolist()

    # Replace empty captions
    texts = [t if t.strip() else "[no_caption]" for t in texts]

    try:
        embeds = text_model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        cap_emb[i:end] = embeds
        np.save(cap_emb_path, cap_emb)
    except Exception as e:
        logger.error(f"Caption batch failed {i}-{end}: {e}")

logger.info("Caption embeddings completed")

# OCR Embeddings
torch.cuda.empty_cache()
for i in tqdm(range(0, n, batch_size_text), desc="OCR embeddings"):
    end = min(i + batch_size_text, n)
    texts = df['ocr_clean'].iloc[i:end].tolist()

    # Replace placeholder
    texts = [t if t != "[no_text]" else "" for t in texts]

    try:
        embeds = text_model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        ocr_emb[i:end] = embeds
        np.save(ocr_emb_path, ocr_emb)
    except Exception as e:
        logger.error(f"OCR batch failed {i}-{end}: {e}")

logger.info("OCR embeddings completed")

# Final save (just in case)
np.save(img_emb_path, img_emb)
np.save(cap_emb_path, cap_emb)
np.save(ocr_emb_path, ocr_emb)

logger.info("All embeddings saved successfully.")
logger.info(f"Files created:")
logger.info(f"  • {img_emb_path}")
logger.info(f"  • {cap_emb_path}")
logger.info(f"  • {ocr_emb_path}")