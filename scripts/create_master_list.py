# Import libraries
import os
import hashlib
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Config
images_dir = "../data/images"
output_csv = "../data/master_images.csv"
log_file = "../logs/image_cleaning.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Collect all the image paths
extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
image_paths = []
for root, _, files in os.walk(images_dir):
    for f in files:
        if f.lower().endswith(extensions):
            image_paths.append(os.path.join(root, f))

# print(f"Found {len(image_paths)} potential images")

# Basic validation + dedup by hash
valid_rows = []
seen_hash = set()
skipped_count = 0
for path in tqdm(image_paths, desc="Validating images", colour='green'):
    try:
        with Image.open(path) as img:
            img.verify()  # Check if image is corrupted

        # Reopen image for processing
        with Image.open(path) as img:
            img = img.convert('RGB')
            hash_val = hashlib.md5(img.tobytes()).hexdigest()

        if hash_val in seen_hash:
            logger.debug(f"Duplicate image skipped: {path}")
            skipped_count += 1
            continue
        seen_hash.add(hash_val)
        valid_rows.append({
            "image_path": os.path.relpath(path, start=images_dir),
            "image_id": len(valid_rows),
            "full_path": path
        })

    except Exception as e:
        logger.warning(f"Skipping invalid/corrupted image: {path} | Reason: {str(e)}")
        skipped_count += 1


df = pd.DataFrame(valid_rows)
df.to_csv(output_csv, index=False)

logger.info(f"Clean master list saved: {len(df)} valid images")
logger.info(f"Total skipped (corrupted + duplicates): {skipped_count}")
logger.info(f"Output file: {output_csv}")
logger.info(f"Log file: {log_file}")