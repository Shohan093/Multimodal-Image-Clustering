import os
import json
import logging
import pandas as pd
from tqdm import tqdm
import pytesseract
import cv2

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Config
master_csv = "../data/master_images.csv"
ocr_json_path = "../data/ocr_results.json"
progress_file = "../data/ocr_progress.txt"
log_file = "../logs/ocr_extraction.log"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load CSV
df = pd.read_csv(master_csv)
logger.info(f"Loaded {len(df)} images from master list")

# Resume index
start_index = 0
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        content = f.read().strip()
        if content.isdigit():
            start_index = int(content)
            logger.info(f"Resuming from index {start_index}")
        else:
            logger.warning("Progress file is empty or corrupted. Starting from 0.")


# Load previous results
results = []
if os.path.exists(ocr_json_path):
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

processed = start_index
empty_ocr = 0

# Image processing function
def preprocess_for_meme(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    return thresh

for i in tqdm(range(start_index, len(df)), desc="Extracting OCR", colour="green"):
    row = df.iloc[i]
    rel_path = row["image_path"]
    full_path = row["full_path"]

    try:
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError("Image not readable")

        processed_img = preprocess_for_meme(img)
        text1 = pytesseract.image_to_string(
            processed_img,
            lang="eng+ben",
            config="--psm 6"
        ).strip()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text2 = pytesseract.image_to_string(
            gray,
            lang="eng+ben",
            config="--psm 6"
        ).strip()
        text = text1 if len(text1) >= len(text2) else text2
        if not text:
            empty_ocr += 1

        results.append({
            "image_path": rel_path,
            "text": text
        })

    except Exception as e:
        logger.warning(f"OCR failed for {full_path}: {str(e)}")
        results.append({
            "image_path": rel_path,
            "text": ""
        })

    processed += 1

    # Save progress after every image
    with open(progress_file, "w") as f:
        f.write(str(processed))

    # Save partial results every 100 images
    if processed % 100 == 0:
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

# Final save
with open(ocr_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logger.info("OCR extraction finished")
logger.info(f"Processed {processed} images")
logger.info(f"Images with no detected text: {empty_ocr}")
logger.info(f"Output file: {ocr_json_path}")
