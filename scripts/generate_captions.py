# Import libraries
import os
import json
import logging
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import transformers

transformers.logging.set_verbosity_error()

# Config
master_csv = "../data/master_images.csv"
progress_file = "../data/caption_progress.txt"
caption_file = "../data/captions.jsonl"
log_file = "../logs/caption_generation.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Model configuration
model_name = "nlpconnect/vit-gpt2-image-captioning"
batch_size = 4
max_new_tokens = 32
sample_limit = None

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Auto detect GPU
logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

# Load model
try:
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device)
    
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"Model loaded successfully: {model_name}")
except Exception as e:
    logger.error(f"Failed to load PyTorch model: {e}")
    raise

# Load data
df = pd.read_csv(master_csv)
if sample_limit is not None:
    df = df.sample(sample_limit).reset_index(drop=True)
logger.info(f"Total images to process: {len(df)}")

start_index = 0
already_done = set()

# Check existing output
if os.path.exists(caption_file):
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                already_done.add(data["image_path"])

if os.path.exists(progress_file):
    try:
        with open(progress_file, "r") as f:
            content = f.read().strip()
            if content.isdigit():
                start_index = int(content) + 1  # continue from NEXT
                logger.info(f"Resuming from index {start_index}")
    except:
        logger.warning("Progress file unreadable → starting from 0")

# Process
process_count = 0
for idx in tqdm(range(start_index, len(df)), desc="Generating captions", initial=start_index, total=len(df)):
    row = df.iloc[idx]
    rel_path = row['image_path']
    full_path = row['full_path']

    # Skip if already done
    if rel_path in already_done:
        process_count += 1
        continue

    try:
        img = Image.open(full_path).convert('RGB')
        pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=max_new_tokens
            )
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # Append immediately
        with open(caption_file, 'a', encoding='utf-8') as f:
            json.dump({
                'image_path': rel_path,
                'caption': caption,
                'index': idx
            }, f, ensure_ascii=False)
            f.write('\n')

        process_count += 1

        # Update progress after every image
        with open(progress_file, 'w') as f:
            f.write(str(idx))

    except Exception as e:
        logger.warning(f"Failed on {rel_path} (index {idx}): {str(e)}")
        # Still mark as progessed
        with open(progress_file, 'w') as f:
            f.write(str(idx))

        # optionally add an empty camption
        with open(caption_file, "a", encoding="utf-8") as f:
            json.dump({
                "image_path": rel_path,
                "caption": "",
                "index": idx,
                "error": str(e)
            }, f, ensure_ascii=False)
            f.write("\n")

logger.info(f"Finished. Processed {process_count} new captions.")
logger.info(f"Output (JSONL): {caption_file}")
logger.info(f"Progress saved at: {progress_file}")
logger.info(f"Check log for any failures: {log_file}")