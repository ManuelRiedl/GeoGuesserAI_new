import os
import shutil
from pathlib import Path

# --- Config ---
VAL_IMAGE_FOLDER = "project_1/images/val"  # contains cropped validation images
VAL_LABEL_FOLDER = "project_1/labels/val"
ORIGINAL_DATA_ROOT = "data/images_unlabeled"  # where full original images and labels are
OUTPUT_IMG_DIR = "original_validation/images"
OUTPUT_LABEL_DIR = "original_validation/labels"

# Subfolder structure inside ORIGINAL_DATA_ROOT
DATASETS = {
    "austria": os.path.join(ORIGINAL_DATA_ROOT, "austria", "austria_bollards"),
    "germany": os.path.join(ORIGINAL_DATA_ROOT, "germany", "germany_bollards"),
    "luxenburg": os.path.join(ORIGINAL_DATA_ROOT, "luxenburg", "luxenburg_bollards"),
    "portugal": os.path.join(ORIGINAL_DATA_ROOT, "portugal", "portugal_bollards"),
    "spain": os.path.join(ORIGINAL_DATA_ROOT, "spain", "spain_bollards"),
    "france": os.path.join(ORIGINAL_DATA_ROOT, "france", "france_bollards"),
    "italy": os.path.join(ORIGINAL_DATA_ROOT, "italy", "italy_bollards"),
    "slowenia": os.path.join(ORIGINAL_DATA_ROOT, "slowenia", "slowenia_bollards"),
}

# Create output dirs
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Helper: check for .jpg or .png file
def find_image(base_name):
    for ext in ['.jpg', '.png']:
        for folder in DATASETS.values():
            candidate = os.path.join(folder, "images", base_name + ext)
            if os.path.exists(candidate):
                return candidate
    return None

def find_label(base_name):
    for folder in DATASETS.values():
        candidate = os.path.join(folder, "labels", base_name + ".txt")
        if os.path.exists(candidate):
            return candidate
    return None

# Collect original image names from cropped filenames
original_names = set()
for file in os.listdir(VAL_IMAGE_FOLDER):
    if "_crop_" in file:
        base = file.split("_crop_")[0]
        original_names.add(base)

print(f"🔍 Found {len(original_names)} original base names from validation crops.")

# Copy files
for name in original_names:
    img_src = find_image(name)
    label_src = find_label(name)

    if img_src:
        shutil.copy(img_src, os.path.join(OUTPUT_IMG_DIR, os.path.basename(img_src)))
    else:
        print(f"⚠️ Image not found for: {name}")

    if label_src:
        shutil.copy(label_src, os.path.join(OUTPUT_LABEL_DIR, os.path.basename(label_src)))
    else:
        print(f"⚠️ Label not found for: {name}")

print(f"\n✅ Copied all available original validation images and labels to `original_validation/`")
