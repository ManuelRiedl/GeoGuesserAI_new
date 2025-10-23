import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIG ---
BASE_DIR = "data/images_cropped"
OUTPUT_DIR = "project_1"
TRAIN_RATIO = 0.85
IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
MODEL_NAME = "yolov8l.pt"
EXPERIMENT_NAME = "all_classes"

# --- Class Mapping ---
# Map original class IDs (as strings) to general classes
# Example: specific bollards -> 0, guardrails -> 1, chevrons -> 2
"""
CLASS_MAPPING = {
    '0': '0',  # austria_bollard
    '1': '0',  # germany_bollard
    '2': '0',  # luxenburg_bollard
    '5': '0',  # portugal_bollard
    '10': '0',  # spain_bollard
    '11': '0',  # france_bollard
    '13': '0',  # italy_bollard
    '14': '0',  # slovenia_bollard
}
# The corresponding class names (order must match new class IDs 0, 1, 2...)
CLASS_NAMES = [
    "bollard",
]
"""



CLASS_MAPPING = {
    '0': '0',  # austria_bollard
    '1': '1',  # germany_bollard
    '2': '2',  # luxenburg_bollard
    '3': '3',  # luxenburg_bollard_reflector
    '4': '4',  # germany_bollard_reflector
    '5': '5',  # portugal_bollard
    '10': '6',  # spain_bollard
    '11': '7',  # france_bollard
    '12': '8',  # italy_bollard
    '13': '9',  # slovenia_bollard
    '14': '10',  # slovenia_bollard_reflector
}
# The corresponding class names (order must match new class IDs 0, 1, 2...)
CLASS_NAMES = [
    "austria_bollard",
    "germany_bollard",
    "luxenburg_bollard",
    "luxenburg_bollard_reflector",
    "germany_bollard_reflector",
    "portugal_bollard",
    "spain_bollard",
    "france_bollard",
    "italy_bollard",
    "slovenia_bollard",
    "slovenia_bollard_reflector"
]


# --- Datasets to include ---
DATASETS = {
    "austrian_bollard": os.path.join(BASE_DIR, "austria", "austria_bollards"),
    "german_bollard": os.path.join(BASE_DIR, "germany", "germany_bollards"),
    "luxenburg_bollard": os.path.join(BASE_DIR, "luxenburg", "luxenburg_bollards"),
    "portugal_bollards": os.path.join(BASE_DIR, "portugal", "portugal_bollards"),
    "spain_bollards": os.path.join(BASE_DIR, "spain", "spain_bollards"),
    "france_bollards": os.path.join(BASE_DIR, "france", "france_bollards"),
    "italy_bollards": os.path.join(BASE_DIR, "italy", "italy_bollards"),
    "slowenia_bollards": os.path.join(BASE_DIR, "slowenia", "slowenia_bollards"),
}

# --- Set up output folders ---
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)
# --- Collect files ---
def get_class_files(path):
    images_folder = os.path.join(path, "images")
    labels_folder = os.path.join(path, "labels")
    img_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".png"))]

    files = []
    for img_file in img_files:
        files.append({
            "img_path": os.path.join(images_folder, img_file),
            "label_path": os.path.join(labels_folder, Path(img_file).stem + ".txt"),
            "filename": img_file,
        })
    return files

train_files = []
val_files = []

for _, path in DATASETS.items():
    class_files = get_class_files(path)
    random.shuffle(class_files)
    split_idx = int(len(class_files) * TRAIN_RATIO)
    train_files.extend(class_files[:split_idx])
    val_files.extend(class_files[split_idx:])

# --- Copy files and remap labels ---
def copy_and_fix_labels(files, split):
    print(f"\nCopying {len(files)} files to {split} set...")
    for file_info in tqdm(files):
        dst_img = os.path.join(OUTPUT_DIR, "images", split, file_info["filename"])
        dst_label = os.path.join(OUTPUT_DIR, "labels", split, Path(file_info["filename"]).stem + ".txt")

        shutil.copyfile(file_info["img_path"], dst_img)

        fixed_lines = []

        if os.path.exists(file_info["label_path"]):
            with open(file_info["label_path"], "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    original_class = parts[0]
                    if original_class in CLASS_MAPPING:
                        parts[0] = CLASS_MAPPING[original_class]
                        fixed_lines.append(" ".join(parts) + "\n")
                    # else: silently ignore class
        else:
            print(f"⚠️ Missing label file: {file_info['label_path']}")

        # Always write the label file — even if it's empty
        with open(dst_label, "w") as f:
            f.writelines(fixed_lines)


copy_and_fix_labels(train_files, "train")
copy_and_fix_labels(val_files, "val")

# --- Write data.yaml for YOLOv8 ---
yaml_path = "data_1.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")

print(f"\n✅ Dataset preparation complete. YOLO config saved to: {yaml_path}")



import os
import shutil
from pathlib import Path

# --- Config ---
VAL_IMAGE_FOLDER = f"{OUTPUT_DIR}/images/val"  # contains cropped validation images
VAL_LABEL_FOLDER = f"{OUTPUT_DIR}/labels/val"
ORIGINAL_DATA_ROOT = "data/images_unlabeled"  # where full original images and labels are
OUTPUT_IMG_DIR = f"{OUTPUT_DIR}/original_validation/images"
OUTPUT_LABEL_DIR = f"{OUTPUT_DIR}/original_validation/labels"

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
    if "_crop" in file:
        base = file.split("_crop")[0]
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
