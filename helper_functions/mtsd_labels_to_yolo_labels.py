import os
import json
from tqdm import tqdm

# Paths
ANNOTATION_DIR = "../data/mtsd/labels_original/mtsd_v2_fully_annotated/annotations"
IMAGE_DIR = "../data/mtsd/images"
OUTPUT_LABEL_DIR = "../data/mtsd/labels_yolo"
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
CLASS_NAMES_PATH = "../data/mtsd/classes.txt"

if not os.path.exists(ANNOTATION_DIR):
    raise FileNotFoundError(f"Annotation directory not found: {ANNOTATION_DIR}")
# Step 1: Collect all unique labels
label_set = set()
annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json")]

for filename in tqdm(annotation_files, desc="Scanning labels"):
    with open(os.path.join(ANNOTATION_DIR, filename), "r") as f:
        data = json.load(f)
        for obj in data.get("objects", []):
            label_set.add(obj["label"])

# Step 2: Build class map
class_map = {label: idx for idx, label in enumerate(sorted(label_set))}

# Step 3: Convert annotations to YOLO format
for filename in tqdm(annotation_files, desc="Converting annotations"):
    with open(os.path.join(ANNOTATION_DIR, filename), "r") as f:
        data = json.load(f)

    img_name = filename.replace(".json", ".jpg")
    label_path = os.path.join(OUTPUT_LABEL_DIR, filename.replace(".json", ".txt"))

    w, h = data["width"], data["height"]

    with open(label_path, "w") as out:
        for obj in data.get("objects", []):
            label = obj["label"]
            if label not in class_map:
                continue

            bbox = obj["bbox"]
            x_center = ((bbox["xmin"] + bbox["xmax"]) / 2) / w
            y_center = ((bbox["ymin"] + bbox["ymax"]) / 2) / h
            width = (bbox["xmax"] - bbox["xmin"]) / w
            height = (bbox["ymax"] - bbox["ymin"]) / h

            out.write(f"{class_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

with open(CLASS_NAMES_PATH, "w") as f:
    for label in sorted(class_map.keys(), key=lambda x: class_map[x]):
        f.write(f"{label}\n")