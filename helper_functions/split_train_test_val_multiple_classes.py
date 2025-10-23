import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIG
BASE_DIR = "data/images_cropped"
OUTPUT_DIR = "project_1"  # final YOLO dataset root
CLASSES = ["austria_bollard","germany_bollard","luxenburg_bollard","luxenburg_bollard_reflector","germany_bollard_reflector"]
TRAIN_RATIO = 0.8

# Source folders
DATASETS = {
    "austrian_bollard": os.path.join(BASE_DIR, "austria", "austria_bollards"),
    "german_bollard": os.path.join(BASE_DIR, "germany", "germany_bollards"),
    "luxenburg_bollard": os.path.join(BASE_DIR, "luxenburg", "luxenburg_bollards"),
}

# Create output folder structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

def get_class_files(class_name, path):
    images_folder = os.path.join(path, "images")
    labels_folder = os.path.join(path, "labels")
    img_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".png"))]

    files = []
    for img_file in img_files:
        files.append({
            "class_name": class_name,
            "img_path": os.path.join(images_folder, img_file),
            "label_path": os.path.join(labels_folder, Path(img_file).stem + ".txt"),
            "filename": img_file,
        })
    return files

# Split each class separately, then combine
train_files = []
val_files = []

for class_idx, (class_name, path) in enumerate(DATASETS.items()):
    class_files = get_class_files(class_name, path)
    random.shuffle(class_files)
    split_idx = int(len(class_files) * TRAIN_RATIO)
    train_split = class_files[:split_idx]
    val_split = class_files[split_idx:]

    # Add class_idx to each dict for later label fix
    for f in train_split:
        f['class_idx'] = class_idx
    for f in val_split:
        f['class_idx'] = class_idx

    train_files.extend(train_split)
    val_files.extend(val_split)

def copy_and_fix_labels(files, split):
    print(f"\nCopying {len(files)} files to {split} set...")
    for file_info in tqdm(files):
        dst_img = os.path.join(OUTPUT_DIR, "images", split, file_info["filename"])
        dst_label = os.path.join(OUTPUT_DIR, "labels", split, Path(file_info["filename"]).stem + ".txt")

        shutil.copyfile(file_info["img_path"], dst_img)

        if os.path.exists(file_info["label_path"]):
            with open(file_info["label_path"], "r") as f:
                lines = f.readlines()

            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    fixed_lines.append(" ".join(parts) + "\n")  # Preserve original class

            with open(dst_label, "w") as f:
                f.writelines(fixed_lines)
        else:
            open(dst_label, "w").close()

copy_and_fix_labels(train_files, "train")
copy_and_fix_labels(val_files, "val")

# Write data.yaml
with open("data_1.yaml", "w") as f:
    f.write(f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
""")

print("\nBalanced dataset preparation complete!")
print(f"Train images: {len(train_files)}, Val images: {len(val_files)}")
