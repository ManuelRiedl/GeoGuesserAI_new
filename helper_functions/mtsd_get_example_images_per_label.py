import os
from PIL import Image, ImageDraw
from collections import defaultdict

# Paths
LABELS_DIR = "../data/mtsd/labels_yolo"
IMAGES_DIR = "../data/mtsd/images"
CLASS_NAMES_FILE = "../data/mtsd/classes.txt"
OUTPUT_DIR = "../data/mtsd/class_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load class names
with open(CLASS_NAMES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Track one image per class
class_to_file = defaultdict(list)

# Index label files by class
for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt") or label_file == "classes.txt":
        continue

    with open(os.path.join(LABELS_DIR, label_file), "r") as f:
        for line in f:
            class_id = int(line.split()[0])
            class_to_file[class_id].append(label_file)

# Annotate one image per class with large enough box
for class_id, files in class_to_file.items():
    for label_file in files:
        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(IMAGES_DIR, img_file)
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            found = False

            with open(os.path.join(LABELS_DIR, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cid = int(parts[0])
                    if cid != class_id:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:])
                    xmin = (xc - bw / 2) * w
                    ymin = (yc - bh / 2) * h
                    xmax = (xc + bw / 2) * w
                    ymax = (yc + bh / 2) * h
                    box_width = xmax - xmin
                    box_height = ymax - ymin

                    if box_width < 200 and box_height < 200:
                        continue

                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
                    draw.text((xmin, max(0, ymin - 12)), class_names[cid], fill="red")
                    found = True

            if found:
                out_path = os.path.join(OUTPUT_DIR, f"{class_names[class_id].replace('/', '_')}.jpg")
                img.save(out_path)
                break  # move to next class
