import os
from PIL import Image
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

# Track label files per class
class_to_file = defaultdict(list)
for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt") or label_file == "classes.txt":
        continue
    with open(os.path.join(LABELS_DIR, label_file), "r") as f:
        for line in f:
            class_id = int(line.split()[0])
            class_to_file[class_id].append(label_file)

# Create collage for each class
for class_id, files in class_to_file.items():
    crops = []
    for label_file in files:
        if len(crops) >= 10:
            break
        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(IMAGES_DIR, img_file)
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            w, h = img.size
            with open(os.path.join(LABELS_DIR, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cid = int(parts[0])
                    if cid != class_id:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:])
                    xmin = int((xc - bw / 2) * w)
                    ymin = int((yc - bh / 2) * h)
                    xmax = int((xc + bw / 2) * w)
                    ymax = int((yc + bh / 2) * h)
                    box_width = xmax - xmin
                    box_height = ymax - ymin

                    if box_width < 20 or box_height < 20:
                        continue

                    crop = img.crop((xmin, ymin, xmax, ymax)).resize((200, 200))
                    crops.append(crop)
                    if len(crops) >= 10:
                        break

    if crops:
        # Create canvas (2 rows × 5 columns)
        canvas = Image.new("RGB", (5 * 200, 2 * 200), "white")
        for i, crop in enumerate(crops):
            x = (i % 5) * 200
            y = (i // 5) * 200
            canvas.paste(crop, (x, y))
        out_path = os.path.join(OUTPUT_DIR, f"{class_names[class_id].replace('/', '_')}.jpg")
        canvas.save(out_path)