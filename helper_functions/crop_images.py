import os
from pathlib import Path
from PIL import Image

# --- CONFIG ---
PRIMARY_LABEL_ID = "12"

ORIG_IMG_DIR = "../../Geogusser_Learn_AI/data/images_unlabeled/italy/italy_bollards/images"
ORIG_LABEL_DIR = "../../Geogusser_Learn_AI/data/images_unlabeled/italy/italy_bollards/labels"
CROPPED_IMG_DIR = "../../Geogusser_Learn_AI/data/images_cropped/italy/italy_bollards/images"
CROPPED_LABEL_DIR = "../../Geogusser_Learn_AI/data/images_cropped/italy/italy_bollards/labels"

os.makedirs(CROPPED_IMG_DIR, exist_ok=True)
os.makedirs(CROPPED_LABEL_DIR, exist_ok=True)

# --- Coordinate Conversion ---
def xywh_to_xyxy(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h
    w_abs = w * img_w
    h_abs = h * img_h
    x1 = int(cx - w_abs / 2)
    y1 = int(cy - h_abs / 2)
    x2 = int(cx + w_abs / 2)
    y2 = int(cy + h_abs / 2)
    return x1, y1, x2, y2

def xyxy_to_xywh(x1, y1, x2, y2, crop_w, crop_h):
    cx = (x1 + x2) / 2 / crop_w
    cy = (y1 + y2) / 2 / crop_h
    w = (x2 - x1) / crop_w
    h = (y2 - y1) / crop_h
    return cx, cy, w, h

# --- Main Loop ---
for label_file in os.listdir(ORIG_LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    img_file = label_file.replace(".txt", ".png")
    img_path = os.path.join(ORIG_IMG_DIR, img_file)
    label_path = os.path.join(ORIG_LABEL_DIR, label_file)

    if not os.path.exists(img_path):
        print(f"Image {img_path} missing, skipping")
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse labels
    all_boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cls_id, x, y, w, h = parts[0], *map(float, parts[1:5])
            all_boxes.append((cls_id, x, y, w, h))

    # Only crop around primary class
    for i, (cls_id, x, y, w, h) in enumerate(all_boxes):
        if cls_id != PRIMARY_LABEL_ID:
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h, img_w, img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        crop = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop.size

        crop_img_name = f"{Path(img_file).stem}_crop{i}.jpg"
        crop_label_name = f"{Path(img_file).stem}_crop{i}.txt"
        crop.save(os.path.join(CROPPED_IMG_DIR, crop_img_name))

        with open(os.path.join(CROPPED_LABEL_DIR, crop_label_name), "w") as lf:
            for cls, x_, y_, w_, h_ in all_boxes:
                # Get train box in pixel coordinates
                bx1, by1, bx2, by2 = xywh_to_xyxy(x_, y_, w_, h_, img_w, img_h)

                # Check if box center is inside crop
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    # Translate box relative to crop
                    rel_x1 = max(0, bx1 - x1)
                    rel_y1 = max(0, by1 - y1)
                    rel_x2 = min(crop_w, bx2 - x1)
                    rel_y2 = min(crop_h, by2 - y1)

                    # Avoid negative or flipped boxes
                    if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
                        continue

                    cx_n, cy_n, w_n, h_n = xyxy_to_xywh(rel_x1, rel_y1, rel_x2, rel_y2, crop_w, crop_h)
                    lf.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")
