import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
MODEL_STAGE1 = "saved_models/bollard_only/bollard_yolov8n/weights/best.pt"
MODEL_STAGE2 = "saved_models/big_model_2_07/bollard_yolov8n/weights/best.pt"

INPUT_FOLDER = "data/testset/images"
OUTPUT_STAGE0 = "predictions_pipeline/new/stage0"
OUTPUT_STAGE1 = "predictions_pipeline/new/stage1"

IMAGE_SIZE_STAGE1 = 640
IMAGE_SIZE_STAGE2 = 640
CONF_STAGE1 = 0.4
CONF_STAGE2 = 0.35

# Create output directories
os.makedirs(OUTPUT_STAGE0, exist_ok=True)
os.makedirs(OUTPUT_STAGE1, exist_ok=True)

# --- Load YOLOv8 models ---
model_stage1 = YOLO(MODEL_STAGE1)
model_stage2 = YOLO(MODEL_STAGE2)

def run_pipeline():
    print("🚀 Running two-stage bollard detection pipeline...")
    test_images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png'))]

    for image_file in tqdm(test_images, desc="📷 Processing test images"):
        full_path = os.path.join(INPUT_FOLDER, image_file)
        original_img = cv2.imread(full_path)

        # Stage 1: Generic bollard detection
        stage1_result = model_stage1.predict(full_path, imgsz=IMAGE_SIZE_STAGE1, conf=CONF_STAGE1, stream=False)[0]
        detections = stage1_result.boxes

        img_stage0 = original_img.copy()
        img_stage1 = original_img.copy()

        if detections is None or len(detections) == 0:
            continue

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # --- Stage 0 output ---
            cv2.rectangle(img_stage0, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(img_stage0, f"bollard {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            # --- Crop for Stage 2 ---
            crop = original_img[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue  # skip tiny regions

            result = model_stage2.predict(crop, imgsz=IMAGE_SIZE_STAGE2, conf=CONF_STAGE2, stream=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for sub_box in result.boxes:
                cls = int(sub_box.cls[0])
                label = model_stage2.names[cls]
                conf_stage2 = float(sub_box.conf[0])
                label_conf = f"{label} {conf_stage2:.2f}"

                sx1, sy1, sx2, sy2 = map(int, sub_box.xyxy[0].cpu().numpy())
                abs_x1 = x1 + sx1
                abs_y1 = y1 + sy1
                abs_x2 = x1 + sx2
                abs_y2 = y1 + sy2

                # --- Stage 1 output: Only draw subtype, not generic bollard ---
                cv2.rectangle(img_stage1, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                cv2.putText(img_stage1, label_conf, (abs_x1, abs_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save both annotated outputs
        cv2.imwrite(os.path.join(OUTPUT_STAGE0, image_file), img_stage0)
        cv2.imwrite(os.path.join(OUTPUT_STAGE1, image_file), img_stage1)

    print(f"\n✅ Done! Stage 0 predictions saved in: {OUTPUT_STAGE0}")
    print(f"✅ Done! Stage 1 predictions saved in: {OUTPUT_STAGE1}")

if __name__ == "__main__":
    run_pipeline()
