import os
import cv2
from ultralytics import YOLO
from ocr_utils import (
    preprocess_numeric,
    preprocess_text,
    ocr_numeric,
    ocr_text,
)

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_DIR = "data/raw_images"
OUT_DIR = "outputs"

NUMERIC_FIELDS = {
    "premium",
    "total",
    "late_fee",
    "receipt_no",
    "policy_no",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD YOLO ----------------
model = YOLO(MODEL_PATH)

# ---------------- RUN INFERENCE ----------------
results = model(IMAGE_DIR, conf=0.25)

final_results = []

for r in results:
    img = cv2.imread(r.path)
    h, w = img.shape[:2]

    record = {
        "image": os.path.basename(r.path),
        "fields": {},
    }

    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Safety clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # -------- FIELD-SPECIFIC PREPROCESS + OCR --------
        if cls_name in NUMERIC_FIELDS:
            proc = preprocess_numeric(roi)
            text = ocr_numeric(proc)
        else:
            proc = preprocess_text(roi)
            text = ocr_text(proc)

        record["fields"][cls_name] = text

        # Debug save (optional)
        # Save raw ROI
        cv2.imwrite(
            f"{OUT_DIR}/{record['image']}_{cls_name}_raw.jpg",
            roi,
        )

        # Save preprocessed ROI
        cv2.imwrite(
            f"{OUT_DIR}/{record['image']}_{cls_name}_proc.jpg",
            proc,
        )


    final_results.append(record)

# ---------------- OUTPUT ----------------
print("\n--- FINAL OCR OUTPUT ---")
for r in final_results:
    print(r)
