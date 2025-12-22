import cv2
from ultralytics import YOLO
import os

IMG_PATH = "data/raw_images/lic_01.jpeg"
OUT_PATH = "outputs/yolo_test.jpg"

os.makedirs("outputs", exist_ok=True)

# Load model
model = YOLO("models/yolov8n.pt")

# Read image
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

# Run inference
results = model(img)[0]

# Draw boxes
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{cls}:{conf:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

# Save result
cv2.imwrite(OUT_PATH, img)

print("YOLO inference complete. Check outputs/yolo_test.jpg")
