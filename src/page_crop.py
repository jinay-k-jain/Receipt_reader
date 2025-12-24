import cv2
import numpy as np
import os

IMG_PATH = "data/raw_images/lic_03.jpg"
OUT_DIR = "data/processed_images"
OUT_PATH = os.path.join(OUT_DIR, "receipt_clean.jpg")

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Receipt image not found")

h, w = img.shape[:2]

# --- Convert to HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

# --- Paper mask: bright + low saturation ---
paper_mask = cv2.inRange(
    hsv,
    (0, 0, 150),     # low saturation, high value
    (180, 80, 255)
)

# --- Clean mask ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)
paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel)

# --- Find contours on mask ---
contours, _ = cv2.findContours(
    paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError("No paper-like region detected")

# Pick largest white region (receipt)
receipt_cnt = max(contours, key=cv2.contourArea)

x, y, bw, bh = cv2.boundingRect(receipt_cnt)

# Small padding
pad = 10
x = max(0, x - pad)
y = max(0, y - pad)
bw = min(w - x, bw + 2 * pad)
bh = min(h - y, bh + 2 * pad)

receipt = img[y:y+bh, x:x+bw]

cv2.imwrite(OUT_PATH, receipt)
cv2.imwrite(os.path.join(OUT_DIR, "paper_mask.jpg"), paper_mask)

print("✅ Receipt cropped using color-based segmentation")
print("Saved:", OUT_PATH)
