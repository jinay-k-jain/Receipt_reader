import cv2
from ocr_masking import ocr_with_mask
from ocr_masking import preprocess_for_ocr

IMG_PATH = "data/raw_images/lic_01.jpeg"

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

h, w = img.shape[:2]

# ---- TEMP ROI (we will adjust) ----
x1 = int(0.20 * w)
y1 = int(0.40 * h)
x2 = int(0.80 * w)
y2 = int(0.55 * h)

roi = img[y1:y2, x1:x2]
proc_roi = preprocess_for_ocr(roi)

cv2.imwrite("outputs/roi_processed.jpg", proc_roi)

# Draw ROI on original image
vis = img.copy()
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("outputs/roi_debug.jpg", vis)
cv2.imwrite("outputs/roi_crop.jpg", roi)

print("ROI images saved to outputs/")

result = ocr_with_mask(roi)
print("RAW POLICY NO :", result["raw"])
print("MASKED POLICY:", result["masked"])
print("CONFIDENCES  :", result["confidences"])
