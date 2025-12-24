import cv2

IMG_PATH = "data/processed_images/threshold.jpg"

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Receipt image not found")

h, w = img.shape[:2]

ROIS = {
    "receipt_no":   (0.69, 0.18, 0.85, 0.22),
    "date":         (0.23, 0.245, 0.43, 0.265),
    "policy_no":    (0.20, 0.40, 0.80, 0.55),
    "name":         (0.10, 0.28, 0.70, 0.36),
    "premium":      (0.55, 0.60, 0.90, 0.66),
    "late_fee":     (0.55, 0.66, 0.90, 0.72),
    "total":        (0.55, 0.72, 0.90, 0.78),
}

vis = img.copy()

for field, (x1n, y1n, x2n, y2n) in ROIS.items():
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, field, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite("outputs/roi_all_debug.jpg", vis)
print("ROI debug image saved: outputs/roi_all_debug.jpg")




# import cv2
# import pytesseract
# from ocr_masking import ocr_with_mask, preprocess_for_ocr

# IMG_PATH = "data/processed_images/threshold.jpg"

# img = cv2.imread(IMG_PATH)
# if img is None:
#     raise FileNotFoundError("Receipt image not found")

# h, w = img.shape[:2]

# # -------- TEMP ROIs (normalized layout-based) --------
# ROIS = {
#     "receipt_no":   (0.69, 0.18, 0.85, 0.22),
#     "date":         (0.23, 0.245, 0.43, 0.265),
#     "policy_no":    (0.25, 0.42, 0.75, 0.55),
#     "name":         (0.10, 0.30, 0.70, 0.38),
#     "premium":      (0.55, 0.60, 0.90, 0.66),
#     "late_fee":     (0.55, 0.66, 0.90, 0.72),
#     "total":        (0.55, 0.72, 0.90, 0.78),
# }

# output = {}

# for field, (x1n, y1n, x2n, y2n) in ROIS.items():
#     x1, y1 = int(x1n * w), int(y1n * h)
#     x2, y2 = int(x2n * w), int(y2n * h)

#     roi = img[y1:y2, x1:x2]

#     # Numeric fields → masking
#     if field in ["policy_no", "receipt_no", "premium", "late_fee", "total"]:
#         result = ocr_with_mask(roi)
#         output[field] = result["masked"]

#     # Text fields
#     else:
#         proc = preprocess_for_ocr(roi)
#         text = pytesseract.image_to_string(proc, config="--psm 6")
#         output[field] = text.strip().replace("\n", " ")

# print("\n--- EXTRACTED DATA ---")
# for k, v in output.items():
#     print(f"{k:12}: {v}")

