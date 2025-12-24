import cv2
import os

IMG_PATH = "data/raw_images/4.jpeg"
OUT_DIR = "data/processed_images"

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image not found. Check path.")

# --- Preprocessing ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    5
)

# Save outputs for inspection
cv2.imwrite(f"{OUT_DIR}/gray.jpg", gray)
cv2.imwrite(f"{OUT_DIR}/threshold.jpg", th)

print("Preprocessing complete. Check data/processed_images/")
