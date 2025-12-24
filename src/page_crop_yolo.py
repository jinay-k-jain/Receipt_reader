import cv2
import pytesseract
import os

IMG_PATH = "data/raw_images/4.jpeg"
OUT_DIR = "data/processed_images"
OUT_PATH = os.path.join(OUT_DIR, "receipt_text_crop.jpg")

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FAST OCR (layout only, not accuracy)
data = pytesseract.image_to_data(
    gray,
    config="--psm 6",
    output_type=pytesseract.Output.DICT
)

xs, ys, xe, ye = [], [], [], []

for i in range(len(data["text"])):
    txt = data["text"][i].strip()
    conf = int(data["conf"][i])

    if conf > 30 and len(txt) > 2:
        x = data["left"][i]
        y = data["top"][i]
        w_ = data["width"][i]
        h_ = data["height"][i]

        xs.append(x)
        ys.append(y)
        xe.append(x + w_)
        ye.append(y + h_)

if not xs:
    raise RuntimeError("No text detected for density crop")

# Union box
x1 = max(0, min(xs) - 20)
y1 = max(0, min(ys) - 20)
x2 = min(w, max(xe) + 20)
y2 = min(h, max(ye) + 20)

crop = img[y1:y2, x1:x2]

cv2.imwrite(OUT_PATH, crop)
print("✅ Receipt cropped using TEXT DENSITY:", OUT_PATH)
