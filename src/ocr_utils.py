import cv2
import pytesseract
import re

CONF_THRESHOLD = 60
def preprocess_numeric(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th
    
def preprocess_text(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray
    
def ocr_numeric(img):
    config = "--psm 7 -c tessedit_char_whitelist=0123456789./"
    data = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT
    )

    out = ""
    for i, txt in enumerate(data["text"]):
        if txt.strip() and int(data["conf"][i]) >= CONF_THRESHOLD:
            out += txt

    return re.sub(r"[^0-9./]", "", out)


def ocr_text(img):
    config = "--psm 6"
    txt = pytesseract.image_to_string(img, config=config)
    return txt.strip()
