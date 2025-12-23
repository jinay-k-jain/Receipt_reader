import cv2
import pytesseract
from pytesseract import Output

DIGIT_CONF_THRESHOLD = 70
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )
    return th
    
def ocr_with_mask(img):
    proc = preprocess_for_ocr(img)
    data = pytesseract.image_to_data(
        proc,
        config="--psm 6 -c tessedit_char_whitelist=0123456789",
        output_type=Output.DICT)
    raw = ""
    masked = ""
    confidences = []
    for txt, conf in zip(data["text"], data["conf"]):
        if not txt.strip():
            continue
        if conf == "-1":
            continue
        conf = int(conf)
        for ch in txt:
            if ch.isdigit():
                raw += ch
                confidences.append(conf)
                masked += ch if conf >= DIGIT_CONF_THRESHOLD else "*"
    return {
        "raw": raw,
        "masked": masked,
        "confidences": confidences
    }
