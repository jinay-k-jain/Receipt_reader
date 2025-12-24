# Receipt Reader – YOLOv8 + OCR Pipeline

An end-to-end computer vision system to automatically extract structured information from LIC payment receipts captured using a mobile camera.

The system detects receipt fields using a trained YOLOv8 model, crops each field, preprocesses it, and applies OCR to extract text with a focus on robustness against noisy, dot-matrix printed receipts.

---

## Problem Statement

LIC receipts are often printed using dot-matrix printers and later captured using mobile cameras.  
These images suffer from:
- Background clutter (tables, surfaces)
- Low contrast text
- Misprints and broken characters
- Inconsistent layouts

Traditional full-page OCR fails badly in such cases.

---

## Solution Overview

This project follows a **detect → crop → preprocess → OCR** architecture:

1. **YOLOv8** detects the receipt fields directly from raw images.
2. Each detected field is **cropped independently**.
3. **Field-specific preprocessing** is applied to improve OCR quality.
4. **OCR (Tesseract)** extracts text from each field.
5. Output is returned as **structured key-value data**.

This design separates *layout understanding* from *text recognition*, which significantly improves accuracy.

---

## Fields Extracted

- Receipt Number
- Policy Number
- Date
- Name
- Premium Amount
- Late Fee
- Total Amount

---

## Tech Stack

- **Python 3.12**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **Tesseract OCR**
- **NumPy**
- **Ubuntu 24.04**

---

## Project Structure

receipt_reader/

├── data/

│ └── raw_images/ # Input receipt images

├── models/

│ └── yolov8n.pt # Base YOLO model

├── src/

│ ├── ocr_utils.py # OCR + preprocessing utilities

│ └── yolo_ocr_pipeline.py

├── outputs/ # Debug crops (ignored in git)

├── runs/ # YOLO training runs (ignored)

└── README.md
---

## How It Works

1. Run YOLOv8 on **raw images** to detect receipt fields.
2. Crop each detected bounding box.
3. Apply **field-specific preprocessing**:
   - Strong thresholding for numeric fields
   - Light filtering for text fields
4. Run OCR on each processed ROI.
5. Aggregate results into structured output.

---

## Running the Pipeline

Activate environment and run:

```bash
python src/yolo_ocr_pipeline.py
```
The output is printed to console and cropped ROIs are saved in outputs/ for inspection.

##Key Design Decisions

-YOLO is applied on raw images only (no preprocessing before detection).
-Preprocessing is applied after cropping, not globally.
-OCR is tuned differently for numeric vs text fields.
-Debug artifacts are saved to visually validate each pipeline stage.

##Current Limitations

-Model trained on a small dataset (prototype stage).
-Loose bounding boxes can affect OCR in some cases.
-Confidence-based character masking can be improved further.

##Future Improvements

-Add more training samples for tighter field localization.
-Re-enable digit-level confidence masking.
-Add regex-based validation for dates and currency.
-Deploy as a REST API using FastAPI.
-Support multi-receipt batch processing.

##Status

This project is a working prototype demonstrating a real-world document AI pipeline and can be extended into a production-ready system with additional data and refinements.
