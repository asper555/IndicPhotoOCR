from IndicPhotoOCR.ocr import OCR
import os
import cv2
import numpy as np

# 🔥 GPU
ocr_system = OCR(verbose=False, device="cuda")

IMAGE_DIR = "val_data/images"
PRED_DIR = "pred"
VIS_DIR = "visualizations"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


def save_predictions(polygons, save_path):
    with open(save_path, "w") as f:
        for poly in polygons:
            coords = []
            for x, y in poly:
                coords.append(str(int(x)))
                coords.append(str(int(y)))
            f.write(",".join(coords) + "\n")


for img_name in os.listdir(IMAGE_DIR):

    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_DIR, img_name)

    print(f"\nProcessing: {img_name}")

    # 🔍 Detection
    detections = ocr_system.detect(image_path)

    # ---- SAVE TXT ----
    txt_name = img_name.rsplit(".", 1)[0] + ".txt"
    txt_path = os.path.join(PRED_DIR, txt_name)
    save_predictions(detections, txt_path)

    # ---- VISUALIZATION ----
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image")
        continue

    for poly in detections:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    vis_path = os.path.join(VIS_DIR, img_name)
    cv2.imwrite(vis_path, img)

    print(f"Saved: {txt_path} | {vis_path}")

print("\n✅ ALL IMAGES PROCESSED")