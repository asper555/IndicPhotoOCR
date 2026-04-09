from IndicPhotoOCR.ocr import OCR

ocr_system = OCR(verbose=True, device="cuda")

image_path = "test_images/image_24.jpg"

detections = ocr_system.detect(image_path)

# 🔥 THIS LINE CREATES IMAGE
ocr_system.visualize_detection(image_path, detections)

print("Done! Check test.png")