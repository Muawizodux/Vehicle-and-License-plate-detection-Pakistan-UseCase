import cv2
import numpy as np
from paddleocr import PaddleOCR
from utils import preprocess_lp

image = cv2.imread('')
cropped_region = preprocess_lp(image)
image_np = np.array(cropped_region)

ocr = PaddleOCR()
result = ocr.ocr(image_np)

license_plate_text = '\n'.join([' '.join(str(item) for item in line) for _, line in result[0]])
print("License plate Number :", license_plate_text.split()[0])