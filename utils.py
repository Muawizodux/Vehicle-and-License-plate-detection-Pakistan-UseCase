import numpy as np
import cv2
from PIL import Image

def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

def correct_skew(image, delta=20, limit=14):
    thresh =image
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected

def histogram(matrix):
  row, col = matrix.shape
  pixel = list(range(256))
  histogram_dict= dict((a, 0) for a in pixel)
  for i in range(row):
    for j in range(col):
      label = histogram_dict.get(matrix[i][j])+1
      histogram_dict[matrix[i][j]] = label
  return histogram_dict

def preprocess_lp(image) -> Image.Image:
    min_area = 200 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalize_image = cv2.equalizeHist(gray_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))  
    opened_image = cv2.morphologyEx(equalize_image, cv2.MORPH_OPEN, kernel)
    _, thresholded_image = cv2.threshold(equalize_image-opened_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    # print(f"Number of contours: {len(filtered_contours)}")
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = equalize_image[y:y+h, x:x+w]  # Extract ROI
    angle, corrected = correct_skew(roi)
    return corrected

