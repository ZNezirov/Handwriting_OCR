import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    if not isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    else:
        image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned