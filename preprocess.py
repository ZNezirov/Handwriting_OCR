import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    image: either a file path (str) or a BytesIO from Streamlit
    """
    if not isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    else:
        image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh
