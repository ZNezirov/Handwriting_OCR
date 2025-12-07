import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    if not isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    else:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # optional: resize if very large
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    return image
