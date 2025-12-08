import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'], gpu=True)

def ocr_image(image_input):
    if not isinstance(image_input, str):
        image = np.array(Image.open(image_input).convert("RGB"))
    else:
        image = image_input
    
    results = reader.readtext(image)
    
    sorted_results = sorted(results, key=lambda x: x[0][0][1])
    
    lines = []
    current_line = []
    current_y = None
    
    for (bbox, text, conf) in sorted_results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        
        if current_y is None:
            current_y = y_center
            current_line.append((bbox[0][0], text))
        elif abs(y_center - current_y) < 20:
            current_line.append((bbox[0][0], text))
        else:
            current_line.sort(key=lambda x: x[0])
            lines.append(" ".join([t for _, t in current_line]))
            current_line = [(bbox[0][0], text)]
            current_y = y_center
    
    if current_line:
        current_line.sort(key=lambda x: x[0])
        lines.append(" ".join([t for _, t in current_line]))
    
    return "\n".join(lines)