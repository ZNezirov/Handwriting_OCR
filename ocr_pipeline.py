import easyocr
import numpy as np
from PIL import Image
import cv2

reader = easyocr.Reader(['en'], gpu=True)

def ocr_image_with_boxes(image_input):
    if not isinstance(image_input, str):
        image = np.array(Image.open(image_input).convert("RGB"))
    else:
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = reader.readtext(image)
    
    annotated = image.copy()
    for (bbox, text, conf) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        cv2.putText(annotated, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
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
    
    return "\n".join(lines), annotated