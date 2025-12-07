import cv2
import numpy as np

def segment_lines(image):
    horizontal_proj = cv2.reduce(image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(-1)
    
    mean_val = np.mean(horizontal_proj)
    threshold = mean_val * 0.15
    
    in_line = False
    start_idx = None
    lines = []
    min_gap = 5
    gap_counter = 0
    
    for i, val in enumerate(horizontal_proj):
        if val > threshold:
            if not in_line:
                start_idx = i
                in_line = True
            gap_counter = 0
        else:
            if in_line:
                gap_counter += 1
                if gap_counter >= min_gap:
                    if i - start_idx > 10:
                        lines.append(image[start_idx:i-min_gap+1, :])
                    in_line = False
    
    if in_line and start_idx is not None:
        if len(horizontal_proj) - start_idx > 10:
            lines.append(image[start_idx:, :])
    
    return lines