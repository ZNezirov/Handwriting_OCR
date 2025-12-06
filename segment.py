import cv2

def segment_lines(image):
    """
    image: preprocessed binary image
    returns: list of line images
    """
    horizontal_sum = cv2.reduce(image, 1, cv2.REDUCE_AVG).reshape(-1)
    lines = []
    start_idx = None

    for i, val in enumerate(horizontal_sum):
        if val > 0 and start_idx is None:
            start_idx = i
        elif val == 0 and start_idx is not None:
            lines.append(image[start_idx:i, :])
            start_idx = None
    if start_idx is not None:
        lines.append(image[start_idx:, :])
    return lines
