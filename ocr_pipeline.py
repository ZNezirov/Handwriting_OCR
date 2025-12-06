# ocr_pipeline.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
import cv2

from preprocess import preprocess_image
from segment import segment_lines

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def ocr_image(image_input):
    """
    image_input: str path or Streamlit uploaded file (BytesIO)
    returns: extracted text
    """
    preprocessed = preprocess_image(image_input)

    lines = segment_lines(preprocessed)

    results = []
    for line in lines:
        if len(line.shape) == 2:
            pil_image = Image.fromarray(line).convert("RGB")
        else:
            pil_image = Image.fromarray(line)

        pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append(text)

    return "\n".join(results)
