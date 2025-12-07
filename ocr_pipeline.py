from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

from preprocess import preprocess_image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def ocr_image(image_input):
    image = preprocess_image(image_input)
    pil_image = Image.fromarray(image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text
