from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
mod = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

dev = "cuda" if torch.cuda.is_available() else "cpu"
mod.to(dev)

def ocr_image(img_in):
    try:
        if isinstance(img_in, str):
            im = Image.open(img_in).convert("RGB")
        else:
            im = Image.open(img_in).convert("RGB")
        
        px = proc(im, return_tensors="pt").pixel_values
        px = px.to(dev)
        
        prompt = "<s_cord-v2>"
        dec_in = proc.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        dec_in = dec_in.to(dev)
        
        out = mod.generate(
            px,
            decoder_input_ids=dec_in,
            max_length=mod.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=proc.tokenizer.pad_token_id,
            eos_token_id=proc.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[proc.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        seq = proc.batch_decode(out.sequences)[0]
        seq = seq.replace(proc.tokenizer.eos_token,"").replace(proc.tokenizer.pad_token,"")
        seq = re.sub(r"<.*?>","", seq, count=1).strip()
        
        txt = proc.token2json(seq)
        
        if isinstance(txt, dict):
            res = []
            def go(d):
                if isinstance(d, dict):
                    for v in d.values(): go(v)
                elif isinstance(d, list):
                    for i in d: go(i)
                elif isinstance(d, str):
                    res.append(d)
            go(txt)
            return "\n".join(res)
        
        return str(txt)
        
    except Exception as e:
        return "error:"+str(e)
