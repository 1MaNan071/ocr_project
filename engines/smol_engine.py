# engines/smol_engine.py
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

_model = None
_processor = None

def ocr_smol(img_path, model_name="ds4sd/SmolDocling-256M-preview"):
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    img = Image.open(img_path).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt").to(device)
    out = _model.generate(**inputs, max_new_tokens=256)
    # Processor decoding may vary; try processor.decode then tokenizer fallback
    try:
        text = _processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        text = tok.decode(out[0], skip_special_tokens=True)
    return text
