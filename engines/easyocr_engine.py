# engines/easyocr_engine.py
import easyocr
from PIL import Image
import numpy as np

_reader = None

def ocr_easyocr(img_path):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    res = _reader.readtext(arr)
    texts = [t[1] for t in res]
    return " ".join(texts)
