from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

_predictor = None

def ocr_doctr(img_path):
    global _predictor
    if _predictor is None:
        # For reproducibility and to avoid potential device conflicts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _predictor = ocr_predictor(pretrained=True).to(device)

    # Load the image and move it to the correct device
    doc = DocumentFile.from_images(img_path)
    
    # Run prediction
    result = _predictor(doc)
    
    # Reconstruct the text in proper reading order
    full_text = result.render()
    
    # Clean up potential extra whitespace
    return " ".join(full_text.split())