from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

_model = None
_processor = None

# We've changed the default model to one fine-tuned for receipt/document parsing
def ocr_donut(img_path, model_name="naver-clova-ix/donut-base-finetuned-cord-v2"):
    global _model, _processor
    if _model is None:
        print("Donut: Loading model and processor for the first time...")
        _processor = DonutProcessor.from_pretrained(model_name)
        _model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _model.eval()
        print("Donut: Model and processor loaded.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    
    img = Image.open(img_path).convert("RGB")
    
    print(f"Donut: Processing image {img_path}...")
    # Prepare inputs for the model
    pixel_values = _processor(img, return_tensors="pt").pixel_values.to(device)
    
    print("Donut: Generating output... (This may take a while on CPU)")
    # Generate output
    outputs = _model.generate(pixel_values, max_length=1024) # Increased max_length for longer docs
    
    print("Donut: Decoding output...")
    # Decode the output
    try:
        decoded = _processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception:
        decoded = _processor.decode(outputs[0], skip_special_tokens=True)
        
    print("Donut: Finished processing.")
    return decoded