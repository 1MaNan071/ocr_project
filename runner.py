import os, argparse, importlib, gc
from utils.metrics import normalize_text, wer, cer, bleu_simple, rouge_l
import pandas as pd
import torch

MODULES = {
    "easyocr": "engines.easyocr_engine",
    "doctr": "engines.doctr_engine",
    "donut": "engines.donut_engine",
    "smol": "engines.smol_engine"
}

def get_ocr_function(engine_key):
    # Dynamically gets the specific function from the engine module
    mod = importlib.import_module(MODULES[engine_key])
    if hasattr(mod, f"ocr_{engine_key}"):
        return getattr(mod, f"ocr_{engine_key}")
    raise ValueError(f"OCR function not found for engine: {engine_key}")

def clear_memory():
    # Garbage collection and clearing CUDA cache if available
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(images_dir, gt_dir, results_dir, engines):
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    
    if not imgs:
        print("No images found in", images_dir)
        return

    # Loop through engines first to manage memory
    for engine_key in engines:
        print(f"\n--- Starting tests for engine: {engine_key} ---")
        try:
            ocr_fn = get_ocr_function(engine_key)
            
            for im in imgs:
                img_path = os.path.join(images_dir, im)
                base = os.path.splitext(im)[0]
                gt_file = os.path.join(gt_dir, base + '.txt')
                
                if not os.path.exists(gt_file):
                    print(f"Skipping {im} (no GT file)")
                    continue

                with open(gt_file, 'r', encoding='utf-8') as fg:
                    gt_raw = fg.read().strip()
                gt_norm = normalize_text(gt_raw)

                print(f"Running {engine_key} on {im} ...")
                
                try:
                    raw = ocr_fn(img_path)
                except Exception as exc:
                    raw = ""
                    print(f"   Error running {engine_key} on {im}: {exc}")
                
                norm = normalize_text(raw)
                w = wer(gt_norm, norm)
                c = cer(gt_norm, norm)
                b = bleu_simple(gt_norm, norm)
                r = rouge_l(gt_norm, norm)
                
                rows.append({
                    'image': im, 'engine': engine_key, 'gt': gt_raw, 'ocr_raw': raw,
                    'wer': w, 'cer': c, 'bleu': b, 'rouge_l': r
                })

                out_file = os.path.join(results_dir, f"{base}__{engine_key}.txt")
                with open(out_file, 'w', encoding='utf-8') as fo:
                    fo.write(raw)
                
                print(f"   {engine_key} WER={w:.3f} CER={c:.3f} BLEU={b:.3f} ROUGE_L={r:.3f}")

        finally:
            # This part is crucial for freeing memory
            if MODULES[engine_key] in sys.modules:
                del sys.modules[MODULES[engine_key]]
            clear_memory()
            print(f"--- Cleared memory after running {engine_key} ---\n")


    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "ocr_results.csv")
    df.to_csv(csv_path, index=False)
    print("Saved final results to", csv_path)

if __name__ == "__main__":
    import sys # Need to import sys here for the memory clearing part
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--gts', required=True)
    p.add_argument('--results', default='results')
    p.add_argument('--engines', default='easyocr,doctr,donut,smol')
    args = p.parse_args()
    engines = [e.strip() for e in args.engines.split(',') if e.strip()]
    main(args.images, args.gts, args.results, engines)