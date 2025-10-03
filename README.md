# OCR Benchmarking Framework

This project provides a flexible framework to benchmark the performance of various Optical Character Recognition (OCR) engines on a custom dataset of images. It calculates common OCR evaluation metrics such as Word Error Rate (WER), Character Error Rate (CER), BLEU, and ROUGE-L.

## Project Structure

* `runner.py`: The main executable script that runs the benchmarking process.
* `engines/`: Contains Python modules for each OCR engine, acting as wrappers for their respective libraries.
    * `easyocr_engine.py`
    * `doctr_engine.py`
    * *(Other engines can be added here)*
* `utils/`: Contains helper scripts.
    * `metrics.py`: Implements the evaluation metrics (WER, CER, etc.).
* `images/`: (Ignored by Git) Place your input image files (`.png`, `.jpg`) in this directory.
* `gt/`: (Ignored by Git) Place the corresponding ground truth text files (`.txt`) here. The filename must match the image filename (e.g., `my_image.png` and `my_image.txt`).
* `output/`: (Ignored by Git) This folder is created by `runner.py` to store the raw text output from each engine and the final `ocr_results.csv` summary.

## Setup

1.  **Clone the repository:**
    ```
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment:**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```

## Usage

1.  Add your images to the `images/` folder.
2.  Add the corresponding ground truth text to the `gt/` folder.
3.  Run the benchmarking script from your terminal:
    ```
    python runner.py --images ./images --gts ./gt --results ./output --engines easyocr,doctr
    ```

### Command-Line Arguments

* `--images`: Path to the directory containing your images.
* `--gts`: Path to the directory containing your ground truth text files.
* `--results`: Path to the directory where results will be saved.
* `--engines`: A comma-separated list of the engines to test (e.g., `easyocr,doctr`).

After the script finishes, the benchmark summary will be available in `output/ocr_results.csv`.
