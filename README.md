# Fake Job Detection System

This repository contains a pipeline for detecting fake job offers. It uses a combination of Optical Character Recognition (OCR), structured entity extraction via local Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG) using a local vector database.

## Prerequisites & Installation

1. **Python Environment**
   It is highly recommended to use a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies**
   Install the required Python packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Models Setup

This project relies on several machine learning models. Please follow these steps to ensure all models are properly configured before running the code.

### 1. Ollama & Ministral 3
Every LLM step (entity extraction, credibility analysis, M-RAV alignment and the voice agent) uses one local Ollama model, set by the `OLLAMA_MODEL` environment variable (default: `ministral-3`). You need to install Ollama and pull the model locally.
- **Install Ollama**: Download and install from [ollama.com](https://ollama.com/)
- **Pull the Model**:
  Open your terminal and run:
  ```bash
  ollama pull ministral-3
  ```
  *(Make sure the Ollama application/server is running in the background before executing the Python scripts).*

### 2. Sentence Transformers (Embeddings)
The system uses the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face for generating vector embeddings (via `HuggingFaceEmbeddings` in `build_vector_db.py`).
- **No manual installation needed**: This model is downloaded **automatically** the first time you run `build_vector_db.py` or any script that initializes the embeddings. Ensure you have an active internet connection on your first run.

### 3. EasyOCR Models
For extracting text from images (e.g., job offer letters), the project uses `easyocr` (in `ocr_engine.py`).
- **No manual installation needed**: The required text detection and recognition models for English will be downloaded **automatically** to your `~/.EasyOCR` folder the first time you run an OCR function.

## Getting Started

1. **Build the Vector Database**:
   Before running the search engine, build the local FAISS index from the EMSCAD training split (`data/splits/train.csv`, 12,272 labelled job postings; `data/splits/` also holds `val.csv` and `test.csv`):
   ```bash
   python ingest_emscad.py
   ```
   This embeds every training posting and saves the index in the `faiss_index/` directory, which the app loads.

   Then train the scam text classifier (TF-IDF + linear SVM, writes `scam_classifier.pkl`):
   ```bash
   python scam_classifier.py --train data/splits/train.csv
   ```
   The app and the full pipeline use it as an extra evidence source; without the file they run without it.

   `python build_vector_db.py` builds a small demo index from `data/evidence_seed.json` into `faiss_index_seed/` instead; it does not replace the EMSCAD index.

2. **Run the Pipeline**:
   Test the entire pipeline (including OCR, extraction, and credibility checks) by running:
   ```bash
   python test_full_pipeline.py
   ```
   Or run the main application:
   ```bash
   python main.py
   ```

## Evaluation

`evaluate_test_split.py` evaluates the text pipeline (Objectives 1, 2 and 4; OCR and the voice
agent are excluded) on an EMSCAD split. The Ollama steps are cached so the rest can be re-scored quickly:

```bash
export OLLAMA_MODEL=llama3.2
python evaluate_test_split.py extract   evaluation_results/test_sample_1000.csv evaluation_results/entities_test.jsonl
python evaluate_test_split.py llm       evaluation_results/test_sample_1000.csv evaluation_results/entities_test.jsonl evaluation_results/llm_mrav_test.jsonl
python evaluate_test_split.py score     evaluation_results/test_sample_1000.csv evaluation_results/entities_test.jsonl evaluation_results/final_test evaluation_results/llm_mrav_test.jsonl
python evaluate_test_split.py metrics   evaluation_results/final_test
```

Thresholds and the Platt model are fitted on the validation split only (`fit-platt`); the test sample
is scored once. Results and the full report are in `evaluation_results/PERFORMANCE_REPORT.md`.
