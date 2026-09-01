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

### 1. Ollama & Llama 3.2
The system uses **Llama 3.2** for structured entity extraction (via `langchain_ollama` inside `models.py`). You need to install Ollama and pull the model locally.
- **Install Ollama**: Download and install from [ollama.com](https://ollama.com/)
- **Pull the Model**:
  Open your terminal and run:
  ```bash
  ollama pull llama3.2
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
   Before running the search engine, you need to build the local FAISS index by running:
   ```bash
   python build_vector_db.py
   ```
   This will embed the data and save the index locally in the `faiss_index/` directory.

2. **Run the Pipeline**:
   Test the entire pipeline (including OCR, extraction, and credibility checks) by running:
   ```bash
   python test_full_pipeline.py
   ```
   Or run the main application:
   ```bash
   python main.py
   ```
