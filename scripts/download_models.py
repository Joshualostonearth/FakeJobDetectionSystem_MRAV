"""
Pre-download everything the app needs from the internet, so it can then run offline.

  DistilBERT tokenizer + config (distilbert-base-uncased)   ~0.7 MB   (weights come from models/stage2.pt)
  sentence-transformers/all-MiniLM-L6-v2                   ~90 MB
  EasyOCR English models (craft_mlt_25k + english_g2)       ~100 MB

Already-cached items are skipped. Not downloaded here (not on the internet):
models/mravplus_bundle.joblib and models/stage2.pt come from the Kaggle MRAV+ notebook.

    python scripts/download_models.py          # asks before downloading
    python scripts/download_models.py --yes

Behind a TLS-inspecting proxy / antivirus, downloads may fail with CERTIFICATE_VERIFY_FAILED;
point REQUESTS_CA_BUNDLE (and SSL_CERT_FILE) at your organisation's CA bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mravplus_infer import DISTILBERT_NAME, MINILM_NAME  # noqa: E402

DISTILBERT_FILES = ["config.json", "tokenizer_config.json", "vocab.txt", "tokenizer.json"]
EASYOCR_DIR = Path.home() / ".EasyOCR" / "model"
EASYOCR_FILES = ["craft_mlt_25k.pth", "english_g2.pth"]


def hf_cached(repo: str, files=None) -> bool:
    from huggingface_hub import snapshot_download
    try:
        path = Path(snapshot_download(repo, local_files_only=True))
    except Exception:
        return False
    return all((path / f).exists() for f in (files or ["config.json"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="download without asking")
    args = ap.parse_args()

    todo = []
    if not hf_cached(DISTILBERT_NAME, DISTILBERT_FILES):
        todo.append(("DistilBERT tokenizer + config", "~0.7 MB"))
    if not hf_cached(MINILM_NAME, ["modules.json", "config.json"]):
        todo.append(("MiniLM sentence encoder", "~90 MB"))
    if not all((EASYOCR_DIR / f).exists() for f in EASYOCR_FILES):
        todo.append(("EasyOCR English models", "~100 MB"))

    try:
        import torch
        print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch is not installed; install requirements.txt first.")
        return 1

    if not todo:
        print("Everything is already cached. The app can run offline.")
        return 0
    print("Will download:")
    for name, size in todo:
        print(f"  - {name:32s} {size}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 1

    from huggingface_hub import hf_hub_download, snapshot_download
    for f in DISTILBERT_FILES:
        hf_hub_download(DISTILBERT_NAME, f)
    snapshot_download(MINILM_NAME, allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*",
                                                   "sentence_bert_config.json", "modules.json"])
    import easyocr
    easyocr.Reader(["en"], gpu=False)  # downloads its models on first use
    print("Done. The app can now run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
