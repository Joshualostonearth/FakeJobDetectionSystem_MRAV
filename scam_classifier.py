"""
scam_classifier.py — Supervised text classifier evidence source (Objective 1b).

A TF-IDF (word 1-2 grams + character 2-5 grams) linear SVM trained on the EMSCAD
training split, with sigmoid calibration so it outputs P(scam | posting text).
Its output is turned into a Dempster-Shafer mass (obj2dst.classifier_mass) and
combined with the FAISS evidence in Objective 4.

Train once (writes scam_classifier.pkl):
    python scam_classifier.py --train data/splits/train.csv

Use:
    from scam_classifier import predict_scam_probability
    p = predict_scam_probability(posting_text)
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

MODEL_PATH = Path(__file__).parent / "scam_classifier.pkl"

# Columns of an EMSCAD row rendered as the text a user would paste into the app
TEXT_FIELDS = [
    ("title", "Job Title"),
    ("location", "Location"),
    ("department", "Department"),
    ("salary_range", "Salary Range"),
    ("employment_type", "Employment Type"),
    ("required_experience", "Required Experience"),
    ("required_education", "Required Education"),
    ("industry", "Industry"),
    ("function", "Function"),
    ("company_profile", "Company Profile"),
    ("description", "Description"),
    ("requirements", "Requirements"),
    ("benefits", "Benefits"),
]


def posting_to_text(row: dict) -> str:
    """Render an EMSCAD posting as plain job-offer text (label column is never used)."""
    parts = []
    for col, label in TEXT_FIELDS:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            parts.append(f"{label}: {val.strip()}")
    return "\n".join(parts)


def build_model():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.svm import LinearSVC

    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300000),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000),
    )
    svm = LinearSVC(C=0.5, class_weight="balanced")
    return make_pipeline(features, CalibratedClassifierCV(svm, method="sigmoid", cv=5))


def train(csv_path: str, model_path: Path = MODEL_PATH):
    import pandas as pd

    df = pd.read_csv(csv_path)
    texts = [posting_to_text(r) for r in df.to_dict("records")]
    model = build_model()
    model.fit(texts, df["fraudulent"].astype(int).values)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained on {len(df)} postings ({int(df['fraudulent'].sum())} fraudulent) -> {model_path}")
    return model


_model = None


def load_model(model_path: Path = MODEL_PATH):
    global _model
    if _model is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path.name} not found. Train it with: python scam_classifier.py --train data/splits/train.csv"
            )
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict_scam_probability(text: str) -> Optional[float]:
    """P(scam) for a job posting text, or None if the text is empty."""
    if not text or not text.strip():
        return None
    return float(load_model().predict_proba([text])[0][1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the scam text classifier")
    parser.add_argument("--train", required=True, help="CSV with EMSCAD columns and a 'fraudulent' label")
    args = parser.parse_args()
    train(args.train)
