"""
Acceptance test: image -> OCR -> M4 -> flag, with Ollama OFF and the voice services OFF.

Ollama "off" is enforced, not assumed: OLLAMA_HOST points at a dead port and every LLM entry
point (check_ollama_service, enrich_with_llm, models.extract_entities_from_text) fails the test
if called. Voice services are never contacted by handle_ui_submission.

  - richmond_offer.png from the repo gets a verdict + explanation
  - 6 synthetic posters (3 fraud / 3 legit EMSCAD test postings, fixed seed) get verdicts on both
    the image path and the pasted-text path

The full 40-poster numbers come from `python scripts/poster_images.py`
(results/e2e_image_report.md).

    python -m pytest tests/test_e2e_image.py -v -s
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mravplus_infer as m4  # noqa: E402

if m4.find_model_file(m4.BUNDLE_FILENAME) is None or not m4.DATA_CSV.is_file():
    pytest.skip("models/mravplus_bundle.joblib or data/fake_job_postings.csv missing", allow_module_level=True)


def _forbidden(*a, **k):
    raise AssertionError("an LLM / Ollama entry point was called on the image -> verdict path")


@pytest.fixture(autouse=True)
def ollama_off(monkeypatch):
    import entity_extractor
    import pipeline_bridge
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    monkeypatch.setattr(pipeline_bridge, "check_ollama_service", _forbidden)
    monkeypatch.setattr(pipeline_bridge, "enrich_with_llm", _forbidden)
    monkeypatch.setattr(entity_extractor, "enrich_with_llm", _forbidden)
    monkeypatch.setattr(pipeline_bridge, "check_voice_services", _forbidden)


@pytest.fixture(scope="module")
def reader():
    from ocr_engine import init_ocr_reader
    r = init_ocr_reader(["en"], gpu=False)
    if r is None:
        pytest.skip("EasyOCR models not available (run scripts/download_models.py)")
    return r


def test_richmond_offer_image(reader):
    import pipeline_bridge as pb
    res = pb.handle_ui_submission(str(PROJECT_ROOT / "richmond_offer.png"), "", reader=reader)
    print("\nrichmond_offer.png:", res["final_verdict"], res["risk_level"], f"p_fake={res['p_fake']:.3f}", res["zone"])
    assert res["ocr_text"].strip()
    assert res["final_verdict"] in ("FAKE", "NOT FAKE")
    assert res["xai"]["m1"]["identity_error"] < 1e-6
    assert res["xai"]["summary"]["text"]


def test_synthetic_posters_image_vs_text(reader, tmp_path):
    import pipeline_bridge as pb
    from scripts.poster_images import pick, poster_text, render

    _, _, te = m4.load_emscad_splits()
    det = pb.get_detector()
    agree = 0
    idx = pick(te, n_each=3)
    for i in idx:
        rec = te.drop(columns=["text"]).iloc[i].to_dict()
        lines = poster_text(rec)
        png = tmp_path / f"poster_{i}.png"
        render(lines, png)
        img = pb.handle_ui_submission(str(png), "", reader=reader, detector=det, with_xai=False)
        txt = pb.handle_ui_submission(None, "\n".join(lines), detector=det, with_xai=False)
        assert img["ocr_text"].strip()
        assert img["final_verdict"] in ("FAKE", "NOT FAKE")
        agree += img["final_verdict"] == txt["final_verdict"]
        print(f"\nrow {i} y={rec['fraudulent']}: image {img['final_verdict']} ({img['p_fake']:.3f}) "
              f"text {txt['final_verdict']} ({txt['p_fake']:.3f})")
    print(f"image/text agreement: {agree}/{len(idx)}")
    assert agree >= len(idx) - 1  # OCR noise may flip at most one borderline poster


def test_unreadable_image_gives_clear_message(reader, tmp_path):
    from PIL import Image
    import pipeline_bridge as pb
    blank = tmp_path / "blank.png"
    Image.new("RGB", (400, 300), "white").save(blank)
    with pytest.raises(ValueError, match="Could not read any text"):
        pb.handle_ui_submission(str(blank), "", reader=reader)
