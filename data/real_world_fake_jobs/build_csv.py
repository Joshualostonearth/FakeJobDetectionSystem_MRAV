"""Build fake_jobs.csv and texts/*.txt from fake_jobs.json.

The CSV uses the same column names as test_dataset.csv where they overlap
(offer_id, company_name, job_role, recruiter_email, domain, location,
source_type, doc_content, is_fraudulent, true_label, category) plus
provenance columns. All rows are real scams: is_fraudulent=1, true_label=0.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    rows = json.loads((HERE / "fake_jobs.json").read_text(encoding="utf-8"))
    texts = HERE / "texts"
    texts.mkdir(exist_ok=True)
    for r in rows:
        for img in r["image_files"]:
            assert (HERE / img).is_file(), f"{r['offer_id']}: missing {img}"
        (texts / f"{r['offer_id']}.txt").write_text(r["text"] + "\n", encoding="utf-8")

    cols = ["offer_id", "company_name", "job_role", "recruiter_email", "domain",
            "location", "source_type", "doc_content", "is_fraudulent", "true_label",
            "category", "scam_type", "date", "red_flags", "image_files",
            "source_publisher", "source_url"]
    with open(HERE / "fake_jobs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                **{k: r[k] for k in ("offer_id", "company_name", "job_role", "recruiter_email",
                                     "domain", "location", "scam_type", "date",
                                     "source_publisher", "source_url")},
                "source_type": r["channel"],
                "doc_content": r["text"],
                "is_fraudulent": 1,
                "true_label": 0,
                "category": "Real-world Scam",
                "red_flags": "; ".join(r["red_flags"]),
                "image_files": ";".join(r["image_files"]),
            })
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
