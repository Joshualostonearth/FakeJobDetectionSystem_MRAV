"""
eval_llm_stage.py — Run (and cache) the Ollama-backed steps of the pipeline for an
evaluation CSV, so the deterministic steps can be re-scored without re-calling the LLM.

    python eval_llm_stage.py extract   <csv> <out.jsonl>   # Obj 1a entity extraction
    python eval_llm_stage.py align     <csv> <out.jsonl>   # Obj 2 M-RAV verdict + per-doc alignment triples

Every record is appended as soon as it is produced; re-running skips finished job_ids.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

from evaluate_test_split import posting_to_text


def _done_ids(out: Path) -> set:
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {json.loads(line)["job_id"] for line in f}


def extract(csv_path: str, out_path: str):
    from models import extract_entities_from_text, OLLAMA_MODEL

    out = Path(out_path)
    df = pd.read_csv(csv_path)
    done = _done_ids(out)
    todo = df[~df["job_id"].isin(done)]
    print(f"[extract] model={OLLAMA_MODEL} rows={len(df)} done={len(done)} todo={len(todo)}", flush=True)
    t_start = time.time()
    for n, (_, r) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        ent = extract_entities_from_text(posting_to_text(r.to_dict()))
        rec = {"job_id": int(r["job_id"]), "model": OLLAMA_MODEL,
               "entities": ent.model_dump(), "seconds": round(time.time() - t0, 2)}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if n % 10 == 0 or n == len(todo):
            rate = (time.time() - t_start) / n
            print(f"[extract] {len(done) + n}/{len(df)}  {rate:.1f}s/row  ETA {rate * (len(todo) - n) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    cmd, csv_path, out_path = sys.argv[1:4]
    {"extract": extract}[cmd](csv_path, out_path)
