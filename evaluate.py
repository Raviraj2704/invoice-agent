"""Step 8b: evaluate the whole agent on 30 normal + 6 attack invoices, for one or more models.

Reports per model: decision accuracy (normal and attack sets), extraction failures,
average seconds per invoice, tokens used and estimated cost per invoice.

Run:  python make_attacks.py          (once, creates the attack invoices)
      python evaluate.py              (compares gpt-oss-120b and gpt-oss-20b)
      python evaluate.py --models openai/gpt-oss-120b
Output: printed table + data/evaluation.csv
"""
import argparse
import contextlib
import csv
import io
import os
import sqlite3
import sys
import time

import extract
from agent import DB_PATH, build_graph, reset_run_data, run_batch
from extract import PAUSE_SECONDS, PDF_DIR, ROOT, extract_invoice, read_pdf_text

ATTACK_DIR = ROOT / "data" / "attacks"
OUT_PATH = ROOT / "data" / "evaluation.csv"

# USD per 1M tokens (input, output) from Groq's pricing page in Oct 2026. Check it before quoting.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def read_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def evaluate_model(conn, client, model, pdfs, labels, pause=0):
    os.environ["GROQ_MODEL"] = model
    extract.USAGE.update(prompt=0, completion=0)
    seconds = []

    def extract_fn(path):
        start = time.time()
        inv, _ = extract_invoice(client, read_pdf_text(path))
        seconds.append(time.time() - start)
        return inv.model_dump(mode="json")

    reset_run_data(conn)
    graph = build_graph(conn, extract_fn)
    with contextlib.redirect_stdout(io.StringIO()):  # hide the per-file lines
        rows = run_batch(graph, pdfs, labels, pause=pause)

    def stats(subset):
        n = len(subset)
        return n, sum(r["outcome_ok"] for r in subset), sum("extraction_failed" in r["flags"] for r in subset)

    normal = [r for r in rows if not r["file"].startswith("attack_")]
    attack = [r for r in rows if r["file"].startswith("attack_")]
    n_all = len(rows)
    tokens_in, tokens_out = extract.USAGE["prompt"], extract.USAGE["completion"]
    price = PRICES.get(model)
    cost = (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000 if price else None

    n_normal, ok_normal, fail_normal = stats(normal)
    n_attack, ok_attack, fail_attack = stats(attack)
    return {
        "model": model,
        "normal_correct": f"{ok_normal}/{n_normal}",
        "attack_correct": f"{ok_attack}/{n_attack}",
        "extraction_failures": fail_normal + fail_attack,
        "avg_seconds": round(sum(seconds) / len(seconds), 2) if seconds else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_per_invoice_usd": round(cost / n_all, 6) if cost is not None else None,
        "wrong_files": ",".join(r["file"] for r in rows if not r["outcome_ok"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="openai/gpt-oss-120b,openai/gpt-oss-20b")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not found in .env")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    labels = read_labels(ROOT / "data" / "labels.csv")
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if (ATTACK_DIR / "labels.csv").exists():
        labels.update(read_labels(ATTACK_DIR / "labels.csv"))
        pdfs += sorted(ATTACK_DIR.glob("*.pdf"))
    else:
        print("No attack invoices found. Run make_attacks.py first to include them.")

    conn = sqlite3.connect(DB_PATH)
    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"Evaluating {model} on {len(pdfs)} invoices ...")
        results.append(evaluate_model(conn, client, model, pdfs, labels, pause=PAUSE_SECONDS))
    conn.close()

    print("\n--- Results ---")
    for r in results:
        print(f"\n{r['model']}")
        for k, v in r.items():
            if k != "model":
                print(f"  {k}: {v}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()