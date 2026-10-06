"""Step 6: the LangGraph agent.

Flow:  extract -> match -> route -> approve | human_queue | reject -> END
       (if extraction fails: extract -> failed -> END, sent to a human)

Every invoice gets a row in `decisions` and entries in `audit_log`.

Run:  python agent.py             (all 30 PDFs, then scores against data/labels.csv)
      python agent.py --limit 5   (quick test)
      python agent.py --keep      (do NOT clear earlier results before running)
"""
import argparse
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from extract import PAUSE_SECONDS, PDF_DIR, ROOT, extract_invoice, load_labels, read_pdf_text
from matcher import match_invoice

DB_PATH = ROOT / "invoice_agent.db"
RESULTS_PATH = ROOT / "data" / "results.csv"


class State(TypedDict, total=False):
    pdf: str
    invoice: dict
    outcome: str
    flags: list
    explanation: str
    error: str


# ---------- database helpers ----------

def reset_run_data(conn):
    """Clear earlier agent results but keep the seeded 'previously processed' invoices."""
    conn.execute(
        "DELETE FROM invoice_lines WHERE invoice_id IN "
        "(SELECT id FROM invoices WHERE source_file != 'previously_processed')"
    )
    conn.execute("DELETE FROM decisions")
    conn.execute("DELETE FROM audit_log")
    conn.execute("DELETE FROM invoices WHERE source_file != 'previously_processed'")
    conn.commit()


def log(conn, invoice_id, step, detail):
    conn.execute(
        "INSERT INTO audit_log (invoice_id, step, detail) VALUES (?, ?, ?)",
        (invoice_id, step, detail),
    )


def store_invoice(conn, inv, filename):
    """Return the invoice id, inserting the invoice if it is new."""
    if inv is None:  # extraction failed: keep a placeholder so the decision has a home
        inv = {"invoice_number": f"UNREADABLE-{filename}", "vendor": None, "po_number": None,
               "invoice_date": None, "tax": None, "total": None, "lines": []}
    row = conn.execute("SELECT id FROM vendors WHERE name = ?", (inv["vendor"],)).fetchone()
    vendor_id = row[0] if row else None
    existing = conn.execute(
        "SELECT id FROM invoices WHERE vendor_id IS ? AND invoice_number = ?",
        (vendor_id, inv["invoice_number"]),
    ).fetchone()
    if existing:
        return existing[0]
    cur = conn.execute(
        "INSERT INTO invoices (invoice_number, vendor_id, po_number, invoice_date, tax, total, source_file) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (inv["invoice_number"], vendor_id, inv["po_number"], inv["invoice_date"],
         inv["tax"], inv["total"], filename),
    )
    for l in inv["lines"]:
        conn.execute(
            "INSERT INTO invoice_lines (invoice_id, item, quantity, unit_price) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, l["item"], l["quantity"], l["unit_price"]),
        )
    return cur.lastrowid


def record(conn, state, outcome, decided_by):
    filename = Path(state["pdf"]).name
    invoice_id = store_invoice(conn, state.get("invoice"), filename)
    conn.execute(
        "INSERT INTO decisions (invoice_id, outcome, flags, explanation, decided_by) VALUES (?, ?, ?, ?, ?)",
        (invoice_id, outcome, ",".join(state.get("flags", [])), state.get("explanation", ""), decided_by),
    )
    log(conn, invoice_id, "decide", f"{filename}: {outcome} ({decided_by})")
    conn.commit()


# ---------- the graph ----------

def build_graph(conn, extract_fn):
    """extract_fn(pdf_path) -> invoice dict. Passed in so it can be swapped in tests."""

    def extract_node(state):
        try:
            return {"invoice": extract_fn(state["pdf"])}
        except Exception as err:
            return {"error": str(err)}

    def failed_node(state):
        state = {**state, "flags": ["extraction_failed"],
                 "explanation": f"Could not extract the invoice: {state['error']}"}
        record(conn, state, "review", "pending_human")
        return {"outcome": "review", "flags": state["flags"], "explanation": state["explanation"]}

    def match_node(state):
        result = match_invoice(conn, state["invoice"])
        return {"outcome": result["outcome"], "flags": result["flags"], "explanation": result["explanation"]}

    def approve_node(state):
        record(conn, state, "approve", "agent")
        return {}

    def human_queue_node(state):
        record(conn, state, "review", "pending_human")  # a person approves this in a later step
        return {}

    def reject_node(state):
        record(conn, state, "reject", "agent")
        return {}

    g = StateGraph(State)
    g.add_node("extract", extract_node)
    g.add_node("failed", failed_node)
    g.add_node("match", match_node)
    g.add_node("approve", approve_node)
    g.add_node("human_queue", human_queue_node)
    g.add_node("reject", reject_node)

    g.add_edge(START, "extract")
    g.add_conditional_edges(
        "extract", lambda s: "failed" if s.get("error") else "match",
        {"failed": "failed", "match": "match"},
    )
    g.add_conditional_edges(
        "match", lambda s: s["outcome"],
        {"approve": "approve", "review": "human_queue", "reject": "reject"},
    )
    for node in ("failed", "approve", "human_queue", "reject"):
        g.add_edge(node, END)
    return g.compile()


# ---------- batch run ----------

def run_batch(graph, pdfs, labels, pause=0):
    rows = []
    for pdf in pdfs:
        final = graph.invoke({"pdf": str(pdf)})
        label = labels[pdf.name]
        expected_flags = [f for f in label["expected_flags"].split(",") if f]
        row = {
            "file": pdf.name,
            "expected": label["expected_outcome"],
            "got": final["outcome"],
            "flags": ",".join(final.get("flags", [])),
            "outcome_ok": final["outcome"] == label["expected_outcome"],
            "flags_ok": all(f in final.get("flags", []) for f in expected_flags),
            "explanation": final.get("explanation", ""),
        }
        rows.append(row)
        mark = "ok " if row["outcome_ok"] and row["flags_ok"] else "BAD"
        print(f"{mark} {pdf.name}: expected {row['expected']}, got {row['got']} [{row['flags']}]")
        if pause:
            time.sleep(pause)
    return rows


def print_summary(rows):
    n = len(rows)
    outcome_hits = sum(r["outcome_ok"] for r in rows)
    flag_hits = sum(r["flags_ok"] for r in rows)
    print("\n--- Summary ---")
    print(f"Decision accuracy: {outcome_hits}/{n} ({100 * outcome_hits / n:.0f}%)")
    print(f"Expected flags found: {flag_hits}/{n} ({100 * flag_hits / n:.0f}%)")
    bad = [r for r in rows if not (r["outcome_ok"] and r["flags_ok"])]
    for r in bad:
        print(f"  check {r['file']}: expected {r['expected']}, got {r['got']} [{r['flags']}] - {r['explanation']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep", action="store_true", help="do not clear earlier results first")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not found in .env")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def extract_fn(path):
        inv, _ = extract_invoice(client, read_pdf_text(path))
        return inv.model_dump(mode="json")

    conn = sqlite3.connect(DB_PATH)
    if not args.keep:
        reset_run_data(conn)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        sys.exit("No PDFs found. Run generate_data.py first.")

    graph = build_graph(conn, extract_fn)
    rows = run_batch(graph, pdfs, load_labels(), pause=PAUSE_SECONDS)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"\nDetails saved to {RESULTS_PATH}")
    conn.close()


if __name__ == "__main__":
    main()