"""Step 7: human review screen for invoices the agent could not decide alone.

Run:   python review_app.py
Open:  http://127.0.0.1:8000

Local use only: there is no login, so do not expose this to the internet as it is.
All invoice text comes from untrusted PDFs, so everything is HTML-escaped before display.
"""
import html
import json
import os
import secrets
import sqlite3
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from extract import ROOT
from vendor_email import draft_vendor_email

load_dotenv(ROOT / ".env")
DB_PATH = ROOT / "invoice_agent.db"

# Login: set REVIEW_PASSWORD in .env to require HTTP Basic login (username REVIEW_USER, default "reviewer").
# With APP_ENV=production the app refuses to start without a password.
REVIEW_USER = os.getenv("REVIEW_USER", "reviewer")
REVIEW_PASSWORD = os.getenv("REVIEW_PASSWORD")
if os.getenv("APP_ENV") == "production" and not REVIEW_PASSWORD:
    raise RuntimeError("Set REVIEW_PASSWORD when APP_ENV=production")

security = HTTPBasic(auto_error=False)


def require_login(creds: HTTPBasicCredentials | None = Depends(security)):
    if not REVIEW_PASSWORD:
        return  # no password configured: local development mode
    ok = (
        creds is not None
        and secrets.compare_digest(creds.username.encode(), REVIEW_USER.encode())
        and secrets.compare_digest(creds.password.encode(), REVIEW_PASSWORD.encode())
    )
    if not ok:
        raise HTTPException(401, "Login required", headers={"WWW-Authenticate": "Basic"})


def check_origin(request: Request):
    """Block cross-site form posts: browsers send Origin on POST and it must match this host."""
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "Cross-site request blocked")


AUTH = [Depends(require_login)]
app = FastAPI(title="Invoice Review")

CSS = """
body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;color:#1f2937;background:#f9fafb}
h1,h2{margin:.4em 0}a{color:#2563eb;text-decoration:none}
table{width:100%;border-collapse:collapse;background:#fff;margin:12px 0}
th,td{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:left;font-size:14px}
th{background:#f3f4f6}.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin:12px 0}
.stats span{display:inline-block;margin-right:16px;font-weight:600}
.flag{background:#fef3c7;border-radius:4px;padding:2px 6px;margin-right:4px;font-size:12px}
button{padding:8px 14px;border:0;border-radius:6px;color:#fff;cursor:pointer;font-size:14px;margin-right:6px}
.ok{background:#16a34a}.no{background:#dc2626}.neutral{background:#2563eb}
input,textarea{width:100%;padding:8px;margin:4px 0 10px;box-sizing:border-box;font:inherit}
pre{white-space:pre-wrap;background:#f3f4f6;padding:10px;border-radius:6px}
"""

DECISION_SQL = """
SELECT d.id, d.outcome, d.flags, d.explanation, d.decided_by,
       i.id AS invoice_id, i.invoice_number, i.po_number, i.invoice_date, i.tax, i.total,
       v.name AS vendor, v.email AS vendor_email
FROM decisions d
JOIN invoices i ON i.id = d.invoice_id
LEFT JOIN vendors v ON v.id = i.vendor_id
"""


def esc(x):
    return html.escape("" if x is None else str(x))


def money(x):
    return "-" if x is None else f"Rs. {x:,.2f}"


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def page(title, body):
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"
    )


def flag_tags(flags):
    return "".join(f"<span class='flag'>{esc(f)}</span>" for f in (flags or "").split(",") if f)


def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    from groq import Groq
    return Groq(api_key=key)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, dependencies=AUTH)
def home():
    conn = db()
    pending = conn.execute(DECISION_SQL + " WHERE d.decided_by = 'pending_human' ORDER BY d.id").fetchall()
    stats = conn.execute(
        "SELECT SUM(decided_by='agent' AND outcome='approve'), SUM(decided_by='agent' AND outcome='reject'), "
        "SUM(decided_by NOT IN ('agent','pending_human')) FROM decisions"
    ).fetchone()
    auto_ok, auto_reject, by_humans = (s or 0 for s in stats)

    rows = "".join(
        f"<tr><td><a href='/review/{d['id']}'>{esc(d['invoice_number'])}</a></td><td>{esc(d['vendor'])}</td>"
        f"<td>{money(d['total'])}</td><td>{flag_tags(d['flags'])}</td></tr>"
        for d in pending
    ) or "<tr><td colspan='4'>Nothing waiting for review.</td></tr>"

    body = (
        "<h1>Invoice review queue</h1>"
        f"<div class='stats'><span>Auto-approved: {auto_ok}</span><span>Auto-rejected: {auto_reject}</span>"
        f"<span>Decided by humans: {by_humans}</span><span>Waiting: {len(pending)}</span></div>"
        "<table><tr><th>Invoice</th><th>Vendor</th><th>Total</th><th>Flags</th></tr>" + rows + "</table>"
    )
    return page("Invoice review queue", body)


@app.get("/review/{decision_id}", response_class=HTMLResponse, dependencies=AUTH)
def review(decision_id: int):
    conn = db()
    d = conn.execute(DECISION_SQL + " WHERE d.id = ?", (decision_id,)).fetchone()
    if d is None:
        raise HTTPException(404, "Decision not found")
    lines = conn.execute(
        "SELECT item, quantity, unit_price FROM invoice_lines WHERE invoice_id = ?", (d["invoice_id"],)
    ).fetchall()
    draft_row = conn.execute(
        "SELECT detail FROM audit_log WHERE invoice_id = ? AND step = 'email_draft' ORDER BY id DESC LIMIT 1",
        (d["invoice_id"],),
    ).fetchone()

    line_rows = "".join(
        f"<tr><td>{esc(l['item'])}</td><td>{l['quantity']}</td><td>{money(l['unit_price'])}</td>"
        f"<td>{money(l['quantity'] * l['unit_price'])}</td></tr>"
        for l in lines
    ) or "<tr><td colspan='4'>No line items.</td></tr>"

    body = (
        "<a href='/'>&larr; Back to queue</a>"
        f"<h1>Invoice {esc(d['invoice_number'])}</h1>"
        f"<div class='card'><b>Vendor:</b> {esc(d['vendor'])}<br><b>PO:</b> {esc(d['po_number'])}<br>"
        f"<b>Date:</b> {esc(d['invoice_date'])}<br><b>Tax:</b> {money(d['tax'])}<br>"
        f"<b>Total:</b> {money(d['total'])}</div>"
        "<table><tr><th>Item</th><th>Qty</th><th>Rate</th><th>Amount</th></tr>" + line_rows + "</table>"
        f"<div class='card'><b>Agent says:</b> {esc(d['outcome'])} {flag_tags(d['flags'])}"
        f"<p>{esc(d['explanation'])}</p></div>"
    )

    if draft_row:
        e = json.loads(draft_row["detail"])
        body += (
            "<div class='card'><h2>Draft vendor email</h2>"
            f"<b>To:</b> {esc(e.get('to'))}<br><b>Subject:</b> {esc(e.get('subject'))}"
            f"<pre>{esc(e.get('body'))}</pre>"
            f"<small>Written by: {esc(e.get('source'))}. Draft only: copy it into your email app after checking it.</small></div>"
        )

    if d["decided_by"] == "pending_human":
        body += (
            f"<form method='post' action='/draft/{d['id']}'><button class='neutral'>Draft vendor email</button></form>"
            f"<form method='post' action='/decide/{d['id']}' class='card'><h2>Your decision</h2>"
            "Your name<input name='reviewer' required maxlength='50'>"
            "Note (optional)<textarea name='note' rows='3' maxlength='500'></textarea>"
            "<button class='ok' name='action' value='approve'>Approve</button>"
            "<button class='no' name='action' value='reject'>Reject</button></form>"
        )
    else:
        body += f"<div class='card'>Final decision: <b>{esc(d['outcome'])}</b> by {esc(d['decided_by'])}</div>"
    return page(f"Invoice {d['invoice_number']}", body)


@app.post("/draft/{decision_id}", dependencies=AUTH)
def draft(request: Request, decision_id: int):
    check_origin(request)
    conn = db()
    d = conn.execute(DECISION_SQL + " WHERE d.id = ?", (decision_id,)).fetchone()
    if d is None:
        raise HTTPException(404, "Decision not found")
    email = draft_vendor_email(
        get_client(), d["vendor"], d["vendor_email"], d["invoice_number"], d["po_number"], d["explanation"]
    )
    conn.execute(
        "INSERT INTO audit_log (invoice_id, step, detail) VALUES (?, 'email_draft', ?)",
        (d["invoice_id"], json.dumps(email)),
    )
    conn.commit()
    return RedirectResponse(f"/review/{decision_id}", status_code=303)


@app.post("/decide/{decision_id}", dependencies=AUTH)
def decide(request: Request, decision_id: int, action: str = Form(...), reviewer: str = Form(...), note: str = Form("")):
    check_origin(request)
    if action not in ("approve", "reject"):
        raise HTTPException(400, "Invalid action")
    reviewer, note = reviewer.strip()[:50], note.strip()[:500]
    if not reviewer:
        raise HTTPException(400, "Reviewer name is required")

    conn = db()
    d = conn.execute(
        "SELECT outcome, decided_by, explanation, invoice_id FROM decisions WHERE id = ?", (decision_id,)
    ).fetchone()
    if d is None:
        raise HTTPException(404, "Decision not found")
    if d["decided_by"] != "pending_human":
        raise HTTPException(409, "This invoice was already decided")

    explanation = d["explanation"] + (f" | Human note ({reviewer}): {note}" if note else "")
    conn.execute(
        "UPDATE decisions SET outcome = ?, decided_by = ?, explanation = ? WHERE id = ?",
        (action, reviewer, explanation, decision_id),
    )
    conn.execute(
        "INSERT INTO audit_log (invoice_id, step, detail) VALUES (?, 'human_decision', ?)",
        (d["invoice_id"], json.dumps({"agent_outcome": d["outcome"], "human_outcome": action, "reviewer": reviewer})),
    )
    conn.commit()
    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)