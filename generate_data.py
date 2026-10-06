"""Step 3: synthetic data generator for the invoice reconciliation agent.

Creates:
  invoice_agent.db        SQLite DB with vendors, POs, delivery receipts and
                          previously processed invoices (for duplicate checks)
  data/invoices/*.pdf     30 invoices to feed the agent (clean + deliberately broken)
  data/labels.csv         ground truth: expected outcome and flags per invoice

Run:  python generate_data.py
"""
import copy
import csv
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).parent
DB_PATH = ROOT / "invoice_agent.db"
SCHEMA_PATH = ROOT / "schema.sql"
PDF_DIR = ROOT / "data" / "invoices"
LABELS_PATH = ROOT / "data" / "labels.csv"

GST_RATE = 0.18
HIGH_VALUE_LIMIT = 100000  # invoices above this need human review
random.seed(42)

VENDORS = [
    ("Sri Lakshmi IT Supplies", "accounts@srilakshmiit.example"),
    ("Deccan Office Solutions", "billing@deccanoffice.example"),
    ("Charminar Networks Pvt Ltd", "finance@charminarnet.example"),
    ("Hitech Hardware Traders", "invoices@hitechhw.example"),
    ("Godavari Stationers", "sales@godavaristat.example"),
    ("Nizam Electronics", "ar@nizamelectronics.example"),
]

CATALOG = {
    "Laptop 14 inch": 55000.0,
    "27 inch Monitor": 12000.0,
    "Mechanical Keyboard": 1500.0,
    "Office Chair": 6500.0,
    "WiFi Router": 4500.0,
    "Laser Printer": 18000.0,
    "Toner Cartridge": 3200.0,
    "USB Hub": 900.0,
}

SCENARIO_COUNTS = {
    "clean": 14,
    "clean_high_value": 2,
    "wrong_amount": 4,
    "over_billed": 4,
    "duplicate": 3,
    "missing_po": 3,
}

_po_counter = 1000
_inv_counter = 2000
_day_counter = 0


def next_po_number():
    global _po_counter
    _po_counter += 1
    return f"PO-{_po_counter}"


def next_invoice_number():
    global _inv_counter
    _inv_counter += 1
    return f"INV-{_inv_counter}"


def random_date():
    return date(2026, 8, 1) + timedelta(days=random.randint(0, 40))


def pick_lines(min_total, max_total, min_qty=1):
    """Pick 1-3 catalog items whose total (with GST) falls in the given range."""
    for _ in range(5000):
        items = random.sample(list(CATALOG), random.randint(1, 3))
        lines = []
        for item in items:
            price = CATALOG[item]
            cap = 12 if price < 20000 else 6
            qty = random.randint(min_qty, max(cap, min_qty))
            lines.append({"item": item, "qty": qty, "price": price})
        total = sum(l["qty"] * l["price"] for l in lines) * (1 + GST_RATE)
        if min_total <= total <= max_total:
            return lines
    raise RuntimeError("Could not pick lines in range")


def init_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    for name, email in VENDORS:
        conn.execute("INSERT INTO vendors (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    return conn


def make_po(conn, vendor_id, lines, shortfall_on_first=0):
    """Insert a PO with a delivery receipt. shortfall_on_first = units not delivered."""
    po_number = next_po_number()
    order_date = random_date()
    cur = conn.execute(
        "INSERT INTO purchase_orders (po_number, vendor_id, order_date) VALUES (?, ?, ?)",
        (po_number, vendor_id, order_date.isoformat()),
    )
    po_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO delivery_receipts (po_id, received_date) VALUES (?, ?)",
        (po_id, (order_date + timedelta(days=7)).isoformat()),
    )
    receipt_id = cur.lastrowid
    for i, l in enumerate(lines):
        conn.execute(
            "INSERT INTO po_lines (po_id, item, quantity, unit_price) VALUES (?, ?, ?, ?)",
            (po_id, l["item"], l["qty"], l["price"]),
        )
        received = l["qty"] - (shortfall_on_first if i == 0 else 0)
        conn.execute(
            "INSERT INTO receipt_lines (receipt_id, item, quantity_received) VALUES (?, ?, ?)",
            (receipt_id, l["item"], received),
        )
    conn.commit()
    return po_number


def build_invoice(vendor_name, po_number, lines, invoice_number=None, inv_date=None):
    subtotal = round(sum(l["qty"] * l["price"] for l in lines), 2)
    tax = round(subtotal * GST_RATE, 2)
    return {
        "invoice_number": invoice_number or next_invoice_number(),
        "vendor": vendor_name,
        "po_number": po_number,
        "date": inv_date or random_date(),
        "lines": lines,
        "subtotal": subtotal,
        "tax": tax,
        "total": round(subtotal + tax, 2),
    }


def insert_processed_invoice(conn, vendor_id, inv):
    """Store an invoice as 'already processed', so a resubmission is a duplicate."""
    cur = conn.execute(
        "INSERT INTO invoices (invoice_number, vendor_id, po_number, invoice_date, tax, total, source_file) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (inv["invoice_number"], vendor_id, inv["po_number"], inv["date"].isoformat(),
         inv["tax"], inv["total"], "previously_processed"),
    )
    for l in inv["lines"]:
        conn.execute(
            "INSERT INTO invoice_lines (invoice_id, item, quantity, unit_price) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, l["item"], l["qty"], l["price"]),
        )
    conn.commit()


def money(x):
    return f"Rs. {x:,.2f}"


def write_pdf(inv, template, path):
    """Write a text-based PDF. Two templates so extraction sees different layouts."""
    c = canvas.Canvas(str(path), pagesize=A4)
    _, h = A4
    y = h - 60

    if template == "A":
        labels = ("Invoice No:", "PO Number:", "Date:")
        date_text = inv["date"].isoformat()
        title = "INVOICE"
    else:
        labels = ("Bill #:", "Against PO:", "Dated:")
        date_text = inv["date"].strftime("%d %b %Y")
        title = "TAX INVOICE"

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, inv["vendor"])
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(545, y, title)
    y -= 40

    c.setFont("Helvetica", 11)
    for label, value in zip(labels, (inv["invoice_number"], inv["po_number"], date_text)):
        c.drawString(50, y, label)
        c.drawString(150, y, value)
        y -= 18
    y -= 20

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Description")
    c.drawRightString(330, y, "Qty")
    c.drawRightString(430, y, "Rate")
    c.drawRightString(545, y, "Amount")
    y -= 6
    c.line(50, y, 545, y)
    y -= 16

    c.setFont("Helvetica", 10)
    for l in inv["lines"]:
        c.drawString(50, y, l["item"])
        c.drawRightString(330, y, str(l["qty"]))
        c.drawRightString(430, y, money(l["price"]))
        c.drawRightString(545, y, money(l["qty"] * l["price"]))
        y -= 18

    y -= 10
    c.line(330, y, 545, y)
    y -= 18
    c.drawString(330, y, "Subtotal")
    c.drawRightString(545, y, money(inv["subtotal"]))
    y -= 18
    c.drawString(330, y, "GST (18%)")
    c.drawRightString(545, y, money(inv["tax"]))
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(330, y, "Total")
    c.drawRightString(545, y, money(inv["total"]))
    c.save()


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for old in PDF_DIR.glob("*.pdf"):
        old.unlink()

    conn = init_db()
    vendor_rows = conn.execute("SELECT id, name FROM vendors").fetchall()

    scenarios = [s for s, n in SCENARIO_COUNTS.items() for _ in range(n)]
    random.shuffle(scenarios)

    labels = []
    for idx, scenario in enumerate(scenarios, start=1):
        vendor_id, vendor_name = random.choice(vendor_rows)
        flags = ""
        outcome = ""

        if scenario == "clean":
            lines = pick_lines(3000, 95000)
            po = make_po(conn, vendor_id, lines)
            inv = build_invoice(vendor_name, po, lines)
            outcome = "approve"

        elif scenario == "clean_high_value":
            lines = pick_lines(110000, 400000)
            po = make_po(conn, vendor_id, lines)
            inv = build_invoice(vendor_name, po, lines)
            outcome, flags = "review", "high_value"

        elif scenario == "wrong_amount":
            lines = pick_lines(5000, 150000)
            po = make_po(conn, vendor_id, lines)
            billed = copy.deepcopy(lines)
            target = random.choice(billed)
            target["price"] = round(target["price"] * random.uniform(1.05, 1.25), 2)
            inv = build_invoice(vendor_name, po, billed)
            outcome, flags = "review", "wrong_amount"

        elif scenario == "over_billed":
            lines = pick_lines(5000, 150000, min_qty=4)
            short = random.randint(1, min(3, lines[0]["qty"] - 1))
            po = make_po(conn, vendor_id, lines, shortfall_on_first=short)
            inv = build_invoice(vendor_name, po, lines)  # bills the full ordered qty
            outcome, flags = "review", "over_billed"

        elif scenario == "duplicate":
            lines = pick_lines(3000, 95000)
            po = make_po(conn, vendor_id, lines)
            original = build_invoice(vendor_name, po, lines)
            insert_processed_invoice(conn, vendor_id, original)
            inv = copy.deepcopy(original)
            inv["date"] = original["date"] + timedelta(days=5)  # vendor resends it
            outcome, flags = "reject", "duplicate"

        elif scenario == "missing_po":
            lines = pick_lines(3000, 95000)
            if random.random() < 0.67:
                po = f"PO-{random.randint(9000, 9999)}"  # PO does not exist
            else:
                other = random.choice([v for v in vendor_rows if v[0] != vendor_id])
                po = make_po(conn, other[0], lines)  # PO belongs to another vendor
            inv = build_invoice(vendor_name, po, lines)
            outcome, flags = "reject", "missing_po"

        filename = f"invoice_{idx:02d}.pdf"
        write_pdf(inv, random.choice("AB"), PDF_DIR / filename)
        labels.append({
            "file": filename,
            "scenario": scenario,
            "expected_outcome": outcome,
            "expected_flags": flags,
            "vendor": inv["vendor"],
            "invoice_number": inv["invoice_number"],
            "po_number": inv["po_number"],
            "total": inv["total"],
        })

    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
        writer.writeheader()
        writer.writerows(labels)

    conn.close()
    print(f"Database: {DB_PATH}")
    print(f"Invoices: {len(labels)} PDFs in {PDF_DIR}")
    print(f"Labels:   {LABELS_PATH}")
    for s, n in SCENARIO_COUNTS.items():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()