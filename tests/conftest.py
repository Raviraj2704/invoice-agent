import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def db_path(tmp_path):
    """A small hand-made database: one PO, partial delivery, one previously processed invoice."""
    path = tmp_path / "test.db"
    c = sqlite3.connect(path)
    c.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    c.execute("INSERT INTO vendors (id, name, email) VALUES (1, 'Acme Supplies', 'ap@acme.example'), "
              "(2, 'Beta Traders', 'ap@beta.example')")
    c.execute("INSERT INTO purchase_orders (id, po_number, vendor_id, order_date) VALUES (1, 'PO-1001', 1, '2026-08-01')")
    c.executemany("INSERT INTO po_lines (po_id, item, quantity, unit_price) VALUES (?, ?, ?, ?)",
                  [(1, "Laptop", 10, 50000.0), (1, "Mouse", 5, 500.0)])
    c.execute("INSERT INTO delivery_receipts (id, po_id, received_date) VALUES (1, 1, '2026-08-08')")
    c.executemany("INSERT INTO receipt_lines (receipt_id, item, quantity_received) VALUES (?, ?, ?)",
                  [(1, "Laptop", 8), (1, "Mouse", 5)])  # only 8 of 10 laptops delivered
    c.execute("INSERT INTO invoices (id, invoice_number, vendor_id, po_number, invoice_date, tax, total, source_file) "
              "VALUES (1, 'INV-OLD', 1, 'PO-1001', '2026-08-10', 90, 590, 'previously_processed')")
    c.commit()
    c.close()
    return path


@pytest.fixture
def conn(db_path):
    c = sqlite3.connect(db_path)
    yield c
    c.close()


@pytest.fixture
def make_invoice():
    def _make(**overrides):
        inv = {
            "invoice_number": "INV-NEW", "vendor": "Acme Supplies", "po_number": "PO-1001",
            "invoice_date": "2026-08-15",
            "lines": [{"item": "Mouse", "quantity": 5, "unit_price": 500.0}],
            "subtotal": 2500.0, "tax": 450.0, "total": 2950.0,
        }
        inv.update(overrides)
        return inv
    return _make