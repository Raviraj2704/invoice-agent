"""Tests for the upload page."""
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import review_app
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@pytest.fixture
def sample_pdf():
    """Create a minimal valid invoice PDF in memory."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Test Vendor")
    c.drawString(500, 750, "INVOICE")
    c.setFont("Helvetica", 11)
    c.drawString(50, 700, "Invoice No: TST-001")
    c.drawString(50, 680, "PO Number: PO-1")
    c.drawString(50, 660, "Date: 2026-09-10")
    c.drawString(50, 600, "Mouse")
    c.drawRightString(300, 600, "10")
    c.drawRightString(400, 600, "Rs. 500.0")
    c.drawRightString(500, 600, "Rs. 5000.0")
    c.drawString(50, 500, "Subtotal")
    c.drawRightString(500, 500, "Rs. 5000.0")
    c.drawString(50, 480, "GST (18%)")
    c.drawRightString(500, 480, "Rs. 900.0")
    c.drawString(50, 460, "Total")
    c.drawRightString(500, 460, "Rs. 5900.0")
    c.save()
    buf.seek(0)
    return buf


@pytest.fixture
def client_with_db(tmp_path, monkeypatch):
    """Client with a test database."""
    import sqlite3
    from generate_data import DB_PATH, ROOT
    from agent import reset_run_data
    
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(review_app, "DB_PATH", db_path)
    monkeypatch.setattr(review_app, "get_client", lambda: None)
    monkeypatch.setattr(review_app, "REVIEW_PASSWORD", None)
    
    c = sqlite3.connect(db_path)
    c.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    c.execute("INSERT INTO vendors (id, name, email) VALUES (1, 'Test Vendor', 'test@example.com')")
    c.execute("INSERT INTO purchase_orders (id, po_number, vendor_id, order_date) VALUES (1, 'PO-1', 1, '2026-09-01')")
    c.execute("INSERT INTO po_lines (po_id, item, quantity, unit_price) VALUES (1, 'Mouse', 10, 500.0)")
    c.execute("INSERT INTO delivery_receipts (id, po_id, received_date) VALUES (1, 1, '2026-09-08')")
    c.execute("INSERT INTO receipt_lines (receipt_id, item, quantity_received) VALUES (1, 'Mouse', 10)")
    c.commit()
    c.close()
    
    return TestClient(review_app.app)


def test_upload_page_loads(client_with_db):
    """Upload page should load and show the form."""
    r = client_with_db.get("/upload-page")
    assert r.status_code == 200
    assert "Drag a PDF here" in r.text


def test_upload_pdf_shows_decision(client_with_db, sample_pdf, monkeypatch):
    """Uploading a PDF should show the agent's decision."""
    # Mock the extraction to avoid needing Groq
    def fake_extract(client, text):
        from extract import InvoiceData
        inv = InvoiceData(
            invoice_number="TST-001",
            vendor="Test Vendor",
            po_number="PO-1",
            invoice_date="2026-09-10",
            lines=[{"item": "Mouse", "quantity": 10, "unit_price": 500.0}],
            subtotal=5000.0,
            tax=900.0,
            total=5900.0,
        )
        return inv, 1
    
    # Assuming review_app imports it as `from extract import extract_invoice` or similar.
    # The safest way is to mock it at its source module.
    monkeypatch.setattr("extract.extract_invoice", fake_extract)
    
    r = client_with_db.post(
        "/upload",
        files={"file": ("invoice.pdf", sample_pdf, "application/pdf")}
    )
    assert r.status_code == 200
    assert "TST-001" in r.text
    assert "approve" in r.text or "review" in r.text


def test_upload_non_pdf_is_rejected(client_with_db):
    """Uploading a non-PDF should show an error."""
    r = client_with_db.post(
        "/upload",
        files={"file": ("data.txt", io.BytesIO(b"not a pdf"), "text/plain")}
    )
    assert r.status_code == 200
    assert "PDF file only" in r.text


def test_upload_requires_login_when_password_is_set(client_with_db, monkeypatch):
    """Upload page should require login if password is set."""
    monkeypatch.setattr(review_app, "REVIEW_PASSWORD", "secret")
    assert client_with_db.get("/upload-page").status_code == 401
    assert client_with_db.get("/upload-page", auth=("reviewer", "secret")).status_code == 200