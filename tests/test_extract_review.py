import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import agent
import extract
import review_app
from vendor_email import draft_vendor_email

GOOD = {
    "invoice_number": "INV-1", "vendor": "Acme Supplies", "po_number": "PO-1001", "invoice_date": "2026-08-15",
    "lines": [{"item": "Mouse", "quantity": 5, "unit_price": 500.0}],
    "subtotal": 2500.0, "tax": 450.0, "total": 2950.0,
}


def fake_client(replies):
    """Stands in for the Groq client: returns the given replies one by one."""
    replies = list(replies)

    def create(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return NS(choices=[NS(message=NS(content=reply))], usage=NS(prompt_tokens=10, completion_tokens=5))

    return NS(chat=NS(completions=NS(create=create)))


# ---------- extraction ----------

def test_valid_output_is_accepted_first_time():
    inv, attempts = extract.extract_invoice(fake_client([json.dumps(GOOD)]), "text")
    assert attempts == 1 and inv.total == 2950.0


def test_invalid_output_is_retried_then_accepted():
    replies = ["not json", json.dumps({**GOOD, "lines": []}), json.dumps(GOOD)]
    inv, attempts = extract.extract_invoice(fake_client(replies), "text")
    assert attempts == 3 and inv.invoice_number == "INV-1"


def test_gives_up_after_three_bad_attempts():
    with pytest.raises(extract.ExtractionError):
        extract.extract_invoice(fake_client(["{}", "{}", "{}"]), "text")


def test_negative_quantity_is_rejected():
    bad = {**GOOD, "lines": [{"item": "Mouse", "quantity": -1, "unit_price": 5.0}]}
    with pytest.raises(ValidationError):
        extract.InvoiceData.model_validate(bad)


def test_token_usage_is_counted():
    extract.USAGE.update(prompt=0, completion=0)
    extract.extract_invoice(fake_client([json.dumps(GOOD)]), "text")
    assert extract.USAGE == {"prompt": 10, "completion": 5}


def test_arithmetic_warnings_flag_inconsistent_totals():
    inv = extract.InvoiceData.model_validate({**GOOD, "total": 9999.0})
    assert extract.arithmetic_warnings(inv)
    assert extract.arithmetic_warnings(extract.InvoiceData.model_validate(GOOD)) == []


# ---------- vendor email drafts ----------

def test_email_falls_back_to_template_without_a_client():
    e = draft_vendor_email(None, "Acme", "ap@acme.example", "INV-1", "PO-1001", "Price is wrong.")
    assert e["source"] == "template" and "INV-1" in e["subject"] and "Price is wrong." in e["body"]


def test_email_uses_llm_when_it_returns_valid_json():
    client = fake_client([json.dumps({"subject": "Question", "body": "Hello"})])
    e = draft_vendor_email(client, "Acme", "ap@acme.example", "INV-1", "PO-1001", "x")
    assert e["source"] == "llm" and e["subject"] == "Question"


@pytest.mark.parametrize("reply", ["garbage", RuntimeError("down"), json.dumps({"subject": "", "body": ""})])
def test_email_falls_back_when_llm_fails(reply):
    e = draft_vendor_email(fake_client([reply]), "Acme", "x@y.example", "INV-1", "PO-1001", "x")
    assert e["source"] == "template"


# ---------- review web app ----------

@pytest.fixture
def client(db_path, conn, make_invoice, monkeypatch):
    monkeypatch.setattr(review_app, "DB_PATH", db_path)
    monkeypatch.setattr(review_app, "get_client", lambda: None)
    monkeypatch.setattr(review_app, "REVIEW_PASSWORD", None)
    # one invoice waiting for a human
    inv = make_invoice(lines=[{"item": "Mouse", "quantity": 5, "unit_price": 700.0}])
    agent.build_graph(conn, lambda p: inv).invoke({"pdf": "invoice_01.pdf"})
    return TestClient(review_app.app)


def pending_id(conn):
    return conn.execute("SELECT id FROM decisions WHERE decided_by = 'pending_human'").fetchone()[0]


def test_queue_lists_waiting_invoice(client):
    page = client.get("/")
    assert page.status_code == 200 and "INV-NEW" in page.text and "wrong_amount" in page.text


def test_hostile_text_is_escaped(client, conn):
    conn.execute("UPDATE decisions SET explanation = '<script>alert(1)</script>'")
    conn.commit()
    page = client.get(f"/review/{pending_id(conn)}").text
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_draft_email_is_shown_on_the_review_page(client, conn):
    page = client.post(f"/draft/{pending_id(conn)}", follow_redirects=True)
    assert page.status_code == 200 and "Draft vendor email" in page.text and "Subject:" in page.text


def test_human_can_approve_once_only(client, conn):
    did = pending_id(conn)
    ok = client.post(f"/decide/{did}", data={"action": "approve", "reviewer": "Ravi", "note": "Confirmed"},
                     follow_redirects=False)
    assert ok.status_code == 303
    assert conn.execute("SELECT outcome, decided_by FROM decisions WHERE id = ?", (did,)).fetchone() == ("approve", "Ravi")
    again = client.post(f"/decide/{did}", data={"action": "reject", "reviewer": "Other"})
    assert again.status_code == 409


@pytest.mark.parametrize("data", [
    {"action": "hack", "reviewer": "Ravi"},
    {"action": "approve", "reviewer": "   "},
])
def test_bad_decisions_are_refused(client, conn, data):
    assert client.post(f"/decide/{pending_id(conn)}", data=data).status_code == 400


def test_unknown_decision_is_404(client):
    assert client.get("/review/99999").status_code == 404


def test_cross_site_post_is_blocked(client, conn):
    did = pending_id(conn)
    data = {"action": "approve", "reviewer": "Ravi"}
    assert client.post(f"/decide/{did}", data=data, headers={"origin": "http://evil.example"}).status_code == 403
    assert client.post(f"/decide/{did}", data=data, headers={"origin": "http://testserver"},
                       follow_redirects=False).status_code == 303


def test_login_is_required_when_a_password_is_set(client, monkeypatch):
    monkeypatch.setattr(review_app, "REVIEW_PASSWORD", "secret")
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("reviewer", "wrong")).status_code == 401
    assert client.get("/", auth=("reviewer", "secret")).status_code == 200
    assert client.get("/health").status_code == 200  # health check stays open