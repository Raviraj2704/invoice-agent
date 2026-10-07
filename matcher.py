"""Step 5: rule-based invoice matching. Plain Python, no LLM, so the maths is exact.

match_invoice(conn, invoice) -> {"outcome": approve|review|reject, "flags": [...], "explanation": "..."}

Checks:
  duplicate      same vendor + invoice number already in the invoices table
  missing_po     PO number not found, or it belongs to a different vendor
  wrong_amount   invoiced unit price differs from the PO price by more than PRICE_TOLERANCE
  over_billed    invoiced quantity is more than the quantity delivered
  item_not_on_po invoiced item does not exist on the PO
  unknown_vendor vendor name not in the vendors table
  high_value     invoice total above HIGH_VALUE_LIMIT

Outcome: reject if duplicate or missing_po; review if any other flag; else approve.
"""

HIGH_VALUE_LIMIT = 100000
PRICE_TOLERANCE = 1.0
REJECT_FLAGS = {"duplicate", "missing_po"}

from extract import arithmetic_warnings

def money(x):
    return f"Rs. {x:,.2f}"


def match_invoice(conn, inv):
    flags, notes = [], []

    row = conn.execute("SELECT id FROM vendors WHERE name = ?", (inv["vendor"],)).fetchone()
    vendor_id = row[0] if row else None
    if vendor_id is None:
        flags.append("unknown_vendor")
        notes.append(f"Vendor '{inv['vendor']}' is not in the vendor list.")

    # Duplicate check
    if vendor_id is not None:
        dup = conn.execute(
            "SELECT 1 FROM invoices WHERE vendor_id = ? AND invoice_number = ?",
            (vendor_id, inv["invoice_number"]),
        ).fetchone()
        if dup:
            flags.append("duplicate")
            notes.append(f"Invoice {inv['invoice_number']} from {inv['vendor']} was already processed.")

    # PO check
    po = conn.execute(
        "SELECT id, vendor_id FROM purchase_orders WHERE po_number = ?", (inv["po_number"],)
    ).fetchone()
    if po is None:
        flags.append("missing_po")
        notes.append(f"PO {inv['po_number']} does not exist.")
    elif vendor_id is not None and po[1] != vendor_id:
        flags.append("missing_po")
        notes.append(f"PO {inv['po_number']} belongs to a different vendor.")
    else:
        po_id = po[0]
        ordered = {
            item: price
            for item, price in conn.execute(
                "SELECT item, unit_price FROM po_lines WHERE po_id = ?", (po_id,)
            )
        }
        received = {
            item: qty
            for item, qty in conn.execute(
                "SELECT rl.item, SUM(rl.quantity_received) FROM receipt_lines rl "
                "JOIN delivery_receipts dr ON dr.id = rl.receipt_id "
                "WHERE dr.po_id = ? GROUP BY rl.item",
                (po_id,),
            )
        }
        for line in inv["lines"]:
            name = line["item"]
            if name not in ordered:
                flags.append("item_not_on_po")
                notes.append(f"'{name}' is not on PO {inv['po_number']}.")
                continue
            if abs(line["unit_price"] - ordered[name]) > PRICE_TOLERANCE:
                flags.append("wrong_amount")
                notes.append(
                    f"'{name}' billed at {money(line['unit_price'])} but PO price is {money(ordered[name])}."
                )
            got = received.get(name, 0)
            if line["quantity"] > got:
                flags.append("over_billed")
                notes.append(f"'{name}': billed {line['quantity']} but only {got} delivered.")

            # Check arithmetic: line items must sum to subtotal, and subtotal + tax must equal total
            warnings = arithmetic_warnings(inv)
            if warnings:
               flags.append("arithmetic_mismatch")
            for w in warnings:
               notes.append(f"Arithmetic issue: {w}")

    if inv["total"] > HIGH_VALUE_LIMIT:
        flags.append("high_value")
        notes.append(f"Total {money(inv['total'])} is above the {money(HIGH_VALUE_LIMIT)} approval limit.")

    flags = list(dict.fromkeys(flags))  # remove repeats, keep order
    if REJECT_FLAGS & set(flags):
        outcome = "reject"
    elif flags:
        outcome = "review"
    else:
        outcome = "approve"

    return {
        "outcome": outcome,
        "flags": flags,
        "explanation": " ".join(notes) if notes else "All checks passed.",
    }