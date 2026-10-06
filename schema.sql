CREATE TABLE vendors (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  email TEXT
);

CREATE TABLE purchase_orders (
  id INTEGER PRIMARY KEY,
  po_number TEXT NOT NULL UNIQUE,
  vendor_id INTEGER NOT NULL REFERENCES vendors(id),
  order_date TEXT
);

CREATE TABLE po_lines (
  id INTEGER PRIMARY KEY,
  po_id INTEGER NOT NULL REFERENCES purchase_orders(id),
  item TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  unit_price REAL NOT NULL
);

CREATE TABLE delivery_receipts (
  id INTEGER PRIMARY KEY,
  po_id INTEGER NOT NULL REFERENCES purchase_orders(id),
  received_date TEXT
);

CREATE TABLE receipt_lines (
  id INTEGER PRIMARY KEY,
  receipt_id INTEGER NOT NULL REFERENCES delivery_receipts(id),
  item TEXT NOT NULL,
  quantity_received INTEGER NOT NULL
);

CREATE TABLE invoices (
  id INTEGER PRIMARY KEY,
  invoice_number TEXT NOT NULL,
  vendor_id INTEGER REFERENCES vendors(id),
  po_number TEXT,
  invoice_date TEXT,
  tax REAL,
  total REAL,
  source_file TEXT,
  UNIQUE (vendor_id, invoice_number)
);

CREATE TABLE invoice_lines (
  id INTEGER PRIMARY KEY,
  invoice_id INTEGER NOT NULL REFERENCES invoices(id),
  item TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  unit_price REAL NOT NULL
);

CREATE TABLE decisions (
  id INTEGER PRIMARY KEY,
  invoice_id INTEGER NOT NULL REFERENCES invoices(id),
  outcome TEXT NOT NULL,        -- approve | review | reject
  flags TEXT,                   -- e.g. "wrong_amount,over_billed"
  explanation TEXT,
  decided_by TEXT,              -- agent | human name
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY,
  invoice_id INTEGER,
  step TEXT,                    -- extract | match | decide | approve
  detail TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);