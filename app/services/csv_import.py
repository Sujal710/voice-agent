"""Parse the daily contact CSV (two columns: name, phone number).

Real-world CSVs are messy, so this accepts:
* headers in any case/spacing: "Name", "Customer Name", "Phone", "Phone Number", "Mobile No." ...
* files with no header row (first column = name, second = phone)
* Excel exports with a UTF-8 BOM, `;` or tab separators
* Indian numbers written any common way: 98765 43210, 098765-43210, 919876543210, +91 98765 43210

Every rejected row is reported back with its line number and a reason, so the
person uploading can fix the file.
"""

import csv
import io
import re

MAX_ROWS = 5000
MAX_BYTES = 2 * 1024 * 1024

NAME_HEADERS = {"name", "customername", "fullname", "customer", "contactname", "naam"}
PHONE_HEADERS = {
    "phone", "phonenumber", "phoneno", "mobile", "mobilenumber", "mobileno", "contact",
    "contactnumber", "contactno", "number", "cell", "whatsapp", "phone1",
}  # fmt: skip

E164 = re.compile(r"^\+[1-9]\d{7,14}$")


class CSVImportError(ValueError):
    """The file as a whole can't be used (wrong format, too big, no columns found)."""


def _key(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.lower())


def normalize_phone(raw: str, default_country_code: str = "91") -> str | None:
    """Return an E.164 number (+919876543210) or None if it can't be a valid number."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or re.search(r"[eE]\+?\d", text):  # Excel scientific notation, e.g. 9.19876E+11
        return None
    has_plus = text.startswith("+")
    digits = re.sub(r"\D", "", text)
    if has_plus:
        number = "+" + digits
    elif text.startswith("00"):
        number = "+" + digits[2:]
    elif default_country_code == "91" and len(digits) == 10 and digits[0] in "6789":
        number = "+91" + digits  # Indian mobile
    elif default_country_code == "91" and len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        number = "+91" + digits[1:]  # 0 trunk prefix
    elif default_country_code == "91" and len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        number = "+" + digits  # 91 without the plus
    elif len(digits) == 10:
        number = f"+{default_country_code}{digits}"
    else:
        return None
    if not E164.match(number):
        return None
    if number.startswith("+91") and (len(number) != 13 or number[3] not in "6789"):
        return None
    return number


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CSVImportError("Could not read the file. Please save it as CSV (UTF-8).")


def parse_contacts(data: bytes, default_country_code: str = "91") -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows).

    valid_rows:    [{"line": 2, "name": "Sujal", "phone": "+919876543210"}, ...]  (deduplicated by phone)
    rejected_rows: [{"line": 5, "name": "...", "phone": "...", "reason": "..."}, ...]
    """
    if len(data) > MAX_BYTES:
        raise CSVImportError("File is larger than 2 MB. Please split it into smaller files.")
    text = _decode(data).strip()
    if not text:
        raise CSVImportError("The file is empty.")

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise CSVImportError("The file has no rows.")

    header = [_key(c) for c in rows[0]]
    name_idx = next((i for i, h in enumerate(header) if h in NAME_HEADERS), None)
    phone_idx = next((i for i, h in enumerate(header) if h in PHONE_HEADERS), None)
    start = 1
    if name_idx is None and phone_idx is None:
        # No header row: if the first row already has a phone number, treat it as data
        if len(rows[0]) >= 2 and normalize_phone(rows[0][1], default_country_code):
            name_idx, phone_idx, start = 0, 1, 0
        else:
            raise CSVImportError("Couldn't find the columns. The first row should be a header like: name,phone")
    elif name_idx is None or phone_idx is None:
        missing = "name" if name_idx is None else "phone number"
        raise CSVImportError(f"Couldn't find the {missing} column. The header should look like: name,phone")

    if len(rows) - start > MAX_ROWS:
        raise CSVImportError(f"Too many rows ({len(rows) - start}). The limit is {MAX_ROWS} per upload.")

    valid: list[dict] = []
    rejected: list[dict] = []
    seen: dict[str, int] = {}
    for offset, row in enumerate(rows[start:]):
        line = offset + start + 1  # 1-based line number as shown in Excel
        name = row[name_idx].strip() if name_idx < len(row) else ""
        raw_phone = row[phone_idx].strip() if phone_idx < len(row) else ""
        name = re.sub(r"\s+", " ", name)[:100]
        if not raw_phone:
            rejected.append({"line": line, "name": name, "phone": raw_phone, "reason": "phone number is empty"})
            continue
        phone = normalize_phone(raw_phone, default_country_code)
        if phone is None:
            rejected.append({"line": line, "name": name, "phone": raw_phone, "reason": "not a valid phone number"})
            continue
        if phone in seen:
            rejected.append(
                {"line": line, "name": name, "phone": raw_phone, "reason": f"duplicate of line {seen[phone]}"}
            )
            continue
        seen[phone] = line
        valid.append({"line": line, "name": name or "Customer", "phone": phone})
    return valid, rejected
