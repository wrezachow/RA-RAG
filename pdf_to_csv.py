# pip install pdfplumber pandas

import os
import re
import pdfplumber
import pandas as pd

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "")                 # soft hyphen
    s = re.sub(r"-\n(\w)", r"\1", s)            # fix hyphen line-wraps
    s = re.sub(r"\n+", "\n", s)                 # collapse many newlines
    s = re.sub(r"[ \t]+", " ", s)               # collapse spaces/tabs
    return s.strip()

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150):
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_chars:
        raise ValueError("overlap must be < chunk_chars")

    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def default_out_csv(pdf_path: str) -> str:
    # papers/foo.pdf -> papers/foo_chunks.csv
    folder = os.path.dirname(pdf_path) or "."
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(folder, f"{base}_chunks.csv")

def pdf_to_csv(
    pdf_path: str,
    out_csv: str | None = None,
    chunk_chars: int = 1200,
    overlap: int = 150,
    only_if_newer: bool = True,
) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if out_csv is None or out_csv.strip() == "":
        out_csv = default_out_csv(pdf_path)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Skip reconversion if CSV is already current
    if only_if_newer and os.path.exists(out_csv):
        if os.path.getmtime(out_csv) >= os.path.getmtime(pdf_path):
            return out_csv

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""

            txt = clean_text(raw)
            if not txt:
                continue

            for chunk_idx, chunk in enumerate(chunk_text(txt, chunk_chars, overlap), start=1):
                rows.append({
                    "source_file": os.path.basename(pdf_path),
                    "page": page_idx,
                    "chunk_id": f"p{page_idx:03d}_c{chunk_idx:03d}",
                    "text": chunk,
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if len(rows) == 0:
        print("Warning: extracted 0 chunks. PDF may be scanned (image-only).")

    return out_csv

if __name__ == "__main__":
    pdf = input("PDF path (e.g. papers/paper.pdf): ").strip()
    out = input("Output CSV (Enter for default next to PDF): ").strip() or None
    saved = pdf_to_csv(pdf, out)
    print("Saved to:", saved)
