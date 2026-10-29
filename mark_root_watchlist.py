# -*- coding: utf-8 -*-
"""
実行中のルート直下ファイルが参照する my_watchlist.json（ルート直下）に
TOPIX入替の目印を付ける。PDF(topix_j.pdf)から新規追加/移行措置を再抽出。
"""
import os
import re
import json

STOCK_APP = r"c:\Users\FNHR\Documents\document_kiro\stock_app"
ROOT = r"c:\Users\FNHR\Documents\document_kiro"
PDF = os.path.join(STOCK_APP, "topix_j.pdf")

# ---- PDFテキスト抽出 ----
def extract_pdf_text(path):
    for fn in ("_pp", "_fitz", "_pypdf"):
        pass
    # pdfplumber → fitz → pypdf の順
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                out.append(pg.extract_text() or "")
        return "\n".join(out)
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(path)
        return "\n".join(pg.get_text() for pg in doc)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n".join((pg.extract_text() or "") for pg in reader.pages)

text = extract_pdf_text(PDF)

H_ADD = "（１）新規追加銘柄"
H_TRANS = "（２）移行措置銘柄"
H_CONST = "（３）構成銘柄"
i_add = text.find(H_ADD)
i_trans = text.find(H_TRANS)
i_const = text.find(H_CONST)

sec_add = text[i_add:i_trans]
sec_trans = text[i_trans:i_const]

CODE_RE = re.compile(r"^\s*\d+\s+([0-9]{3}[0-9A-Z]|[0-9]{4})\s+(.+?)\s+(プライム|スタンダード|グロース)\s*$")

def extract(section):
    out = {}
    for line in section.splitlines():
        m = CODE_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out

additions = extract(sec_add)
transitions = extract(sec_trans)

MARK_DATE = "2026-10-30"
TAG_ADD = "TOPIX新規追加"
TAG_TRANS = "TOPIX移行措置(除外)"

report = {
    "pdf_add": len(additions),
    "pdf_trans": len(transitions),
    "files": {},
}

def mark_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    changed = []
    for code, entry in d.items():
        if not isinstance(entry, dict):
            continue
        tag = None
        if code in additions:
            tag = TAG_ADD
        elif code in transitions:
            tag = TAG_TRANS
        if tag:
            entry["topix_2026"] = tag
            entry["topix_updated"] = MARK_DATE
            changed.append(f"{code} {entry.get('name','')} → {tag}")
        else:
            entry.pop("topix_2026", None)
            entry.pop("topix_updated", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return changed

for p in [os.path.join(ROOT, "my_watchlist.json"),
          os.path.join(STOCK_APP, "my_watchlist.json")]:
    ch = mark_file(p)
    report["files"][p] = {"marked": len(ch) if ch is not None else "N/A", "changes": ch or []}

with open(os.path.join(STOCK_APP, "_mark_root_out.txt"), "w", encoding="utf-8") as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2))
