# -*- coding: utf-8 -*-
"""
topix_j.pdf の
  (1) 新規追加銘柄
  (2) 移行措置銘柄（経過措置＝実質的な除外対象）
を抽出し、ウォッチリスト（my_watchlist.json / fundamentals_watchlist.json）の
該当銘柄に目印を付ける。
"""
import os
import re
import json

D = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(D, "_topix_text.txt")

with open(TXT, "r", encoding="utf-8") as f:
    text = f.read()

# セクション見出し
H_ADD = "（１）新規追加銘柄"
H_TRANS = "（２）移行措置銘柄"
H_CONST = "（３）構成銘柄"

i_add = text.find(H_ADD)
i_trans = text.find(H_TRANS)
i_const = text.find(H_CONST)
assert i_add != -1 and i_trans != -1 and i_const != -1, (i_add, i_trans, i_const)

sec_add = text[i_add:i_trans]
sec_trans = text[i_trans:i_const]

# 行フォーマット: "No コード 銘柄名 市場区分"
# コードは 4桁数字 または 数字3桁+英字1（例 141A, 581A, 429A）
CODE_RE = re.compile(r"^\s*\d+\s+([0-9]{3}[0-9A-Z]|[0-9]{4})\s+(.+?)\s+(プライム|スタンダード|グロース)\s*$")


def extract(section):
    out = {}
    for line in section.splitlines():
        m = CODE_RE.match(line)
        if m:
            code = m.group(1)
            name = m.group(2).strip()
            out[code] = name
    return out


additions = extract(sec_add)
transitions = extract(sec_trans)

print(f"新規追加: {len(additions)}件")
print(f"移行措置(除外相当): {len(transitions)}件")

# ===== ウォッチリストに目印を付ける =====
MARK_DATE = "2026-10-30"  # 入替実施日
TAG_ADD = "TOPIX新規追加"
TAG_TRANS = "TOPIX移行措置(除外)"

changes = {"my_watchlist": [], "fundamentals_watchlist": []}

# --- my_watchlist.json: {code: {name, added, ...}} ---
MW = os.path.join(D, "my_watchlist.json")
with open(MW, "r", encoding="utf-8") as f:
    mw = json.load(f)

for code, entry in mw.items():
    tag = None
    if code in additions:
        tag = TAG_ADD
    elif code in transitions:
        tag = TAG_TRANS
    if tag:
        entry["topix_2026"] = tag
        entry["topix_updated"] = MARK_DATE
        changes["my_watchlist"].append(f"{code} {entry.get('name','')} → {tag}")
    else:
        # 以前の誤マークが残っていれば消す（冪等性）
        entry.pop("topix_2026", None)
        entry.pop("topix_updated", None)

with open(MW, "w", encoding="utf-8") as f:
    json.dump(mw, f, ensure_ascii=False, indent=2)

# --- fundamentals_watchlist.json: {"code.T": name} → マーク付与のため構造拡張が必要 ---
# この形式は値が文字列なので、別途サイドカーJSONにマークを保存して非破壊にする。
FW = os.path.join(D, "fundamentals_watchlist.json")
with open(FW, "r", encoding="utf-8") as f:
    fw = json.load(f)

fw_marks = {}
for tcode, name in fw.items():
    code = tcode.replace(".T", "")
    if code in additions:
        fw_marks[tcode] = {"name": name, "topix_2026": TAG_ADD, "topix_updated": MARK_DATE}
        changes["fundamentals_watchlist"].append(f"{tcode} {name} → {TAG_ADD}")
    elif code in transitions:
        fw_marks[tcode] = {"name": name, "topix_2026": TAG_TRANS, "topix_updated": MARK_DATE}
        changes["fundamentals_watchlist"].append(f"{tcode} {name} → {TAG_TRANS}")

FW_MARKS = os.path.join(D, "fundamentals_watchlist_topix_marks.json")
with open(FW_MARKS, "w", encoding="utf-8") as f:
    json.dump(fw_marks, f, ensure_ascii=False, indent=2)

# ===== レポート =====
report = {
    "新規追加銘柄数(PDF)": len(additions),
    "移行措置銘柄数(PDF)": len(transitions),
    "my_watchlist_マーク数": len(changes["my_watchlist"]),
    "fundamentals_watchlist_マーク数": len(changes["fundamentals_watchlist"]),
    "my_watchlist_変更": changes["my_watchlist"],
    "fundamentals_watchlist_変更": changes["fundamentals_watchlist"],
}
with open(os.path.join(D, "_mark_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n=== my_watchlist マーク ===")
for c in changes["my_watchlist"]:
    print(" ", c)
print("\n=== fundamentals_watchlist マーク ===")
for c in changes["fundamentals_watchlist"]:
    print(" ", c)
