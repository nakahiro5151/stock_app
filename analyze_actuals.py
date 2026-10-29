# -*- coding: utf-8 -*-
"""実績35銘柄が、予想CSV上で実際にどう判定されたかの内訳を集計する。"""
import pandas as pd

CSV = r"c:\Users\FNHR\Documents\document_kiro\stock_app\topix_prediction_20261007_174706.csv"

ACTUAL = {
    "1723": "日本電技", "1965": "テクノ菱和", "2160": "ジーエヌアイグループ",
    "2702": "日本マクドナルドHD", "2782": "セリア", "2805": "ヱスビー食品",
    "4107": "伊勢化学工業", "4478": "フリー", "4588": "オンコリスバイオファーマ",
    "4816": "東映アニメーション", "4966": "上村工業", "5161": "西川ゴム工業",
    "6227": "ＡＩメカテック", "6366": "千代田化工建設", "6613": "ＱＤレーザ",
    "6736": "サン電子", "6777": "santec HD", "6834": "精工技研",
    "6890": "フェローテック", "6960": "フクダ電子", "7287": "日本精機",
    "7564": "ワークマン", "7716": "ナカニシ", "7806": "ＭＴＧ",
    "7906": "ヨネックス", "8117": "中央自動車工業", "8508": "Ｊトラスト",
    "9436": "沖縄セルラー電話", "141A": "トライアルHD", "186A": "アストロスケール",
    "215A": "タイミー", "290A": "Synspective", "485A": "パワーエックス",
    "581A": "ＧＯ", "593A": "ティアフォー",
}

df = pd.read_csv(CSV, dtype={"銘柄コード": str})
df["銘柄コード"] = df["銘柄コード"].astype(str).str.strip()

rows = []
for code, nm in ACTUAL.items():
    r = df[df["銘柄コード"] == code]
    if r.empty:
        rows.append({"コード": code, "銘柄": nm, "判定": "(CSVに無し)",
                     "回転率": None, "累積比率": None, "浮動株時価総額億": None})
        continue
    r = r.iloc[0]
    rows.append({
        "コード": code, "銘柄": nm,
        "判定": r.get("TOPIX判定"),
        "回転率": r.get("年間売買回転率"),
        "累積比率": r.get("累積時価総額比率"),
        "浮動株時価総額億": r.get("浮動株時価総額(億円)"),
    })

_D = r"c:\Users\FNHR\Documents\document_kiro\stock_app"
out = pd.DataFrame(rows)
out.to_csv(_D + r"\actuals_breakdown.csv", index=False, encoding="utf-8-sig")

# 判定カテゴリ別の集計
summary = out["判定"].value_counts(dropna=False)
with open(_D + r"\actuals_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== 実績35銘柄の予想側判定の内訳 ===\n")
    f.write(summary.to_string())
    f.write("\n\n=== 明細 ===\n")
    f.write(out.to_string(index=False))

print("判定内訳:")
print(summary.to_string())
print()
print(out.to_string(index=False))
