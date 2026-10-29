# -*- coding: utf-8 -*-
"""
TOPIX新基準スクリーニング（予想）を stock_theme_dashboard.py のロジックそのままに
ヘッドレスで実行し、JPX公表の実績（topix_j.pdf）と照合して的中率を出す。

- 予想: アプリ tab10 と同一ロジック（_topix_run_screening）で「新規追加候補」を抽出
- 実績: topix_j.pdf の「(1) 新規追加銘柄」35銘柄
"""
import io
import re
import sys
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ===== アプリと同一の定数 =====
_TOPIX_TRADED_VALUE_RATIO_THRESHOLD = 0.14
_TOPIX_CUMULATIVE_MCAP_PCT = 0.97


# ===== アプリ関数の移植（@st.cache_data / st.* を除去、ロジックは不変）=====
def _topix_fetch_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        return pd.DataFrame()
    content = io.BytesIO(resp.content)
    try:
        df = pd.read_excel(content, engine="calamine")
    except Exception:
        content.seek(0)
        try:
            df = pd.read_excel(content, engine="openpyxl")
        except Exception:
            content.seek(0)
            df = pd.read_excel(content)
    col_map = {}
    used = set()
    for c in df.columns:
        s = str(c)
        if "コード" in s and "33業種" not in s and "17業種" not in s and "code" not in used:
            col_map[c] = "code"; used.add("code")
        elif "銘柄名" in s and "name" not in used:
            col_map[c] = "name"; used.add("name")
        elif "市場・商品区分" in s and "market" not in used:
            col_map[c] = "market"; used.add("market")
        elif "33業種" in s and "コード" not in s and "sector" not in used:
            col_map[c] = "sector"; used.add("sector")
    df = df.rename(columns=col_map)
    for n in ["code", "name", "market", "sector"]:
        if n not in df.columns:
            df[n] = ""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df["code"] = df["code"].astype(str).str.strip()
    # 従来の4桁数字コードに加え、新形式コード（英数字4桁、例 141A）も対象に含める
    df = df[df["code"].str.match(r"^[0-9A-Z]{4}$")]
    df["ticker"] = df["code"] + ".T"
    return df[["ticker", "code", "name", "market", "sector"]].reset_index(drop=True)


import os
import pickle

_PRICE_CACHE = "_topix_price_cache.pkl"


def _topix_fetch_market_data(tickers, period="1y"):
    if not tickers:
        return {}
    # ディスクキャッシュ（当日中は再ダウンロードしない）
    cached = {}
    if os.path.exists(_PRICE_CACHE):
        try:
            age_h = (time.time() - os.path.getmtime(_PRICE_CACHE)) / 3600
            if age_h < 24:
                with open(_PRICE_CACHE, "rb") as f:
                    cached = pickle.load(f)
                print(f"  price cache hit: {len(cached)}件 (age {age_h:.1f}h)", flush=True)
        except Exception:
            cached = {}
    need = [t for t in tickers if t not in cached]
    print(f"  price: cached={len(cached)} / need={len(need)}", flush=True)

    results = dict(cached)
    batch_size = 50
    for i in range(0, len(need), batch_size):
        batch = need[i:i + batch_size]
        try:
            data = yf.download(batch, period=period, progress=False, threads=True)
            if data.empty:
                continue
            for t in batch:
                try:
                    if len(batch) == 1:
                        close = data["Close"].dropna()
                        volume = data["Volume"].dropna()
                    else:
                        close = data["Close"][t].dropna()
                        volume = data["Volume"][t].dropna()
                    if len(close) > 20:
                        results[t] = {"close": close, "volume": volume}
                except Exception:
                    pass
        except Exception:
            pass
        # バッチごとに逐次キャッシュ保存（中断してもレジューム可能）
        if (i // batch_size) % 5 == 0:
            print(f"  market_data: {min(i+batch_size, len(need))}/{len(need)}", flush=True)
            try:
                with open(_PRICE_CACHE, "wb") as f:
                    pickle.dump(results, f)
            except Exception:
                pass
    # 最終キャッシュ保存
    try:
        with open(_PRICE_CACHE, "wb") as f:
            pickle.dump(results, f)
    except Exception:
        pass
    return results


_FUND_CACHE = "_topix_fund_cache.json"


def _topix_fetch_fundamentals(tickers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json as _json
    results = {}

    # ウォームアップ: 先に1銘柄 info を叩いて crumb/cookie を確立しておくと
    # 後続の並列 info 取得で 401(Invalid Crumb) が大幅に減る。
    try:
        _warm = yf.Ticker("7203.T")  # トヨタ（確実に存在）
        _ = _warm.info
        print("  fund warmup done", flush=True)
    except Exception as e:
        print(f"  fund warmup failed: {e}", flush=True)

    # レジューム: 既存キャッシュを読み込み、未取得分だけ取りに行く
    if os.path.exists(_FUND_CACHE):
        try:
            age_h = (time.time() - os.path.getmtime(_FUND_CACHE)) / 3600
            if age_h < 24:
                with open(_FUND_CACHE, "r", encoding="utf-8") as f:
                    results = _json.load(f)
                print(f"  fund cache hit: {len(results)}件 (age {age_h:.1f}h)", flush=True)
        except Exception:
            results = {}
    need = [t for t in tickers if t not in results]
    print(f"  fundamentals: cached={len(results)} / need={len(need)}", flush=True)

    def _fetch_one(t):
        # fast_info 優先（crumb不要で速い）。info は floatShares/PER等のため最小限に。
        out = {
            "marketCap": None, "floatShares": None, "sharesOutstanding": None,
            "dividendYield": None, "trailingPE": None, "forwardPE": None,
            "returnOnEquity": None, "priceToBook": None, "currentPrice": None,
        }
        tk = yf.Ticker(t)
        try:
            fi = tk.fast_info
            out["currentPrice"] = getattr(fi, "last_price", None)
            out["marketCap"] = getattr(fi, "market_cap", None)
            out["sharesOutstanding"] = getattr(fi, "shares", None)
        except Exception:
            pass
        # info は floatShares/PER/ROE 等に必須。ウォームアップ済みなら概ね1発で通る。
        # 失敗時のみ1回だけ軽く再試行（長い待ちは入れない）。
        info = None
        for attempt in range(2):
            try:
                info = tk.info
                if info and (info.get("sharesOutstanding") or info.get("floatShares") or info.get("marketCap")):
                    break
            except Exception:
                info = None
            time.sleep(0.3)
        if info:
            out["marketCap"] = info.get("marketCap") or out["marketCap"]
            out["floatShares"] = info.get("floatShares")
            out["sharesOutstanding"] = info.get("sharesOutstanding") or out["sharesOutstanding"]
            out["trailingPE"] = info.get("trailingPE")
            out["forwardPE"] = info.get("forwardPE")
            out["returnOnEquity"] = info.get("returnOnEquity")
            out["priceToBook"] = info.get("priceToBook")
            out["currentPrice"] = info.get("currentPrice", info.get("regularMarketPrice")) or out["currentPrice"]
            out["dividendYield"] = info.get("trailingAnnualDividendYield") or info.get("dividendYield")
        return t, out

    done = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in need}
        for fut in as_completed(futures):
            try:
                t, data = fut.result(timeout=30)
                results[t] = data
            except Exception:
                results[futures[fut]] = {}
            done += 1
            if done % 100 == 0:
                # 逐次保存（中断してもレジューム可能）
                try:
                    with open(_FUND_CACHE, "w", encoding="utf-8") as f:
                        _json.dump(results, f, ensure_ascii=False)
                except Exception:
                    pass
                print(f"  fundamentals: {done}/{len(need)}", flush=True)
    try:
        with open(_FUND_CACHE, "w", encoding="utf-8") as f:
            _json.dump(results, f, ensure_ascii=False)
    except Exception:
        pass
    return results


def _topix_calc_turnover(close_series, volume_series, float_shares=None, shares_outstanding=None):
    if close_series is None or len(close_series) < 60:
        return None
    shares = float_shares or shares_outstanding
    if not shares or shares <= 0:
        return None
    df = pd.DataFrame({"close": close_series, "volume": volume_series})
    df["traded_value"] = df["close"] * df["volume"]
    df["month"] = df.index.to_period("M")
    monthly_ratios = []
    for _, grp in df.groupby("month"):
        if len(grp) < 5:
            continue
        median_daily_tv = grp["traded_value"].median()
        biz_days = len(grp)
        month_end_price = grp["close"].iloc[-1]
        ff_mcap = month_end_price * shares
        if ff_mcap > 0:
            monthly_ratios.append((median_daily_tv * biz_days) / ff_mcap)
    if len(monthly_ratios) < 6:
        return None
    return sum(monthly_ratios[-12:])


def _topix_run_screening(stock_df, market_data, fundamentals, current_topix_codes=None):
    if current_topix_codes is None:
        current_topix_codes = set()
    rows = []
    for _, row in stock_df.iterrows():
        t = row["ticker"]
        md = market_data.get(t)
        fund = fundamentals.get(t, {})
        if not md or not fund:
            continue
        close = md.get("close")
        volume = md.get("volume")
        if close is None or len(close) < 20:
            continue
        price = fund.get("currentPrice") or (close.iloc[-1] if len(close) > 0 else None)
        if not price or price <= 0:
            continue
        mcap = fund.get("marketCap")
        float_shares = fund.get("floatShares")
        shares_out = fund.get("sharesOutstanding")
        ff_mcap = (float_shares * price) if (float_shares and price) else (mcap * 0.7 if mcap else None)
        turnover = _topix_calc_turnover(close, volume, float_shares, shares_out)
        div_yield = fund.get("dividendYield")
        if div_yield is not None:
            div_yield = round(div_yield * 100, 2)
        pe = fund.get("trailingPE") or fund.get("forwardPE")
        pbr = fund.get("priceToBook")
        roe = fund.get("returnOnEquity")
        if roe is not None:
            roe = round(roe * 100, 2)
        rows.append({
            "銘柄コード": row["code"], "銘柄名": row["name"],
            "市場区分": row["market"], "業種": row["sector"],
            "株価": round(price, 1) if price else None,
            "時価総額(億円)": round(mcap / 1e8, 1) if mcap else None,
            "浮動株時価総額(億円)": round(ff_mcap / 1e8, 1) if ff_mcap else None,
            "年間売買回転率": round(turnover, 4) if turnover else None,
            "配当利回り(%)": div_yield,
            "PER(倍)": round(pe, 2) if pe else None,
            "PBR(倍)": round(pbr, 2) if pbr else None,
            "ROE(%)": roe, "ticker": t,
            "_is_current": row["code"] in current_topix_codes,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    valid = result.dropna(subset=["浮動株時価総額(億円)", "年間売買回転率"])
    valid = valid[valid["年間売買回転率"] >= _TOPIX_TRADED_VALUE_RATIO_THRESHOLD]
    valid = valid.sort_values("浮動株時価総額(億円)", ascending=False)
    total_ff = valid["浮動株時価総額(億円)"].sum()
    if total_ff > 0:
        valid["累積時価総額比率"] = valid["浮動株時価総額(億円)"].cumsum() / total_ff
    else:
        valid["累積時価総額比率"] = None
    result = result.merge(valid[["ticker", "累積時価総額比率"]], on="ticker", how="left")

    def _judge(r):
        tr = r.get("年間売買回転率")
        cum = r.get("累積時価総額比率")
        is_current = r.get("_is_current", False)
        if tr is None or pd.isna(tr):
            return "データ不足"
        if tr < _TOPIX_TRADED_VALUE_RATIO_THRESHOLD:
            return "除外警戒"
        if cum is not None and not pd.isna(cum) and cum <= _TOPIX_CUMULATIVE_MCAP_PCT:
            return "継続安全" if is_current else "新規追加候補"
        elif cum is not None and not pd.isna(cum) and cum <= 0.99:
            return "ボーダーライン"
        return "除外警戒"

    result["TOPIX判定"] = result.apply(_judge, axis=1)
    return result.sort_values("浮動株時価総額(億円)", ascending=False).reset_index(drop=True)


def _topix_fetch_current_constituents():
    codes = set()
    try:
        url = "https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv"
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and len(resp.content) > 1000:
            text = resp.content.decode("shift_jis", errors="ignore")
            for line in text.strip().split("\n")[1:]:
                parts = line.split(",")
                if len(parts) >= 3:
                    code_str = parts[2].strip()
                    m = re.match(r"^(\d{4})$", code_str)
                    if m:
                        codes.add(m.group(1))
    except Exception:
        pass
    return codes


# ===== 実績（PDF）=====
# topix_j.pdf (1)新規追加銘柄 35社。PDF抽出テキスト(_topix_text.txt)より確定。
ACTUAL_ADDITIONS = {
    "1723", "1965", "2160", "2702", "2782", "2805", "4107", "4478", "4588",
    "4816", "4966", "5161", "6227", "6366", "6613", "6736", "6777", "6834",
    "6890", "6960", "7287", "7564", "7716", "7806", "7906", "8117", "8508",
    "9436", "141A", "186A", "215A", "290A", "485A", "581A", "593A",
}


def main():
    # スクリプトのあるディレクトリを作業ディレクトリにして、キャッシュ/出力の場所を固定
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=== TOPIX新基準スクリーニング（予想生成）===", flush=True)
    print(f"cwd={os.getcwd()}", flush=True)
    t0 = time.time()
    stock_df = _topix_fetch_stock_list()
    if stock_df.empty:
        print("ERROR: 銘柄リスト取得失敗", flush=True)
        sys.exit(1)
    print(f"銘柄リスト: {len(stock_df)}件", flush=True)

    current_codes = _topix_fetch_current_constituents()
    print(f"現行TOPIX構成銘柄: {len(current_codes)}件", flush=True)

    # アプリ既定: 全市場・対象数1000（index=3 の 1000）。ただし精度優先で全件対象にできるよう引数化。
    max_stocks = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    # 現行構成を除外（アプリ既定 exclude_current=True 相当）して新規採用候補を見る
    if len(current_codes) > 100:
        before = len(stock_df)
        screen_df = stock_df[~stock_df["code"].isin(current_codes)].copy()
        print(f"現行構成 {before - len(screen_df)}件を除外 → 残り {len(screen_df)}件", flush=True)
    else:
        screen_df = stock_df.copy()
        print("WARNING: 現行構成取得不足。除外せず全件対象", flush=True)

    screen_df = screen_df.head(max_stocks)
    print(f"スクリーニング対象: {len(screen_df)}件（max={max_stocks}）", flush=True)

    tickers = screen_df["ticker"].tolist()
    print("価格データ取得中...", flush=True)
    md = _topix_fetch_market_data(tickers, period="1y")
    print(f"価格データ取得: {len(md)}件", flush=True)
    print("ファンダメンタルズ取得中...", flush=True)
    fund = _topix_fetch_fundamentals(tickers)
    print(f"ファンダメンタルズ取得: {len(fund)}件", flush=True)

    print("スクリーニング計算中...", flush=True)
    result = _topix_run_screening(screen_df, md, fund, current_topix_codes=current_codes)
    if result.empty:
        print("ERROR: スクリーニング結果が空", flush=True)
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"topix_prediction_{stamp}.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"予想CSV保存: {out_csv}", flush=True)

    # 予想: 新規追加候補
    pred_add = result[result["TOPIX判定"] == "新規追加候補"]
    pred_codes = set(pred_add["銘柄コード"].astype(str))
    print(f"\n予想「新規追加候補」: {len(pred_codes)}件", flush=True)

    # ===== 照合 =====
    hit = ACTUAL_ADDITIONS & pred_codes          # 当たり（予想した & 実際に追加された）
    miss = ACTUAL_ADDITIONS - pred_codes          # 見逃し（実際に追加されたが予想しなかった）
    false_pos = pred_codes - ACTUAL_ADDITIONS     # 空振り（予想したが追加されなかった）

    # 実際追加銘柄が「そもそもスクリーニング対象に乗っていたか」も確認
    screened_codes = set(result["銘柄コード"].astype(str))
    actual_in_scope = ACTUAL_ADDITIONS & screened_codes
    actual_out_scope = ACTUAL_ADDITIONS - screened_codes

    name_map = dict(zip(stock_df["code"].astype(str), stock_df["name"]))

    def fmt(codes):
        return [f"{c} {name_map.get(c, '?')}" for c in sorted(codes)]

    report = {
        "実行時刻": stamp,
        "スクリーニング対象数": len(result),
        "実績_新規追加銘柄数": len(ACTUAL_ADDITIONS),
        "予想_新規追加候補数": len(pred_codes),
        "的中(hit)": sorted(hit),
        "的中数": len(hit),
        "見逃し(miss)": fmt(miss),
        "見逃し数": len(miss),
        "空振り(false_positive)数": len(false_pos),
        "実績銘柄のうちスクリーニング対象に含まれた数": len(actual_in_scope),
        "実績銘柄のうち対象外(データ取得不可等)": fmt(actual_out_scope),
        "recall(実績に対する的中率%)": round(len(hit) / len(ACTUAL_ADDITIONS) * 100, 1),
        "precision(予想の正解率%)": round(len(hit) / len(pred_codes) * 100, 1) if pred_codes else None,
        "recall_対象内(対象に乗った実績に対する的中率%)": round(len(hit) / len(actual_in_scope) * 100, 1) if actual_in_scope else None,
    }

    with open(f"topix_backtest_report_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 照合レポート =====", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\n的中銘柄: {fmt(hit)}", flush=True)
    print(f"\n経過時間: {time.time()-t0:.0f}秒", flush=True)


if __name__ == "__main__":
    main()
