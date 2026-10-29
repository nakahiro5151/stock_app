"""
EARNINGS INTELLIGENCE - 決算分析ダッシュボード
- 総合10段階スコア（決算モメンタム / EPS予想修正 / 営業利益率 / ROE・自己資本 / バリュエーション）
- 決算・業績ページ（Q実績 vs 会社計画、成長セグメント、業績推移）
- 株価・バリュエーション・財務ページ（理論株価レンジ、PEGレシオ含む）
- 複数銘柄の一括比較
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import requests as _requests

# .envからAPIキーを読み込み
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

_EDINETDB_API_KEY = os.environ.get("EDINETDB_API_KEY", "")

st.set_page_config(
    page_title="EARNINGS INTELLIGENCE",
    page_icon="📊",
    layout="wide",
)

# =====================================================================
# スタイル
# =====================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 100%; }
    
    /* ヘッダーバー */
    .header-bar {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d1b4e 50%, #4a1942 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .header-bar h1 { margin: 0; font-size: 1.8rem; }
    .header-bar .subtitle { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
    
    /* スコアカード */
    .score-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .score-card .value { font-size: 2rem; font-weight: bold; color: #1a1a3e; }
    .score-card .label { font-size: 0.8rem; color: #666; margin-top: 4px; }
    .score-card .change-positive { color: #26a69a; font-weight: bold; }
    .score-card .change-negative { color: #ef5350; font-weight: bold; }
    
    /* メトリックカード */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem !important;
        color: #666 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #1a1a3e !important;
    }
    
    /* 読みどころカード */
    .insight-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid;
    }
    .insight-positive { background: #f0fdf4; border-color: #22c55e; }
    .insight-watch { background: #fffbeb; border-color: #f59e0b; }
    .insight-outlook { background: #f0f9ff; border-color: #3b82f6; }
    .insight-title { font-weight: bold; font-size: 0.85rem; margin-bottom: 4px; }
    .insight-text { font-size: 0.82rem; color: #444; }
    
    /* スコアバー */
    .score-bar-container {
        background: #f1f3f5;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e9ecef;
    }
    .score-bar-title { font-weight: bold; font-size: 0.9rem; }
    .score-bar-value { float: right; font-weight: bold; color: #1a1a3e; }
    .score-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 8px;
        background: #e9ecef;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .score-bar-detail { font-size: 0.75rem; color: #888; margin-top: 6px; }
    
    /* タブ */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# ウォッチリスト
# =====================================================================
WATCHLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fundamentals_watchlist.json")

DEFAULT_WATCHLIST = {
    "8136.T": "サンリオ",
    "7203.T": "トヨタ自動車",
    "8306.T": "三菱UFJFG",
    "6758.T": "ソニーG",
    "6861.T": "キーエンス",
    "9984.T": "ソフトバンクG",
    "8035.T": "東京エレクトロン",
    "6501.T": "日立製作所",
    "4063.T": "信越化学工業",
    "7011.T": "三菱重工業",
    "9432.T": "NTT",
    "6902.T": "デンソー",
    "8316.T": "三井住友FG",
    "4568.T": "第一三共",
    "6367.T": "ダイキン工業",
    "7741.T": "HOYA",
}


def load_watchlist():
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_WATCHLIST


def save_watchlist(wl):
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)


# =====================================================================
# データ取得
# =====================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(ticker_str):
    """yfinanceから個別銘柄のデータを一括取得"""
    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info or {}
        
        # 年次・四半期財務データ
        income_annual = None
        income_quarterly = None
        balance = None
        cashflow = None
        try:
            income_annual = tk.income_stmt
            income_quarterly = tk.quarterly_income_stmt
            balance = tk.balance_sheet
            cashflow = tk.cashflow
        except Exception:
            pass

        return {
            "info": info,
            "income_annual": income_annual,
            "income_quarterly": income_quarterly,
            "balance": balance,
            "cashflow": cashflow,
            "ticker": ticker_str,
        }
    except Exception:
        return None


def fetch_all_stocks(watchlist_dict):
    """複数銘柄を並列取得"""
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_stock_data, t): t for t in watchlist_dict.keys()}
        for f in as_completed(futures):
            ticker = futures[f]
            try:
                data = f.result()
                if data and data.get("info"):
                    results[ticker] = data
            except Exception:
                pass
    return results


# =====================================================================
# EDINET DB データ取得（スコア・バリュエーション用）+ ディスクキャッシュ
# =====================================================================
import hashlib as _hashlib
import tempfile as _tempfile

_EDB_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".edinetdb_cache")
os.makedirs(_EDB_DISK_CACHE_DIR, exist_ok=True)


def _edb_cache_path(ticker_code):
    """銘柄ごとのキャッシュファイルパス"""
    return os.path.join(_EDB_DISK_CACHE_DIR, f"{ticker_code.replace('.T','')}.json")


def _edb_load_cache(ticker_code):
    """ディスクキャッシュからロード（7日以内なら有効）"""
    path = _edb_cache_path(ticker_code)
    if os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
            age_days = (time.time() - mtime) / 86400
            if age_days < 7:  # 7日間有効
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
    return None


def _edb_save_cache(ticker_code, data):
    """ディスクキャッシュに保存"""
    path = _edb_cache_path(ticker_code)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_edinetdb_full(ticker_code):
    """EDINET DB APIからfinancials/earnings/ratiosを取得（ディスクキャッシュ付き）"""
    # 1. ディスクキャッシュ確認
    cached = _edb_load_cache(ticker_code)
    if cached:
        return cached

    # 2. API取得
    if not _EDINETDB_API_KEY:
        return cached  # キャッシュもなければNone

    code = ticker_code.replace(".T", "")
    headers = {"X-API-Key": _EDINETDB_API_KEY, "User-Agent": "Mozilla/5.0"}
    try:
        sr = _requests.get(f"https://edinetdb.jp/v1/search?q={code}", timeout=15, headers=headers)
        if sr.status_code == 429:
            # レート制限 → ディスクキャッシュ（期限切れでも）を返す
            return _edb_load_expired_cache(ticker_code)
        if sr.status_code != 200:
            return cached
        search_data = sr.json()
        data_list = search_data.get("data") if isinstance(search_data, dict) else search_data
        if not data_list or len(data_list) == 0:
            return cached
        edinet_code = data_list[0].get("edinet_code") or data_list[0].get("edinetCode")
        if not edinet_code:
            return cached

        result = {"edinet_code": edinet_code}
        # earnings（決算短信）
        er = _requests.get(f"https://edinetdb.jp/v1/companies/{edinet_code}/earnings", timeout=15, headers=headers)
        if er.status_code == 200:
            result["earnings"] = er.json()
        elif er.status_code == 429:
            return _edb_load_expired_cache(ticker_code)
        # ratios
        rr = _requests.get(f"https://edinetdb.jp/v1/companies/{edinet_code}/ratios", timeout=15, headers=headers)
        if rr.status_code == 200:
            result["ratios"] = rr.json()
        elif rr.status_code == 429:
            return _edb_load_expired_cache(ticker_code)
        # financials（年次時系列: FY2012~）
        fr = _requests.get(f"https://edinetdb.jp/v1/companies/{edinet_code}/financials", timeout=15, headers=headers)
        if fr.status_code == 200:
            result["financials"] = fr.json()
        elif fr.status_code == 429:
            return _edb_load_expired_cache(ticker_code)

        # 3. キャッシュに保存
        _edb_save_cache(ticker_code, result)
        return result
    except Exception:
        return cached


def _edb_load_expired_cache(ticker_code):
    """期限切れキャッシュも含めてロード（API制限時のフォールバック）"""
    path = _edb_cache_path(ticker_code)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_edinetdb_screener(conditions, sort_by=None, sort_order="desc", limit=200):
    """EDINET DB /v1/screener で全上場企業を横断スクリーニングする（1リクエスト＝1ユニット）。

    conditions: [(metric, operator, value), ...]
      metric 例: "pbr","per","roa","operating-margin","equity-ratio",
                 "net-cash-ratio","psr","net-margin","market-cap","dividend-yield"
      operator: "lte" / "gte" / "eq"
      value: 数値（market-cap は百万円単位、利益率・利回りは%、比率は倍）
    Returns: dict {"companies": [...], "total": int, "error": str|None, "as_of": str|None}
    """
    if not _EDINETDB_API_KEY:
        return {"companies": [], "total": 0, "error": "APIキー未設定（.envのEDINETDB_API_KEY）", "as_of": None}
    headers = {"X-API-Key": _EDINETDB_API_KEY, "User-Agent": "Mozilla/5.0"}
    params = []
    for metric, op, val in conditions:
        # shorthand: metricの'-'を'_'に、例 operating-margin -> operating_margin_gte
        key = metric.replace("-", "_") + "_" + op
        params.append(f"{key}={val}")
    if sort_by:
        params.append(f"sort_by={sort_by}")
        params.append(f"sort_order={sort_order}")
    params.append(f"limit={int(limit)}")
    url = "https://edinetdb.jp/v1/screener?" + "&".join(params)
    try:
        r = _requests.get(url, timeout=30, headers=headers)
        if r.status_code == 429:
            return {"companies": [], "total": 0,
                    "error": "APIレート上限に到達（日本時間0時にリセット）", "as_of": None}
        if r.status_code != 200:
            return {"companies": [], "total": 0,
                    "error": f"APIエラー（HTTP {r.status_code}）", "as_of": None}
        j = r.json()
        data = j.get("data", {})
        return {
            "companies": data.get("companies", []),
            "total": data.get("total", 0),
            "conditions": data.get("conditions", []),
            "error": None,
            "as_of": (j.get("meta", {}) or {}).get("data_as_of"),
        }
    except Exception as e:
        return {"companies": [], "total": 0, "error": f"取得エラー: {e}", "as_of": None}


# =====================================================================
# スコア計算ロジック（元サイト共通ルール準拠 + EDINET DB データ使用）
# =====================================================================

def _common_growth_score(pct):
    """修正率・前年比・計画差の共通採点（10段階）
    +15%以上=10, +10%=9, +7%=8, +4%=7, +1%=6, -1%=5, -4%=4, -7%=3, -10%=2, <-10%=1
    """
    if pct is None:
        return 5.0
    if pct >= 15:
        return 10.0
    elif pct >= 10:
        return 9.0
    elif pct >= 7:
        return 8.0
    elif pct >= 4:
        return 7.0
    elif pct >= 1:
        return 6.0
    elif pct >= -1:
        return 5.0
    elif pct >= -4:
        return 4.0
    elif pct >= -7:
        return 3.0
    elif pct >= -10:
        return 2.0
    else:
        return 1.0


def _margin_diff_score(diff_pt):
    """営業利益率前年差の採点（pt単位）"""
    if diff_pt is None:
        return 5.0
    if diff_pt >= 3.0:
        return 10.0
    elif diff_pt >= 2.0:
        return 9.0
    elif diff_pt >= 1.5:
        return 8.0
    elif diff_pt >= 1.0:
        return 7.0
    elif diff_pt >= 0.3:
        return 6.0
    elif diff_pt >= -0.3:
        return 5.0
    elif diff_pt >= -1.0:
        return 4.0
    elif diff_pt >= -1.5:
        return 3.0
    elif diff_pt >= -2.0:
        return 2.0
    else:
        return 1.0


def _roe_score(roe_pct):
    """ROEスコア"""
    if roe_pct is None:
        return 5.0
    if roe_pct >= 15:
        return 10.0
    elif roe_pct >= 12:
        return 9.0
    elif roe_pct >= 10:
        return 8.0
    elif roe_pct >= 8:
        return 7.0
    elif roe_pct >= 6:
        return 6.0
    elif roe_pct >= 4:
        return 5.0
    elif roe_pct >= 2:
        return 4.0
    elif roe_pct >= 0:
        return 3.0
    elif roe_pct >= -5:
        return 2.0
    else:
        return 1.0


def _equity_ratio_score(eq_pct):
    """自己資本比率スコア"""
    if eq_pct is None:
        return 5.0
    if eq_pct >= 70:
        return 10.0
    elif eq_pct >= 60:
        return 9.0
    elif eq_pct >= 50:
        return 8.0
    elif eq_pct >= 40:
        return 7.0
    elif eq_pct >= 30:
        return 6.0
    elif eq_pct >= 20:
        return 5.0
    elif eq_pct >= 10:
        return 4.0
    elif eq_pct >= 0:
        return 3.0
    else:
        return 1.0


def _per_score(per, base_per=30):
    """PERスコア: max(1, min(10, 5.5 + (1 - 現在PER÷基準PER) × 15))"""
    if per is None or per <= 0:
        return 5.0
    score = 5.5 + (1 - per / base_per) * 15
    return max(1.0, min(10.0, round(score, 1)))


def calc_total_score(data, edb_data=None):
    """総合スコア計算（EDINET DBデータ優先）
    決算モメンタム25% + EPS予想修正25% + 営業利益率20% + ROE・自己資本15% + バリュエーション15%
    """
    info = data.get("info", {})

    # EDINET DB earningsから最新四半期データを取得
    latest_q = None
    prev_annual = None
    latest_annual = None
    if edb_data and edb_data.get("earnings"):
        earnings_data = edb_data["earnings"]
        # 構造: {"data": {"count": N, "earnings": [...], "edinet_code": "..."}}
        if isinstance(earnings_data, dict):
            inner = earnings_data.get("data", {})
            if isinstance(inner, dict):
                earnings_list = inner.get("earnings", [])
            elif isinstance(inner, list):
                earnings_list = inner
            else:
                earnings_list = []
        else:
            earnings_list = []

        if earnings_list:
            # 最新の四半期データ（Q1-Q3）
            for e in earnings_list:
                q = e.get("quarter")
                if q and q < 4:
                    latest_q = e
                    break
            # 最新の通期データ（Q4）
            for e in earnings_list:
                if e.get("quarter") == 4:
                    latest_annual = e
                    break

    # EDINET DB ratiosから最新年次データ
    latest_ratio = None
    prev_ratio = None
    if edb_data and edb_data.get("ratios"):
        ratios_data = edb_data["ratios"]
        if isinstance(ratios_data, dict):
            ratios_list = ratios_data.get("data", [])
        elif isinstance(ratios_data, list):
            ratios_list = ratios_data
        else:
            ratios_list = []
        if ratios_list and len(ratios_list) >= 1:
            sorted_ratios = sorted(ratios_list, key=lambda x: x.get("fiscal_year", 0), reverse=True)
            latest_ratio = sorted_ratios[0]
            prev_ratio = sorted_ratios[1] if len(sorted_ratios) >= 2 else None

    # --- 決算モメンタム ---
    # Q1前年比: 売上10% + 営業利益50% + 純利益20% + EPS20%
    if latest_q:
        rev_chg = latest_q.get("revenue_change")  # already %
        oi_chg = latest_q.get("operating_income_change")
        ni_chg = latest_q.get("net_income_change")
        # EPS変化率を推定（net_income_changeで代用）
        eps_chg = ni_chg
        momentum = (
            _common_growth_score(rev_chg) * 0.10 +
            _common_growth_score(oi_chg) * 0.50 +
            _common_growth_score(ni_chg) * 0.20 +
            _common_growth_score(eps_chg) * 0.20
        )
    else:
        # yfinanceフォールバック
        rev_g = (info.get("revenueGrowth") or 0) * 100
        eq_g = (info.get("earningsQuarterlyGrowth") or 0) * 100
        eg = (info.get("earningsGrowth") or 0) * 100
        momentum = (
            _common_growth_score(rev_g) * 0.10 +
            _common_growth_score(eq_g) * 0.50 +
            _common_growth_score(eg) * 0.20 +
            _common_growth_score(eg) * 0.20
        )
    momentum = round(momentum, 1)

    # --- EPS予想修正 ---
    # 直前会社予想からの修正率。据え置き0% = 5点。
    # Q1発表時のforecast_eps vs Q4(通期決算)発表時のforecast_eps（=翌期予想）
    eps_rev_score = 5.0
    if latest_q and latest_annual:
        curr_forecast = latest_q.get("forecast_eps")
        prev_forecast = latest_annual.get("forecast_eps")
        if curr_forecast and prev_forecast and prev_forecast > 0:
            revision_pct = (curr_forecast - prev_forecast) / prev_forecast * 100
            eps_rev_score = _common_growth_score(revision_pct)
        elif curr_forecast and prev_forecast and curr_forecast == prev_forecast:
            eps_rev_score = 5.0  # 据え置き
    elif latest_q:
        # 前回予想がない場合、据え置き扱い
        eps_rev_score = 5.0
    else:
        trailing_eps = info.get("trailingEps")
        forward_eps_val = info.get("forwardEps")
        if trailing_eps and forward_eps_val and trailing_eps > 0:
            revision_pct = (forward_eps_val - trailing_eps) / trailing_eps * 100
            eps_rev_score = _common_growth_score(revision_pct)

    # --- 営業利益率 ---
    # Q1前年差ptベース
    margin_score = 5.0
    if latest_q and latest_q.get("revenue") and latest_q.get("operating_income"):
        q_margin = latest_q["operating_income"] / latest_q["revenue"] * 100
        # 前年Q1の利益率を推定: OI変化率とRevenue変化率から
        oi_chg_val = latest_q.get("operating_income_change")
        rev_chg_val = latest_q.get("revenue_change")
        if oi_chg_val is not None and rev_chg_val is not None:
            # 前年OI = 今期OI / (1 + change/100)
            prev_oi = latest_q["operating_income"] / (1 + oi_chg_val / 100) if oi_chg_val != -100 else 0
            prev_rev = latest_q["revenue"] / (1 + rev_chg_val / 100) if rev_chg_val != -100 else 0
            prev_margin = (prev_oi / prev_rev * 100) if prev_rev > 0 else 0
            margin_diff = q_margin - prev_margin
            margin_score = _margin_diff_score(margin_diff)
    elif latest_ratio and prev_ratio:
        # 年次の利益率差で代用
        curr_margin = (latest_ratio.get("net_margin") or 0) * 100
        prev_margin_val = (prev_ratio.get("net_margin") or 0) * 100
        margin_score = _margin_diff_score(curr_margin - prev_margin_val)

    # --- ROE・自己資本 ---
    # ROE点60% + 自己資本比率点40%
    roe_pct = None
    eq_ratio_pct = None
    if latest_ratio:
        roe_pct = (latest_ratio.get("roe_official") or latest_ratio.get("roe") or 0) * 100
    elif info.get("returnOnEquity"):
        roe_pct = info["returnOnEquity"] * 100

    if latest_q and latest_q.get("equity_ratio"):
        eq_ratio_pct = latest_q["equity_ratio"]  # already %
    elif latest_ratio and latest_ratio.get("equity_ratio_official"):
        eq_ratio_pct = latest_ratio["equity_ratio_official"] * 100

    roe_eq_score = round(_roe_score(roe_pct) * 0.60 + _equity_ratio_score(eq_ratio_pct) * 0.40, 1)

    # --- バリュエーション ---
    # PER点70% + PBR点30%（PBRは同業比較未取得のため中立5点）
    per_val_for_score = None
    if latest_q and latest_q.get("forecast_eps"):
        # 会社予想PER = 現在株価 / forecast_eps
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price and latest_q["forecast_eps"] > 0:
            per_val_for_score = price / latest_q["forecast_eps"]
    if per_val_for_score is None:
        per_val_for_score = info.get("forwardPE") or info.get("trailingPE")

    valuation_score = round(_per_score(per_val_for_score, base_per=30) * 0.70 + 5.0 * 0.30, 1)

    # --- 総合 ---
    total = (momentum * 0.25 + eps_rev_score * 0.25 + margin_score * 0.20 +
             roe_eq_score * 0.15 + valuation_score * 0.15)

    return {
        "total": round(total, 1),
        "momentum": momentum,
        "eps_revision": eps_rev_score,
        "margin": margin_score,
        "roe_equity": roe_eq_score,
        "valuation": valuation_score,
    }


def score_label(score):
    """スコアのラベル"""
    if score >= 8:
        return "強気"
    elif score >= 6:
        return "やや強気"
    elif score >= 5:
        return "中立"
    elif score >= 3:
        return "やや弱気"
    else:
        return "弱気"


def calc_peg_ratio(info):
    """PEGレシオ計算（複数ソースのフォールバック付き）。

    Returns: (peg: float or None, source: str)
      source は算出根拠（"yfinance", "EPS成長率", "四半期EPS成長率",
      "予想EPS成長率", "売上成長率(暫定)" など）。
    PEG=PER÷EPS成長率(%)。成長率が負（減益）の場合は本来定義できないが、
    暫定として売上成長率で代用する。
    """
    # PER（trailing優先、無ければforward、無ければ株価÷EPSで自前計算）
    per = info.get("trailingPE") or info.get("forwardPE")
    if not per or per <= 0:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        eps = info.get("trailingEps") or info.get("forwardEps")
        if price and eps and eps > 0:
            per = price / eps

    # 1. yfinance提供のPEG（pegRatio / trailingPegRatio）
    for key in ("pegRatio", "trailingPegRatio"):
        v = info.get(key)
        if v is not None and -1000 < v < 1000:
            return (round(float(v), 2), "yfinance")

    if not per or per <= 0:
        return (None, "PER取得不可")

    # 2. 年次EPS成長率
    eg = info.get("earningsGrowth")
    if eg is not None and eg > 0:
        return (round(per / (eg * 100), 2), "EPS成長率")

    # 3. 四半期EPS成長率
    eqg = info.get("earningsQuarterlyGrowth")
    if eqg is not None and eqg > 0:
        return (round(per / (eqg * 100), 2), "四半期EPS成長率")

    # 4. 予想EPS成長率（trailing→forward）
    teps = info.get("trailingEps")
    feps = info.get("forwardEps")
    if teps and feps and teps > 0:
        g = (feps - teps) / teps
        if g > 0:
            return (round(per / (g * 100), 2), "予想EPS成長率")

    # 5. 売上成長率で代用（暫定。減益・成長率欠損時の最終手段）
    rg = info.get("revenueGrowth")
    if rg is not None and rg > 0:
        return (round(per / (rg * 100), 2), "売上成長率(暫定)")

    # 成長率がすべて負 or 欠損 → 減益のため定義不能
    return (None, "減益のため算出不可")


def calc_theoretical_price(info):
    """理論株価レンジ計算（EPS × PERレンジ）"""
    eps = info.get("forwardEps") or info.get("trailingEps")
    if not eps or eps <= 0:
        return None
    
    # PER 24倍（悲観）/ 30倍（基準）/ 36倍（楽観）
    return {
        "pessimistic": round(eps * 24, 0),
        "base": round(eps * 30, 0),
        "optimistic": round(eps * 36, 0),
        "eps": eps,
    }


def generate_insights(data):
    """今回の読みどころ生成"""
    info = data.get("info", {})
    insights = {"positive": [], "watch": [], "outlook": []}
    
    # POSITIVE
    rev_growth = info.get("revenueGrowth")
    op_margin = info.get("operatingMargins")
    eq_growth = info.get("earningsQuarterlyGrowth")
    
    if rev_growth and rev_growth > 0.1:
        insights["positive"].append(f"売上高は前年比+{rev_growth*100:.1f}%と好調。")
    if eq_growth and eq_growth > 0.1:
        insights["positive"].append(f"四半期利益は前年比+{eq_growth*100:.1f}%の成長。")
    if op_margin and op_margin > 0.2:
        insights["positive"].append(f"営業利益率{op_margin*100:.1f}%と高水準を維持。")
    
    # WATCH
    per = info.get("trailingPE") or info.get("forwardPE")
    if per and per > 30:
        insights["watch"].append(f"PER {per:.1f}倍とバリュエーションが高め。利益成長が鈍化した場合の株価調整リスクに注意。")
    if op_margin and rev_growth:
        if op_margin < 0.1 and rev_growth > 0.05:
            insights["watch"].append("売上増加に対して利益率が低め。コスト管理の動向を注視。")
    
    # OUTLOOK
    forward_eps = info.get("forwardEps")
    trailing_eps = info.get("trailingEps")
    market_cap = info.get("marketCap")
    if forward_eps and trailing_eps and trailing_eps > 0:
        eps_growth = (forward_eps - trailing_eps) / trailing_eps * 100
        insights["outlook"].append(f"EPS予想は前期比{eps_growth:+.1f}%。")
    if market_cap:
        insights["outlook"].append(f"時価総額{market_cap/1e8:,.0f}億円。")
    
    return insights


# =====================================================================
# 売買シグナル（優良銘柄マーク: 5条件判定）
# =====================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_price_history(ticker_str):
    """5年分の株価履歴を取得（年足/月足/週足判定用）。yfinanceから何度でも更新可能。"""
    try:
        hist = yf.Ticker(ticker_str).history(period="5y")
        if hist.empty:
            return None
        return hist
    except Exception:
        return None


def get_dividend_yield_pct(info):
    """配当利回り(%)を正しく計算して返す。
    yfinanceのdividendYieldは既にパーセント値(例: 2.66)で返ることがあるため、
    最も信頼できる dividendRate / price を優先する。
    """
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    div_rate = info.get("dividendRate")
    # 1. dividendRate / price（最も正確）
    if div_rate and price and price > 0:
        return div_rate / price * 100
    # 2. dividendYield（値の大きさで小数/％を判定）
    dy = info.get("dividendYield")
    if dy is not None:
        # 1未満なら小数(0.026)、1以上ならパーセント(2.6)と判定
        return dy * 100 if dy < 1 else dy
    # 3. trailingAnnualDividendYield（小数）
    tady = info.get("trailingAnnualDividendYield")
    if tady is not None:
        return tady * 100
    return 0


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividend_yield_history(ticker_str, years=5):
    """株価と配当履歴から、実績配当利回り(%)の推移を日次で計算して返す。

    各日時点の TTM配当（過去365日の1株配当合計）÷ 当日株価 × 100 を実績利回りとする。
    参考: 株価情報サイトの「実績配当利回り推移」と同じ考え方。

    Returns: dict {
        "df": DataFrame(index=日付, columns=["close","yield_pct","ttm_div"]),
        "div_dates": [Timestamp...]   # 配当落ち日（マーカー用）
    } or None
    """
    try:
        import pandas as _pd
        tk = yf.Ticker(ticker_str)
        hist = tk.history(period=f"{years}y", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].astype(float)
        # 配当イベント（配当落ち日と金額）
        if "Dividends" in hist.columns:
            div_series = hist["Dividends"].astype(float)
        else:
            div_series = _pd.Series(0.0, index=hist.index)
        div_series = div_series[div_series > 0]

        # 全配当履歴（historyの期間外も含めてTTMを正確化）
        try:
            all_div = tk.dividends
        except Exception:
            all_div = div_series
        if all_div is None or len(all_div) == 0:
            all_div = div_series

        # タイムゾーンを揃える
        idx = close.index
        # 各日のTTM配当（過去365日の配当合計）
        ttm = []
        # all_div のインデックスを close と同じtz基準で扱う
        div_idx = all_div.index
        div_vals = all_div.values
        for d in idx:
            lo = d - _pd.Timedelta(days=365)
            try:
                mask = (div_idx > lo) & (div_idx <= d)
                ttm.append(float(div_vals[mask].sum()))
            except Exception:
                ttm.append(0.0)
        ttm = _pd.Series(ttm, index=idx)

        yld = (ttm / close * 100).where(close > 0)
        df = _pd.DataFrame({"close": close, "yield_pct": yld, "ttm_div": ttm})
        df = df.dropna(subset=["yield_pct"])
        # TTM配当が0の初期区間（配当データが揃う前）は除外
        df = df[df["ttm_div"] > 0]
        if df.empty:
            return None

        div_dates = list(div_series.index)
        return {"df": df, "div_dates": div_dates}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_per_history(ticker_str, years=5):
    """株価とEPS履歴から、実績PER（株価収益率）の推移を日次で計算して返す。

    EPSは年次決算のEPS（取得できない場合は純利益÷発行済株式数）を用い、
    決算発表ラグ（期末の約3ヶ月後に判明）を考慮して各日に「その時点で判明していた
    最新の年次EPS」を割り当てる。各日のPER = 株価 ÷ 適用EPS。

    Returns: dict {
        "df": DataFrame(index=日付, columns=["close","per","eps"]),
        "eps_points": [(Timestamp, eps), ...]   # EPS切替点（マーカー用）
    } or None
    """
    try:
        import pandas as _pd
        import numpy as _np
        tk = yf.Ticker(ticker_str)
        hist = tk.history(period=f"{years}y", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].astype(float)

        # --- 年次EPSの系列を構築（決算期末日 -> EPS）---
        eps_by_date = {}  # 発表日(=期末+ラグ) -> eps
        try:
            ais = tk.income_stmt
        except Exception:
            ais = None

        shares = tk.info.get("sharesOutstanding") if hasattr(tk, "info") else None
        if ais is not None and not ais.empty:
            # EPS行を優先、無ければ純利益÷株式数
            eps_row = None
            for cand in ["Diluted EPS", "Basic EPS"]:
                if cand in ais.index:
                    eps_row = ais.loc[cand]
                    break
            ni_row = None
            for cand in ["Net Income From Continuing Operation Net Minority Interest",
                         "Net Income", "Net Income Common Stockholders"]:
                if cand in ais.index:
                    ni_row = ais.loc[cand]
                    break
            for col in ais.columns:
                fy_end = _pd.Timestamp(col)
                eps_val = None
                if eps_row is not None:
                    v = eps_row.get(col)
                    if v is not None and not (isinstance(v, float) and _np.isnan(v)):
                        eps_val = float(v)
                if eps_val is None and ni_row is not None and shares:
                    v = ni_row.get(col)
                    if v is not None and not (isinstance(v, float) and _np.isnan(v)) and shares > 0:
                        eps_val = float(v) / float(shares)
                if eps_val is not None and eps_val != 0:
                    # 決算発表ラグ: 期末の約3ヶ月後に市場へ反映
                    avail = fy_end + _pd.Timedelta(days=90)
                    eps_by_date[avail] = eps_val

        # 最新のtrailingEpsも末尾に追加（直近の反映を担保）
        _teps = tk.info.get("trailingEps") if hasattr(tk, "info") else None
        if _teps and _teps > 0:
            eps_by_date[close.index.max()] = float(_teps)

        if not eps_by_date:
            return None

        # tz合わせ: close.index の tz に揃える
        tz = close.index.tz
        norm = {}
        for d, e in eps_by_date.items():
            d2 = _pd.Timestamp(d)
            if tz is not None:
                d2 = d2.tz_localize(tz) if d2.tzinfo is None else d2.tz_convert(tz)
            norm[d2] = e
        eps_dates = sorted(norm.keys())

        # 各日に「その日以前で最新の」EPSを割り当て（ステップ関数）
        eps_series = []
        for d in close.index:
            applicable = [ed for ed in eps_dates if ed <= d]
            if applicable:
                eps_series.append(norm[applicable[-1]])
            else:
                eps_series.append(_np.nan)
        eps_series = _pd.Series(eps_series, index=close.index)

        per = (close / eps_series).where(eps_series > 0)
        df = _pd.DataFrame({"close": close, "per": per, "eps": eps_series})
        df = df.dropna(subset=["per"])
        # 異常値（赤字でPERマイナス、極端な高PER）を軽く制限
        df = df[(df["per"] > 0) & (df["per"] < 300)]
        if df.empty:
            return None

        eps_points = [(d, norm[d]) for d in eps_dates
                      if d >= df.index.min() and d <= df.index.max()]
        return {"df": df, "eps_points": eps_points}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nav_discount_history(ticker_str, nav_per_share, years=5):
    """SBG専用: NAVディスカウント率の推移を日次で計算して返す。

    公式SOTPページは「現在の」1株NAVのみ開示するため、最新の1株NAVを基準値として
    固定し、各日の株価に対するディスカウント率を計算する。
        ディスカウント率(%) = (1株NAV − 株価) / 1株NAV × 100
    NAVは最新値で固定しているため、過去の変動は主に株価変動に由来する近似である点に注意。

    Returns: dict {
        "df": DataFrame(index=日付, columns=["close","disc"]),
        "nav_per_share": float,
    } or None
    """
    try:
        import pandas as _pd
        if not nav_per_share or nav_per_share <= 0:
            return None
        tk = yf.Ticker(ticker_str)
        hist = tk.history(period=f"{years}y", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].astype(float)
        disc = (float(nav_per_share) - close) / float(nav_per_share) * 100.0
        df = _pd.DataFrame({"close": close, "disc": disc}).dropna()
        if df.empty:
            return None
        return {"df": df, "nav_per_share": float(nav_per_share)}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arm_price_history(years=1):
    """ARM Holdings（NASDAQ: ARM）の株価履歴（終値, USD）を返す。2023年9月上場。

    Returns: pandas.Series(index=日付, 値=終値USD) or None
    """
    try:
        tk = yf.Ticker("ARM")
        hist = tk.history(period=f"{years}y", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        return hist["Close"].astype(float)
    except Exception:
        return None


# OpenAI 推定時価総額のマイルストーン（十億ドル）。各日付以降はその評価額を適用（階段状）。
OPENAI_VALUATION_MILESTONES = [
    ("2019-07-01", 14),
    ("2023-01-01", 29),
    ("2024-02-01", 80),
    ("2024-10-01", 157),
    ("2025-03-01", 300),
    ("2025-10-01", 500),
    ("2026-03-01", 852),
]


def build_openai_valuation_series(index):
    """与えられた日付インデックスに対し、OpenAI推定時価総額（兆ドル）を階段状で割り当てたSeriesを返す。"""
    try:
        import pandas as _pd
        import numpy as _np
        tz = getattr(index, "tz", None)
        ms = []
        for d, v in OPENAI_VALUATION_MILESTONES:
            ts = _pd.Timestamp(d)
            if tz is not None:
                ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
            ms.append((ts, v / 1000.0))  # 十億ドル → 兆ドル
        ms.sort(key=lambda x: x[0])
        vals = []
        for d in index:
            applicable = [v for ts, v in ms if ts <= d]
            vals.append(applicable[-1] if applicable else _np.nan)
        return _pd.Series(vals, index=index)
    except Exception:
        return None


def calc_asset_value_metrics(balance, info):
    """資産バリュー性の指標を計算する。

    1) 清原式ネットキャッシュ比率（『わが投資術』）
       = （流動資産 ＋ 投資有価証券 × 70％ － 負債合計） ÷ 時価総額
    2) 正味流動資産（ネットネット, かぶ1000流）
       = （現金預金 ＋ 売掛金 ＋ 有価証券 ＋ 投資有価証券） － （負債合計 ＋ 貸倒引当金）

    balance: yfinance の balance_sheet（DataFrame）。info: yfinance info(dict)。
    金額は円。取得できない項目は None として、計算可否を返す。

    Returns: dict {
        "current_assets", "investment_securities", "total_liabilities",
        "cash", "receivables", "short_term_securities", "allowance_doubtful",
        "market_cap",
        "kiyohara_net_cash", "kiyohara_ratio",   # 清原式：金額と時価総額比
        "net_net_value", "net_net_ratio",         # 正味流動資産：金額と時価総額比
        "note": 補足メッセージ or None,
    } すべて取れない場合は None。
    """
    if balance is None or getattr(balance, "empty", True):
        return None

    def _b(*labels):
        """balance_sheet 最新列から、候補ラベルのいずれかの値を返す（無ければ None）。"""
        try:
            col = balance.iloc[:, 0]
        except Exception:
            return None
        for lb in labels:
            if lb in col.index:
                v = col.get(lb)
                if v is not None and not (isinstance(v, float) and (v != v)):  # NaN除外
                    return float(v)
        return None

    current_assets = _b("Current Assets")
    investment_securities = _b("Investmentin Financial Assets", "Available For Sale Securities",
                               "Long Term Equity Investment", "Investments And Advances")
    total_liabilities = _b("Total Liabilities Net Minority Interest", "Total Liabilities")
    cash = _b("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    receivables = _b("Accounts Receivable", "Gross Accounts Receivable", "Receivables")
    short_term_securities = _b("Other Short Term Investments")
    # 貸倒引当金は yfinance に独立項目がないことが多い（売掛金は控除後）。取れなければ0扱い。
    allowance_doubtful = _b("Allowance For Doubtful Accounts Receivable")

    market_cap = info.get("marketCap")

    # --- 清原式ネットキャッシュ ---
    kiyohara_net_cash = None
    kiyohara_ratio = None
    if current_assets is not None and total_liabilities is not None:
        inv = investment_securities or 0.0
        kiyohara_net_cash = current_assets + inv * 0.70 - total_liabilities
        if market_cap:
            kiyohara_ratio = kiyohara_net_cash / market_cap

    # --- 換金性資産の正味額（かぶ1000流ネットネット）---
    # 換金性資産 ＝ 現金＋売掛金＋有価証券＋投資有価証券（貸倒引当金控除後）
    # 正味額     ＝ 換金性資産 － 総負債
    liquid_assets = None
    net_net_value = None      # 正味額（換金性資産－総負債）
    net_net_ratio = None      # 正味額 ÷ 時価総額（大きいほど割安）
    net_net_index = None      # かぶ1000流ネットネット指数 = 時価総額 ÷ 正味額（1未満が割安）
    liquid_coverage = None    # 換金性資産カバー率 = 換金性資産 ÷ 時価総額（負債控除前・連続評価用）
    _plus = [x for x in (cash, receivables, short_term_securities, investment_securities) if x is not None]
    if _plus and total_liabilities is not None:
        liquid_assets = sum(_plus) - (allowance_doubtful or 0.0)  # 換金性資産（貸倒引当金控除）
        net_net_value = liquid_assets - total_liabilities
        if market_cap:
            net_net_ratio = net_net_value / market_cap
            liquid_coverage = liquid_assets / market_cap
            # ネットネット指数: 正味額がプラスのときのみ意味を持つ（マイナスは対象外）
            if net_net_value > 0:
                net_net_index = market_cap / net_net_value

    if kiyohara_net_cash is None and net_net_value is None:
        return None

    note = None
    if allowance_doubtful is None:
        note = "貸倒引当金は個別開示が取得できないため0として計算（売掛金は通常、貸倒引当金控除後の純額）。"

    return {
        "current_assets": current_assets,
        "investment_securities": investment_securities,
        "total_liabilities": total_liabilities,
        "cash": cash,
        "receivables": receivables,
        "short_term_securities": short_term_securities,
        "allowance_doubtful": allowance_doubtful,
        "market_cap": market_cap,
        "liquid_assets": liquid_assets,
        "kiyohara_net_cash": kiyohara_net_cash,
        "kiyohara_ratio": kiyohara_ratio,
        "net_net_value": net_net_value,
        "net_net_ratio": net_net_ratio,
        "net_net_index": net_net_index,
        "liquid_coverage": liquid_coverage,
        "note": note,
    }


def judge_asset_value(av):
    """資産バリュー指標から総合判定を返す。
    Returns: dict {"level": 0-3, "label": str, "icon": str, "reasons": [str]} or None
    """
    if not av:
        return None
    kr = av.get("kiyohara_ratio")
    nni = av.get("net_net_index")
    lc = av.get("liquid_coverage")
    level = 0
    reasons = []
    kr_strong = kr is not None and kr > 1
    nni_strong = nni is not None and nni < 1
    if kr_strong and nni_strong:
        reasons.append("清原NC比率>1 かつ ネットネット指数<1")
        level = 3
    else:
        if kr_strong:
            reasons.append("清原NC比率>1")
            level = max(level, 3)
        if nni is not None and nni < 0.67:
            reasons.append("ネットネット指数<0.67")
            level = max(level, 2)
        elif nni is not None and nni < 1:
            reasons.append("ネットネット指数<1")
            level = max(level, 2)
        if lc is not None and lc >= 1.0:
            reasons.append("換金性資産カバー率≥1")
            level = max(level, 2)
        elif lc is not None and lc >= 0.67:
            reasons.append("換金性資産カバー率≥0.67")
            level = max(level, 1)
        if kr is not None and 0.5 < kr <= 1:
            reasons.append("清原NC比率>0.5")
            level = max(level, 1)
    label, icon = {
        3: ("非常に強い", "💎"),
        2: ("割安", "🟢"),
        1: ("下支えあり", "🟡"),
        0: ("限定的", "⚪"),
    }[level]
    return {"level": level, "label": label, "icon": icon, "reasons": reasons}


def judge_tachan_screening(info, balance, financials):
    """『50万円を50億円に増やした 投資家の父から娘への教え』（たーちゃん著）の
    3つのバリュー株スクリーニング条件への適合度を判定する。

    1) 資産バリュー株: PBR 0.5倍以下 かつ 自己資本比率 60%以上
       （収益性も見るなら PER 12倍以下 を加点）／グレアム指数 PER×PBR≦22.5
    2) 収益バリュー株: 営業利益率10%以上・PER10倍以下・PBR1.5倍以下・ROA7%以上・時価総額300億円以下
       （緩和: 営業利益率8%、PER15倍まで許容）
    3) シクリカルバリュー株: 景気循環業種 × 直近赤字（赤字こそ妙味）× PSR低位（0.5倍以下が割安目安）
       ※景気循環の「谷」を狙う投資。黒字の最高益圏は対象外。

    info: yfinance info(dict)、balance: balance_sheet(DataFrame)、financials: income_stmt(DataFrame)
    Returns: dict {
      "asset": {...}, "earnings": {...}, "cyclical": {...},
      "best": "asset"|"earnings"|"cyclical"|None,
    } 判定不能な指標は None。
    """
    if not info:
        return None

    # ---- 基礎指標の取得 ----
    pbr = info.get("priceToBook")
    per = info.get("trailingPE") or info.get("forwardPE")
    roa = info.get("returnOnAssets")            # 比率（0.07 = 7%）
    op_margin = info.get("operatingMargins")    # 比率（0.10 = 10%）
    market_cap = info.get("marketCap")          # 円
    revenue = info.get("totalRevenue")          # 円
    net_income = info.get("netIncomeToCommon")  # 円

    # 自己資本比率（%）: 純資産 ÷ 総資産。balance から算出
    equity_ratio_pct = None

    def _b(*labels):
        if balance is None or getattr(balance, "empty", True):
            return None
        try:
            col = balance.iloc[:, 0]
        except Exception:
            return None
        for lb in labels:
            if lb in col.index:
                v = col.get(lb)
                if v is not None and not (isinstance(v, float) and (v != v)):
                    return float(v)
        return None

    _equity = _b("Stockholders Equity", "Total Equity Gross Minority Interest",
                 "Common Stock Equity")
    _total_assets = _b("Total Assets")
    if _equity and _total_assets and _total_assets > 0:
        equity_ratio_pct = _equity / _total_assets * 100.0

    # PSR = 時価総額 ÷ 売上高
    psr = None
    if market_cap and revenue and revenue > 0:
        psr = market_cap / revenue

    # グレアム指数 = PER × PBR
    graham = None
    if per and pbr and per > 0 and pbr > 0:
        graham = per * pbr

    mcap_oku = (market_cap / 1e8) if market_cap else None  # 億円

    # ---- 直近の赤字判定（シクリカル用）: 純利益がマイナス ----
    is_loss = None
    recent_losses = 0
    if financials is not None and not getattr(financials, "empty", True):
        try:
            ni_row = None
            for cand in ("Net Income", "Net Income Common Stockholders",
                         "Net Income From Continuing Operation Net Minority Interest"):
                if cand in financials.index:
                    ni_row = financials.loc[cand]
                    break
            if ni_row is not None:
                vals = [float(v) for v in ni_row.values[:3]
                        if v is not None and not (isinstance(v, float) and (v != v))]
                if vals:
                    is_loss = vals[0] < 0
                    recent_losses = sum(1 for v in vals if v < 0)
        except Exception:
            pass
    if is_loss is None and net_income is not None:
        is_loss = net_income < 0

    # 景気循環業種か（sector/industry のヒューリスティック）
    cyclical_sectors = {
        "Basic Materials", "Energy", "Industrials", "Consumer Cyclical",
        "Financial Services", "Real Estate",
    }
    cyclical_kw = ["steel", "ship", "marine", "shipping", "semiconductor",
                   "oil", "gas", "mining", "metal", "chemical", "auto",
                   "construction", "machinery", "paper", "aluminum", "copper"]
    sector = (info.get("sector") or "")
    industry = (info.get("industry") or "").lower()
    is_cyclical_industry = (sector in cyclical_sectors) or any(k in industry for k in cyclical_kw)

    # ============ ① 資産バリュー株 ============
    a_reasons, a_fail = [], []
    a_core = []  # 必須条件の達成フラグ
    if pbr is not None:
        if pbr <= 0.5:
            a_reasons.append(f"PBR {pbr:.2f}倍 ≤ 0.5")
            a_core.append(True)
        else:
            a_fail.append(f"PBR {pbr:.2f}倍 >0.5")
            a_core.append(False)
    if equity_ratio_pct is not None:
        if equity_ratio_pct >= 60:
            a_reasons.append(f"自己資本比率 {equity_ratio_pct:.0f}% ≥ 60%")
            a_core.append(True)
        else:
            a_fail.append(f"自己資本比率 {equity_ratio_pct:.0f}% <60%")
            a_core.append(False)
    # 加点: PER12倍以下 / グレアム指数22.5以下
    if per is not None and per <= 12:
        a_reasons.append(f"PER {per:.1f}倍 ≤ 12（収益性も割安）")
    if graham is not None and graham <= 22.5:
        a_reasons.append(f"グレアム指数 PER×PBR={graham:.1f} ≤ 22.5")
    asset_match = len(a_core) > 0 and all(a_core)

    # ============ ② 収益バリュー株 ============
    e_reasons, e_fail = [], []
    e_core = []
    e_relaxed = False
    if op_margin is not None:
        omp = op_margin * 100
        if op_margin >= 0.10:
            e_reasons.append(f"営業利益率 {omp:.1f}% ≥ 10%")
            e_core.append(True)
        elif op_margin >= 0.08:
            e_reasons.append(f"営業利益率 {omp:.1f}%（緩和基準8%）")
            e_core.append(True)
            e_relaxed = True
        else:
            e_fail.append(f"営業利益率 {omp:.1f}% <8%")
            e_core.append(False)
    if per is not None:
        if per <= 10:
            e_reasons.append(f"PER {per:.1f}倍 ≤ 10")
            e_core.append(True)
        elif per <= 15:
            e_reasons.append(f"PER {per:.1f}倍（緩和基準15倍）")
            e_core.append(True)
            e_relaxed = True
        else:
            e_fail.append(f"PER {per:.1f}倍 >15")
            e_core.append(False)
    if pbr is not None:
        if pbr <= 1.5:
            e_reasons.append(f"PBR {pbr:.2f}倍 ≤ 1.5")
            e_core.append(True)
        else:
            e_fail.append(f"PBR {pbr:.2f}倍 >1.5")
            e_core.append(False)
    if roa is not None:
        rap = roa * 100
        if roa >= 0.07:
            e_reasons.append(f"ROA {rap:.1f}% ≥ 7%")
            e_core.append(True)
        else:
            e_fail.append(f"ROA {rap:.1f}% <7%")
            e_core.append(False)
    if mcap_oku is not None:
        if mcap_oku <= 300:
            e_reasons.append(f"時価総額 {mcap_oku:,.0f}億円 ≤ 300億円（小型）")
            e_core.append(True)
        else:
            e_fail.append(f"時価総額 {mcap_oku:,.0f}億円 >300億円")
            e_core.append(False)
    earnings_match = len(e_core) > 0 and all(e_core)

    # ============ ③ シクリカルバリュー株 ============
    c_reasons, c_fail = [], []
    if is_cyclical_industry:
        c_reasons.append(f"景気循環業種（{sector or industry or '—'}）")
    else:
        c_fail.append("景気循環業種に該当しない可能性")
    if is_loss is True:
        if recent_losses >= 2:
            c_reasons.append(f"直近{recent_losses}期が赤字（谷＝妙味大）")
        else:
            c_reasons.append("直近が赤字（回復期待の仕込み局面）")
    elif is_loss is False:
        c_fail.append("黒字（景気の谷ではない＝対象外の可能性）")
    if psr is not None:
        if psr <= 0.5:
            c_reasons.append(f"PSR {psr:.2f}倍 ≤ 0.5（売上対比で割安）")
        elif psr >= 20:
            c_fail.append(f"PSR {psr:.1f}倍 ≥ 20（期待先行で割高）")
    # シクリカル成立: 景気循環業種 かつ 直近赤字（PSRは割安度の補助）
    cyclical_match = bool(is_cyclical_industry and is_loss is True)

    # ---- ベスト適合（資産→収益→シクリカルの優先で代表を選ぶ）----
    best = None
    if earnings_match:
        best = "earnings"
    elif asset_match:
        best = "asset"
    elif cyclical_match:
        best = "cyclical"

    return {
        "asset": {"match": asset_match, "reasons": a_reasons, "fail": a_fail,
                  "pbr": pbr, "equity_ratio": equity_ratio_pct, "per": per, "graham": graham},
        "earnings": {"match": earnings_match, "relaxed": e_relaxed,
                     "reasons": e_reasons, "fail": e_fail,
                     "op_margin": op_margin, "per": per, "pbr": pbr, "roa": roa,
                     "mcap_oku": mcap_oku},
        "cyclical": {"match": cyclical_match, "reasons": c_reasons, "fail": c_fail,
                     "is_cyclical_industry": is_cyclical_industry, "is_loss": is_loss,
                     "recent_losses": recent_losses, "psr": psr,
                     "sector": sector, "industry": info.get("industry")},
        "best": best,
    }


def _generate_composite_signal(dow, breakout, deviation, momentum):
    """総合売買シグナル生成（-100〜100スコア）— 別アプリ同仕様
    dow: 上昇/下降/なし, breakout: 高値/安値/MA20上/MA20下/なし,
    deviation: 理論株価乖離%, momentum: モメンタム(-10〜10)
    """
    score = 0
    # ダウ理論
    if dow == "上昇トレンド":
        score += 25
    elif dow == "下降トレンド":
        score -= 25
    # ブレイクアウト
    if "高値ブレイク" in breakout:
        score += 15
    elif "安値ブレイク" in breakout:
        score -= 15
    elif "MA20上" in breakout:
        score += 5
    elif "MA20下" in breakout:
        score -= 5
    # 理論株価乖離（割安なら加点）
    if deviation is not None:
        if deviation < -10:
            score += 15
        elif deviation < -5:
            score += 8
        elif deviation > 10:
            score -= 15
        elif deviation > 5:
            score -= 8
    # モメンタム
    score += momentum * 1.5
    score = max(-100, min(100, score))

    if score > 30:
        return "強い買い 🟢", "#00c853", round(score, 1)
    elif score > 10:
        return "買い 🔵", "#2196f3", round(score, 1)
    elif score > -10:
        return "中立 ⚪", "#9e9e9e", round(score, 1)
    elif score > -30:
        return "売り 🟠", "#ff9800", round(score, 1)
    else:
        return "強い売り 🔴", "#f44336", round(score, 1)


def calc_trade_signal(ticker_str, info):
    """優良銘柄マーク（5条件） + 売買シグナル + 期待値スコア
    Returns: dict
    """
    hist = fetch_price_history(ticker_str)

    cond_yearly = False
    cond_monthly = False
    cond_weekly = False
    yearly_label = ""
    dow = "なし"
    breakout = "なし"
    momentum = 0.0

    if hist is not None and len(hist) > 60:
        close = hist["Close"]
        last = float(close.iloc[-1])

        # ① 年足陽線
        try:
            yearly = close.resample("YE").last()
            yearly_open = close.resample("YE").first()
            bullish_years = 0
            consecutive = 0
            for yc, yo in zip(yearly.values, yearly_open.values):
                if yc > yo:
                    bullish_years += 1
                    consecutive += 1
                else:
                    consecutive = 0
            cond_yearly = consecutive >= 1 or bullish_years >= 2
            yearly_label = f"陽線{bullish_years}/{len(yearly)}年 連続{consecutive}"
        except Exception:
            pass

        # ② 月足↑
        try:
            monthly = close.resample("ME").last()
            if len(monthly) >= 4:
                cond_monthly = monthly.iloc[-1] > monthly.iloc[-4]
        except Exception:
            pass

        # ③ 週足↑
        try:
            weekly = close.resample("W").last()
            if len(weekly) >= 5:
                cond_weekly = weekly.iloc[-1] > weekly.iloc[-5]
        except Exception:
            pass

        # ダウ理論（直近の高値・安値切り上げ判定）
        try:
            ma25 = close.rolling(25).mean()
            ma75 = close.rolling(75).mean()
            if last > ma25.iloc[-1] > ma75.iloc[-1]:
                dow = "上昇トレンド"
            elif last < ma25.iloc[-1] < ma75.iloc[-1]:
                dow = "下降トレンド"
        except Exception:
            pass

        # ブレイクアウト
        try:
            high_20 = close.iloc[-20:].max()
            low_20 = close.iloc[-20:].min()
            ma20 = close.rolling(20).mean().iloc[-1]
            if last >= high_20:
                breakout = "高値ブレイク"
            elif last <= low_20:
                breakout = "安値ブレイク"
            elif last > ma20:
                breakout = "MA20上"
            else:
                breakout = "MA20下"
        except Exception:
            pass

        # モメンタム（20日リターン%を-10〜10に正規化）
        try:
            if len(close) >= 21:
                ret_20d = (last / float(close.iloc[-21]) - 1) * 100
                momentum = max(-10, min(10, ret_20d / 2))
        except Exception:
            pass

    # ④ 配当利回り2.5%以上
    dy_pct = get_dividend_yield_pct(info)
    cond_yield = dy_pct >= 2.5

    # ⑤ 好業績
    roe = info.get("returnOnEquity") or 0
    eps_growth = info.get("earningsGrowth") or 0
    rev_growth = info.get("revenueGrowth") or 0
    cond_biz = (roe >= 0.10) or (eps_growth > 0) or (rev_growth > 0.05)

    cond_count = sum([cond_yearly, cond_monthly, cond_weekly, cond_yield, cond_biz])

    if cond_count >= 5:
        mark, gem_label = "💎", "完璧"
    elif cond_count >= 4:
        mark, gem_label = "🌟", "優良"
    elif cond_count >= 3:
        mark, gem_label = "⭐", "注目"
    else:
        mark, gem_label = "", ""

    # 理論株価乖離（PER基準の簡易版: forwardPEが業界より高いか）
    per = info.get("forwardPE") or info.get("trailingPE")
    deviation = None
    if per:
        # PER30を基準に乖離を推定（高PER=割高=正の乖離）
        deviation = (per / 30 - 1) * 100

    # 総合売買シグナル
    signal_text, signal_color, signal_score = _generate_composite_signal(dow, breakout, deviation, momentum)

    # 期待値スコア（売買シグナル + 優良マークボーナス + 好業績ボーナス）
    expect = signal_score
    expect += cond_count * 5  # 5条件ボーナス（最大25点）
    if cond_biz:
        expect += 5
    if eps_growth and eps_growth > 0.1:
        expect += 5
    expect = max(-100, min(100, round(expect, 0)))

    return {
        "mark": mark,
        "label": gem_label,
        "cond_count": cond_count,
        "cond_yearly": cond_yearly,
        "cond_monthly": cond_monthly,
        "cond_weekly": cond_weekly,
        "cond_yield": cond_yield,
        "cond_biz": cond_biz,
        "yearly_label": yearly_label,
        "dy_pct": dy_pct,
        "dow": dow,
        "breakout": breakout,
        "signal_text": signal_text,
        "signal_color": signal_color,
        "signal_score": signal_score,
        "expect": expect,
        "detail": (
            f"{'✅' if cond_yearly else '❌'}年足 "
            f"{'✅' if cond_monthly else '❌'}月足 "
            f"{'✅' if cond_weekly else '❌'}週足 "
            f"{'✅' if cond_yield else '❌'}利回{dy_pct:.1f}% "
            f"{'✅' if cond_biz else '❌'}業績"
        ),
    }


# =====================================================================
# 銘柄辞書（予測変換用）
# =====================================================================
STOCK_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_dict_cache.json")

@st.cache_data(ttl=86400, show_spinner=False)
def load_stock_dict():
    """全上場銘柄の辞書を読み込む（コード→銘柄名）"""
    if os.path.exists(STOCK_DICT_PATH):
        try:
            with open(STOCK_DICT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# 起動時に辞書ロード
_STOCK_DICT = load_stock_dict()

# 逆引き辞書（銘柄名→コード）も構築
_NAME_TO_CODE = {}
for _code, _name in _STOCK_DICT.items():
    _NAME_TO_CODE[_name] = _code


def search_stocks(query, stock_dict, max_results=20):
    """銘柄コードまたは銘柄名で部分一致検索"""
    if not query or not query.strip():
        return []
    q = query.strip()
    results = []
    
    # コード検索（前方一致）
    for code, name in stock_dict.items():
        code_num = code.replace(".T", "")
        if code_num.startswith(q):
            results.append((code, name))
            if len(results) >= max_results:
                break
    
    # 名前検索（部分一致）— コード検索でまだ足りない場合
    if len(results) < max_results:
        for code, name in stock_dict.items():
            if q in name and (code, name) not in results:
                results.append((code, name))
                if len(results) >= max_results:
                    break
    
    return results


# =====================================================================
# メイン
# =====================================================================
watchlist = load_watchlist()

# サイドバー
with st.sidebar:
    st.markdown("### EARNINGS INTELLIGENCE")
    st.caption(f"{len(watchlist)} COMPANIES")
    
    st.divider()
    st.subheader("銘柄コード・会社名")
    
    # 銘柄選択
    ticker_options = [f"{code.replace('.T','')} | {name}" for code, name in watchlist.items()]
    selected_option = st.selectbox("検索", ticker_options, label_visibility="collapsed")
    selected_ticker = list(watchlist.keys())[ticker_options.index(selected_option)]
    
    st.divider()
    
    # 銘柄追加（予測変換付き）
    st.subheader("ウォッチリスト管理")
    
    # 検索入力
    search_query = st.text_input(
        "🔍 銘柄検索（コード or 名前を入力）",
        "",
        placeholder="例: 8136, サンリオ, トヨタ",
        key="stock_search",
    )
    
    # 検索結果を表示
    if search_query:
        search_results = search_stocks(search_query, _STOCK_DICT)
        if search_results:
            # selectboxで候補を表示
            candidates = [f"{code.replace('.T','')} | {name}" for code, name in search_results]
            selected_candidate = st.selectbox(
                f"検索結果（{len(search_results)}件）",
                candidates,
                key="search_results_select",
            )
            
            # 追加ボタン
            if st.button("✅ ウォッチリストに追加", key="add_from_search"):
                # 選択された候補からコードを取得
                sel_code = search_results[candidates.index(selected_candidate)][0]
                sel_name = search_results[candidates.index(selected_candidate)][1]
                if sel_code not in watchlist:
                    watchlist[sel_code] = sel_name
                    save_watchlist(watchlist)
                    st.success(f"✅ {sel_code.replace('.T','')} {sel_name} を追加しました")
                    st.rerun()
                else:
                    st.info(f"{sel_name} は既にウォッチリストに登録済みです")
        else:
            # 辞書にない場合でも直接コードで追加可能にする
            st.caption("辞書に該当なし。新形式コード（151Aなど）は下のボタンで直接追加できます。")
            if st.button(f"📌 「{search_query.strip()}」をそのまま追加", key="add_direct"):
                direct_code = search_query.strip().upper()
                if not direct_code.endswith(".T"):
                    direct_code += ".T"
                if direct_code not in watchlist:
                    # yfinanceで名前を取得試行
                    try:
                        _tk_info = yf.Ticker(direct_code).info or {}
                        _name = _tk_info.get("shortName") or _tk_info.get("longName") or direct_code
                    except Exception:
                        _name = direct_code
                    watchlist[direct_code] = _name
                    save_watchlist(watchlist)
                    st.success(f"✅ {direct_code.replace('.T','')} {_name} を追加しました")
                    st.rerun()
                else:
                    st.info("既に登録済みです")
    
    st.divider()
    
    # 削除
    st.caption("銘柄を削除")
    remove_option = st.selectbox("削除対象", ["---"] + ticker_options, key="remove_select")
    if st.button("🗑️ 削除", key="remove_btn") and remove_option != "---":
        rm_ticker = list(watchlist.keys())[ticker_options.index(remove_option)]
        del watchlist[rm_ticker]
        save_watchlist(watchlist)
        st.rerun()
    
    st.divider()
    st.subheader("🔄 データ更新")
    st.caption("株価・売買シグナルはyfinanceから何度でも更新可能（EDINETは7日キャッシュ）")

    if st.button("🔄 株価・シグナルを更新", key="refresh_price"):
        # yfinance系キャッシュのみクリア（EDINETディスクキャッシュは保持）
        fetch_stock_data.clear()
        fetch_price_history.clear()
        st.success("株価・シグナルを更新しました")
        st.rerun()

    # EDINETキャッシュ状態表示
    _edb_cache_file = _edb_cache_path(selected_ticker)
    if os.path.exists(_edb_cache_file):
        _age_days = (time.time() - os.path.getmtime(_edb_cache_file)) / 86400
        _remaining = max(0, 7 - _age_days)
        st.caption(f"💾 {selected_ticker.replace('.T','')} 決算キャッシュ: あと{_remaining:.1f}日有効")
        if st.button("🗑️ この銘柄の決算キャッシュを削除", key="clear_edb"):
            try:
                os.remove(_edb_cache_file)
                fetch_edinetdb_full.clear()
                st.success("決算キャッシュを削除しました。次回EDINETから再取得します。")
                st.rerun()
            except Exception as e:
                st.error(f"削除失敗: {e}")
    else:
        st.caption(f"💾 {selected_ticker.replace('.T','')} 決算キャッシュ: なし（次回EDINET取得）")

    st.divider()
    st.markdown("""
    **📌 リンク**
    - [株探](https://kabutan.jp/)
    - [四季報](https://shikiho.toyokeizai.net/)
    - [バフェットコード](https://www.buffett-code.com/)
    """)

# データ取得
with st.spinner("データ取得中..."):
    all_stock_data = fetch_all_stocks(watchlist)

if selected_ticker not in all_stock_data:
    st.error(f"{selected_ticker} のデータを取得できませんでした")
    st.stop()

current_data = all_stock_data[selected_ticker]
info = current_data["info"]
company_name = watchlist.get(selected_ticker, info.get("shortName", selected_ticker))

# =====================================================================
# ヘッダー
# =====================================================================
st.markdown(f"""
<div class="header-bar">
    <div style="font-size:0.75rem; color:#aaa; letter-spacing:2px;">EARNINGS INTELLIGENCE / {len(watchlist)} COMPANIES</div>
    <h1>{company_name}（{selected_ticker.replace('.T','')}）決算ダッシュボード</h1>
    <div class="subtitle">株価 {info.get('currentPrice') or info.get('regularMarketPrice', 'N/A'):,}円 / {datetime.now().strftime('%Y-%m-%d')}</div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 全上場スクリーニング（かぶ1000・たーちゃん式）— EDINET DB /v1/screener
#   全上場企業(約3,900社)を1リクエストで横断フィルタ。結果は1日キャッシュ。
# =====================================================================
with st.expander("🔍 全上場スクリーニング（かぶ1000・たーちゃん式）", expanded=False):
    st.caption("EDINET DB の全上場企業データを横断して、著名投資家のスクリーニング条件に合致する銘柄を抽出します。"
               "1回の抽出につきAPI 1リクエスト（1日キャッシュ）。データ出典: EDINET DB (https://edinetdb.jp)")

    # プリセット条件定義: (metric, operator, value)。market-capは百万円、利益率/利回りは%、比率は倍。
    _SCREEN_PRESETS = {
        "かぶ1000流ネットネット（清原NC比率≥1）": {
            "desc": "清原NC比率（ネットキャッシュ÷時価総額）≥1。換金性資産だけで時価総額をカバーする超割安。",
            "conditions": [("net-cash-ratio", "gte", 1.0)],
            "sort_by": "net-cash-ratio", "sort_order": "desc",
            "cols": ["net-cash-ratio", "pbr", "per"],
        },
        "かぶ1000流ネットネット（やや緩め・比率≥0.5）": {
            "desc": "清原NC比率≥0.5 かつ PBR≤1。資産の下支えがある割安株を広めに抽出。",
            "conditions": [("net-cash-ratio", "gte", 0.5), ("pbr", "lte", 1.0)],
            "sort_by": "net-cash-ratio", "sort_order": "desc",
            "cols": ["net-cash-ratio", "pbr", "per"],
        },
        "たーちゃん①資産バリュー株": {
            "desc": "PBR≤0.5 かつ 自己資本比率≥60%。『安いから買うんじゃない、資産があるから買う』。",
            "conditions": [("pbr", "lte", 0.5), ("equity-ratio", "gte", 60.0)],
            "sort_by": "pbr", "sort_order": "asc",
            "cols": ["pbr", "equity-ratio", "per"],
        },
        "たーちゃん②収益バリュー株": {
            "desc": "営業利益率≥10%・PER≤10倍・PBR≤1.5倍・ROA≥7%・時価総額≤300億円。稼ぐ力に対して割安な小型株。",
            "conditions": [("operating-margin", "gte", 10.0), ("per", "lte", 10.0),
                           ("pbr", "lte", 1.5), ("roa", "gte", 7.0), ("market-cap", "lte", 30000.0)],
            "sort_by": "operating-margin", "sort_order": "desc",
            "cols": ["operating-margin", "per", "pbr", "roa", "market-cap"],
        },
        "たーちゃん②収益バリュー株（緩和基準）": {
            "desc": "営業利益率≥8%・PER≤15倍・PBR≤1.5倍・ROA≥7%・時価総額≤300億円。該当が少ない時の緩和版。",
            "conditions": [("operating-margin", "gte", 8.0), ("per", "lte", 15.0),
                           ("pbr", "lte", 1.5), ("roa", "gte", 7.0), ("market-cap", "lte", 30000.0)],
            "sort_by": "operating-margin", "sort_order": "desc",
            "cols": ["operating-margin", "per", "pbr", "roa", "market-cap"],
        },
        "たーちゃん③シクリカルバリュー株（赤字×低PSR）": {
            "desc": "直近純利益が赤字 かつ PSR≤0.5倍。景気の谷で仕込む候補（業種の景気局面はご自身で確認）。",
            "conditions": [("net-margin", "lte", 0.0), ("psr", "lte", 0.5)],
            "sort_by": "psr", "sort_order": "asc",
            "cols": ["psr", "net-margin", "market-cap"],
        },
        "グレアム割安（PBR≤1・PER≤10・自己資本≥50%）": {
            "desc": "ベンジャミン・グレアム系の保守的割安。PBR≤1・PER≤10・自己資本比率≥50%。",
            "conditions": [("pbr", "lte", 1.0), ("per", "lte", 10.0), ("equity-ratio", "gte", 50.0)],
            "sort_by": "per", "sort_order": "asc",
            "cols": ["pbr", "per", "equity-ratio"],
        },
    }

    _scr_choice = st.selectbox("スクリーニング条件", list(_SCREEN_PRESETS.keys()), key="screen_preset")
    _preset = _SCREEN_PRESETS[_scr_choice]
    st.caption(f"📋 {_preset['desc']}")
    _cond_txt = " ／ ".join(
        f"{m} {'≤' if o=='lte' else '≥' if o=='gte' else '='} {v:g}" for m, o, v in _preset["conditions"]
    )
    st.caption(f"条件: {_cond_txt}（market-capは百万円）")

    _scr_c1, _scr_c2 = st.columns([1, 3])
    with _scr_c1:
        _scr_limit = st.number_input("最大表示件数", min_value=10, max_value=500, value=100, step=10,
                                     key="screen_limit")
    with _scr_c2:
        _scr_only_watch = st.checkbox("ウォッチリスト追加ボタンを表示", value=True, key="screen_addbtn")

    if st.button("🔎 この条件で全上場をスクリーニング", key="run_screen"):
        st.session_state["screen_run"] = True
        st.session_state["screen_run_key"] = _scr_choice

    if st.session_state.get("screen_run") and st.session_state.get("screen_run_key") == _scr_choice:
        with st.spinner("EDINET DB で全上場企業を抽出中…"):
            _res = fetch_edinetdb_screener(
                _preset["conditions"], sort_by=_preset["sort_by"],
                sort_order=_preset["sort_order"], limit=int(_scr_limit),
            )
        if _res.get("error"):
            st.error(f"スクリーニング失敗: {_res['error']}")
        else:
            _total = _res.get("total", 0)
            _as_of = _res.get("as_of") or ""
            st.success(f"✅ 該当 {_total:,}社（上位{len(_res['companies'])}社を表示）"
                       + (f" ／ データ基準日 {_as_of}" if _as_of else ""))

            # 列ラベルと整形
            _col_label = {
                "net-cash-ratio": "清原NC比率", "pbr": "PBR", "per": "PER",
                "equity-ratio": "自己資本比率%", "operating-margin": "営業利益率%",
                "roa": "ROA%", "psr": "PSR", "net-margin": "純利益率%",
                "market-cap": "時価総額(億円)", "dividend-yield": "配当利回り%",
            }
            _rows = []
            for c in _res["companies"]:
                _sec = str(c.get("secCode", "")).rstrip("0") if c.get("secCode") else ""
                # secCodeは末尾0が付く5桁表記（例 61510→6151）。4桁化。
                _code4 = str(c.get("secCode", ""))
                if len(_code4) == 5 and _code4.endswith("0"):
                    _code4 = _code4[:4]
                _row = {
                    "コード": _code4,
                    "銘柄名": c.get("filerName", ""),
                    "業種": c.get("industry", ""),
                    "市場": c.get("marketSegment", ""),
                }
                for mk in _preset["cols"]:
                    v = c.get(mk)
                    if v is None:
                        _row[_col_label.get(mk, mk)] = None
                    elif mk == "market-cap":
                        _row[_col_label[mk]] = round(v / 100.0, 0)  # 百万円→億円
                    else:
                        _row[_col_label.get(mk, mk)] = round(v, 2)
                _rows.append(_row)

            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True, height=440)

            # ウォッチリストへ一括追加
            if _scr_only_watch and _rows:
                _add_codes = st.multiselect(
                    "ウォッチリストに追加する銘柄を選択",
                    [f'{r["コード"]} | {r["銘柄名"]}' for r in _rows if r["コード"]],
                    key="screen_add_sel",
                )
                if st.button("➕ 選択した銘柄をウォッチリストに追加", key="screen_add_btn") and _add_codes:
                    _added = 0
                    for _sel in _add_codes:
                        _cd = _sel.split(" | ")[0]
                        _nm = _sel.split(" | ", 1)[1] if " | " in _sel else _cd
                        _tk = f"{_cd}.T"
                        if _tk not in watchlist:
                            watchlist[_tk] = _nm
                            _added += 1
                    if _added:
                        try:
                            save_watchlist(watchlist)
                            st.success(f"{_added}銘柄をウォッチリストに追加しました。")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失敗: {e}")
                    else:
                        st.info("選択した銘柄はすべて既にウォッチリストにあります。")
            st.caption("清原NC比率=ネットキャッシュ÷時価総額。PSR=時価総額÷売上高。"
                       "数値はEDINET DBの最新有報ベース（毎日8時更新）。データ出典: EDINET DB (https://edinetdb.jp)")


# =====================================================================
# 四季報オンライン サマリー（ヘッダー直下に常時表示）
# 一度取得すれば7日キャッシュから自動表示。他セクションからも参照できるよう
# session_state に保持する。
# =====================================================================
try:
    from shikiho_scraper import load_cached_shikiho, fetch_shikiho_subprocess as _fetch_shikiho
except Exception:
    load_cached_shikiho = None
    _fetch_shikiho = None

_shikiho_data = None
if load_cached_shikiho is not None:
    _shikiho_data = load_cached_shikiho(selected_ticker)  # 期限無視で読む（基本毎日変わらない情報）

# 銘柄追加時の自動取得: キャッシュが無い銘柄を初めて開いたら自動で1回だけ取得を試みる。
# （Streamlitの再描画で何度も走らないよう session_state でガード）
_sk_auto_tried = st.session_state.setdefault("shikiho_auto_tried", set())
if (_shikiho_data is None and _fetch_shikiho is not None
        and selected_ticker not in _sk_auto_tried
        and os.environ.get("SHIKIHO_EMAIL") and os.environ.get("SHIKIHO_PASSWORD")):
    _sk_auto_tried.add(selected_ticker)
    with st.spinner("📘 四季報を自動取得中…（この銘柄は初回のみ・30秒前後）"):
        try:
            _auto_got = _fetch_shikiho(selected_ticker, force=False, cache_days=90, timeout=150)
            if _auto_got and not _auto_got.get("error"):
                _shikiho_data = _auto_got
        except Exception:
            pass

with st.container():
    _sk_head1, _sk_head2 = st.columns([5, 1])
    with _sk_head2:
        _sk_btn = st.button("📘 四季報更新", key="shikiho_refresh_top",
                            help="会社四季報オンラインから最新を取得（要ログイン・初回10〜30秒）")
    if _sk_btn and _fetch_shikiho is not None:
        with st.spinner("四季報オンラインから取得中…（別プロセスでブラウザ起動・初回30秒前後）"):
            try:
                _got = _fetch_shikiho(selected_ticker, force=True, timeout=150)  # 別プロセスで強制取得
                if not _got.get("error"):
                    _shikiho_data = _got
                else:
                    st.warning(f"⚠️ 四季報取得に失敗: {_got['error']}")
            except Exception as e:
                st.warning(f"⚠️ 四季報取得エラー: {e}")

    # session_state に保持（PER推移チャート等から参照）
    st.session_state["shikiho_data"] = _shikiho_data

    if _shikiho_data:
        _sc = _shikiho_data.get("shikiho_score")
        _items = _shikiho_data.get("score_items") or {}
        _ph_hi = _shikiho_data.get("per_high_avg")
        _ph_lo = _shikiho_data.get("per_low_avg")
        _comments = _shikiho_data.get("comments") or []
        _age = _shikiho_data.get("cache_age_days")

        _score_color = "#16a34a" if (_sc or 0) >= 4 else ("#ffa726" if (_sc or 0) == 3 else "#90a4ae")
        _stars = ("★" * int(_sc) + "☆" * (5 - int(_sc))) if _sc else "—"
        _items_order = ["成長性", "規模", "収益性", "割安度", "安全性", "値上がり"]
        _items_html = " ／ ".join(f"{k} {_items[k]}" for k in _items_order if k in _items)
        _comment_html = ""
        for _c in _comments[:2]:
            _comment_html += (
                f'<div style="margin-top:6px; font-size:0.84rem;">'
                f'<b style="color:#c0392b;">【{_c.get("headline","")}】</b> {_c.get("body","")}</div>'
            )
        _per_badge = ""
        if _ph_hi and _ph_lo:
            _per_badge = (f'<span style="background:#eef2ff; color:#3b5bdb; padding:2px 8px; '
                          f'border-radius:4px; font-size:0.78rem; margin-left:8px;">'
                          f'実績PER 高値平均{_ph_hi:.1f}倍 / 安値平均{_ph_lo:.1f}倍</span>')
        # 会社予想PER（四季報）バッジ
        _fper_badge = ""
        _fp_list = _shikiho_data.get("forecast_per") or []
        if _fp_list:
            _fp_txt = " / ".join(f'{x.get("period","")} {x.get("per"):.2f}倍' for x in _fp_list[:2] if x.get("per"))
            if _fp_txt:
                _fper_badge = (f'<span style="background:#e8f5e9; color:#2e7d32; padding:2px 8px; '
                               f'border-radius:4px; font-size:0.78rem; margin-left:8px;">'
                               f'予想PER {_fp_txt}</span>')
        _age_txt = f"（{_age}日前取得）" if _age is not None else ""

        # --- 現在PERを算出し、四季報の高値平均/安値平均で割安/割高を判定 ---
        _cur_per_hdr = info.get("trailingPE")
        if not _cur_per_hdr or _cur_per_hdr <= 0:
            _px = info.get("currentPrice") or info.get("regularMarketPrice")
            _eps_ttm = info.get("trailingEps")
            if _px and _eps_ttm and _eps_ttm > 0:
                _cur_per_hdr = _px / _eps_ttm

        _verdict_html = ""
        if _ph_hi and _ph_lo and _ph_hi > _ph_lo and _cur_per_hdr and _cur_per_hdr > 0:
            _pos_hdr = (_cur_per_hdr - _ph_lo) / (_ph_hi - _ph_lo) * 100
            if _cur_per_hdr >= _ph_hi:
                _vc, _vl = "#ef5350", "割高圏"
            elif _pos_hdr >= 66:
                _vc, _vl = "#ff7043", "やや割高（高値圏）"
            elif _pos_hdr >= 34:
                _vc, _vl = "#ffa726", "中間圏"
            elif _cur_per_hdr > _ph_lo:
                _vc, _vl = "#26a69a", "やや割安（安値圏）"
            else:
                _vc, _vl = "#16a34a", "割安圏"
            _verdict_html = (
                f'<div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-top:8px;">'
                f'<span style="background:{_vc}; color:#fff; padding:4px 14px; border-radius:20px; '
                f'font-size:1.05rem; font-weight:bold;">💰 {_vl}</span>'
                f'<span style="font-size:0.9rem; color:#333;">現在PER <b>{_cur_per_hdr:.1f}倍</b>'
                f'（安値平均{_ph_lo:.1f}倍〜高値平均{_ph_hi:.1f}倍、レンジ位置 {max(0,min(100,_pos_hdr)):.0f}%）</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:linear-gradient(90deg,#fff8e1,#ffffff); '
            f'border:1px solid #f3d27a; border-left:6px solid #f6a623; border-radius:10px; '
            f'padding:12px 16px; margin:4px 0 2px 0;">'
            f'{_verdict_html}'
            f'<div style="display:flex; align-items:center; flex-wrap:wrap; gap:6px; margin-top:{"8px" if _verdict_html else "0"};">'
            f'<span style="font-size:0.8rem; color:#888; font-weight:bold;">📘 会社四季報</span>'
            f'<span style="font-size:1.3rem; font-weight:bold; color:{_score_color};">四季報スコア {_stars} {_sc if _sc is not None else "—"}</span>'
            f'{_fper_badge}'
            f'{_per_badge}'
            f'<span style="font-size:0.78rem; color:#999;">{_age_txt}</span>'
            f'<a href="https://shikiho.toyokeizai.net/stocks/{selected_ticker.replace(".T","")}" '
            f'target="_blank" style="font-size:0.78rem; color:#1565c0; text-decoration:none; '
            f'margin-left:8px;">四季報オンラインで開く ↗</a>'
            f'</div>'
            f'<div style="font-size:0.8rem; color:#555; margin-top:4px;">{_items_html}</div>'
            f'{_comment_html}'
            f'</div>', unsafe_allow_html=True)
    else:
        st.info("📘 会社四季報の情報は未取得です。右上の「📘 四季報更新」で取得できます（7日キャッシュ）。")

# =====================================================================
# ソフトバンクグループ（9984）専用: NAVディスカウント
#   NAV / 保有株式 / 純負債 / 1株NAV は公式SOTPページから自動取得
#   （https://group.softbank/ir/stock/sotp、1日キャッシュ）。
#   ディスカウント率 = (1株NAV − 株価) / 1株NAV × 100（1株ベースを優先）
#   取得失敗時は手入力フォールバック。
# =====================================================================
if selected_ticker.replace(".T", "").upper() in ("9984", "9984A"):
    st.markdown("**🏦 NAVディスカウント（ソフトバンクグループ専用）**")

    import sbg_nav_scraper
    _reload_nav = st.button("🔄 公式NAVを再取得", key="sbg_nav_reload",
                            help="SBG公式SOTPページから最新NAVを取得（通常は1日キャッシュ）")
    with st.spinner("公式NAVページを取得中..."):
        _navd = sbg_nav_scraper.fetch_sbg_nav(force=_reload_nav)

    _auto_ok = (not _navd.get("error")) and _navd.get("nav") and _navd.get("nav_per_share")

    if _auto_ok:
        _nav = float(_navd["nav"])              # NAV（兆円）
        _holdings = float(_navd.get("holdings") or 0.0)
        _net_debt = float(_navd.get("net_debt") or 0.0)
        _nav_ps = float(_navd["nav_per_share"])  # 1株NAV（円）
        _px_now = (info.get("currentPrice") or info.get("regularMarketPrice")
                   or _navd.get("price"))
        _as_of = _navd.get("as_of") or ""
        _src_note = "🟢 公式SOTPページ自動取得" + (f"（{_as_of}時点）" if _as_of else "")
        if _navd.get("from_cache"):
            _src_note += "・キャッシュ"
        st.caption(_src_note + f"／ 出典: {_navd.get('source')}")

        _mcap = info.get("marketCap")
        _mcap_cho = (_mcap / 1e12) if _mcap else None

        # 1株NAV基準のディスカウント率（最も確実）
        _disc = None
        if _px_now and _nav_ps > 0:
            _disc = (_nav_ps - float(_px_now)) / _nav_ps * 100
        # フォールバック: 時価総額ベース
        if _disc is None and _mcap_cho and _nav > 0:
            _disc = (_nav - _mcap_cho) / _nav * 100

        if _disc is not None:
            if _disc >= 50:
                _dc, _dl = "#16a34a", "大幅ディスカウント（割安）"
            elif _disc >= 30:
                _dc, _dl = "#26a69a", "ディスカウント（やや割安）"
            elif _disc >= 10:
                _dc, _dl = "#ffa726", "小幅ディスカウント"
            elif _disc >= -10:
                _dc, _dl = "#90a4ae", "ほぼNAV並み"
            else:
                _dc, _dl = "#ef5350", "プレミアム（割高）"

            _ps_note = f"1株NAV {_nav_ps:,.0f}円"
            if _px_now:
                _ps_note += f" vs 株価 {float(_px_now):,.0f}円"
            _mcap_note = f"／ 時価総額 {_mcap_cho:.2f}兆円" if _mcap_cho else ""
            st.markdown(
                f'<div style="background:#f8f9fa; border-radius:12px; padding:16px 20px; '
                f'border-left:6px solid {_dc}; margin-top:6px;">'
                f'<div style="font-size:0.8rem; color:#666;">NAVディスカウント率（1株NAV基準）</div>'
                f'<div style="font-size:1.8rem; font-weight:bold; color:{_dc};">{_disc:+.1f}%</div>'
                f'<div style="font-size:0.95rem; color:{_dc}; font-weight:bold;">{_dl}</div>'
                f'<div style="font-size:0.82rem; color:#555; margin-top:6px;">'
                f'NAV {_nav:.2f}兆円（保有株式 {_holdings:.2f} − 純負債 {_net_debt:.2f}）'
                f' {_mcap_note}<br>{_ps_note}</div>'
                f'</div>', unsafe_allow_html=True)
            st.caption("ディスカウント率 = (1株NAV − 株価) ÷ 1株NAV ×100。プラスが大きいほどNAV対比で割安。")

            # --- 過去NAVディスカウント推移 × 株価（二軸）＋ 割安/割高判定 ---
            _ndh_c1, _ndh_c2 = st.columns([3, 1])
            with _ndh_c1:
                st.markdown("**📉 NAVディスカウント率 × 株価の推移**")
                st.caption("ディスカウント率（左軸）と株価（右軸）。1株NAVは最新の公式値で固定した近似です。"
                           "過去にさかのぼるほど実際のNAVとの乖離が大きくなるため、直近1年以内に限定しています。"
                           "右軸は株価・ARM・OpenAIを「期間初日=100」の指数に統一し、変化率を同じ高さで比較できます。"
                           "凡例の「ARM株価」「OpenAI推定時価」はクリックで表示/非表示でき、実数値はホバーで確認できます。")
            with _ndh_c2:
                _ndh_period = st.radio(
                    "期間", ["3ヶ月", "6ヶ月", "1年"], index=2, horizontal=True,
                    key="nav_disc_hist_period", label_visibility="collapsed"
                )

            with st.spinner("NAVディスカウント推移を計算中..."):
                _ndh = fetch_nav_discount_history(selected_ticker, _nav_ps, years=1)

            if _ndh and _ndh.get("df") is not None and not _ndh["df"].empty:
                import pandas as _pd_nd
                _ndf_all = _ndh["df"]
                _ymap_nd = {"3ヶ月": 90, "6ヶ月": 180, "1年": 365}
                _dnd = _ymap_nd[_ndh_period]
                _cut_nd = _ndf_all.index.max() - _pd_nd.Timedelta(days=_dnd)
                _ndf = _ndf_all[_ndf_all.index >= _cut_nd]
                if _ndf.empty:
                    _ndf = _ndf_all

                _d = _ndf["disc"]
                _d_max, _d_min, _d_avg = float(_d.max()), float(_d.min()), float(_d.mean())

                # 右軸は「指数（期間初日=100）」に統一し、株価・ARM・OpenAIを同じ高さ帯で比較
                def _to_index100(_s):
                    _v = _s.dropna()
                    if _v.empty:
                        return None
                    _base = float(_v.iloc[0])
                    if _base == 0:
                        return None
                    return _s / _base * 100.0

                fig_nd = make_subplots(specs=[[{"secondary_y": True}]])

                # 株価（指数・右軸）
                _px_idx = _to_index100(_ndf["close"])
                fig_nd.add_trace(
                    go.Scatter(
                        x=_ndf.index, y=_px_idx.values, name="株価（指数・右軸）",
                        line=dict(color="#6ea8e6", width=1.6),
                        customdata=_ndf["close"].values,
                        hovertemplate="株価 %{customdata:,.0f}円（指数%{y:.0f}）<extra></extra>",
                    ),
                    secondary_y=True,
                )
                # NAVディスカウント率（左軸・%）
                fig_nd.add_trace(
                    go.Scatter(
                        x=_ndf.index, y=_ndf["disc"], name="NAVディスカウント率（左軸）",
                        line=dict(color="#8e44ad", width=2),
                    ),
                    secondary_y=False,
                )

                # ARM株価（指数・右軸、デフォルト非表示）
                _arm = fetch_arm_price_history(years=1)
                if _arm is not None and not _arm.empty:
                    _arm_r = _arm.reindex(
                        _arm.index.union(_ndf.index)
                    ).interpolate().reindex(_ndf.index)
                    _arm_idx = _to_index100(_arm_r)
                    if _arm_idx is not None:
                        fig_nd.add_trace(
                            go.Scatter(
                                x=_ndf.index, y=_arm_idx.values, name="ARM株価（指数・参考）",
                                line=dict(color="#e67e22", width=1.6, dash="dot"),
                                visible="legendonly",
                                customdata=_arm_r.values,
                                hovertemplate="ARM %{customdata:.1f}USD（指数%{y:.0f}）<extra></extra>",
                            ),
                            secondary_y=True,
                        )

                # OpenAI推定時価（指数・右軸、階段状、デフォルト非表示）
                _oa_ser = build_openai_valuation_series(_ndf.index)
                if _oa_ser is not None and _oa_ser.notna().any():
                    _oa_idx = _to_index100(_oa_ser)
                    if _oa_idx is not None:
                        fig_nd.add_trace(
                            go.Scatter(
                                x=_ndf.index, y=_oa_idx.values, name="OpenAI推定時価（指数・参考）",
                                line=dict(color="#16a085", width=1.8, shape="hv"),
                                visible="legendonly",
                                customdata=_oa_ser.values,
                                hovertemplate="OpenAI %{customdata:.3f}兆ドル（指数%{y:.0f}）<extra></extra>",
                            ),
                            secondary_y=True,
                        )

                for _val, _lab in [(_d_max, "最大（割安側）"), (_d_avg, "平均"), (_d_min, "最小（割高側）")]:
                    fig_nd.add_hline(
                        y=_val, line_dash="dash", line_color="#555", line_width=1,
                        annotation_text=f"{_lab} {_val:+.1f}%", annotation_position="right",
                        annotation_font_size=10, secondary_y=False,
                    )
                fig_nd.update_yaxes(title_text="ディスカウント率(%)", secondary_y=False,
                                    ticksuffix="%", color="#8e44ad")
                fig_nd.update_yaxes(title_text="指数（期間初日=100）", secondary_y=True, color="#6ea8e6")
                fig_nd.update_layout(
                    height=380, template="plotly_white",
                    margin=dict(l=10, r=70, t=20, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_nd, use_container_width=True)

                # 過去レンジ内での現在位置 → 割安/割高判定
                # 最小(=割高側)=0%, 最大(=割安側)=100% として現在ディスカウントの相対位置
                _cur_d = float(_ndf["disc"].iloc[-1])
                if _d_max > _d_min:
                    _pos_nd = (_cur_d - _d_min) / (_d_max - _d_min) * 100
                else:
                    _pos_nd = 50.0
                if _pos_nd >= 80:
                    _ndc, _ndl = "#16a34a", "割安圏（ディスカウント過去最大水準）"
                elif _pos_nd >= 60:
                    _ndc, _ndl = "#26a69a", "やや割安（ディスカウント拡大）"
                elif _pos_nd >= 40:
                    _ndc, _ndl = "#ffa726", "中間圏"
                elif _pos_nd >= 20:
                    _ndc, _ndl = "#ff7043", "やや割高（ディスカウント縮小）"
                else:
                    _ndc, _ndl = "#ef5350", "割高圏（ディスカウント過去最小水準）"
                st.markdown(
                    f'<div style="background:#f8f9fa; border-radius:10px; padding:14px 18px; '
                    f'border-left:6px solid {_ndc}; margin-top:8px;">'
                    f'<div style="font-size:0.8rem; color:#666;">過去ディスカウントレンジでの位置づけ'
                    f'（{_ndh_period}: 最小 {_d_min:+.1f}% 〜 最大 {_d_max:+.1f}%）</div>'
                    f'<div style="font-size:1.3rem; font-weight:bold; color:{_ndc};">'
                    f'現在 {_cur_d:+.1f}% → {_ndl}</div>'
                    f'<div style="font-size:0.82rem; color:#555; margin-top:4px;">'
                    f'レンジ内ポジション: {max(0,min(100,_pos_nd)):.0f}%'
                    f'（0%=ディスカウント最小/割高, 100%=ディスカウント最大/割安）／ 平均 {_d_avg:+.1f}%</div>'
                    f'</div>', unsafe_allow_html=True)
                st.caption(
                    f"更新日: {_ndf.index.max().strftime('%Y-%m-%d')} ／ "
                    f"ディスカウントが大きい（=NAV対比で株価が安い）ほど割安。"
                    "1株NAVは最新の公式値で固定した近似のため、直近1年以内に限定しています。"
                    "過去の実際のNAVはARM上場（2023年9月）等で大きく変動しており、長期の絶対水準比較には使えません。"
                )
            else:
                st.info("NAVディスカウント推移を計算できません（株価データ不足）。")
        else:
            st.info("株価が取得できずディスカウント率を計算できません。")
    else:
        # ---- フォールバック: 手入力 ----
        _err = _navd.get("error")
        st.warning(f"⚠️ 公式NAV自動取得に失敗（{_err or '不明'}）。手入力モードに切替えます。")
        st.caption("保有株式価値・純有利子負債は会社IR（決算説明資料の「NAV」ページ）から入力。単位は兆円。")

        _nav_state = st.session_state.setdefault("sbg_nav", {
            "holdings": 20.0, "net_debt": 2.0,
            "openai_val_usd": 1.4, "openai_stake": 7.0, "usdjpy": 150.0,
        })
        _nav_c1, _nav_c2, _nav_c3 = st.columns(3)
        with _nav_c1:
            _holdings_other = st.number_input("OpenAI以外の保有価値（兆円）", min_value=0.0, step=0.1,
                                              value=float(_nav_state["holdings"]), key="sbg_holdings")
        with _nav_c2:
            _net_debt = st.number_input("調整後純有利子負債（兆円）", step=0.1,
                                        value=float(_nav_state["net_debt"]), key="sbg_net_debt")
        with _nav_c3:
            _mcap = info.get("marketCap")
            _mcap_cho = (_mcap / 1e12) if _mcap else None
            st.metric("時価総額（兆円）", f"{_mcap_cho:.2f}" if _mcap_cho else "N/A")

        _oa1, _oa2, _oa3 = st.columns(3)
        with _oa1:
            _openai_val_usd = st.number_input("OpenAI推定時価総額（兆ドル）", min_value=0.0, step=0.1,
                                              value=float(_nav_state["openai_val_usd"]), key="sbg_openai_val")
        with _oa2:
            _openai_stake = st.number_input("SBGのOpenAI出資比率（%）", min_value=0.0, max_value=100.0, step=0.5,
                                            value=float(_nav_state["openai_stake"]), key="sbg_openai_stake")
        with _oa3:
            _usdjpy = st.number_input("ドル円", min_value=1.0, step=1.0,
                                      value=float(_nav_state["usdjpy"]), key="sbg_usdjpy")

        _openai_value_cho = _openai_val_usd * (_openai_stake / 100.0) * _usdjpy
        _nav_state.update({
            "holdings": _holdings_other, "net_debt": _net_debt,
            "openai_val_usd": _openai_val_usd, "openai_stake": _openai_stake, "usdjpy": _usdjpy,
        })
        _holdings = _holdings_other + _openai_value_cho
        st.caption(f"💡 OpenAI持分価値 ≈ {_openai_value_cho:.2f}兆円／ 保有価値合計 {_holdings:.2f}兆円")

        _nav = _holdings - _net_debt
        if _nav and _mcap_cho and _nav > 0:
            _disc = (_nav - _mcap_cho) / _nav * 100
            if _disc >= 50:
                _dc, _dl = "#16a34a", "大幅ディスカウント（割安）"
            elif _disc >= 30:
                _dc, _dl = "#26a69a", "ディスカウント（やや割安）"
            elif _disc >= 10:
                _dc, _dl = "#ffa726", "小幅ディスカウント"
            elif _disc >= -10:
                _dc, _dl = "#90a4ae", "ほぼNAV並み"
            else:
                _dc, _dl = "#ef5350", "プレミアム（割高）"
            _per_share_note = ""
            _shares = info.get("sharesOutstanding")
            if _shares:
                _nav_ps = _nav * 1e12 / _shares
                _px_now = info.get("currentPrice") or info.get("regularMarketPrice")
                _per_share_note = f"／ 1株NAV ≈ {_nav_ps:,.0f}円（現在株価 {_px_now:,.0f}円）" if _px_now else f"／ 1株NAV ≈ {_nav_ps:,.0f}円"
            st.markdown(
                f'<div style="background:#f8f9fa; border-radius:12px; padding:16px 20px; '
                f'border-left:6px solid {_dc}; margin-top:6px;">'
                f'<div style="font-size:0.8rem; color:#666;">NAVディスカウント率</div>'
                f'<div style="font-size:1.8rem; font-weight:bold; color:{_dc};">{_disc:+.1f}%</div>'
                f'<div style="font-size:0.95rem; color:{_dc}; font-weight:bold;">{_dl}</div>'
                f'<div style="font-size:0.82rem; color:#555; margin-top:6px;">'
                f'NAV {_nav:.2f}兆円（保有 {_holdings:.2f} − 純負債 {_net_debt:.2f}）'
                f' vs 時価総額 {_mcap_cho:.2f}兆円 {_per_share_note}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.info("NAVディスカウントを計算できません（保有株式価値・時価総額を確認してください）。")


# =====================================================================
# 共通変数
# =====================================================================
# EDINET DB データ取得（スコア計算用）
with st.spinner("EDINET DB データ取得中..."):
    edb_full = fetch_edinetdb_full(selected_ticker)

scores = calc_total_score(current_data, edb_data=edb_full)

# --- EDINET DBから正確な値を取得（yfinanceフォールバック） ---
# earnings（決算短信）からの値
_edb_latest_q = None
_edb_latest_annual = None
_edb_latest_ratio = None
if edb_full and edb_full.get("earnings"):
    _ed = edb_full["earnings"]
    _inner = _ed.get("data", {}) if isinstance(_ed, dict) else {}
    _el = _inner.get("earnings", []) if isinstance(_inner, dict) else (_inner if isinstance(_inner, list) else [])
    for _e in _el:
        if _e.get("quarter") and _e["quarter"] < 4 and _edb_latest_q is None:
            _edb_latest_q = _e
        if _e.get("quarter") == 4 and _edb_latest_annual is None:
            _edb_latest_annual = _e
if edb_full and edb_full.get("ratios"):
    _rd = edb_full["ratios"]
    _rl = _rd.get("data", []) if isinstance(_rd, dict) else (_rd if isinstance(_rd, list) else [])
    if _rl:
        _sorted_r = sorted(_rl, key=lambda x: x.get("fiscal_year", 0), reverse=True)
        _edb_latest_ratio = _sorted_r[0]

# 会社予想EPS（EDINET DB優先）
_forecast_eps = None
if _edb_latest_q and _edb_latest_q.get("forecast_eps"):
    _forecast_eps = _edb_latest_q["forecast_eps"]

# 株価
_current_price = info.get("currentPrice") or info.get("regularMarketPrice")

# 会社予想PER
# 優先順位: ①四季報オンラインの会社予想PER（今期=先頭） ②EDINET DB会社予想EPSから算出 ③yfinance
per_val = None
per_source = None
_sk_fp = (_shikiho_data or {}).get("forecast_per") if "_shikiho_data" in dir() else None
if _sk_fp and len(_sk_fp) > 0 and _sk_fp[0].get("per"):
    per_val = round(float(_sk_fp[0]["per"]), 1)
    per_source = f"四季報（{_sk_fp[0].get('period','今期')}）"
elif _forecast_eps and _current_price and _forecast_eps > 0:
    per_val = round(_current_price / _forecast_eps, 1)
    per_source = "EDINET会社予想EPS"
else:
    per_val = info.get("forwardPE") or info.get("trailingPE")
    per_source = "yfinance"

# BPS（EDINET DB優先: 最新Q4のbps、分割調整済み）
_bps = None
if _edb_latest_annual and _edb_latest_annual.get("bps"):
    _bps = _edb_latest_annual["bps"]
elif _edb_latest_ratio and _edb_latest_ratio.get("bps"):
    _bps = _edb_latest_ratio["bps"]
else:
    _bps = info.get("bookValue")

# PBR = 株価 / BPS
if _bps and _current_price and _bps > 0:
    pbr_val = round(_current_price / _bps, 1)
else:
    pbr_val = info.get("priceToBook")

# ROE（EDINET DB: roe_official）
if _edb_latest_ratio and _edb_latest_ratio.get("roe_official"):
    roe_val = _edb_latest_ratio["roe_official"]  # 小数 0.416
elif info.get("returnOnEquity"):
    roe_val = info["returnOnEquity"]
else:
    roe_val = None

# 自己資本比率（EDINET DB: 最新四半期のequity_ratio）
if _edb_latest_q and _edb_latest_q.get("equity_ratio"):
    equity_ratio = _edb_latest_q["equity_ratio"]  # already %
elif _edb_latest_annual and _edb_latest_annual.get("equity_ratio"):
    equity_ratio = _edb_latest_annual["equity_ratio"]
else:
    # yfinanceフォールバック
    balance = current_data.get("balance")
    equity_ratio = None
    if balance is not None and not balance.empty:
        try:
            latest_b = balance.iloc[:, 0]
            ta = latest_b.get("Total Assets")
            te = latest_b.get("Stockholders Equity") or latest_b.get("Total Equity Gross Minority Interest")
            if ta and te and ta > 0:
                equity_ratio = te / ta * 100
        except Exception:
            pass

# 基準理論株価 = forecast_eps × 30
if _forecast_eps and _forecast_eps > 0:
    theo = {
        "pessimistic": round(_forecast_eps * 24, 0),
        "base": round(_forecast_eps * 30, 0),
        "optimistic": round(_forecast_eps * 36, 0),
        "eps": _forecast_eps,
    }
else:
    theo = calc_theoretical_price(info)

# その他
forward_eps = _forecast_eps or info.get("forwardEps")
trailing_eps = (_edb_latest_annual.get("eps") if _edb_latest_annual else None) or info.get("trailingEps")
rev_g = info.get("revenueGrowth")
op_margin = info.get("operatingMargins")
eq_growth = info.get("earningsQuarterlyGrowth")
peg_val, peg_source = calc_peg_ratio(info)

# =====================================================================
# PAGE 01: 決算・業績
# =====================================================================
st.markdown('<div style="font-size:0.75rem; color:#ef5350; font-weight:bold; letter-spacing:1px;">PAGE 01</div>', unsafe_allow_html=True)
st.subheader("決算・業績")

# --- KPI行 ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_s = scores["total"]
    st.metric("総合スコア / 10", f"{total_s}", score_label(total_s))
with col2:
    # 売上高前年比（EDINET DB優先）
    _rev_chg_display = None
    if _edb_latest_q and _edb_latest_q.get("revenue_change") is not None:
        _rev_chg_display = _edb_latest_q["revenue_change"]
    elif rev_g is not None:
        _rev_chg_display = rev_g * 100
    if _rev_chg_display is not None:
        st.metric("Q1 売上高 前年比", f"+{_rev_chg_display:.1f}%" if _rev_chg_display >= 0 else f"{_rev_chg_display:.1f}%")
    else:
        st.metric("Q1 売上高 前年比", "N/A")
with col3:
    # 営業利益前年比（EDINET DB優先）
    _oi_chg_display = None
    if _edb_latest_q and _edb_latest_q.get("operating_income_change") is not None:
        _oi_chg_display = _edb_latest_q["operating_income_change"]
    elif eq_growth is not None:
        _oi_chg_display = eq_growth * 100
    if _oi_chg_display is not None:
        st.metric("Q1 営業利益 前年比", f"+{_oi_chg_display:.1f}%" if _oi_chg_display >= 0 else f"{_oi_chg_display:.1f}%")
    else:
        st.metric("Q1 営業利益 前年比", "N/A")
with col4:
    eps_display = forward_eps or trailing_eps
    st.metric("通期会社EPS", f"{eps_display:.2f}円" if eps_display else "N/A")
with col5:
    # 営業利益 通期計画進捗率 = Q1実績 / 通期予想 * 100
    _progress_annual = None
    _progress_h1 = None
    if _edb_latest_q:
        _q1_oi = _edb_latest_q.get("operating_income")
        _forecast_oi = _edb_latest_q.get("forecast_operating_income")
        if _q1_oi and _forecast_oi and _forecast_oi > 0:
            # earningsのOIは百万円単位
            _progress_annual = _q1_oi / _forecast_oi * 100
            _progress_h1 = _q1_oi / (_forecast_oi * 0.5) * 100  # H1目安=通期の50%
    if _progress_annual is not None:
        st.metric("営業利益 通期計画進捗率", f"{_progress_annual:.1f}%", f"上期計画進捗率 {_progress_h1:.1f}%")
    elif op_margin is not None:
        st.metric("営業利益率", f"{op_margin*100:.1f}%")
    else:
        st.metric("営業利益率", "N/A")

st.divider()

# --- Q1実績とH1会社計画テーブル + 読みどころ ---
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("**Q1実績とH1会社計画**")
    st.caption("金額：百万円")

    # EDINET DBからQ1データで構築
    if _edb_latest_q:
        _q = _edb_latest_q
        # Q1実績
        q1_rev = _q.get("revenue")  # 百万円
        q1_oi = _q.get("operating_income")  # 百万円
        q1_ni = _q.get("net_income")  # 百万円
        q1_eps = _q.get("eps")
        q1_rev_chg = _q.get("revenue_change")  # %
        q1_oi_chg = _q.get("operating_income_change")  # %
        q1_ni_chg = _q.get("net_income_change")  # %

        # 前年Q1を逆算
        prev_q1_rev = q1_rev / (1 + q1_rev_chg / 100) if q1_rev and q1_rev_chg and q1_rev_chg != -100 else None
        prev_q1_oi = q1_oi / (1 + q1_oi_chg / 100) if q1_oi and q1_oi_chg and q1_oi_chg != -100 else None
        prev_q1_ni = q1_ni / (1 + q1_ni_chg / 100) if q1_ni and q1_ni_chg and q1_ni_chg != -100 else None
        prev_q1_eps = q1_eps / (1 + q1_ni_chg / 100) if q1_eps and q1_ni_chg and q1_ni_chg != -100 else None

        # 営業利益率
        q1_oi_margin = (q1_oi / q1_rev * 100) if q1_rev and q1_oi and q1_rev > 0 else None
        prev_q1_oi_margin = (prev_q1_oi / prev_q1_rev * 100) if prev_q1_rev and prev_q1_oi and prev_q1_rev > 0 else None
        margin_diff = (q1_oi_margin - prev_q1_oi_margin) if q1_oi_margin is not None and prev_q1_oi_margin is not None else None

        # H1計画（forecast は通期予想。H1計画は別途取得不可のため通期の半分で近似）
        # ただしearningsに forecast_operating_income等があるのは通期予想
        forecast_rev = _q.get("forecast_revenue")  # 通期予想（百万円）
        forecast_oi = _q.get("forecast_operating_income")
        forecast_ni = _q.get("forecast_net_income")
        forecast_eps_val = _q.get("forecast_eps")

        # H1計画 = 通期予想の概ね半分（季節性未調整）
        # 但し、一部企業はH1予想を別途開示 → ここでは通期/2で近似
        h1_rev = forecast_rev * 0.5 if forecast_rev else None
        h1_oi = forecast_oi * 0.5 if forecast_oi else None
        h1_ni = forecast_ni * 0.5 if forecast_ni else None
        h1_eps = forecast_eps_val * 0.5 if forecast_eps_val else None

        # 進捗率 = Q1実績 / H1計画
        prog_rev = (q1_rev / h1_rev * 100) if q1_rev and h1_rev and h1_rev > 0 else None
        prog_oi = (q1_oi / h1_oi * 100) if q1_oi and h1_oi and h1_oi > 0 else None
        prog_ni = (q1_ni / h1_ni * 100) if q1_ni and h1_ni and h1_ni > 0 else None
        prog_eps = (q1_eps / h1_eps * 100) if q1_eps and h1_eps and h1_eps > 0 else None

        # H1計画の営業利益率
        h1_margin = (h1_oi / h1_rev * 100) if h1_rev and h1_oi and h1_rev > 0 else None
        h1_margin_diff = (q1_oi_margin - h1_margin) if q1_oi_margin is not None and h1_margin is not None else None

        # テーブル構築
        _fy_prev = int(_q.get("fiscal_year_end", "2027")[:4]) - 1
        rows = []
        rows.append({
            "指標": "売上高",
            f"FY{str(_fy_prev)[-2:]} Q1": f"{prev_q1_rev:,.0f}" if prev_q1_rev else "-",
            f"FY{str(_fy_prev+1)[-2:]} Q1": f"{q1_rev:,.0f}" if q1_rev else "-",
            "前年比": f"+{q1_rev_chg:.1f}%" if q1_rev_chg and q1_rev_chg >= 0 else (f"{q1_rev_chg:.1f}%" if q1_rev_chg else "-"),
            f"FY{str(_fy_prev+1)[-2:]} H1計画": f"{h1_rev:,.0f}" if h1_rev else "-",
            "上期計画進捗率": f"{prog_rev:.1f}%" if prog_rev else "-",
        })
        rows.append({
            "指標": "営業利益",
            f"FY{str(_fy_prev)[-2:]} Q1": f"{prev_q1_oi:,.0f}" if prev_q1_oi else "-",
            f"FY{str(_fy_prev+1)[-2:]} Q1": f"{q1_oi:,.0f}" if q1_oi else "-",
            "前年比": f"+{q1_oi_chg:.1f}%" if q1_oi_chg and q1_oi_chg >= 0 else (f"{q1_oi_chg:.1f}%" if q1_oi_chg else "-"),
            f"FY{str(_fy_prev+1)[-2:]} H1計画": f"{h1_oi:,.0f}" if h1_oi else "-",
            "上期計画進捗率": f"{prog_oi:.1f}%" if prog_oi else "-",
        })
        rows.append({
            "指標": "親会社株主帰属利益",
            f"FY{str(_fy_prev)[-2:]} Q1": f"{prev_q1_ni:,.0f}" if prev_q1_ni else "-",
            f"FY{str(_fy_prev+1)[-2:]} Q1": f"{q1_ni:,.0f}" if q1_ni else "-",
            "前年比": f"+{q1_ni_chg:.1f}%" if q1_ni_chg and q1_ni_chg >= 0 else (f"{q1_ni_chg:.1f}%" if q1_ni_chg else "-"),
            f"FY{str(_fy_prev+1)[-2:]} H1計画": f"{h1_ni:,.0f}" if h1_ni else "-",
            "上期計画進捗率": f"{prog_ni:.1f}%" if prog_ni else "-",
        })
        rows.append({
            "指標": "EPS",
            f"FY{str(_fy_prev)[-2:]} Q1": f"{prev_q1_eps:.2f}" if prev_q1_eps else "-",
            f"FY{str(_fy_prev+1)[-2:]} Q1": f"{q1_eps:.2f}" if q1_eps else "-",
            "前年比": f"+{q1_ni_chg:.1f}%" if q1_ni_chg and q1_ni_chg >= 0 else (f"{q1_ni_chg:.1f}%" if q1_ni_chg else "-"),
            f"FY{str(_fy_prev+1)[-2:]} H1計画": f"{h1_eps:.2f}" if h1_eps else "-",
            "上期計画進捗率": f"{prog_eps:.1f}%" if prog_eps else "-",
        })
        rows.append({
            "指標": "営業利益率",
            f"FY{str(_fy_prev)[-2:]} Q1": f"{prev_q1_oi_margin:.1f}%" if prev_q1_oi_margin else "-",
            f"FY{str(_fy_prev+1)[-2:]} Q1": f"{q1_oi_margin:.1f}%" if q1_oi_margin else "-",
            "前年比": f"{margin_diff:+.1f}pt" if margin_diff is not None else "-",
            f"FY{str(_fy_prev+1)[-2:]} H1計画": f"{h1_margin:.1f}%" if h1_margin else "-",
            "上期計画進捗率": f"{h1_margin_diff:+.1f}pt" if h1_margin_diff is not None else "-",
        })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        # yfinanceフォールバック
        revenue = info.get("totalRevenue")
        op_income = info.get("operatingIncome") or info.get("ebitda")
        net_income = info.get("netIncomeToCommon")
        table_df = pd.DataFrame({
            "指標": ["売上高", "営業利益", "純利益", "EPS"],
            "実績値": [
                f"{revenue/1e6:,.0f}百万円" if revenue else "N/A",
                f"{op_income/1e6:,.0f}百万円" if op_income else "N/A",
                f"{net_income/1e6:,.0f}百万円" if net_income else "N/A",
                f"{trailing_eps:.2f}円" if trailing_eps else "N/A",
            ],
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

with col_right:
    st.markdown("**今回の読みどころ**")
    insights = generate_insights(current_data)
    if insights["positive"]:
        pos_text = "<br>".join(insights["positive"])
        st.markdown(f'<div class="insight-card insight-positive"><div class="insight-title">POSITIVE</div><div class="insight-text">{pos_text}</div></div>', unsafe_allow_html=True)
    if insights["watch"]:
        w_text = "<br>".join(insights["watch"])
        st.markdown(f'<div class="insight-card insight-watch"><div class="insight-title">WATCH</div><div class="insight-text">{w_text}</div></div>', unsafe_allow_html=True)
    if insights["outlook"]:
        o_text = "<br>".join(insights["outlook"])
        st.markdown(f'<div class="insight-card insight-outlook"><div class="insight-title">COMPANY OUTLOOK</div><div class="insight-text">{o_text}</div></div>', unsafe_allow_html=True)

st.divider()

# =====================================================================
# 成長セグメント（EDINET DB API）
# =====================================================================
st.markdown("**成長セグメント**")
st.caption("売上高が前年比5%以上、かつ営業利益が増加したセグメントを表示")

income_quarterly = current_data.get("income_quarterly")
income_annual = current_data.get("income_annual")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_edinetdb_segments(ticker_code):
    """EDINET DB APIからセグメント別業績を取得（ディスクキャッシュ付き）"""
    # ディスクキャッシュ確認
    cache_path = os.path.join(_EDB_DISK_CACHE_DIR, f"{ticker_code.replace('.T','')}_segments.json")
    if os.path.exists(cache_path):
        try:
            age = (time.time() - os.path.getmtime(cache_path)) / 86400
            if age < 7:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    if not _EDINETDB_API_KEY:
        return None
    code = ticker_code.replace(".T", "")
    headers = {"X-API-Key": _EDINETDB_API_KEY, "User-Agent": "Mozilla/5.0"}
    try:
        sr = _requests.get(f"https://edinetdb.jp/v1/search?q={code}", timeout=15, headers=headers)
        if sr.status_code == 429:
            # レート制限→期限切れキャッシュ
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        if sr.status_code != 200:
            return None
        search_data = sr.json()
        if not search_data:
            return None
        edinet_code = None
        if isinstance(search_data, dict):
            data_list = search_data.get("data") or search_data.get("results") or []
            if data_list:
                edinet_code = data_list[0].get("edinet_code") or data_list[0].get("edinetCode")
        elif isinstance(search_data, list) and len(search_data) > 0:
            edinet_code = search_data[0].get("edinet_code") or search_data[0].get("edinetCode")
        if not edinet_code:
            return None
        seg_resp = _requests.get(f"https://edinetdb.jp/v1/companies/{edinet_code}/segments", timeout=15, headers=headers)
        if seg_resp.status_code == 429:
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        if seg_resp.status_code != 200:
            return None
        result = seg_resp.json()
        # キャッシュ保存
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, default=str)
        except Exception:
            pass
        return result
    except Exception:
        return None


def build_segment_reason_comment(seg):
    """セグメントの増益/減益要因を一言で推定して返す。
    売上前年比・営業利益増減・営業利益率(今期/前期)から、
    「増収と採算改善が寄与」「増収も採算悪化が利益を圧迫」等のコメントを生成する。
    """
    rev_g = seg.get("売上高前年比")           # %
    oi_diff = seg.get("営業利益増減")          # 円
    rev_curr = seg.get("売上高(今期)")
    rev_prev = seg.get("売上高(前期)")
    oi_curr = seg.get("営業利益(今期)")
    oi_prev = seg.get("営業利益(前期)")

    # 営業利益率の今期/前期（前期はデータがあれば算出）
    m_curr = seg.get("営業利益率")
    if m_curr is None and rev_curr and oi_curr is not None and rev_curr != 0:
        m_curr = oi_curr / rev_curr
    m_prev = None
    if rev_prev and oi_prev is not None and rev_prev != 0:
        m_prev = oi_prev / rev_prev

    # 利益率の前年差（ポイント）
    margin_chg_pt = None
    if m_curr is not None and m_prev is not None:
        margin_chg_pt = (m_curr - m_prev) * 100

    # 判定できるだけの材料がない場合
    if rev_g is None and oi_diff is None:
        return None

    rev_up = rev_g is not None and rev_g > 0
    rev_down = rev_g is not None and rev_g < 0
    oi_up = oi_diff is not None and oi_diff > 0
    oi_down = oi_diff is not None and oi_diff < 0

    # 利益率の方向を言葉に
    if margin_chg_pt is not None and margin_chg_pt >= 0.5:
        margin_word = f"採算改善（利益率+{margin_chg_pt:.1f}pt）"
    elif margin_chg_pt is not None and margin_chg_pt <= -0.5:
        margin_word = f"採算悪化（利益率{margin_chg_pt:.1f}pt）"
    else:
        margin_word = "採算横ばい" if margin_chg_pt is not None else None

    # 増益パターン
    if oi_up:
        if rev_up and margin_chg_pt is not None and margin_chg_pt >= 0.5:
            return f"増収に加え{margin_word}が利益を押し上げ"
        if rev_up and margin_chg_pt is not None and margin_chg_pt <= -0.5:
            return f"増収効果が{margin_word}を吸収して増益"
        if rev_up:
            return "売上増（数量・単価）が利益を牽引"
        if rev_down:
            return f"減収も{margin_word or 'コスト削減'}で増益を確保"
        return "営業利益が増加"

    # 減益パターン
    if oi_down:
        if rev_up and margin_chg_pt is not None and margin_chg_pt <= -0.5:
            return f"増収も{margin_word}（コスト増）が利益を圧迫"
        if rev_up:
            return "増収も費用増が利益を上回り減益"
        if rev_down and margin_chg_pt is not None and margin_chg_pt <= -0.5:
            return f"減収と{margin_word}が重なり減益"
        if rev_down:
            return "減収（数量減）が利益を押し下げ"
        return "営業利益が減少"

    # 利益横ばい
    if rev_up:
        return "増収も利益はほぼ横ばい"
    if rev_down:
        return "減収も利益はほぼ横ばい"
    return None


# セグメントデータ取得
seg_data = fetch_edinetdb_segments(selected_ticker)

if seg_data and isinstance(seg_data, (list, dict)):
    # レスポンス形式: {"data": [...]}
    segments_list = seg_data.get("data") if isinstance(seg_data, dict) else seg_data

    if segments_list and len(segments_list) > 0:
        # fiscal_yearでグルーピング（2年分ある場合は自分で前年比計算）
        from collections import defaultdict
        _seg_by_fy = defaultdict(list)
        for seg in segments_list:
            fy = seg.get("fiscalYear") or seg.get("fiscal_year") or 0
            _seg_by_fy[fy].append(seg)

        # 最新年度と前年度を特定
        _fy_list = sorted(_seg_by_fy.keys(), reverse=True)
        _latest_fy = _fy_list[0] if _fy_list else None
        _prev_fy = _fy_list[1] if len(_fy_list) >= 2 else None

        # 前年度セグメントを名前でインデックス化
        _prev_seg_map = {}
        if _prev_fy:
            for s in _seg_by_fy[_prev_fy]:
                name = s.get("segmentName") or s.get("segment_name") or ""
                _prev_seg_map[name] = s

        growth_segs = []
        all_segs = []

        for seg in _seg_by_fy.get(_latest_fy, []):
            seg_name = seg.get("segmentName") or seg.get("segment_name") or "不明"
            revenue = seg.get("revenue") or 0
            oi = seg.get("operatingIncome") or seg.get("operating_income") or 0
            rev_yoy = seg.get("revenueYoy")
            oi_yoy = seg.get("oiYoy")
            oi_margin = seg.get("oiMargin")

            # 前年比計算: APIの値があればそれを使う、なければ前年度セグメントから計算
            revenue_prev = None
            oi_prev = None
            if rev_yoy is not None and rev_yoy != -1:
                revenue_prev = revenue / (1 + rev_yoy)
                rev_growth_pct = rev_yoy * 100
            elif seg_name in _prev_seg_map:
                revenue_prev = _prev_seg_map[seg_name].get("revenue") or 0
                rev_growth_pct = ((revenue - revenue_prev) / revenue_prev * 100) if revenue_prev and revenue_prev > 0 else None
            else:
                rev_growth_pct = None

            if oi_yoy is not None and oi_yoy != -1:
                oi_prev = oi / (1 + oi_yoy)
            elif seg_name in _prev_seg_map:
                oi_prev = _prev_seg_map[seg_name].get("operatingIncome") or _prev_seg_map[seg_name].get("operating_income") or 0
            
            oi_diff = (oi - oi_prev) if oi_prev is not None else None

            seg_record = {
                "セグメント": seg_name,
                "売上高(今期)": revenue,
                "売上高(前期)": revenue_prev,
                "売上高前年比": rev_growth_pct,
                "営業利益(今期)": oi,
                "営業利益(前期)": oi_prev,
                "営業利益増減": oi_diff,
                "営業利益率": oi_margin,
            }
            all_segs.append(seg_record)

            # 成長セグメント: 売上5%以上 & 営業利益増加
            if rev_growth_pct is not None and oi_diff is not None:
                if rev_growth_pct >= 5 and oi_diff > 0:
                    growth_segs.append(seg_record)

        # 成長セグメントをカード表示
        if growth_segs:
            st.caption(f"{len(growth_segs)}件を自動抽出")
            seg_cols = st.columns(min(len(growth_segs), 3))
            for i, seg in enumerate(growth_segs[:3]):
                with seg_cols[i]:
                    rev_g_pct = seg["売上高前年比"]
                    oi_diff_val = seg["営業利益増減"]
                    _reason = build_segment_reason_comment(seg)
                    _reason_html = (
                        f'<div style="margin-top:8px; font-size:0.78rem; color:#166534; '
                        f'background:#dcfce7; border-radius:6px; padding:6px 8px;">💬 {_reason}</div>'
                        if _reason else ""
                    )
                    st.markdown(
                        f'<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:16px;">'
                        f'<div style="font-size:0.7rem; color:#666;">セグメント</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{seg["セグメント"]}</div>'
                        f'<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                        f'<div><div style="font-size:0.7rem; color:#666;">売上高前年比</div>'
                        f'<div style="font-size:1.2rem; font-weight:bold; color:#16a34a;">+{rev_g_pct:.1f}%</div></div>'
                        f'<div><div style="font-size:0.7rem; color:#666;">営業利益増減</div>'
                        f'<div style="font-size:1rem; font-weight:bold; color:#16a34a;">+{oi_diff_val/1e6:,.0f}百万円</div></div>'
                        f'</div>'
                        f'<div style="margin-top:6px;"><span style="background:#dcfce7; color:#16a34a; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:bold;">増収増益</span></div>'
                        f'{_reason_html}'
                        f'</div>', unsafe_allow_html=True)
        elif all_segs:
            # 前年比が計算できない場合でもセグメント構成は表示
            st.info("セグメント前年比データなし（前年度セグメントが未取得）。現在のセグメント構成を表示します。")
            seg_cols = st.columns(min(len(all_segs), 3))
            for i, seg in enumerate(all_segs[:3]):
                with seg_cols[i]:
                    _s_rev = seg["売上高(今期)"]
                    _s_oi = seg["営業利益(今期)"]
                    _s_margin = seg.get("営業利益率")
                    st.markdown(
                        f'<div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:10px; padding:16px;">'
                        f'<div style="font-size:0.7rem; color:#666;">セグメント</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{seg["セグメント"]}</div>'
                        f'<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                        f'<div><div style="font-size:0.7rem; color:#666;">売上高</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{_s_rev/1e6:,.0f}百万円</div></div>'
                        f'<div><div style="font-size:0.7rem; color:#666;">営業利益</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{_s_oi/1e6:,.0f}百万円</div></div>'
                        f'</div>'
                        f'<div style="margin-top:6px; font-size:0.75rem; color:#666;">営業利益率: {_s_margin*100:.1f}%</div>'
                        f'</div>' if _s_margin else
                        f'<div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:10px; padding:16px;">'
                        f'<div style="font-size:0.7rem; color:#666;">セグメント</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{seg["セグメント"]}</div>'
                        f'<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                        f'<div><div style="font-size:0.7rem; color:#666;">売上高</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{_s_rev/1e6:,.0f}百万円</div></div>'
                        f'<div><div style="font-size:0.7rem; color:#666;">営業利益</div>'
                        f'<div style="font-size:1rem; font-weight:bold;">{_s_oi/1e6:,.0f}百万円</div></div>'
                        f'</div></div>', unsafe_allow_html=True)
        else:
            st.info("前年比5%以上かつ営業利益増のセグメントはありません")

        # 全セグメント詳細テーブル
        with st.expander(f"全{len(all_segs)}セグメントの実績を見る"):
            table_rows = []
            for seg in all_segs:
                rev_prev = seg["売上高(前期)"]
                rev_curr = seg["売上高(今期)"]
                oi_prev = seg["営業利益(前期)"]
                oi_curr = seg["営業利益(今期)"]
                rev_g_val = seg["売上高前年比"]
                oi_diff_val = seg["営業利益増減"]
                oi_margin_val = seg.get("営業利益率")

                # 判定
                if rev_g_val is not None and oi_diff_val is not None:
                    if rev_g_val >= 5 and oi_diff_val > 0:
                        judge = "増収増益"
                    elif rev_g_val >= 0 and oi_diff_val <= 0:
                        judge = "増収減益"
                    elif rev_g_val < 0 and oi_diff_val > 0:
                        judge = "減収増益"
                    else:
                        judge = "減収減益"
                else:
                    judge = "-"

                _seg_reason = build_segment_reason_comment(seg) or "-"

                table_rows.append({
                    "セグメント": seg["セグメント"],
                    "売上高(百万円)": f"{rev_curr/1e6:,.0f}" if rev_curr else "-",
                    "前年比": f"{rev_g_val:+.1f}%" if rev_g_val is not None else "-",
                    "営業利益(百万円)": f"{oi_curr/1e6:,.0f}" if oi_curr else "-",
                    "営業利益増減(百万円)": f"{oi_diff_val/1e6:+,.0f}" if oi_diff_val is not None else "-",
                    "営業利益率": f"{oi_margin_val*100:.1f}%" if oi_margin_val is not None else "-",
                    "判定": judge,
                    "増減益の要因": _seg_reason,
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("セグメントデータが空です")
else:
    # EDINET DB取得失敗時はyfinance四半期データでフォールバック
    if income_quarterly is not None and not income_quarterly.empty and len(income_quarterly.columns) >= 2:
        try:
            latest_q = income_quarterly.iloc[:, 0]
            prev_q = income_quarterly.iloc[:, 1]
            segment_items = []
            for item_name, display_name in [("Total Revenue", "売上高"), ("Operating Income", "営業利益"), ("Net Income", "純利益")]:
                if item_name in latest_q.index and item_name in prev_q.index:
                    curr_val = latest_q[item_name]
                    prev_val = prev_q[item_name]
                    if prev_val and prev_val != 0 and not pd.isna(curr_val) and not pd.isna(prev_val):
                        growth_pct = (curr_val - prev_val) / abs(prev_val) * 100
                        diff = curr_val - prev_val
                        if growth_pct > 5 and diff > 0:
                            segment_items.append({"項目": display_name, "前年比": f"+{growth_pct:.1f}%", "増減額": f"+{diff/1e6:,.0f}百万円"})
            if segment_items:
                for seg in segment_items:
                    st.markdown(f"- **{seg['項目']}**: {seg['前年比']} ({seg['増減額']})")
            else:
                st.info("成長セグメントデータなし（四半期比較データ不足）")
        except Exception:
            st.info("セグメント分析データを取得できませんでした")
    else:
        st.info("この銘柄はセグメント別データが未開示です（単一セグメント企業、または有報にセグメント情報なし）")

# --- 📄 決算PDF読込（セグメント構造化抽出） ---
st.divider()
with st.expander("📄 決算PDF読込（セグメント情報を構造化抽出）"):
    st.caption("決算短信PDF（TDnet版推奨）をアップロード → セグメント別業績を構造化。複数ファイル対応。")
    st.caption("⚠️ 画像型PDFはテキスト抽出不可。TDnetの決算短信PDFを使用してください。")

    # --- 🔄 TDnet 決算短信 自動取得 ---
    _auto_col1, _auto_col2 = st.columns([1, 2])
    with _auto_col1:
        _do_auto = st.button("🔄 決算短信を自動取得", key="tdnet_auto",
                             help="TDnetの直近約1か月の開示からこの銘柄の決算短信PDFを自動取得します")
    with _auto_col2:
        st.caption(f"対象: {company_name}（{selected_ticker.replace('.T','')}）／ kabutan（過去分可）→TDnetの順で取得")

    _pdf_results = []  # [(表示名, result_dict)]
    if _do_auto:
        from pdf_segment_parser import process_kessan_auto
        with st.spinner("決算短信を検索・取得中（kabutan → TDnet）..."):
            _tdnet_res = process_kessan_auto(
                selected_ticker, cache_dir=_EDB_DISK_CACHE_DIR, cache_days=7, tdnet_days_back=45
            )
        if _tdnet_res.get("error"):
            if _tdnet_res["error"] == "NOT_FOUND":
                st.warning("⚠️ 決算短信を自動取得できませんでした（kabutan・TDnetのいずれからも見つかりませんでした）。")
                st.caption(
                    "💡 上場から日が浅い、非上場化、コード不一致などの可能性があります。"
                    "手元に決算短信PDFがあれば、下のアップローダーから今すぐ読み込めます。"
                )
            else:
                st.warning(f"⚠️ 自動取得できませんでした: {_tdnet_res['error']}")
                st.caption("↓ 下のアップローダーから手動でPDFを読み込むこともできます。")
        else:
            _m = _tdnet_res.get("tdnet_meta") or {}
            _origin = _m.get("source", "tdnet")
            _origin_label = {"kabutan": "kabutan", "tdnet": "TDnet"}.get(_origin, _origin)
            _src = "キャッシュ" if _tdnet_res.get("from_cache") else _origin_label
            st.success(f"✅ {_src}から取得: {_m.get('title','決算短信')}（開示日 {_m.get('date','?')}）")
            st.caption(f"PDF: {_m.get('pdf_url','')}")
            _pdf_results.append((f"📄 {_m.get('title','決算短信')}", _tdnet_res))

    uploaded_pdfs = st.file_uploader(
        "PDFをドラッグ＆ドロップ（複数可）", type=["pdf"],
        key="pdf_upload", accept_multiple_files=True
    )
    if uploaded_pdfs:
        from pdf_segment_parser import process_pdf_full
        for pdf_file in uploaded_pdfs:
            _pdf_results.append((f"📄 {pdf_file.name}", process_pdf_full(pdf_file, filename=pdf_file.name)))

    if _pdf_results:
        for _disp_name, result in _pdf_results:
            st.markdown(f"#### {_disp_name}")
            if result["error"]:
                st.warning(f"⚠️ {result['error']}")
                continue

            structured = result["segments"]
            orders = result["orders"]
            kpis = result["kpis"]
            growth = result["growth_indicators"]

            # --- セグメント ---
            if structured:
                # 成長判定: 「増収率≥5% かつ 増益」 または 「増収率≥10%」
                def _is_growth(s):
                    rg = s.get("revenue_growth_rate")
                    oc = s.get("operating_profit_change")
                    if rg is None:
                        return False
                    if rg >= 10:
                        return True
                    return rg >= 5 and oc is not None and oc > 0
                growth_segs = [s for s in structured if _is_growth(s)]
                st.markdown(f"**成長セグメント** — {len(growth_segs)}件を自動抽出")
                if growth_segs:
                    g_cols = st.columns(min(len(growth_segs), 3))
                    for i, seg in enumerate(growth_segs[:3]):
                        with g_cols[i]:
                            _qc = seg.get("qual_comment")
                            _qc_html = (
                                f'<div style="margin-top:8px; font-size:0.78rem; color:#166534; '
                                f'background:#dcfce7; border-radius:6px; padding:6px 8px;">💬 {_qc}</div>'
                                if _qc else ""
                            )
                            st.markdown(
                                f'<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:16px;">'
                                f'<div style="font-size:0.7rem; color:#666;">セグメント</div>'
                                f'<div style="font-size:1rem; font-weight:bold;">{seg["segment_name"]}</div>'
                                f'<div style="display:flex; justify-content:space-between; margin-top:8px;">'
                                f'<div><div style="font-size:0.7rem; color:#666;">売上高前年比</div>'
                                f'<div style="font-size:1.2rem; font-weight:bold; color:#16a34a;">+{seg["revenue_growth_rate"]:.1f}%</div></div>'
                                f'<div><div style="font-size:0.7rem; color:#666;">営業利益増減</div>'
                                f'<div style="font-size:1rem; font-weight:bold; color:#16a34a;">+{seg["operating_profit_change"]:,.0f}</div></div>'
                                f'</div>'
                                f'<div style="margin-top:4px; font-size:0.75rem; color:#666;">'
                                f'売上: {seg["revenue_previous"]:,.0f} → {seg["revenue_current"]:,.0f}</div>'
                                f'<div style="font-size:0.75rem; color:#666;">'
                                f'営利: {seg["operating_profit_previous"]:,.0f} → {seg["operating_profit_current"]:,.0f}</div>'
                                f'<div style="margin-top:6px;"><span style="background:#dcfce7; color:#16a34a; '
                                f'padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:bold;">増収増益</span></div>'
                                f'{_qc_html}'
                                f'</div>', unsafe_allow_html=True)
                with st.expander(f"全{len(structured)}セグメントの実績を見る"):
                    tbl = []
                    for seg in structured:
                        tbl.append({
                            "セグメント": seg["segment_name"],
                            "売上(前期)": f'{seg["revenue_previous"]:,.0f}' if seg["revenue_previous"] else "-",
                            "売上(当期)": f'{seg["revenue_current"]:,.0f}' if seg["revenue_current"] else "-",
                            "前年比": f'{seg["revenue_growth_rate"]:+.1f}%' if seg["revenue_growth_rate"] is not None else "-",
                            "営利(前期)": f'{seg["operating_profit_previous"]:,.0f}' if seg["operating_profit_previous"] is not None else "-",
                            "営利(当期)": f'{seg["operating_profit_current"]:,.0f}' if seg["operating_profit_current"] is not None else "-",
                            "営利増減": f'{seg["operating_profit_change"]:+,.0f}' if seg["operating_profit_change"] is not None else "-",
                            "判定": seg["category"],
                            "増減益の要因": seg.get("qual_comment") or "-",
                        })
                    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
                # カテゴリ分類
                categories = {}
                for seg in structured:
                    categories.setdefault(seg["category"], []).append(seg["segment_name"])
                for cat, names in categories.items():
                    emoji = {"増収増益": "🟢", "増収減益": "🟡", "減収増益": "🔵", "減収減益": "🔴"}.get(cat, "⚪")
                    st.markdown(f"{emoji} **{cat}**: {', '.join(names)}")

            # --- 受注残 ---
            if orders:
                st.markdown("---")
                st.markdown("**📦 受注関連**")
                ord_tbl = []
                for o in orders:
                    # 数値を期間付きで表示
                    if isinstance(o["values"], list) and o["values"]:
                        if isinstance(o["values"][0], dict):
                            val_str = " → ".join([f'{v.get("period","")}: {v["value"]:,.0f}' if v.get("period") else f'{v["value"]:,.0f}' for v in o["values"]])
                        else:
                            val_str = " / ".join([f"{v:,.0f}" for v in o["values"]])
                    else:
                        val_str = str(o["values"])
                    source = f'{o.get("source_file", "")}'
                    if o.get("source_page"):
                        source += f' p.{o["source_page"]}'
                    ord_tbl.append({
                        "項目": o["item"],
                        "数値（推移）": val_str,
                        "単位": o["unit"],
                        "出所": source,
                        "コンテキスト": o["context"][:60],
                    })
                st.dataframe(pd.DataFrame(ord_tbl), use_container_width=True, hide_index=True)
                # 受注残倍率
                if growth.get("order_backlog_ratio"):
                    st.metric("受注残高 / 売上高 倍率", f"{growth['order_backlog_ratio']:.2f}倍")
                if growth.get("order_yoy") is not None:
                    st.metric("受注高 前年比", f"{growth['order_yoy']:+.1f}%")

            # --- KPI ---
            if kpis:
                st.markdown("---")
                st.markdown("**📊 KPI指標**")
                kpi_tbl = []
                for k in kpis:
                    kpi_tbl.append({
                        "KPI": k["kpi_name"],
                        "数値": " / ".join([f"{v:,.0f}" for v in k["values"]]),
                        "単位": k["unit"],
                        "コンテキスト": k["context"][:80],
                    })
                st.dataframe(pd.DataFrame(kpi_tbl), use_container_width=True, hide_index=True)

                # KPI成長率
                if growth.get("kpi_growth"):
                    st.markdown("**KPI成長率**")
                    for kg in growth["kpi_growth"]:
                        color = "#16a34a" if kg["yoy"] >= 10 else ("#f59e0b" if kg["yoy"] >= 0 else "#ef4444")
                        st.markdown(
                            f'<div style="display:inline-block; background:#f8f9fa; border-radius:8px; '
                            f'padding:8px 16px; margin:4px; border-left:4px solid {color};">'
                            f'<b>{kg["name"]}</b>: {kg["previous"]:,.0f} → {kg["current"]:,.0f} '
                            f'<span style="color:{color}; font-weight:bold;">({kg["yoy"]:+.1f}%)</span>'
                            f'</div>', unsafe_allow_html=True)

            # --- 成長判定サマリー ---
            if orders or kpis:
                st.markdown("---")
                st.markdown("**🎯 成長判定**")
                badges = []
                if growth.get("order_yoy") and growth["order_yoy"] >= 10:
                    badges.append("🟢 受注高 YoY +10%超")
                if growth.get("kpi_growth"):
                    for kg in growth["kpi_growth"]:
                        if kg["yoy"] >= 10:
                            badges.append(f"🟢 {kg['name']} YoY +{kg['yoy']:.0f}%")
                if badges:
                    st.markdown(" / ".join(badges))
                else:
                    st.caption("成長判定: 該当なし（YoY +10%超の項目なし）")

            if not structured and not orders and not kpis:
                st.info(
                    "セグメント別の内訳は検出されませんでした。単一セグメント企業（事業セグメントが1つ）、"
                    "またはこのPDFにセグメント表・本文説明が含まれていない可能性があります。"
                    "上部の数値ベース要因コメント／業績推移をご参照ください。"
                )

st.divider()

# --- 業績推移チャート ---
st.markdown("**通期業績推移**")

# EDINET DB financials（年次時系列FY2012~）+ earnings Q4 + FY予測
_fin_annual_list = []

# 1. financialsから年次データを取得（revenue, ordinary_incomeあり）
if edb_full and edb_full.get("financials"):
    _fd = edb_full["financials"]
    _fl = _fd.get("data", []) if isinstance(_fd, dict) else (_fd if isinstance(_fd, list) else [])
    for f_item in _fl:
        fy = f_item.get("fiscal_year")
        if fy and fy >= 2022:
            _fin_annual_list.append({
                "FY": f"FY{str(fy)[-2:]}",
                "fiscal_year": fy,
                "revenue": f_item.get("revenue"),  # 円単位
                "operating_income": f_item.get("operating_income") or f_item.get("ordinary_income"),  # 営業利益優先、なければ経常利益
                "net_income": f_item.get("net_income"),
                "eps": f_item.get("adjusted_eps") or f_item.get("eps"),
            })

# 2. earningsのQ4（通期決算短信）から営業利益を補完（百万円単位）
if edb_full and edb_full.get("earnings"):
    _ed2 = edb_full["earnings"]
    _inner2 = _ed2.get("data", {}) if isinstance(_ed2, dict) else {}
    _el2 = _inner2.get("earnings", []) if isinstance(_inner2, dict) else []
    for e in _el2:
        if e.get("quarter") == 4:
            fy_end = e.get("fiscal_year_end", "")
            fy_num = int(fy_end[:4]) if fy_end and len(fy_end) >= 4 else 0
            if fy_num >= 2022:
                # 既存エントリを更新 or 追加
                found = False
                for existing in _fin_annual_list:
                    if existing["fiscal_year"] == fy_num:
                        # earnings（百万円）で上書き
                        if e.get("operating_income"):
                            existing["operating_income"] = e["operating_income"] * 1e6  # 百万→円
                        if e.get("revenue") and not existing.get("revenue"):
                            existing["revenue"] = e["revenue"] * 1e6
                        if e.get("net_income") and not existing.get("net_income"):
                            existing["net_income"] = e["net_income"] * 1e6
                        found = True
                        break
                if not found:
                    _fin_annual_list.append({
                        "FY": f"FY{str(fy_num)[-2:]}",
                        "fiscal_year": fy_num,
                        "revenue": (e.get("revenue") or 0) * 1e6,
                        "operating_income": (e.get("operating_income") or 0) * 1e6,
                        "net_income": (e.get("net_income") or 0) * 1e6,
                        "eps": e.get("eps"),
                    })

# 3. FY27E予測を追加
if _edb_latest_q:
    _fy_end = _edb_latest_q.get("fiscal_year_end", "")
    _fy_e_num = int(_fy_end[:4]) if _fy_end and len(_fy_end) >= 4 else 0
    if _fy_e_num >= 2022:
        # 重複チェック
        _exists = any(d["fiscal_year"] == _fy_e_num for d in _fin_annual_list)
        if not _exists:
            _fin_annual_list.append({
                "FY": f"FY{str(_fy_e_num)[-2:]}E",
                "fiscal_year": _fy_e_num,
                "revenue": (_edb_latest_q.get("forecast_revenue") or 0) * 1e6,
                "operating_income": (_edb_latest_q.get("forecast_operating_income") or 0) * 1e6,
                "net_income": (_edb_latest_q.get("forecast_net_income") or 0) * 1e6,
                "eps": _edb_latest_q.get("forecast_eps"),
                "is_forecast": True,
            })

# ソート＆重複除去
_fin_annual_list = sorted(_fin_annual_list, key=lambda x: x.get("fiscal_year", 0))
# fiscal_year重複があれば後のもの（earningsで上書き済み）を残す
_seen_fy = {}
for d in _fin_annual_list:
    _seen_fy[d["fiscal_year"]] = d
_fin_annual_list = sorted(_seen_fy.values(), key=lambda x: x["fiscal_year"])

if _fin_annual_list and len(_fin_annual_list) >= 2:
    col_chart1, col_chart2, col_chart3 = st.columns(3)

    with col_chart1:
        labels = [d["FY"] for d in _fin_annual_list]
        revenues_oku = [(d.get("revenue") or 0) / 1e8 for d in _fin_annual_list]
        oi_oku = [(d.get("operating_income") or 0) / 1e8 for d in _fin_annual_list]

        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(x=labels, y=revenues_oku, name="売上高", marker_color="rgba(239,83,80,0.7)"))
        fig_rev.add_trace(go.Scatter(x=labels, y=oi_oku, mode="lines+markers", name="営業利益", line=dict(color="#ff7043", width=3)))
        fig_rev.update_layout(title="売上高・営業利益（億円）", height=320, template="plotly_white", margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_chart2:
        margin_labels = []
        margin_vals = []
        for d in _fin_annual_list:
            rev = d.get("revenue") or 0
            oi = d.get("operating_income") or 0
            if rev > 0 and oi != 0:
                margin_labels.append(d["FY"])
                margin_vals.append(oi / rev * 100)
        if margin_labels and len(margin_labels) >= 2:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=margin_labels, y=margin_vals, mode="lines+markers", line=dict(color="#26a69a", width=3), fill="tozeroy", fillcolor="rgba(38,166,154,0.1)"))
            fig_m.update_layout(title="営業利益率（%）", height=320, template="plotly_white", margin=dict(l=10,r=10,t=40,b=10), yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("営業利益率データ不足")

    with col_chart3:
        eps_labels = [d["FY"] for d in _fin_annual_list if d.get("eps")]
        eps_vals = [d["eps"] for d in _fin_annual_list if d.get("eps")]
        if eps_labels and len(eps_labels) >= 2:
            fig_eps = go.Figure()
            fig_eps.add_trace(go.Scatter(x=eps_labels, y=eps_vals, mode="lines+markers", line=dict(color="#ef5350", width=3), marker=dict(size=8)))
            fig_eps.update_layout(title="会社EPS予想の推移", height=320, template="plotly_white", margin=dict(l=10,r=10,t=40,b=10), yaxis=dict(ticksuffix="円"))
            st.plotly_chart(fig_eps, use_container_width=True)
        else:
            st.info("EPSデータ不足")

elif income_annual is not None and not income_annual.empty:
    col_chart1, col_chart2, col_chart3 = st.columns(3)
    with col_chart1:
        rev_items = [item for item in ["Total Revenue", "Operating Income"] if item in income_annual.index]
        if rev_items:
            chart_df = income_annual.loc[rev_items].T / 1e8
            chart_df = chart_df.sort_index()
            chart_df.index = pd.to_datetime(chart_df.index).strftime("FY%y")
            fig_rev = go.Figure()
            if "Total Revenue" in chart_df.columns:
                fig_rev.add_trace(go.Bar(x=chart_df.index, y=chart_df["Total Revenue"], name="売上高", marker_color="rgba(239,83,80,0.7)"))
            if "Operating Income" in chart_df.columns:
                fig_rev.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Operating Income"], mode="lines+markers", name="営業利益", line=dict(color="#ff7043", width=3)))
            fig_rev.update_layout(title="売上高・営業利益（億円）", height=320, template="plotly_white", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_rev, use_container_width=True)
    with col_chart2:
        pass
    with col_chart3:
        pass
else:
    st.info("年次業績データを取得できませんでした")

st.divider()

# =====================================================================
# PAGE 02: 株価・バリュエーション・財務
# =====================================================================
st.markdown('<div style="font-size:0.75rem; color:#ef5350; font-weight:bold; letter-spacing:1px; margin-top:20px;">PAGE 02</div>', unsafe_allow_html=True)
st.subheader("株価・バリュエーション・財務")

# --- KPI行 ---
_div_yield_pct = get_dividend_yield_pct(info)
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("会社予想PER", f"{per_val:.1f}倍" if per_val else "N/A",
              help=f"出所: {per_source}" if per_source else None)
with col2:
    st.metric("PBR", f"{pbr_val:.1f}倍" if pbr_val else "N/A")
with col3:
    st.metric("配当利回り", f"{_div_yield_pct:.2f}%" if _div_yield_pct else "N/A")
with col4:
    st.metric("基準理論株価", f"{theo['base']:,.0f}円" if theo else "N/A")
with col5:
    st.metric("ROE", f"{roe_val*100:.1f}%" if roe_val else "N/A")
with col6:
    st.metric("自己資本比率", f"{equity_ratio:.1f}%" if equity_ratio else "N/A")

# 配当の補足情報（1株配当・配当性向）
_div_rate = info.get("dividendRate")
_payout_ratio = info.get("payoutRatio")
_div_notes = []
if _div_rate:
    _div_notes.append(f"1株配当（予想） {_div_rate:,.0f}円")
if _payout_ratio is not None:
    _po = _payout_ratio * 100 if _payout_ratio < 1 else _payout_ratio
    _div_notes.append(f"配当性向 {_po:.1f}%")
if _div_notes:
    st.caption("💰 " + " ／ ".join(_div_notes))

st.divider()

# --- EPS更新による予想PERの変化 ---
_eps_per_col1, _eps_per_col2 = st.columns([3, 1])
with _eps_per_col1:
    st.markdown("**EPS更新による予想PERの変化**")
with _eps_per_col2:
    st.caption("同一株価で計算")

# 同一fiscal_year_end内での最初のforecast_eps vs 最新のforecast_epsを比較
_prev_forecast_eps = None
_curr_forecast_eps = None
_curr_disclosure_date_prev = "期初"
_curr_disclosure_date = "最新"

if edb_full and edb_full.get("earnings"):
    _ed_eps = edb_full["earnings"]
    _inner_eps = _ed_eps.get("data", {}) if isinstance(_ed_eps, dict) else {}
    _el_eps = _inner_eps.get("earnings", []) if isinstance(_inner_eps, dict) else []

    if _el_eps:
        # 最新のearnings項目のfiscal_year_endを特定
        _latest_fye = _el_eps[0].get("fiscal_year_end", "")

        # 同じfiscal_year_endのearningsを全て取得
        _same_fye = [e for e in _el_eps if e.get("fiscal_year_end") == _latest_fye and e.get("forecast_eps")]

        if _same_fye:
            # 最新（リスト先頭）= 現在のforecast_eps
            _curr_forecast_eps = _same_fye[0].get("forecast_eps")
            _d_curr = _same_fye[0].get("disclosure_date", "")
            try:
                from datetime import datetime as _dt2
                _p = _dt2.strptime(_d_curr, "%a, %d %b %Y %H:%M:%S %Z") if "," in _d_curr else _dt2.strptime(_d_curr[:10], "%Y-%m-%d")
                _curr_disclosure_date = _p.strftime("%m/%d") + f" Q{_same_fye[0].get('quarter','')}後"
            except Exception:
                _curr_disclosure_date = f"Q{_same_fye[0].get('quarter','')}後"

            # 最古（リスト末尾）= 期初のforecast_eps
            _prev_forecast_eps = _same_fye[-1].get("forecast_eps")
            _d_prev = _same_fye[-1].get("disclosure_date", "")
            try:
                _p2 = _dt2.strptime(_d_prev, "%a, %d %b %Y %H:%M:%S %Z") if "," in _d_prev else _dt2.strptime(_d_prev[:10], "%Y-%m-%d")
                _curr_disclosure_date_prev = _p2.strftime("%m/%d") + " 期初"
            except Exception:
                _curr_disclosure_date_prev = "期初"

        # 同一FYEに1件しかない場合、前年度Q4のforecast_eps（=今期の期初予想）を探す
        if not _prev_forecast_eps or (_prev_forecast_eps == _curr_forecast_eps and len(_same_fye) <= 1):
            # 前年度のQ4データ（翌期予想として出す）
            _other_fye = [e for e in _el_eps if e.get("fiscal_year_end") != _latest_fye and e.get("quarter") == 4 and e.get("forecast_eps")]
            if _other_fye:
                _prev_forecast_eps = _other_fye[0].get("forecast_eps")
                _d_prev2 = _other_fye[0].get("disclosure_date", "")
                try:
                    _p3 = _dt2.strptime(_d_prev2, "%a, %d %b %Y %H:%M:%S %Z") if "," in _d_prev2 else _dt2.strptime(_d_prev2[:10], "%Y-%m-%d")
                    _curr_disclosure_date_prev = _p3.strftime("%m/%d") + " 期初"
                except Exception:
                    _curr_disclosure_date_prev = "期初"

if _curr_forecast_eps and _prev_forecast_eps and _current_price:
    _per_before = _current_price / _prev_forecast_eps if _prev_forecast_eps > 0 else None
    _per_after = _current_price / _curr_forecast_eps if _curr_forecast_eps > 0 else None

    if _per_before and _per_after:
        # バーチャート
        fig_per_change = go.Figure()
        fig_per_change.add_trace(go.Bar(
            x=[_curr_disclosure_date_prev, _curr_disclosure_date],
            y=[_per_before, _per_after],
            marker_color=["rgba(219,112,147,0.5)", "rgba(219,112,147,0.9)"],
            text=[f"{_per_before:.1f}x", f"{_per_after:.1f}x"],
            textposition="outside",
            textfont=dict(size=14, color="#1a1a3e"),
            width=0.4,
        ))
        _max_per = max(_per_before, _per_after) * 1.2
        fig_per_change.update_layout(
            height=280,
            template="plotly_white",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, _max_per], ticksuffix="x"),
            xaxis=dict(tickfont=dict(size=12)),
            showlegend=False,
        )
        st.plotly_chart(fig_per_change, use_container_width=True)

        # 注記
        if abs(_curr_forecast_eps - _prev_forecast_eps) < 0.01:
            st.caption("会社EPS予想は据え置きのため、同一株価で見た予想PERに変化はありません。")
        else:
            _eps_chg_pct = (_curr_forecast_eps - _prev_forecast_eps) / _prev_forecast_eps * 100
            st.caption(f"EPS予想: {_prev_forecast_eps:.2f}円 → {_curr_forecast_eps:.2f}円（{_eps_chg_pct:+.1f}%）により、予想PERが {_per_before:.1f}x → {_per_after:.1f}x に変化。")
elif _curr_forecast_eps and _current_price:
    # 前回予想がない場合は現在のPERのみ表示
    _per_now = _current_price / _curr_forecast_eps if _curr_forecast_eps > 0 else None
    if _per_now:
        st.metric("現在の会社予想PER", f"{_per_now:.1f}x")
        st.caption("期初予想データなし。EPS修正比較は表示できません。")
else:
    st.info("EPS予想データが不足しています")

st.divider()

# --- 配当利回り × 株価 の推移（二軸）---
_dh_col1, _dh_col2 = st.columns([3, 1])
with _dh_col1:
    st.markdown("**💹 配当利回り × 株価の推移**")
    st.caption("実績配当利回り（左軸・各日のTTM配当÷株価）と株価（右軸）。配当落ち日をマーカー表示。")
with _dh_col2:
    _dh_period = st.radio(
        "期間", ["1年", "2年", "3年", "全て"], index=3, horizontal=True,
        key="div_hist_period", label_visibility="collapsed"
    )

with st.spinner("配当利回り推移を計算中..."):
    _dh = fetch_dividend_yield_history(selected_ticker, years=5)

if _dh and _dh.get("df") is not None and not _dh["df"].empty:
    import pandas as _pd_dh
    _df_all = _dh["df"]
    # 期間フィルタ
    _years_map = {"1年": 1, "2年": 2, "3年": 3, "全て": None}
    _yn = _years_map[_dh_period]
    if _yn:
        _cutoff = _df_all.index.max() - _pd_dh.Timedelta(days=365 * _yn)
        _df = _df_all[_df_all.index >= _cutoff]
    else:
        _df = _df_all
    if _df.empty:
        _df = _df_all

    _y = _df["yield_pct"]
    _y_max, _y_min, _y_avg = float(_y.max()), float(_y.min()), float(_y.mean())

    fig_dh = make_subplots(specs=[[{"secondary_y": True}]])
    # 株価（右軸・エリア塗りの薄い青）
    fig_dh.add_trace(
        go.Scatter(
            x=_df.index, y=_df["close"], name="株価（右軸）",
            line=dict(color="#6ea8e6", width=1.2), fill="tozeroy",
            fillcolor="rgba(110,168,230,0.15)",
        ),
        secondary_y=True,
    )
    # 実績配当利回り（左軸・赤線）
    fig_dh.add_trace(
        go.Scatter(
            x=_df.index, y=_df["yield_pct"], name="実績配当利回り",
            line=dict(color="#c0392b", width=2),
        ),
        secondary_y=False,
    )
    # 補助線（最大/平均/最小）
    for _val, _lab in [(_y_max, "最大値"), (_y_avg, "平均値"), (_y_min, "最小値")]:
        fig_dh.add_hline(
            y=_val, line_dash="dash", line_color="#555", line_width=1,
            annotation_text=f"{_lab} {_val:.2f}%", annotation_position="right",
            annotation_font_size=10, secondary_y=False,
        )
    # 配当落ち日マーカー（ピンクのドット、左軸の下部に配置）
    _marker_dates = [d for d in _dh["div_dates"] if d >= _df.index.min() and d <= _df.index.max()]
    if _marker_dates:
        _marker_y = [_y_min * 0.9] * len(_marker_dates)
        fig_dh.add_trace(
            go.Scatter(
                x=_marker_dates, y=_marker_y, name="配当落ち日", mode="markers",
                marker=dict(color="#f6a5c0", size=9, line=dict(color="#e07b9a", width=1)),
            ),
            secondary_y=False,
        )
    fig_dh.update_yaxes(title_text="配当利回り(%)", secondary_y=False, ticksuffix="%",
                        rangemode="tozero", color="#c0392b")
    fig_dh.update_yaxes(title_text="株価(円)", secondary_y=True, color="#6ea8e6")
    fig_dh.update_layout(
        height=420, template="plotly_white",
        margin=dict(l=10, r=70, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_dh, use_container_width=True)

    # 下部サマリーテーブル（現在値）
    _latest = _df.iloc[-1]
    _cur_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    _s1, _s2, _s3, _s4 = st.columns(4)
    with _s1:
        st.metric("株価", f"{_latest['close']:,.0f}円")
    with _s2:
        st.metric("1株配当（TTM）", f"{_latest['ttm_div']:,.1f}円")
    with _s3:
        st.metric("実績配当利回り", f"{_latest['yield_pct']:.2f}%")
    with _s4:
        st.metric("期間平均利回り", f"{_y_avg:.2f}%")
    st.caption(
        f"更新日: {_df.index.max().strftime('%Y-%m-%d')} ／ "
        f"期間{_dh_period}の利回りレンジ: {_y_min:.2f}% 〜 {_y_max:.2f}%（平均 {_y_avg:.2f}%）。"
        "実績配当利回り=過去1年の1株配当合計÷株価。株価・配当はyfinance由来。"
    )
else:
    st.info("配当利回り推移を計算できません（配当履歴または株価データが不足。無配企業の可能性もあります）。")

st.divider()

# --- PER × 株価 の推移（二軸）---
_ph_col1, _ph_col2 = st.columns([3, 1])
with _ph_col1:
    st.markdown("**📉 PER × 株価の推移**")
    st.caption("実績PER（左軸・各日の株価÷適用EPS）と株価（右軸）。EPSは年次決算ベース（発表ラグ約3ヶ月考慮）。")
with _ph_col2:
    _ph_period = st.radio(
        "期間", ["1年", "2年", "3年", "全て"], index=3, horizontal=True,
        key="per_hist_period", label_visibility="collapsed"
    )

with st.spinner("PER推移を計算中..."):
    _ph = fetch_per_history(selected_ticker, years=5)

if _ph and _ph.get("df") is not None and not _ph["df"].empty:
    import pandas as _pd_ph
    _pdf_all = _ph["df"]
    _years_map_p = {"1年": 1, "2年": 2, "3年": 3, "全て": None}
    _ypn = _years_map_p[_ph_period]
    if _ypn:
        _cutoff_p = _pdf_all.index.max() - _pd_ph.Timedelta(days=365 * _ypn)
        _pdf = _pdf_all[_pdf_all.index >= _cutoff_p]
    else:
        _pdf = _pdf_all
    if _pdf.empty:
        _pdf = _pdf_all

    _p = _pdf["per"]
    _p_max, _p_min, _p_avg = float(_p.max()), float(_p.min()), float(_p.mean())

    fig_ph = make_subplots(specs=[[{"secondary_y": True}]])
    # 株価（右軸・薄い青エリア）
    fig_ph.add_trace(
        go.Scatter(
            x=_pdf.index, y=_pdf["close"], name="株価（右軸）",
            line=dict(color="#6ea8e6", width=1.2), fill="tozeroy",
            fillcolor="rgba(110,168,230,0.15)",
        ),
        secondary_y=True,
    )
    # 実績PER（左軸・紫線）
    fig_ph.add_trace(
        go.Scatter(
            x=_pdf.index, y=_pdf["per"], name="実績PER",
            line=dict(color="#8e44ad", width=2),
        ),
        secondary_y=False,
    )
    # 補助線（最大/平均/最小）
    for _val, _lab in [(_p_max, "最大値"), (_p_avg, "平均値"), (_p_min, "最小値")]:
        fig_ph.add_hline(
            y=_val, line_dash="dash", line_color="#555", line_width=1,
            annotation_text=f"{_lab} {_val:.1f}倍", annotation_position="right",
            annotation_font_size=10, secondary_y=False,
        )
    # EPS切替点マーカー（決算反映タイミング）
    _eps_pts = _ph.get("eps_points") or []
    _eps_marks = [(d, e) for d, e in _eps_pts if d >= _pdf.index.min() and d <= _pdf.index.max()]
    if _eps_marks:
        _mx = [d for d, _ in _eps_marks]
        _my = [_p_min * 0.95] * len(_eps_marks)
        _mt = [f"EPS {e:.1f}円" for _, e in _eps_marks]
        fig_ph.add_trace(
            go.Scatter(
                x=_mx, y=_my, name="EPS更新", mode="markers",
                marker=dict(color="#f1c40f", size=10, symbol="diamond",
                            line=dict(color="#c9a406", width=1)),
                text=_mt, hovertemplate="%{text}<extra></extra>",
            ),
            secondary_y=False,
        )

    # --- 四季報 実績PER 高値平均/安値平均を基準線として追加 ---
    _sk_for_per = st.session_state.get("shikiho_data") or {}
    _sk_hi = _sk_for_per.get("per_high_avg")
    _sk_lo = _sk_for_per.get("per_low_avg")
    if _sk_hi:
        fig_ph.add_hline(
            y=_sk_hi, line_dash="dot", line_color="#ef5350", line_width=2,
            annotation_text=f"四季報 高値平均 {_sk_hi:.1f}倍", annotation_position="left",
            annotation_font_size=11, annotation_font_color="#ef5350", secondary_y=False,
        )
    if _sk_lo:
        fig_ph.add_hline(
            y=_sk_lo, line_dash="dot", line_color="#16a34a", line_width=2,
            annotation_text=f"四季報 安値平均 {_sk_lo:.1f}倍", annotation_position="left",
            annotation_font_size=11, annotation_font_color="#16a34a", secondary_y=False,
        )

    fig_ph.update_yaxes(title_text="PER(倍)", secondary_y=False, ticksuffix="倍",
                        rangemode="tozero", color="#8e44ad")
    fig_ph.update_yaxes(title_text="株価(円)", secondary_y=True, color="#6ea8e6")
    fig_ph.update_layout(
        height=420, template="plotly_white",
        margin=dict(l=10, r=70, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ph, use_container_width=True)

    _latest_p = _pdf.iloc[-1]
    _cur_per = float(_latest_p["per"])
    _ps1, _ps2, _ps3, _ps4 = st.columns(4)
    with _ps1:
        st.metric("株価", f"{_latest_p['close']:,.0f}円")
    with _ps2:
        st.metric("適用EPS", f"{_latest_p['eps']:,.1f}円")
    with _ps3:
        st.metric("実績PER", f"{_cur_per:.1f}倍")
    with _ps4:
        st.metric("期間平均PER", f"{_p_avg:.1f}倍")

    # --- 四季報の高値平均/安値平均を基準にした 高値圏/安値圏 判定 ---
    _sk_hi2 = (st.session_state.get("shikiho_data") or {}).get("per_high_avg")
    _sk_lo2 = (st.session_state.get("shikiho_data") or {}).get("per_low_avg")
    if _sk_hi2 and _sk_lo2 and _sk_hi2 > _sk_lo2:
        # 安値平均=0%, 高値平均=100% として現在PERの相対位置
        _pos = (_cur_per - _sk_lo2) / (_sk_hi2 - _sk_lo2) * 100
        if _cur_per >= _sk_hi2:
            _zc, _zl = "#ef5350", "割高圏（四季報高値平均を上回る）"
        elif _pos >= 66:
            _zc, _zl = "#ff7043", "高値圏"
        elif _pos >= 34:
            _zc, _zl = "#ffa726", "中間圏"
        elif _cur_per > _sk_lo2:
            _zc, _zl = "#26a69a", "安値圏"
        else:
            _zc, _zl = "#16a34a", "割安圏（四季報安値平均を下回る）"
        st.markdown(
            f'<div style="background:#f8f9fa; border-radius:10px; padding:14px 18px; '
            f'border-left:6px solid {_zc}; margin-top:8px;">'
            f'<div style="font-size:0.8rem; color:#666;">四季報PERレンジでの位置づけ（安値平均{_sk_lo2:.1f}倍〜高値平均{_sk_hi2:.1f}倍）</div>'
            f'<div style="font-size:1.3rem; font-weight:bold; color:{_zc};">'
            f'現在 {_cur_per:.1f}倍 → {_zl}</div>'
            f'<div style="font-size:0.82rem; color:#555; margin-top:4px;">'
            f'レンジ内ポジション: {max(0,min(100,_pos)):.0f}%（0%=安値平均, 100%=高値平均）</div>'
            f'</div>', unsafe_allow_html=True)
    elif not (_sk_hi2 and _sk_lo2):
        st.caption("💡 四季報の実績PER（高値平均/安値平均）を取得すると、現在PERが高値圏か安値圏か判定できます（ヘッダー右上の「📘 四季報更新」）。")

    st.caption(
        f"更新日: {_pdf.index.max().strftime('%Y-%m-%d')} ／ "
        f"期間{_ph_period}のPERレンジ: {_p_min:.1f}倍 〜 {_p_max:.1f}倍（平均 {_p_avg:.1f}倍）。"
        "実績PER=株価÷年次EPS（発表ラグ約3ヶ月考慮）。EPSは年1回更新のため段差が生じます。株価・EPSはyfinance由来。"
    )
else:
    st.info("PER推移を計算できません（EPS履歴または株価データが不足。赤字企業や新規上場で履歴が短い可能性もあります）。")

st.divider()

# --- PEGレシオ + 理論株価 ---
peg_col1, peg_col2 = st.columns([1, 1])

with peg_col1:
    st.markdown("**📈 PEGレシオ**")
    if peg_val is not None:
        if peg_val < 0:
            peg_color, peg_label = "#888", "減益（参考外）"
        elif peg_val < 1.0:
            peg_color, peg_label = "#26a69a", "割安"
        elif peg_val < 1.5:
            peg_color, peg_label = "#ffa726", "適正"
        elif peg_val < 2.0:
            peg_color, peg_label = "#ff7043", "やや割高"
        else:
            peg_color, peg_label = "#ef5350", "割高"

        # 暫定ソース（売上成長率代用など）はバッジ表示
        _is_provisional = peg_source in ("売上成長率(暫定)",)
        _src_badge = (
            f'<span style="background:#fff3cd; color:#856404; padding:2px 8px; border-radius:4px; '
            f'font-size:0.72rem; font-weight:bold;">暫定</span> ' if _is_provisional else ""
        )
        st.markdown(
            f'<div style="background:#f8f9fa; border-radius:12px; padding:24px; border-left:5px solid {peg_color};">'
            f'<div style="font-size:2.5rem; font-weight:bold; color:{peg_color};">{peg_val:.2f}</div>'
            f'<div style="font-size:0.9rem; color:#666; margin-top:4px;">PEGレシオ = PER ÷ 成長率(%)</div>'
            f'<div style="font-size:1.1rem; font-weight:bold; color:{peg_color}; margin-top:8px;">判定: {peg_label}</div>'
            f'<div style="font-size:0.8rem; color:#888; margin-top:6px;">{_src_badge}算出根拠: {peg_source}</div>'
            f'</div>', unsafe_allow_html=True)

        # 内訳テーブル（成長率の実値を表示）
        _eg = info.get("earningsGrowth")
        _eqg = info.get("earningsQuarterlyGrowth")
        _rg = info.get("revenueGrowth")
        _rows_peg = [f"| 項目 | 値 |", "|------|-----|"]
        if per_val:
            _rows_peg.append(f"| PER | {per_val:.1f}倍 |")
        if _eg is not None:
            _rows_peg.append(f"| EPS成長率(年次) | {_eg*100:+.1f}% |")
        if _eqg is not None:
            _rows_peg.append(f"| EPS成長率(四半期) | {_eqg*100:+.1f}% |")
        if _rg is not None:
            _rows_peg.append(f"| 売上成長率 | {_rg*100:+.1f}% |")
        _rows_peg.append(f"| PEGレシオ | {peg_val:.2f} |")
        st.markdown("\n".join(_rows_peg))
        if _is_provisional:
            st.caption("※ EPS成長率が取得できない/減益のため、売上成長率で代用した暫定値です。")
    else:
        # 計算できない理由を具体的に表示
        _eg = info.get("earningsGrowth")
        _rg = info.get("revenueGrowth")
        _detail = []
        if _eg is not None:
            _detail.append(f"EPS成長率{_eg*100:+.1f}%")
        if _rg is not None:
            _detail.append(f"売上成長率{_rg*100:+.1f}%")
        _dtxt = f"（{peg_source}。" + ("／".join(_detail) + "）" if _detail else "）")
        st.info(f"PEGレシオを計算できません{_dtxt}")
        st.caption("PEGは PER÷成長率 で算出します。成長率が負（減益）だと指標の意味が崩れるため表示していません。")

with peg_col2:
    st.markdown("**理論株価レンジ**")
    if theo:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        colors = ["#ff7043", "#ffa726", "#26a69a"]
        labels = [f"悲観 24x", f"基準 30x", f"楽観 36x"]
        values = [theo["pessimistic"], theo["base"], theo["optimistic"]]
        fig_theo = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:,.0f}" for v in values], textposition="outside"))
        if current_price:
            fig_theo.add_hline(y=current_price, line_dash="dash", line_color="#1a1a3e", annotation_text=f"現在 {current_price:,.0f}円")
        fig_theo.update_layout(title=f"会社EPS {theo['eps']:.2f}円 × PER", height=350, template="plotly_white", margin=dict(l=10,r=10,t=50,b=10), showlegend=False)
        st.plotly_chart(fig_theo, use_container_width=True)
    else:
        st.info("理論株価を計算できません")

st.divider()

# --- 資産バリュー性（清原式ネットキャッシュ / 正味流動資産）---
st.markdown("**💰 資産バリュー性**")
st.caption("清原達郎氏『わが投資術』のネットキャッシュ比率と、かぶ1000流の正味流動資産（ネットネット）")

_av = calc_asset_value_metrics(current_data.get("balance"), info)
if _av:
    av_col1, av_col2 = st.columns(2)

    # 清原式ネットキャッシュ比率
    with av_col1:
        _kr = _av["kiyohara_ratio"]
        _knc = _av["kiyohara_net_cash"]
        if _kr is not None:
            # 判定: 1.0以上=時価総額を上回る現金的価値（超割安）, 0以上=ネットキャッシュプラス
            if _kr >= 1.0:
                _kc, _kl = "#16a34a", "時価総額超（超割安）"
            elif _kr >= 0.5:
                _kc, _kl = "#26a69a", "極めて割安"
            elif _kr >= 0:
                _kc, _kl = "#ffa726", "ネットキャッシュ プラス"
            else:
                _kc, _kl = "#ef5350", "ネットキャッシュ マイナス"
            st.markdown(
                f'<div style="background:#f8f9fa; border-radius:12px; padding:20px; border-left:5px solid {_kc};">'
                f'<div style="font-size:0.8rem; color:#666;">清原式ネットキャッシュ比率</div>'
                f'<div style="font-size:2.2rem; font-weight:bold; color:{_kc};">{_kr:.2f}</div>'
                f'<div style="font-size:0.85rem; color:{_kc}; font-weight:bold;">{_kl}</div>'
                f'<div style="font-size:0.8rem; color:#666; margin-top:8px;">'
                f'ネットキャッシュ額: {_knc/1e8:,.0f}億円</div>'
                f'</div>', unsafe_allow_html=True)
            st.caption("＝（流動資産＋投資有価証券×70%－負債合計）÷時価総額")
        else:
            st.info("清原式ネットキャッシュ比率を計算できません（BS項目または時価総額が不足）")

    # 換金性資産による割安度（ネットネット指数＋カバー率）
    with av_col2:
        _nnv = _av["net_net_value"]
        _nni = _av["net_net_index"]
        _lc = _av.get("liquid_coverage")   # 換金性資産 ÷ 時価総額（負債控除前・連続評価用）
        if _lc is not None:
            # カバー率で割安度を連続評価（ほぼ全銘柄で値が出る）
            # 1.0以上=換金性資産だけで時価総額をカバー, 0.5以上=半分, …
            if _lc >= 1.0:
                _nc, _nl = "#16a34a", "換金性資産が時価総額を上回る（非常に割安）"
            elif _lc >= 0.67:
                _nc, _nl = "#26a69a", "換金性資産が時価総額の2/3以上（割安）"
            elif _lc >= 0.4:
                _nc, _nl = "#ffa726", "換金性資産が時価総額の4割以上"
            else:
                _nc, _nl = "#90a4ae", "換金性資産による下支えは限定的"
            # ネットネット指数が成立していれば強調表示
            _idx_line = ""
            if _nni is not None:
                _idx_badge = "✅ ネットネット指数 " + f"{_nni:.2f}（1未満＝超割安）" if _nni < 1 else f"ネットネット指数 {_nni:.2f}"
                _idx_line = f'<div style="font-size:0.8rem; color:#16a34a; margin-top:6px; font-weight:bold;">{_idx_badge}</div>'
            else:
                _idx_line = '<div style="font-size:0.78rem; color:#999; margin-top:6px;">ネットネット指数: 対象外（換金性資産＜総負債）</div>'
            st.markdown(
                f'<div style="background:#f8f9fa; border-radius:12px; padding:20px; border-left:5px solid {_nc};">'
                f'<div style="font-size:0.8rem; color:#666;">換金性資産カバー率（かぶ1000流ベース）</div>'
                f'<div style="font-size:2.2rem; font-weight:bold; color:{_nc};">{_lc:.2f}</div>'
                f'<div style="font-size:0.85rem; color:{_nc}; font-weight:bold;">{_nl}</div>'
                f'{_idx_line}'
                f'<div style="font-size:0.8rem; color:#666; margin-top:8px;">'
                f'換金性資産 {_av["liquid_assets"]/1e8:,.0f}億円 ／ 正味額 {_nnv/1e8:,.0f}億円</div>'
                f'</div>', unsafe_allow_html=True)
            st.caption("カバー率＝換金性資産÷時価総額（1以上で非常に割安）｜ネットネット指数＝時価総額÷正味額（1未満が割安）")
        else:
            st.info("換金性資産による割安度を計算できません（BS項目または時価総額が不足）")

    # 内訳テーブル
    with st.expander("資産バリュー性の計算内訳を見る"):
        def _oku(v):
            return f"{v/1e8:,.0f}億円" if v is not None else "取得不可"
        _rows = [
            {"項目": "流動資産", "金額": _oku(_av["current_assets"])},
            {"項目": "投資有価証券", "金額": _oku(_av["investment_securities"])},
            {"項目": "現金及び預金", "金額": _oku(_av["cash"])},
            {"項目": "売掛金・受取手形", "金額": _oku(_av["receivables"])},
            {"項目": "短期有価証券", "金額": _oku(_av["short_term_securities"])},
            {"項目": "貸倒引当金", "金額": _oku(_av["allowance_doubtful"])},
            {"項目": "換金性資産（貸倒控除後）", "金額": _oku(_av["liquid_assets"])},
            {"項目": "負債合計", "金額": _oku(_av["total_liabilities"])},
            {"項目": "換金性資産の正味額", "金額": _oku(_av["net_net_value"])},
            {"項目": "時価総額", "金額": _oku(_av["market_cap"])},
        ]
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        if _av.get("note"):
            st.caption("※ " + _av["note"])
        st.caption("BS項目はyfinance（直近本決算）由来。EDINET開示との差異や、投資有価証券の含み損益は未反映です。")

    # --- 資産バリュー総合判定 ---
    _kr = _av["kiyohara_ratio"]
    _nni = _av["net_net_index"]       # ネットネット指数（1未満が割安、小さいほど割安）
    _lc = _av.get("liquid_coverage")  # 換金性資産カバー率（大きいほど割安、負債控除前）
    _av_msgs = []          # 該当した根拠
    _av_level = 0          # 0:限定的 1:下支え 2:割安 3:非常に強い
    # 清原NC比率・カバー率は「大きいほど割安」、ネットネット指数は「小さいほど割安（1未満が基準）」
    _kr_strong = _kr is not None and _kr > 1
    _nni_strong = _nni is not None and _nni < 1
    if _kr_strong and _nni_strong:
        _av_msgs.append("清原NC比率>1 かつ ネットネット指数<1（換金性資産だけで時価総額をカバー）")
        _av_level = 3
    else:
        if _kr_strong:
            _av_msgs.append("清原NC比率 > 1（時価総額を上回るネットキャッシュ）")
            _av_level = max(_av_level, 3)
        if _nni is not None and _nni < 0.67:
            _av_msgs.append("ネットネット指数 < 0.67（かなり割安）")
            _av_level = max(_av_level, 2)
        elif _nni is not None and _nni < 1:
            _av_msgs.append("ネットネット指数 < 1（ネットネット割安）")
            _av_level = max(_av_level, 2)
        # 換金性資産カバー率による評価（指数が対象外でも連続的に拾える）
        if _lc is not None and _lc >= 1.0:
            _av_msgs.append("換金性資産カバー率 ≥ 1（換金性資産が時価総額以上）")
            _av_level = max(_av_level, 2)
        elif _lc is not None and _lc >= 0.67:
            _av_msgs.append("換金性資産カバー率 ≥ 0.67（時価総額の2/3以上を換金性資産でカバー）")
            _av_level = max(_av_level, 1)
        if _kr is not None and 0.5 < _kr <= 1:
            _av_msgs.append("清原NC比率 > 0.5（資産による下支えあり）")
            _av_level = max(_av_level, 1)

    _av_label, _av_color = {
        3: ("非常に強い資産バリュー", "#16a34a"),
        2: ("資産価値で割安（ネットネット）", "#26a69a"),
        1: ("資産による下支えあり", "#ffa726"),
        0: ("資産バリュー性は限定的", "#90a4ae"),
    }[_av_level]

    _reason_html = ""
    if _av_msgs:
        _reason_html = (
            '<ul style="margin:8px 0 0 0; padding-left:18px; font-size:0.82rem; color:#444;">'
            + "".join(f"<li>{m}</li>" for m in _av_msgs)
            + "</ul>"
        )
    st.markdown(
        f'<div style="background:#f8f9fa; border-radius:12px; padding:16px 20px; '
        f'border-left:6px solid {_av_color}; margin-top:12px;">'
        f'<div style="font-size:0.8rem; color:#666;">資産バリュー総合判定</div>'
        f'<div style="font-size:1.4rem; font-weight:bold; color:{_av_color};">{_av_label}</div>'
        f'{_reason_html}'
        f'</div>', unsafe_allow_html=True)
    st.caption("判定基準: 清原NC比率>1=非常に強い・>0.5=下支え／ネットネット指数<1=割安・<0.67=かなり割安／換金性資産カバー率≥1=割安・≥0.67=下支え（指数が対象外でもカバー率で評価）")
else:
    st.info("資産バリュー性を計算できません（貸借対照表データを取得できませんでした）")

st.divider()

# --- たーちゃん式バリュー株スクリーニング（『50万円を50億円に増やした』）---
st.markdown("**📗 たーちゃん式スクリーニング判定**")
st.caption("『50万円を50億円に増やした 投資家の父から娘への教え』（たーちゃん著）の3つのバリュー株条件への適合度。"
           "資産バリュー株／収益バリュー株／シクリカルバリュー株。")

_ts = judge_tachan_screening(info, current_data.get("balance"), current_data.get("income_annual"))
if _ts:
    def _ts_card(title, sub, d, cond_lines):
        matched = d.get("match")
        if matched:
            _c, _lab = "#16a34a", "✅ 条件に適合"
        elif d.get("reasons"):
            _c, _lab = "#ffa726", "△ 一部該当"
        else:
            _c, _lab = "#90a4ae", "× 非該当"
        _relaxed = "（緩和基準を含む）" if d.get("relaxed") else ""
        _ok_html = "".join(
            f'<div style="font-size:0.8rem; color:#2e7d32;">・{r}</div>' for r in d.get("reasons", [])
        )
        _ng_html = "".join(
            f'<div style="font-size:0.8rem; color:#b71c1c;">・{r}</div>' for r in d.get("fail", [])
        )
        return (
            f'<div style="background:#f8f9fa; border-radius:12px; padding:16px 18px; '
            f'border-left:6px solid {_c}; margin-bottom:4px; height:100%;">'
            f'<div style="font-size:0.95rem; font-weight:bold; color:#333;">{title}</div>'
            f'<div style="font-size:0.72rem; color:#888; margin-bottom:6px;">{sub}</div>'
            f'<div style="font-size:1.0rem; font-weight:bold; color:{_c}; margin-bottom:6px;">{_lab}{_relaxed}</div>'
            f'<div style="font-size:0.72rem; color:#aaa; margin-bottom:2px;">{cond_lines}</div>'
            f'{_ok_html}{_ng_html}'
            f'</div>'
        )

    _best_map = {"asset": "資産バリュー株", "earnings": "収益バリュー株", "cyclical": "シクリカルバリュー株"}
    if _ts.get("best"):
        st.markdown(
            f'<div style="background:#e8f5e9; border-radius:10px; padding:10px 16px; margin-bottom:10px; '
            f'border-left:6px solid #16a34a;">'
            f'<span style="font-size:0.95rem; font-weight:bold; color:#16a34a;">'
            f'🎯 「{_best_map[_ts["best"]]}」の条件に適合</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#f5f5f5; border-radius:10px; padding:10px 16px; margin-bottom:10px; '
            'border-left:6px solid #90a4ae;">'
            '<span style="font-size:0.9rem; color:#666;">いずれの条件にも完全適合せず（下記で各条件の達成状況を確認）</span></div>',
            unsafe_allow_html=True)

    _tc1, _tc2, _tc3 = st.columns(3)
    with _tc1:
        st.markdown(_ts_card(
            "① 資産バリュー株", "資産に対して割安（難易度★）",
            _ts["asset"],
            "条件: PBR≤0.5倍 かつ 自己資本比率≥60%（+PER≤12倍/グレアム指数≤22.5で加点）",
        ), unsafe_allow_html=True)
    with _tc2:
        st.markdown(_ts_card(
            "② 収益バリュー株", "稼ぐ力に対して割安（難易度★★）",
            _ts["earnings"],
            "条件: 営業利益率≥10%・PER≤10倍・PBR≤1.5倍・ROA≥7%・時価総額≤300億円",
        ), unsafe_allow_html=True)
    with _tc3:
        st.markdown(_ts_card(
            "③ シクリカルバリュー株", "景気の谷で仕込む（難易度★★★）",
            _ts["cyclical"],
            "条件: 景気循環業種 × 直近赤字（谷＝妙味）。PSR≤0.5倍で割安度補強",
        ), unsafe_allow_html=True)

    st.caption("※ 自己資本比率・営業利益率・ROA・時価総額・PSRはyfinance由来。"
               "シクリカルは業種と赤字の自動判定のため、実際の景気サイクル局面はご自身でご確認ください。"
               "赤字企業はPERが算出されない点に注意（たーちゃんはシクリカルではPERよりPSRを重視）。")
else:
    st.info("たーちゃん式スクリーニングを判定できません（企業データを取得できませんでした）")

st.divider()

# --- 10段階スコア内訳 + 財務健全性 ---
score_col1, score_col2 = st.columns([3, 2])

with score_col1:
    st.markdown("**10段階スコアの内訳**")
    total = scores["total"]
    st.markdown(f"総合スコア **{total}点** = {scores['momentum']*0.25:.2f} + {scores['eps_revision']*0.25:.2f} + {scores['margin']*0.20:.2f} + {scores['roe_equity']*0.15:.2f} + {scores['valuation']*0.15:.2f}")

    score_items = [
        ("決算モメンタム", scores["momentum"], 25, "#ef5350"),
        ("会社EPS予想修正", scores["eps_revision"], 25, "#ffa726"),
        ("営業利益率", scores["margin"], 20, "#ff7043"),
        ("ROE・自己資本", scores["roe_equity"], 15, "#26a69a"),
        ("バリュエーション", scores["valuation"], 15, "#4da6ff"),
    ]
    for name, score_val, weight, color in score_items:
        pct = score_val / 10 * 100
        weighted = score_val * weight / 100
        bar_color = color if score_val >= 5 else "#ef5350"
        st.markdown(
            f'<div class="score-bar-container">'
            f'<span class="score-bar-title">{name}</span>'
            f'<span class="score-bar-value">{score_val:.1f} / 10</span>'
            f'<div class="score-bar"><div class="score-bar-fill" style="width:{pct}%; background:{bar_color};"></div></div>'
            f'<div class="score-bar-detail">素点 {score_val:.1f}点 ｜ ウェイト {weight}% ｜ 寄与点 {weighted:.2f}点</div>'
            f'</div>', unsafe_allow_html=True)

with score_col2:
    st.markdown("**財務健全性と前提**")
    bps = info.get("bookValue")
    st.markdown(f"**BPS**: {bps:.2f}円" if bps else "**BPS**: N/A")
    st.markdown(f"**PBR**: {pbr_val:.1f}倍" if pbr_val else "**PBR**: N/A")
    st.markdown(f"**ROE**: {roe_val*100:.1f}%" if roe_val else "**ROE**: N/A")
    st.markdown(f"**自己資本比率**: {equity_ratio:.1f}%" if equity_ratio else "**自己資本比率**: N/A")

    # FCF
    cashflow_data = current_data.get("cashflow")
    fcf = None
    if cashflow_data is not None and not cashflow_data.empty:
        try:
            if "Free Cash Flow" in cashflow_data.index:
                fcf = cashflow_data.loc["Free Cash Flow"].iloc[0]
        except Exception:
            pass
    st.markdown(f"**FCF**: {fcf/1e8:,.1f}億円" if fcf else "**FCF**: N/A")

    # ネットキャッシュ
    net_cash = None
    _balance = current_data.get("balance")
    if _balance is not None and not _balance.empty:
        try:
            latest_b2 = _balance.iloc[:, 0]
            cash = latest_b2.get("Cash And Cash Equivalents") or 0
            debt = latest_b2.get("Total Debt") or 0
            if cash or debt:
                net_cash = cash - debt
        except Exception:
            pass
    st.markdown(f"**ネットキャッシュ**: {net_cash/1e8:,.1f}億円" if net_cash else "**ネットキャッシュ**: N/A")

    st.divider()
    st.caption("理論株価は「最新会社EPS × PERシナリオ」の簡易方式。PER 24・30・36倍を使用。")

# --- 共通の点数基準と判定ルール（プルダウン） ---
with st.expander("▶ 共通の点数基準と判定ルールを見る"):
    st.markdown("#### 総合スコア ＝ Σ（各評価項目の10段階スコア × ウェイト）")
    st.markdown("#### 評価項目と対象会社のルール")

    # 各スコアの詳細説明を動的生成
    _q = _edb_latest_q
    _mom_detail = ""
    _eps_rev_detail = ""
    _margin_detail = ""
    _roe_detail = ""
    _val_detail = ""

    if _q:
        _rev_chg = _q.get("revenue_change")
        _oi_chg = _q.get("operating_income_change")
        _ni_chg = _q.get("net_income_change")
        _mom_detail = (
            f"前年比は売上{_rev_chg:+.1f}%、営業利益{_oi_chg:+.1f}%、純利益{_ni_chg:+.1f}%で加重。"
            if _rev_chg is not None and _oi_chg is not None and _ni_chg is not None else ""
        )
        _f_eps = _q.get("forecast_eps")
        _prev_f_eps = _edb_latest_annual.get("forecast_eps") if _edb_latest_annual else None
        if _f_eps and _prev_f_eps:
            if abs(_f_eps - _prev_f_eps) < 0.01:
                _eps_rev_detail = f"EPS予想{_f_eps:.2f}円は据え置き。修正率0.0%のため5点。"
            else:
                _chg = (_f_eps - _prev_f_eps) / _prev_f_eps * 100
                _eps_rev_detail = f"EPS予想{_prev_f_eps:.2f}円→{_f_eps:.2f}円（{_chg:+.1f}%）。"

        if _q.get("revenue") and _q.get("operating_income"):
            _q_margin = _q["operating_income"] / _q["revenue"] * 100
            if _oi_chg is not None and _rev_chg is not None and _oi_chg != -100 and _rev_chg != -100:
                _p_oi = _q["operating_income"] / (1 + _oi_chg / 100)
                _p_rev = _q["revenue"] / (1 + _rev_chg / 100)
                _p_margin = _p_oi / _p_rev * 100
                _diff = _q_margin - _p_margin
                _margin_detail = f"Q1営業利益率は{_p_margin:.2f}%から{_q_margin:.2f}%へ{_diff:+.2f}pt。"

    if _edb_latest_ratio:
        _roe_off = _edb_latest_ratio.get("roe_official")
        if _roe_off:
            _roe_detail = f"ROE{_roe_off*100:.2f}%は{_roe_score(_roe_off*100):.0f}点、"
    if equity_ratio:
        _roe_detail += f"自己資本比率{equity_ratio:.1f}%は{_equity_ratio_score(equity_ratio):.0f}点。"

    if per_val:
        _per_s = _per_score(per_val, 30)
        _val_detail = f"株価{_current_price:,.0f}円で予想PER{per_val:.2f}倍。基準PER30倍に対するPER点は{_per_s:.1f}点、PBRは同業比較なしの中立5点。"

    _score_table = f"""
| 項目 | 点 | ウェイト | 寄与点 | ルール・判定根拠 |
|------|-----|---------|--------|----------------|
| 決算モメンタム | {scores['momentum']:.1f} | 25% | {scores['momentum']*0.25:.2f} | Q1前年比（売上10%・営業利益50%・純利益20%・EPS20%）。{_mom_detail} |
| 会社EPS予想修正 | {scores['eps_revision']:.1f} | 25% | {scores['eps_revision']*0.25:.2f} | 直前会社予想からの修正率。{_eps_rev_detail} |
| 営業利益率 | {scores['margin']:.1f} | 20% | {scores['margin']*0.20:.2f} | Q1前年差ptベース。{_margin_detail} |
| ROE・自己資本 | {scores['roe_equity']:.1f} | 15% | {scores['roe_equity']*0.15:.2f} | ROE点60%＋自己資本比率点40%。{_roe_detail} |
| バリュエーション | {scores['valuation']:.1f} | 15% | {scores['valuation']*0.15:.2f} | PER点70%＋PBR点30%。{_val_detail} |
"""
    st.markdown(_score_table)

    st.markdown("---")
    st.markdown("#### 修正率・前年比・計画差の共通採点")
    st.markdown("""
| 点数 | 基準 |
|------|------|
| 10点 | +15%以上 |
| 9点 | +10%以上 |
| 8点 | +7%以上 |
| 7点 | +4%以上 |
| 6点 | +1%以上 |
| 5点 | -1%以上 |
| 4点 | -4%以上 |
| 3点 | -7%以上 |
| 2点 | -10%以上 |
| 1点 | -10%未満 |
""")

    st.markdown("#### 営業利益率前年差の採点")
    st.markdown("""
| 点数 | 基準 |
|------|------|
| 10点 | +3.0pt以上 |
| 9点 | +2.0pt以上 |
| 8点 | +1.5pt以上 |
| 7点 | +1.0pt以上 |
| 6点 | +0.3pt以上 |
| 5点 | -0.3pt以上 |
| 4点 | -1.0pt以上 |
| 3点 | -1.5pt以上 |
| 2点 | -2.0pt以上 |
| 1点 | -2.0pt未満 |
""")

    st.markdown("#### 財務スコアの基準")
    st.markdown("""
**ROE**: 15%以上=10点、12%以上=9点、10%以上=8点、8%以上=7点、6%以上=6点、4%以上=5点、2%以上=4点、0%以上=3点、-5%以上=2点、-5%未満=1点

**自己資本比率**: 70%以上=10点、60%以上=9点、50%以上=8点、40%以上=7点、30%以上=6点、20%以上=5点、10%以上=4点、0%以上=3点、0%未満=1点
""")

    st.markdown("#### バリュエーションと判定帯")
    st.markdown("""
PER点＝max(1, min(10, 5.5＋(1－現在PER÷基準PER)×15))。PBR点は同業比較未取得のため5点。

| スコア | 判定 |
|--------|------|
| 9.0–10.0 | 非常に強い |
| 7.0–8.9 | 強い |
| 5.0–6.9 | 中立 |
| 3.0–4.9 | 弱い |
| 1.0–2.9 | 非常に弱い |

スコアは機械的な比較用です。PBRは同業比較を自動取得していないため中立5点。事業の質、需給、ガバナンスなどは別途確認が必要です。
""")

# =====================================================================
# 売買シグナル（ウォッチリスト全銘柄）
# =====================================================================
st.divider()
st.markdown('<div style="font-size:0.75rem; color:#ef5350; font-weight:bold; letter-spacing:1px; margin-top:20px;">SIGNAL</div>', unsafe_allow_html=True)
st.subheader("📈 売買シグナル（ウォッチリスト）")
st.caption("5条件（①年足陽線 ②月足↑ ③週足↑ ④配当利回り2.5%以上 ⑤好業績）のうち何個を満たすかで判定")

# --- 四季報 一括取得／一括更新（ウォッチリスト全銘柄を1ログインで巡回）---
_sk_bc1, _sk_bc2, _sk_bc3 = st.columns([1.2, 1.2, 4])
with _sk_bc1:
    _sk_batch_fill = st.button("📘 四季報 一括取得", key="shikiho_batch_fill",
                               help="未取得・期限切れ（90日超）の銘柄だけまとめて取得")
with _sk_bc2:
    _sk_batch_all = st.button("🔄 四季報 全更新", key="shikiho_batch_all",
                              help="四半期更新時に全銘柄を強制リフレッシュ")
with _sk_bc3:
    # キャッシュ済み件数の表示
    try:
        from shikiho_scraper import load_cached_shikiho as _lcs
        _cached_cnt = sum(1 for t in watchlist if _lcs(t) is not None)
        st.caption(f"四季報キャッシュ: {_cached_cnt}/{len(watchlist)} 銘柄（四半期＝約90日で自動的に“古い”扱い）")
    except Exception:
        st.caption("")

if _sk_batch_fill or _sk_batch_all:
    try:
        from shikiho_scraper import fetch_shikiho_batch_subprocess
        _force_batch = bool(_sk_batch_all)
        _tickers_all = list(watchlist.keys())
        with st.spinner(f"四季報を一括取得中…（{len(_tickers_all)}銘柄・1ログインで巡回、数分かかる場合があります）"):
            _batch_res = fetch_shikiho_batch_subprocess(
                _tickers_all, cache_days=90, force=_force_batch, timeout=1200
            )
        if isinstance(_batch_res, dict) and _batch_res.get("error"):
            st.warning(f"⚠️ 一括取得エラー: {_batch_res['error']}")
        else:
            _ok = sum(1 for v in _batch_res.values() if isinstance(v, dict) and v.get("ok"))
            _ng = sum(1 for v in _batch_res.values() if isinstance(v, dict) and not v.get("ok"))
            st.success(f"✅ 一括取得完了: 成功{_ok}件 / 失敗{_ng}件")
            if _ng:
                _ng_codes = [k for k, v in _batch_res.items() if isinstance(v, dict) and not v.get("ok")]
                st.caption("失敗: " + ", ".join(_ng_codes[:20]))
    except Exception as e:
        st.warning(f"⚠️ 一括取得エラー: {e}")

with st.spinner("売買シグナルを計算中..."):
    signal_rows = []
    for ticker, data in all_stock_data.items():
        sig = calc_trade_signal(ticker, data.get("info", {}))
        signal_rows.append({
            "ticker": ticker,
            "name": watchlist.get(ticker, data.get("info", {}).get("shortName", "")),
            "signal": sig,
        })

# 集計
_gem5 = sum(1 for r in signal_rows if r["signal"]["mark"] == "💎")
_gem4 = sum(1 for r in signal_rows if r["signal"]["mark"] == "🌟")
_gem3 = sum(1 for r in signal_rows if r["signal"]["mark"] == "⭐")

if _gem5 + _gem4 + _gem3 > 0:
    parts = []
    if _gem5:
        parts.append(f"💎完璧 {_gem5}件")
    if _gem4:
        parts.append(f"🌟優良 {_gem4}件")
    if _gem3:
        parts.append(f"⭐注目 {_gem3}件")
    st.success(f"{' / '.join(parts)} — 年足陽線+月足週足↑+利回り+好業績（5条件中3つ以上）")
else:
    st.info("優良銘柄マーク該当なし（5条件中3つ以上を満たす銘柄なし）")

# テーブル表示（期待値順に上位に）
sig_tbl = []
for r in sorted(signal_rows, key=lambda x: x["signal"]["expect"], reverse=True):
    sig = r["signal"]
    _sdata = all_stock_data[r["ticker"]]
    _sinfo = _sdata.get("info", {})
    price = _sinfo.get("currentPrice") or _sinfo.get("regularMarketPrice")

    # 総合スコア（ウォッチリストはyfinanceベース。EDINET DBは選択銘柄のみ取得のため未使用）
    try:
        _sc = calc_total_score(_sdata, None)
        _total_score = _sc["total"]
    except Exception:
        _total_score = None

    # PEGレシオ（暫定値は末尾に*）
    try:
        _peg_v, _peg_src = calc_peg_ratio(_sinfo)
        if _peg_v is not None:
            _peg_txt = f"{_peg_v:.2f}*" if _peg_src == "売上成長率(暫定)" else f"{_peg_v:.2f}"
        else:
            _peg_txt = "—"
    except Exception:
        _peg_txt = "—"

    # 資産バリュー性
    _av_row = calc_asset_value_metrics(_sdata.get("balance"), _sinfo)
    _avj = judge_asset_value(_av_row)
    if _avj:
        _asset_label = f'{_avj["icon"]}{_avj["label"]}'
        _kr_v = _av_row.get("kiyohara_ratio")
        _lc_v = _av_row.get("liquid_coverage")
        _kr_txt = f"{_kr_v:.2f}" if _kr_v is not None else "—"
        _lc_txt = f"{_lc_v:.2f}" if _lc_v is not None else "—"
    else:
        _asset_label, _kr_txt, _lc_txt = "—", "—", "—"

    # たーちゃん式スクリーニング適合
    try:
        _tsr = judge_tachan_screening(_sinfo, _sdata.get("balance"), _sdata.get("income_annual"))
        _ts_best_map = {"asset": "💠資産", "earnings": "💵収益", "cyclical": "🔄シクリカル"}
        if _tsr and _tsr.get("best"):
            _tachan_label = _ts_best_map[_tsr["best"]]
        elif _tsr and (_tsr["asset"]["reasons"] or _tsr["earnings"]["reasons"] or _tsr["cyclical"]["reasons"]):
            _tachan_label = "△一部"
        else:
            _tachan_label = "—"
    except Exception:
        _tachan_label = "—"

    sig_tbl.append({
        "": sig["mark"],
        "コード": r["ticker"].replace(".T", ""),
        "銘柄名": r["name"],
        "株価": f"¥{price:,.0f}" if price else "—",
        "シグナル": sig["signal_text"],
        "期待値": sig["expect"],
        "年足": "✅" if sig["cond_yearly"] else "❌",
        "月足": "✅" if sig["cond_monthly"] else "❌",
        "週足": "✅" if sig["cond_weekly"] else "❌",
        "利回り": f'✅{sig["dy_pct"]:.1f}%' if sig["cond_yield"] else f'❌{sig["dy_pct"]:.1f}%',
        "業績": "✅" if sig["cond_biz"] else "❌",
        "達成": f'{sig["cond_count"]}/5',
        "判定": sig["label"],
        "総合スコア": _total_score if _total_score is not None else None,
        "評価": score_label(_total_score) if _total_score is not None else "—",
        "PEG": _peg_txt,
        "資産バリュー": _asset_label,
        "清原NC": _kr_txt,
        "換金カバー率": _lc_txt,
        "たーちゃん": _tachan_label,
    })
st.dataframe(
    pd.DataFrame(sig_tbl),
    use_container_width=True,
    hide_index=True,
    height=500,
    column_config={
        "期待値": st.column_config.ProgressColumn(
            "期待値", min_value=-100, max_value=100, format="%d"
        ),
        "総合スコア": st.column_config.ProgressColumn(
            "総合スコア", min_value=0, max_value=10, format="%.1f"
        ),
    },
)
st.caption("シグナル: 強い買い🟢(>30) / 買い🔵(>10) / 中立⚪ / 売り🟠 / 強い売り🔴。期待値=シグナルスコア+優良条件ボーナス（-100〜100）。")
st.caption("💎完璧=5条件 / 🌟優良=4条件 / ⭐注目=3条件。年足=陽線傾向、月足=直近3ヶ月上昇、週足=直近4週上昇。")
st.caption("総合スコア=10段階（決算モメンタム25%+EPS予想修正25%+営業利益率20%+ROE自己資本15%+バリュエーション15%）。ウォッチリストはyfinanceベース簡易版（個別ページはEDINETデータ優先でより精緻）。")
st.caption("PEG=PER÷成長率。1未満=割安・1〜2=適正・2超=割高の目安。末尾*は売上成長率で代用した暫定値（減益等でEPS成長率が取れない場合）。")
st.caption("資産バリュー: 💎非常に強い / 🟢割安 / 🟡下支えあり / ⚪限定的。清原NC=清原式ネットキャッシュ比率、換金カバー率=換金性資産÷時価総額。")
st.caption("たーちゃん=『50万円を50億円に増やした』式スクリーニング適合。💠資産バリュー株 / 💵収益バリュー株 / 🔄シクリカルバリュー株 / △一部該当 / —非該当。")

# =====================================================================
# フッター
# =====================================================================
st.divider()
st.markdown(
    '<div style="background:#f8f9fa; border-radius:8px; padding:12px; border:1px solid #e9ecef;">'
    '<details><summary><b>▼ 出所・データ注記を表示</b></summary>'
    '<p style="font-size:0.82rem; color:#666; margin-top:8px;">'
    'Yahoo Finance API（yfinance）/ 理論株価は会社EPS × PER（24/30/36倍）の簡易推計 / '
    'PEGレシオはyfinance提供値または PER÷EPS成長率(%)で計算 / '
    '外部コンセンサスは使用せず、会社開示・公開株価のみで構成しています。投資判断は利用者ご自身で行ってください。'
    '</p></details></div>', unsafe_allow_html=True)
st.caption(f"🕐 最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ データソース: Yahoo Finance ｜ 銘柄数: {len(all_stock_data)}")
