"""
米国株投資分析ダッシュボード
- ウォッチリスト管理（ティッカー/銘柄名）
- 中長期売買タイミング分析（MA/RSI/MACD/ボリンジャーバンド）
- NVIDIAの投資先・出資先ニュース追跡
- NVIDIA関連銘柄バリュエーション比較
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
import requests
import json
import os
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

warnings.filterwarnings("ignore")

st.set_page_config(page_title="US Stock Investment Dashboard", page_icon="🇺🇸", layout="wide")

# =====================================================================
# ファイルパス
# =====================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(_BASE_DIR, "us_watchlist.json")
NVIDIA_INVESTMENTS_PATH = os.path.join(_BASE_DIR, "nvidia_investments.json")

# =====================================================================
# ウォッチリスト管理
# =====================================================================
def load_watchlist():
    if os.path.exists(WATCHLIST_PATH):
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_watchlist(data):
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_nvidia_investments():
    if os.path.exists(NVIDIA_INVESTMENTS_PATH):
        with open(NVIDIA_INVESTMENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"investments": [], "nvidia_ecosystem_tickers": []}


# =====================================================================
# テクニカル指標計算
# =====================================================================
def calc_sma(series, window):
    """単純移動平均"""
    return series.rolling(window=window).mean()

def calc_ema(series, span):
    """指数移動平均"""
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, period=14):
    """RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(series, window=20, num_std=2):
    """ボリンジャーバンド"""
    sma = calc_sma(series, window)
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower

def calc_atr(high, low, close, period=14):
    """ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    """ストキャスティクス"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


# =====================================================================
# 売買シグナル生成（中長期向け）
# =====================================================================
def generate_trading_signals(df):
    """
    中長期保有目線の売買シグナル生成
    - MA クロス (50/200日)
    - RSI 過熱/過冷
    - MACD クロス
    - ボリンジャーバンド位置
    - 出来高トレンド
    """
    if df is None or len(df) < 50:
        return {"signal": "データ不足", "score": 0, "details": {}}

    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None

    signals = {}
    score = 0

    # --- 1. MA クロス (50/200) ---
    ma50 = calc_sma(close, 50)
    ma200 = calc_sma(close, 200)
    if len(close) >= 200 and not np.isnan(ma50.iloc[-1]) and not np.isnan(ma200.iloc[-1]):
        if ma50.iloc[-1] > ma200.iloc[-1]:
            signals["MA Cross"] = "ゴールデンクロス (50>200) 🟢"
            score += 2
            # 直近でクロスしたか
            if ma50.iloc[-5] <= ma200.iloc[-5]:
                signals["MA Cross"] += " [直近発生!]"
                score += 1
        else:
            signals["MA Cross"] = "デッドクロス (50<200) 🔴"
            score -= 2
            if ma50.iloc[-5] >= ma200.iloc[-5]:
                signals["MA Cross"] += " [直近発生!]"
                score -= 1
    elif len(close) >= 50:
        ma20 = calc_sma(close, 20)
        if not np.isnan(ma20.iloc[-1]) and not np.isnan(ma50.iloc[-1]):
            if ma20.iloc[-1] > ma50.iloc[-1]:
                signals["MA Cross"] = "短期上昇トレンド (20>50) 🔵"
                score += 1
            else:
                signals["MA Cross"] = "短期下降トレンド (20<50) 🟠"
                score -= 1

    # --- 2. RSI ---
    rsi = calc_rsi(close)
    if not np.isnan(rsi.iloc[-1]):
        rsi_val = rsi.iloc[-1]
        if rsi_val > 70:
            signals["RSI"] = f"過熱圏 ({rsi_val:.1f}) 🔴"
            score -= 1.5
        elif rsi_val > 60:
            signals["RSI"] = f"やや過熱 ({rsi_val:.1f}) 🟠"
            score -= 0.5
        elif rsi_val < 30:
            signals["RSI"] = f"売られすぎ ({rsi_val:.1f}) 🟢"
            score += 2
        elif rsi_val < 40:
            signals["RSI"] = f"やや売られすぎ ({rsi_val:.1f}) 🔵"
            score += 1
        else:
            signals["RSI"] = f"中立圏 ({rsi_val:.1f}) ⚪"

    # --- 3. MACD ---
    macd_line, signal_line, histogram = calc_macd(close)
    if not np.isnan(histogram.iloc[-1]):
        if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
            signals["MACD"] = "買いシグナル (ヒストグラム転換) 🟢"
            score += 2
        elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
            signals["MACD"] = "売りシグナル (ヒストグラム転換) 🔴"
            score -= 2
        elif histogram.iloc[-1] > 0:
            signals["MACD"] = "上昇継続 🔵"
            score += 0.5
        else:
            signals["MACD"] = "下落継続 🟠"
            score -= 0.5

    # --- 4. ボリンジャーバンド ---
    upper, mid, lower = calc_bollinger(close)
    if not np.isnan(upper.iloc[-1]):
        price = close.iloc[-1]
        bb_position = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        if bb_position > 1.0:
            signals["Bollinger"] = f"上限突破 (位置: {bb_position:.2f}) 🔴"
            score -= 1
        elif bb_position > 0.8:
            signals["Bollinger"] = f"上限接近 (位置: {bb_position:.2f}) 🟠"
            score -= 0.5
        elif bb_position < 0.0:
            signals["Bollinger"] = f"下限突破 (位置: {bb_position:.2f}) 🟢"
            score += 1.5
        elif bb_position < 0.2:
            signals["Bollinger"] = f"下限接近 (位置: {bb_position:.2f}) 🔵"
            score += 1
        else:
            signals["Bollinger"] = f"バンド内 (位置: {bb_position:.2f}) ⚪"

    # --- 5. 出来高トレンド ---
    if volume is not None and len(volume) >= 20:
        vol_ma20 = volume.rolling(20).mean()
        if not np.isnan(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0:
            vol_ratio = volume.iloc[-5:].mean() / vol_ma20.iloc[-1]
            if vol_ratio > 1.5 and close.iloc[-1] > close.iloc[-5]:
                signals["Volume"] = f"出来高急増+上昇 ({vol_ratio:.1f}x) 🟢"
                score += 1
            elif vol_ratio > 1.5 and close.iloc[-1] < close.iloc[-5]:
                signals["Volume"] = f"出来高急増+下落 ({vol_ratio:.1f}x) 🔴"
                score -= 1
            else:
                signals["Volume"] = f"通常 ({vol_ratio:.1f}x) ⚪"

    # --- 6. 価格トレンド (OLS) ---
    if len(close) >= 60:
        y = close.iloc[-60:].values
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value ** 2
        trend_pct = (slope * 60) / close.iloc[-60] * 100
        if trend_pct > 5 and r2 > 0.5:
            signals["Trend"] = f"強い上昇トレンド (+{trend_pct:.1f}%, R²={r2:.2f}) 🟢"
            score += 1.5
        elif trend_pct > 0:
            signals["Trend"] = f"緩やかな上昇 (+{trend_pct:.1f}%, R²={r2:.2f}) 🔵"
            score += 0.5
        elif trend_pct < -5 and r2 > 0.5:
            signals["Trend"] = f"強い下降トレンド ({trend_pct:.1f}%, R²={r2:.2f}) 🔴"
            score -= 1.5
        else:
            signals["Trend"] = f"緩やかな下落 ({trend_pct:.1f}%, R²={r2:.2f}) 🟠"
            score -= 0.5

    # --- 総合判定 ---
    if score >= 4:
        overall = "強い買い 🟢🟢"
    elif score >= 2:
        overall = "買い 🟢"
    elif score >= 0.5:
        overall = "やや買い 🔵"
    elif score >= -0.5:
        overall = "中立 ⚪"
    elif score >= -2:
        overall = "やや売り 🟠"
    elif score >= -4:
        overall = "売り 🔴"
    else:
        overall = "強い売り 🔴🔴"

    return {
        "signal": overall,
        "score": round(score, 2),
        "details": signals
    }


# =====================================================================
# データ取得
# =====================================================================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period="1y"):
    """個別銘柄の株価データ取得"""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_stock_info(ticker):
    """銘柄の基本情報取得"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_multiple_prices(tickers, period="1y"):
    """複数銘柄の終値を一括取得"""
    try:
        df = yf.download(tickers, period=period, progress=False, threads=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df[["Close"]]
            if len(tickers) == 1:
                close.columns = tickers
        return close
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_nvidia_news():
    """NVIDIA投資関連ニュースをRSSから取得"""
    feeds = [
        ("https://news.google.com/rss/search?q=NVIDIA+investment+acquisition&hl=en&gl=US&ceid=US:en", "NVIDIA Investment"),
        ("https://news.google.com/rss/search?q=NVIDIA+startup+funding+portfolio&hl=en&gl=US&ceid=US:en", "NVIDIA Startup"),
        ("https://news.google.com/rss/search?q=NVIDIA+partnership+deal&hl=en&gl=US&ceid=US:en", "NVIDIA Partnership"),
        ("https://news.google.com/rss/search?q=NVIDIA+エヌビディア+投資+出資&hl=ja&gl=JP&ceid=JP:ja", "NVIDIA投資(JP)"),
        ("https://news.google.com/rss/search?q=NVIDIA+AI+infrastructure+data+center&hl=en&gl=US&ceid=US:en", "NVIDIA AI Infra"),
    ]

    articles = []
    seen_titles = set()

    for url, source in feeds:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()

                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                dt_obj = None
                time_str = ""
                if pub_date:
                    try:
                        dt_obj = parsedate_to_datetime(pub_date)
                        time_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        time_str = pub_date[:16]

                articles.append({
                    "title": title,
                    "link": link,
                    "time": time_str,
                    "dt": dt_obj,
                    "source": source,
                })
        except Exception:
            continue

    articles.sort(key=lambda x: x["dt"] if x.get("dt") else datetime.min.replace(tzinfo=None), reverse=True)
    return articles[:50]

@st.cache_data(ttl=300)
def fetch_us_market_news():
    """米国株市場全般のニュース取得"""
    feeds = [
        ("https://news.google.com/rss/search?q=US+stock+market+S%26P500&hl=en&gl=US&ceid=US:en", "US Market"),
        ("https://news.google.com/rss/search?q=Fed+interest+rate+economy&hl=en&gl=US&ceid=US:en", "Fed Policy"),
        ("https://news.google.com/rss/search?q=semiconductor+AI+chip+stocks&hl=en&gl=US&ceid=US:en", "Semiconductor"),
        ("https://news.google.com/rss/search?q=米国株+ナスダック+S%26P500&hl=ja&gl=JP&ceid=JP:ja", "米国株(JP)"),
    ]

    articles = []
    seen_titles = set()

    for url, source in feeds:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()

                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                dt_obj = None
                time_str = ""
                if pub_date:
                    try:
                        dt_obj = parsedate_to_datetime(pub_date)
                        time_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        time_str = pub_date[:16]

                articles.append({
                    "title": title,
                    "link": link,
                    "time": time_str,
                    "dt": dt_obj,
                    "source": source,
                })
        except Exception:
            continue

    articles.sort(key=lambda x: x["dt"] if x.get("dt") else datetime.min.replace(tzinfo=None), reverse=True)
    return articles[:40]


# =====================================================================
# バリュエーション分析
# =====================================================================
def get_valuation_metrics(ticker):
    """バリュエーション指標を取得"""
    info = fetch_stock_info(ticker)
    if not info:
        return {}
    return {
        "ticker": ticker,
        "name": info.get("shortName", ticker),
        "price": info.get("currentPrice", info.get("regularMarketPrice", None)),
        "market_cap": info.get("marketCap", None),
        "pe_trailing": info.get("trailingPE", None),
        "pe_forward": info.get("forwardPE", None),
        "ps_ratio": info.get("priceToSalesTrailing12Months", None),
        "pb_ratio": info.get("priceToBook", None),
        "ev_ebitda": info.get("enterpriseToEbitda", None),
        "ev_revenue": info.get("enterpriseToRevenue", None),
        "profit_margin": info.get("profitMargins", None),
        "revenue_growth": info.get("revenueGrowth", None),
        "earnings_growth": info.get("earningsGrowth", None),
        "dividend_yield": info.get("dividendYield", None),
        "beta": info.get("beta", None),
        "52w_high": info.get("fiftyTwoWeekHigh", None),
        "52w_low": info.get("fiftyTwoWeekLow", None),
        "50d_ma": info.get("fiftyDayAverage", None),
        "200d_ma": info.get("twoHundredDayAverage", None),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
    }


# =====================================================================
# チャート描画
# =====================================================================
def plot_technical_chart(df, ticker, show_bb=True, show_ma=True):
    """テクニカルチャート（ローソク足 + 指標）"""
    if df is None or df.empty:
        return None

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[f"{ticker} Price", "RSI", "MACD", "Volume"]
    )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_price = df["Open"]

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index, open=open_price, high=high, low=low, close=close,
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # 移動平均線
    if show_ma:
        ma20 = calc_sma(close, 20)
        ma50 = calc_sma(close, 50)
        ma200 = calc_sma(close, 200)
        fig.add_trace(go.Scatter(x=df.index, y=ma20, name="MA20",
                                 line=dict(color="#ffeb3b", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma50, name="MA50",
                                 line=dict(color="#ff9800", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ma200, name="MA200",
                                 line=dict(color="#e91e63", width=2)), row=1, col=1)

    # ボリンジャーバンド
    if show_bb:
        upper, mid, lower = calc_bollinger(close)
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="BB Upper",
                                 line=dict(color="rgba(128,128,128,0.3)", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="BB Lower",
                                 line=dict(color="rgba(128,128,128,0.3)", dash="dash"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)

    # RSI
    rsi = calc_rsi(close)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
                             line=dict(color="#7c4dff", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    macd_line, signal_line, histogram = calc_macd(close)
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]
    fig.add_trace(go.Bar(x=df.index, y=histogram, name="MACD Hist",
                         marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                             line=dict(color="#2196f3", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal",
                             line=dict(color="#ff9800", width=1)), row=3, col=1)

    # 出来高
    if "Volume" in df.columns:
        vol_colors = ["#26a69a" if close.iloc[i] >= open_price.iloc[i] else "#ef5350"
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color=vol_colors, opacity=0.7), row=4, col=1)

    fig.update_layout(
        height=800, showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=30)
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    return fig


def plot_comparison_chart(tickers, period="1y"):
    """複数銘柄の正規化比較チャート"""
    prices = fetch_multiple_prices(tickers, period)
    if prices.empty:
        return None

    # 正規化（初日=100）
    normalized = prices / prices.iloc[0] * 100

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col],
            name=col, line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title="正規化パフォーマンス比較 (初日=100)",
        height=500, template="plotly_dark",
        yaxis_title="Normalized Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 100%; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px; padding: 16px; margin-bottom: 8px;
        border-left: 4px solid #0066ff;
    }
    .signal-badge {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-weight: bold; font-size: 0.85rem;
    }
    .nvidia-card {
        background: linear-gradient(135deg, #1a2e1a 0%, #0d3e16 100%);
        border-radius: 10px; padding: 16px; margin-bottom: 8px;
        border-left: 4px solid #76b900;
    }
    .news-item {
        padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    div[data-baseweb="tab-list"] {
        flex-wrap: wrap;
        gap: 2px;
    }
    div[data-baseweb="tab-list"] button {
        font-size: 0.85rem;
        padding: 6px 12px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# メインUI
# =====================================================================
st.title("🇺🇸 米国株投資分析ダッシュボード")
st.caption("中長期保有目線での売買タイミング分析 | NVIDIAエコシステム追跡 | バリュエーション比較")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    analysis_period = st.selectbox("分析期間", ["6mo", "1y", "2y", "5y"], index=1)
    st.divider()

    st.header("📋 ウォッチリスト管理")
    watchlist = load_watchlist()

    # 銘柄追加
    with st.expander("➕ 銘柄を追加", expanded=False):
        new_ticker = st.text_input("ティッカーシンボル", placeholder="例: AAPL, TSLA").upper().strip()
        new_notes = st.text_input("メモ（任意）", placeholder="例: AI関連")
        if st.button("追加", use_container_width=True):
            if new_ticker and new_ticker not in watchlist:
                info = fetch_stock_info(new_ticker)
                watchlist[new_ticker] = {
                    "name": info.get("shortName", new_ticker),
                    "sector": info.get("sector", ""),
                    "added": datetime.now().strftime("%Y-%m-%d"),
                    "notes": new_notes
                }
                save_watchlist(watchlist)
                st.success(f"✅ {new_ticker} を追加しました")
                st.rerun()
            elif new_ticker in watchlist:
                st.warning("既に登録済みです")

    # 銘柄削除
    if watchlist:
        with st.expander("🗑️ 銘柄を削除", expanded=False):
            del_ticker = st.selectbox("削除する銘柄", list(watchlist.keys()))
            if st.button("削除", use_container_width=True):
                del watchlist[del_ticker]
                save_watchlist(watchlist)
                st.success(f"🗑️ {del_ticker} を削除しました")
                st.rerun()

    st.divider()
    st.caption(f"ウォッチリスト: {len(watchlist)}銘柄")

# =====================================================================
# タブ構成
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ウォッチリスト概要",
    "📈 個別銘柄分析",
    "🎯 売買シグナル一覧",
    "🟢 NVIDIAエコシステム",
    "📰 NVIDIA投資ニュース",
    "📋 バリュエーション比較"
])

# =====================================================================
# Tab 1: ウォッチリスト概要
# =====================================================================
with tab1:
    st.subheader("📊 ウォッチリスト パフォーマンス概要")

    if not watchlist:
        st.info("サイドバーから銘柄を追加してください")
    else:
        tickers = list(watchlist.keys())

        with st.spinner("株価データ取得中..."):
            prices = fetch_multiple_prices(tickers, period=analysis_period)

        if not prices.empty:
            # パフォーマンスサマリー
            returns_1d = prices.pct_change().iloc[-1] if len(prices) > 1 else pd.Series()
            returns_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) > 5 else pd.Series()
            returns_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 21 else pd.Series()
            returns_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) > 63 else pd.Series()

            # テーブル表示
            summary_data = []
            for ticker in tickers:
                if ticker not in prices.columns:
                    continue
                row = {
                    "Ticker": ticker,
                    "銘柄名": watchlist[ticker].get("name", ""),
                    "現在値": f"${prices[ticker].iloc[-1]:.2f}" if not np.isnan(prices[ticker].iloc[-1]) else "N/A",
                    "日次": f"{returns_1d.get(ticker, 0)*100:+.2f}%" if ticker in returns_1d.index else "N/A",
                    "週次": f"{returns_1w.get(ticker, 0)*100:+.2f}%" if ticker in returns_1w.index else "N/A",
                    "月次": f"{returns_1m.get(ticker, 0)*100:+.2f}%" if ticker in returns_1m.index else "N/A",
                    "3ヶ月": f"{returns_3m.get(ticker, 0)*100:+.2f}%" if ticker in returns_3m.index else "N/A",
                    "セクター": watchlist[ticker].get("sector", ""),
                    "メモ": watchlist[ticker].get("notes", ""),
                }
                summary_data.append(row)

            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

            # 比較チャート
            st.subheader("📈 正規化パフォーマンス比較")
            selected_for_chart = st.multiselect(
                "比較する銘柄を選択",
                tickers,
                default=tickers[:5]
            )
            if selected_for_chart:
                fig = plot_comparison_chart(selected_for_chart, period=analysis_period)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # セクター別パフォーマンス
            st.subheader("🏷️ セクター別リターン")
            sector_returns = {}
            for ticker in tickers:
                sector = watchlist[ticker].get("sector", "Other")
                if sector not in sector_returns:
                    sector_returns[sector] = []
                if ticker in returns_1m.index:
                    sector_returns[sector].append(returns_1m[ticker])

            sector_avg = {k: np.mean(v) * 100 for k, v in sector_returns.items() if v}
            if sector_avg:
                fig_sector = go.Figure(go.Bar(
                    x=list(sector_avg.keys()),
                    y=list(sector_avg.values()),
                    marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in sector_avg.values()]
                ))
                fig_sector.update_layout(
                    title="セクター別 月次平均リターン (%)",
                    template="plotly_dark", height=350,
                    yaxis_title="Return (%)"
                )
                st.plotly_chart(fig_sector, use_container_width=True)


# =====================================================================
# Tab 2: 個別銘柄分析
# =====================================================================
with tab2:
    st.subheader("📈 個別銘柄テクニカル分析")

    watchlist = load_watchlist()
    all_tickers = list(watchlist.keys()) if watchlist else ["NVDA", "AAPL", "MSFT"]

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_ticker = st.selectbox("銘柄を選択", all_tickers, index=0)
    with col2:
        chart_period = st.selectbox("チャート期間", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="chart_period")

    if selected_ticker:
        with st.spinner(f"{selected_ticker} のデータ取得中..."):
            df = fetch_stock_data(selected_ticker, period=chart_period)
            info = fetch_stock_info(selected_ticker)

        if df is not None and not df.empty:
            # 基本情報
            col_a, col_b, col_c, col_d = st.columns(4)
            price = df["Close"].iloc[-1]
            prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
            change = (price - prev_price) / prev_price * 100

            col_a.metric("現在値", f"${price:.2f}", f"{change:+.2f}%")
            col_b.metric("52週高値", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            col_c.metric("52週安値", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
            col_d.metric("時価総額", f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A")

            # テクニカルチャート
            show_bb = st.checkbox("ボリンジャーバンド表示", value=True)
            show_ma = st.checkbox("移動平均線表示", value=True)

            fig = plot_technical_chart(df, selected_ticker, show_bb=show_bb, show_ma=show_ma)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 売買シグナル
            st.subheader("🎯 売買シグナル分析")
            signal_result = generate_trading_signals(df)
            st.markdown(f"### 総合判定: {signal_result['signal']} (スコア: {signal_result['score']})")

            if signal_result["details"]:
                for indicator, detail in signal_result["details"].items():
                    st.markdown(f"- **{indicator}**: {detail}")

            # 追加情報
            with st.expander("📊 詳細バリュエーション"):
                val_col1, val_col2, val_col3 = st.columns(3)
                val_col1.metric("PER (Forward)", f"{info.get('forwardPE', 'N/A'):.1f}x" if info.get('forwardPE') else "N/A")
                val_col1.metric("PER (Trailing)", f"{info.get('trailingPE', 'N/A'):.1f}x" if info.get('trailingPE') else "N/A")
                val_col2.metric("PSR", f"{info.get('priceToSalesTrailing12Months', 'N/A'):.1f}x" if info.get('priceToSalesTrailing12Months') else "N/A")
                val_col2.metric("PBR", f"{info.get('priceToBook', 'N/A'):.1f}x" if info.get('priceToBook') else "N/A")
                val_col3.metric("EV/EBITDA", f"{info.get('enterpriseToEbitda', 'N/A'):.1f}x" if info.get('enterpriseToEbitda') else "N/A")
                val_col3.metric("配当利回り", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
        else:
            st.error(f"{selected_ticker} のデータを取得できませんでした")

# =====================================================================
# Tab 3: 売買シグナル一覧
# =====================================================================
with tab3:
    st.subheader("🎯 全銘柄 売買シグナル一覧")
    st.caption("中長期保有目線: MA(50/200), RSI, MACD, ボリンジャーバンド, 出来高, OLSトレンドの複合判定")

    watchlist = load_watchlist()
    if not watchlist:
        st.info("サイドバーから銘柄を追加してください")
    else:
        signal_data = []
        progress = st.progress(0)

        for i, (ticker, meta) in enumerate(watchlist.items()):
            progress.progress((i + 1) / len(watchlist))
            df = fetch_stock_data(ticker, period="1y")
            result = generate_trading_signals(df)
            signal_data.append({
                "Ticker": ticker,
                "銘柄名": meta.get("name", ""),
                "総合判定": result["signal"],
                "スコア": result["score"],
                "セクター": meta.get("sector", ""),
            })

        progress.empty()

        df_signals = pd.DataFrame(signal_data)
        df_signals = df_signals.sort_values("スコア", ascending=False)

        # 色分け表示
        st.dataframe(df_signals, use_container_width=True, hide_index=True)

        # スコア分布
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 シグナルスコア分布")
            fig_dist = go.Figure(go.Bar(
                x=df_signals["Ticker"],
                y=df_signals["スコア"],
                marker_color=[
                    "#00c853" if s >= 3 else "#2196f3" if s >= 1 else
                    "#9e9e9e" if s >= -1 else "#ff9800" if s >= -3 else "#f44336"
                    for s in df_signals["スコア"]
                ],
                text=df_signals["スコア"].apply(lambda x: f"{x:.1f}"),
                textposition="outside"
            ))
            fig_dist.update_layout(template="plotly_dark", height=400, yaxis_title="Score")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.subheader("🏷️ シグナル分類")
            signal_counts = df_signals["総合判定"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                hole=0.4
            ))
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # 買いシグナル銘柄ハイライト
        buy_signals = df_signals[df_signals["スコア"] >= 2]
        if not buy_signals.empty:
            st.subheader("🟢 買いシグナル銘柄")
            st.dataframe(buy_signals, use_container_width=True, hide_index=True)

        sell_signals = df_signals[df_signals["スコア"] <= -2]
        if not sell_signals.empty:
            st.subheader("🔴 売りシグナル銘柄")
            st.dataframe(sell_signals, use_container_width=True, hide_index=True)


# =====================================================================
# Tab 4: NVIDIAエコシステム
# =====================================================================
with tab4:
    st.subheader("🟢 NVIDIAエコシステム & 投資先分析")
    st.caption("NVIDIAが出資・投資・買収した企業群のパフォーマンスとバリュエーション")

    nvidia_data = load_nvidia_investments()
    investments = nvidia_data.get("investments", [])
    ecosystem_tickers = nvidia_data.get("nvidia_ecosystem_tickers", [])

    # 投資先一覧テーブル
    st.subheader("📋 NVIDIA投資先一覧")
    inv_df = pd.DataFrame(investments)
    if not inv_df.empty:
        st.dataframe(
            inv_df[["company", "ticker", "category", "type", "year", "notes"]],
            use_container_width=True, hide_index=True,
            column_config={
                "company": "企業名",
                "ticker": "ティッカー",
                "category": "カテゴリ",
                "type": "投資タイプ",
                "year": "年",
                "notes": "備考"
            }
        )

    # 上場銘柄のパフォーマンス
    st.subheader("📈 上場投資先パフォーマンス")
    listed_tickers = [inv["ticker"] for inv in investments if inv.get("ticker")]
    listed_tickers = [t for t in listed_tickers if t]  # None除外

    if listed_tickers:
        # NVDAも含めて比較
        compare_tickers = ["NVDA"] + listed_tickers
        with st.spinner("投資先株価データ取得中..."):
            prices = fetch_multiple_prices(compare_tickers, period=analysis_period)

        if not prices.empty:
            # 正規化チャート
            normalized = prices / prices.iloc[0] * 100
            fig = go.Figure()
            colors = ["#76b900"] + list(px.colors.qualitative.Set2)
            for i, col in enumerate(normalized.columns):
                width = 3 if col == "NVDA" else 1.5
                fig.add_trace(go.Scatter(
                    x=normalized.index, y=normalized[col],
                    name=col, line=dict(color=colors[i % len(colors)], width=width)
                ))
            fig.update_layout(
                title="NVIDIA vs 投資先 正規化パフォーマンス (初日=100)",
                height=500, template="plotly_dark",
                yaxis_title="Normalized Price",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # リターン比較テーブル
            st.subheader("📊 リターン比較")
            ret_data = []
            for ticker in compare_tickers:
                if ticker not in prices.columns:
                    continue
                p = prices[ticker].dropna()
                if len(p) < 2:
                    continue
                ret_1w = (p.iloc[-1] / p.iloc[-5] - 1) * 100 if len(p) > 5 else 0
                ret_1m = (p.iloc[-1] / p.iloc[-21] - 1) * 100 if len(p) > 21 else 0
                ret_3m = (p.iloc[-1] / p.iloc[-63] - 1) * 100 if len(p) > 63 else 0
                ret_data.append({
                    "Ticker": ticker,
                    "現在値": f"${p.iloc[-1]:.2f}",
                    "週次": f"{ret_1w:+.1f}%",
                    "月次": f"{ret_1m:+.1f}%",
                    "3ヶ月": f"{ret_3m:+.1f}%",
                })
            if ret_data:
                st.dataframe(pd.DataFrame(ret_data), use_container_width=True, hide_index=True)

    # エコシステム全体
    st.subheader("🌐 NVIDIAエコシステム銘柄")
    st.caption("NVIDIA関連サプライチェーン・パートナー企業")

    eco_col1, eco_col2 = st.columns(2)
    with eco_col1:
        selected_eco = st.multiselect(
            "比較銘柄を選択",
            ecosystem_tickers,
            default=ecosystem_tickers[:6]
        )
    with eco_col2:
        eco_period = st.selectbox("期間", ["3mo", "6mo", "1y"], index=1, key="eco_period")

    if selected_eco:
        fig_eco = plot_comparison_chart(selected_eco, period=eco_period)
        if fig_eco:
            fig_eco.update_layout(title="NVIDIAエコシステム銘柄 パフォーマンス比較")
            st.plotly_chart(fig_eco, use_container_width=True)

# =====================================================================
# Tab 5: NVIDIAニュース
# =====================================================================
with tab5:
    st.subheader("📰 NVIDIA投資・出資関連ニュース")
    st.caption("NVIDIAの新規投資・出資・買収・パートナーシップに関する最新ニュース")

    with st.spinner("ニュース取得中..."):
        nvidia_news = fetch_nvidia_news()

    if nvidia_news:
        # フィルタ
        news_filter = st.text_input("🔍 キーワードフィルタ", placeholder="例: investment, acquisition, AI")

        filtered_news = nvidia_news
        if news_filter:
            filtered_news = [n for n in nvidia_news if news_filter.lower() in n["title"].lower()]

        st.markdown(f"**{len(filtered_news)}件のニュース** (最新50件)")
        st.divider()

        for article in filtered_news[:30]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"[{article['title']}]({article['link']})")
            with col2:
                st.caption(article["time"])
            st.caption(f"Source: {article['source']}")
            st.divider()
    else:
        st.warning("ニュースを取得できませんでした")

    # 市場全般ニュース
    st.subheader("📰 米国株市場ニュース")
    with st.spinner("市場ニュース取得中..."):
        market_news = fetch_us_market_news()

    if market_news:
        for article in market_news[:15]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"[{article['title']}]({article['link']})")
            with col2:
                st.caption(article["time"])
            st.divider()


# =====================================================================
# Tab 6: バリュエーション比較
# =====================================================================
with tab6:
    st.subheader("📋 バリュエーション比較分析")
    st.caption("NVIDIA関連銘柄 & ウォッチリスト銘柄のバリュエーション指標比較")

    nvidia_data = load_nvidia_investments()
    watchlist = load_watchlist()

    # 比較対象選択
    val_source = st.radio(
        "比較対象",
        ["NVIDIAエコシステム", "ウォッチリスト", "カスタム"],
        horizontal=True
    )

    if val_source == "NVIDIAエコシステム":
        val_tickers = nvidia_data.get("nvidia_ecosystem_tickers", [])[:12]
    elif val_source == "ウォッチリスト":
        val_tickers = list(watchlist.keys())[:15]
    else:
        custom_input = st.text_input("ティッカーをカンマ区切りで入力", "NVDA, AMD, AVGO, TSM, SMCI, ARM")
        val_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

    if val_tickers:
        with st.spinner("バリュエーションデータ取得中..."):
            val_data = []
            for ticker in val_tickers:
                metrics = get_valuation_metrics(ticker)
                if metrics:
                    val_data.append(metrics)
                time.sleep(0.2)  # API制限対策

        if val_data:
            df_val = pd.DataFrame(val_data)

            # メイン指標テーブル
            display_cols = {
                "ticker": "Ticker",
                "name": "銘柄名",
                "price": "株価($)",
                "market_cap": "時価総額",
                "pe_forward": "PER(予想)",
                "ps_ratio": "PSR",
                "pb_ratio": "PBR",
                "ev_ebitda": "EV/EBITDA",
                "revenue_growth": "売上成長率",
                "profit_margin": "利益率",
                "beta": "β値",
            }

            df_display = df_val[[c for c in display_cols.keys() if c in df_val.columns]].copy()
            df_display.columns = [display_cols.get(c, c) for c in df_display.columns]

            # フォーマット
            if "時価総額" in df_display.columns:
                df_display["時価総額"] = df_display["時価総額"].apply(
                    lambda x: f"${x/1e9:.1f}B" if x and x > 1e9 else f"${x/1e6:.0f}M" if x else "N/A"
                )
            if "売上成長率" in df_display.columns:
                df_display["売上成長率"] = df_display["売上成長率"].apply(
                    lambda x: f"{x*100:.1f}%" if x else "N/A"
                )
            if "利益率" in df_display.columns:
                df_display["利益率"] = df_display["利益率"].apply(
                    lambda x: f"{x*100:.1f}%" if x else "N/A"
                )

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # バリュエーション比較チャート
            st.subheader("📊 バリュエーション比較チャート")

            chart_metric = st.selectbox(
                "比較指標",
                ["PER(予想)", "PSR", "PBR", "EV/EBITDA"],
                index=0
            )

            metric_map = {"PER(予想)": "pe_forward", "PSR": "ps_ratio", "PBR": "pb_ratio", "EV/EBITDA": "ev_ebitda"}
            metric_col = metric_map[chart_metric]

            chart_data = df_val[["ticker", metric_col]].dropna()
            if not chart_data.empty:
                chart_data = chart_data.sort_values(metric_col, ascending=False)
                fig_val = go.Figure(go.Bar(
                    x=chart_data["ticker"],
                    y=chart_data[metric_col],
                    marker_color=px.colors.qualitative.Set2[:len(chart_data)],
                    text=chart_data[metric_col].apply(lambda x: f"{x:.1f}x"),
                    textposition="outside"
                ))
                fig_val.update_layout(
                    title=f"{chart_metric} 比較",
                    template="plotly_dark", height=400,
                    yaxis_title=chart_metric
                )
                st.plotly_chart(fig_val, use_container_width=True)

            # 割安/割高判定
            st.subheader("💡 バリュエーション判定")
            for _, row in df_val.iterrows():
                ticker = row["ticker"]
                pe = row.get("pe_forward")
                ps = row.get("ps_ratio")
                growth = row.get("revenue_growth")

                if pe and growth and growth > 0:
                    peg = pe / (growth * 100)
                    if peg < 1:
                        st.markdown(f"🟢 **{ticker}**: PEG={peg:.2f} → 成長対比で割安")
                    elif peg < 2:
                        st.markdown(f"⚪ **{ticker}**: PEG={peg:.2f} → 適正水準")
                    else:
                        st.markdown(f"🔴 **{ticker}**: PEG={peg:.2f} → 成長対比で割高")
                elif pe:
                    if pe < 15:
                        st.markdown(f"🟢 **{ticker}**: PER={pe:.1f}x → バリュー圏")
                    elif pe < 30:
                        st.markdown(f"⚪ **{ticker}**: PER={pe:.1f}x → 適正水準")
                    else:
                        st.markdown(f"🟠 **{ticker}**: PER={pe:.1f}x → グロース・プレミアム")

            # 52週高値/安値からの位置
            st.subheader("📏 52週レンジ内の位置")
            range_data = []
            for _, row in df_val.iterrows():
                high = row.get("52w_high")
                low = row.get("52w_low")
                price = row.get("price")
                if high and low and price and high > low:
                    position = (price - low) / (high - low) * 100
                    range_data.append({
                        "Ticker": row["ticker"],
                        "現在値": price,
                        "52週安値": low,
                        "52週高値": high,
                        "位置(%)": position
                    })

            if range_data:
                df_range = pd.DataFrame(range_data)
                fig_range = go.Figure()
                for _, r in df_range.iterrows():
                    color = "#26a69a" if r["位置(%)"] < 40 else "#ff9800" if r["位置(%)"] < 70 else "#ef5350"
                    fig_range.add_trace(go.Bar(
                        x=[r["Ticker"]], y=[r["位置(%)"]],
                        marker_color=color,
                        text=f"{r['位置(%)']:.0f}%",
                        textposition="outside",
                        showlegend=False
                    ))
                fig_range.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3)
                fig_range.update_layout(
                    title="52週レンジ内の位置 (0%=安値, 100%=高値)",
                    template="plotly_dark", height=400,
                    yaxis_title="Position (%)", yaxis_range=[0, 110]
                )
                st.plotly_chart(fig_range, use_container_width=True)
                st.caption("🟢 40%以下: 安値圏 | 🟠 40-70%: 中間 | 🔴 70%以上: 高値圏")

# =====================================================================
# フッター
# =====================================================================
st.divider()
st.caption("⚠️ 本ダッシュボードは情報提供のみを目的としています。投資判断は自己責任でお願いします。")
st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} | データソース: Yahoo Finance, Google News RSS")
