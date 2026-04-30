import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
import time
import re

st.set_page_config(
    page_title="マーケット分析ダッシュボード",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 100%; }
    div[data-testid="stMetric"] {
        background: #1a1a2e; border-radius: 8px; padding: 12px;
        border-left: 4px solid #0066ff;
    }
    div[data-testid="stMetric"] label {
        font-size: 1rem !important; color: #ffffff !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important; color: #ffffff !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.95rem !important;
    }
    .tdnet-row {
        padding: 8px 12px; border-radius: 6px; margin-bottom: 4px;
        border-left: 3px solid #0066ff; background: #1a1a2e;
    }
    .tdnet-row:hover { background: #252545; }
    .tdnet-title { color: #fff; font-size: 0.95rem; }
    .tdnet-meta { color: #888; font-size: 0.8rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

TICKERS = {
    "日経225": "^N225",
    "TOPIX ETF": "1306.T",
    "ドル/円": "USDJPY=X",
    "米10年債利回り": "^TNX",
    "VIX指数": "^VIX",
    "FF金利先物": "ZQ=F",
    "原油 (WTI)": "CL=F",
    "金 (Gold)": "GC=F",
    "銅 (Copper)": "HG=F",
}

CATEGORIES = {
    "📊 株価指数・為替": ["日経225", "TOPIX ETF", "ドル/円"],
    "⚠️ リスク・金利指標": ["米10年債利回り", "VIX指数", "FF金利先物"],
    "🛢️ コモディティ": ["原油 (WTI)", "金 (Gold)", "銅 (Copper)"],
}


@st.cache_data(ttl=120)
def fetch(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


@st.cache_data(ttl=180)
def fetch_tdnet(target_date_str):
    date_fmt = target_date_str.replace("-", "")
    url = f"https://www.release.tdnet.info/inbs/I_list_001_{date_fmt}.html"
    try:
        resp = requests.get(url, timeout=15)
        resp.encoding = resp.apparent_encoding
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text("\n")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        disclosures = []
        i = 0
        while i < len(lines):
            if re.match(r"^\d{2}:\d{2}$", lines[i]):
                t = lines[i]
                code = lines[i+1] if i+1 < len(lines) else ""
                company = lines[i+2] if i+2 < len(lines) else ""
                title = lines[i+3] if i+3 < len(lines) else ""
                disclosures.append({
                    "time": t, "code": code.strip(),
                    "company": company.strip(), "title": title.strip(),
                })
                i += 4
            else:
                i += 1
        return disclosures
    except Exception:
        return []

def make_chart(df, name, height=500):
    if df is None or df.empty:
        return None
    fig = go.Figure()
    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=name,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            name=name, line=dict(color="#26a69a", width=2),
            fill="tozeroy", fillcolor="rgba(38,166,154,0.1)",
        ))
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[0])
    pct = (last - prev) / prev * 100 if prev else 0
    color = "#26a69a" if pct >= 0 else "#ef5350"
    fig.update_layout(
        title=dict(
            text=f"{name}　<span style='color:{color}'>{last:,.2f}（{pct:+.2f}%）</span>",
            font=dict(size=16),
        ),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        showlegend=False,
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
    )
    return fig


# ===== メイン =====
st.title("📈 マーケット分析ダッシュボード")

# クイックリンク
st.markdown("""
<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
    <a href="https://shikiho.toyokeizai.net/" target="_blank" style="padding:4px 12px; background:#1a1a2e; border-radius:4px; color:#4da6ff; text-decoration:none; font-size:0.82rem; border:1px solid #333;">四季報</a>
    <a href="https://kabutan.jp/" target="_blank" style="padding:4px 12px; background:#1a1a2e; border-radius:4px; color:#4da6ff; text-decoration:none; font-size:0.82rem; border:1px solid #333;">株探</a>
    <a href="https://kabubiz.com/riron/" target="_blank" style="padding:4px 12px; background:#1a1a2e; border-radius:4px; color:#4da6ff; text-decoration:none; font-size:0.82rem; border:1px solid #333;">株Biz理論株価</a>
    <a href="https://www.monex.co.jp/" target="_blank" style="padding:4px 12px; background:#1a1a2e; border-radius:4px; color:#4da6ff; text-decoration:none; font-size:0.82rem; border:1px solid #333;">マネックス</a>
    <a href="https://www.sbisec.co.jp/ETGate" target="_blank" style="padding:4px 12px; background:#1a1a2e; border-radius:4px; color:#4da6ff; text-decoration:none; font-size:0.82rem; border:1px solid #333;">SBI証券</a>
    <a href="https://www.matsui.co.jp/" target="_blank" style="padding:4px 12px; background:#1a1a2e; border-radius:4px; color:#4da6ff; text-decoration:none; font-size:0.82rem; border:1px solid #333;">松井証券</a>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 設定")
    auto_refresh = st.toggle("自動更新（2分）", value=False)
    chart_period = st.selectbox("期間", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval_map = {"1d": "5m", "5d": "5m", "1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1wk"}
    interval = interval_map[chart_period]

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"🕐 最終更新: {now_str}　｜　期間: {chart_period} / {interval}足")

for cat_name, keys in CATEGORIES.items():
    st.subheader(cat_name)
    mcols = st.columns(len(keys))
    data_map = {}
    for i, key in enumerate(keys):
        df = fetch(TICKERS[key], chart_period, interval)
        data_map[key] = df
        with mcols[i]:
            if df is not None and not df.empty:
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[0])
                chg = last - prev
                pct = (chg / prev * 100) if prev else 0
                st.metric(key, f"{last:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            else:
                st.metric(key, "取得失敗", "N/A")
    ccols = st.columns(len(keys))
    for i, key in enumerate(keys):
        with ccols[i]:
            fig = make_chart(data_map[key], key, height=480)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{key}: データ取得失敗")
    st.divider()

# --- 相関ヒートマップ ---
st.subheader("🔗 銘柄間相関")
try:
    closes = {}
    for name, sym in TICKERS.items():
        df = fetch(sym, "3mo", "1d")
        if df is not None and not df.empty:
            closes[name] = df["Close"]
    if closes:
        combined = pd.DataFrame(closes).dropna()
        corr = combined.pct_change().dropna().corr()
        fig_h = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=12),
        ))
        fig_h.update_layout(
            height=500, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_h, use_container_width=True)
except Exception as e:
    st.warning(f"相関計算エラー: {e}")

st.divider()

# --- TDnet 適時開示 ---
st.subheader("📋 TDnet 適時開示情報")
tdnet_date = st.date_input("開示日", value=date.today())
target_str = tdnet_date.strftime("%Y-%m-%d")
search_kw = st.text_input("🔍 キーワード検索（企業名・開示内容）", "")

with st.spinner("TDnet から取得中..."):
    disclosures = fetch_tdnet(target_str)

if disclosures:
    filtered = disclosures
    if search_kw:
        kw = search_kw.lower()
        filtered = [d for d in filtered if
                    kw in d["company"].lower() or
                    kw in d["title"].lower() or
                    kw in d["code"].lower()]
    st.caption(f"📄 {len(filtered)} 件表示（全 {len(disclosures)} 件）")
    for d in filtered[:200]:
        st.markdown(f"""
        <div class="tdnet-row">
            <div class="tdnet-title">📄 {d['company']}（{d['code']}）— {d['title']}</div>
            <div class="tdnet-meta">🕐 {d['time']}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info(f"{target_str} の開示情報はありません（休日または未公開）")

# 自動更新
if auto_refresh:
    time.sleep(120)
    st.rerun()
