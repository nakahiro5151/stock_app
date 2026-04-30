"""
TOPIX新基準スクリーニング ダッシュボード
- 次世代TOPIX選定基準に基づく銘柄スクリーニング
- 年間売買回転率 ≥ 0.14
- 累積浮動株時価総額上位97%
- 配当利回り・PER・ROE併記
- 第1回定期見直し: 2026年10月 / 第2回: 2028年10月
参考: https://www.jpx.co.jp/english/markets/indices/revisions-indices/02.html
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import json
import os
import time
from datetime import datetime, timedelta
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TOPIX新基準スクリーニング",
    page_icon="📊",
    layout="wide",
)

# ── 定数 ──
CACHE_DIR = "topix_cache"
STOCK_LIST_CACHE = os.path.join(CACHE_DIR, "stock_list.json")
SCREENING_CACHE = os.path.join(CACHE_DIR, "screening_data.json")
CACHE_TTL = 86400  # 24h

# 次世代TOPIX基準
TRADED_VALUE_RATIO_THRESHOLD = 0.14
CUMULATIVE_MCAP_PCT = 0.97

os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# データ取得
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_stock_list_from_jpx():
    """JPX上場銘柄一覧を取得し、市場区分・業種付きで返す"""
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        return pd.DataFrame()
    df = pd.read_excel(io.BytesIO(resp.content))
    # カラム名を正規化
    col_map = {}
    for c in df.columns:
        s = str(c)
        if "コード" in s:
            col_map[c] = "code"
        elif "銘柄名" in s:
            col_map[c] = "name"
        elif "市場・商品区分" in s:
            col_map[c] = "market"
        elif "33業種" in s and "コード" not in s:
            col_map[c] = "sector"
        elif "33業種" in s and "コード" in s:
            col_map[c] = "sector_code"
        elif "規模" in s:
            col_map[c] = "size"
    df = df.rename(columns=col_map)
    needed = ["code", "name", "market", "sector"]
    for n in needed:
        if n not in df.columns:
            df[n] = ""
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"].str.match(r"^\d{4}$")]
    df["ticker"] = df["code"] + ".T"
    return df[["ticker", "code", "name", "market", "sector"]].reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_data_batch(tickers_tuple, period="1y"):
    """yfinanceでバッチ取得: 価格・出来高"""
    tickers = list(tickers_tuple)
    if not tickers:
        return {}
    results = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
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
    return results


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_fundamentals_batch(tickers_tuple):
    """yfinanceで基本情報をバッチ取得: 時価総額, 浮動株, 配当利回り, PER, ROE"""
    tickers = list(tickers_tuple)
    results = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            results[t] = {
                "marketCap": info.get("marketCap", None),
                "floatShares": info.get("floatShares", None),
                "sharesOutstanding": info.get("sharesOutstanding", None),
                "dividendYield": info.get("dividendYield", None),
                "trailingPE": info.get("trailingPE", None),
                "forwardPE": info.get("forwardPE", None),
                "returnOnEquity": info.get("returnOnEquity", None),
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice", None)),
            }
        except Exception:
            results[t] = {}
    return results


# ═══════════════════════════════════════════════════════════════
# スクリーニング計算
# ═══════════════════════════════════════════════════════════════

def calc_traded_value_ratio(close_series, volume_series):
    """
    年間売買回転率を計算
    月次売買回転率 = (月中日次売買代金の中央値 × 営業日数) / 月末浮動株時価総額
    年間 = 直近12ヶ月の月次売買回転率の合計
    ※ 浮動株時価総額が不明な場合は時価総額で代用
    """
    if close_series is None or len(close_series) < 60:
        return None
    df = pd.DataFrame({"close": close_series, "volume": volume_series})
    df["traded_value"] = df["close"] * df["volume"]
    df["month"] = df.index.to_period("M")

    monthly_ratios = []
    for month, grp in df.groupby("month"):
        if len(grp) < 5:
            continue
        median_daily_tv = grp["traded_value"].median()
        biz_days = len(grp)
        month_end_mcap = grp["close"].iloc[-1]  # 株価（後で株数を掛ける）
        if month_end_mcap > 0:
            # 売買代金 / 時価総額 の代理指標
            monthly_ratio = (median_daily_tv * biz_days) / (month_end_mcap * 1)
            monthly_ratios.append(monthly_ratio)

    if len(monthly_ratios) < 6:
        return None
    return sum(monthly_ratios[-12:])


def calc_annual_turnover_ratio(close_series, volume_series, float_shares=None, shares_outstanding=None):
    """
    年間売買回転率（改良版）
    = Σ(月次: 日次売買代金中央値 × 営業日数 / 月末浮動株時価総額)
    """
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
            ratio = (median_daily_tv * biz_days) / ff_mcap
            monthly_ratios.append(ratio)

    if len(monthly_ratios) < 6:
        return None
    return sum(monthly_ratios[-12:])


def run_screening(stock_df, market_data, fundamentals):
    """スクリーニング実行: 全銘柄に対して基準判定"""
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

        # 浮動株時価総額
        if float_shares and price:
            ff_mcap = float_shares * price
        elif mcap:
            ff_mcap = mcap * 0.7  # 概算: 浮動株比率70%と仮定
        else:
            ff_mcap = None

        # 年間売買回転率
        turnover = calc_annual_turnover_ratio(close, volume, float_shares, shares_out)

        # ファンダメンタルズ
        div_yield = fund.get("dividendYield")
        if div_yield is not None:
            div_yield = round(div_yield * 100, 2)

        pe = fund.get("trailingPE")
        if pe is None:
            pe = fund.get("forwardPE")

        roe = fund.get("returnOnEquity")
        if roe is not None:
            roe = round(roe * 100, 2)

        rows.append({
            "銘柄コード": row["code"],
            "銘柄名": row["name"],
            "市場区分": row["market"],
            "業種": row["sector"],
            "株価": round(price, 1) if price else None,
            "時価総額(億円)": round(mcap / 1e8, 1) if mcap else None,
            "浮動株時価総額(億円)": round(ff_mcap / 1e8, 1) if ff_mcap else None,
            "年間売買回転率": round(turnover, 4) if turnover else None,
            "配当利回り(%)": div_yield,
            "PER(倍)": round(pe, 2) if pe else None,
            "ROE(%)": roe,
            "ticker": t,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # 累積浮動株時価総額比率の計算
    valid = result.dropna(subset=["浮動株時価総額(億円)", "年間売買回転率"])
    valid = valid[valid["年間売買回転率"] >= TRADED_VALUE_RATIO_THRESHOLD]
    valid = valid.sort_values("浮動株時価総額(億円)", ascending=False)
    total_ff_mcap = valid["浮動株時価総額(億円)"].sum()
    if total_ff_mcap > 0:
        valid["累積時価総額比率"] = valid["浮動株時価総額(億円)"].cumsum() / total_ff_mcap
    else:
        valid["累積時価総額比率"] = None

    # 結果をマージ
    result = result.merge(
        valid[["ticker", "累積時価総額比率"]],
        on="ticker", how="left"
    )

    # TOPIX判定
    def judge(row):
        tr = row.get("年間売買回転率")
        cum = row.get("累積時価総額比率")
        if tr is None or pd.isna(tr):
            return "データ不足"
        if tr < TRADED_VALUE_RATIO_THRESHOLD:
            return "❌ 回転率不足"
        if cum is not None and not pd.isna(cum) and cum <= CUMULATIVE_MCAP_PCT:
            return "✅ 採用圏内"
        elif cum is not None and not pd.isna(cum) and cum <= 0.99:
            return "⚠️ ボーダー"
        else:
            return "❌ 圏外"

    result["TOPIX判定"] = result.apply(judge, axis=1)
    return result.sort_values("浮動株時価総額(億円)", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

def main():
    st.title("📊 TOPIX 新基準スクリーニング")
    st.caption(
        "次世代TOPIX（第2段階見直し）の選定基準に基づくスクリーニング　"
        "| 第1回定期見直し: 2026年10月 | 第2回: 2028年10月"
    )

    # ── 基準説明 ──
    with st.expander("📋 次世代TOPIX 選定基準", expanded=False):
        st.markdown("""
        **対象ユニバース**: 全市場区分（プライム・スタンダード・グロース）

        **選定基準**:
        1. **年間売買回転率** ≥ 0.14
           - 月次売買回転率 = (月中日次売買代金の中央値 × 営業日数) / 月末浮動株時価総額
           - 年間 = 直近12ヶ月の月次売買回転率の合計
        2. **累積浮動株時価総額** 上位97%以内
           - 回転率基準を満たす銘柄を浮動株時価総額順に並べ、累積97%以内

        **経過措置**: 第1回見直しで非選定となった銘柄は四半期ごとに8段階でウエイト低減

        [JPX公式資料](https://www.jpx.co.jp/english/markets/indices/revisions-indices/02.html)
        """)

    # ── サイドバー ──
    st.sidebar.header("🔧 フィルター設定")

    # 銘柄数制限（パフォーマンス対策）
    max_stocks = st.sidebar.selectbox(
        "スクリーニング対象数",
        [50, 100, 200, 500, 1000, 2000],
        index=2,
        help="多いほど時間がかかります（初回は特に）"
    )

    market_filter = st.sidebar.multiselect(
        "市場区分",
        ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"],
        default=["プライム（内国株式）"]
    )

    sort_by = st.sidebar.selectbox(
        "ソート基準",
        ["浮動株時価総額(億円)", "年間売買回転率", "配当利回り(%)", "PER(倍)", "ROE(%)"],
        index=0
    )
    sort_asc = st.sidebar.checkbox("昇順", value=False)

    topix_filter = st.sidebar.multiselect(
        "TOPIX判定フィルター",
        ["✅ 採用圏内", "⚠️ ボーダー", "❌ 回転率不足", "❌ 圏外", "データ不足"],
        default=["✅ 採用圏内", "⚠️ ボーダー"]
    )

    # 追加フィルター
    st.sidebar.subheader("ファンダメンタルズ")
    min_div = st.sidebar.number_input("配当利回り下限(%)", value=0.0, step=0.5)
    max_per = st.sidebar.number_input("PER上限(倍)", value=100.0, step=5.0)
    min_roe = st.sidebar.number_input("ROE下限(%)", value=0.0, step=1.0)

    # ── データ取得 ──
    if st.sidebar.button("🚀 スクリーニング実行", type="primary", use_container_width=True):
        with st.spinner("JPXから銘柄リストを取得中..."):
            stock_df = fetch_stock_list_from_jpx()

        if stock_df.empty:
            st.error("銘柄リストの取得に失敗しました")
            return

        # 市場フィルター適用
        if market_filter:
            stock_df = stock_df[stock_df["market"].isin(market_filter)]

        stock_df = stock_df.head(max_stocks)
        total = len(stock_df)
        st.info(f"対象銘柄数: {total}件")

        tickers = tuple(stock_df["ticker"].tolist())

        # 価格データ取得
        progress = st.progress(0, text="価格データ取得中...")
        market_data = fetch_market_data_batch(tickers, period="1y")
        progress.progress(50, text="ファンダメンタルズ取得中...")

        # ファンダメンタルズ取得（時間がかかる）
        fundamentals = fetch_fundamentals_batch(tickers)
        progress.progress(80, text="スクリーニング計算中...")

        # スクリーニング実行
        result = run_screening(stock_df, market_data, fundamentals)
        progress.progress(100, text="完了")
        time.sleep(0.3)
        progress.empty()

        if result.empty:
            st.warning("スクリーニング結果がありません")
            return

        # セッションに保存
        st.session_state["screening_result"] = result

    # ── 結果表示 ──
    if "screening_result" not in st.session_state:
        st.info("👈 サイドバーから「スクリーニング実行」を押してください")
        return

    result = st.session_state["screening_result"]

    # フィルター適用
    filtered = result.copy()
    if topix_filter:
        filtered = filtered[filtered["TOPIX判定"].isin(topix_filter)]
    if min_div > 0:
        filtered = filtered[filtered["配当利回り(%)"].fillna(0) >= min_div]
    if max_per < 100:
        filtered = filtered[filtered["PER(倍)"].fillna(999) <= max_per]
    if min_roe > 0:
        filtered = filtered[filtered["ROE(%)"].fillna(0) >= min_roe]

    # ソート
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=sort_asc, na_position="last")

    # ── サマリー ──
    col1, col2, col3, col4, col5 = st.columns(5)
    total_count = len(result)
    pass_count = len(result[result["TOPIX判定"] == "✅ 採用圏内"])
    border_count = len(result[result["TOPIX判定"] == "⚠️ ボーダー"])
    fail_count = len(result[result["TOPIX判定"].str.contains("❌", na=False)])
    col1.metric("対象銘柄", f"{total_count}")
    col2.metric("✅ 採用圏内", f"{pass_count}")
    col3.metric("⚠️ ボーダー", f"{border_count}")
    col4.metric("❌ 圏外/不足", f"{fail_count}")
    col5.metric("表示中", f"{len(filtered)}")

    st.divider()

    # ── 表示モード切替 ──
    view_mode = st.radio(
        "表示モード", ["📋 テーブル", "🃏 カード"],
        horizontal=True, label_visibility="collapsed"
    )

    if view_mode == "📋 テーブル":
        # ── テーブル表示 ──
        display_cols = [
            "銘柄コード", "銘柄名", "市場区分", "業種", "株価",
            "時価総額(億円)", "浮動株時価総額(億円)", "年間売買回転率",
            "累積時価総額比率", "TOPIX判定",
            "配当利回り(%)", "PER(倍)", "ROE(%)"
        ]
        display_df = filtered[[c for c in display_cols if c in filtered.columns]]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config={
                "株価": st.column_config.NumberColumn(format="¥%.0f"),
                "時価総額(億円)": st.column_config.NumberColumn(format="%.0f"),
                "浮動株時価総額(億円)": st.column_config.NumberColumn(format="%.0f"),
                "年間売買回転率": st.column_config.NumberColumn(format="%.4f"),
                "累積時価総額比率": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.2f"
                ),
                "配当利回り(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "PER(倍)": st.column_config.NumberColumn(format="%.1f"),
                "ROE(%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    else:
        # ── カード表示（複数行レイアウト） ──
        page_size = 20
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input("ページ", min_value=1, max_value=total_pages, value=1, step=1)
        page_df = filtered.iloc[(page - 1) * page_size : page * page_size]
        st.caption(f"{len(filtered)}件中 {(page-1)*page_size+1}〜{min(page*page_size, len(filtered))}件")

        for _, r in page_df.iterrows():
            judge = r.get("TOPIX判定", "")
            if "✅" in str(judge):
                border_color = "#2ecc71"
            elif "⚠️" in str(judge):
                border_color = "#f39c12"
            elif "❌" in str(judge):
                border_color = "#e74c3c"
            else:
                border_color = "#95a5a6"

            with st.container():
                st.markdown(
                    f"""<div style="border-left:4px solid {border_color};
                    padding:8px 12px; margin-bottom:8px;
                    background:#fafafa; border-radius:4px;">
                    </div>""",
                    unsafe_allow_html=True,
                )
                # 1行目: 銘柄基本情報
                c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
                c1.markdown(f"**{r.get('銘柄コード', '')}**")
                c2.markdown(f"**{r.get('銘柄名', '')}**")
                c3.markdown(f"🏷️ {r.get('市場区分', '')}")
                c4.markdown(f"{judge}")

                # 2行目: 時価総額・回転率
                c5, c6, c7, c8 = st.columns(4)
                price = r.get("株価")
                c5.metric("株価", f"¥{price:,.0f}" if pd.notna(price) else "—", label_visibility="visible")
                mcap = r.get("時価総額(億円)")
                c6.metric("時価総額", f"{mcap:,.0f}億" if pd.notna(mcap) else "—")
                ff = r.get("浮動株時価総額(億円)")
                c7.metric("浮動株時価総額", f"{ff:,.0f}億" if pd.notna(ff) else "—")
                tr = r.get("年間売買回転率")
                c8.metric("売買回転率", f"{tr:.4f}" if pd.notna(tr) else "—")

                # 3行目: ファンダメンタルズ
                c9, c10, c11, c12 = st.columns(4)
                dy = r.get("配当利回り(%)")
                c9.metric("配当利回り", f"{dy:.2f}%" if pd.notna(dy) else "—")
                pe = r.get("PER(倍)")
                c10.metric("PER", f"{pe:.1f}倍" if pd.notna(pe) else "—")
                roe = r.get("ROE(%)")
                c11.metric("ROE", f"{roe:.1f}%" if pd.notna(roe) else "—")
                cum = r.get("累積時価総額比率")
                c12.metric("累積比率", f"{cum:.2%}" if pd.notna(cum) else "—")

                st.markdown("---")

    # ── CSV ダウンロード ──
    display_cols = [
        "銘柄コード", "銘柄名", "市場区分", "業種", "株価",
        "時価総額(億円)", "浮動株時価総額(億円)", "年間売買回転率",
        "累積時価総額比率", "TOPIX判定",
        "配当利回り(%)", "PER(倍)", "ROE(%)"
    ]
    display_df = filtered[[c for c in display_cols if c in filtered.columns]]
    csv = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 CSV ダウンロード",
        csv,
        f"topix_screening_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


    # ── 可視化 ──
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 分布チャート", "🏭 業種別分析", "📈 ボーダーライン銘柄"])

    with tab1:
        valid_data = filtered.dropna(subset=["年間売買回転率", "浮動株時価総額(億円)"])
        if not valid_data.empty:
            fig = px.scatter(
                valid_data,
                x="年間売買回転率",
                y="浮動株時価総額(億円)",
                color="TOPIX判定",
                hover_name="銘柄名",
                hover_data=["銘柄コード", "配当利回り(%)", "PER(倍)", "ROE(%)"],
                log_y=True,
                title="売買回転率 vs 浮動株時価総額",
                color_discrete_map={
                    "✅ 採用圏内": "#2ecc71",
                    "⚠️ ボーダー": "#f39c12",
                    "❌ 回転率不足": "#e74c3c",
                    "❌ 圏外": "#95a5a6",
                    "データ不足": "#bdc3c7",
                },
            )
            fig.add_vline(x=TRADED_VALUE_RATIO_THRESHOLD, line_dash="dash",
                          line_color="red", annotation_text="回転率基準 0.14")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sector_data = filtered.dropna(subset=["業種", "TOPIX判定"])
        if not sector_data.empty:
            sector_summary = sector_data.groupby(["業種", "TOPIX判定"]).size().reset_index(name="件数")
            fig2 = px.bar(
                sector_summary, x="業種", y="件数", color="TOPIX判定",
                title="業種別 TOPIX判定分布",
                color_discrete_map={
                    "✅ 採用圏内": "#2ecc71",
                    "⚠️ ボーダー": "#f39c12",
                    "❌ 回転率不足": "#e74c3c",
                    "❌ 圏外": "#95a5a6",
                    "データ不足": "#bdc3c7",
                },
            )
            fig2.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        border = filtered[filtered["TOPIX判定"] == "⚠️ ボーダー"]
        if not border.empty:
            st.markdown("### ボーダーライン銘柄（累積97%〜99%）")
            st.markdown("これらの銘柄は次回見直しで採用/除外が変わる可能性があります。")
            st.dataframe(
                border[["銘柄コード", "銘柄名", "市場区分", "浮動株時価総額(億円)",
                         "年間売買回転率", "累積時価総額比率",
                         "配当利回り(%)", "PER(倍)", "ROE(%)"]],
                use_container_width=True,
            )
        else:
            st.info("ボーダーライン銘柄はありません")


if __name__ == "__main__":
    main()
