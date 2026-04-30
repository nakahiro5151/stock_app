"""
日本株テーマ別資金フロー分析ダッシュボード
- テーマ別累積リターン・パフォーマンスチャート
- 注目テーマランキング（日次/週次/月次）
- テーマ別代表銘柄と資金流入解説
- OLS回帰・相関分析・日米リードラグ
- 売買判断シグナル
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
from urllib.parse import quote
import requests
import time
import json
import os
import io
import tempfile
from bs4 import BeautifulSoup

# .envからAPIキーを読み込み（python-dotenvがあれば使用、なければos.environのみ）
try:
    from dotenv import load_dotenv
    # スクリプトと同じディレクトリの.envを明示的に指定
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

warnings.filterwarnings("ignore")

st.set_page_config(page_title="テーマ別資金フロー分析", page_icon="🎯", layout="wide")

# =====================================================================
# テーマ定義: テーマ名 → 代表銘柄(5社) + 対応する米国ETF + 概要
# =====================================================================
THEMES = {
    "蓄電池": {
        "stocks": {"6752.T": "パナソニック", "6762.T": "TDK", "6981.T": "村田製作所",
                   "6674.T": "GSユアサ", "6810.T": "マクセル"},
        "us_etf": "QCLN", "desc": "EV・再エネ普及に伴う蓄電需要拡大", "kabutan": "蓄電池",
    },
    "防衛": {
        "stocks": {"7011.T": "三菱重工業", "7012.T": "川崎重工業", "7013.T": "IHI",
                   "5631.T": "日本製鋼所", "7721.T": "東京計器"},
        "us_etf": "ITA", "desc": "地政学リスク・防衛費増額", "kabutan": "防衛",
    },
    "レアアース": {
        "stocks": {"5713.T": "住友金属鉱山", "5714.T": "DOWAホールディングス",
                   "5706.T": "三井金属鉱業", "5711.T": "三菱マテリアル", "5727.T": "東邦チタニウム"},
        "us_etf": "REMX", "desc": "資源ナショナリズム・サプライチェーン再構築", "kabutan": "レアアース",
    },
    "地方銀行": {
        "stocks": {"7186.T": "コンコルディアFG", "8354.T": "ふくおかFG",
                   "7167.T": "めぶきFG", "8331.T": "千葉銀行", "8355.T": "静岡銀行"},
        "us_etf": "KRE", "desc": "金利上昇・地銀再編期待", "kabutan": "地方銀行",
    },
    "TOPIXコア30": {
        "stocks": {"7203.T": "トヨタ自動車", "8306.T": "三菱UFJFG", "6758.T": "ソニーG",
                   "6861.T": "キーエンス", "9984.T": "ソフトバンクG"},
        "us_etf": "SPY", "desc": "大型株への資金回帰・安定志向", "kabutan": "TOPIXコア30",
    },
    "半導体": {
        "stocks": {"8035.T": "東京エレクトロン", "6920.T": "レーザーテック",
                   "6857.T": "アドバンテスト", "6723.T": "ルネサスエレクトロニクス", "6146.T": "ディスコ"},
        "us_etf": "SOXX", "desc": "AI需要・半導体サイクル回復", "kabutan": "半導体",
    },
    "ペロブスカイト太陽電池": {
        "stocks": {"4005.T": "住友化学", "4183.T": "三井化学", "5019.T": "出光興産",
                   "6988.T": "日東電工", "4204.T": "積水化学工業"},
        "us_etf": "TAN", "desc": "次世代太陽電池の実用化進展", "kabutan": "ペロブスカイト太陽電池",
    },
    "JPX日経400": {
        "stocks": {"8058.T": "三菱商事", "8031.T": "三井物産", "9432.T": "NTT",
                   "4063.T": "信越化学工業", "6501.T": "日立製作所"},
        "us_etf": "SPY", "desc": "資本効率重視・ガバナンス改革", "kabutan": "JPX日経400",
    },
    "肥料": {
        "stocks": {"4208.T": "UBE", "4401.T": "ADEKA", "4996.T": "クミアイ化学工業",
                   "4997.T": "日本農薬", "4955.T": "アグロカネショウ"},
        "us_etf": "MOO", "desc": "食料安全保障・農業DX", "kabutan": "肥料",
    },
    "石炭": {
        "stocks": {"1518.T": "三井松島HD", "1514.T": "住石HD",
                   "5765.T": "五洋インテックス", "8031.T": "三井物産", "8002.T": "丸紅"},
        "us_etf": "XLE", "desc": "エネルギー安全保障・石炭価格高騰", "kabutan": "石炭",
    },
    "海水淡水化": {
        "stocks": {"6370.T": "栗田工業", "6368.T": "オルガノ", "6361.T": "荏原製作所",
                   "6366.T": "千代田化工建設", "6013.T": "タクミナ"},
        "us_etf": "PHO", "desc": "水資源不足・インフラ需要", "kabutan": "海水淡水化",
    },
    "宇宙開発": {
        "stocks": {"7012.T": "川崎重工業", "7011.T": "三菱重工業", "6208.T": "石川製作所",
                   "9412.T": "スカパーJSAT", "4274.T": "細谷火工"},
        "us_etf": "UFO", "desc": "宇宙ビジネス商業化・安全保障", "kabutan": "宇宙開発",
    },
    "人工知能": {
        "stocks": {"6758.T": "ソニーG", "6861.T": "キーエンス", "9984.T": "ソフトバンクG",
                   "4686.T": "ジャストシステム", "3993.T": "PKSHA Technology"},
        "us_etf": "BOTZ", "desc": "生成AI普及・企業DX加速", "kabutan": "人工知能",
    },
    "データセンター": {
        "stocks": {"3042.T": "セキュアヴェイル", "3778.T": "さくらインターネット",
                   "9434.T": "ソフトバンク", "4812.T": "ISID", "3844.T": "コムチュア"},
        "us_etf": "VPN", "desc": "クラウド需要・AI計算基盤拡大", "kabutan": "データセンター",
    },
    "フィジカルAI": {
        "stocks": {"6506.T": "安川電機", "6954.T": "ファナック", "7735.T": "SCREENホールディングス",
                   "6861.T": "キーエンス", "6645.T": "オムロン"},
        "us_etf": "ROBO", "desc": "ロボティクス・自動化のAI融合", "kabutan": "フィジカルAI",
    },
    "半導体製造装置": {
        "stocks": {"8035.T": "東京エレクトロン", "7735.T": "SCREENホールディングス",
                   "6146.T": "ディスコ", "7729.T": "東京精密", "6125.T": "岡本工作機械製作所"},
        "us_etf": "SOXX", "desc": "先端プロセス投資・EUV関連", "kabutan": "半導体製造装置",
    },
    "再生可能エネルギー": {
        "stocks": {"9519.T": "レノバ", "9517.T": "イーレックス", "1407.T": "ウエストHD",
                   "9514.T": "エフオン", "3856.T": "Abalance"},
        "us_etf": "ICLN", "desc": "脱炭素政策・GX推進", "kabutan": "再生可能エネルギー",
    },
    "銀行": {
        "stocks": {"8306.T": "三菱UFJFG", "8316.T": "三井住友FG", "8411.T": "みずほFG",
                   "8309.T": "三井住友トラスト", "7182.T": "ゆうちょ銀行"},
        "us_etf": "XLF", "desc": "金利正常化・利ざや改善", "kabutan": "銀行",
    },
    "SaaS": {
        "stocks": {"4478.T": "フリー", "4443.T": "Sansan", "4776.T": "サイボウズ",
                   "3923.T": "ラクス", "4475.T": "HENNGE"},
        "us_etf": "WCLD", "desc": "企業DX・サブスクリプション成長", "kabutan": "SaaS",
    },
    "核融合発電": {
        "stocks": {"6501.T": "日立製作所", "6502.T": "東芝", "5631.T": "日本製鋼所",
                   "6366.T": "千代田化工建設", "5727.T": "東邦チタニウム"},
        "us_etf": "NLR", "desc": "次世代エネルギー・核融合実用化期待", "kabutan": "核融合発電",
    },
    "水素": {
        "stocks": {"6902.T": "デンソー", "7203.T": "トヨタ自動車", "8088.T": "岩谷産業",
                   "6366.T": "千代田化工建設", "5020.T": "ENEOSホールディングス"},
        "us_etf": "ICLN", "desc": "水素社会実現・グリーン水素", "kabutan": "水素",
    },
    "総合商社": {
        "stocks": {"8058.T": "三菱商事", "8031.T": "三井物産", "8001.T": "伊藤忠商事",
                   "8053.T": "住友商事", "8002.T": "丸紅"},
        "us_etf": "SPY", "desc": "バフェット効果・資源高・株主還元", "kabutan": "総合商社",
    },
    "インバウンド": {
        "stocks": {"9603.T": "エイチ・アイ・エス", "6191.T": "エアトリ",
                   "9024.T": "西武HD", "2809.T": "キユーピー", "7832.T": "バンダイナムコHD"},
        "us_etf": "JETS", "desc": "訪日外国人増加・円安恩恵", "kabutan": "インバウンド",
    },
    "海運": {
        "stocks": {"9101.T": "日本郵船", "9104.T": "商船三井", "9107.T": "川崎汽船",
                   "9110.T": "NSユナイテッド海運", "9119.T": "飯野海運"},
        "us_etf": "SEA", "desc": "海運市況・コンテナ運賃動向", "kabutan": "海運",
    },
    "鉄道": {
        "stocks": {"9020.T": "JR東日本", "9021.T": "JR西日本", "9022.T": "JR東海",
                   "9005.T": "東急", "9009.T": "京成電鉄"},
        "us_etf": "XLI", "desc": "インバウンド・不動産開発・運賃改定", "kabutan": "鉄道",
    },
    "不動産": {
        "stocks": {"8801.T": "三井不動産", "8802.T": "三菱地所", "8830.T": "住友不動産",
                   "3289.T": "東急不動産HD", "8804.T": "東京建物"},
        "us_etf": "IYR", "desc": "金利動向・オフィス回帰・再開発", "kabutan": "不動産",
    },
    "建設": {
        "stocks": {"1801.T": "大成建設", "1802.T": "大林組", "1812.T": "鹿島建設",
                   "1803.T": "清水建設", "1808.T": "長谷工コーポレーション"},
        "us_etf": "XLI", "desc": "国土強靭化・再開発・半導体工場建設", "kabutan": "建設",
    },
    "光デバイス": {
        "stocks": {"6146.T": "ディスコ", "6920.T": "レーザーテック", "6962.T": "大真空",
                   "6976.T": "太陽誘電", "6728.T": "アルバック"},
        "us_etf": "SOXX", "desc": "光半導体・光通信・シリコンフォトニクス", "kabutan": "光デバイス",
    },
}

# =====================================================================
# データ取得
# =====================================================================

@st.cache_data(ttl=600)
def fetch_prices(tickers, period="6mo"):
    """複数ティッカーの終値を一括取得"""
    try:
        df = yf.download(tickers, period=period, progress=False, threads=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df[["Close"]]
            close.columns = tickers if isinstance(tickers, list) else [tickers]
        return close
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_us_etfs(period="6mo"):
    """米国ETFの終値を取得（リードラグ分析用）- 個別取得で堅牢化"""
    etfs = sorted(set(info["us_etf"] for info in THEMES.values()))
    closes = {}
    for etf in etfs:
        try:
            df = yf.download(etf, period=period, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    closes[etf] = df["Close"].iloc[:, 0]
                else:
                    closes[etf] = df["Close"]
        except Exception:
            continue
    if closes:
        result = pd.DataFrame(closes)
        result.index = result.index.tz_localize(None) if result.index.tz is not None else result.index
        return result
    return pd.DataFrame()


def build_theme_index(period="6mo"):
    """全テーマの等金額加重インデックス（日次リターン）を構築"""
    all_tickers = set()
    for info in THEMES.values():
        all_tickers.update(info["stocks"].keys())
    all_tickers = sorted(all_tickers)

    prices = fetch_prices(all_tickers, period=period)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    # テーマ別等金額加重リターン
    theme_returns = {}
    for theme, info in THEMES.items():
        cols = [t for t in info["stocks"].keys() if t in prices.columns]
        if len(cols) == 0:
            continue
        ret = prices[cols].pct_change().dropna(how="all")
        theme_returns[theme] = ret.mean(axis=1)

    theme_ret_df = pd.DataFrame(theme_returns).dropna(how="all")
    theme_cum = (1 + theme_ret_df).cumprod() - 1  # 累積リターン

    return theme_ret_df, theme_cum



# =====================================================================
# ニュース取得・テーマ紐づけ
# =====================================================================
# 銘柄→テーマ 逆引き辞書（ポートフォリオのテーマ分類用）
TICKER_TO_THEMES = {}
for _theme, _info in THEMES.items():
    for _ticker in _info["stocks"].keys():
        if _ticker not in TICKER_TO_THEMES:
            TICKER_TO_THEMES[_ticker] = []
        TICKER_TO_THEMES[_ticker].append(_theme)

# 銘柄名→テーマ推定用キーワードマッピング
_NAME_THEME_KEYWORDS = {
    "半導体": ["半導体", "エレクトロン", "アドバンテスト", "レーザーテック", "ディスコ",
               "ルネサス", "ソシオネクスト", "SCREEN", "東京精密", "SUMCO"],
    "半導体製造装置": ["エレクトロン", "SCREEN", "ディスコ", "東京精密", "岡本工作"],
    "人工知能": ["AI", "ソニー", "ソフトバンク", "PKSHA", "ジャストシステム", "NEC", "富士通", "NTTデータ"],
    "防衛": ["三菱重工", "川崎重工", "IHI", "日本製鋼", "東京計器", "石川製作", "細谷火工"],
    "宇宙開発": ["三菱重工", "川崎重工", "スカパー", "石川製作", "細谷火工"],
    "蓄電池": ["パナソニック", "TDK", "村田製作", "GSユアサ", "マクセル", "バッテリー"],
    "銀行": ["三菱UFJ", "三井住友F", "みずほ", "ゆうちょ", "トラスト"],
    "地方銀行": ["千葉銀", "静岡銀", "ふくおか", "コンコルディア", "めぶき"],
    "総合商社": ["三菱商事", "三井物産", "伊藤忠", "住友商事", "丸紅", "豊田通商"],
    "インバウンド": ["エイチ・アイ・エス", "エアトリ", "西武HD", "バンダイナムコ", "オリエンタルランド"],
    "海運": ["日本郵船", "商船三井", "川崎汽船", "NSユナイテッド", "飯野海運"],
    "鉄道": ["JR", "東急", "京成", "小田急", "京王", "西武"],
    "水素": ["デンソー", "トヨタ", "岩谷産業", "千代田化工", "ENEOS"],
    "再生可能エネルギー": ["レノバ", "イーレックス", "ウエストHD", "エフオン", "Abalance"],
    "核融合発電": ["日立製作", "東芝", "日本製鋼", "千代田化工", "東邦チタニウム"],
    "ペロブスカイト太陽電池": ["住友化学", "三井化学", "出光興産", "日東電工", "積水化学"],
    "データセンター": ["さくらインターネット", "ソフトバンク", "ISID", "コムチュア", "セキュアヴェイル", "NTT"],
    "フィジカルAI": ["安川電機", "ファナック", "SCREEN", "キーエンス", "オムロン"],
    "SaaS": ["フリー", "Sansan", "サイボウズ", "ラクス", "HENNGE"],
    "レアアース": ["住友金属鉱山", "DOWA", "三井金属", "三菱マテリアル", "東邦チタニウム"],
    "海水淡水化": ["栗田工業", "オルガノ", "荏原製作", "千代田化工", "タクミナ"],
    "石炭": ["三井松島", "住石", "五洋インテックス"],
    "肥料": ["UBE", "ADEKA", "クミアイ化学", "日本農薬", "アグロカネショウ"],
    "TOPIXコア30": ["トヨタ", "ソニー", "キーエンス", "ソフトバンクG"],
    "JPX日経400": ["三菱商事", "三井物産", "NTT", "信越化学", "日立製作"],
    "不動産": ["三井不動産", "三菱地所", "住友不動産", "東急不動産", "東京建物",
              "野村不動産", "霞ヶ関キャピタル", "ロードスター", "いちご", "ヒューリック"],
    "光デバイス": ["ディスコ", "レーザーテック", "大真空", "太陽誘電", "アルバック",
                  "浜松ホトニクス", "ウシオ電機", "サンテック", "フォトニクス"],
}

def guess_themes_by_name(ticker, name):
    """銘柄名からテーマを推定"""
    # まず逆引き辞書にあればそれを返す
    if ticker in TICKER_TO_THEMES:
        return TICKER_TO_THEMES[ticker]
    # 銘柄名でキーワードマッチング
    matched = []
    for theme, keywords in _NAME_THEME_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                if theme not in matched:
                    matched.append(theme)
                break
    return matched

# テーマごとのキーワード辞書（ニュースマッチング用）
THEME_KEYWORDS = {}
for _theme, _info in THEMES.items():
    kws = [_theme]
    # 銘柄名もキーワードに
    kws.extend(_info["stocks"].values())
    # descからもキーワード抽出
    kws.extend([w for w in _info["desc"].replace("・", "　").replace("、", "　").split("　") if len(w) >= 2])
    THEME_KEYWORDS[_theme] = kws

# 追加の汎用キーワード→テーマ マッピング
EXTRA_KEYWORD_MAP = {
    "半導体": ["半導体", "TSMC", "エヌビディア", "NVIDIA", "GPU", "AI半導体", "先端半導体",
               "半導体規制", "対中規制", "チップ", "ファウンドリ"],
    "人工知能": ["AI", "人工知能", "生成AI", "ChatGPT", "LLM", "機械学習", "ディープラーニング",
                "OpenAI", "Gemini", "Claude", "大規模言語モデル", "NVIDIA", "GPU"],
    "防衛": ["防衛", "ミサイル", "安全保障", "NATO", "軍事", "防衛費", "戦争", "紛争",
            "ウクライナ", "ロシア", "北朝鮮", "台湾有事", "地政学", "武器", "弾薬"],
    "蓄電池": ["蓄電池", "EV", "電気自動車", "リチウムイオン", "全固体電池", "バッテリー",
              "テスラ", "BYD", "充電"],
    "銀行": ["金利", "利上げ", "日銀", "マイナス金利", "YCC", "長期金利", "金融政策",
            "植田", "FRB", "FOMC", "利下げ", "メガバンク"],
    "地方銀行": ["地銀", "地方銀行", "金利上昇", "再編"],
    "総合商社": ["商社", "バフェット", "資源", "配当", "三菱商事", "三井物産", "伊藤忠"],
    "インバウンド": ["インバウンド", "訪日", "観光", "免税", "円安", "旅行", "ホテル", "百貨店"],
    "海運": ["海運", "コンテナ", "運賃", "バルチック", "紅海", "スエズ", "ホルムズ海峡", "物流"],
    "水素": ["水素", "グリーン水素", "燃料電池", "FCV", "アンモニア"],
    "再生可能エネルギー": ["再エネ", "再生可能エネルギー", "太陽光", "風力", "脱炭素", "GX",
                        "カーボンニュートラル", "グリーントランスフォーメーション"],
    "ペロブスカイト太陽電池": ["ペロブスカイト", "次世代太陽電池"],
    "核融合発電": ["核融合", "フュージョン", "ITER", "プラズマ"],
    "宇宙開発": ["宇宙", "ロケット", "衛星", "スペース", "JAXA", "SpaceX", "宇宙ビジネス"],
    "データセンター": ["データセンター", "DC", "クラウド", "サーバー", "AWS", "Azure",
                     "量子コンピュータ", "量子"],
    "フィジカルAI": ["ロボット", "ロボティクス", "自動化", "ファナック", "安川",
                    "自動運転", "ドローン", "ヒューマノイド"],
    "SaaS": ["SaaS", "クラウドサービス", "サブスク", "DX", "デジタルトランスフォーメーション"],
    "レアアース": ["レアアース", "希土類", "レアメタル", "資源ナショナリズム", "鉱物資源"],
    "海水淡水化": ["淡水化", "水処理", "水不足", "水インフラ"],
    "石炭": ["石炭", "火力発電", "エネルギー安全保障"],
    "肥料": ["肥料", "農業", "食料安全保障", "食料危機", "穀物"],
    "鉄道": ["鉄道", "JR", "運賃改定", "リニア", "新幹線"],
    "半導体製造装置": ["半導体製造装置", "EUV", "露光装置", "東京エレクトロン", "ディスコ",
                     "ASML", "先端パッケージング"],
    "TOPIXコア30": ["TOPIX", "大型株", "日経平均", "株価指数"],
    "JPX日経400": ["JPX", "ガバナンス", "ROE", "資本効率", "PBR", "東証改革"],
    "光デバイス": ["光デバイス", "光半導体", "光通信", "シリコンフォトニクス", "フォトニクス",
                  "光トランシーバ", "レーザー", "光ファイバ", "光電変換", "VCSEL",
                  "光配線", "Co-Packaged Optics", "CPO"],
    # 地政学・政策テーマ（既存テーマに横断的に紐づけ）
    "防衛": ["トランプ", "関税", "制裁", "経済安全保障", "国策"],
    "総合商社": ["原油", "WTI", "OPEC", "石油", "資源価格"],
    "銀行": ["国債", "債券", "イールドカーブ"],
}
for _theme, _extra_kws in EXTRA_KEYWORD_MAP.items():
    if _theme in THEME_KEYWORDS:
        THEME_KEYWORDS[_theme].extend(_extra_kws)


@st.cache_data(ttl=300)
def fetch_news_rss():
    """複数RSSソースから経済ニュースを取得"""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    feeds = [
        # --- 日本市場・経済 ---
        ("https://news.google.com/rss/search?q=日経平均+株価+マーケット&hl=ja&gl=JP&ceid=JP:ja", "日本株式"),
        ("https://news.google.com/rss/search?q=決算+業績+株主還元+増配&hl=ja&gl=JP&ceid=JP:ja", "決算"),
        # --- 金融政策・中央銀行 ---
        ("https://news.google.com/rss/search?q=日銀+金融政策+利上げ+YCC&hl=ja&gl=JP&ceid=JP:ja", "日銀政策"),
        ("https://news.google.com/rss/search?q=FRB+利下げ+FOMC+金利&hl=ja&gl=JP&ceid=JP:ja", "米金融政策"),
        # --- 地政学・戦争・安全保障 ---
        ("https://news.google.com/rss/search?q=戦争+紛争+安全保障+地政学&hl=ja&gl=JP&ceid=JP:ja", "地政学"),
        ("https://news.google.com/rss/search?q=トランプ+関税+制裁+米中&hl=ja&gl=JP&ceid=JP:ja", "トランプ政策"),
        ("https://news.google.com/rss/search?q=イラン+中東+ホルムズ海峡+原油&hl=ja&gl=JP&ceid=JP:ja", "中東情勢"),
        ("https://news.google.com/rss/search?q=ウクライナ+ロシア+NATO+防衛費&hl=ja&gl=JP&ceid=JP:ja", "ウクライナ"),
        ("https://news.google.com/rss/search?q=台湾+中国+半導体規制+デカップリング&hl=ja&gl=JP&ceid=JP:ja", "台湾海峡"),
        # --- コモディティ・資源 ---
        ("https://news.google.com/rss/search?q=原油+WTI+OPEC+石油&hl=ja&gl=JP&ceid=JP:ja", "原油"),
        ("https://news.google.com/rss/search?q=金価格+ゴールド+安全資産&hl=ja&gl=JP&ceid=JP:ja", "金"),
        ("https://news.google.com/rss/search?q=レアアース+レアメタル+資源&hl=ja&gl=JP&ceid=JP:ja", "資源"),
        # --- テクノロジー・AI ---
        ("https://news.google.com/rss/search?q=AI+人工知能+生成AI+NVIDIA&hl=ja&gl=JP&ceid=JP:ja", "AI"),
        ("https://news.google.com/rss/search?q=半導体+TSMC+先端半導体&hl=ja&gl=JP&ceid=JP:ja", "半導体"),
        ("https://news.google.com/rss/search?q=データセンター+クラウド+量子コンピュータ&hl=ja&gl=JP&ceid=JP:ja", "テック基盤"),
        ("https://news.google.com/rss/search?q=ロボット+自動運転+ドローン&hl=ja&gl=JP&ceid=JP:ja", "ロボティクス"),
        # --- 国策・政策テーマ ---
        ("https://news.google.com/rss/search?q=国策+経済安全保障+GX+DX&hl=ja&gl=JP&ceid=JP:ja", "国策"),
        ("https://news.google.com/rss/search?q=防衛費+宇宙開発+サイバーセキュリティ&hl=ja&gl=JP&ceid=JP:ja", "防衛宇宙"),
        ("https://news.google.com/rss/search?q=脱炭素+再エネ+水素+原発再稼働&hl=ja&gl=JP&ceid=JP:ja", "エネルギー政策"),
        ("https://news.google.com/rss/search?q=核融合+ペロブスカイト+次世代エネルギー&hl=ja&gl=JP&ceid=JP:ja", "次世代エネ"),
        # --- 為替・債券 ---
        ("https://news.google.com/rss/search?q=ドル円+為替+円安+円高&hl=ja&gl=JP&ceid=JP:ja", "為替"),
        # --- 海外メディア（英語・日本語混在） ---
        ("https://news.google.com/rss/search?q=Japan+stocks+Nikkei+economy&hl=ja&gl=JP&ceid=JP:ja", "海外日本株"),
        ("https://news.google.com/rss/search?q=Bloomberg+Japan+market&hl=ja&gl=JP&ceid=JP:ja", "Bloomberg"),
        # --- セクター ---
        ("https://news.google.com/rss/search?q=インバウンド+観光+訪日&hl=ja&gl=JP&ceid=JP:ja", "インバウンド"),
        ("https://news.google.com/rss/search?q=海運+コンテナ+物流&hl=ja&gl=JP&ceid=JP:ja", "海運物流"),
        ("https://news.google.com/rss/search?q=銀行+地銀+メガバンク+金利&hl=ja&gl=JP&ceid=JP:ja", "銀行"),
        ("https://news.google.com/rss/search?q=商社+バフェット+資源株&hl=ja&gl=JP&ceid=JP:ja", "商社"),
    ]

    articles = []
    seen_titles = set()
    # タイトル正規化→出現フィード数カウント用
    title_feed_count = {}

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
                desc = item.findtext("description", "").strip()

                if not title:
                    continue

                # 複数フィード出現カウント
                title_norm = title.lower().strip()
                title_feed_count[title_norm] = title_feed_count.get(title_norm, 0) + 1

                # 重複排除
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # 日時パース
                time_str = ""
                dt_obj = None
                if pub_date:
                    try:
                        dt_obj = parsedate_to_datetime(pub_date)
                        time_str = dt_obj.strftime("%m/%d %H:%M")
                    except Exception:
                        time_str = pub_date[:16]

                articles.append({
                    "title": title,
                    "link": link,
                    "time": time_str,
                    "pub_date": pub_date,
                    "dt": dt_obj,
                    "source": source,
                    "desc": desc,
                })
        except Exception:
            continue

    # 各記事に複数フィード出現数を付与
    for a in articles:
        a["feed_count"] = title_feed_count.get(a["title"].lower().strip(), 1)

    # 時刻でソート（新しい順）— datetimeオブジェクトで正確にソート
    now = datetime.now(articles[0]["dt"].tzinfo) if articles and articles[0].get("dt") else datetime.now()
    articles.sort(key=lambda x: x["dt"] if x.get("dt") else datetime.min.replace(tzinfo=now.tzinfo), reverse=True)
    return articles[:200]


def calc_attention_score(article, theme_matches):
    """注目度スコアを算出（テーマ関連度 × フィード出現数 × 鮮度）"""
    # テーママッチスコア（キーワードヒット数合計）
    theme_score = sum(m["score"] for m in theme_matches) if theme_matches else 0

    # 複数フィード出現ボーナス（複数カテゴリで拾われた＝注目度高い）
    feed_bonus = article.get("feed_count", 1)

    # 鮮度スコア（直近ほど高い）
    freshness = 1.0
    dt = article.get("dt")
    if dt:
        try:
            now = datetime.now(dt.tzinfo)
            hours_ago = (now - dt).total_seconds() / 3600
            if hours_ago <= 1:
                freshness = 3.0
            elif hours_ago <= 3:
                freshness = 2.5
            elif hours_ago <= 6:
                freshness = 2.0
            elif hours_ago <= 12:
                freshness = 1.5
            elif hours_ago <= 24:
                freshness = 1.0
            else:
                freshness = 0.5
        except Exception:
            pass

    score = (theme_score * 2 + feed_bonus) * freshness
    return round(score, 1)


def match_themes_to_article(title, desc=""):
    """ニュース記事タイトル+説明からマッチするテーマを検出"""
    text = (title + " " + desc).lower()
    matched = []
    for theme, keywords in THEME_KEYWORDS.items():
        score = 0
        matched_kws = []
        for kw in keywords:
            if kw.lower() in text:
                score += 1
                matched_kws.append(kw)
        if score >= 1:
            matched.append({"theme": theme, "score": score, "keywords": matched_kws})
    # スコア順
    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched[:5]


# =====================================================================
# 分析関数
# =====================================================================
def calc_period_returns(theme_ret_df, period="1d"):
    """日次/週次/月次リターンを算出"""
    if theme_ret_df.empty:
        return pd.Series(dtype=float)
    if period == "1d":
        return theme_ret_df.iloc[-1] if len(theme_ret_df) > 0 else pd.Series(dtype=float)
    elif period == "1w":
        return theme_ret_df.iloc[-5:].sum() if len(theme_ret_df) >= 5 else theme_ret_df.sum()
    elif period == "1m":
        return theme_ret_df.iloc[-21:].sum() if len(theme_ret_df) >= 21 else theme_ret_df.sum()
    elif period == "3m":
        return theme_ret_df.iloc[-63:].sum() if len(theme_ret_df) >= 63 else theme_ret_df.sum()
    return theme_ret_df.sum()


def ols_trend(series):
    """OLS回帰でトレンド傾き・R²を算出"""
    y = series.dropna().values
    if len(y) < 5:
        return 0, 0
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, r_value ** 2


def calc_momentum_score(theme_ret_df):
    """モメンタムスコア: 短期(5d) + 中期(20d) + 長期(60d) の加重平均"""
    scores = {}
    for col in theme_ret_df.columns:
        s = theme_ret_df[col].dropna()
        if len(s) < 5:
            scores[col] = 0
            continue
        r5 = s.iloc[-5:].sum()
        r20 = s.iloc[-20:].sum() if len(s) >= 20 else s.sum()
        r60 = s.iloc[-60:].sum() if len(s) >= 60 else s.sum()
        # 短期重視の加重
        scores[col] = r5 * 0.5 + r20 * 0.3 + r60 * 0.2
    return pd.Series(scores).sort_values(ascending=False)


def calc_lead_lag(jp_returns, us_returns, max_lag=5):
    """日米リードラグ相関（クロス相関）"""
    results = {}
    for theme, info in THEMES.items():
        if theme not in jp_returns.columns:
            continue
        etf = info["us_etf"]
        if etf not in us_returns.columns:
            continue
        jp = jp_returns[theme].dropna()
        us = us_returns[etf].dropna()
        # 日付を揃える（タイムゾーン除去して日付のみで比較）
        jp.index = jp.index.tz_localize(None) if jp.index.tz is not None else jp.index
        us.index = us.index.tz_localize(None) if us.index.tz is not None else us.index
        jp.index = jp.index.normalize()
        us.index = us.index.normalize()
        common = jp.index.intersection(us.index)
        if len(common) < 10:
            continue

        jp_c = jp.loc[common].values
        us_c = us.loc[common].values
        best_lag, best_corr = 0, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                c = np.corrcoef(us_c[:-lag], jp_c[lag:])[0, 1] if len(us_c) > lag else 0
            elif lag < 0:
                c = np.corrcoef(us_c[-lag:], jp_c[:lag])[0, 1] if len(jp_c) > -lag else 0
            else:
                c = np.corrcoef(us_c, jp_c)[0, 1]
            if abs(c) > abs(best_corr):
                best_corr = c
                best_lag = lag
        results[theme] = {"etf": etf, "best_lag": best_lag, "correlation": best_corr}
    return pd.DataFrame(results).T


def generate_signal(theme_ret_df):
    """売買シグナル生成: モメンタム + OLSトレンド + ボラティリティ調整"""
    signals = {}
    for col in theme_ret_df.columns:
        s = theme_ret_df[col].dropna()
        if len(s) < 20:
            signals[col] = {"signal": "データ不足", "score": 0, "color": "gray"}
            continue

        # モメンタム
        r5 = s.iloc[-5:].sum()
        r20 = s.iloc[-20:].sum()

        # OLSトレンド（直近20日）
        slope, r2 = ols_trend(s.iloc[-20:].cumsum())

        # ボラティリティ（直近20日）
        vol = s.iloc[-20:].std()
        sharpe_like = r20 / vol if vol > 0 else 0

                # 総合スコア（外れ値をクリップ）
        score = np.clip(r5 * 100, -20, 20) * 0.3 + np.clip(r20 * 100, -50, 50) * 0.2 + np.clip(slope * 1000, -10, 10) * 0.3 + np.clip(sharpe_like, -5, 5) * 0.2

        if score > 2:
            sig, color = "強い買い 🟢", "#00c853"
        elif score > 0.5:
            sig, color = "買い 🔵", "#2196f3"
        elif score > -0.5:
            sig, color = "中立 ⚪", "#9e9e9e"
        elif score > -2:
            sig, color = "売り 🟠", "#ff9800"
        else:
            sig, color = "強い売り 🔴", "#f44336"

        signals[col] = {"signal": sig, "score": round(score, 2), "color": color,
                        "r5": round(r5 * 100, 2), "r20": round(r20 * 100, 2),
                        "slope": round(slope * 1000, 3), "sharpe": round(sharpe_like, 2)}
    return signals


def generate_commentary(theme, info, ret_1d, ret_1w, ret_1m, signal_data):
    """テーマ別の自動解説コメント生成"""
    desc = info["desc"]
    sig = signal_data.get("signal", "N/A")
    r5 = signal_data.get("r5", 0)
    r20 = signal_data.get("r20", 0)

    if r5 > 0 and r20 > 0:
        momentum_text = "短期・中期ともに上昇トレンド継続中"
    elif r5 > 0 and r20 <= 0:
        momentum_text = "短期的に反発の兆し。中期トレンドはまだ弱い"
    elif r5 <= 0 and r20 > 0:
        momentum_text = "中期上昇トレンドの中で短期調整局面"
    else:
        momentum_text = "短期・中期ともに下落圧力"

    d_pct = round(ret_1d * 100, 2) if not np.isnan(ret_1d) else 0
    w_pct = round(ret_1w * 100, 2) if not np.isnan(ret_1w) else 0
    m_pct = round(ret_1m * 100, 2) if not np.isnan(ret_1m) else 0

    comment = (
        f"【{theme}】{desc}。"
        f"日次{d_pct:+.2f}% / 週次{w_pct:+.2f}% / 月次{m_pct:+.2f}%。"
        f"{momentum_text}。判定: {sig}"
    )
    return comment


# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 100%; }
    .theme-card {
        background: #1a1a2e; border-radius: 10px; padding: 16px;
        margin-bottom: 8px; border-left: 4px solid #0066ff;
    }
    .signal-badge {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-weight: bold; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# メインUI
# =====================================================================
st.title("🎯 日本株テーマ別資金フロー分析")
st.caption("テーマ別の資金動向・モメンタム・日米リードラグから売買判断を支援")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    analysis_period = st.selectbox("分析期間", ["3mo", "6mo", "1y", "2y"], index=1)
    selected_themes = st.multiselect(
        "表示テーマ", list(THEMES.keys()), default=list(THEMES.keys())
    )
    show_individual = st.toggle("個別銘柄チャートも表示", value=False)

# データ取得
with st.spinner("📊 株価データ取得中..."):
    theme_ret_df, theme_cum = build_theme_index(period=analysis_period)

if theme_ret_df.empty:
    st.error("データ取得に失敗しました。しばらく待ってから再試行してください。")
    st.stop()

# フィルタ
cols_available = [c for c in selected_themes if c in theme_ret_df.columns]
theme_ret_filtered = theme_ret_df[cols_available]
theme_cum_filtered = theme_cum[cols_available]

# =====================================================================
# Tab 1: テーマランキング
# =====================================================================
# タブを折り返し表示にするCSS（スクロールなしで全タブ表示）
st.markdown("""
<style>
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([
    "📊 テーマランキング", "📈 累積リターン",
    "🔬 テーマ詳細", "🌏 日米リードラグ", "🎯 売買シグナル",
    "🔥 HOTニュース", "📋 ポートフォリオ", "🐉 ドラゴンキング",
    "📢 出来高急増", "📊 TOPIX新基準", "🏛️ PBR改革",
    "🦈 アクティビスト", "👁️ ウォッチリスト", "📰 決算速報"
])

with tab1:
    st.subheader("テーマ別リターンランキング")

    c1, c2, c3, c4 = st.columns(4)
    ret_1d = calc_period_returns(theme_ret_filtered, "1d")
    ret_1w = calc_period_returns(theme_ret_filtered, "1w")
    ret_1m = calc_period_returns(theme_ret_filtered, "1m")
    ret_3m = calc_period_returns(theme_ret_filtered, "3m")

    # 週次ランキング順位を事前計算（日次の急変検出用）
    rank_1w = (ret_1w * 100).sort_values(ascending=False)
    rank_1w_order = {theme: i for i, theme in enumerate(rank_1w.index)}
    rank_1d_order = {theme: i for i, theme in enumerate((ret_1d * 100).sort_values(ascending=False).index)}

    def show_ranking(col, title, data, emoji, show_surge=False):
        with col:
            st.markdown(f"**{emoji} {title}**")
            if data.empty:
                st.write("データなし")
                return
            ranked = (data * 100).sort_values(ascending=False)
            for i, (theme, val) in enumerate(ranked.items()):
                color = "#00c853" if val > 0 else "#f44336"
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."

                # 急上昇/急落マーク（日次のみ）
                badge = ""
                if show_surge and theme in rank_1w_order and theme in rank_1d_order:
                    diff = rank_1w_order[theme] - rank_1d_order[theme]  # 正=順位上昇
                    if diff >= 5:
                        badge = " <span style='color:#00e676;font-size:0.85rem' title='週次比 順位+{}'>�急上昇</span>".format(diff)
                    elif diff >= 3:
                        badge = " <span style='color:#69f0ae;font-size:0.85rem' title='週次比 順位+{}'>⬆️上昇</span>".format(diff)
                    elif diff <= -5:
                        badge = " <span style='color:#ff5252;font-size:0.85rem' title='週次比 順位{}'>💥急落</span>".format(diff)
                    elif diff <= -3:
                        badge = " <span style='color:#ff8a80;font-size:0.85rem' title='週次比 順位{}'>⬇️下落</span>".format(diff)

                # 株探リンク
                kw = THEMES.get(theme, {}).get("kabutan", theme)
                kb_url = f"https://kabutan.jp/themes/?theme={quote(kw)}"
                theme_link = f"<a href='{kb_url}' target='_blank' style='color:inherit;text-decoration:none;border-bottom:1px dotted #666;'>{theme}</a>"

                st.markdown(
                    f"{medal} {theme_link}: "
                    f"<span style='color:{color};font-weight:bold'>{val:+.2f}%</span>"
                    f"{badge}",
                    unsafe_allow_html=True
                )

    show_ranking(c1, "日次", ret_1d, "📅", show_surge=True)
    show_ranking(c2, "週次", ret_1w, "📆")
    show_ranking(c3, "月次", ret_1m, "🗓️")
    show_ranking(c4, "3ヶ月", ret_3m, "📊")

    # ヒートマップ
    st.subheader("テーマ別リターン ヒートマップ")
    heatmap_data = pd.DataFrame({
        "日次": ret_1d * 100, "週次": ret_1w * 100,
        "月次": ret_1m * 100, "3ヶ月": ret_3m * 100
    }).dropna()

    if not heatmap_data.empty:
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdYlGn",
            text=heatmap_data.values.round(2),
            texttemplate="%{text:.2f}%",
            textfont=dict(size=11),
            colorbar=dict(title="%"),
        ))
        fig_heat.update_layout(
            height=max(400, len(heatmap_data) * 28),
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── リスクリワード分析（数カ月スイング向け） ──
    st.divider()
    st.markdown("#### 📐 テーマ別リスクリワード分析（中長期スイング向け）")
    st.caption("数カ月〜年単位の保有で指数を超えるリターンを狙うためのテーマ選定")

    if theme_ret_filtered is not None and not theme_ret_filtered.empty:
        rr_rows = []
        for theme in theme_ret_filtered.columns:
            ts = theme_ret_filtered[theme].dropna()
            if len(ts) < 40:
                continue
            ret_3m = (ts.iloc[-1] / ts.iloc[-min(63, len(ts))] - 1) * 100 if len(ts) > 20 else None
            ret_6m = (ts.iloc[-1] / ts.iloc[0] - 1) * 100
            daily_ret = ts.pct_change().dropna()
            if len(daily_ret) < 20:
                continue
            ann_ret = daily_ret.mean() * 252 * 100
            ann_vol = daily_ret.std() * np.sqrt(252) * 100
            sharpe = (ann_ret - 0.1) / ann_vol if ann_vol > 0 else 0
            downside = daily_ret[daily_ret < 0]
            down_vol = downside.std() * np.sqrt(252) * 100 if len(downside) > 5 else ann_vol
            sortino = (ann_ret - 0.1) / down_vol if down_vol > 0 else 0
            cummax = ts.cummax()
            drawdown = (ts - cummax) / cummax * 100
            max_dd = drawdown.min()
            monthly = ts.resample("ME").last().pct_change().dropna()
            win_rate = (monthly > 0).mean() * 100 if len(monthly) > 2 else None
            calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
            from_high = (ts.iloc[-1] / ts.max() - 1) * 100
            rr_rows.append({
                "テーマ": theme,
                "3Mリターン(%)": round(ret_3m, 1) if ret_3m else None,
                "6Mリターン(%)": round(ret_6m, 1),
                "年率リターン(%)": round(ann_ret, 1),
                "年率ボラ(%)": round(ann_vol, 1),
                "シャープ比": round(sharpe, 2),
                "ソルティノ比": round(sortino, 2),
                "最大DD(%)": round(max_dd, 1),
                "カルマー比": round(calmar, 2),
                "月次勝率(%)": round(win_rate, 0) if win_rate else None,
                "高値比(%)": round(from_high, 1),
            })

        if rr_rows:
            rr_df = pd.DataFrame(rr_rows).sort_values("ソルティノ比", ascending=False)

            st.markdown("##### 🎯 推奨テーマ（リスクリワード上位 × 押し目）")
            recommend = rr_df[(rr_df["ソルティノ比"] > 0) & (rr_df["高値比(%)"] < -3)].head(5)
            if not recommend.empty:
                for _, r in recommend.iterrows():
                    sc = "#2ecc71" if r["ソルティノ比"] > 1 else "#f1c40f" if r["ソルティノ比"] > 0.5 else "#e74c3c"
                    wr_str = f"勝率{r['月次勝率(%)']:.0f}%" if pd.notna(r.get("月次勝率(%)")) else ""
                    _card = (
                        f'<div style="padding:8px 12px;margin-bottom:4px;border-radius:6px;'
                        f'border-left:4px solid {sc};background:rgba(128,128,128,0.08);">'
                        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
                        f'<span style="font-weight:bold;font-size:1rem;">{r["テーマ"]}</span>'
                        f'<span style="color:{sc};font-weight:bold;">ソルティノ {r["ソルティノ比"]:.2f}</span>'
                        f'<span style="color:#3498db;">高値比 {r["高値比(%)"]:+.1f}%</span>'
                        f'<span style="opacity:0.7;">6M {r["6Mリターン(%)"]:+.1f}%</span>'
                        f'<span style="opacity:0.7;">最大DD {r["最大DD(%)"]:.1f}%</span>'
                        f'<span style="opacity:0.7;">{wr_str}</span>'
                        f'</div></div>'
                    )
                    st.markdown(_card, unsafe_allow_html=True)
                st.caption("ソルティノ比が高く、直近高値から下落中 = 押し目買いのチャンス")
            else:
                st.info("現在、押し目条件を満たすテーマはありません")

            st.markdown("##### 📋 全テーマ リスクリワード一覧")
            st.dataframe(
                rr_df, use_container_width=True, height=500,
                column_config={
                    "シャープ比": st.column_config.NumberColumn(format="%.2f"),
                    "ソルティノ比": st.column_config.NumberColumn(format="%.2f"),
                    "カルマー比": st.column_config.NumberColumn(format="%.2f"),
                    "最大DD(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "月次勝率(%)": st.column_config.NumberColumn(format="%.0f%%"),
                    "高値比(%)": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

            rr_plot = rr_df.dropna(subset=["月次勝率(%)"])
            if not rr_plot.empty:
                fig_rr = px.scatter(
                    rr_plot, x="年率ボラ(%)", y="年率リターン(%)",
                    text="テーマ", size="月次勝率(%)",
                    color="ソルティノ比", color_continuous_scale="RdYlGn",
                    title="リスク vs リターン（サイズ: 勝率、色: ソルティノ比）",
                )
                fig_rr.update_traces(textposition="top center", textfont_size=9)
                fig_rr.update_layout(height=450)
                st.plotly_chart(fig_rr, use_container_width=True)

# =====================================================================
# Tab 2: 累積リターンチャート
# =====================================================================
with tab2:
    st.subheader("テーマ別累積リターン推移")

    fig_cum = go.Figure()
    for col in theme_cum_filtered.columns:
        fig_cum.add_trace(go.Scatter(
            x=theme_cum_filtered.index,
            y=theme_cum_filtered[col] * 100,
            mode="lines", name=col,
            line=dict(width=2),
        ))
    fig_cum.update_layout(
        yaxis_title="累積リターン (%)",
        height=700,
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="top", y=-0.05, font=dict(size=11)),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=30, b=200),
    )

    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_cum, use_container_width=True)

    # テーマ間相関マトリクス
    st.subheader("テーマ間相関マトリクス（日次リターン）")
    corr = theme_ret_filtered.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text:.2f}",
        textfont=dict(size=10),
    ))
    fig_corr.update_layout(
        height=max(500, len(corr) * 25),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# =====================================================================
# Tab 3: テーマ詳細分析
# =====================================================================
with tab3:
    st.subheader("テーマ別詳細分析")
    detail_theme = st.selectbox("テーマを選択", cols_available)

    if detail_theme:
        info = THEMES[detail_theme]
        kabutan_kw = info.get("kabutan", detail_theme)
        kabutan_url = f"https://kabutan.jp/themes/?theme={quote(kabutan_kw)}"
        st.markdown(
            f"<a href='{kabutan_url}' target='_blank' style='display:inline-block;"
            f"padding:6px 16px;background:#1a1a2e;border-radius:6px;color:#4da6ff;"
            f"text-decoration:none;font-size:0.9rem;border:1px solid #333;"
            f"margin-bottom:10px;'>🔗 株探「{kabutan_kw}」テーマページを開く</a>",
            unsafe_allow_html=True
        )
        signals = generate_signal(theme_ret_filtered)
        sig_data = signals.get(detail_theme, {})

        # 解説コメント
        d1 = ret_1d.get(detail_theme, 0)
        w1 = ret_1w.get(detail_theme, 0)
        m1 = ret_1m.get(detail_theme, 0)
        commentary = generate_commentary(detail_theme, info, d1, w1, m1, sig_data)
        st.info(commentary)

        # 代表銘柄テーブル
        st.markdown("**代表銘柄**")
        stock_tickers = list(info["stocks"].keys())
        stock_names = list(info["stocks"].values())

        stock_prices = fetch_prices(stock_tickers, period=analysis_period)
        if not stock_prices.empty:


            stock_ret = stock_prices.pct_change().dropna()
            stock_perf = pd.DataFrame({
                "銘柄": [info["stocks"].get(t, t) for t in stock_ret.columns],
                "ティッカー": stock_ret.columns,
                "週次リターン(%)": (stock_ret.iloc[-5:].sum() * 100).values.round(2) if len(stock_ret) >= 5 else (stock_ret.sum() * 100).values.round(2),
                "月次リターン(%)": (stock_ret.iloc[-21:].sum() * 100).values.round(2) if len(stock_ret) >= 21 else (stock_ret.sum() * 100).values.round(2),
            })
            st.dataframe(stock_perf.sort_values("週次リターン(%)", ascending=False),
                         use_container_width=True, hide_index=True)

            # 個別銘柄チャート
            if show_individual:
                st.markdown("**個別銘柄 累積リターン**")
                stock_cum = (1 + stock_ret).cumprod() - 1
                fig_stock = go.Figure()
                for t in stock_cum.columns:
                    name = info["stocks"].get(t, t)
                    fig_stock.add_trace(go.Scatter(
                        x=stock_cum.index, y=stock_cum[t] * 100,
                        mode="lines", name=name, line=dict(width=2),
                    ))
                fig_stock.update_layout(
                    yaxis_title="累積リターン (%)", height=450,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    hovermode="x unified",
                )
                fig_stock.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(fig_stock, use_container_width=True)

        # OLSトレンド分析
        st.markdown("**OLSトレンド分析（直近60日）**")
        if detail_theme in theme_ret_filtered.columns:
            s = theme_ret_filtered[detail_theme].dropna()
            cum_s = s.cumsum()
            if len(cum_s) >= 10:
                window = min(60, len(cum_s))
                y = cum_s.iloc[-window:].values
                x = np.arange(len(y))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                trend_line = intercept + slope * x

                fig_ols = go.Figure()
                fig_ols.add_trace(go.Scatter(
                    x=cum_s.iloc[-window:].index, y=y * 100,
                    mode="lines", name="累積リターン",
                    line=dict(color="#26a69a", width=2),
                ))
                fig_ols.add_trace(go.Scatter(
                    x=cum_s.iloc[-window:].index, y=trend_line * 100,
                    mode="lines", name=f"OLS回帰 (傾き={slope*1000:.3f}, R²={r_value**2:.3f})",
                    line=dict(color="#ff9800", width=2, dash="dash"),
                ))
                fig_ols.update_layout(
                    yaxis_title="累積リターン (%)", height=400,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_ols, use_container_width=True)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("OLS傾き (×1000)", f"{slope*1000:.4f}")
                col_b.metric("R²", f"{r_value**2:.4f}")
                col_c.metric("p値", f"{p_value:.4f}")

# =====================================================================
# Tab 4: 日米リードラグ分析
# =====================================================================
with tab4:
    st.subheader("🌏 日米セクター リードラグ分析")
    st.caption("米国ETFと日本テーマの時差相関を分析。正のラグ=米国が先行、負のラグ=日本が先行")

    with st.spinner("米国ETFデータ取得中..."):
        us_prices = fetch_us_etfs(period=analysis_period)

    if not us_prices.empty:
        us_ret = us_prices.pct_change().dropna()
        lead_lag_df = calc_lead_lag(theme_ret_filtered, us_ret, max_lag=5)

        if not lead_lag_df.empty:
            lead_lag_display = lead_lag_df.copy()
            lead_lag_display["correlation"] = pd.to_numeric(lead_lag_display["correlation"], errors="coerce").round(3)
            lead_lag_display["best_lag"] = pd.to_numeric(lead_lag_display["best_lag"], errors="coerce").fillna(0).astype(int)
            lead_lag_display.columns = ["米国ETF", "最適ラグ(日)", "相関係数"]

            # ラグの解釈列を追加
            def interpret_lag(row):
                lag = row["最適ラグ(日)"]
                corr = row["相関係数"]
                if abs(corr) < 0.1:
                    return "相関なし"
                if lag > 0:
                    return f"米国が{lag}日先行"
                elif lag < 0:
                    return f"日本が{-lag}日先行"
                else:
                    return "同時連動"

            lead_lag_display["解釈"] = lead_lag_display.apply(interpret_lag, axis=1)
            st.dataframe(lead_lag_display, use_container_width=True)

            # リードラグ可視化
            fig_lag = make_subplots(rows=1, cols=2,
                                   subplot_titles=["最適ラグ（日数）", "相関係数"])
            themes_sorted = lead_lag_df.sort_values("best_lag")

            fig_lag.add_trace(go.Bar(
                x=themes_sorted.index,
                y=themes_sorted["best_lag"].astype(float),
                marker_color=["#f44336" if v < 0 else "#26a69a"
                              for v in themes_sorted["best_lag"].astype(float)],
                name="ラグ",
            ), row=1, col=1)

            fig_lag.add_trace(go.Bar(
                x=themes_sorted.index,
                y=themes_sorted["correlation"].astype(float),
                marker_color=["#2196f3" if v > 0 else "#ff9800"
                              for v in themes_sorted["correlation"].astype(float)],
                name="相関",
            ), row=1, col=2)

            fig_lag.update_layout(
                height=450, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                showlegend=False,
            )
            st.plotly_chart(fig_lag, use_container_width=True)

            # 詳細クロス相関
            st.markdown("**個別テーマ クロス相関詳細**")
            lag_detail_theme = st.selectbox("テーマ選択（リードラグ）",
                                           [t for t in cols_available if t in lead_lag_df.index])
            if lag_detail_theme:
                etf = THEMES[lag_detail_theme]["us_etf"]
                if etf in us_ret.columns and lag_detail_theme in theme_ret_filtered.columns:
                    jp = theme_ret_filtered[lag_detail_theme].dropna()
                    us = us_ret[etf].dropna()
                    common = jp.index.intersection(us.index)
                    if len(common) >= 20:
                        jp_c = jp.loc[common].values
                        us_c = us.loc[common].values
                        lags = list(range(-10, 11))
                        corrs = []
                        for lag in lags:
                            if lag > 0:
                                c = np.corrcoef(us_c[:-lag], jp_c[lag:])[0, 1] if len(us_c) > lag else 0
                            elif lag < 0:
                                c = np.corrcoef(us_c[-lag:], jp_c[:lag])[0, 1] if len(jp_c) > -lag else 0
                            else:
                                c = np.corrcoef(us_c, jp_c)[0, 1]
                            corrs.append(c)

                        fig_xcorr = go.Figure()
                        fig_xcorr.add_trace(go.Bar(
                            x=lags, y=corrs,
                            marker_color=["#26a69a" if c > 0 else "#f44336" for c in corrs],
                        ))
                        fig_xcorr.update_layout(
                            title=f"{lag_detail_theme} vs {etf} クロス相関",
                            xaxis_title="ラグ（日）正=米国先行",
                            yaxis_title="相関係数",
                            height=350, template="plotly_dark",
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        )
                        fig_xcorr.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_xcorr, use_container_width=True)
    else:
        st.warning("米国ETFデータの取得に失敗しました")

# =====================================================================
# Tab 5: 売買シグナル
# =====================================================================
with tab5:
    st.subheader("🎯 テーマ別 売買判断シグナル")
    st.caption("モメンタム + OLSトレンド + シャープ比で総合判定")

    signals = generate_signal(theme_ret_filtered)

    # 買い推奨・売り推奨を分離
    buy_themes = {k: v for k, v in signals.items() if v.get("score", 0) > 0.5}
    sell_themes = {k: v for k, v in signals.items() if v.get("score", 0) < -0.5}
    neutral_themes = {k: v for k, v in signals.items()
                      if -0.5 <= v.get("score", 0) <= 0.5}

    col_buy, col_sell = st.columns(2)
    with col_buy:
        st.markdown("### 🟢 買い推奨テーマ")
        if buy_themes:
            for theme, data in sorted(buy_themes.items(),
                                       key=lambda x: x[1]["score"], reverse=True):
                st.markdown(
                    f"<div style='background:#1b3a1b;padding:10px;border-radius:8px;"
                    f"margin-bottom:6px;border-left:4px solid #00c853;color:#ffffff'>"
                    f"<b>{theme}</b> — {data['signal']}<br>"
                    f"スコア: {data['score']} | 5日: {data.get('r5',0):+.2f}% | "
                    f"20日: {data.get('r20',0):+.2f}%</div>",
                    unsafe_allow_html=True
                )

        else:
            st.info("現在、明確な買い推奨テーマはありません")

    with col_sell:
        st.markdown("### 🔴 売り推奨テーマ")
        if sell_themes:
            for theme, data in sorted(sell_themes.items(),
                                       key=lambda x: x[1]["score"]):
                st.markdown(
                    f"<div style='background:#3a1b1b;padding:10px;border-radius:8px;"
                    f"margin-bottom:6px;border-left:4px solid #f44336;color:#ffffff'>"
                    f"<b>{theme}</b> — {data['signal']}<br>"
                    f"スコア: {data['score']} | 5日: {data.get('r5',0):+.2f}% | "
                    f"20日: {data.get('r20',0):+.2f}%</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("現在、明確な売り推奨テーマはありません")

    # 全テーマシグナル一覧
    st.subheader("全テーマ シグナル一覧")
    sig_rows = []
    for theme, data in signals.items():
        sig_rows.append({
            "テーマ": theme,
            "シグナル": data.get("signal", "N/A"),
            "総合スコア": data.get("score", 0),
            "5日リターン(%)": data.get("r5", 0),
            "20日リターン(%)": data.get("r20", 0),
            "OLS傾き": data.get("slope", 0),
            "シャープ比": data.get("sharpe", 0),
        })
    sig_df = pd.DataFrame(sig_rows).sort_values("総合スコア", ascending=False)
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

    # スコア棒グラフ
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Bar(
        x=sig_df["テーマ"], y=sig_df["総合スコア"],
        marker_color=[
            "#00c853" if v > 2 else "#2196f3" if v > 0.5
            else "#9e9e9e" if v > -0.5 else "#ff9800" if v > -2
            else "#f44336" for v in sig_df["総合スコア"]
        ],
    ))
    fig_sig.update_layout(
        yaxis_title="総合スコア", height=400,
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    fig_sig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sig.add_hline(y=2, line_dash="dot", line_color="#00c853",
                      annotation_text="強い買い")
    fig_sig.add_hline(y=-2, line_dash="dot", line_color="#f44336",
                      annotation_text="強い売り")
    st.plotly_chart(fig_sig, use_container_width=True)

    # モメンタムスコア
    st.subheader("モメンタムスコア（短期重視加重）")
    mom = calc_momentum_score(theme_ret_filtered)
    fig_mom = go.Figure()
    fig_mom.add_trace(go.Bar(
        x=mom.index, y=mom.values * 100,
        marker_color=["#26a69a" if v > 0 else "#ef5350" for v in mom.values],
    ))
    fig_mom.update_layout(
        yaxis_title="モメンタムスコア (%)", height=400,
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    fig_mom.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_mom, use_container_width=True)

# =====================================================================
# Tab 6: HOTニュース
# =====================================================================
with tab6:
    st.subheader("🔥 HOT経済ニュース × テーマ紐づけ")
    st.caption("5分ごとに自動更新 — ニュースからテーマ・関連銘柄を自動検出")

    col_news_ctrl1, col_news_ctrl2 = st.columns([1, 3])
    with col_news_ctrl1:
        news_auto = st.toggle("自動更新（5分）", value=False, key="news_auto")
    with col_news_ctrl2:
        news_filter = st.text_input("🔍 ニュース絞り込み", "", key="news_filter")

    with st.spinner("📰 最新ニュース取得中..."):
        news_articles = fetch_news_rss()

    if news_articles:
        # フィルタ
        if news_filter:
            nf = news_filter.lower()
            news_articles = [a for a in news_articles if nf in a["title"].lower() or nf in a.get("desc", "").lower()]

        # テーマ紐づけ済み / 未紐づけ に分離
        hot_articles = []
        other_articles = []
        for article in news_articles:
            matches = match_themes_to_article(article["title"], article.get("desc", ""))
            article["theme_matches"] = matches
            article["attention"] = calc_attention_score(article, matches)
            if matches:
                hot_articles.append(article)
            else:
                other_articles.append(article)

        # ソート選択
        sort_col1, sort_col2 = st.columns([1, 3])
        with sort_col1:
            sort_mode = st.radio("並び順", ["🕐 新しい順", "🔥 注目度順"], horizontal=True, key="news_sort")

        if sort_mode == "🔥 注目度順":
            hot_articles.sort(key=lambda x: x.get("attention", 0), reverse=True)
        else:
            hot_articles.sort(key=lambda x: x["dt"] if x.get("dt") else datetime.min, reverse=True)

        # HOTニュース（テーマ紐づけあり）
        st.markdown(f"### 🎯 テーマ関連ニュース（{len(hot_articles)}件）")
        if hot_articles:
            for article in hot_articles:
                matches = article["theme_matches"]
                attention = article.get("attention", 0)

                # 注目度バッジ
                if attention >= 15:
                    att_badge = "<span style='background:#d50000;color:#fff;padding:1px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;margin-right:6px;'>🔥🔥🔥 超注目</span>"
                elif attention >= 10:
                    att_badge = "<span style='background:#ff6d00;color:#fff;padding:1px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;margin-right:6px;'>🔥🔥 注目</span>"
                elif attention >= 5:
                    att_badge = "<span style='background:#f9a825;color:#000;padding:1px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;margin-right:6px;'>🔥 話題</span>"
                else:
                    att_badge = ""

                # テーマバッジ
                theme_badges = ""
                for m in matches:
                    theme_name = m["theme"]
                    kw = THEMES.get(theme_name, {}).get("kabutan", theme_name)
                    kb_url = f"https://kabutan.jp/themes/?theme={quote(kw)}"
                    theme_badges += (
                        f"<a href='{kb_url}' target='_blank' style='display:inline-block;"
                        f"padding:2px 8px;margin:2px;background:#0d47a1;border-radius:10px;"
                        f"color:#90caf9;text-decoration:none;font-size:0.78rem;'>"
                        f"🎯 {theme_name}</a>"
                    )

                # 関連銘柄（上位テーマの代表銘柄）
                related_stocks = []
                for m in matches[:2]:
                    info = THEMES.get(m["theme"], {})
                    for ticker, name in list(info.get("stocks", {}).items())[:3]:
                        if name not in related_stocks:
                            related_stocks.append(name)
                stock_text = "、".join(related_stocks[:6])

                st.markdown(
                    f"<div style='background:#1a1a2e;padding:12px 16px;border-radius:8px;"
                    f"margin-bottom:6px;border-left:4px solid {'#d50000' if attention >= 15 else '#ff6d00' if attention >= 10 else '#f9a825' if attention >= 5 else '#555'};'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:start;'>"
                    f"<div style='flex:1;'>"
                    f"<div>{att_badge}"
                    f"<span style='color:#888;font-size:0.72rem;'>スコア {attention}</span></div>"
                    f"<a href='{article['link']}' target='_blank' style='color:#fff;"
                    f"text-decoration:none;font-size:0.95rem;font-weight:500;'>"
                    f"📰 {article['title']}</a>"
                    f"<div style='margin-top:6px;'>{theme_badges}</div>"
                    f"<div style='color:#aaa;font-size:0.78rem;margin-top:4px;'>"
                    f"💼 関連: {stock_text}</div>"
                    f"</div>"
                    f"<div style='color:#888;font-size:0.75rem;white-space:nowrap;margin-left:12px;'>"
                    f"🕐 {article['time']}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("現在テーマに紐づくニュースはありません")

        # テーマ別ニュース集計
        st.markdown("### 📊 テーマ別ニュース出現数")
        theme_counts = {}
        for article in hot_articles:
            for m in article["theme_matches"]:
                t = m["theme"]
                theme_counts[t] = theme_counts.get(t, 0) + 1
        if theme_counts:
            tc_sorted = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Bar(
                x=[t[0] for t in tc_sorted],
                y=[t[1] for t in tc_sorted],
                marker_color="#ff6d00",
            ))
            fig_tc.update_layout(
                yaxis_title="ニュース件数", height=350,
                template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            )
            st.plotly_chart(fig_tc, use_container_width=True)

        # その他のニュース（折りたたみ）
        if other_articles:
            with st.expander(f"📋 その他の経済ニュース（{len(other_articles)}件）"):
                for article in other_articles[:30]:
                    st.markdown(
                        f"<div style='padding:4px 0;border-bottom:1px solid #222;'>"
                        f"<a href='{article['link']}' target='_blank' style='color:#ccc;"
                        f"text-decoration:none;font-size:0.88rem;'>"
                        f"{article['title']}</a>"
                        f" <span style='color:#666;font-size:0.75rem;'>{article['time']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.warning("ニュースの取得に失敗しました。しばらく待ってから再試行してください。")

    # ニュース自動更新
    if news_auto:
        time.sleep(300)
        st.rerun()

# =====================================================================
# Tab 7: マイポートフォリオ
# =====================================================================

# デフォルト: 空のポートフォリオ
# session_state で管理: {"ticker": "name", ...}

# 全上場銘柄辞書（ローカルキャッシュ + 株探から段階的に構築）
_STOCK_DICT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_dict_cache.json")

def _fetch_stock_list_from_jpx():
    """JPXの上場銘柄一覧CSVから辞書を構築"""
    try:
        # JPXのCSV版（東証上場銘柄一覧）
        url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return {}
        # xlsバイナリをpandasで読む（xlrdが必要）
        import io
        df = pd.read_excel(io.BytesIO(resp.content))
        code_col = [c for c in df.columns if "コード" in str(c)]
        name_col = [c for c in df.columns if "銘柄名" in str(c)]
        if code_col and name_col:
            result = {}
            for _, row in df.iterrows():
                code = str(row[code_col[0]]).strip()
                name = str(row[name_col[0]]).strip()
                if code.isdigit() and name and name != "nan":
                    result[f"{code}.T"] = name
            return result
    except Exception:
        pass
    return {}

def _build_basic_stock_dict():
    """主要銘柄の基本辞書（フォールバック用）"""
    return {
        "1605.T": "INPEX", "1801.T": "大成建設", "1802.T": "大林組",
        "1812.T": "鹿島建設", "1925.T": "大和ハウス工業", "1928.T": "積水ハウス",
        "2002.T": "日清製粉グループ本社", "2269.T": "明治HD",
        "2413.T": "エムスリー", "2502.T": "アサヒグループHD",
        "2503.T": "キリンHD", "2802.T": "味の素", "2914.T": "日本たばこ産業",
        "3038.T": "神戸物産", "3064.T": "MonotaRO", "3092.T": "ZOZO",
        "3099.T": "三越伊勢丹HD", "3382.T": "セブン&アイHD",
        "3407.T": "旭化成", "3436.T": "SUMCO",
        "3778.T": "さくらインターネット", "3923.T": "ラクス",
        "3993.T": "PKSHA Technology",
        "4004.T": "レゾナック", "4005.T": "住友化学",
        "4063.T": "信越化学工業", "4183.T": "三井化学",
        "4188.T": "三菱ケミカルグループ", "4204.T": "積水化学工業",
        "4307.T": "野村総合研究所", "4385.T": "メルカリ",
        "4443.T": "Sansan", "4452.T": "花王", "4478.T": "フリー",
        "4502.T": "武田薬品工業", "4503.T": "アステラス製薬",
        "4507.T": "塩野義製薬", "4519.T": "中外製薬",
        "4523.T": "エーザイ", "4543.T": "テルモ",
        "4568.T": "第一三共", "4578.T": "大塚HD",
        "4661.T": "オリエンタルランド", "4684.T": "オービック",
        "4689.T": "LINEヤフー", "4704.T": "トレンドマイクロ",
        "4755.T": "楽天グループ", "4901.T": "富士フイルムHD",
        "4911.T": "資生堂",
        "5019.T": "出光興産", "5020.T": "ENEOSホールディングス",
        "5108.T": "ブリヂストン", "5201.T": "AGC",
        "5401.T": "日本製鉄", "5411.T": "JFEホールディングス",
        "5631.T": "日本製鋼所", "5706.T": "三井金属鉱業",
        "5711.T": "三菱マテリアル", "5713.T": "住友金属鉱山",
        "5714.T": "DOWAホールディングス", "5727.T": "東邦チタニウム",
        "5801.T": "古河電気工業", "5802.T": "住友電気工業", "5803.T": "フジクラ",
        "6098.T": "リクルートHD", "6146.T": "ディスコ",
        "6178.T": "日本郵政", "6273.T": "SMC",
        "6301.T": "小松製作所", "6326.T": "クボタ",
        "6361.T": "荏原製作所", "6367.T": "ダイキン工業",
        "6368.T": "オルガノ", "6370.T": "栗田工業",
        "6479.T": "ミネベアミツミ",
        "6501.T": "日立製作所", "6503.T": "三菱電機",
        "6504.T": "富士電機", "6506.T": "安川電機",
        "6526.T": "ソシオネクスト", "6594.T": "ニデック",
        "6645.T": "オムロン", "6674.T": "GSユアサ",
        "6701.T": "NEC", "6702.T": "富士通",
        "6723.T": "ルネサスエレクトロニクス",
        "6752.T": "パナソニックHD", "6758.T": "ソニーグループ",
        "6762.T": "TDK", "6810.T": "マクセル",
        "6841.T": "横河電機", "6857.T": "アドバンテスト",
        "6861.T": "キーエンス", "6869.T": "シスメックス",
        "6902.T": "デンソー", "6920.T": "レーザーテック",
        "6954.T": "ファナック", "6971.T": "京セラ",
        "6976.T": "太陽誘電", "6981.T": "村田製作所",
        "6988.T": "日東電工",
        "7003.T": "三井E&S", "7011.T": "三菱重工業",
        "7012.T": "川崎重工業", "7013.T": "IHI",
        "7182.T": "ゆうちょ銀行", "7186.T": "コンコルディアFG",
        "7201.T": "日産自動車", "7203.T": "トヨタ自動車",
        "7261.T": "マツダ", "7267.T": "本田技研工業",
        "7269.T": "スズキ", "7270.T": "SUBARU",
        "7309.T": "シマノ", "7453.T": "良品計画",
        "7532.T": "パン・パシフィック・インターナショナルHD",
        "7721.T": "東京計器", "7729.T": "東京精密",
        "7731.T": "ニコン", "7733.T": "オリンパス",
        "7735.T": "SCREENホールディングス",
        "7741.T": "HOYA", "7751.T": "キヤノン",
        "7832.T": "バンダイナムコHD", "7974.T": "任天堂",
        "8001.T": "伊藤忠商事", "8002.T": "丸紅",
        "8015.T": "豊田通商", "8031.T": "三井物産",
        "8035.T": "東京エレクトロン", "8053.T": "住友商事",
        "8058.T": "三菱商事", "8088.T": "岩谷産業",
        "8233.T": "高島屋", "8267.T": "イオン",
        "8306.T": "三菱UFJFG", "8309.T": "三井住友トラスト",
        "8316.T": "三井住友FG", "8331.T": "千葉銀行",
        "8354.T": "ふくおかFG", "8355.T": "静岡銀行",
        "8411.T": "みずほFG", "8591.T": "オリックス",
        "8601.T": "大和証券グループ本社", "8604.T": "野村HD",
        "8630.T": "SOMPOホールディングス",
        "8725.T": "MS&ADインシュアランスグループHD",
        "8750.T": "第一生命HD", "8766.T": "東京海上HD",
        "8801.T": "三井不動産", "8802.T": "三菱地所",
        "8830.T": "住友不動産",
        "9005.T": "東急", "9009.T": "京成電鉄",
        "9020.T": "JR東日本", "9021.T": "JR西日本",
        "9022.T": "JR東海", "9024.T": "西武HD",
        "9101.T": "日本郵船", "9104.T": "商船三井",
        "9107.T": "川崎汽船",
        "9201.T": "日本航空", "9202.T": "ANA HD",
        "9412.T": "スカパーJSAT",
        "9432.T": "NTT", "9433.T": "KDDI", "9434.T": "ソフトバンク",
        "9501.T": "東京電力HD", "9502.T": "中部電力",
        "9503.T": "関西電力", "9519.T": "レノバ",
        "9532.T": "大阪ガス", "9602.T": "東宝",
        "9613.T": "NTTデータグループ",
        "9684.T": "スクウェア・エニックスHD", "9697.T": "カプコン",
        "9735.T": "セコム", "9766.T": "コナミグループ",
        "9843.T": "ニトリHD", "9983.T": "ファーストリテイリング",
        "9984.T": "ソフトバンクグループ",
    }

def _load_stock_dict():
    """銘柄辞書を取得（キャッシュ → JPX → 基本辞書の順）"""
    # 1. キャッシュが新しければそれを使う
    try:
        if os.path.exists(_STOCK_DICT_CACHE_FILE):
            age = time.time() - os.path.getmtime(_STOCK_DICT_CACHE_FILE)
            if age < 86400 * 7:  # 7日以内
                with open(_STOCK_DICT_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if len(data) > 100:
                        return data
    except Exception:
        pass

    # 2. JPXから取得を試みる
    jpx_dict = _fetch_stock_list_from_jpx()
    if len(jpx_dict) > 100:
        try:
            with open(_STOCK_DICT_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(jpx_dict, f, ensure_ascii=False)
        except Exception:
            pass
        return jpx_dict

    # 3. 古いキャッシュがあれば使う
    try:
        if os.path.exists(_STOCK_DICT_CACHE_FILE):
            with open(_STOCK_DICT_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return data
    except Exception:
        pass

    # 4. 最終フォールバック: 基本辞書
    basic = _build_basic_stock_dict()
    try:
        with open(_STOCK_DICT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(basic, f, ensure_ascii=False)
    except Exception:
        pass
    return basic

STOCK_DICT = _load_stock_dict()
if STOCK_DICT:
    st.sidebar.caption(f"📋 銘柄辞書: {len(STOCK_DICT)}件")

# 検索用: 「銘柄名（コード）」のリストを構築
_SEARCH_OPTIONS = sorted(
    [f"{name}（{ticker.replace('.T', '')}）" for ticker, name in STOCK_DICT.items()],
    key=lambda x: x
)
_OPTION_TO_TICKER = {}
for ticker, name in STOCK_DICT.items():
    label = f"{name}（{ticker.replace('.T', '')}）"
    _OPTION_TO_TICKER[label] = ticker

# ポートフォリオ永続化（JSONファイル）
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_portfolio.json")

def save_portfolio(stocks):
    """ポートフォリオをJSONファイルに保存"""
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(stocks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_portfolio():
    """ポートフォリオをJSONファイルから読み込み"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

@st.cache_data(ttl=86400, show_spinner=False)
def resolve_stock_name(ticker):
    """ティッカーから日本語銘柄名を取得（株探 → JPX辞書 → yfinance の順）"""
    code = ticker.replace(".T", "")

    # 1. 株探からスクレイピング
    try:
        url = f"https://kabutan.jp/stock/?code={code}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.encoding = resp.apparent_encoding
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # <div class="company_block"><h3>銘柄名</h3> のパターン
            h3 = soup.select_one("div.company_block h3")
            if h3:
                name = h3.get_text(strip=True)
                if name and len(name) >= 2:
                    return name
            # <title>タグから抽出: "銘柄名【コード】..." のパターン
            title_tag = soup.find("title")
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                if "【" in title_text:
                    name = title_text.split("【")[0].strip()
                    if name and len(name) >= 2:
                        return name
    except Exception:
        pass

    # 2. JPX辞書
    if ticker in STOCK_DICT:
        return STOCK_DICT[ticker]

    # 3. yfinance（最終手段）
    try:
        info = yf.Ticker(ticker).info
        long_name = info.get("longName", "")
        short_name = info.get("shortName", "")
        for n in [long_name, short_name]:
            if n and any(ord(c) > 0x3000 for c in n):
                return n
        return long_name or short_name or ticker
    except Exception:
        return ticker


@st.cache_data(ttl=600)
def fetch_portfolio_data(tickers_tuple, period="6mo"):
    """ポートフォリオ銘柄のOHLCV+出来高データを取得"""
    tickers = list(tickers_tuple)
    result = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            result[t] = df
        except Exception:
            continue
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_info(ticker):
    """決算日・業績データを取得"""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # 次回決算日
        next_earnings = None
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        next_earnings = pd.Timestamp(ed[0]) if isinstance(ed, list) else pd.Timestamp(ed)
                elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
                    if hasattr(ed, 'iloc'):
                        next_earnings = pd.Timestamp(ed.iloc[0])
                    else:
                        next_earnings = pd.Timestamp(ed)
        except Exception:
            pass

        # 直近の業績（四半期）
        earnings_score = 0
        earnings_detail = ""
        try:
            # 売上・利益の成長率
            rev_growth = info.get("revenueGrowth")
            earn_growth = info.get("earningsGrowth")
            profit_margin = info.get("profitMargins")
            rec = info.get("recommendationMean")  # 1=Strong Buy, 5=Sell

            # 会社予想 vs 実績
            # trailingEps vs forwardEps で予想の方向性
            trailing_eps = info.get("trailingEps")
            forward_eps = info.get("forwardEps")

            details = []

            # 売上成長率
            if rev_growth is not None:
                if rev_growth > 0.15:
                    earnings_score += 20
                    details.append(f"売上+{rev_growth*100:.0f}%")
                elif rev_growth > 0.05:
                    earnings_score += 10
                    details.append(f"売上+{rev_growth*100:.0f}%")
                elif rev_growth > 0:
                    earnings_score += 5
                    details.append(f"売上+{rev_growth*100:.0f}%")
                else:
                    earnings_score -= 10
                    details.append(f"売上{rev_growth*100:.0f}%")

            # 利益成長率
            if earn_growth is not None:
                if earn_growth > 0.2:
                    earnings_score += 25
                    details.append(f"利益+{earn_growth*100:.0f}%")
                elif earn_growth > 0.05:
                    earnings_score += 12
                    details.append(f"利益+{earn_growth*100:.0f}%")
                elif earn_growth > 0:
                    earnings_score += 5
                    details.append(f"利益+{earn_growth*100:.0f}%")
                else:
                    earnings_score -= 15
                    details.append(f"利益{earn_growth*100:.0f}%")

            # EPS予想（Forward > Trailing = 増益予想）
            if trailing_eps and forward_eps and trailing_eps > 0:
                eps_growth = (forward_eps - trailing_eps) / abs(trailing_eps)
                if eps_growth > 0.1:
                    earnings_score += 15
                    details.append(f"EPS予想+{eps_growth*100:.0f}%")
                elif eps_growth > 0:
                    earnings_score += 5
                    details.append(f"EPS予想+{eps_growth*100:.0f}%")
                elif eps_growth < -0.1:
                    earnings_score -= 15
                    details.append(f"EPS予想{eps_growth*100:.0f}%")

            # 利益率
            if profit_margin is not None:
                if profit_margin > 0.15:
                    earnings_score += 10
                elif profit_margin > 0.05:
                    earnings_score += 5
                elif profit_margin < 0:
                    earnings_score -= 10

            # アナリスト評価
            if rec is not None:
                if rec <= 2.0:
                    earnings_score += 10
                elif rec >= 4.0:
                    earnings_score -= 10

            earnings_score = max(min(earnings_score, 100), -100)
            earnings_detail = " / ".join(details) if details else "データなし"

        except Exception:
            earnings_detail = "取得失敗"

        return {
            "next_earnings": next_earnings,
            "earnings_score": earnings_score,
            "earnings_detail": earnings_detail,
        }
    except Exception:
        return {
            "next_earnings": None,
            "earnings_score": 0,
            "earnings_detail": "取得失敗",
        }


@st.cache_data(ttl=600)
def fetch_tdnet_for_portfolio(target_date_str, stock_codes):
    """TDnetから適時開示を取得し、ポートフォリオ銘柄でフィルタ"""
    import re
    date_fmt = target_date_str.replace("-", "")
    url = f"https://www.release.tdnet.info/inbs/I_list_001_{date_fmt}.html"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.encoding = resp.apparent_encoding
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        disclosures = []

        # 方法1: テーブル行ベースのパース
        for tr in soup.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 4:
                # TDnetのテーブル: 時刻, コード, 会社名, 開示タイトル, ...
                cell_texts = [td.get_text(strip=True) for td in tds]
                # 時刻パターンを探す
                time_val = ""
                code_val = ""
                company_val = ""
                title_val = ""
                for j, ct in enumerate(cell_texts):
                    if re.match(r"^\d{2}:\d{2}$", ct) and not time_val:
                        time_val = ct
                    elif re.match(r"^\d{4,5}$", ct) and not code_val:
                        code_val = ct
                if time_val and code_val and code_val in stock_codes:
                    # 会社名とタイトルを残りのセルから取得
                    remaining = [ct for ct in cell_texts if ct != time_val and ct != code_val and len(ct) > 1]
                    company_val = remaining[0] if remaining else ""
                    title_val = remaining[1] if len(remaining) > 1 else (remaining[0] if remaining else "")
                    # リンクからタイトルを取得（より正確）
                    links = tr.find_all("a")
                    for a in links:
                        a_text = a.get_text(strip=True)
                        if len(a_text) > 5:
                            title_val = a_text
                            break
                    disclosures.append({
                        "time": time_val, "code": code_val,
                        "company": company_val, "title": title_val,
                    })

        # 方法2: テーブルで見つからなかった場合、テキストベースのフォールバック
        if not disclosures:
            text = soup.get_text("\n")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            i = 0
            while i < len(lines):
                if re.match(r"^\d{2}:\d{2}$", lines[i]):
                    t = lines[i]
                    # 後続の行からコードを探す
                    code_found = ""
                    company = ""
                    title = ""
                    for k in range(1, min(6, len(lines) - i)):
                        candidate = lines[i + k].strip()
                        if re.match(r"^\d{4,5}$", candidate) and not code_found:
                            code_found = candidate
                        elif code_found and not company:
                            company = candidate
                        elif company and not title:
                            title = candidate
                            break
                    if code_found and code_found in stock_codes:
                        disclosures.append({
                            "time": t, "code": code_found,
                            "company": company, "title": title,
                        })
                    i += 1
                else:
                    i += 1

        return disclosures
    except Exception:
        return []


def _dow_judge(close_values, window):
    """指定窓サイズでダウ理論判定（高値安値の切上げ/切下げ）"""
    close = close_values
    if len(close) < window * 3:
        return None, None
    highs = []
    lows = []
    half = window // 2
    for i in range(half, len(close) - half):
        seg = close[max(0, i - half):i + half + 1]
        if close[i] == max(seg):
            highs.append((i, close[i]))
        if close[i] == min(seg):
            lows.append((i, close[i]))
    if len(highs) < 2 or len(lows) < 2:
        return None, None
    h1, h2 = highs[-2][1], highs[-1][1]
    l1, l2 = lows[-2][1], lows[-1][1]
    if h2 > h1 and l2 > l1:
        return "上昇", f"高値切上({h1:.0f}→{h2:.0f})"
    elif h2 < h1 and l2 < l1:
        return "下降", f"高値切下({h1:.0f}→{h2:.0f})"
    else:
        return "レンジ", "方向感なし"


def analyze_dow_theory(df):
    """ダウ理論: 日足(10日窓)・週足相当(5日集約)・月足相当(20日集約)で総合判定"""
    if df is None or len(df) < 20:
        return "データ不足", "gray", ""

    close = df["Close"].values

    # 日足ベース（窓10日 = 短期スイング）
    d_trend, d_detail = _dow_judge(close, 10) or (None, None)

    # 週足相当（5日ごとの終値を使用）
    weekly = close[::5] if len(close) >= 25 else close
    w_trend, w_detail = _dow_judge(weekly, 4) or (None, None)

    # 月足相当（20日ごとの終値を使用）
    monthly = close[::20] if len(close) >= 60 else None
    m_trend, m_detail = (None, None)
    if monthly is not None and len(monthly) >= 4:
        m_trend, m_detail = _dow_judge(monthly, 2) or (None, None)

    # 総合判定（長期 > 中期 > 短期の重み）
    score = 0
    details = []
    for label, trend, detail in [("月", m_trend, m_detail), ("週", w_trend, w_detail), ("日", d_trend, d_detail)]:
        if trend == "上昇":
            score += 1
            details.append(f"{label}↑")
        elif trend == "下降":
            score -= 1
            details.append(f"{label}↓")
        elif trend == "レンジ":
            details.append(f"{label}→")

    detail_str = " ".join(details) if details else ""

    if not details:
        return "判定不可", "gray", ""
    elif score >= 2:
        return "上昇トレンド", "#00c853", detail_str
    elif score <= -2:
        return "下降トレンド", "#f44336", detail_str
    elif score == 1:
        return "やや上昇", "#26a69a", detail_str
    elif score == -1:
        return "やや下降", "#ef5350", detail_str
    else:
        return "レンジ", "#9e9e9e", detail_str


def analyze_breakout(df):
    """チャートブレイク検出（直近高値/安値ブレイク、ボリンジャーバンド±2σ/±3σ）"""
    if df is None or len(df) < 25:
        return "データ不足", "gray"

    close = df["Close"]
    last = float(close.iloc[-1])

    # 直近20日の高値・安値
    high_20 = float(close.iloc[-21:-1].max())
    low_20 = float(close.iloc[-21:-1].min())

    # ボリンジャーバンド（20日）
    ma20 = float(close.iloc[-20:].mean())
    std20 = float(close.iloc[-20:].std())
    bb_2u = ma20 + 2 * std20
    bb_2l = ma20 - 2 * std20
    bb_3u = ma20 + 3 * std20
    bb_3l = ma20 - 3 * std20

    signals = []
    if last > high_20:
        signals.append("📈 20日高値ブレイク")
    if last < low_20:
        signals.append("📉 20日安値ブレイク")

    # BB ±3σ（過熱感強い）
    if last > bb_3u:
        signals.append("🔥 BB+3σ超 過熱")
    elif last > bb_2u:
        signals.append("⚡ BB+2σ超")
    if last < bb_3l:
        signals.append("🧊 BB-3σ超 売られすぎ")
    elif last < bb_2l:
        signals.append("⚡ BB-2σ超")

    if not signals:
        if last > ma20:
            return "MA20上", "#26a69a"
        else:
            return "MA20下", "#ef5350"

    # 色: 過熱=赤系、売られすぎ=青系、ブレイク=緑/赤
    if any("3σ超 過熱" in s for s in signals):
        color = "#d50000"
    elif any("3σ超 売られ" in s for s in signals):
        color = "#2962ff"
    elif any("高値" in s or "+2σ" in s for s in signals):
        color = "#00c853"
    else:
        color = "#f44336"
    return " / ".join(signals), color


def calc_theoretical_price(df):
    """簡易理論株価（PER×EPS推定、移動平均乖離率ベース）"""
    if df is None or len(df) < 60:
        return None, None

    close = df["Close"]
    last = float(close.iloc[-1])

    # 60日移動平均からの理論値（平均回帰ベース）
    ma60 = float(close.iloc[-60:].mean())
    # 120日があれば使う
    if len(close) >= 120:
        ma120 = float(close.iloc[-120:].mean())
    else:
        ma120 = ma60

    # 理論株価 = 長期MAと中期MAの加重平均
    theoretical = ma60 * 0.6 + ma120 * 0.4
    deviation = (last - theoretical) / theoretical * 100

    return round(theoretical, 1), round(deviation, 1)


def calc_supply_demand(df):
    """需給分析（出来高トレンド + 価格出来高相関）"""
    if df is None or len(df) < 20 or "Volume" not in df.columns:
        return "データ不足", "gray", 0

    vol = df["Volume"].iloc[-20:]
    close = df["Close"].iloc[-20:]

    # 出来高5日平均 vs 20日平均
    vol_5 = float(vol.iloc[-5:].mean())
    vol_20 = float(vol.mean())
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1

    # 価格上昇日の出来高 vs 下落日の出来高
    ret = close.pct_change().dropna()
    vol_aligned = vol.iloc[-len(ret):]
    up_vol = float(vol_aligned[ret > 0].mean()) if (ret > 0).any() else 0
    down_vol = float(vol_aligned[ret <= 0].mean()) if (ret <= 0).any() else 0
    buy_pressure = up_vol / down_vol if down_vol > 0 else 2.0

    # 需給スコア
    score = (vol_ratio - 1) * 50 + (buy_pressure - 1) * 30
    score = np.clip(score, -100, 100)

    if score > 20:
        return "買い優勢", "#00c853", round(score, 1)
    elif score > 0:
        return "やや買い", "#26a69a", round(score, 1)
    elif score > -20:
        return "やや売り", "#ef5350", round(score, 1)
    else:
        return "売り優勢", "#f44336", round(score, 1)


def calc_stock_momentum(df):
    """個別銘柄モメンタムスコア"""
    if df is None or len(df) < 20:
        return 0

    close = df["Close"]
    ret = close.pct_change().dropna()

    r5 = float(ret.iloc[-5:].sum()) if len(ret) >= 5 else 0
    r20 = float(ret.iloc[-20:].sum()) if len(ret) >= 20 else float(ret.sum())
    r60 = float(ret.iloc[-60:].sum()) if len(ret) >= 60 else float(ret.sum())

    # RSI (14日)
    if len(ret) >= 14:
        gains = ret.iloc[-14:].clip(lower=0).mean()
        losses = (-ret.iloc[-14:].clip(upper=0)).mean()
        rsi = float(gains / (gains + losses) * 100) if (gains + losses) > 0 else 50
    else:
        rsi = 50

    # 総合モメンタム
    score = r5 * 100 * 0.35 + r20 * 100 * 0.25 + r60 * 100 * 0.15 + (rsi - 50) * 0.25
    return round(np.clip(score, -30, 30), 2)


def calc_daijunkan_macd(df):
    """大循環MACD分析（小次郎講師方式）
    3本のEMA(5,20,40)から3本のMACDを算出し、ステージ判定・売買シグナルを返す。
    Returns: (stage, signal_text, signal_emoji, detail)
    """
    if df is None or len(df) < 45:
        return None, "データ不足", "⚪", ""

    close = df["Close"]
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema40 = close.ewm(span=40, adjust=False).mean()

    # 3本のMACD
    macd1 = ema5 - ema20   # 短期-中期
    macd2 = ema5 - ema40   # 短期-長期
    macd3 = ema20 - ema40  # 中期-長期（帯MACD）

    m1 = float(macd1.iloc[-1])
    m2 = float(macd2.iloc[-1])
    m3 = float(macd3.iloc[-1])

    # 前日値（クロス検出用）
    m1_prev = float(macd1.iloc[-2])
    m2_prev = float(macd2.iloc[-2])
    m3_prev = float(macd3.iloc[-2])

    # EMA順序でステージ判定
    e5 = float(ema5.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e40 = float(ema40.iloc[-1])

    if e5 >= e20 >= e40:
        stage = 1  # 上昇期
    elif e20 >= e5 >= e40:
        stage = 2  # 上昇期の調整
    elif e20 >= e40 >= e5:
        stage = 3  # 下降への転換
    elif e40 >= e20 >= e5:
        stage = 4  # 下降期
    elif e40 >= e5 >= e20:
        stage = 5  # 下降期の調整
    elif e5 >= e40 >= e20:
        stage = 6  # 上昇への転換
    else:
        stage = 0

    # シグナル判定
    details = []

    # 0ラインクロス検出
    if m1_prev <= 0 < m1:
        details.append("MACD1↑0突破")
    elif m1_prev >= 0 > m1:
        details.append("MACD1↓0割れ")
    if m2_prev <= 0 < m2:
        details.append("MACD2↑0突破")
    elif m2_prev >= 0 > m2:
        details.append("MACD2↓0割れ")
    if m3_prev <= 0 < m3:
        details.append("MACD3↑0突破")
    elif m3_prev >= 0 > m3:
        details.append("MACD3↓0割れ")

    # 帯MACDの勢い（傾き）
    if len(macd3) >= 5:
        m3_slope = float(macd3.iloc[-1] - macd3.iloc[-5])
        if abs(m3) > 0 and abs(m3_slope / max(abs(m3), 0.01)) < 0.1:
            details.append("帯MACD横ばい⚠️")

    # ステージに基づくシグナル
    stage_map = {
        1: ("S1 上昇期 🟢", "🟢"),
        2: ("S2 上昇調整 🔵", "🔵"),
        3: ("S3 下降転換 🟠", "🟠"),
        4: ("S4 下降期 🔴", "🔴"),
        5: ("S5 下降調整 🟡", "🟡"),
        6: ("S6 上昇転換 🟣", "🟣"),
    }
    sig_text, sig_emoji = stage_map.get(stage, ("判定不可", "⚪"))
    detail_str = " / ".join(details) if details else ""

    return stage, sig_text, sig_emoji, detail_str


def calc_oshime_signal(df):
    """押し目エントリーシグナル（バックテスト検証済み: 20日的中率68%, PF1.95）。

    中長期上昇トレンド中の短期調整局面を複合条件で検出する。
    条件の合致数に応じてスコアリングし、5条件中4つ以上で「押し目買い」判定。

    検出条件:
      1. トレンド: MA60 > MA120 かつ MA60上向き（20日変化率 > 0）
      2. 押し目: MA20からの乖離率 -5%〜0%（MA20付近まで調整）
      3. RSI反転: RSI(5)が前日より上昇（短期的な反転兆候）
      4. MACD改善: MACDヒストグラムが前日より改善
      5. 陽線: 終値 > 始値（買い圧力の回復）

    Returns:
        dict: signal, emoji, score(-30〜+30), detail, hits(合致数/5),
              rsi14, dev_ma20, drawdown
    """
    empty = {"signal": "様子見", "emoji": "⚪", "score": 0, "detail": "",
             "hits": 0, "rsi14": None, "dev_ma20": None, "drawdown": None,
             "N": None, "V": None, "E": None, "NT": None, "wave_type": "—"}
    if df is None or len(df) < 65:
        return empty

    close = df["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    # ── 指標計算 ──
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean() if len(close) >= 120 else None

    # MA乖離率
    dev_ma20 = float((last - ma20.iloc[-1]) / ma20.iloc[-1] * 100) if ma20.iloc[-1] > 0 else None

    # MA60の傾き（20日変化率）
    ma60_slope = float((ma60.iloc[-1] - ma60.iloc[-21]) / ma60.iloc[-21] * 100) if len(ma60.dropna()) > 21 and ma60.iloc[-21] > 0 else None

    # RSI(14)
    delta = close.diff()
    gain14 = delta.clip(lower=0).rolling(14).mean()
    loss14 = (-delta.clip(upper=0)).rolling(14).mean()
    rs14 = gain14 / loss14.replace(0, np.nan)
    rsi14_series = 100 - (100 / (1 + rs14))
    rsi14 = float(rsi14_series.iloc[-1]) if not np.isnan(rsi14_series.iloc[-1]) else None

    # RSI(5)
    gain5 = delta.clip(lower=0).rolling(5).mean()
    loss5 = (-delta.clip(upper=0)).rolling(5).mean()
    rs5 = gain5 / loss5.replace(0, np.nan)
    rsi5_series = 100 - (100 / (1 + rs5))
    rsi5_now = float(rsi5_series.iloc[-1]) if not np.isnan(rsi5_series.iloc[-1]) else None
    rsi5_prev = float(rsi5_series.iloc[-2]) if not np.isnan(rsi5_series.iloc[-2]) else None

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_hist_now = float(macd_hist.iloc[-1])
    macd_hist_prev = float(macd_hist.iloc[-2])

    # 陽線判定
    bullish = last > float(df["Open"].iloc[-1])

    # 直近高値からの下落率
    high_20d = float(close.iloc[-20:].max())
    drawdown = round((last - high_20d) / high_20d * 100, 1)

    # ── 5条件チェック ──
    hits = 0
    details = []

    # 1. トレンド: MA60 > MA120 かつ MA60上向き（MA120がない場合はMA60の傾きのみ）
    trend_ok = False
    if ma120 is not None and len(ma60.dropna()) > 0 and len(ma120.dropna()) > 0:
        trend_ok = float(ma60.iloc[-1]) > float(ma120.iloc[-1]) and ma60_slope is not None and ma60_slope > 0
    elif len(ma60.dropna()) > 0:
        # MA120がない場合: MA60上向き かつ 株価>MA60
        trend_ok = ma60_slope is not None and ma60_slope > 0 and last > float(ma60.iloc[-1])
    if trend_ok:
        hits += 1
        details.append("トレンド↑")
    else:
        details.append("トレンド✗")

    # 2. MA20押し: 乖離率 -5%〜0%
    pullback_ok = dev_ma20 is not None and -5 <= dev_ma20 <= 0
    if pullback_ok:
        hits += 1
        details.append(f"MA20押し{dev_ma20:+.1f}%")
    elif dev_ma20 is not None:
        details.append(f"MA20乖離{dev_ma20:+.1f}%")

    # 3. RSI(5)反転
    rsi_turn_ok = rsi5_now is not None and rsi5_prev is not None and rsi5_now > rsi5_prev
    if rsi_turn_ok:
        hits += 1
        details.append("RSI反転↑")
    else:
        details.append("RSI反転✗")

    # 4. MACDヒストグラム改善
    macd_ok = macd_hist_now > macd_hist_prev
    if macd_ok:
        hits += 1
        details.append("MACD改善↑")
    else:
        details.append("MACD改善✗")

    # 5. 陽線
    if bullish:
        hits += 1
        details.append("陽線○")
    else:
        details.append("陰線✗")

    # ── スコアリング ──
    result = dict(empty)
    result["hits"] = hits
    result["rsi14"] = round(rsi14, 1) if rsi14 is not None else None
    result["dev_ma20"] = round(dev_ma20, 1) if dev_ma20 is not None else None
    result["drawdown"] = drawdown

    # ── 波動理論 N/V/E/NT値（環境認識用） ──
    swing_window = 10
    close_arr = close.values.astype(float)
    n_bars = len(close_arr)
    swing_highs = []
    swing_lows = []
    if n_bars > swing_window * 2:
        for i in range(swing_window, n_bars - swing_window):
            wl = close_arr[i - swing_window:i]
            wr = close_arr[i + 1:i + swing_window + 1]
            if close_arr[i] >= max(wl) and close_arr[i] >= max(wr):
                swing_highs.append((i, close_arr[i]))
            if close_arr[i] <= min(wl) and close_arr[i] <= min(wr):
                swing_lows.append((i, close_arr[i]))

    # 上昇波動: 安値A → 高値B → 安値C（押し目）→ 目標D
    up_found = False
    for ci in range(len(swing_lows) - 1, -1, -1):
        if up_found:
            break
        c_idx, c_p = swing_lows[ci]
        for bi in range(len(swing_highs) - 1, -1, -1):
            if up_found:
                break
            b_idx, b_p = swing_highs[bi]
            if b_idx >= c_idx:
                continue
            for ai in range(len(swing_lows) - 1, -1, -1):
                a_idx, a_p = swing_lows[ai]
                if a_idx >= b_idx:
                    continue
                if a_p < b_p and c_p < b_p and c_p > a_p:
                    result["wave_type"] = "上昇"
                    result["N"] = round(c_p + (b_p - a_p), 1)
                    result["V"] = round(b_p + (b_p - c_p), 1)
                    result["E"] = round(b_p + (b_p - a_p), 1)
                    result["NT"] = round(c_p + (c_p - a_p), 1)
                    up_found = True
                    break

    # 上昇波動が見つからなければ下降波動を探す
    if not up_found:
        down_found = False
        for ci in range(len(swing_highs) - 1, -1, -1):
            if down_found:
                break
            c_idx, c_p = swing_highs[ci]
            for bi in range(len(swing_lows) - 1, -1, -1):
                if down_found:
                    break
                b_idx, b_p = swing_lows[bi]
                if b_idx >= c_idx:
                    continue
                for ai in range(len(swing_highs) - 1, -1, -1):
                    a_idx, a_p = swing_highs[ai]
                    if a_idx >= b_idx:
                        continue
                    if a_p > b_p and c_p > b_p and c_p < a_p:
                        result["wave_type"] = "下降"
                        result["N"] = round(c_p - (a_p - b_p), 1)
                        result["V"] = round(b_p - (c_p - b_p), 1)
                        result["E"] = round(b_p - (a_p - b_p), 1)
                        result["NT"] = round(c_p - (c_p - b_p), 1)
                        down_found = True
                        break

    if hits == 5 and trend_ok:
        result["signal"] = "押し目買い🎯"
        result["emoji"] = "🟢"
        result["score"] = 25
    elif hits == 4 and trend_ok:
        result["signal"] = "押し目接近"
        result["emoji"] = "🔵"
        result["score"] = 15
    elif hits >= 3 and trend_ok:
        result["signal"] = "調整中"
        result["emoji"] = "🟡"
        result["score"] = 5
    elif not trend_ok and hits >= 3:
        result["signal"] = "トレンド待ち"
        result["emoji"] = "⚪"
        result["score"] = 0
    elif not trend_ok:
        result["signal"] = "様子見"
        result["emoji"] = "⚪"
        result["score"] = -5
    else:
        result["signal"] = "条件不足"
        result["emoji"] = "⚪"
        result["score"] = 0

    result["detail"] = " / ".join(details)
    return result


def generate_composite_signal(dow, breakout_text, deviation, demand_score, momentum, wave_score=0):
    """総合売買シグナル生成"""
    score = 0

    # ダウ理論
    if dow == "上昇トレンド":
        score += 25
    elif dow == "下降トレンド":
        score -= 25

    # ブレイクアウト
    if "高値ブレイク" in breakout_text or "上限突破" in breakout_text:
        score += 15
    elif "安値ブレイク" in breakout_text or "下限突破" in breakout_text:
        score -= 15
    elif "MA20上" in breakout_text:
        score += 5
    elif "MA20下" in breakout_text:
        score -= 5

    # 理論株価乖離
    if deviation is not None:
        if deviation < -10:
            score += 15  # 割安
        elif deviation < -5:
            score += 8
        elif deviation > 10:
            score -= 15  # 割高
        elif deviation > 5:
            score -= 8

    # 需給
    score += demand_score * 0.3

    # モメンタム
    score += momentum * 1.5

    # 一目均衡表 波動理論
    score += wave_score * 0.5

    score = np.clip(score, -100, 100)

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


with tab7:
    st.subheader("📋 マイポートフォリオ 総合分析")
    st.caption("登録銘柄のテクニカル・理論株価・需給・モメンタムから総合売買シグナルを算出")

    # session_state 初期化（ファイルから復元）
    if "my_portfolio" not in st.session_state:
        st.session_state["my_portfolio"] = load_portfolio()

    my_stocks = st.session_state["my_portfolio"]

    # ローマ字名の銘柄を日本語名に自動修正
    _needs_save = False
    for ticker, name in list(my_stocks.items()):
        # ASCII文字のみ or ティッカーそのまま → 日本語名を再取得
        if name == ticker or (name and all(ord(c) < 0x3000 for c in name)):
            jp_name = resolve_stock_name(ticker)
            if jp_name != ticker and jp_name != name:
                my_stocks[ticker] = jp_name
                _needs_save = True
    if _needs_save:
        st.session_state["my_portfolio"] = my_stocks
        save_portfolio(my_stocks)
    st.markdown("#### ➕ 銘柄を追加")

    # 銘柄名検索（辞書から候補表示）
    selected = st.selectbox(
        "🔍 銘柄名で検索（辞書にある銘柄）",
        options=[""] + _SEARCH_OPTIONS,
        index=0,
        key="stock_search",
        placeholder="銘柄名を入力して検索...",
    )
    if st.button("➕ 検索結果から追加", key="add_search_btn") and selected:
        ticker = _OPTION_TO_TICKER.get(selected)
        if ticker:
            if ticker in my_stocks:
                st.warning(f"{STOCK_DICT.get(ticker, ticker)} は既に登録済みです")
            else:
                my_stocks[ticker] = STOCK_DICT[ticker]
                st.session_state["my_portfolio"] = my_stocks
                save_portfolio(my_stocks)
                st.rerun()
    st.caption("辞書にない銘柄は証券コードで直接追加できます")
    add_col1, add_col2 = st.columns([2, 1])
    with add_col1:
        new_ticker_input = st.text_input(
            "証券コード（例: 9560）",
            key="new_ticker_input",
            placeholder="9560",
        )
    with add_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        add_clicked = st.button("➕ コードで追加", key="add_stock_btn", use_container_width=True)

    if add_clicked and new_ticker_input:
        raw = new_ticker_input.strip()
        ticker = raw if "." in raw else f"{raw}.T"
        ticker = ticker.upper()
        if ticker in my_stocks:
            st.warning(f"{my_stocks[ticker]} は既に登録済みです")
        else:
            with st.spinner(f"{ticker} の銘柄情報を取得中..."):
                name = resolve_stock_name(ticker)
            if name == ticker:
                st.warning(f"{ticker} の銘柄情報が取得できませんでした。ティッカーが正しいか確認してください。追加はされます。")
            my_stocks[ticker] = name
            st.session_state["my_portfolio"] = my_stocks
            save_portfolio(my_stocks)
            st.rerun()
    if my_stocks:
        st.markdown(f"#### 📊 登録銘柄（{len(my_stocks)}件）")
        # 3列で表示
        del_cols = st.columns(3)
        tickers_to_delete = []
        for idx, (ticker, name) in enumerate(my_stocks.items()):
            col = del_cols[idx % 3]
            with col:
                code = ticker.replace(".T", "")
                if st.button(f"❌ {name}（{code}）", key=f"del_{ticker}", use_container_width=True):
                    tickers_to_delete.append(ticker)

        if tickers_to_delete:
            for t in tickers_to_delete:
                my_stocks.pop(t, None)
            st.session_state["my_portfolio"] = my_stocks
            save_portfolio(my_stocks)
            st.rerun()

        st.divider()

        # データ取得 & 分析
        with st.spinner(f"📈 {len(my_stocks)}銘柄のデータ取得中..."):
            pf_data = fetch_portfolio_data(tuple(my_stocks.keys()), period="6mo")

        if not pf_data:
            st.error("データ取得に失敗しました。")
        else:
            # 全銘柄分析
            results = []
            for ticker, name in my_stocks.items():
                df = pf_data.get(ticker)
                if df is None or df.empty:
                    continue

                last_price = float(df["Close"].iloc[-1])
                prev_price = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_price
                day_chg = (last_price - prev_price) / prev_price * 100

                dow_trend, dow_color, dow_detail = analyze_dow_theory(df)
                breakout_text, breakout_color = analyze_breakout(df)
                theoretical, deviation = calc_theoretical_price(df)
                demand_label, demand_color, demand_score = calc_supply_demand(df)
                momentum = calc_stock_momentum(df)

                # 決算情報
                ei = fetch_earnings_info(ticker)
                next_earn = ei["next_earnings"]
                earn_score = ei["earnings_score"]
                earn_detail = ei["earnings_detail"]

                # 決算日までの日数
                days_to_earn = None
                if next_earn is not None:
                    try:
                        today = pd.Timestamp.now().normalize()
                        next_earn = pd.Timestamp(next_earn).tz_localize(None) if next_earn.tzinfo else next_earn
                        days_to_earn = (next_earn - today).days
                    except Exception:
                        pass

                signal, signal_color, composite_score = generate_composite_signal(
                    dow_trend, breakout_text, deviation, demand_score, momentum
                )

                results.append({
                    "ticker": ticker,
                    "name": name,
                    "themes": guess_themes_by_name(ticker, name),
                    "price": last_price,
                    "day_chg": day_chg,
                    "dow": dow_trend,
                    "dow_color": dow_color,
                    "dow_detail": dow_detail,
                    "breakout": breakout_text,
                    "breakout_color": breakout_color,
                    "theoretical": theoretical,
                    "deviation": deviation,
                    "demand": demand_label,
                    "demand_color": demand_color,
                    "demand_score": demand_score,
                    "momentum": momentum,
                    "signal": signal,
                    "signal_color": signal_color,
                    "composite": composite_score,
                    "next_earnings": next_earn,
                    "days_to_earn": days_to_earn,
                    "earn_score": earn_score,
                    "earn_detail": earn_detail,
                })

            # スコア順にソート
            results.sort(key=lambda x: x["composite"], reverse=True)

            # サマリーメトリクス
            buy_count = sum(1 for r in results if r["composite"] > 10)
            sell_count = sum(1 for r in results if r["composite"] < -10)
            neutral_count = len(results) - buy_count - sell_count

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("分析銘柄数", len(results))
            mc2.metric("買いシグナル", f"{buy_count}銘柄", delta=None)
            mc3.metric("中立", f"{neutral_count}銘柄")
            mc4.metric("売りシグナル", f"{sell_count}銘柄", delta=None)

            # シグナル一覧テーブル
            st.markdown("### 🎯 総合売買シグナル一覧")
            for r in results:
                chg_color = "#00c853" if r["day_chg"] >= 0 else "#f44336"
                dev_text = f"{r['deviation']:+.1f}%" if r['deviation'] is not None else "N/A"
                dev_color = "#00c853" if r['deviation'] is not None and r['deviation'] < -5 else "#f44336" if r['deviation'] is not None and r['deviation'] > 5 else "#aaa"
                theo_text = f"¥{r['theoretical']:,.0f}" if r['theoretical'] is not None else "N/A"

                # 株探リンク
                code = r["ticker"].replace(".T", "")
                kabutan_stock_url = f"https://kabutan.jp/stock/?code={code}"

                # テーマバッジ
                themes = r.get("themes", [])
                theme_badges_html = ""
                if themes:
                    for th in themes:
                        kw = THEMES.get(th, {}).get("kabutan", th)
                        kb_url = f"https://kabutan.jp/themes/?theme={quote(kw)}"
                        theme_badges_html += (
                            f"<a href='{kb_url}' target='_blank' style='display:inline-block;"
                            f"padding:1px 6px;margin:1px;background:#1a237e;border-radius:8px;"
                            f"color:#7986cb;text-decoration:none;font-size:0.68rem;'>"
                            f"{th}</a>"
                        )

                # 決算日表示の準備
                dte = r.get("days_to_earn")
                ne = r.get("next_earnings")
                es = r.get("earn_score", 0)
                ed = r.get("earn_detail", "")
                if ne is not None:
                    earn_date_str = ne.strftime("%m/%d")
                    if dte is not None and 0 <= dte <= 7:
                        earn_date_html = f"<span style='background:#d50000;color:#fff;padding:1px 6px;border-radius:6px;font-size:0.75rem;font-weight:bold;'>📅 {earn_date_str} ({dte}日後)</span>"
                    elif dte is not None and dte < 0:
                        earn_date_html = f"<span style='color:#666;font-size:0.75rem;'>{earn_date_str}(済)</span>"
                    elif dte is not None:
                        earn_date_html = f"<span style='color:#aaa;font-size:0.75rem;'>{earn_date_str} ({dte}日後)</span>"
                    else:
                        earn_date_html = f"<span style='color:#aaa;font-size:0.75rem;'>{earn_date_str}</span>"
                else:
                    earn_date_html = "<span style='color:#666;font-size:0.75rem;'>未定</span>"

                es_color = "#00c853" if es > 15 else "#26a69a" if es > 0 else "#f44336" if es < -15 else "#ef5350" if es < 0 else "#888"

                st.markdown(
                    f"<div style='background:#1a1a2e;padding:10px 14px;border-radius:8px;"
                    f"margin-bottom:4px;border-left:5px solid {r['signal_color']};'>"
                    # 上段: グリッドで各指標
                    f"<div style='display:grid;grid-template-columns:190px 100px 90px 160px 110px 75px 75px 120px;align-items:center;gap:6px;'>"
                    # 銘柄名・価格
                    f"<div style='overflow:hidden;'>"
                    f"<a href='{kabutan_stock_url}' target='_blank' style='color:#fff;text-decoration:none;font-weight:600;font-size:0.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;'>"
                    f"{r['name']}</a>"
                    f"<span style='color:#888;font-size:0.72rem;'>{code}</span>"
                    f" <span style='font-size:1rem;color:#fff;'>¥{r['price']:,.0f}</span>"
                    f" <span style='color:{chg_color};font-size:0.8rem;'>{r['day_chg']:+.2f}%</span>"
                    f"</div>"
                    # シグナル
                    f"<div style='text-align:center;'>"
                    f"<div style='color:{r['signal_color']};font-weight:bold;font-size:0.85rem;'>{r['signal']}</div>"
                    f"<div style='color:#888;font-size:0.7rem;'>スコア {r['composite']}</div>"
                    f"</div>"
                    # ダウ理論
                    f"<div>"
                    f"<div style='color:#888;font-size:0.68rem;'>ダウ理論</div>"
                    f"<div style='color:{r['dow_color']};font-size:0.8rem;'>{r['dow']}</div>"
                    f"<div style='color:#666;font-size:0.68rem;'>{r['dow_detail']}</div>"
                    f"</div>"
                    # ブレイク/BB
                    f"<div>"
                    f"<div style='color:#888;font-size:0.68rem;'>ブレイク/BB</div>"
                    f"<div style='color:{r['breakout_color']};font-size:0.8rem;'>{r['breakout']}</div>"
                    f"</div>"
                    # 理論株価
                    f"<div>"
                    f"<div style='color:#888;font-size:0.68rem;'>理論株価</div>"
                    f"<div style='font-size:0.8rem;'>{theo_text}"
                    f" <span style='color:{dev_color};'>({dev_text})</span></div>"
                    f"</div>"
                    # 需給
                    f"<div>"
                    f"<div style='color:#888;font-size:0.68rem;'>需給</div>"
                    f"<div style='color:{r['demand_color']};font-size:0.8rem;'>{r['demand']}</div>"
                    f"</div>"
                    # モメンタム
                    f"<div>"
                    f"<div style='color:#888;font-size:0.68rem;'>モメンタム</div>"
                    f"<div style='color:{'#00c853' if r['momentum'] > 0 else '#f44336'};font-size:0.8rem;'>{r['momentum']:+.1f}</div>"
                    f"</div>"
                    # 決算
                    f"<div>"
                    f"<div style='color:#888;font-size:0.68rem;'>次回決算</div>"
                    f"<div>{earn_date_html}</div>"
                    f"<div style='color:{es_color};font-size:0.72rem;'>業績 {es:+d}点</div>"
                    f"</div>"
                    f"</div>"
                    # 下段: テーマバッジ + 業績詳細
                    + (f"<div style='margin-top:4px;'>{theme_badges_html}"
                       + (f" <span style='color:#888;font-size:0.68rem;margin-left:8px;'>📊 {ed}</span>" if ed and ed != "データなし" and ed != "取得失敗" else "")
                       + f"</div>" if theme_badges_html or (ed and ed not in ("データなし", "取得失敗")) else "")
                    + f"</div>",
                    unsafe_allow_html=True
                )

            # --- 適時開示（IR）セクション ---
            st.markdown("### 📋 登録銘柄のIR・適時開示")

            # 四季報リンク一覧
            st.markdown(
                "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;'>"
                + "".join(
                    f"<a href='https://shikiho.toyokeizai.net/stocks/{t.replace('.T','')}' target='_blank' "
                    f"style='padding:3px 10px;background:#1a1a2e;border-radius:4px;color:#4da6ff;"
                    f"text-decoration:none;font-size:0.78rem;border:1px solid #333;'>"
                    f"📊 {n}</a>"
                    for t, n in my_stocks.items()
                )
                + "</div>",
                unsafe_allow_html=True
            )

            # GoogleニュースRSSで各銘柄のIR/決算ニュースを取得
            @st.cache_data(ttl=600)
            def fetch_ir_news_for_portfolio(stock_names_tuple):
                """登録銘柄のIR関連ニュースをGoogleニュースRSSから取得"""
                import xml.etree.ElementTree as ET
                from email.utils import parsedate_to_datetime as _pdt
                all_news = []
                seen = set()
                for name in stock_names_tuple:
                    # 短い名前は検索精度が低いのでスキップ
                    if len(name) < 2:
                        continue
                    search_q = quote(f"{name} 決算 OR 業績 OR 配当 OR IR OR 開示")
                    rss_url = f"https://news.google.com/rss/search?q={search_q}&hl=ja&gl=JP&ceid=JP:ja"
                    try:
                        resp = requests.get(rss_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                        if resp.status_code != 200:
                            continue
                        root = ET.fromstring(resp.content)
                        for item in root.findall(".//item")[:5]:
                            title = item.findtext("title", "").strip()
                            link = item.findtext("link", "").strip()
                            pub = item.findtext("pubDate", "").strip()
                            if not title or title in seen:
                                continue
                            seen.add(title)
                            dt_obj = None
                            time_str = ""
                            if pub:
                                try:
                                    dt_obj = _pdt(pub)
                                    time_str = dt_obj.strftime("%m/%d %H:%M")
                                except Exception:
                                    time_str = pub[:16]
                            all_news.append({
                                "stock": name, "title": title,
                                "link": link, "time": time_str, "dt": dt_obj,
                            })
                    except Exception:
                        continue
                # 新しい順
                all_news.sort(key=lambda x: x["dt"] if x.get("dt") else datetime.min, reverse=True)
                return all_news[:50]

            pf_name_list = tuple(n for n in my_stocks.values() if n)
            with st.spinner("📰 IR関連ニュース取得中..."):
                ir_news = fetch_ir_news_for_portfolio(pf_name_list)

            if ir_news:
                st.caption(f"📄 {len(ir_news)} 件のIR関連ニュース（登録銘柄）")
                for d in ir_news:
                    title_text = d["title"]
                    is_important = any(kw in title_text for kw in [
                        "決算", "業績", "配当", "自己株式", "株式分割",
                        "上方修正", "下方修正", "MBO", "TOB", "合併",
                        "増資", "減資", "特別損失", "特別利益", "開示",
                        "修正", "予想", "中間", "通期",
                    ])
                    border_color = "#ff6d00" if is_important else "#333"
                    badge = "<span style='background:#ff6d00;color:#fff;padding:1px 5px;border-radius:4px;font-size:0.65rem;margin-right:4px;'>重要</span>" if is_important else ""

                    st.markdown(
                        f"<div style='padding:8px 12px;border-radius:6px;margin-bottom:3px;"
                        f"border-left:3px solid {border_color};background:#1a1a2e;'>"
                        f"<div style='color:#fff;font-size:0.88rem;'>{badge}"
                        f"<a href='{d['link']}' target='_blank' style='color:#fff;text-decoration:none;'>"
                        f"📄 {title_text}</a></div>"
                        f"<div style='color:#888;font-size:0.72rem;margin-top:2px;'>"
                        f"🏢 {d['stock']}　🕐 {d['time']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("登録銘柄のIR関連ニュースが見つかりませんでした")

            # --- 出来高急増セクション ---
            st.markdown("### 📊 出来高急増アラート")
            st.caption("日足・週足・月足で出来高が平均の2倍以上に急増した銘柄を検出")

            vol_alerts = []
            for ticker, name in my_stocks.items():
                df = pf_data.get(ticker)
                if df is None or df.empty or "Volume" not in df.columns or len(df) < 25:
                    continue

                vol = df["Volume"].astype(float)
                close = df["Close"].astype(float)
                last_price = float(close.iloc[-1])
                prev_price = float(close.iloc[-2]) if len(close) >= 2 else last_price
                chg = (last_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

                alerts = []

                # 日足: 当日出来高 vs 20日平均
                vol_today = float(vol.iloc[-1])
                vol_avg_20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
                if vol_avg_20 > 0:
                    ratio_d = vol_today / vol_avg_20
                    if ratio_d >= 2.0:
                        alerts.append(("日足", round(ratio_d, 1), vol_today, vol_avg_20))

                # 週足相当: 直近5日合計 vs 前4週(20日)の週平均
                if len(vol) >= 25:
                    vol_week = float(vol.iloc[-5:].sum())
                    vol_prev_weeks = float(vol.iloc[-25:-5].mean()) * 5
                    if vol_prev_weeks > 0:
                        ratio_w = vol_week / vol_prev_weeks
                        if ratio_w >= 1.8:
                            alerts.append(("週足", round(ratio_w, 1), vol_week, vol_prev_weeks))

                # 月足相当: 直近20日合計 vs 前3ヶ月(60日)の月平均
                if len(vol) >= 80:
                    vol_month = float(vol.iloc[-20:].sum())
                    vol_prev_months = float(vol.iloc[-80:-20].mean()) * 20
                    if vol_prev_months > 0:
                        ratio_m = vol_month / vol_prev_months
                        if ratio_m >= 1.8:
                            alerts.append(("月足", round(ratio_m, 1), vol_month, vol_prev_months))

                if alerts:
                    # 最大倍率でソート用
                    max_ratio = max(a[1] for a in alerts)
                    vol_alerts.append({
                        "ticker": ticker, "name": name,
                        "price": last_price, "chg": chg,
                        "alerts": alerts, "max_ratio": max_ratio,
                    })

            vol_alerts.sort(key=lambda x: x["max_ratio"], reverse=True)

            if vol_alerts:
                # ニュース取得関数（キャッシュ付き）
                @st.cache_data(ttl=600)
                def _fetch_vol_news(stock_name):
                    import xml.etree.ElementTree as ET
                    from email.utils import parsedate_to_datetime as _pdt2
                    try:
                        q = quote(f"{stock_name} 株")
                        rss = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
                        resp = requests.get(rss, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
                        if resp.status_code != 200:
                            return ""
                        root = ET.fromstring(resp.content)
                        items = root.findall(".//item")
                        if items:
                            title = items[0].findtext("title", "").strip()
                            link = items[0].findtext("link", "").strip()
                            pub = items[0].findtext("pubDate", "")
                            time_s = ""
                            try:
                                time_s = _pdt2(pub).strftime("%m/%d")
                            except Exception:
                                pass
                            return f"<a href='{link}' target='_blank' style='color:#90caf9;text-decoration:none;font-size:0.75rem;'>📰 {title[:60]}{'...' if len(title)>60 else ''}</a> <span style='color:#666;font-size:0.68rem;'>{time_s}</span>"
                    except Exception:
                        pass
                    return ""

                st.caption(f"🔔 {len(vol_alerts)} 銘柄で出来高急増を検出")
                for va in vol_alerts:
                    code = va["ticker"].replace(".T", "")
                    chg_color = "#00c853" if va["chg"] >= 0 else "#f44336"

                    # アラートバッジ
                    alert_badges = ""
                    for timeframe, ratio, vol_cur, vol_avg in va["alerts"]:
                        if ratio >= 3.0:
                            bg = "#d50000"
                        elif ratio >= 2.5:
                            bg = "#ff6d00"
                        else:
                            bg = "#f9a825"
                        alert_badges += (
                            f"<span style='background:{bg};color:#fff;padding:2px 8px;"
                            f"border-radius:8px;font-size:0.72rem;font-weight:bold;margin-right:4px;'>"
                            f"{timeframe} {ratio:.1f}倍</span>"
                        )

                    # ニュース取得
                    news_html = _fetch_vol_news(va["name"])

                    st.markdown(
                        f"<div style='background:#1a1a2e;padding:10px 14px;border-radius:8px;"
                        f"margin-bottom:4px;border-left:5px solid {'#d50000' if va['max_ratio']>=3 else '#ff6d00' if va['max_ratio']>=2.5 else '#f9a825'};'>"
                        f"<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"
                        f"<div style='min-width:160px;'>"
                        f"<a href='https://kabutan.jp/stock/?code={code}' target='_blank' "
                        f"style='color:#fff;text-decoration:none;font-weight:600;font-size:0.9rem;'>"
                        f"{va['name']}</a>"
                        f" <span style='color:#888;font-size:0.72rem;'>{code}</span><br>"
                        f"<span style='color:#fff;'>¥{va['price']:,.0f}</span>"
                        f" <span style='color:{chg_color};font-size:0.8rem;'>{va['chg']:+.2f}%</span>"
                        f"</div>"
                        f"<div>{alert_badges}</div>"
                        f"</div>"
                        + (f"<div style='margin-top:4px;'>{news_html}</div>" if news_html else "")
                        + f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("現在、出来高急増の銘柄はありません")

            # スコア分布チャート
            st.markdown("### 📊 総合スコア分布")
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Bar(
                x=[r["name"] for r in results],
                y=[r["composite"] for r in results],
                marker_color=[r["signal_color"] for r in results],
                text=[r["signal"].split(" ")[0] for r in results],
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig_pf.update_layout(
                yaxis_title="総合スコア", height=450,
                template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(b=120),
            )
            fig_pf.add_hline(y=30, line_dash="dot", line_color="#00c853", annotation_text="強い買い")
            fig_pf.add_hline(y=-30, line_dash="dot", line_color="#f44336", annotation_text="強い売り")
            fig_pf.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pf, use_container_width=True)

            # データテーブル（ダウンロード用）
            with st.expander("📥 データテーブル（CSV出力可）"):
                table_df = pd.DataFrame([{
                    "銘柄": r["name"],
                    "ティッカー": r["ticker"],
                    "テーマ": " / ".join(r.get("themes", [])) if r.get("themes") else "-",
                    "株価": r["price"],
                    "日次変動%": round(r["day_chg"], 2),
                    "総合シグナル": r["signal"],
                    "総合スコア": r["composite"],
                    "ダウ理論": r["dow"],
                    "ブレイク": r["breakout"],
                    "理論株価": r["theoretical"],
                    "乖離率%": r["deviation"],
                    "需給": r["demand"],
                    "需給スコア": r["demand_score"],
                    "モメンタム": r["momentum"],
                    "次回決算": r["next_earnings"].strftime("%Y-%m-%d") if r.get("next_earnings") else "-",
                    "決算まで(日)": r.get("days_to_earn", "-"),
                    "業績スコア": r.get("earn_score", 0),
                    "業績詳細": r.get("earn_detail", ""),
                } for r in results])
                st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 上の入力欄から証券コードを入力して銘柄を追加してください（例: 7011, 8306, 6758）")

# =====================================================================
# Tab 8: ドラゴンキング分析
# =====================================================================

# --- DK統計検定のPython実装 ---
def dk_test_stat(vals, r):
    """DK検定統計量（Pisarenko & Sornette 2012）
    vals: 数値配列, r: DK候補数
    returns: (test_stat, p_value)
    """
    from scipy.stats import f as f_dist
    x = np.sort(vals)
    n = len(x)
    if n < r + 2 or r < 1:
        return 0, 1.0
    # 上位r個の合計 - r * (n-r)番目の値
    numerator = np.sum(x[-r:]) - r * x[n - r - 1]
    # 全体の合計 - n * 最小値
    denominator = np.sum(x) - n * x[0]
    if denominator == 0:
        return 0, 1.0
    T = numerator / denominator
    # F分布近似: df1=2r, df2=2(n-r)
    f_stat = T * (n - r) / r if r > 0 else 0
    p_value = 1 - f_dist.cdf(f_stat, 2 * r, 2 * (n - r))
    return round(float(f_stat), 4), round(float(p_value), 6)


def ss_stat(vals, r):
    """Sum-Sum検定統計量: 上位r個の合計 / 全体の合計"""
    x = np.sort(vals)
    total = np.sum(x)
    if total == 0:
        return 0
    return round(float(np.sum(x[-r:]) / total), 4)


def detect_super_exponential(prices, window=60):
    """超指数的成長の検出（バブルの兆候）
    log(price)に対して線形回帰 vs 2次回帰のR²比較
    """
    if len(prices) < window:
        return 0, "データ不足"
    y = np.log(prices[-window:])
    x = np.arange(len(y))

    # 線形フィット
    slope1, intercept1, r1, _, _ = stats.linregress(x, y)
    r2_linear = r1 ** 2

    # 2次フィット
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 2次項の係数が正 = 加速的上昇（バブル兆候）
    accel = coeffs[0]
    improvement = r2_quad - r2_linear

    if accel > 0 and improvement > 0.02:
        score = min(improvement * 100 + accel * 10000, 100)
        return round(score, 1), "加速的上昇検出"
    elif accel < 0 and improvement > 0.02:
        score = -min(improvement * 100 + abs(accel) * 10000, 100)
        return round(score, 1), "加速的下落検出"
    return 0, "通常"


@st.cache_data(ttl=600)
def fetch_dk_index_data(symbol, period="2y"):
    """指数データ取得"""
    try:
        df = yf.download(symbol, period=period, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def analyze_dragon_king(df, name):
    """指数に対するドラゴンキング総合分析"""
    if df is None or len(df) < 60:
        return None

    close = df["Close"]
    prices = close.values.astype(float)
    returns = np.diff(np.log(prices))  # 対数リターン

    # 絶対リターン（極端なイベント検出用）
    abs_ret = np.abs(returns)

    # 直近の値
    last_price = float(prices[-1])
    ret_1m = float((prices[-1] / prices[-21] - 1) * 100) if len(prices) > 21 else 0
    ret_3m = float((prices[-1] / prices[-63] - 1) * 100) if len(prices) > 63 else 0

    # 1. DK検定（直近250日のリターン分布で上位3個をテスト）
    recent_abs = abs_ret[-250:] if len(abs_ret) >= 250 else abs_ret
    dk_f, dk_p = dk_test_stat(recent_abs, r=3)

    # 2. SS統計量
    ss = ss_stat(recent_abs, r=3)

    # 3. 超指数的成長検出
    se_score, se_label = detect_super_exponential(prices, window=60)

    # 4. ボラティリティ・クラスタリング（直近20日 vs 全期間）
    vol_recent = float(np.std(returns[-20:])) if len(returns) >= 20 else 0
    vol_all = float(np.std(returns))
    vol_ratio = vol_recent / vol_all if vol_all > 0 else 1

    # 5. MA乖離率（200日線）
    if len(prices) >= 200:
        ma200 = float(np.mean(prices[-200:]))
        ma_dev = (last_price - ma200) / ma200 * 100
    else:
        ma200 = float(np.mean(prices))
        ma_dev = (last_price - ma200) / ma200 * 100

    # 6. 最大ドローダウン（直近1年）
    recent_prices = prices[-252:] if len(prices) >= 252 else prices
    peak = np.maximum.accumulate(recent_prices)
    drawdown = (recent_prices - peak) / peak * 100
    max_dd = float(np.min(drawdown))

    # --- バブル度スコア（0-100） ---
    bubble_score = 0
    # DK検定: p値が小さい = 極端なイベントが統計的に有意
    if dk_p < 0.01:
        bubble_score += 25
    elif dk_p < 0.05:
        bubble_score += 15
    elif dk_p < 0.1:
        bubble_score += 8

    # 超指数的成長
    if se_score > 0:
        bubble_score += min(se_score * 0.4, 25)

    # MA200乖離
    if ma_dev > 20:
        bubble_score += 20
    elif ma_dev > 10:
        bubble_score += 10
    elif ma_dev > 5:
        bubble_score += 5

    # ボラティリティ急上昇
    if vol_ratio > 2.0:
        bubble_score += 15
    elif vol_ratio > 1.5:
        bubble_score += 8

    # 急騰（3ヶ月で20%以上）
    if ret_3m > 20:
        bubble_score += 15
    elif ret_3m > 10:
        bubble_score += 8

    bubble_score = min(round(bubble_score), 100)

    # --- 暴落シグナル ---
    crash_signals = []
    if dk_p < 0.05:
        crash_signals.append("⚠️ DK検定有意（極端イベント検出）")
    if se_score > 30:
        crash_signals.append("🔥 超指数的上昇（バブル兆候）")
    if vol_ratio > 2.0:
        crash_signals.append("📊 ボラティリティ急上昇")
    if ma_dev > 20:
        crash_signals.append("📈 MA200から大幅乖離")
    if max_dd < -15:
        crash_signals.append("📉 大幅ドローダウン進行中")

    return {
        "name": name,
        "price": last_price,
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "dk_f": dk_f,
        "dk_p": dk_p,
        "ss": ss,
        "se_score": se_score,
        "se_label": se_label,
        "vol_ratio": round(vol_ratio, 2),
        "ma_dev": round(ma_dev, 1),
        "max_dd": round(max_dd, 1),
        "bubble_score": bubble_score,
        "crash_signals": crash_signals,
        "prices": prices,
        "returns": returns,
        "dates": df.index,
    }


DK_INDICES = {
    "S&P 500": "^GSPC",
    "日経225": "^N225",
    "NASDAQ": "^IXIC",
    "TOPIX": "1473.T",
    "東証グロース250": "2516.T",
    "ラッセル2000": "^RUT",
    "ドル/円": "USDJPY=X",
    "VIX": "^VIX",
    "米10年債": "^TNX",
    "金(Gold)": "GC=F",
    "原油(WTI)": "CL=F",
    "ビットコイン": "BTC-USD",
}

with tab8:
    st.subheader("🐉 ドラゴンキング分析 — バブル度・暴落シグナル")
    st.caption("Sornette理論に基づく極端イベント検出。DK検定・超指数的成長・ボラティリティ分析で市場の過熱/暴落リスクを評価")

    dk_period = st.selectbox("分析期間", ["1y", "2y", "5y"], index=1, key="dk_period")

    with st.spinner("🐉 指数データ取得・分析中..."):
        dk_results = []
        for name, symbol in DK_INDICES.items():
            df = fetch_dk_index_data(symbol, period=dk_period)
            result = analyze_dragon_king(df, name)
            if result:
                dk_results.append(result)

    if dk_results:
        # バブル度ランキング
        dk_results.sort(key=lambda x: x["bubble_score"], reverse=True)

        st.markdown("### 🎯 バブル度スコア一覧")
        for r in dk_results:
            # バブル度に応じた色
            if r["bubble_score"] >= 60:
                bs_color = "#d50000"
                bs_label = "🔴 高リスク"
            elif r["bubble_score"] >= 40:
                bs_color = "#ff6d00"
                bs_label = "🟠 警戒"
            elif r["bubble_score"] >= 20:
                bs_color = "#f9a825"
                bs_label = "🟡 注意"
            else:
                bs_color = "#00c853"
                bs_label = "🟢 正常"

            signals_html = ""
            if r["crash_signals"]:
                signals_html = "<div style='margin-top:4px;'>" + " ".join(
                    f"<span style='background:#3a1b1b;padding:2px 6px;border-radius:6px;font-size:0.7rem;color:#ff8a80;'>{s}</span>"
                    for s in r["crash_signals"]
                ) + "</div>"

            st.markdown(
                f"<div style='background:#1a1a2e;padding:12px 16px;border-radius:8px;"
                f"margin-bottom:6px;border-left:5px solid {bs_color};'>"
                f"<div style='display:grid;grid-template-columns:140px 80px 80px 80px 100px 90px 80px 80px;align-items:center;gap:8px;'>"
                # 指数名
                f"<div>"
                f"<div style='color:#fff;font-weight:600;font-size:0.95rem;'>{r['name']}</div>"
                f"<div style='color:{bs_color};font-size:0.8rem;font-weight:bold;'>{bs_label} {r['bubble_score']}点</div>"
                f"</div>"
                # 価格
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>現在値</div>"
                f"<div style='color:#fff;font-size:0.85rem;'>{r['price']:,.1f}</div>"
                f"</div>"
                # 1M / 3M
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>1ヶ月</div>"
                f"<div style='color:{'#00c853' if r['ret_1m'] >= 0 else '#f44336'};font-size:0.8rem;'>{r['ret_1m']:+.1f}%</div>"
                f"</div>"
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>3ヶ月</div>"
                f"<div style='color:{'#00c853' if r['ret_3m'] >= 0 else '#f44336'};font-size:0.8rem;'>{r['ret_3m']:+.1f}%</div>"
                f"</div>"
                # DK検定
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>DK検定 p値</div>"
                f"<div style='color:{'#f44336' if r['dk_p'] < 0.05 else '#ff9800' if r['dk_p'] < 0.1 else '#aaa'};font-size:0.8rem;'>"
                f"{'⚠️' if r['dk_p'] < 0.05 else ''} {r['dk_p']:.4f}</div>"
                f"</div>"
                # 超指数的成長
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>加速度</div>"
                f"<div style='color:{'#f44336' if r['se_score'] > 20 else '#ff9800' if r['se_score'] > 0 else '#aaa'};font-size:0.8rem;'>{r['se_score']:+.0f}</div>"
                f"</div>"
                # Vol比率
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>Vol比率</div>"
                f"<div style='color:{'#f44336' if r['vol_ratio'] > 2 else '#ff9800' if r['vol_ratio'] > 1.5 else '#aaa'};font-size:0.8rem;'>{r['vol_ratio']:.2f}x</div>"
                f"</div>"
                # MA乖離
                f"<div>"
                f"<div style='color:#888;font-size:0.68rem;'>MA200乖離</div>"
                f"<div style='color:{'#f44336' if abs(r['ma_dev']) > 20 else '#ff9800' if abs(r['ma_dev']) > 10 else '#aaa'};font-size:0.8rem;'>{r['ma_dev']:+.1f}%</div>"
                f"</div>"
                f"</div>"
                + signals_html
                + f"</div>",
                unsafe_allow_html=True
            )

        # バブル度バーチャート
        st.markdown("### 📊 バブル度スコア比較")
        fig_dk = go.Figure()
        fig_dk.add_trace(go.Bar(
            x=[r["name"] for r in dk_results],
            y=[r["bubble_score"] for r in dk_results],
            marker_color=[
                "#d50000" if r["bubble_score"] >= 60
                else "#ff6d00" if r["bubble_score"] >= 40
                else "#f9a825" if r["bubble_score"] >= 20
                else "#00c853"
                for r in dk_results
            ],
            text=[f"{r['bubble_score']}点" for r in dk_results],
            textposition="outside",
        ))
        fig_dk.update_layout(
            yaxis_title="バブル度スコア", height=400,
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis=dict(range=[0, 110]),
        )
        fig_dk.add_hline(y=60, line_dash="dot", line_color="#d50000", annotation_text="高リスク")
        fig_dk.add_hline(y=40, line_dash="dot", line_color="#ff6d00", annotation_text="警戒")
        fig_dk.add_hline(y=20, line_dash="dot", line_color="#f9a825", annotation_text="注意")
        st.plotly_chart(fig_dk, use_container_width=True)

        # リターン分布 + DK検出（選択した指数）
        st.markdown("### 🔬 リターン分布 & ドラゴンキング検出")
        dk_detail = st.selectbox("指数を選択", [r["name"] for r in dk_results], key="dk_detail_select")
        sel = next((r for r in dk_results if r["name"] == dk_detail), None)
        if sel:
            abs_ret = np.abs(sel["returns"])
            sorted_ret = np.sort(abs_ret)

            fig_dist = make_subplots(rows=1, cols=2,
                                     subplot_titles=["リターン分布（ヒストグラム）", "対数ランクプロット"])

            # ヒストグラム
            fig_dist.add_trace(go.Histogram(
                x=sel["returns"] * 100,
                nbinsx=80,
                marker_color="#26a69a",
                opacity=0.7,
                name="日次リターン",
            ), row=1, col=1)

            # 上位3個をマーク
            top3 = np.sort(abs_ret)[-3:]
            for v in top3:
                fig_dist.add_vline(x=v * 100, line_dash="dash", line_color="#f44336",
                                   row=1, col=1)
                fig_dist.add_vline(x=-v * 100, line_dash="dash", line_color="#f44336",
                                   row=1, col=1)

            # 対数ランクプロット（パワーロー逸脱の可視化）
            ranks = np.arange(1, len(sorted_ret) + 1)[::-1]
            fig_dist.add_trace(go.Scatter(
                x=np.log10(sorted_ret + 1e-10),
                y=np.log10(ranks),
                mode="markers",
                marker=dict(size=3, color="#26a69a"),
                name="全データ",
            ), row=1, col=2)

            # 上位3個を赤でハイライト
            top3_idx = np.argsort(abs_ret)[-3:]
            fig_dist.add_trace(go.Scatter(
                x=np.log10(sorted_ret[-3:] + 1e-10),
                y=np.log10([3, 2, 1]),
                mode="markers",
                marker=dict(size=10, color="#f44336", symbol="star"),
                name="DK候補",
            ), row=1, col=2)

            fig_dist.update_layout(
                height=400, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                showlegend=True,
            )
            fig_dist.update_xaxes(title_text="日次リターン (%)", row=1, col=1)
            fig_dist.update_xaxes(title_text="log10(|リターン|)", row=1, col=2)
            fig_dist.update_yaxes(title_text="log10(ランク)", row=1, col=2)
            st.plotly_chart(fig_dist, use_container_width=True)

            # 詳細メトリクス
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("DK検定 F値", f"{sel['dk_f']:.2f}")
            dc2.metric("DK検定 p値", f"{sel['dk_p']:.4f}",
                        delta="有意" if sel['dk_p'] < 0.05 else "非有意",
                        delta_color="inverse" if sel['dk_p'] < 0.05 else "off")
            dc3.metric("SS統計量", f"{sel['ss']:.4f}")
            dc4.metric("最大DD", f"{sel['max_dd']:.1f}%")

            # 累積リターンチャート（価格推移 + MA200 + バブル度推移）
            st.markdown("### 📈 価格推移 & バブル度モニター")
            prices_series = sel["prices"]
            dates = sel["dates"]

            fig_price = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.65, 0.35],
                subplot_titles=["価格推移 & 移動平均", "ローリング・バブル度スコア"],
                vertical_spacing=0.08,
            )

            # 価格
            fig_price.add_trace(go.Scatter(
                x=dates, y=prices_series,
                mode="lines", name="価格",
                line=dict(color="#26a69a", width=2),
            ), row=1, col=1)

            # MA50
            if len(prices_series) >= 50:
                ma50 = pd.Series(prices_series).rolling(50).mean().values
                fig_price.add_trace(go.Scatter(
                    x=dates, y=ma50,
                    mode="lines", name="MA50",
                    line=dict(color="#42a5f5", width=1, dash="dot"),
                ), row=1, col=1)

            # MA200
            if len(prices_series) >= 200:
                ma200 = pd.Series(prices_series).rolling(200).mean().values
                fig_price.add_trace(go.Scatter(
                    x=dates, y=ma200,
                    mode="lines", name="MA200",
                    line=dict(color="#ff9800", width=1.5, dash="dash"),
                ), row=1, col=1)

            # ローリング・バブル度スコア（60日窓でDK検定 + MA乖離を計算）
            rolling_window = 60
            bubble_scores_rolling = []
            bubble_dates = []
            for i in range(rolling_window, len(prices_series)):
                window_prices = prices_series[i - rolling_window:i + 1]
                window_ret = np.abs(np.diff(np.log(window_prices)))
                if len(window_ret) < 20:
                    bubble_scores_rolling.append(0)
                    bubble_dates.append(dates[i])
                    continue

                # 簡易バブル度
                _, p_val = dk_test_stat(window_ret, r=2)
                ma_val = np.mean(window_prices)
                dev = (window_prices[-1] - ma_val) / ma_val * 100 if ma_val > 0 else 0
                vol_r = np.std(window_ret[-10:]) / np.std(window_ret) if np.std(window_ret) > 0 else 1

                sc = 0
                if p_val < 0.05:
                    sc += 30
                elif p_val < 0.1:
                    sc += 15
                if abs(dev) > 15:
                    sc += 20
                elif abs(dev) > 8:
                    sc += 10
                if vol_r > 2:
                    sc += 20
                elif vol_r > 1.5:
                    sc += 10
                bubble_scores_rolling.append(min(sc, 100))
                bubble_dates.append(dates[i])

            # バブル度推移
            fig_price.add_trace(go.Scatter(
                x=bubble_dates, y=bubble_scores_rolling,
                mode="lines", name="バブル度",
                line=dict(color="#ff6d00", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,109,0,0.15)",
            ), row=2, col=1)

            # 閾値ライン
            fig_price.add_hline(y=60, line_dash="dot", line_color="#d50000",
                                annotation_text="高リスク", row=2, col=1)
            fig_price.add_hline(y=40, line_dash="dot", line_color="#ff6d00",
                                annotation_text="警戒", row=2, col=1)

            fig_price.update_layout(
                height=650, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=1.02, font=dict(size=10)),
                margin=dict(l=50, r=20, t=40, b=30),
            )
            fig_price.update_yaxes(title_text="価格", row=1, col=1)
            fig_price.update_yaxes(title_text="バブル度", range=[0, 110], row=2, col=1)
            st.plotly_chart(fig_price, use_container_width=True)

            # 全指数の累積リターン比較
            st.markdown("### 📊 全指数 累積リターン比較")
            fig_cum_dk = go.Figure()
            for r in dk_results:
                if len(r["returns"]) > 0:
                    cum_ret = (np.exp(np.cumsum(r["returns"])) - 1) * 100
                    fig_cum_dk.add_trace(go.Scatter(
                        x=r["dates"][1:],
                        y=cum_ret,
                        mode="lines", name=r["name"],
                        line=dict(width=2),
                    ))
            fig_cum_dk.update_layout(
                yaxis_title="累積リターン (%)",
                height=500, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="top", y=-0.05, font=dict(size=10)),
                margin=dict(l=50, r=20, t=30, b=150),
            )
            fig_cum_dk.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_cum_dk, use_container_width=True)

    else:
        st.warning("データ取得に失敗しました。")

    st.caption("📖 ドラゴンキング理論: Sornette (2009) に基づく。パワーロー分布から逸脱する極端イベント（DK）を統計的に検出し、バブル崩壊の前兆を捉える手法。")

# =====================================================================
# Tab 9: 出来高急増ランキング
# =====================================================================

@st.cache_data(ttl=600)
def fetch_kabudragon_volume_ranking():
    """株ドラゴンから出来高急増ランキングを取得"""
    import re as _re
    url = "https://www.kabudragon.com/ranking/dekizou.html"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.encoding = resp.apparent_encoding
        if resp.status_code != 200:
            return [], ""
        soup = BeautifulSoup(resp.text, "html.parser")

        date_str = ""
        page_text = soup.get_text()
        m = _re.search(r"\d{4}/\d{2}/\d{2}", page_text)
        if m:
            date_str = m.group()

        rows = []
        for table in soup.find_all("table"):
            trs = table.find_all("tr")
            if len(trs) < 3:
                continue
            for tr in trs:
                tds = tr.find_all("td")
                # 列数が10〜15の行のみ対象（ヘッダー行や結合セル行を除外）
                if len(tds) < 10 or len(tds) > 20:
                    continue
                texts = [td.get_text(strip=True) for td in tds]
                if not texts[0].isdigit():
                    continue
                try:
                    # 列構成: 順位, コード, 名称, 市場, 日付, 取引値, 前日比, 前日比%, 急増率, 出来高, ...
                    offset = 4
                    if _re.match(r"^\d+/\d+", texts[offset]):
                        offset = 5  # 日付列あり
                    rows.append({
                        "rank": texts[0],
                        "code": texts[1],
                        "name": texts[2],
                        "market": texts[3],
                        "price": texts[offset] if offset < len(texts) else "",
                        "change": texts[offset+1] if offset+1 < len(texts) else "",
                        "change_pct": texts[offset+2] if offset+2 < len(texts) else "",
                        "vol_ratio": texts[offset+3] if offset+3 < len(texts) else "",
                        "volume": texts[offset+4] if offset+4 < len(texts) else "",
                    })
                except IndexError:
                    continue
            if rows:
                break

        return rows[:50], date_str
    except Exception:
        return [], ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividend_yield(code):
    """配当利回りを取得（yfinance info から計算、異常値フィルタ付き）"""
    try:
        ticker = f"{code}.T" if not code.endswith(".T") else code
        info = yf.Ticker(ticker).info

        price = info.get("previousClose") or info.get("currentPrice") or info.get("regularMarketPrice")
        if not price or price <= 0:
            return None

        # 方法1: dividendRate（予想年間配当）/ 株価
        div_rate = info.get("dividendRate")
        if div_rate and div_rate > 0:
            yld = div_rate / price * 100
            if 0.1 < yld < 6:
                return round(yld, 2)

        # 方法2: trailingAnnualDividendRate（実績年間配当）/ 株価
        trailing = info.get("trailingAnnualDividendRate")
        if trailing and trailing > 0:
            yld = trailing / price * 100
            if 0.1 < yld < 6:
                return round(yld, 2)

        # 方法3: dividendYield（yfinance算出値）をそのまま使う
        dy = info.get("dividendYield")
        if dy and dy > 0:
            yld = dy * 100 if dy < 0.5 else dy
            if 0.1 < yld < 6:
                return round(yld, 2)
    except Exception:
        pass
    return None


@st.cache_data(ttl=600)
def fetch_news_for_stock(stock_name):
    """銘柄の直近ニュースを1件取得"""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime as _pdt3
    try:
        q = quote(f"{stock_name}")
        rss = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        resp = requests.get(rss, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        if items:
            title = items[0].findtext("title", "").strip()
            link = items[0].findtext("link", "").strip()
            return f"<a href='{link}' target='_blank' style='color:#90caf9;text-decoration:none;font-size:0.73rem;'>{title[:70]}{'...' if len(title)>70 else ''}</a>"
    except Exception:
        pass
    return ""


with tab9:
    st.subheader("📢 出来高急増ランキング（全市場）")
    st.caption("株ドラゴン提供データ — 5日平均比で出来高が急増した銘柄TOP20 + 関連ニュース")

    with st.spinner("📊 出来高急増ランキング取得中..."):
        vol_ranking, vol_date = fetch_kabudragon_volume_ranking()

    if vol_ranking:
        st.markdown(f"📅 データ日: {vol_date}　|　全{len(vol_ranking)}件取得")

        # 上位20件のニュースを一括並列取得（配当は遅延ロード）
        top20 = vol_ranking[:20]
        from concurrent.futures import ThreadPoolExecutor

        @st.cache_data(ttl=600)
        def _batch_fetch_news_and_ytd(names_tuple, codes_tuple):
            """ニュースと年初来高値/安値を並列取得"""
            news_map = {}
            ytd_map = {}  # code -> {"ytd_high": float, "ytd_low": float}
            def _get_news(name):
                return name, fetch_news_for_stock(name)
            def _get_ytd(code):
                try:
                    ticker = f"{code}.T"
                    df = yf.download(ticker, period="ytd", progress=False)
                    if df.empty:
                        return code, None
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    ytd_high = float(df["High"].max())
                    ytd_low = float(df["Low"].min())
                    last_close = float(df["Close"].iloc[-1])
                    # 直近5日の安値・高値
                    recent_low = float(df["Low"].iloc[-5:].min()) if len(df) >= 5 else ytd_low
                    recent_high = float(df["High"].iloc[-5:].max()) if len(df) >= 5 else ytd_high
                    return code, {
                        "ytd_high": ytd_high, "ytd_low": ytd_low,
                        "last": last_close,
                        "near_ytd_low": recent_low <= ytd_low * 1.03,  # 年初来安値の3%以内
                        "near_ytd_high": recent_high >= ytd_high * 0.97,  # 年初来高値の3%以内
                    }
                except Exception:
                    return code, None
            with ThreadPoolExecutor(max_workers=10) as ex:
                news_futs = [ex.submit(_get_news, n) for n in names_tuple]
                ytd_futs = [ex.submit(_get_ytd, c) for c in codes_tuple]
                for f in news_futs:
                    try:
                        n, html = f.result(timeout=8)
                        news_map[n] = html
                    except Exception:
                        pass
                for f in ytd_futs:
                    try:
                        c, data = f.result(timeout=15)
                        if data:
                            ytd_map[c] = data
                    except Exception:
                        pass
            return news_map, ytd_map

        with st.spinner("📰 関連ニュース・年初来データ取得中..."):
            names_tuple = tuple(r["name"] for r in top20)
            all_codes_tuple = tuple(r["code"] for r in vol_ranking[:50])
            news_cache, ytd_cache = _batch_fetch_news_and_ytd(names_tuple, all_codes_tuple)

        # 上位20件を表示
        for r in top20:
            code = r["code"]
            name = r["name"]
            vol_ratio = r["vol_ratio"]
            change_pct = r["change_pct"]
            kabutan_url = f"https://kabutan.jp/stock/?code={code}"
            shikiho_url = f"https://shikiho.toyokeizai.net/stocks/{code}"

            # 変動率の色と売り/買い判定
            try:
                pct_val = float(change_pct.replace("%", "").replace("+", ""))
                chg_color = "#00c853" if pct_val >= 0 else "#f44336"
            except Exception:
                chg_color = "#aaa"
                pct_val = 0

            # 売り/買い材料判定バッジ
            if pct_val >= 5:
                signal_badge = "<span style='background:#1b3a1b;color:#00e676;padding:2px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;'>🚀 急騰・買い材料</span>"
            elif pct_val >= 1:
                signal_badge = "<span style='background:#1b3a1b;color:#69f0ae;padding:2px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;'>📈 買い優勢</span>"
            elif pct_val > -1:
                signal_badge = "<span style='background:#2a2a2a;color:#bbb;padding:2px 8px;border-radius:8px;font-size:0.72rem;'>➡️ 様子見</span>"
            elif pct_val > -5:
                signal_badge = "<span style='background:#3a1b1b;color:#ff8a80;padding:2px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;'>📉 売り優勢</span>"
            else:
                signal_badge = "<span style='background:#3a1b1b;color:#ff5252;padding:2px 8px;border-radius:8px;font-size:0.72rem;font-weight:bold;'>💥 急落・売り材料</span>"

            # 出来高倍率の色
            try:
                ratio_val = float(vol_ratio.replace("%", ""))
                if ratio_val >= 400:
                    ratio_color = "#d50000"
                    ratio_bg = "#3a1b1b"
                elif ratio_val >= 300:
                    ratio_color = "#ff6d00"
                    ratio_bg = "#3a2a1b"
                else:
                    ratio_color = "#f9a825"
                    ratio_bg = "#3a3a1b"
            except Exception:
                ratio_color = "#aaa"
                ratio_bg = "#1a1a2e"
                ratio_val = 0

            news_html = news_cache.get(name, "")

            # テーマ分類バッジ
            vol_themes = guess_themes_by_name(f"{code}.T", name)
            theme_badges_vol = ""
            if vol_themes:
                theme_badges_vol = "".join(
                    f"<a href='https://kabutan.jp/themes/?theme={quote(THEMES.get(th, {}).get('kabutan', th))}' target='_blank' "
                    f"style='display:inline-block;padding:1px 6px;margin:1px;background:#1a237e;border-radius:8px;"
                    f"color:#7986cb;text-decoration:none;font-size:0.65rem;'>{th}</a>"
                    for th in vol_themes
                )

            # セリクラ/バイクラ判定
            climax_badge = ""
            ytd_info = ytd_cache.get(code)
            if ytd_info:
                # セリングクライマックス: 年初来安値付近 + 出来高急増 + 株価上昇
                if ytd_info["near_ytd_low"] and pct_val > 1 and ratio_val >= 200:
                    climax_badge = (
                        "<span style='background:#1a237e;color:#82b1ff;padding:2px 8px;"
                        "border-radius:8px;font-size:0.72rem;font-weight:bold;margin-left:4px;'>"
                        "🔄 セリクラ候補（安値圏で出来高急増＋反発）</span>"
                    )
                # バイイングクライマックス: 年初来高値付近 + 出来高急増 + 株価下落
                elif ytd_info["near_ytd_high"] and pct_val < -1 and ratio_val >= 200:
                    climax_badge = (
                        "<span style='background:#4a148c;color:#ea80fc;padding:2px 8px;"
                        "border-radius:8px;font-size:0.72rem;font-weight:bold;margin-left:4px;'>"
                        "🔄 バイクラ候補（高値圏で出来高急増＋反落）</span>"
                    )

            # 仕手株リスク判定
            shite_warnings = []
            try:
                price_val = float(r["price"].replace(",", ""))
                if price_val < 300:
                    shite_warnings.append("低位株")
            except Exception:
                pass
            if r["market"] in ("東スタ", "東グロ", "名メイ", "ネク"):
                shite_warnings.append("小型市場")
            if ratio_val >= 400:
                shite_warnings.append("異常急増")
            # 出来高が少ない（流動性低い）
            try:
                vol_val = int(r["volume"].replace(",", ""))
                if vol_val < 50000:
                    shite_warnings.append("薄商い")
            except Exception:
                pass
            shite_badge = ""
            if len(shite_warnings) >= 2:
                shite_badge = (
                    f"<span style='background:#4a148c;color:#ce93d8;padding:2px 8px;"
                    f"border-radius:8px;font-size:0.7rem;font-weight:bold;margin-left:4px;'>"
                    f"⚠️ 仕手注意（{'/'.join(shite_warnings)}）</span>"
                )

            # 材料コメント生成（ニュースタイトルからキーワード分析）
            material_comment = ""
            news_title = news_cache.get(name + "_title", "")
            if not news_title and news_html:
                # news_htmlからタイトルテキストを抽出
                import re as _re2
                m = _re2.search(r">([^<]+)</a>", news_html)
                news_title = m.group(1) if m else ""

            if news_title:
                t_lower = news_title
                if any(k in t_lower for k in ["決算", "業績", "増益", "最高益", "上方修正"]):
                    material_comment = "📊 好決算・業績好調"
                elif any(k in t_lower for k in ["下方修正", "減益", "赤字", "損失"]):
                    material_comment = "📉 業績悪化・下方修正"
                elif any(k in t_lower for k in ["配当", "増配", "株主還元", "自社株買い", "自己株式"]):
                    material_comment = "💰 株主還元強化"
                elif any(k in t_lower for k in ["TOB", "MBO", "買収", "合併", "統合", "提携"]):
                    material_comment = "🤝 M&A・資本提携"
                elif any(k in t_lower for k in ["新製品", "新サービス", "受注", "契約", "採用"]):
                    material_comment = "🆕 新規事業・受注"
                elif any(k in t_lower for k in ["AI", "半導体", "データセンター", "量子"]):
                    material_comment = "🤖 テクノロジー関連"
                elif any(k in t_lower for k in ["防衛", "軍事", "安全保障", "宇宙"]):
                    material_comment = "🛡️ 防衛・安全保障"
                elif any(k in t_lower for k in ["株式分割", "分割"]):
                    material_comment = "✂️ 株式分割"
                elif any(k in t_lower for k in ["ストップ高", "急騰", "年初来高値"]):
                    material_comment = "🚀 株価急騰"
                elif any(k in t_lower for k in ["ストップ安", "急落", "年初来安値"]):
                    material_comment = "💥 株価急落"
                elif any(k in t_lower for k in ["優待", "株主優待"]):
                    material_comment = "🎁 株主優待"
                elif any(k in t_lower for k in ["格上げ", "目標株価", "レーティング"]):
                    material_comment = "📈 アナリスト評価"
                elif any(k in t_lower for k in ["不祥事", "不正", "行政処分"]):
                    material_comment = "⚠️ 不祥事・コンプライアンス"
                else:
                    material_comment = "📰 ニュース材料"

            material_html = ""
            if material_comment:
                mc_color = "#ff8a80" if "悪化" in material_comment or "急落" in material_comment or "不祥事" in material_comment else "#69f0ae" if "好決算" in material_comment or "還元" in material_comment or "急騰" in material_comment else "#90caf9"
                material_html = f"<span style='color:{mc_color};font-size:0.75rem;font-weight:bold;margin-right:6px;'>{material_comment}</span>"

            st.markdown(
                f"<div style='background:#1a1a2e;padding:10px 14px;border-radius:8px;"
                f"margin-bottom:4px;border-left:5px solid {ratio_color};'>"
                f"<div style='display:grid;grid-template-columns:40px 190px 100px 90px 90px 1fr;align-items:center;gap:8px;'>"
                # 順位
                f"<div style='color:#888;font-size:1.1rem;font-weight:bold;text-align:center;'>"
                f"{'🥇' if r['rank']=='1' else '🥈' if r['rank']=='2' else '🥉' if r['rank']=='3' else r['rank']}</div>"
                # 銘柄名
                f"<div style='overflow:hidden;'>"
                f"<a href='{kabutan_url}' target='_blank' style='color:#fff;text-decoration:none;font-weight:600;font-size:0.88rem;'>"
                f"{name}</a>"
                f" <span style='color:#888;font-size:0.7rem;'>{code}</span>"
                f" <a href='{shikiho_url}' target='_blank' style='color:#4da6ff;font-size:0.65rem;text-decoration:none;'>四季報</a>"
                f"{shite_badge}"
                f"<div style='color:#888;font-size:0.68rem;'>{r['market']}</div>"
                f"</div>"
                # 株価 + 売り/買い判定
                f"<div>"
                f"<div style='color:#fff;font-size:0.9rem;'>{r['price']}</div>"
                f"<div style='color:{chg_color};font-size:0.8rem;'>{r['change']} ({change_pct})</div>"
                f"</div>"
                # 出来高急増率
                f"<div style='background:{ratio_bg};padding:4px 8px;border-radius:6px;text-align:center;'>"
                f"<div style='color:#888;font-size:0.65rem;'>急増率</div>"
                f"<div style='color:{ratio_color};font-size:1rem;font-weight:bold;'>{vol_ratio}</div>"
                f"</div>"
                # 出来高
                f"<div>"
                f"<div style='color:#888;font-size:0.65rem;'>出来高</div>"
                f"<div style='color:#aaa;font-size:0.82rem;'>{r['volume']}</div>"
                f"</div>"
                # ニュース（理由）
                f"<div>"
                f"<div style='color:#888;font-size:0.65rem;'>関連ニュース</div>"
                f"<div>{news_html if news_html else '<span style=\"color:#666;font-size:0.72rem;\">—</span>'}</div>"
                f"</div>"
                f"</div>"
                # 下段: 売り/買い判定 + 材料コメント + セリクラ/バイクラ + テーマ
                f"<div style='margin-top:4px;'>{signal_badge} {material_html}{climax_badge}</div>"
                + (f"<div style='margin-top:2px;'>{theme_badges_vol}</div>" if theme_badges_vol else "")
                + f"</div>",
                unsafe_allow_html=True
            )

        # 配当利回り（展開時のみ取得）
        with st.expander("💰 上位20銘柄の配当利回りを確認"):
            with st.spinner("配当利回り取得中..."):
                @st.cache_data(ttl=3600)
                def _batch_yields(codes_tuple):
                    result = {}
                    with ThreadPoolExecutor(max_workers=10) as ex:
                        def _gy(c):
                            return c, fetch_dividend_yield(c)
                        futs = [ex.submit(_gy, c) for c in codes_tuple]
                        for f in futs:
                            try:
                                c, y = f.result(timeout=10)
                                result[c] = y
                            except Exception:
                                pass
                    return result
                yld_data = _batch_yields(tuple(r["code"] for r in top20))
            high_yield = [(r["name"], r["code"], yld_data.get(r["code"])) for r in top20 if yld_data.get(r["code"]) and yld_data[r["code"]] >= 3.0]
            if high_yield:
                for nm, cd, yld in sorted(high_yield, key=lambda x: x[2], reverse=True):
                    yc = "#00c853" if yld >= 4.0 else "#f9a825"
                    st.markdown(f"<span style='color:{yc};font-weight:bold;'>💰 {yld:.2f}%</span> {nm}（{cd}）", unsafe_allow_html=True)
            else:
                st.caption("利回り3%以上の銘柄はありません")

        # 21-50位は折りたたみ
        if len(vol_ranking) > 20:
            with st.expander(f"📋 21〜{len(vol_ranking)}位を表示"):
                # 配当利回りを展開時に並列取得
                @st.cache_data(ttl=3600)
                def _batch_yields_rest(codes_tuple_rest):
                    result = {}
                    with ThreadPoolExecutor(max_workers=10) as ex:
                        def _gy(c):
                            return c, fetch_dividend_yield(c)
                        futs = [ex.submit(_gy, c) for c in codes_tuple_rest]
                        for f in futs:
                            try:
                                c, y = f.result(timeout=10)
                                result[c] = y
                            except Exception:
                                pass
                    return result
                rest_codes = tuple(r["code"] for r in vol_ranking[20:])
                with st.spinner("配当利回り取得中..."):
                    yield_rest = _batch_yields_rest(rest_codes)

                for r in vol_ranking[20:]:
                    code = r["code"]
                    kabutan_url = f"https://kabutan.jp/stock/?code={code}"
                    try:
                        pct_val = float(r["change_pct"].replace("%", "").replace("+", ""))
                        chg_color = "#00c853" if pct_val >= 0 else "#f44336"
                    except Exception:
                        chg_color = "#aaa"
                        pct_val = 0
                    try:
                        ratio_val_r = float(r["vol_ratio"].replace("%", ""))
                    except Exception:
                        ratio_val_r = 0

                    # セリクラ/バイクラ判定
                    badges = ""
                    ytd_r = ytd_cache.get(code)
                    if ytd_r:
                        if ytd_r["near_ytd_low"] and pct_val > 1 and ratio_val_r >= 200:
                            badges += " <span style='background:#1a237e;color:#82b1ff;padding:1px 6px;border-radius:6px;font-size:0.68rem;font-weight:bold;'>🔄 セリクラ</span>"
                        elif ytd_r["near_ytd_high"] and pct_val < -1 and ratio_val_r >= 200:
                            badges += " <span style='background:#4a148c;color:#ea80fc;padding:1px 6px;border-radius:6px;font-size:0.68rem;font-weight:bold;'>🔄 バイクラ</span>"

                    # 配当利回り
                    dy = yield_rest.get(code)
                    if dy and dy >= 3.0:
                        yc = "#00c853" if dy >= 4.0 else "#f9a825"
                        badges += f" <span style='background:#1b3a1b;border:1px solid {yc};color:{yc};padding:1px 5px;border-radius:6px;font-size:0.68rem;font-weight:bold;'>💰 {dy:.1f}%</span>"

                    # テーマ分類
                    rest_themes = guess_themes_by_name(f"{code}.T", r["name"])
                    if rest_themes:
                        badges += " " + "".join(
                            f"<span style='background:#1a237e;color:#7986cb;padding:1px 5px;border-radius:6px;font-size:0.62rem;margin:0 1px;'>{th}</span>"
                            for th in rest_themes
                        )

                    st.markdown(
                        f"<div style='padding:4px 8px;border-bottom:1px solid #222;display:flex;gap:12px;align-items:center;flex-wrap:wrap;'>"
                        f"<span style='color:#888;min-width:30px;'>{r['rank']}</span>"
                        f"<a href='{kabutan_url}' target='_blank' style='color:#ccc;text-decoration:none;min-width:140px;font-size:0.85rem;'>"
                        f"{r['name']} ({code})</a>"
                        f"<span style='color:#fff;min-width:70px;'>{r['price']}</span>"
                        f"<span style='color:{chg_color};min-width:80px;'>{r['change_pct']}</span>"
                        f"<span style='color:#ff6d00;font-weight:bold;min-width:70px;'>{r['vol_ratio']}</span>"
                        f"{badges}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        st.caption("データ出典: [株ドラゴン](https://www.kabudragon.com/ranking/dekizou.html)")
    else:
        st.warning("出来高急増ランキングの取得に失敗しました。")

# =====================================================================
# Tab 10: TOPIX新基準スクリーニング
# =====================================================================
_TOPIX_TRADED_VALUE_RATIO_THRESHOLD = 0.14
_TOPIX_CUMULATIVE_MCAP_PCT = 0.97
_TOPIX_CACHE_TTL = 86400

@st.cache_data(ttl=_TOPIX_CACHE_TTL, show_spinner=False)
def _topix_fetch_stock_list():
    """JPX上場銘柄一覧を取得（市場区分・業種付き）"""
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        return pd.DataFrame()
    df = pd.read_excel(io.BytesIO(resp.content))
    col_map = {}
    used = set()
    for c in df.columns:
        s = str(c)
        if "コード" in s and "33業種" not in s and "17業種" not in s and "code" not in used:
            col_map[c] = "code"
            used.add("code")
        elif "銘柄名" in s and "name" not in used:
            col_map[c] = "name"
            used.add("name")
        elif "市場・商品区分" in s and "market" not in used:
            col_map[c] = "market"
            used.add("market")
        elif "33業種" in s and "コード" not in s and "sector" not in used:
            col_map[c] = "sector"
            used.add("sector")
    df = df.rename(columns=col_map)
    for n in ["code", "name", "market", "sector"]:
        if n not in df.columns:
            df[n] = ""
    # 重複カラム名があれば最初のものだけ残す
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"].str.match(r"^\d{4}$")]
    df["ticker"] = df["code"] + ".T"
    return df[["ticker", "code", "name", "market", "sector"]].reset_index(drop=True)

@st.cache_data(ttl=_TOPIX_CACHE_TTL, show_spinner=False)
def _topix_fetch_market_data(tickers_tuple, period="1y"):
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

@st.cache_data(ttl=_TOPIX_CACHE_TTL, show_spinner=False)
def _topix_fetch_fundamentals(tickers_tuple):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tickers = list(tickers_tuple)
    results = {}

    def _fetch_one(t):
        try:
            tk = yf.Ticker(t)
            info = tk.info
            div_yield_calc = None
            try:
                divs = tk.dividends
                price = info.get("currentPrice", info.get("regularMarketPrice"))
                if divs is not None and len(divs) > 0 and price and price > 0:
                    one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
                    recent_divs = divs[divs.index >= one_year_ago]
                    if len(recent_divs) > 0:
                        div_yield_calc = recent_divs.sum() / price
            except Exception:
                pass
            if div_yield_calc is None:
                div_yield_calc = info.get("trailingAnnualDividendYield") or info.get("dividendYield")
            return t, {
                "marketCap": info.get("marketCap"),
                "floatShares": info.get("floatShares"),
                "sharesOutstanding": info.get("sharesOutstanding"),
                "dividendYield": div_yield_calc,
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "returnOnEquity": info.get("returnOnEquity"),
                "priceToBook": info.get("priceToBook"),
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice")),
            }
        except Exception:
            return t, {}

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                t, data = fut.result(timeout=15)
                results[t] = data
            except Exception:
                results[futures[fut]] = {}
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
    """current_topix_codes: 現行TOPIX構成銘柄コードのset（あれば継続安全/新規追加の判別に使用）"""
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
        # 回転率基準を満たさない → 除外警戒
        if tr < _TOPIX_TRADED_VALUE_RATIO_THRESHOLD:
            return "🔴 除外警戒"
        # 回転率OK & 累積97%以内
        if cum is not None and not pd.isna(cum) and cum <= _TOPIX_CUMULATIVE_MCAP_PCT:
            if is_current:
                return "🟢 継続安全"
            else:
                return "⚪ 新規追加候補"
        # 回転率OK & 累積97-99% → ボーダー
        elif cum is not None and not pd.isna(cum) and cum <= 0.99:
            return "🟡 ボーダーライン"
        # 回転率OK だが累積99%超 → 除外警戒
        return "🔴 除外警戒"
    result["TOPIX判定"] = result.apply(_judge, axis=1)

    # ── TOPIX適合スコア計算 ──
    # TOPIXの加重平均特性に近いほど高スコア
    # 基準値: PER≈17, PBR≈1.55, 配当利回り≈2.16%, ROE≈8-10%
    _TOPIX_TARGET_PER = 17.0
    _TOPIX_TARGET_PBR = 1.55
    _TOPIX_TARGET_DIV = 2.16
    _TOPIX_TARGET_ROE = 9.0

    def _calc_fit_score(r):
        score = 0.0
        components = 0

        # 1. 流動性スコア (0-25点): 回転率が基準を超えるほど高い
        tr = r.get("年間売買回転率")
        if tr is not None and not pd.isna(tr):
            if tr >= _TOPIX_TRADED_VALUE_RATIO_THRESHOLD:
                # 0.14で15点、0.5以上で25点
                score += min(25, 15 + (tr - 0.14) / 0.36 * 10)
            else:
                score += max(0, tr / 0.14 * 10)
            components += 1

        # 2. 時価総額スコア (0-25点): 大きいほど高い
        ff = r.get("浮動株時価総額(億円)")
        if ff is not None and not pd.isna(ff) and ff > 0:
            # 100億で10点、500億で18点、1000億以上で25点
            score += min(25, np.log10(max(ff, 1)) / np.log10(1000) * 25)
            components += 1

        # 3. PER適合 (0-15点): 17倍に近いほど高い
        pe = r.get("PER(倍)")
        if pe is not None and not pd.isna(pe) and 0 < pe < 200:
            pe_diff = abs(pe - _TOPIX_TARGET_PER) / _TOPIX_TARGET_PER
            score += max(0, 15 * (1 - pe_diff))
            components += 1

        # 4. PBR適合 (0-10点): 1.55倍に近いほど高い
        pbr = r.get("PBR(倍)")
        if pbr is not None and not pd.isna(pbr) and 0 < pbr < 50:
            pbr_diff = abs(pbr - _TOPIX_TARGET_PBR) / _TOPIX_TARGET_PBR
            score += max(0, 10 * (1 - pbr_diff))
            components += 1

        # 5. 配当利回り適合 (0-10点): 2.16%に近いほど高い
        dy = r.get("配当利回り(%)")
        if dy is not None and not pd.isna(dy) and dy >= 0:
            dy_diff = abs(dy - _TOPIX_TARGET_DIV) / max(_TOPIX_TARGET_DIV, 0.01)
            score += max(0, 10 * (1 - dy_diff))
            components += 1

        # 6. ROE適合 (0-15点): 高いほど良い（8-10%が理想）
        roe = r.get("ROE(%)")
        if roe is not None and not pd.isna(roe):
            if roe >= _TOPIX_TARGET_ROE:
                score += min(15, 10 + (roe - _TOPIX_TARGET_ROE) / 10 * 5)
            elif roe > 0:
                score += roe / _TOPIX_TARGET_ROE * 10
            components += 1

        if components == 0:
            return None
        return round(score, 1)

    result["適合スコア"] = result.apply(_calc_fit_score, axis=1)

    # ランク付け（スコア降順）
    scored = result.dropna(subset=["適合スコア"])
    if not scored.empty:
        scored = scored.sort_values("適合スコア", ascending=False)
        scored["採用確度ランク"] = range(1, len(scored) + 1)
        result = result.merge(scored[["ticker", "採用確度ランク"]], on="ticker", how="left")
    return result.sort_values("浮動株時価総額(億円)", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=_TOPIX_CACHE_TTL, show_spinner=False)
def _topix_fetch_current_constituents():
    """
    現行TOPIX構成銘柄コードのセットを取得する。
    JPX公式のTOPIXウエイトCSVから取得（最も確実）。
    """
    import re
    codes = set()

    # JPX TOPIX構成銘柄ウエイトCSV（Shift_JIS）
    # https://www.jpx.co.jp/markets/indices/topix/ からリンクされている自動生成CSV
    try:
        url = "https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv"
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and len(resp.content) > 1000:
            text = resp.content.decode("shift_jis", errors="ignore")
            for line in text.strip().split("\n")[1:]:  # ヘッダースキップ
                parts = line.split(",")
                if len(parts) >= 3:
                    code_str = parts[2].strip()
                    m = re.match(r"^(\d{4})$", code_str)
                    if m:
                        codes.add(m.group(1))
    except Exception:
        pass

    if len(codes) > 100:
        return codes

    # フォールバック: JPX英語版ページのExcel
    try:
        url2 = "https://www.jpx.co.jp/english/markets/indices/topix/tvdivq00000030ne-att/TOPIX_weight_en.xlsx"
        resp2 = requests.get(url2, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp2.status_code == 200 and len(resp2.content) > 1000:
            df = pd.read_excel(io.BytesIO(resp2.content))
            for c in df.columns:
                s = str(c)
                if "Code" in s or "コード" in s:
                    for val in df[c].dropna().astype(str):
                        m = re.match(r"^(\d{4})", val.strip())
                        if m:
                            codes.add(m.group(1))
                    break
    except Exception:
        pass

    return codes

with tab10:
    st.subheader("📊 TOPIX 新基準スクリーニング")
    st.caption(
        "次世代TOPIX（第2段階見直し）の選定基準に基づくスクリーニング　"
        "| 第1回定期見直し: 2026年10月 | 第2回: 2028年10月"
    )

    with st.expander("📋 次世代TOPIX 選定基準", expanded=False):
        st.markdown("""
        **対象ユニバース**: 全市場区分（プライム・スタンダード・グロース）

        **選定基準**:
        1. **年間売買回転率** ≥ 0.14
           - 月次売買回転率 = (月中日次売買代金の中央値 × 営業日数) / 月末浮動株時価総額
           - 年間 = 直近12ヶ月の月次売買回転率の合計
        2. **累積浮動株時価総額** 上位97%以内

        **経過措置**: 第1回見直しで非選定となった銘柄は四半期ごとに8段階でウエイト低減

        [JPX公式資料](https://www.jpx.co.jp/english/markets/indices/revisions-indices/02.html)
        """)

    # フィルター（タブ内に配置）
    st.markdown("#### 🔍 新規採用候補スクリーニング")
    st.caption(
        "現行TOPIX構成銘柄を除外し、新基準で新規採用される可能性のある銘柄を抽出します。"
        "全市場（プライム含む）から、現行TOPIX非構成の注目銘柄をピックアップします。"
    )

    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc1:
        topix_market_filter = st.multiselect(
            "市場区分",
            ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"],
            default=["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"],
            key="topix_market"
        )
    with tc2:
        topix_max_stocks = st.selectbox(
            "対象数", [100, 200, 500, 1000, 2000, 3000],
            index=3, key="topix_max",
            help="全市場から幅広く拾うには1000以上推奨"
        )
    with tc3:
        topix_judge_filter = st.multiselect(
            "判定フィルター",
            ["⚪ 新規追加候補", "🟢 継続安全", "🟡 ボーダーライン", "🔴 除外警戒", "データ不足"],
            default=["⚪ 新規追加候補", "🟢 継続安全", "🟡 ボーダーライン"],
            key="topix_judge"
        )

    exclude_current = st.checkbox(
        "🚫 現行TOPIX構成銘柄を除外（新規採用候補のみ表示）",
        value=True, key="topix_exclude",
        help="JPX/BlackRockから現行構成銘柄を取得して除外。取得失敗時は除外なしで全銘柄表示。"
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        topix_min_turnover = st.number_input(
            "売買回転率下限", value=0.14, step=0.02, format="%.2f",
            key="topix_turnover",
            help="JPX基準は0.14。0.2以上にすると高流動性銘柄に絞れます"
        )
    with fc2:
        topix_min_div = st.number_input("配当利回り下限(%)", value=0.0, step=0.5, key="topix_div")
    with fc3:
        topix_max_per = st.number_input("PER上限(倍)", value=100.0, step=5.0, key="topix_per")
    with fc4:
        topix_min_roe = st.number_input("ROE下限(%)", value=0.0, step=1.0, key="topix_roe")

    if st.button("🚀 スクリーニング実行", type="primary", key="topix_run"):
        with st.spinner("JPXから銘柄リストを取得中..."):
            stock_df = _topix_fetch_stock_list()
        if stock_df.empty:
            st.error("銘柄リストの取得に失敗しました")
        else:
            # 市場フィルター適用
            if topix_market_filter:
                stock_df = stock_df[stock_df["market"].isin(topix_market_filter)]

            # 常に現行TOPIX構成銘柄リストを取得（色分け判定に使用）
            with st.spinner("現行TOPIX構成銘柄リストを取得中..."):
                current_codes = _topix_fetch_current_constituents()
            if len(current_codes) > 100:
                st.info(f"現行TOPIX構成銘柄: {len(current_codes)}件を取得済み")
            else:
                current_codes = set()
                st.warning("現行TOPIX構成銘柄リストの取得に失敗。継続安全/新規追加の判別は市場区分ベースで推定します。")

            # 除外オプション
            if exclude_current and len(current_codes) > 100:
                before = len(stock_df)
                stock_df = stock_df[~stock_df["code"].isin(current_codes)]
                excluded_count = before - len(stock_df)
                st.success(f"{excluded_count}件の現行構成銘柄を除外 → 残り {len(stock_df)}件")

            stock_df = stock_df.head(topix_max_stocks)
            st.info(f"スクリーニング対象: {len(stock_df)}件")
            tickers = tuple(stock_df["ticker"].tolist())
            progress = st.progress(0, text="価格データ取得中...")
            md = _topix_fetch_market_data(tickers, period="1y")
            progress.progress(50, text="ファンダメンタルズ取得中...")
            fund = _topix_fetch_fundamentals(tickers)
            progress.progress(80, text="スクリーニング計算中...")
            result = _topix_run_screening(stock_df, md, fund, current_topix_codes=current_codes)
            progress.progress(100, text="完了")
            time.sleep(0.3)
            progress.empty()
            if not result.empty:
                st.session_state["topix_result"] = result

    if "topix_result" in st.session_state:
        result = st.session_state["topix_result"]
        filtered = result.copy()
        if topix_judge_filter:
            filtered = filtered[filtered["TOPIX判定"].isin(topix_judge_filter)]
        if topix_min_turnover > 0:
            filtered = filtered[filtered["年間売買回転率"].fillna(0) >= topix_min_turnover]
        if topix_min_div > 0:
            filtered = filtered[filtered["配当利回り(%)"].fillna(0) >= topix_min_div]
        if topix_max_per < 100:
            filtered = filtered[filtered["PER(倍)"].fillna(999) <= topix_max_per]
        if topix_min_roe > 0:
            filtered = filtered[filtered["ROE(%)"].fillna(0) >= topix_min_roe]

        # サマリー
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("対象銘柄", f"{len(result)}")
        m2.metric("⚪ 新規追加候補", f"{len(result[result['TOPIX判定'] == '⚪ 新規追加候補'])}")
        m3.metric("🟢 継続安全", f"{len(result[result['TOPIX判定'] == '🟢 継続安全'])}")
        m4.metric("🟡 ボーダー", f"{len(result[result['TOPIX判定'] == '🟡 ボーダーライン'])}")
        m5.metric("🔴 除外警戒", f"{len(result[result['TOPIX判定'] == '🔴 除外警戒'])}")
        m6.metric("表示中", f"{len(filtered)}")
        st.divider()

        # ── 表示タブ ──
        topix_view_tab1, topix_view_tab2, topix_view_tab3, topix_view_tab4 = st.tabs([
            "🏆 採用確度ランキング", "📋 テーブル", "📊 分布チャート", "🏭 業種別分析"
        ])

        # ── 採用確度ランキング ──
        with topix_view_tab1:
            st.markdown("#### 🏆 TOPIX新規採用 確度ランキング")
            st.caption(
                "TOPIX加重平均特性（PER≈17倍, PBR≈1.55倍, 配当利回り≈2.16%, ROE≈9%）への近さ、"
                "流動性、時価総額を総合評価してスコアリング（100点満点）"
            )

            ranked = filtered.dropna(subset=["適合スコア"]).sort_values("適合スコア", ascending=False)
            if ranked.empty:
                st.info("スコア算出可能な銘柄がありません")
            else:
                # TOP30をカード表示
                top_n = min(50, len(ranked))
                st.markdown(f"##### TOP {top_n}")

                for idx, (_, r) in enumerate(ranked.head(top_n).iterrows()):
                    rank = idx + 1
                    score = r.get("適合スコア", 0)
                    judge = str(r.get("TOPIX判定", ""))

                    # メダル
                    if rank == 1:
                        medal = "🥇"
                    elif rank == 2:
                        medal = "🥈"
                    elif rank == 3:
                        medal = "🥉"
                    elif rank <= 10:
                        medal = "🔹"
                    else:
                        medal = f"{rank}."

                    # スコアバーの色
                    if score >= 60:
                        bar_color = "#2ecc71"
                    elif score >= 40:
                        bar_color = "#f39c12"
                    else:
                        bar_color = "#e74c3c"

                    code = r.get("銘柄コード", "")
                    name = r.get("銘柄名", "")
                    market = r.get("市場区分", "")
                    kabutan_url = f"https://kabutan.jp/stock/?code={code}"

                    _pe = r.get("PER(倍)")
                    _pbr = r.get("PBR(倍)")
                    _dy = r.get("配当利回り(%)")
                    _roe = r.get("ROE(%)")
                    _ff = r.get("浮動株時価総額(億円)")
                    _tr = r.get("年間売買回転率")

                    pe_str = f"PER {_pe:.1f}" if pd.notna(_pe) else "PER —"
                    pbr_str = f"PBR {_pbr:.2f}" if pd.notna(_pbr) else "PBR —"
                    dy_str = f"利回り {_dy:.2f}%" if pd.notna(_dy) else "利回り —"
                    roe_str = f"ROE {_roe:.1f}%" if pd.notna(_roe) else "ROE —"
                    ff_str = f"浮動株時価総額 {_ff:,.0f}億" if pd.notna(_ff) else ""
                    tr_str = f"回転率 {_tr:.3f}" if pd.notna(_tr) else ""

                    bar_width = min(score, 100)

                    st.markdown(
                        f"""<div style="padding:8px 12px;margin-bottom:6px;border-radius:6px;
                        background:linear-gradient(90deg, {bar_color}22 {bar_width}%, transparent {bar_width}%);
                        border-left:4px solid {bar_color};">
                        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                            <span style="font-size:1.2rem;min-width:36px;">{medal}</span>
                            <a href="{kabutan_url}" target="_blank" style="color:inherit;text-decoration:none;font-weight:bold;font-size:1rem;">
                                {code} {name}
                            </a>
                            <span style="color:#888;font-size:0.8rem;">{market}</span>
                            <span style="font-weight:bold;color:{bar_color};margin-left:auto;">
                                {score:.1f}点
                            </span>
                            <span style="font-size:0.75rem;padding:2px 6px;border-radius:4px;
                            background:{'#ffffff33' if '⚪' in judge else '#2ecc7133' if '🟢' in judge else '#f39c1233' if '🟡' in judge else '#e74c3c33'};">
                                {judge}
                            </span>
                        </div>
                        <div style="display:flex;gap:16px;margin-top:4px;font-size:0.78rem;color:#666;flex-wrap:wrap;">
                            <span>{pe_str}</span><span>{pbr_str}</span>
                            <span>{dy_str}</span><span>{roe_str}</span>
                            <span>{ff_str}</span><span>{tr_str}</span>
                        </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                # スコア分布
                st.markdown("---")
                st.markdown("##### スコア分布")
                fig_hist = px.histogram(
                    ranked, x="適合スコア", nbins=20,
                    title="TOPIX適合スコア分布",
                    color_discrete_sequence=["#3498db"],
                )
                fig_hist.add_vline(x=60, line_dash="dash", line_color="green", annotation_text="高確度ライン")
                fig_hist.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="中確度ライン")
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)

        # ── テーブル表示 ──
        with topix_view_tab2:
            disp_cols = [
                "採用確度ランク", "適合スコア",
                "銘柄コード", "銘柄名", "市場区分", "業種", "株価",
                "時価総額(億円)", "浮動株時価総額(億円)", "年間売買回転率",
                "累積時価総額比率", "TOPIX判定",
                "配当利回り(%)", "PER(倍)", "PBR(倍)", "ROE(%)"
            ]
            disp = filtered.sort_values("適合スコア", ascending=False, na_position="last")
            disp = disp[[c for c in disp_cols if c in disp.columns]]
            st.dataframe(
                disp, use_container_width=True, height=600,
                column_config={
                    "採用確度ランク": st.column_config.NumberColumn("ランク", format="%d"),
                    "適合スコア": st.column_config.ProgressColumn("適合スコア", min_value=0, max_value=100, format="%.1f"),
                    "株価": st.column_config.NumberColumn(format="¥%.0f"),
                    "時価総額(億円)": st.column_config.NumberColumn(format="%.0f"),
                    "浮動株時価総額(億円)": st.column_config.NumberColumn(format="%.0f"),
                    "年間売買回転率": st.column_config.NumberColumn(format="%.4f"),
                    "累積時価総額比率": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                    "配当利回り(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "PER(倍)": st.column_config.NumberColumn(format="%.1f"),
                    "PBR(倍)": st.column_config.NumberColumn(format="%.2f"),
                    "ROE(%)": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

        # ── 分布チャート ──
        with topix_view_tab3:
            vd = filtered.dropna(subset=["年間売買回転率", "浮動株時価総額(億円)"])
            if not vd.empty:
                fig = px.scatter(
                    vd, x="年間売買回転率", y="浮動株時価総額(億円)",
                    color="適合スコア" if "適合スコア" in vd.columns else "TOPIX判定",
                    hover_name="銘柄名",
                    hover_data=["銘柄コード", "配当利回り(%)", "PER(倍)", "PBR(倍)", "ROE(%)"],
                    log_y=True, title="売買回転率 vs 浮動株時価総額（色: 適合スコア）",
                    color_continuous_scale="RdYlGn",
                )
                fig.add_vline(x=_TOPIX_TRADED_VALUE_RATIO_THRESHOLD, line_dash="dash",
                              line_color="red", annotation_text="回転率基準 0.14")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        # ── 業種別分析 ──
        with topix_view_tab4:
            sd = filtered.dropna(subset=["業種", "TOPIX判定"])
            if not sd.empty:
                ss = sd.groupby(["業種", "TOPIX判定"]).size().reset_index(name="件数")
                fig2 = px.bar(
                    ss, x="業種", y="件数", color="TOPIX判定", title="業種別 TOPIX判定分布",
                    color_discrete_map={"⚪ 新規追加候補": "#ecf0f1", "🟢 継続安全": "#2ecc71",
                                        "🟡 ボーダーライン": "#f1c40f", "🔴 除外警戒": "#e74c3c", "データ不足": "#bdc3c7"},
                )
                fig2.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)

        # CSV
        st.divider()
        csv_cols = [
            "採用確度ランク", "適合スコア",
            "銘柄コード", "銘柄名", "市場区分", "業種", "株価",
            "時価総額(億円)", "浮動株時価総額(億円)", "年間売買回転率",
            "累積時価総額比率", "TOPIX判定", "配当利回り(%)", "PER(倍)", "PBR(倍)", "ROE(%)"
        ]
        csv_df = filtered.sort_values("適合スコア", ascending=False, na_position="last")
        csv_df = csv_df[[c for c in csv_cols if c in csv_df.columns]]
        st.download_button(
            "📥 CSV ダウンロード", csv_df.to_csv(index=False).encode("utf-8-sig"),
            f"topix_screening_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", key="topix_csv"
        )

# =====================================================================
# Tab 11: PBR1倍割れ改革 スクリーニング
# =====================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def _pbr_fetch_fundamentals(tickers_tuple):
    """PBR改革スクリーニング用: 並列取得で高速化"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tickers = list(tickers_tuple)
    results = {}

    def _fetch_one(t):
        try:
            tk = yf.Ticker(t)
            info = tk.info
            price = info.get("currentPrice", info.get("regularMarketPrice"))

            # 配当利回り・配当性向を1回の dividends 取得で計算
            div_yield_calc = None
            payout = None
            divs = None
            try:
                divs = tk.dividends
                if divs is not None and len(divs) > 0 and price and price > 0:
                    one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
                    recent = divs[divs.index >= one_year_ago]
                    if len(recent) > 0:
                        annual_div = recent.sum()
                        div_yield_calc = annual_div / price
                        eps = info.get("trailingEps")
                        if eps and eps > 0:
                            payout = annual_div / eps
            except Exception:
                pass
            if div_yield_calc is None:
                div_yield_calc = info.get("trailingAnnualDividendYield") or info.get("dividendYield")

            pbr = info.get("priceToBook")
            doe = (div_yield_calc * pbr) if (div_yield_calc and pbr and pbr > 0) else None

            return t, {
                "marketCap": info.get("marketCap"),
                "currentPrice": price,
                "dividendYield": div_yield_calc,
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "priceToBook": pbr,
                "returnOnEquity": info.get("returnOnEquity"),
                "bookValue": info.get("bookValue"),
                "payoutRatio": payout,
                "doe": doe,
                "trailingEps": info.get("trailingEps"),
            }
        except Exception:
            return t, {}

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                t, data = fut.result(timeout=15)
                results[t] = data
            except Exception:
                results[futures[fut]] = {}
    return results


def _pbr_calc_reform_score(r):
    """
    東証改革先回りスコア（100点満点）
    PBR1倍割れ + 低PER + 高ROE + 配当余力 → DOE/累進配当導入の可能性
    """
    score = 0.0

    # 1. PBR割安度 (0-25点): 低いほど改善余地大
    pbr = r.get("PBR(倍)")
    if pbr is not None and not pd.isna(pbr) and 0 < pbr < 1.0:
        score += 25 * (1.0 - pbr)  # PBR0.5→12.5点, PBR0.3→17.5点
    elif pbr is not None and not pd.isna(pbr) and pbr >= 1.0:
        return 0  # PBR1倍以上は対象外

    # 2. PER割安度 (0-20点): 低いほど良い
    pe = r.get("PER(倍)")
    if pe is not None and not pd.isna(pe) and 0 < pe < 10:
        score += 20 * (1 - pe / 10)  # PER5→10点, PER3→14点

    # 3. ROE (0-20点): 高いほど良い（稼ぐ力がある）
    roe = r.get("ROE(%)")
    if roe is not None and not pd.isna(roe) and roe > 0:
        score += min(20, roe / 15 * 20)  # ROE10→13.3点, ROE15→20点

    # 4. 配当余力 (0-15点): 配当性向が低い＝増配余地大
    payout = r.get("配当性向(%)")
    if payout is not None and not pd.isna(payout) and 0 < payout < 100:
        score += 15 * (1 - payout / 100)  # 性向30%→10.5点

    # 5. DOE水準 (0-10点): 低DOE＝DOE導入で株価上昇余地
    doe = r.get("DOE(%)")
    if doe is not None and not pd.isna(doe) and doe >= 0:
        if doe < 3:
            score += 10 * (1 - doe / 3)  # DOE1%→6.7点
        # DOE3%以上は既に高水準

    # 6. 時価総額 (0-10点): 適度な規模
    mcap = r.get("時価総額(億円)")
    if mcap is not None and not pd.isna(mcap) and mcap > 0:
        if 50 <= mcap <= 3000:
            score += 10  # スタンダード市場の適正規模
        elif mcap > 3000:
            score += 7
        elif mcap > 10:
            score += 5

    return round(score, 1)


with tab11:
    st.subheader("🏛️ PBR1倍割れ改革 スクリーニング")
    st.caption(
        "東証の資本コスト・株価意識改革に未対応のスタンダード市場銘柄から、"
        "DOE導入・累進配当採用の先回り投資候補を抽出"
    )

    with st.expander("📋 スクリーニング条件の考え方", expanded=False):
        st.markdown("""
        **東証改革の背景**: 2023年3月、東証はPBR1倍割れ企業に資本コストや株価を意識した経営の改善を要請。
        プライム市場は対応が進むが、スタンダード市場は未対応企業が多い。

        **先回り投資の着眼点**:
        - **PBR < 1.0**: 純資産割れ → 改善要請の対象
        - **PER < 10**: 利益に対して割安 → 増配・自社株買いの原資あり
        - **ROE > 10%**: 稼ぐ力がある → 株主還元強化の余力
        - **配当性向が低い**: 増配余地が大きい
        - **DOEが低い**: DOE導入で安定配当→株価上昇の触媒に

        **DOE（株主資本配当率）** = 年間配当 / 純資産 ≈ 配当利回り × PBR
        累進配当やDOE採用は、PBR改善の有力な手段として注目されています。
        """)

    # フィルター
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pbr_market = st.multiselect(
            "市場区分",
            ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"],
            default=["スタンダード（内国株式）"],
            key="pbr_market"
        )
    with pc2:
        pbr_max_stocks = st.selectbox(
            "対象数", [100, 200, 500, 1000, 1500],
            index=2, key="pbr_max"
        )
    with pc3:
        pbr_sector_filter = st.text_input(
            "業種キーワード（空欄=全業種）", value="", key="pbr_sector",
            help="例: 建設、機械、卸売"
        )

    pf1, pf2, pf3, pf4 = st.columns(4)
    with pf1:
        pbr_max_pbr = st.number_input("PBR上限(倍)", value=1.0, step=0.1, format="%.1f", key="pbr_pbr")
    with pf2:
        pbr_max_per = st.number_input("PER上限(倍)", value=10.0, step=1.0, key="pbr_per")
    with pf3:
        pbr_min_roe = st.number_input("ROE下限(%)", value=10.0, step=1.0, key="pbr_roe")
    with pf4:
        pbr_min_div = st.number_input("配当利回り下限(%)", value=0.0, step=0.5, key="pbr_div")

    if st.button("🚀 スクリーニング実行", type="primary", key="pbr_run"):
        with st.spinner("銘柄リストを取得中..."):
            stock_df = _topix_fetch_stock_list()
        if stock_df.empty:
            st.error("銘柄リストの取得に失敗しました")
        else:
            if pbr_market:
                stock_df = stock_df[stock_df["market"].isin(pbr_market)]
            if pbr_sector_filter:
                stock_df = stock_df[stock_df["sector"].str.contains(pbr_sector_filter, na=False)]
            stock_df = stock_df.head(pbr_max_stocks)
            st.info(f"対象: {len(stock_df)}件")

            tickers = tuple(stock_df["ticker"].tolist())
            progress = st.progress(0, text="ファンダメンタルズ取得中...")
            fund = _pbr_fetch_fundamentals(tickers)
            progress.progress(100, text="完了")
            time.sleep(0.3)
            progress.empty()

            # 結果構築
            rows = []
            for _, row in stock_df.iterrows():
                t = row["ticker"]
                f = fund.get(t, {})
                if not f:
                    continue
                price = f.get("currentPrice")
                if not price or price <= 0:
                    continue
                pbr = f.get("priceToBook")
                pe = f.get("trailingPE") or f.get("forwardPE")
                roe = f.get("returnOnEquity")
                if roe is not None:
                    roe = round(roe * 100, 2)
                dy = f.get("dividendYield")
                if dy is not None:
                    dy = round(dy * 100, 2)
                payout = f.get("payoutRatio")
                if payout is not None:
                    payout = round(payout * 100, 1)
                doe = f.get("doe")
                if doe is not None:
                    doe = round(doe * 100, 2)
                mcap = f.get("marketCap")
                bps = f.get("bookValue")

                rows.append({
                    "銘柄コード": row["code"], "銘柄名": row["name"],
                    "市場区分": row["market"], "業種": row["sector"],
                    "株価": round(price, 1),
                    "時価総額(億円)": round(mcap / 1e8, 1) if mcap else None,
                    "PBR(倍)": round(pbr, 2) if pbr else None,
                    "PER(倍)": round(pe, 1) if pe else None,
                    "ROE(%)": roe,
                    "配当利回り(%)": dy,
                    "配当性向(%)": payout,
                    "DOE(%)": doe,
                    "BPS": round(bps, 1) if bps else None,
                })

            result = pd.DataFrame(rows)
            if result.empty:
                st.warning("該当銘柄なし")
            else:
                # フィルター適用
                if pbr_max_pbr > 0:
                    result = result[result["PBR(倍)"].fillna(999) <= pbr_max_pbr]
                if pbr_max_per > 0:
                    result = result[result["PER(倍)"].fillna(999) <= pbr_max_per]
                if pbr_min_roe > 0:
                    result = result[result["ROE(%)"].fillna(0) >= pbr_min_roe]
                if pbr_min_div > 0:
                    result = result[result["配当利回り(%)"].fillna(0) >= pbr_min_div]

                # スコア計算
                result["改革先回りスコア"] = result.apply(_pbr_calc_reform_score, axis=1)
                result = result[result["改革先回りスコア"] > 0]
                result = result.sort_values("改革先回りスコア", ascending=False).reset_index(drop=True)
                result["ランク"] = range(1, len(result) + 1)

                st.session_state["pbr_result"] = result

    if "pbr_result" in st.session_state:
        result = st.session_state["pbr_result"]

        m1, m2, m3 = st.columns(3)
        m1.metric("該当銘柄", f"{len(result)}")
        avg_pbr = result["PBR(倍)"].mean()
        m2.metric("平均PBR", f"{avg_pbr:.2f}" if pd.notna(avg_pbr) else "—")
        avg_doe = result["DOE(%)"].mean()
        m3.metric("平均DOE", f"{avg_doe:.2f}%" if pd.notna(avg_doe) else "—")
        st.divider()

        pbr_tab1, pbr_tab2 = st.tabs(["🏆 ランキング", "📋 テーブル"])

        with pbr_tab1:
            st.markdown("#### 🏆 DOE・累進配当 先回り候補ランキング")
            top_n = min(30, len(result))
            for _, r in result.head(top_n).iterrows():
                rank = r["ランク"]
                sc = r["改革先回りスコア"]
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🔹" if rank <= 10 else f"{rank}."
                code = r["銘柄コード"]
                name = r["銘柄名"]
                kabutan = f"https://kabutan.jp/stock/?code={code}"

                _pbr = r.get("PBR(倍)")
                _pe = r.get("PER(倍)")
                _roe = r.get("ROE(%)")
                _dy = r.get("配当利回り(%)")
                _po = r.get("配当性向(%)")
                _doe = r.get("DOE(%)")
                _mcap = r.get("時価総額(億円)")

                bar_w = min(sc, 100)
                bc = "#e74c3c" if sc >= 50 else "#f39c12" if sc >= 30 else "#3498db"

                st.markdown(
                    f"""<div style="padding:8px 12px;margin-bottom:6px;border-radius:6px;
                    background:linear-gradient(90deg, {bc}22 {bar_w}%, transparent {bar_w}%);
                    border-left:4px solid {bc};">
                    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                        <span style="font-size:1.2rem;min-width:36px;">{medal}</span>
                        <a href="{kabutan}" target="_blank" style="color:inherit;text-decoration:none;font-weight:bold;">
                            {code} {name}
                        </a>
                        <span style="color:#888;font-size:0.8rem;">{r.get('業種','')}</span>
                        <span style="font-weight:bold;color:{bc};margin-left:auto;">{sc:.1f}点</span>
                    </div>
                    <div style="display:flex;gap:14px;margin-top:4px;font-size:0.78rem;color:#666;flex-wrap:wrap;">
                        <span>PBR {_pbr:.2f}</span>
                        <span>PER {_pe:.1f}</span>
                        <span>ROE {_roe:.1f}%</span>
                        <span>利回り {_dy:.2f}%</span>
                        <span>性向 {_po:.0f}%</span>
                        <span>DOE {_doe:.2f}%</span>
                        <span>時価総額 {_mcap:,.0f}億</span>
                    </div>
                    </div>""",
                    unsafe_allow_html=True
                )

        with pbr_tab2:
            disp_cols = [
                "ランク", "改革先回りスコア",
                "銘柄コード", "銘柄名", "業種", "株価",
                "時価総額(億円)", "PBR(倍)", "PER(倍)", "ROE(%)",
                "配当利回り(%)", "配当性向(%)", "DOE(%)", "BPS"
            ]
            disp = result[[c for c in disp_cols if c in result.columns]]
            st.dataframe(
                disp, use_container_width=True, height=600,
                column_config={
                    "ランク": st.column_config.NumberColumn(format="%d"),
                    "改革先回りスコア": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                    "株価": st.column_config.NumberColumn(format="¥%.0f"),
                    "時価総額(億円)": st.column_config.NumberColumn(format="%.0f"),
                    "PBR(倍)": st.column_config.NumberColumn(format="%.2f"),
                    "PER(倍)": st.column_config.NumberColumn(format="%.1f"),
                    "ROE(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "配当利回り(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "配当性向(%)": st.column_config.NumberColumn(format="%.0f%%"),
                    "DOE(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "BPS": st.column_config.NumberColumn(format="%.0f"),
                },
            )

        # CSV
        st.divider()
        csv_cols = [
            "ランク", "改革先回りスコア",
            "銘柄コード", "銘柄名", "業種", "株価",
            "時価総額(億円)", "PBR(倍)", "PER(倍)", "ROE(%)",
            "配当利回り(%)", "配当性向(%)", "DOE(%)", "BPS"
        ]
        csv_df = result[[c for c in csv_cols if c in result.columns]]
        st.download_button(
            "📥 CSV ダウンロード", csv_df.to_csv(index=False).encode("utf-8-sig"),
            f"pbr_reform_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", key="pbr_csv"
        )

        # チャート
        st.divider()
        vd = result.dropna(subset=["PBR(倍)", "ROE(%)"])
        if not vd.empty:
            fig = px.scatter(
                vd, x="PBR(倍)", y="ROE(%)",
                size="時価総額(億円)", color="改革先回りスコア",
                hover_name="銘柄名",
                hover_data=["銘柄コード", "PER(倍)", "配当利回り(%)", "DOE(%)"],
                title="PBR vs ROE（サイズ: 時価総額、色: 改革先回りスコア）",
                color_continuous_scale="RdYlGn",
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="PBR 1.0倍")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# Tab 12: アクティビスト追跡
# =====================================================================

# アクティビスト定義
_ACTIVISTS = {
    "シティインデックスイレブンス": {
        "aliases": ["シティインデックス", "City Index Eleventh", "村上ファンド", "村上世彰",
                    "レノ", "Reno", "南青山不動産", "オフィスサポート"],
        "type": "🇯🇵 国内", "style": "株主提案・経営改革要求",
    },
    "ストラテジックキャピタル": {
        "aliases": ["Strategic Capital", "丸木強"],
        "type": "🇯🇵 国内", "style": "株主提案・資本効率改善",
    },
    "オアシス・マネジメント": {
        "aliases": ["Oasis Management", "オアシス"],
        "type": "🇭🇰 香港", "style": "経営改革・株主還元要求",
    },
    "エリオット・インベストメント": {
        "aliases": ["Elliott Investment", "Elliott Management", "エリオット"],
        "type": "🇺🇸 米国", "style": "アクティビスト・イベントドリブン",
    },
    "光通信": {
        "aliases": ["光通信", "HIKARI TSUSHIN"],
        "type": "🇯🇵 国内", "style": "大量取得・長期保有",
    },
    "ダルトン・インベストメンツ": {
        "aliases": ["Dalton Investments", "ダルトン"],
        "type": "🇺🇸 米国", "style": "バリュー投資・経営関与",
    },
    "バリューアクト・キャピタル": {
        "aliases": ["ValueAct Capital", "バリューアクト"],
        "type": "🇺🇸 米国", "style": "取締役派遣・経営改革",
    },
    "旧村上ファンド系": {
        "aliases": ["エスグラントコーポレーション", "野村絢", "C&Iホールディングス",
                    "村上絢", "村上玲"],
        "type": "🇯🇵 国内", "style": "株主提案・TOB",
    },
    "清原達郎": {
        "aliases": ["清原達郎"],
        "type": "🇯🇵 個人", "style": "割安小型株・長期集中投資",
    },
    "片山晃": {
        "aliases": ["片山晃", "五月"],
        "type": "🇯🇵 個人", "style": "成長株・モメンタム投資",
    },
    "井村俊哉": {
        "aliases": ["井村俊哉", "Kaihou", "株式会社Kaihou"],
        "type": "🇯🇵 個人", "style": "バリュー株・割安成長株",
    },
    "ブラックロック": {
        "aliases": ["ブラックロック", "BlackRock", "BLACKROCK"],
        "type": "🇺🇸 米国", "style": "パッシブ/アクティブ・大量保有",
    },
}


_ACTIVIST_CACHE_FILE = os.path.join(CACHE_DIR if os.path.exists("topix_cache") else ".", "activist_cache.json")

def _load_activist_cache():
    """ファイルキャッシュから読み込み"""
    try:
        if os.path.exists(_ACTIVIST_CACHE_FILE):
            with open(_ACTIVIST_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
    except Exception:
        pass
    return []

def _save_activist_cache(data):
    """ファイルキャッシュに保存"""
    try:
        os.makedirs(os.path.dirname(_ACTIVIST_CACHE_FILE) or ".", exist_ok=True)
        with open(_ACTIVIST_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass

def _scrape_edinetdb_activist(query_tuples):
    """edinetdb.jpの大株主検索からアクティビスト保有銘柄を並列取得"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_results = []

    def _fetch_one(query, activist_name):
        results = []
        try:
            url = f"https://edinetdb.jp/shareholders?q={requests.utils.quote(query)}"
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return results
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                return results
            for tr in table.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if len(tds) >= 5:
                    results.append({
                        "アクティビスト": activist_name,
                        "報告者": tds[0].get_text(strip=True),
                        "銘柄名": tds[1].get_text(strip=True),
                        "コード": tds[2].get_text(strip=True),
                        "保有割合": tds[3].get_text(strip=True),
                        "報告日": tds[4].get_text(strip=True),
                    })
        except Exception:
            pass
        return results

    # 3並列で取得（レート制限を考慮しつつ高速化）
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_fetch_one, q, n): n for q, n in query_tuples}
        for fut in as_completed(futures):
            try:
                all_results.extend(fut.result(timeout=25))
            except Exception:
                pass
    return all_results


def _scrape_kabutan_5pct_news():
    """株探の5%ルールニュースからアクティビスト保有を検索"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_results = []

    # アクティビスト名→検索用キーワードのマッピング
    _search_names = {}
    for activist, info in _ACTIVISTS.items():
        for alias in [activist] + info.get("aliases", []):
            _search_names[alias] = activist

    def _check_stock(code):
        """株探の銘柄ページから5%ルールニュースを取得"""
        results = []
        try:
            url = f"https://kabutan.jp/stock/holder/?code={code}"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return results
            soup = BeautifulSoup(resp.text, "html.parser")
            # 銘柄名取得
            stock_name = ""
            title_tag = soup.find("div", class_="si_i1_1")
            if title_tag:
                h2 = title_tag.find("h2")
                if h2:
                    stock_name = h2.get_text(strip=True)
            # ニューステーブルから5%ルール関連を抽出
            for table in soup.find_all("table", class_="s_news_list"):
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) >= 3:
                        badge = tds[1].get_text(strip=True)
                        news_text = tds[2].get_text(strip=True)
                        if "５％" in badge or "5%" in badge:
                            # アクティビスト名をマッチ
                            for name, activist in _search_names.items():
                                if name in news_text:
                                    date_text = tds[0].get_text(strip=True).replace("\xa0", " ")
                                    # 日付を正規化 (26/03/06 -> 2026-03-06)
                                    import re
                                    dm = re.match(r"(\d{2})/(\d{2})/(\d{2})", date_text)
                                    if dm:
                                        y, m, d = dm.groups()
                                        date_norm = f"20{y}-{m}-{d}"
                                    else:
                                        date_norm = date_text[:10]
                                    results.append({
                                        "アクティビスト": activist,
                                        "報告者": name,
                                        "銘柄名": stock_name or str(code),
                                        "コード": str(code),
                                        "保有割合": "",
                                        "報告日": date_norm,
                                        "ソース": "kabutan",
                                    })
                                    break
        except Exception:
            pass
        return results

    # 大株主テーブルからもアクティビスト名を検索
    def _check_holder_table(code):
        results = []
        try:
            url = f"https://kabutan.jp/stock/holder/?code={code}"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return results
            soup = BeautifulSoup(resp.text, "html.parser")
            stock_name = ""
            title_tag = soup.find("div", class_="si_i1_1")
            if title_tag:
                h2 = title_tag.find("h2")
                if h2:
                    stock_name = h2.get_text(strip=True)
            for table in soup.find_all("table", class_="stock_holder_1"):
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) >= 3:
                        holder_name = tds[0].get_text(strip=True)
                        ratio = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                        for name, activist in _search_names.items():
                            if name in holder_name:
                                results.append({
                                    "アクティビスト": activist,
                                    "報告者": holder_name,
                                    "銘柄名": stock_name or str(code),
                                    "コード": str(code),
                                    "保有割合": f"{ratio}%",
                                    "報告日": "",
                                    "ソース": "kabutan_holder",
                                })
                                break
        except Exception:
            pass
        return results

    return _check_stock, _check_holder_table


def _merge_activist_data(existing, new_data):
    """既存キャッシュに新規データをマージ（重複除去）"""
    seen = set()
    merged = []
    for r in new_data + existing:
        key = (r.get("コード",""), r.get("アクティビスト",""), r.get("報告日",""))
        if key not in seen:
            seen.add(key)
            merged.append(r)
    merged.sort(key=lambda x: x.get("報告日", ""), reverse=True)
    return merged


with tab12:
    st.subheader("🦈 アクティビスト追跡")
    st.caption(
        "EDINET DB の大量保有報告書データからアクティビスト投資家の保有銘柄を取得。"
        "初回取得後はキャッシュから即座に表示、差分のみ更新。"
    )

    with st.expander("📋 追跡対象アクティビスト", expanded=False):
        for name, info in _ACTIVISTS.items():
            st.markdown(f"- **{name}** {info['type']} — {info['style']}")

    # 検索クエリ定義（edinetdb.jpの部分一致検索用）
    _ACTIVIST_QUERIES = [
        ("シティインデックス", "シティインデックスイレブンス"),
        ("ストラテジックキャピタル", "ストラテジックキャピタル"),
        ("オアシス", "オアシス・マネジメント"),
        ("エリオット", "エリオット・インベストメント"),
        ("光通信", "光通信"),
        ("ダルトン", "ダルトン・インベストメンツ"),
        ("バリューアクト", "バリューアクト・キャピタル"),
        ("レノ", "シティインデックスイレブンス"),
        ("南青山不動産", "シティインデックスイレブンス"),
        ("野村絢", "旧村上ファンド系"),
        ("C&Iホールディングス", "旧村上ファンド系"),
        ("清原達郎", "清原達郎"),
        ("片山晃", "片山晃"),
        ("井村", "井村俊哉"),
        ("Kaihou", "井村俊哉"),
        ("ブラックロック", "ブラックロック"),
        ("BlackRock", "ブラックロック"),
    ]

    act_selected = st.multiselect(
        "取得するアクティビスト",
        list(_ACTIVISTS.keys()),
        default=list(_ACTIVISTS.keys()),
        key="act_select"
    )

    # キャッシュから自動読み込み（ページ表示時）
    if "act_edinetdb" not in st.session_state:
        cached = _load_activist_cache()
        if cached:
            _cdf = pd.DataFrame(cached)
            _cdf = _cdf.drop_duplicates(subset=["コード", "アクティビスト"])
            _cdf = _cdf.sort_values("報告日", ascending=False).reset_index(drop=True)
            st.session_state["act_edinetdb"] = _cdf

    _bc1, _bc2 = st.columns(2)
    with _bc1:
        if st.button("🔍 初回取得 / 全件再取得", type="primary", key="act_fetch"):
            queries = tuple(
                (q, name) for q, name in _ACTIVIST_QUERIES if name in act_selected
            )
            with st.spinner(f"EDINET DB ({len(queries)}件) + 株探 5%ルール検索中..."):
                raw = _scrape_edinetdb_activist(queries)

                # 株探の5%ルールニュースも検索（EDINET DBにないデータを補完）
                _check_news, _check_holders = _scrape_kabutan_5pct_news()
                # EDINET DBで取得した銘柄コードに対して株探も確認
                edinet_codes = set(r.get("コード","") for r in raw)
                # 追加で主要な注目銘柄も検索
                extra_codes = ["6072", "1518", "1514"]  # 地盤ネット、三井松島、住石
                all_codes = list(edinet_codes | set(extra_codes))
                from concurrent.futures import ThreadPoolExecutor
                kabutan_results = []
                with ThreadPoolExecutor(max_workers=5) as ex:
                    for res in ex.map(_check_news, all_codes[:50]):
                        kabutan_results.extend(res)

                raw.extend(kabutan_results)

            if not raw:
                st.warning("取得失敗。しばらく待ってから再試行してください。")
            else:
                _save_activist_cache(raw)
                df = pd.DataFrame(raw)
                df = df.drop_duplicates(subset=["コード", "アクティビスト"])
                df = df.sort_values("報告日", ascending=False).reset_index(drop=True)
                st.session_state["act_edinetdb"] = df
                kb_count = len(kabutan_results)
                st.success(f"🦈 {len(df)}件取得（EDINET DB + 株探{kb_count}件）、キャッシュ更新済み")
    with _bc2:
        if st.button("🔄 差分更新（最新のみ）", key="act_update"):
            queries = tuple(
                (q, name) for q, name in _ACTIVIST_QUERIES if name in act_selected
            )
            with st.spinner("差分取得中..."):
                raw = _scrape_edinetdb_activist(queries)
            if raw:
                existing = _load_activist_cache()
                merged = _merge_activist_data(existing, raw)
                new_count = len(merged) - len(existing)
                _save_activist_cache(merged)
                df = pd.DataFrame(merged)
                df = df.drop_duplicates(subset=["コード", "アクティビスト"])
                df = df.sort_values("報告日", ascending=False).reset_index(drop=True)
                st.session_state["act_edinetdb"] = df
                if new_count > 0:
                    st.success(f"🆕 {new_count}件の新規報告を追加（合計{len(df)}件）")
                else:
                    st.info(f"新規報告なし（キャッシュ{len(df)}件は最新）")

    if os.path.exists(_ACTIVIST_CACHE_FILE):
        _age = time.time() - os.path.getmtime(_ACTIVIST_CACHE_FILE)
        _age_str = f"{int(_age//3600)}時間{int((_age%3600)//60)}分前" if _age > 3600 else f"{int(_age//60)}分前"
        st.caption(f"💾 キャッシュ最終更新: {_age_str}")

    if "act_edinetdb" in st.session_state:
        df = st.session_state["act_edinetdb"]

        # サマリー
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("保有銘柄数", f"{df['コード'].nunique()}")
        mc2.metric("アクティビスト数", f"{df['アクティビスト'].nunique()}")
        recent = df[df["報告日"] >= (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")]
        mc3.metric("直近6ヶ月の報告", f"{len(recent)}件")
        st.divider()

        # フィルター
        fc1, fc2 = st.columns(2)
        with fc1:
            act_filt = st.multiselect(
                "アクティビスト絞り込み", df["アクティビスト"].unique().tolist(),
                default=df["アクティビスト"].unique().tolist(), key="act_filt2"
            )
        with fc2:
            recent_only = st.checkbox("直近6ヶ月の報告のみ", value=False, key="act_recent")

        filtered = df[df["アクティビスト"].isin(act_filt)]
        if recent_only:
            cutoff = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            filtered = filtered[filtered["報告日"] >= cutoff]

        # テーブル表示
        st.dataframe(
            filtered[["アクティビスト", "銘柄名", "コード", "保有割合", "報告日", "報告者"]],
            use_container_width=True, height=400,
        )

        # 株価パフォーマンス取得
        if st.button("📈 株価パフォーマンス取得", key="act_perf_btn"):
            codes = filtered["コード"].unique().tolist()
            tickers = tuple(sorted(set(f"{c}.T" for c in codes if c.isdigit())))

            @st.cache_data(ttl=1800, show_spinner=False)
            def _fetch_act_prices(tickers_tuple):
                pd_data = {}
                tlist = list(tickers_tuple)
                try:
                    data = yf.download(tlist, period="6mo", progress=False, threads=True)
                    if not data.empty:
                        for t in tlist:
                            try:
                                close = data["Close"][t].dropna() if len(tlist) > 1 else data["Close"].dropna()
                                if len(close) > 5:
                                    pd_data[t] = close
                            except Exception:
                                pass
                except Exception:
                    pass
                return pd_data

            with st.spinner(f"{len(tickers)}銘柄の株価を取得中..."):
                prices = _fetch_act_prices(tickers)

            rows = []
            for _, r in filtered.iterrows():
                code = r["コード"]
                ticker = f"{code}.T"
                close = prices.get(ticker)
                if close is None or len(close) < 10:
                    continue
                cur = close.iloc[-1]
                r1w = (cur / close.iloc[-min(5, len(close))] - 1) * 100
                r1m = (cur / close.iloc[-min(21, len(close))] - 1) * 100
                r3m = (cur / close.iloc[-min(63, len(close))] - 1) * 100

                # ── テクニカル指標 ──
                # RSI(14)
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi_series = 100 - (100 / (1 + rs))
                rsi = round(rsi_series.iloc[-1], 1) if len(rsi_series.dropna()) > 0 else None

                # 25日移動平均乖離率
                ma25 = close.rolling(25).mean()
                dev25 = round((cur / ma25.iloc[-1] - 1) * 100, 1) if pd.notna(ma25.iloc[-1]) and ma25.iloc[-1] > 0 else None

                # 出来高比率（直近5日平均 / 20日平均）
                vol_ratio = None
                try:
                    vol_data = yf.download(ticker, period="2mo", progress=False)
                    if not vol_data.empty and "Volume" in vol_data.columns:
                        vol = vol_data["Volume"].dropna()
                        if len(vol) >= 20:
                            v5 = vol.iloc[-5:].mean()
                            v20 = vol.iloc[-20:].mean()
                            vol_ratio = round(v5 / v20 * 100, 0) if v20 > 0 else None
                except Exception:
                    pass

                # ── 売買シグナル判定 ──
                signals = []
                if rsi is not None:
                    if rsi >= 70:
                        signals.append(("🔴過熱", "#e74c3c"))
                    elif rsi >= 60:
                        signals.append(("🟡やや過熱", "#f1c40f"))
                    elif rsi <= 30:
                        signals.append(("🟢売られすぎ", "#2ecc71"))
                    elif rsi <= 40:
                        signals.append(("🟢割安圏", "#2ecc71"))

                if dev25 is not None:
                    if dev25 > 15:
                        signals.append(("🔴乖離大", "#e74c3c"))
                    elif dev25 > 8:
                        signals.append(("🟡乖離注意", "#f1c40f"))
                    elif dev25 < -10:
                        signals.append(("🟢下方乖離", "#2ecc71"))

                if vol_ratio is not None:
                    if vol_ratio >= 200:
                        signals.append(("📈出来高急増", "#3498db"))

                signal_text = " ".join(
                    f'<span style="color:{c};font-size:0.7rem;font-weight:bold;">{s}</span>'
                    for s, c in signals
                ) if signals else '<span style="opacity:0.5;font-size:0.7rem;">—</span>'

                # 報告日からのリターン
                try:
                    fdate = pd.Timestamp(r["報告日"])
                    after = close[close.index >= fdate]
                    before = close[close.index < fdate]
                    if len(before) > 0 and len(after) > 0:
                        ret_since = (cur / before.iloc[-1] - 1) * 100
                    else:
                        ret_since = None
                except Exception:
                    ret_since = None
                rows.append({
                    "アクティビスト": r["アクティビスト"],
                    "コード": code, "銘柄名": r["銘柄名"],
                    "保有割合": r["保有割合"], "報告日": r["報告日"],
                    "現在株価": round(cur, 1),
                    "1W(%)": round(r1w, 1), "1M(%)": round(r1m, 1),
                    "3M(%)": round(r3m, 1),
                    "報告日来(%)": round(ret_since, 1) if ret_since is not None else None,
                    "RSI": rsi, "乖離率(%)": dev25,
                    "出来高比(%)": vol_ratio,
                    "_signal_html": signal_text,
                })

            perf = pd.DataFrame(rows)
            if not perf.empty:
                perf = perf.sort_values("報告日来(%)", ascending=False, na_position="last")
                st.session_state["act_perf"] = perf

        if "act_perf" in st.session_state:
            perf = st.session_state["act_perf"]
            st.markdown("#### 🏆 保有銘柄パフォーマンス")

            for _, r in perf.iterrows():
                code = r["コード"]
                kabutan = f"https://kabutan.jp/stock/?code={code}"
                ret_since = r.get("報告日来(%)")
                rc = "#2ecc71" if (ret_since or 0) >= 0 else "#e74c3c"
                r1m = r.get("1M(%)", 0)
                r1m_c = "#2ecc71" if r1m >= 0 else "#e74c3c"
                sig = r.get("_signal_html", "")
                _rsi = r.get("RSI")
                _dev = r.get("乖離率(%)")
                _vr = r.get("出来高比(%)")
                rsi_str = f"RSI {_rsi:.0f}" if pd.notna(_rsi) else ""
                dev_str = f"乖離 {_dev:+.1f}%" if pd.notna(_dev) else ""
                vr_str = f"出来高比 {_vr:.0f}%" if pd.notna(_vr) else ""

                since_str = f"{ret_since:+.1f}%" if pd.notna(ret_since) else "—"
                _card = (
                    f'<div style="padding:8px 12px;margin-bottom:4px;border-radius:6px;'
                    f'border-left:4px solid {rc};background:rgba(128,128,128,0.08);">'
                    f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                    f'<a href="{kabutan}" target="_blank" style="color:{rc};text-decoration:none;font-weight:bold;min-width:160px;">'
                    f'{code} {r["銘柄名"]}</a>'
                    f'<span style="background:#34495e;color:#ecf0f1;padding:2px 6px;border-radius:4px;font-size:0.72rem;">{r["アクティビスト"]}</span>'
                    f'<span style="opacity:0.6;font-size:0.75rem;">{r["保有割合"]} | {r["報告日"]}</span>'
                    f'<span style="margin-left:auto;color:inherit;">¥{r["現在株価"]:,.0f}</span>'
                    f'<span style="color:{r1m_c};font-weight:bold;">1M {r1m:+.1f}%</span>'
                    f'<span style="color:{rc};font-weight:bold;">報告日来 {since_str}</span>'
                    f'</div>'
                    f'<div style="display:flex;gap:12px;margin-top:3px;font-size:0.72rem;flex-wrap:wrap;align-items:center;">'
                    f'{sig}'
                    f'<span style="opacity:0.5;">{rsi_str}</span>'
                    f'<span style="opacity:0.5;">{dev_str}</span>'
                    f'<span style="opacity:0.5;">{vr_str}</span>'
                    f'</div></div>'
                )
                st.markdown(_card, unsafe_allow_html=True)

            st.divider()
            st.dataframe(
                perf, use_container_width=True, height=400,
                column_config={
                    "現在株価": st.column_config.NumberColumn(format="¥%.0f"),
                    "1W(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "1M(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "3M(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "報告日来(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "RSI": st.column_config.NumberColumn(format="%.0f"),
                    "乖離率(%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "出来高比(%)": st.column_config.NumberColumn(format="%.0f%%"),
                    "_signal_html": None,  # HTML列は非表示
                },
            )
            st.download_button(
                "📥 CSV", perf.to_csv(index=False).encode("utf-8-sig"),
                f"activist_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", key="act_csv"
            )

            # ══════════════════════════════════════════════
            # 統計分析: 最適エントリータイミング & アクティビスト別パフォーマンス
            # ══════════════════════════════════════════════
            st.divider()
            st.markdown("#### 📐 統計分析: 最適エントリータイミング")
            st.caption("報告日からN日後に買った場合のリターンを統計的に分析")

            if st.button("🔬 統計分析を実行", key="act_stats_btn"):
                # 全銘柄の報告日からN日後リターンを計算
                analysis_rows = []
                codes_in_perf = perf["コード"].unique().tolist()
                tickers_for_stats = tuple(sorted(set(f"{c}.T" for c in codes_in_perf if c.isdigit())))

                @st.cache_data(ttl=1800, show_spinner=False)
                def _fetch_stats_prices(tickers_tuple):
                    pd_data = {}
                    tlist = list(tickers_tuple)
                    try:
                        data = yf.download(tlist, period="2y", progress=False, threads=True)
                        if not data.empty:
                            for t in tlist:
                                try:
                                    close = data["Close"][t].dropna() if len(tlist) > 1 else data["Close"].dropna()
                                    if len(close) > 20:
                                        pd_data[t] = close
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return pd_data

                with st.spinner("2年分の株価データを取得中..."):
                    stats_prices = _fetch_stats_prices(tickers_for_stats)

                # N日後リターン計算
                entry_days = [0, 1, 3, 5, 10, 20, 40, 60]
                hold_periods = [5, 10, 20, 40, 60, 120]

                timing_data = []  # エントリータイミング分析用
                activist_data = []  # アクティビスト別分析用

                for _, row in perf.iterrows():
                    code = row["コード"]
                    ticker = f"{code}.T"
                    close = stats_prices.get(ticker)
                    if close is None or len(close) < 60:
                        continue
                    try:
                        fdate = pd.Timestamp(row["報告日"])
                    except Exception:
                        continue

                    for entry_offset in entry_days:
                        entry_date = fdate + pd.Timedelta(days=entry_offset)
                        # 最も近い営業日を探す
                        entry_candidates = close[close.index >= entry_date]
                        if len(entry_candidates) < 2:
                            continue
                        entry_price = entry_candidates.iloc[0]

                        for hold in hold_periods:
                            exit_date = entry_candidates.index[0] + pd.Timedelta(days=hold)
                            exit_candidates = close[close.index >= exit_date]
                            if len(exit_candidates) < 1:
                                continue
                            exit_price = exit_candidates.iloc[0]
                            ret = (exit_price / entry_price - 1) * 100

                            timing_data.append({
                                "コード": code, "銘柄名": row["銘柄名"],
                                "アクティビスト": row["アクティビスト"],
                                "報告日": row["報告日"],
                                "エントリー日数": entry_offset,
                                "保有日数": hold,
                                "リターン(%)": round(ret, 2),
                            })

                    # アクティビスト別: 報告日翌日買い→各期間のリターン
                    entry_candidates = close[close.index >= fdate]
                    if len(entry_candidates) < 2:
                        continue
                    entry_price = entry_candidates.iloc[min(1, len(entry_candidates)-1)]
                    for hold in [20, 60, 120]:
                        exit_date = entry_candidates.index[0] + pd.Timedelta(days=hold)
                        exit_cands = close[close.index >= exit_date]
                        if len(exit_cands) < 1:
                            continue
                        ret = (exit_cands.iloc[0] / entry_price - 1) * 100
                        activist_data.append({
                            "アクティビスト": row["アクティビスト"],
                            "コード": code, "銘柄名": row["銘柄名"],
                            f"{hold}日後(%)": round(ret, 2),
                        })

                if timing_data:
                    tdf = pd.DataFrame(timing_data)
                    st.session_state["act_timing"] = tdf
                if activist_data:
                    adf = pd.DataFrame(activist_data)
                    st.session_state["act_activist_stats"] = adf

            # ── 結果表示 ──
            if "act_timing" in st.session_state:
                tdf = st.session_state["act_timing"]

                # 1. 最適エントリータイミング
                st.markdown("##### ⏱️ 報告日から何日後に買うのが最適か")
                pivot = tdf.groupby(["エントリー日数", "保有日数"])["リターン(%)"].agg(
                    ["mean", "median", "std", "count"]
                ).reset_index()
                pivot.columns = ["エントリー日数", "保有日数", "平均(%)", "中央値(%)", "標準偏差", "サンプル数"]
                pivot["シャープ比"] = pivot["平均(%)"] / pivot["標準偏差"].replace(0, np.nan)
                pivot = pivot.round(2)

                # ベストタイミング（シャープ比最大）
                best = pivot.loc[pivot["シャープ比"].idxmax()] if not pivot.empty else None
                if best is not None:
                    st.success(
                        f"🏆 最適エントリー: 報告日から **{int(best['エントリー日数'])}日後** に買い、"
                        f"**{int(best['保有日数'])}日間** 保有 → "
                        f"平均リターン **{best['平均(%)']:+.1f}%**（中央値 {best['中央値(%)']:+.1f}%、"
                        f"シャープ比 {best['シャープ比']:.2f}、n={int(best['サンプル数'])}）"
                    )

                # ヒートマップ
                heat_data = pivot.pivot_table(
                    index="エントリー日数", columns="保有日数", values="平均(%)"
                )
                if not heat_data.empty:
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=heat_data.values,
                        x=[f"{c}日" for c in heat_data.columns],
                        y=[f"{r}日後" for r in heat_data.index],
                        colorscale="RdYlGn", zmid=0,
                        text=np.round(heat_data.values, 1),
                        texttemplate="%{text}%",
                        hovertemplate="エントリー: %{y}<br>保有: %{x}<br>平均リターン: %{z:.1f}%",
                    ))
                    fig_heat.update_layout(
                        title="エントリータイミング × 保有期間 → 平均リターン(%)",
                        xaxis_title="保有期間", yaxis_title="報告日からの日数",
                        height=350,
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                # シャープ比ヒートマップ
                sharpe_data = pivot.pivot_table(
                    index="エントリー日数", columns="保有日数", values="シャープ比"
                )
                if not sharpe_data.empty:
                    fig_sharpe = go.Figure(data=go.Heatmap(
                        z=sharpe_data.values,
                        x=[f"{c}日" for c in sharpe_data.columns],
                        y=[f"{r}日後" for r in sharpe_data.index],
                        colorscale="RdYlGn", zmid=0,
                        text=np.round(sharpe_data.values, 2),
                        texttemplate="%{text}",
                    ))
                    fig_sharpe.update_layout(
                        title="シャープ比（リターン/リスク）— 高いほど効率的",
                        xaxis_title="保有期間", yaxis_title="報告日からの日数",
                        height=350,
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)

                # 詳細テーブル
                with st.expander("📋 詳細データ"):
                    st.dataframe(pivot, use_container_width=True)

            # 2. アクティビスト別パフォーマンスランキング
            if "act_timing" in st.session_state:
                tdf = st.session_state["act_timing"]
                st.divider()
                st.markdown("##### 🦈 アクティビスト別パフォーマンスランキング")
                st.caption("報告日翌日エントリー、各保有期間の平均リターンで比較")

                act_rank = tdf[tdf["エントリー日数"] == 1].groupby("アクティビスト").agg(
                    銘柄数=("コード", "nunique"),
                    平均リターン=("リターン(%)", "mean"),
                    中央値=("リターン(%)", "median"),
                    勝率=("リターン(%)", lambda x: (x > 0).mean() * 100),
                    最大=("リターン(%)", "max"),
                    最小=("リターン(%)", "min"),
                ).round(1).sort_values("平均リターン", ascending=False)

                if not act_rank.empty:
                    for i, (activist, row) in enumerate(act_rank.iterrows()):
                        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                        avg = row["平均リターン"]
                        rc = "#2ecc71" if avg >= 0 else "#e74c3c"
                        st.markdown(
                            f"""<div style="padding:8px 12px;margin-bottom:4px;border-radius:6px;
                            border-left:4px solid {rc};background:rgba(128,128,128,0.08);">
                            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                                <span style="font-size:1.1rem;">{medal}</span>
                                <span style="font-weight:bold;">{activist}</span>
                                <span style="color:{rc};font-weight:bold;margin-left:auto;">
                                    平均 {avg:+.1f}%</span>
                                <span style="opacity:0.7;">中央値 {row['中央値']:+.1f}%</span>
                                <span style="opacity:0.7;">勝率 {row['勝率']:.0f}%</span>
                                <span style="opacity:0.7;">{int(row['銘柄数'])}銘柄</span>
                            </div></div>""",
                            unsafe_allow_html=True
                        )

                    st.dataframe(act_rank, use_container_width=True)

            # 3. 最適エントリー日に近い「今が買い時」銘柄
            if "act_timing" in st.session_state and "act_edinetdb" in st.session_state:
                tdf = st.session_state["act_timing"]
                edf = st.session_state["act_edinetdb"]

                # 最適エントリー日数を算出
                pivot = tdf.groupby(["エントリー日数", "保有日数"])["リターン(%)"].agg(
                    ["mean", "std"]
                ).reset_index()
                pivot.columns = ["entry", "hold", "mean", "std"]
                pivot["sharpe"] = pivot["mean"] / pivot["std"].replace(0, np.nan)
                best_row = pivot.loc[pivot["sharpe"].idxmax()] if not pivot.empty else None

                if best_row is not None:
                    best_entry = int(best_row["entry"])
                    best_hold = int(best_row["hold"])
                    margin = max(3, best_entry // 2)  # ±margin日を許容

                    st.divider()
                    st.markdown(f"##### 🎯 今が買い時の銘柄（報告日から{best_entry}日±{margin}日）")
                    st.caption(
                        f"統計的に最もパフォーマンスが良いエントリータイミング "
                        f"（報告日から{best_entry}日後買い→{best_hold}日保有）に近い銘柄を抽出"
                    )

                    today = pd.Timestamp.now().normalize()
                    candidates = []
                    for _, r in edf.iterrows():
                        try:
                            fdate = pd.Timestamp(r["報告日"])
                        except Exception:
                            continue
                        days_since = (today - fdate).days
                        # 最適エントリー日 ± margin の範囲内
                        if best_entry - margin <= days_since <= best_entry + margin:
                            diff = abs(days_since - best_entry)
                            candidates.append({
                                "コード": r["コード"],
                                "銘柄名": r["銘柄名"],
                                "アクティビスト": r["アクティビスト"],
                                "保有割合": r["保有割合"],
                                "報告日": r["報告日"],
                                "経過日数": days_since,
                                "最適日との差": diff,
                            })

                    if candidates:
                        cdf = pd.DataFrame(candidates)
                        cdf = cdf.drop_duplicates(subset=["コード"]).sort_values("最適日との差")

                        # テーマ・配当利回りを一括取得
                        from concurrent.futures import ThreadPoolExecutor
                        _codes_for_info = cdf["コード"].tolist()

                        @st.cache_data(ttl=3600, show_spinner=False)
                        def _fetch_entry_info(codes_tuple):
                            info_map = {}
                            def _get(code):
                                ticker = f"{code}.T"
                                themes = guess_themes_by_name(ticker, "")
                                dy = fetch_dividend_yield(code)
                                return code, themes, dy
                            with ThreadPoolExecutor(max_workers=10) as ex:
                                for code, themes, dy in ex.map(
                                    lambda c: _get(c), list(codes_tuple)
                                ):
                                    info_map[code] = {"themes": themes, "dy": dy}
                            return info_map

                        with st.spinner("テーマ・配当利回りを取得中..."):
                            info_map = _fetch_entry_info(tuple(_codes_for_info))

                        for _, c in cdf.iterrows():
                            code = c["コード"]
                            kabutan = f"https://kabutan.jp/stock/?code={code}"
                            diff = c["最適日との差"]
                            if diff == 0:
                                timing = "🔥 ジャストタイミング"
                                tc = "#e74c3c"
                            elif diff <= 2:
                                timing = "🟢 ほぼ最適"
                                tc = "#2ecc71"
                            else:
                                timing = "🟡 許容範囲"
                                tc = "#f1c40f"

                            # テーマ・配当利回り
                            _info = info_map.get(code, {})
                            _themes = _info.get("themes", [])
                            _dy = _info.get("dy")
                            theme_html = ""
                            if _themes:
                                theme_html = " ".join(
                                    f"<span style='background:#1a237e;color:#7986cb;padding:1px 5px;"
                                    f"border-radius:6px;font-size:0.65rem;'>{th}</span>"
                                    for th in _themes[:3]
                                )
                            dy_html = ""
                            if _dy is not None and _dy > 0:
                                if _dy >= 4.0:
                                    dy_html = f"<span style='color:#00c853;font-weight:bold;'>💰💰 {_dy:.2f}%</span>"
                                elif _dy >= 3.0:
                                    dy_html = f"<span style='color:#00c853;font-weight:bold;'>💰 {_dy:.2f}%</span>"
                                else:
                                    dy_html = f"<span style='opacity:0.7;'>{_dy:.2f}%</span>"

                            _card = (
                                f'<div style="padding:8px 12px;margin-bottom:4px;border-radius:6px;'
                                f'border-left:4px solid {tc};background:rgba(128,128,128,0.08);">'
                                f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                                f'<span style="font-size:0.85rem;color:{tc};font-weight:bold;">{timing}</span>'
                                f'<a href="{kabutan}" target="_blank" style="color:inherit;text-decoration:none;font-weight:bold;">{code} {c["銘柄名"]}</a>'
                                f'<span style="background:#34495e;color:#ecf0f1;padding:2px 6px;border-radius:4px;font-size:0.72rem;">{c["アクティビスト"]}</span>'
                                f'{dy_html} {theme_html}'
                                f'</div>'
                                f'<div style="display:flex;gap:10px;margin-top:3px;font-size:0.75rem;opacity:0.6;flex-wrap:wrap;">'
                                f'<span>{c["保有割合"]}</span>'
                                f'<span>報告日 {c["報告日"]}</span>'
                                f'<span>経過{c["経過日数"]}日</span>'
                                f'</div></div>'
                            )
                            st.markdown(_card, unsafe_allow_html=True)

                        st.caption(
                            f"推奨: これらの銘柄を買い、約{best_hold}日間保有 "
                            f"→ 統計的期待リターン {best_row['mean']:+.1f}%"
                        )
                    else:
                        st.info(
                            f"現在、報告日から{best_entry}日±{margin}日の範囲にある銘柄はありません。"
                            "次の報告を待つか、差分更新で最新データを取得してください。"
                        )

# フッター
st.divider()
st.caption("⚠️ 本ダッシュボードは情報提供目的であり、投資助言ではありません。投資判断はご自身の責任で行ってください。")
st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =====================================================================
# Tab 13: マイウォッチリスト
# =====================================================================
WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_watchlist.json")

def save_watchlist(stocks):
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(stocks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_watchlist():
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

@st.cache_data(ttl=86400, show_spinner=False)
def _wl_fetch_fundamental(codes_tuple):
    """ウォッチリスト用: ファンダメンタル情報（EDINET DB優先、yfinanceフォールバック）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import hashlib
    results = {}
    codes = list(codes_tuple)

    # ── EDINET DB ディスクキャッシュ ──
    _EDB_CACHE_DIR = os.path.join(tempfile.gettempdir(), "edinetdb_cache")
    os.makedirs(_EDB_CACHE_DIR, exist_ok=True)

    def _edb_cache_path(code):
        today = datetime.now().strftime("%Y-%m-%d")
        ck = hashlib.md5(f"{today}:get_company:{code}".encode()).hexdigest()
        return os.path.join(_EDB_CACHE_DIR, f"{ck}.json")

    def _edb_load(code):
        p = _edb_cache_path(code)
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _edb_save(code, data):
        try:
            with open(_edb_cache_path(code), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def _edb_fetch(code):
        """EDINET DB REST API経由で企業情報+財務データ取得（キャッシュ付き）"""
        cached = _edb_load(code)
        if cached is not None:
            return cached
        return None

    def _parse_edinetdb(code, edb):
        """EDINET DB REST APIレスポンスからファンダメンタル情報を抽出（分割調整済み）"""
        res = {}
        if not edb:
            return None

        res["name"] = edb.get("name_ja", "") or edb.get("company_name", "") or edb.get("filerName", "") or edb.get("name", "")
        # STOCK_DICTから日本語名を補完
        ticker = f"{code}.T"
        jp_name = STOCK_DICT.get(ticker, "")
        if jp_name and (not res["name"] or all(ord(c) < 0x3000 for c in res["name"])):
            res["name"] = jp_name

        # 財務データ（リスト、最新が先頭）
        financials = edb.get("financials", [])
        if isinstance(financials, list) and len(financials) > 0:
            fin = financials[0]
        else:
            fin = {}

        # ratios（分割調整済みEPS/BPS/DPS含む）
        ratios = edb.get("ratios", [])
        if isinstance(ratios, list) and len(ratios) > 0:
            rat = ratios[0]  # 最新年度
        else:
            rat = {}

        # PER
        res["per"] = rat.get("per") or fin.get("per")

        # ROE
        roe_val = rat.get("roe") or rat.get("roe_official") or fin.get("roe_official")
        if roe_val is not None:
            rv = float(roe_val)
            res["roe"] = round(rv * 100, 1) if abs(rv) < 1 else round(rv, 1)
        else:
            res["roe"] = None

        # EPS成長率（分割調整済みadjusted_epsを使用、データ少ない銘柄も対応）
        res["eps_cagr"] = None
        res["per_3y"] = None
        adj_eps_list = []
        for r in (ratios if isinstance(ratios, list) else []):
            ae = r.get("adjusted_eps")
            if ae is not None and float(ae) > 0:
                adj_eps_list.append(float(ae))
        # ratiosにadjusted_epsがない場合、financialsのepsを使う
        if len(adj_eps_list) < 2:
            adj_eps_list = []
            for f in (financials if isinstance(financials, list) else []):
                e = f.get("eps")
                if e is not None and float(e) > 0:
                    adj_eps_list.append(float(e))

        per_val = res.get("per")
        if len(adj_eps_list) >= 2:
            latest = adj_eps_list[0]
            n = min(len(adj_eps_list) - 1, 5)
            oldest = adj_eps_list[n]
            if oldest > 0 and latest > 0:
                cagr = (latest / oldest) ** (1 / n) - 1
                if -0.9 < cagr < 2.0:
                    res["eps_cagr"] = round(cagr * 100, 1)
                    if per_val and float(per_val) > 0:
                        res["per_3y"] = round(float(per_val) / ((1 + cagr) ** 3), 1)

        # データが1年分しかない場合（上場間もない銘柄）→ 決算短信の前年比やratiosの成長率を使う
        if res["eps_cagr"] is None:
            # ratiosにeps_growth等があれば使う
            for r in (ratios if isinstance(ratios, list) else []):
                eg = r.get("eps_growth") or r.get("eps_cagr")
                if eg is not None:
                    egf = float(eg)
                    # 小数比率か%かを判定
                    if abs(egf) > 5: egf = egf / 100  # %表記→小数
                    if -0.9 < egf < 2.0:
                        res["eps_cagr"] = round(egf * 100, 1)
                        if per_val and float(per_val) > 0:
                            res["per_3y"] = round(float(per_val) / ((1 + egf) ** 3), 1)
                    break
        # まだ取れない場合のフラグ（yfinanceフォールバックで補完される）
        res["_eps_from_edinetdb"] = res["eps_cagr"] is not None

        # 配当（分割調整済みadjusted_dividend_per_share優先）
        adj_dps = rat.get("adjusted_dividend_per_share") or fin.get("adjusted_dividend_per_share") or fin.get("dividend_per_share")
        res["div_rate"] = float(adj_dps) if adj_dps else None
        res["div_yield"] = None
        res["div_yield_3y"] = None
        res["div_cagr"] = None
        # 配当成長率（分割調整済み）
        div_series = []
        for r in (ratios if isinstance(ratios, list) else []):
            d = r.get("adjusted_dividend_per_share")
            if d is not None and float(d) > 0:
                div_series.append(float(d))
        if len(div_series) < 2:
            for f in (financials if isinstance(financials, list) else []):
                d = f.get("dividend_per_share")
                if d is not None and float(d) > 0:
                    div_series.append(float(d))
        if len(div_series) >= 2:
            n = min(len(div_series) - 1, 5)
            if div_series[n] > 0 and div_series[0] > 0:
                dcagr = (div_series[0] / div_series[n]) ** (1 / n) - 1
                res["div_cagr"] = round(dcagr * 100, 1)

        # 健全性スコア
        res["health_score"] = edb.get("health_score") or edb.get("financial_health_score")

        res["shikiho"] = "—"
        res["next_earnings"] = "—"
        res["themes"] = []
        res["_source"] = "EDINET DB"
        res["biz_trend"] = _judge_earnings_trend(edb)
        return res

    def _judge_earnings_trend(edb):
        """EDINET DB financialsから長期業績トレンドを判定"""
        if not edb:
            return "—"
        financials = edb.get("financials", [])
        if not isinstance(financials, list) or len(financials) < 2:
            return "—"
        # 売上 or 営業利益の推移で判定（最新が先頭）
        rev_list = []
        oi_list = []
        for f in financials:
            r = f.get("revenue") or f.get("net_sales")
            o = f.get("operating_income")
            if r is not None: rev_list.append(float(r))
            if o is not None: oi_list.append(float(o))
        # 営業利益優先、なければ売上
        series = oi_list if len(oi_list) >= 2 else rev_list
        if len(series) < 2:
            return "—"
        latest = series[0]
        oldest = series[-1]
        mid = series[len(series)//2] if len(series) >= 3 else (latest + oldest) / 2
        if oldest <= 0:
            return "📈" if latest > 0 else "—"
        growth = (latest - oldest) / abs(oldest)
        # 途中の推移も見る
        monotone_up = all(series[i] >= series[i+1] * 0.95 for i in range(len(series)-1))
        monotone_down = all(series[i] <= series[i+1] * 1.05 for i in range(len(series)-1))
        if growth > 0.3 and monotone_up:
            return "📈📈"  # 強い右肩上がり
        elif growth > 0.1:
            return "📈"    # 右肩上がり
        elif growth < -0.3 and monotone_down:
            return "📉📉"  # 強い右肩下がり
        elif growth < -0.1:
            return "📉"    # 右肩下がり
        else:
            return "➡️"    # ヨコヨコ

    def _fetch_yutai(code):
        """株探から株主優待情報を取得（優待内容・優待利回り）"""
        try:
            url = f"https://kabutan.jp/stock/yutai?code={code}"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code != 200:
                return None, None
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text()
            # 「優待なし」判定
            if "株主優待はありません" in text or "優待情報はありません" in text:
                return None, None
            # 優待利回りを探す
            import re as _re
            yutai_yield = None
            m = _re.search(r"優待利回り[^\d]*?([\d.]+)\s*%", text)
            if m:
                yutai_yield = float(m.group(1))
            # 優待内容を探す
            yutai_desc = None
            for tag in soup.find_all(["td", "div", "p"]):
                t = tag.get_text(strip=True)
                if any(kw in t for kw in ["割引券", "商品券", "優待券", "ポイント", "カタログ", "食事券", "QUOカード", "クオカード", "自社製品", "お米券", "ギフト"]):
                    if len(t) < 100:
                        yutai_desc = t
                        break
            return yutai_yield, yutai_desc
        except Exception:
            return None, None

    def _get_fund_yf(code):
        """yfinanceフォールバック"""
        ticker = f"{code}.T"
        res = {}
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            res["per"] = info.get("trailingPE") or info.get("forwardPE")
            roe = info.get("returnOnEquity")
            res["roe"] = round(roe * 100, 1) if roe else None
            res["eps_cagr"] = None
            res["per_3y"] = None
            try:
                current_per = res.get("per")
                lp = info.get("currentPrice") or info.get("regularMarketPrice")
                adjusted_eps = lp / current_per if current_per and current_per > 0 and lp else None
                forward_pe = info.get("forwardPE")
                adjusted_fwd = lp / forward_pe if forward_pe and forward_pe > 0 and lp else None
                if adjusted_eps and adjusted_eps > 0 and adjusted_fwd and adjusted_fwd > 0:
                    eg = (adjusted_fwd - adjusted_eps) / abs(adjusted_eps)
                    if -0.9 < eg < 2.0:
                        res["eps_cagr"] = round(eg * 100, 1)
                        if current_per and current_per > 0:
                            res["per_3y"] = round(current_per / ((1 + eg) ** 3), 1)
            except Exception:
                pass
            rec_mean = info.get("recommendationMean")
            if rec_mean is not None:
                if rec_mean <= 1.5: res["shikiho"] = "強気 ⬆️"
                elif rec_mean <= 2.5: res["shikiho"] = "やや強気 ↗️"
                elif rec_mean <= 3.5: res["shikiho"] = "中立 ➡️"
                elif rec_mean <= 4.5: res["shikiho"] = "やや弱気 ↘️"
                else: res["shikiho"] = "弱気 ⬇️"
            else:
                res["shikiho"] = "—"
            try:
                cal = t.calendar
                if cal is not None:
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if ed:
                            res["next_earnings"] = str(pd.Timestamp(ed[0]).date()) if isinstance(ed, list) else str(pd.Timestamp(ed).date())
                        else: res["next_earnings"] = "—"
                    elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                        ed = cal.loc["Earnings Date"]
                        res["next_earnings"] = str(pd.Timestamp(ed.iloc[0] if hasattr(ed, 'iloc') else ed).date())
                    else: res["next_earnings"] = "—"
                else: res["next_earnings"] = "—"
            except Exception:
                res["next_earnings"] = "—"
            res["div_rate"] = info.get("dividendRate")
            res["div_yield"] = None
            res["div_yield_3y"] = None
            res["div_cagr"] = None
            try:
                divs = t.dividends
                if divs is not None and len(divs) >= 2:
                    annual = divs.resample("YE").sum()
                    annual = annual[annual > 0]
                    if len(annual) >= 2:
                        n_years = min(len(annual) - 1, 5)
                        if n_years >= 1 and float(annual.iloc[-(n_years+1)]) > 0:
                            res["div_cagr"] = round(((float(annual.iloc[-1]) / float(annual.iloc[-(n_years+1)])) ** (1/n_years) - 1) * 100, 1)
            except Exception:
                pass
            name = info.get("shortName", "")
            jp_name = STOCK_DICT.get(ticker, "")
            if jp_name and (not name or all(ord(c) < 0x3000 for c in name)):
                name = jp_name
            res["name"] = name
            themes = guess_themes_by_name(ticker, name)
            if not themes and jp_name:
                themes = guess_themes_by_name(ticker, jp_name)
            res["themes"] = themes
            res["_source"] = "yfinance"
            # 業績トレンド（yfinance: earningsGrowth/revenueGrowthから簡易判定）
            eg = info.get("earningsGrowth")
            rg = info.get("revenueGrowth")
            g = eg if eg is not None else rg
            if g is not None:
                if g > 0.2: res["biz_trend"] = "📈📈"
                elif g > 0.05: res["biz_trend"] = "📈"
                elif g > -0.05: res["biz_trend"] = "➡️"
                elif g > -0.2: res["biz_trend"] = "📉"
                else: res["biz_trend"] = "📉📉"
            else:
                res["biz_trend"] = "—"
        except Exception as ex:
            res["name"] = f"取得失敗({ex})"
            res["_source"] = "error"
            for k in ["per","roe","shikiho","next_earnings","themes","div_rate","div_yield","div_yield_3y","div_cagr","eps_cagr","per_3y"]:
                res.setdefault(k, None if k not in ["shikiho","next_earnings","themes"] else ("—" if k != "themes" else []))
        return res

    def _get_fund(code):
        """EDINET DB優先 → yfinanceフォールバック"""
        # 1. EDINET DB
        edb = _edb_fetch(code)
        edb_res = _parse_edinetdb(code, edb) if edb else None

        # 2. yfinance（四季報評価・決算日・テーマはここから）
        yf_res = _get_fund_yf(code)

        # 3. 優待情報
        yutai_yield, yutai_desc = _fetch_yutai(code)

        # 4. マージ（EDINET DB優先、yfinanceで補完）
        if edb_res:
            merged = {**yf_res, **{k: v for k, v in edb_res.items() if v is not None and v != "—" and v != []}}
            merged["_source"] = "EDINET DB"
            for k in ["shikiho", "next_earnings", "themes"]:
                if not merged.get(k) or merged[k] == "—" or merged[k] == []:
                    merged[k] = yf_res.get(k, "—" if k != "themes" else [])
            if not edb_res.get("_eps_from_edinetdb"):
                for k in ["eps_cagr", "per_3y"]:
                    yf_val = yf_res.get(k)
                    if yf_val is not None:
                        merged[k] = yf_val
            merged["yutai_yield"] = yutai_yield
            merged["yutai_desc"] = yutai_desc
            return code, merged
        else:
            yf_res["yutai_yield"] = yutai_yield
            yf_res["yutai_desc"] = yutai_desc
            return code, yf_res

    with ThreadPoolExecutor(max_workers=min(len(codes), 20)) as pool:
        futs = {pool.submit(_get_fund, c): c for c in codes}
        for fut in as_completed(futs):
            try:
                code, res = fut.result()
                results[code] = res
            except Exception:
                pass
    return results


@st.cache_data(ttl=300, show_spinner=False)
def _wl_fetch_prices(codes_tuple):
    """ウォッチリスト用: 株価＋テクニカル＋指数相関（5分キャッシュ）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    codes = list(codes_tuple)

    # TOPIX・日経225の日次リターンを事前取得（TOPIXはETF 1306.Tで代替）
    _idx_returns = {}
    for idx_ticker, idx_name in [("1306.T", "TOPIX"), ("^N225", "日経225")]:
        try:
            ih = yf.Ticker(idx_ticker).history(period="6mo")
            if ih is not None and len(ih) > 5:
                _idx_returns[idx_name] = ih["Close"].pct_change().dropna()
        except Exception:
            pass

    def _judge_higher_tf(hist):
        """週足・月足トレンド判定（上位足シグナル）"""
        result = {"weekly": "—", "monthly": "—", "weekly_emoji": "⚪", "monthly_emoji": "⚪"}
        if hist is None or len(hist) < 20:
            return result
        close = hist["Close"]

        def _trend(series):
            """移動平均の傾きと位置関係からトレンド判定"""
            if len(series) < 5:
                return "データ不足", "⚪"
            vals = series.values
            last = float(vals[-1])
            ma5 = float(series.iloc[-5:].mean())
            ma_all = float(series.mean())
            # 直近値と移動平均の位置関係
            above_short = last > ma5
            above_long = last > ma_all
            # 移動平均の傾き
            if len(series) >= 10:
                ma_early = float(series.iloc[:len(series)//2].mean())
                ma_late = float(series.iloc[len(series)//2:].mean())
                slope_up = ma_late > ma_early * 1.02
                slope_down = ma_late < ma_early * 0.98
            else:
                slope_up = above_short and above_long
                slope_down = not above_short and not above_long
            # 底値圏判定（直近が全期間安値の10%以内）
            period_low = float(series.min())
            period_high = float(series.max())
            rng = period_high - period_low if period_high > period_low else 1
            near_bottom = (last - period_low) / rng < 0.15
            near_top = (period_high - last) / rng < 0.15

            if slope_up and above_short and above_long:
                return "🟢⬆️", "🟢"
            elif slope_down and not above_short and not above_long:
                if near_bottom:
                    return "🟣⏬底", "🟣"
                return "🔴⬇️", "🔴"
            elif near_bottom and not slope_up:
                return "🟣⏬底", "🟣"
            elif near_top and slope_up:
                return "🟠⏫天", "🟠"
            elif not slope_up and not slope_down:
                return "🟡➡️", "🟡"
            elif slope_up:
                return "🔵↗️", "🔵"
            else:
                return "🟠↘️", "🟠"

        # 週足（5日ごとの終値）
        weekly = close.iloc[::5]
        if len(weekly) >= 5:
            result["weekly"], result["weekly_emoji"] = _trend(weekly)

        # 月足（20日ごとの終値）
        monthly = close.iloc[::20]
        if len(monthly) >= 3:
            result["monthly"], result["monthly_emoji"] = _trend(monthly)

        return result

    def _get_price(code):
        ticker = f"{code}.T"
        res = {}
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if hist is not None and len(hist) > 0:
                res["last_price"] = float(hist["Close"].iloc[-1])
                res["hist"] = hist
                # 前日比
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    if prev > 0:
                        res["daily_change_pct"] = round((res["last_price"] - prev) / prev * 100, 2)
                    else:
                        res["daily_change_pct"] = None
                else:
                    res["daily_change_pct"] = None
                # 出来高急増検出（直近5日平均 vs 20日平均）
                if "Volume" in hist.columns and len(hist) >= 20:
                    vol = hist["Volume"]
                    vol_5 = float(vol.iloc[-5:].mean())
                    vol_20 = float(vol.iloc[-20:].mean())
                    if vol_20 > 0:
                        vol_ratio = vol_5 / vol_20
                        res["vol_ratio"] = round(vol_ratio, 2)
                        res["vol_surge"] = vol_ratio >= 1.5
                    else:
                        res["vol_ratio"] = None
                        res["vol_surge"] = False
                else:
                    res["vol_ratio"] = None
                    res["vol_surge"] = False
            else:
                res["last_price"] = None
                res["hist"] = None
                res["vol_ratio"] = None
                res["vol_surge"] = False
                res["daily_change_pct"] = None
        except Exception:
            res["last_price"] = None
            res["hist"] = None
            res["vol_ratio"] = None
            res["vol_surge"] = False
            res["daily_change_pct"] = None

        # テクニカル分析
        if res.get("hist") is not None and len(res["hist"]) >= 20:
            try:
                dow, _dc, dow_detail = analyze_dow_theory(res["hist"])
                breakout_text, _bk = analyze_breakout(res["hist"])
                _theo, deviation = calc_theoretical_price(res["hist"])
                _dl, _dcc, demand_score = calc_supply_demand(res["hist"])
                momentum = calc_stock_momentum(res["hist"])
                wave = calc_oshime_signal(res["hist"])
                sig_text, sig_color, sig_score = generate_composite_signal(
                    dow, breakout_text, deviation, demand_score, momentum, wave.get("score", 0)
                )
                res.update({"signal": sig_text, "signal_color": sig_color, "signal_score": sig_score,
                            "dow": dow, "dow_detail": dow_detail, "breakout": breakout_text,
                            "deviation": deviation, "demand": demand_score, "momentum": momentum})
                dj_stage, dj_sig, _dje, dj_detail = calc_daijunkan_macd(res["hist"])
                res.update({"dj_stage": dj_stage, "dj_signal": dj_sig, "dj_detail": dj_detail})
                res.update({"wave": wave})
            except Exception:
                res.update({"signal": "分析エラー", "signal_color": "gray", "signal_score": 0,
                            "dow": "—", "dow_detail": "", "breakout": "—",
                            "deviation": None, "demand": 0, "momentum": 0,
                            "dj_stage": None, "dj_signal": "—", "dj_detail": "",
                            "wave": {"signal": "様子見", "emoji": "⚪", "score": 0, "detail": "",
                                     "hits": 0, "rsi14": None, "dev_ma20": None, "drawdown": None,
                                     "N": None, "V": None, "E": None, "NT": None, "wave_type": "—"}})
        else:
            res.update({"signal": "データ不足", "signal_color": "gray", "signal_score": 0,
                        "dow": "—", "dow_detail": "", "breakout": "—",
                        "deviation": None, "demand": 0, "momentum": 0,
                        "dj_stage": None, "dj_signal": "—", "dj_detail": "",
                        "wave": {"signal": "様子見", "emoji": "⚪", "score": 0, "detail": "",
                                 "hits": 0, "rsi14": None, "dev_ma20": None, "drawdown": None,
                                 "N": None, "V": None, "E": None, "NT": None, "wave_type": "—"}})

        # 上位足トレンド
        htf = _judge_higher_tf(res.get("hist"))
        res["weekly_trend"] = htf["weekly"]
        res["weekly_emoji"] = htf["weekly_emoji"]
        res["monthly_trend"] = htf["monthly"]
        res["monthly_emoji"] = htf["monthly_emoji"]

        # 75MA押し目シグナル
        res["ma75_signal"] = ""
        if res.get("hist") is not None and len(res["hist"]) >= 75:
            close = res["hist"]["Close"]
            last = float(close.iloc[-1])
            ma75 = float(close.iloc[-75:].mean())
            # MA75からの乖離率
            ma75_dev = (last - ma75) / ma75 * 100
            # タッチ判定（±3%以内）
            touching = abs(ma75_dev) <= 3.0
            approaching = 0 < ma75_dev <= 5.0  # 上から近づいている
            # 直近5日でMA75に接近中か
            if len(close) >= 80:
                prev_dev = (float(close.iloc[-5]) - float(close.iloc[-80:-5].mean())) / float(close.iloc[-80:-5].mean()) * 100
                narrowing = prev_dev > ma75_dev and ma75_dev > 0  # 乖離が縮小中
            else:
                narrowing = False

            # 上位足が上向きの場合のみ押し目シグナル
            wt = res.get("weekly_trend", "")
            mt = res.get("monthly_trend", "")
            upper_bullish = ("⬆️" in wt or "↗️" in wt) and ("⬆️" in mt or "↗️" in mt or "➡️" in mt)

            if touching and upper_bullish:
                res["ma75_signal"] = "🎯押し目！"
            elif approaching and narrowing and upper_bullish:
                res["ma75_signal"] = "👀接近中"
            elif touching and not upper_bullish:
                res["ma75_signal"] = "75MAタッチ"
            elif last < ma75 and upper_bullish:
                res["ma75_signal"] = "⚠️MA下抜け"

        # 指数との相関
        res["corr_topix"] = None
        res["corr_n225"] = None
        if res.get("hist") is not None and len(res["hist"]) > 10:
            stock_ret = res["hist"]["Close"].pct_change().dropna()
            for idx_name, key in [("TOPIX", "corr_topix"), ("日経225", "corr_n225")]:
                idx_ret = _idx_returns.get(idx_name)
                if idx_ret is not None:
                    try:
                        merged = pd.DataFrame({"s": stock_ret, "i": idx_ret}).dropna()
                        if len(merged) > 10:
                            res[key] = round(float(merged["s"].corr(merged["i"])), 2)
                    except Exception:
                        pass

        return code, res

    with ThreadPoolExecutor(max_workers=min(len(codes), 20)) as pool:
        futs = {pool.submit(_get_price, c): c for c in codes}
        for fut in as_completed(futs):
            try:
                code, res = fut.result()
                results[code] = res
            except Exception:
                pass
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def _wl_check_activist(codes_tuple):
    """ウォッチリスト用: アクティビスト有無チェック"""
    act_map = {}
    try:
        cache = _load_activist_cache()
        if cache:
            for entry in cache:
                c = str(entry.get("コード", ""))
                if c in [str(x) for x in codes_tuple]:
                    act_map.setdefault(c, []).append(entry.get("アクティビスト", "不明"))
    except Exception:
        pass
    return act_map


with tab13:
    st.subheader("👁️ マイウォッチリスト")
    st.caption("監視銘柄のエントリータイミングを総合判定 — PEG・PER・ROE・四季報評価・決算日・アクティビスト・テーマ")

    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist()

    wl = st.session_state["watchlist"]

    # 銘柄追加UI
    with st.expander("➕ 銘柄を追加 / 削除", expanded=not bool(wl)):
        # 1. 銘柄名で検索（辞書から候補表示）
        wl_selected = st.selectbox(
            "🔍 銘柄名で検索（辞書にある銘柄）",
            options=[""] + _SEARCH_OPTIONS,
            index=0,
            key="wl_stock_search",
            placeholder="銘柄名を入力して検索...",
        )
        if st.button("➕ 検索結果から追加", key="wl_add_search") and wl_selected:
            ticker = _OPTION_TO_TICKER.get(wl_selected)
            if ticker:
                code = ticker.replace(".T", "")
                if code in wl:
                    st.warning(f"{STOCK_DICT.get(ticker, ticker)} は既に登録済みです")
                else:
                    wl[code] = {"name": STOCK_DICT.get(ticker, code), "added": str(pd.Timestamp.now().date())}
                    save_watchlist(wl)
                    st.rerun()

        # 2. 証券コードで直接追加
        st.caption("辞書にない銘柄は証券コードで直接追加できます（カンマ区切りで複数可）")
        wl_code_input = st.text_input("証券コード", placeholder="8501, 6758, 7203", key="wl_code_input")
        if st.button("➕ コードで追加", key="wl_add_code", use_container_width=True):
            if wl_code_input.strip():
                added = 0
                for raw in wl_code_input.split(","):
                    code = raw.strip().replace(".T", "")
                    if code and code not in wl:
                        with st.spinner(f"{code} の銘柄情報を取得中..."):
                            name = resolve_stock_name(f"{code}.T")
                        wl[code] = {"name": name, "added": str(pd.Timestamp.now().date())}
                        added += 1
                if added:
                    save_watchlist(wl)
                    st.success(f"{added}銘柄追加")
                    st.rerun()

        # 3. 削除
        st.markdown("---")
        if wl:
            del_code = st.selectbox("削除する銘柄", list(wl.keys()),
                                    format_func=lambda c: f"{c} {wl[c].get('name','')}", key="wl_del_sel")
            if st.button("🗑️ 削除", key="wl_del", use_container_width=True):
                wl.pop(del_code, None)
                save_watchlist(wl)
                st.rerun()

    if not wl:
        st.info("ウォッチリストに銘柄を追加してください。")
    else:
        codes = list(wl.keys())
        with st.spinner("ウォッチリスト分析中..."):
            fund_data = _wl_fetch_fundamental(tuple(codes))
            price_data = _wl_fetch_prices(tuple(codes))
            act_map = _wl_check_activist(tuple(codes))
        # マージ（price優先、fundで補完）+ 利回り計算
        wl_data = {}
        for c in codes:
            f = fund_data.get(c, {})
            p = price_data.get(c, {})
            d = {**f, **p, "code": c, "ticker": f"{c}.T"}
            # 利回り計算（予想配当÷最新株価）
            div_rate = f.get("div_rate")
            lp = d.get("last_price")
            if div_rate and div_rate > 0 and lp and lp > 0:
                d["div_yield"] = round(div_rate / lp * 100, 2)
                cagr_pct = f.get("div_cagr")
                if cagr_pct is not None:
                    cagr = cagr_pct / 100
                    future_div = div_rate * (1 + cagr) ** 3
                    d["div_yield_3y"] = round(future_div / lp * 100, 2)
            wl_data[c] = d

        # ── シグナルスコア順にソート ──
        sorted_codes = sorted(codes, key=lambda c: wl_data.get(c, {}).get("signal_score", 0), reverse=True)

        # ── 期待値スコア算出（売買シグナル+上位足+ファンダ総合） ──
        def _calc_expect(d):
            s = d.get("signal_score", 0)
            # 上位足ボーナス
            mt = d.get("monthly_trend", "")
            wt = d.get("weekly_trend", "")
            for t in [mt, wt]:
                if "⬆️" in t: s += 10
                elif "↗️" in t: s += 5
                elif "⬇️" in t: s -= 10
                elif "↘️" in t: s -= 5
                elif "底" in t: s += 8
            # ファンダボーナス
            if d.get("eps_cagr") and d["eps_cagr"] > 15: s += 10
            elif d.get("eps_cagr") and d["eps_cagr"] > 5: s += 5
            if d.get("roe") and d["roe"] > 15: s += 5
            if d.get("div_yield") and d["div_yield"] > 3: s += 5
            if d.get("vol_surge"): s += 5
            # 波動理論ボーナス
            wave = d.get("wave", {})
            ws = wave.get("score", 0)
            s += ws * 0.5
            return max(min(round(s), 100), -100)

        # ── サマリーテーブル ──
        rows = []
        for c in sorted_codes:
            d = wl_data.get(c, {})
            acts = act_map.get(c, [])
            act_str = "🦈" if acts else ""
            themes = d.get("themes", [])
            theme_str = ", ".join(themes[:2]) if themes else ""
            vol_mark = "🔥" if d.get("vol_surge") else ""
            expect = _calc_expect(d)
            yutai_url = f"https://shikiho.toyokeizai.net/stocks/{c}/yutai"
            yutai_val = f"🎁{d['yutai_yield']:.1f}%" if d.get("yutai_yield") else ""
            # 決算日フォーマット（M/D形式、14日以内は🔔付き）
            ne_raw = d.get("next_earnings", "")
            ne_display = ""
            if ne_raw and ne_raw != "—":
                try:
                    ne_dt = pd.Timestamp(ne_raw)
                    days_to = (ne_dt - pd.Timestamp.now()).days
                    ne_display = ne_dt.strftime("%-m/%-d") if os.name != "nt" else f"{ne_dt.month}/{ne_dt.day}"
                    if 0 <= days_to <= 14:
                        ne_display = f"🔔{ne_display}"
                    elif days_to < 0:
                        ne_display = f"✅{ne_display}"
                except Exception:
                    ne_display = ne_raw[:5]

            wave = d.get("wave", {})
            wave_sig = wave.get("signal", "様子見")

            rows.append({
                "コード": c,
                "銘柄": d.get("name", wl[c].get("name", ""))[:8],
                "決算": ne_display,
                "業績": d.get("biz_trend", "—"),
                "月足": d.get("monthly_trend", "—"),
                "週足": d.get("weekly_trend", "—"),
                "75MA": d.get("ma75_signal", ""),
                "株価": f"¥{d['last_price']:,.0f}" if d.get("last_price") else "—",
                "前日比": f"{d['daily_change_pct']:+.1f}%" if d.get("daily_change_pct") is not None else "—",
                "出来高": f"{vol_mark}×{d['vol_ratio']:.1f}" if d.get("vol_ratio") else "",
                "シグナル": d.get("signal", "—"),
                "期待値": expect,
                "MACD": d.get("dj_signal", "—"),
                "押し目": wave_sig,
                "利回": f"{d['div_yield']:.1f}%" if d.get("div_yield") else "",
                "優待": yutai_val,
                "3Y利回": f"{d['div_yield_3y']:.1f}%" if d.get("div_yield_3y") else "",
                "PER": f"{d['per']:.0f}" if d.get("per") else "",
                "3YPER": f"{d['per_3y']:.0f}" if d.get("per_3y") else "",
                "EPS成長": f"{d['eps_cagr']:+.0f}%" if d.get("eps_cagr") is not None else "",
                "ROE": f"{d['roe']:.0f}%" if d.get("roe") else "",
                "ACT": act_str,
                "TOPIX": f"{d['corr_topix']:.1f}" if d.get("corr_topix") is not None else "",
                "テーマ": theme_str,
            })

        df_wl = pd.DataFrame(rows)
        st.dataframe(df_wl, use_container_width=True, hide_index=True, height=min(len(rows)*38+40, 600),
                      column_config={
                          "期待値": st.column_config.ProgressColumn(
                              "期待値", min_value=-100, max_value=100, format="%d"
                          ),
                      })

        # ── 個別銘柄カード（コンパクト版） ──
        st.markdown("---")
        for c in sorted_codes:
            d = wl_data.get(c, {})
            acts = act_map.get(c, [])
            sig = d.get("signal", "—")
            sig_color = d.get("signal_color", "gray")
            sig_score = d.get("signal_score", 0)
            expect = _calc_expect(d)
            dc = d.get("daily_change_pct")
            dc_s = f"{dc:+.1f}%" if dc is not None else ""
            mt = d.get("monthly_trend", "—")
            wt = d.get("weekly_trend", "—")

            ma75s = d.get("ma75_signal", "")

            with st.expander(f"{mt}{wt} {'**'+ma75s+'** ' if ma75s else ''}**{c} {d.get('name','')}** {sig} 期待値:{expect} {dc_s}", expanded=False):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("株価", f"¥{d['last_price']:,.0f}" if d.get("last_price") else "—", delta=f"{dc:+.2f}%" if dc else None)
                c2.metric("PER→3Y", f"{d['per']:.0f}→{d['per_3y']:.0f}" if d.get("per") and d.get("per_3y") else f"{d['per']:.0f}" if d.get("per") else "—")
                c3.metric("ROE", f"{d['roe']:.1f}%" if d.get("roe") else "—")
                c4.metric("利回り", f"{d['div_yield']:.2f}%" if d.get("div_yield") else "—")
                c5.metric("EPS成長", f"{d['eps_cagr']:+.1f}%" if d.get("eps_cagr") is not None else "—")
                c6.metric("MACD", d.get("dj_signal", "—"))

                # 押し目エントリーシグナル
                wave = d.get("wave", {})
                if wave.get("hits", 0) > 0 or wave.get("rsi14") is not None:
                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric("押し目", f"{wave.get('emoji','')} {wave.get('signal','—')}")
                    w2.metric("RSI(14)", f"{wave['rsi14']:.0f}" if wave.get("rsi14") else "—")
                    w3.metric("MA20乖離", f"{wave['dev_ma20']:+.1f}%" if wave.get("dev_ma20") is not None else "—")
                    w4.metric("高値比", f"{wave['drawdown']:.1f}%" if wave.get("drawdown") is not None else "—")
                    # N/V/E/NT値（環境認識）
                    if wave.get("N") is not None:
                        wt = wave.get("wave_type", "—")
                        nv1, nv2, nv3, nv4, nv5 = st.columns(5)
                        nv1.metric("波動", f"{'📈' if wt == '上昇' else '📉'}{wt}")
                        nv2.metric("NT値", f"¥{wave['NT']:,.0f}" if wave.get("NT") else "—")
                        nv3.metric("N値", f"¥{wave['N']:,.0f}" if wave.get("N") else "—")
                        nv4.metric("V値", f"¥{wave['V']:,.0f}" if wave.get("V") else "—")
                        nv5.metric("E値", f"¥{wave['E']:,.0f}" if wave.get("E") else "—")
                    if wave.get("detail"):
                        st.caption(f"📐 押し目判定 ({wave.get('hits',0)}/5): {wave['detail']}")

                # 補足情報を1行で
                info_parts = []
                if d.get("breakout") and d["breakout"] != "—": info_parts.append(f"ブレイク:{d['breakout']}")
                dev = d.get("deviation")
                if dev is not None: info_parts.append(f"乖離:{dev:+.1f}%")
                if d.get("yutai_yield"): info_parts.append(f"[🎁優待{d['yutai_yield']:.1f}%](https://shikiho.toyokeizai.net/stocks/{c}/yutai)")
                if acts: info_parts.append(f"🦈{','.join(set(acts))}")
                themes = d.get("themes", [])
                if themes: info_parts.append(f"テーマ:{','.join(themes[:2])}")
                if info_parts:
                    st.markdown(" / ".join(info_parts))

                # ミニチャート（小さく）
                hist = d.get("hist")
                if hist is not None and len(hist) > 5:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines",
                                             line=dict(color=sig_color, width=1.2), showlegend=False))
                    fig.update_layout(height=120, margin=dict(l=20,r=5,t=5,b=15),
                                      xaxis=dict(showgrid=False,showticklabels=False),
                                      yaxis=dict(showgrid=True,gridcolor="#eee",showticklabels=True,tickfont=dict(size=8)),
                                      plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)

    # ── EDINET API 連携 ──
    st.markdown("---")
    st.markdown("### 📄 EDINET 連携")

    # APIキー読み込み（金融庁EDINET API）
    _edinet_key = os.environ.get("EDINET_API_KEY", "")

    ed_tab1, ed_tab2 = st.tabs(["📊 EDINET DB 財務分析", "📄 EDINET 開示書類"])

    # ── EDINET DB（REST API直接呼び出し） ──
    with ed_tab1:
        if wl:
            edb_action = st.selectbox("分析内容", [
                "企業情報・財務健全性", "財務時系列（最大6年）", "決算短信（速報）",
            ], key="edb_action")
            edb_code = st.selectbox("銘柄", list(wl.keys()),
                format_func=lambda c: f"{c} {wl[c].get('name','')}", key="edb_code")

            def _resolve_edinet_code(sec_code):
                """証券コード→EDINETコード変換（キャッシュから）"""
                return None

            # 一括表示（キャッシュ済みデータ）
            import hashlib as _hl
            _cache_dir = os.path.join(tempfile.gettempdir(), "edinetdb_cache")
            def _tab_edb_load(code):
                today = datetime.now().strftime("%Y-%m-%d")
                ck = _hl.md5(f"{today}:get_company:{code}".encode()).hexdigest()
                p = os.path.join(_cache_dir, f"{ck}.json")
                try:
                    if os.path.exists(p):
                        with open(p, "r", encoding="utf-8") as f:
                            return json.load(f)
                except Exception:
                    pass
                return None

            codes_list = list(wl.keys())
            cached_count = sum(1 for c in codes_list if _tab_edb_load(c) is not None)
            uncached = len(codes_list) - cached_count
            st.caption(f"📊 {len(codes_list)}銘柄中 {cached_count}件キャッシュ済み / 未取得: {uncached}件")

            if cached_count > 0:
                batch_results = []
                for c in codes_list:
                    cached = _tab_edb_load(c)
                    if cached:
                        batch_results.append({"code": c, "name": cached.get("name_ja", cached.get("name", c)), **cached})
                if batch_results:
                    st.dataframe(pd.DataFrame(batch_results)[["code","name"]].rename(columns={"code":"コード","name":"企業名"}),
                                 use_container_width=True, hide_index=True)

            # 個別詳細表示
            edb_code = st.selectbox("詳細表示", codes_list,
                format_func=lambda c: f"{c} {wl[c].get('name','')}", key="edb_code_detail")
            if st.button("📋 詳細JSON表示", key="edb_detail"):
                cached = _tab_edb_load(edb_code)
                if cached:
                    st.json(cached)
                else:
                    st.info(f"{edb_code} のキャッシュがありません。")

            # キャッシュクリア
            st.markdown("---")
            if st.button("🔄 全銘柄キャッシュクリア", key="edb_clear"):
                import glob
                cache_dir = os.path.join(tempfile.gettempdir(), "edinetdb_cache")
                for f in glob.glob(os.path.join(cache_dir, "*.json")):
                    os.remove(f)
                st.success("キャッシュをクリアしました。ページをリロードすると再取得されます。")
        else:
            st.info("ウォッチリストに銘柄を追加してください。")

    # ── 金融庁 EDINET API（開示書類検索） ──
    with ed_tab2:
        if not _edinet_key:
            st.warning("金融庁EDINET APIキーが未設定です。`.env` に `EDINET_API_KEY=your_key` を設定してください。")
        else:
            st.caption("✅ 金融庁EDINET APIキー設定済み")

            @st.cache_data(ttl=3600, show_spinner=False)
            def _edinet_search(date_str, doc_type="2"):
                url = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
                params = {"date": date_str, "type": doc_type, "Subscription-Key": _edinet_key}
                try:
                    resp = requests.get(url, params=params, timeout=30)
                    if resp.status_code == 200:
                        return resp.json().get("results", [])
                    return None
                except Exception:
                    return None

            @st.cache_data(ttl=3600, show_spinner=False)
            def _edinet_fetch_doc_list(date_str, target_codes):
                results = _edinet_search(date_str)
                if results is None:
                    return None
                matched = []
                for doc in results:
                    sec_code = str(doc.get("secCode", ""))[:4]
                    if sec_code in target_codes:
                        matched.append({
                            "日付": date_str, "コード": sec_code,
                            "提出者": doc.get("filerName", ""),
                            "書類種別": doc.get("docDescription", ""),
                            "docID": doc.get("docID", ""),
                        })
                return matched

            ed_col1, ed_col2 = st.columns([2, 1])
            with ed_col1:
                ed_days = st.slider("過去何日分を検索", 1, 30, 7, key="ed_days")
            with ed_col2:
                ed_search = st.button("🔍 EDINET検索", key="ed_search_btn", use_container_width=True)

            if ed_search and wl:
                target_codes = set(wl.keys())
                all_docs = []
                progress = st.progress(0, text="EDINET検索中...")
                for i in range(ed_days):
                    d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                    docs = _edinet_fetch_doc_list(d, target_codes)
                    if docs:
                        all_docs.extend(docs)
                    progress.progress((i + 1) / ed_days, text=f"EDINET検索中... {i+1}/{ed_days}日")
                progress.empty()

                if all_docs:
                    df_ed = pd.DataFrame(all_docs)
                    st.success(f"📄 {len(all_docs)}件の開示書類が見つかりました")
                    st.dataframe(df_ed, use_container_width=True, hide_index=True)
                    hoyu = df_ed[df_ed["書類種別"].str.contains("大量保有|変更報告", na=False)]
                    if len(hoyu) > 0:
                        st.markdown("#### 🦈 大量保有報告書")
                        st.dataframe(hoyu, use_container_width=True, hide_index=True)
                    kessan = df_ed[df_ed["書類種別"].str.contains("有価証券報告|四半期報告|決算短信|半期報告", na=False)]
                    if len(kessan) > 0:
                        st.markdown("#### 📊 決算関連書類")
                        st.dataframe(kessan, use_container_width=True, hide_index=True)
                else:
                    st.info(f"過去{ed_days}日間にウォッチリスト銘柄の開示書類はありませんでした。")


# =====================================================================
# Tab 14: 決算速報
# =====================================================================
with tab14:
    st.subheader("📰 決算速報 — 本日の決算短信チェック")
    st.caption("決算発表銘柄を取得し、EDINET DBで進捗率・ガイダンスを5段階評価。PTS反応も表示。")

    _edb_key_kessan = ""

    @st.cache_data(ttl=900, show_spinner=False)
    def _fetch_kessan_news(date_str):
        """株探の決算速報ページから当日の決算発表銘柄を取得"""
        import re as _re
        rows = []
        existing_codes = set()

        def _parse_kabutan_table(soup):
            """株探の決算テーブルをパース"""
            parsed = []
            trs = soup.select("table.stock_table tr")
            for tr in trs:
                tds = tr.find_all("td")
                ths = tr.find_all("th")
                if len(tds) < 3:
                    continue
                code = tds[0].get_text(strip=True)
                if not _re.match(r"^[0-9A-Z]{4}$", code):
                    continue
                if code in existing_codes:
                    continue
                # 銘柄名は<th>タグ
                name = ""
                for th in ths:
                    t = th.get_text(strip=True)
                    if t and len(t) >= 2 and not _re.match(r"^\d", t):
                        name = t
                        break
                if not name:
                    name = STOCK_DICT.get(f"{code}.T", code)
                # 決算ポイント（ヘッドライン）
                headline = tds[3].get_text(strip=True) if len(tds) > 3 else ""
                # 株価・前日比
                price_str = tds[4].get_text(strip=True) if len(tds) > 4 else ""
                change_str = tds[7].get_text(strip=True) if len(tds) > 7 else ""
                existing_codes.add(code)
                parsed.append({
                    "code": code, "name": name, "headline": headline,
                    "price": price_str, "change": change_str,
                    "url": f"https://kabutan.jp/stock/?code={code}",
                })
            return parsed

        # ソース1: 株探 前日引け後の決算発表（mode=4_3）
        try:
            url1 = "https://kabutan.jp/warning/?mode=4_3"
            resp1 = requests.get(url1, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp1.encoding = resp1.apparent_encoding
            if resp1.status_code == 200:
                rows.extend(_parse_kabutan_table(BeautifulSoup(resp1.text, "html.parser")))
        except Exception:
            pass

        # ソース2: 株探 当日場中の決算発表（mode=4_2）
        try:
            url2 = "https://kabutan.jp/warning/?mode=4_2"
            resp2 = requests.get(url2, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp2.encoding = resp2.apparent_encoding
            if resp2.status_code == 200:
                rows.extend(_parse_kabutan_table(BeautifulSoup(resp2.text, "html.parser")))
        except Exception:
            pass

        # ソース3: 日経適時開示（補完）
        try:
            url3 = "https://www.nikkei.com/markets/kigyo/disclose/?kwd=" + quote("決算短信") + "&SelDateDiff=0&hm=1"
            resp3 = requests.get(url3, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp3.encoding = resp3.apparent_encoding
            if resp3.status_code == 200:
                soup3 = BeautifulSoup(resp3.text, "html.parser")
                for tr in soup3.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 4:
                        continue
                    texts = [td.get_text(strip=True) for td in tds]
                    if not any("決算短信" in t for t in texts):
                        continue
                    name = ""
                    for td in tds:
                        a_tag = td.find("a")
                        if a_tag:
                            n = a_tag.get_text(strip=True)
                            if n and len(n) >= 2 and not _re.match(r"^\d", n):
                                name = n
                                break
                    if not name:
                        continue
                    code = ""
                    for ticker, sname in STOCK_DICT.items():
                        if name in sname:
                            code = ticker.replace(".T", "")
                            break
                    if code and code not in existing_codes:
                        existing_codes.add(code)
                        rows.append({
                            "code": code, "name": name, "headline": "",
                            "price": "", "change": "",
                            "url": f"https://kabutan.jp/stock/?code={code}",
                        })
        except Exception:
            pass
        return rows

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_earnings_detail_v2(code, api_key):
        """EDINET DB REST APIから決算短信データを取得（APIキー不要、キャッシュのみ）"""
        return None

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_pts_price(code):
        """株探から現在値（場中）＋PTS価格＋前日終値を取得"""
        try:
            import re as _re
            from datetime import datetime as _dt
            url = f"https://kabutan.jp/stock/?code={code}"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code != 200:
                return {"close": None, "current": None, "current_change": None,
                        "pts": None, "pts_change": None, "is_market_open": False}
            text = resp.text
            soup = BeautifulSoup(resp.text, "html.parser")
            result = {"close": None, "current": None, "current_change": None,
                      "pts": None, "pts_change": None, "is_market_open": False}

            # 現在値（場中 or 終値）: 最初の「N,NNN.N円」＋前日比
            m = _re.search(r'(\d[\d,]*\.?\d*)\s*円.*?前日比', text, _re.DOTALL)
            if m:
                result["current"] = float(m.group(1).replace(",", ""))

            # 前日比（%）
            m2 = _re.search(r'前日比.*?([+\-]?\d[\d,]*\.?\d*)\s*\(([+\-]?\d+\.?\d*)%\)', text, _re.DOTALL)
            if m2:
                result["current_change"] = float(m2.group(2))
            else:
                # 別パターン: 前日比の後に%
                m2b = _re.search(r'前日比.*?([+\-]?\d+\.?\d*)\s*%', text, _re.DOTALL)
                if m2b:
                    result["current_change"] = float(m2b.group(1))

            # 場中判定（9:00-15:30）
            now = _dt.now()
            hour_min = now.hour * 100 + now.minute
            is_weekday = now.weekday() < 5
            result["is_market_open"] = is_weekday and 900 <= hour_min <= 1530

            # 前日終値（場中なら現在値と異なる）
            # 株探では場中は「現在値」が表示され、終値は別セクション
            close_match = _re.search(r'終値[^\d]*?(\d[\d,]*\.?\d*)', text)
            if close_match:
                result["close"] = float(close_match.group(1).replace(",", ""))
            if not result["close"]:
                # 場が閉まっている場合、現在値＝終値
                result["close"] = result["current"]

            # PTS価格（「PTS ... N,NNN.N円」形式）
            m3 = _re.search(r'PTS.*?(\d[\d,]*\.?\d*)\s*円', text, _re.DOTALL)
            if m3:
                candidate = float(m3.group(1).replace(",", ""))
                if candidate >= 10:
                    result["pts"] = candidate
                    # PTS騰落率（前日終値比）
                    base = result["close"] or result["current"]
                    if base and base > 0:
                        pct = round((candidate - base) / base * 100, 2)
                        if abs(pct) <= 30:
                            result["pts_change"] = pct
                        else:
                            result["pts"] = None  # 異常値

            return result
        except Exception:
            return {"close": None, "current": None, "current_change": None,
                    "pts": None, "pts_change": None, "is_market_open": False}

    @st.cache_data(ttl=3600, show_spinner=False)
    def _parse_kessan_pdf_from_tdnet(code):
        """株探の決算詳細ページから業績データを取得（TDNet PDF → 株探に変更）"""
        import re as _re
        result = {"revenue_yoy": None, "oi_yoy": None, "ni_yoy": None,
                  "forecast_change": None, "dividend_info": None, "buyback_info": None,
                  "summary": "", "pdf_url": None, "tdnet_url": None,
                  "finance_url": None, "per": None, "pbr": None,
                  "latest_revenue": None, "latest_oi": None, "latest_ni": None,
                  "forecast_revenue": None, "forecast_oi": None, "forecast_ni": None,
                  "div_yield": None, "div_current": None, "div_prev": None,
                  "div_forecast": None, "div_change": None,
                  "guidance_label": None, "guidance_detail": None,
                  "fc_revenue_chg": None, "fc_oi_chg": None, "fc_ni_chg": None}
        try:
            finance_url = f"https://kabutan.jp/stock/finance?code={code}"
            result["finance_url"] = finance_url
            resp = requests.get(finance_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code != 200:
                return result
            soup = BeautifulSoup(resp.text, "html.parser")
            tables = soup.find_all("table")

            # --- Table: 年次業績（決算期 | 売上高 | 営業益 | 経常益 | 最終益 | 修正1株益 | 修正1株配 | 発表日）---
            annual_rows = []
            annual_hdrs = []
            for t in tables:
                rows = t.find_all("tr")
                if not rows:
                    continue
                hdrs = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
                if "売上高" in hdrs and "営業益" in hdrs and "発表日" in hdrs:
                    annual_hdrs = hdrs
                    for r in rows[1:]:
                        cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
                        if len(cells) >= 6 and _re.search(r"\d{4}\.\d{2}", cells[0]):
                            annual_rows.append(cells)
                    break

            # ヘッダーインデックス
            _idx_rev = annual_hdrs.index("売上高") if "売上高" in annual_hdrs else 1
            _idx_oi = annual_hdrs.index("営業益") if "営業益" in annual_hdrs else 2
            _idx_ni = annual_hdrs.index("最終益") if "最終益" in annual_hdrs else 4
            _idx_eps = annual_hdrs.index("修正1株益") if "修正1株益" in annual_hdrs else 5
            _idx_dps = annual_hdrs.index("修正1株配") if "修正1株配" in annual_hdrs else 6

            # 直近2期から前年比を計算（「予」=予想行は除外して実績のみ比較）
            def _parse_num(s):
                s = s.replace(",", "").replace("－", "").replace("―", "").strip()
                if not s or s == "-":
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            actual_rows = [r for r in annual_rows if not _re.search(r"予\d{4}", r[0].replace("\xa0", ""))]
            forecast_rows = [r for r in annual_rows if _re.search(r"予\d{4}", r[0].replace("\xa0", ""))]

            if len(actual_rows) >= 2:
                latest = actual_rows[-1]
                prev = actual_rows[-2]
                for idx, key in [(_idx_rev, "revenue"), (_idx_oi, "oi"), (_idx_ni, "ni")]:
                    cur_v = _parse_num(latest[idx]) if len(latest) > idx else None
                    prev_v = _parse_num(prev[idx]) if len(prev) > idx else None
                    if cur_v is not None:
                        result[f"latest_{key}"] = cur_v
                    if cur_v is not None and prev_v and prev_v != 0:
                        yoy = round((cur_v - prev_v) / abs(prev_v) * 100, 1)
                        result[f"{key}_yoy"] = str(yoy)
                # 配当（直近実績 vs 前期）
                cur_dps = _parse_num(latest[_idx_dps]) if len(latest) > _idx_dps else None
                prev_dps = _parse_num(prev[_idx_dps]) if len(prev) > _idx_dps else None
                result["div_current"] = cur_dps
                result["div_prev"] = prev_dps
                if cur_dps is not None and prev_dps is not None and prev_dps > 0:
                    chg = round((cur_dps - prev_dps) / prev_dps * 100, 1)
                    if chg > 1:
                        result["div_change"] = f"増配({chg:+.0f}%)"
                    elif chg < -1:
                        result["div_change"] = f"減配({chg:.0f}%)"
                    else:
                        result["div_change"] = "据置"
                elif cur_dps is not None and (prev_dps is None or prev_dps == 0):
                    if cur_dps > 0:
                        result["div_change"] = "復配"

            # --- 予想行（会社予想 vs 直近実績で成長率＋配当予想）---
            if forecast_rows and actual_rows:
                fc = forecast_rows[-1]
                ac = actual_rows[-1]
                for idx, key in [(_idx_rev, "revenue"), (_idx_oi, "oi"), (_idx_ni, "ni")]:
                    fc_v = _parse_num(fc[idx]) if len(fc) > idx else None
                    ac_v = _parse_num(ac[idx]) if len(ac) > idx else None
                    if fc_v is not None:
                        result[f"forecast_{key}"] = fc_v
                    if fc_v is not None and ac_v and ac_v != 0:
                        chg = round((fc_v - ac_v) / abs(ac_v) * 100, 1)
                        if key == "revenue":
                            result["fc_revenue_chg"] = chg
                        elif key == "oi":
                            result["fc_oi_chg"] = chg
                        elif key == "ni":
                            result["fc_ni_chg"] = chg
                        if result.get(f"{key}_yoy") is None:
                            result[f"{key}_yoy"] = str(chg)
                # 来期配当予想
                fc_dps = _parse_num(fc[_idx_dps]) if len(fc) > _idx_dps else None
                result["div_forecast"] = fc_dps
                cur_dps = result.get("div_current")
                if fc_dps is not None and cur_dps is not None and cur_dps > 0:
                    fc_div_chg = round((fc_dps - cur_dps) / cur_dps * 100, 1)
                    if fc_div_chg > 1:
                        result["div_change"] = (result.get("div_change") or "") + f" → 来期増配({fc_div_chg:+.0f}%)"
                    elif fc_div_chg < -1:
                        result["div_change"] = (result.get("div_change") or "") + f" → 来期減配({fc_div_chg:.0f}%)"

                # --- ガイダンス判定（来期予想の強弱）---
                fc_rev = result.get("fc_revenue_chg")
                fc_oi = result.get("fc_oi_chg")
                fc_ni = result.get("fc_ni_chg")
                vals = [v for v in [fc_rev, fc_oi, fc_ni] if v is not None]
                if vals:
                    avg = sum(vals) / len(vals)
                    parts_g = []
                    if fc_rev is not None:
                        parts_g.append(f"売上{fc_rev:+.0f}%")
                    if fc_oi is not None:
                        parts_g.append(f"営利{fc_oi:+.0f}%")
                    if fc_ni is not None:
                        parts_g.append(f"純利{fc_ni:+.0f}%")
                    result["guidance_detail"] = " / ".join(parts_g)
                    if avg > 20:
                        result["guidance_label"] = "🟢強気"
                    elif avg > 5:
                        result["guidance_label"] = "🔵堅調"
                    elif avg > -5:
                        result["guidance_label"] = "🟡横ばい"
                    elif avg > -15:
                        result["guidance_label"] = "🟠慎重"
                    else:
                        result["guidance_label"] = "🔴弱気"

            # --- 四半期進捗率テーブル（対通期進捗率）---
            for t in tables:
                rows = t.find_all("tr")
                if not rows:
                    continue
                hdrs = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
                if "対通期進捗率" in hdrs:
                    # 前年同期比の行
                    for r in rows:
                        cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
                        if cells and "前年同期比" in cells[0]:
                            # 前年同期比 | 売上 | 営利 | 経常 | 純利 | EPS | ... | (%)
                            for ci, key in [(1, "revenue"), (2, "oi"), (4, "ni")]:
                                if len(cells) > ci:
                                    v = _parse_num(cells[ci])
                                    if v is not None and result.get(f"{key}_yoy") is None:
                                        result[f"{key}_yoy"] = str(v)
                    break

            # --- PER/PBR/利回り ---
            for t in tables:
                rows = t.find_all("tr")
                if not rows:
                    continue
                hdrs = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
                if "PER" in hdrs and "PBR" in hdrs:
                    if len(rows) > 1:
                        vals = [c.get_text(strip=True) for c in rows[1].find_all(["th", "td"])]
                        for i, h in enumerate(hdrs):
                            if h == "PER" and i < len(vals):
                                result["per"] = vals[i].replace("倍", "").strip()
                            if h == "PBR" and i < len(vals):
                                result["pbr"] = vals[i].replace("倍", "").strip()
                            if h == "利回り" and i < len(vals):
                                y = vals[i].replace("％", "").replace("%", "").strip()
                                if y and y != "－":
                                    result["div_yield"] = y
                    break

            # --- 業績修正テーブル ---
            for t in tables:
                rows = t.find_all("tr")
                if not rows:
                    continue
                hdrs = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
                if "修正方向" in hdrs:
                    # 最新の修正行を取得
                    for r in reversed(rows[1:]):
                        cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
                        cell_text = " ".join(cells)
                        if "↑" in cell_text or "上方" in cell_text:
                            result["forecast_change"] = "上方修正"
                            break
                        elif "↓" in cell_text or "下方" in cell_text:
                            result["forecast_change"] = "下方修正"
                            break
                    break

            # --- TDNet適時開示リンク（日付ページからPDFリンク取得）---
            try:
                from datetime import datetime as _dt
                today_str = _dt.now().strftime("%Y%m%d")
                tdnet_date_url = f"https://www.release.tdnet.info/inbs/I_list_001_{today_str}.html"
                tr = requests.get(tdnet_date_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                if tr.status_code == 200:
                    result["tdnet_url"] = tdnet_date_url
                    ts = BeautifulSoup(tr.text, "html.parser")
                    for a in ts.find_all("a", href=True):
                        href = a.get("href", "")
                        text = a.get_text(strip=True)
                        if href.endswith(".pdf") and code in href:
                            result["pdf_url"] = f"https://www.release.tdnet.info/inbs/{href}" if not href.startswith("http") else href
                            break
                    # コードで見つからなければテキストマッチ
                    if not result.get("pdf_url"):
                        stock_name = STOCK_DICT.get(f"{code}.T", "")
                        for a in ts.find_all("a", href=True):
                            href = a.get("href", "")
                            text = a.get_text(strip=True)
                            if href.endswith(".pdf") and ("決算短信" in text or "業績予想" in text):
                                if stock_name and stock_name[:3] in text:
                                    result["pdf_url"] = f"https://www.release.tdnet.info/inbs/{href}" if not href.startswith("http") else href
                                    break
            except Exception:
                pass

            # --- サマリー生成 ---
            parts = []
            for key, label in [("revenue_yoy", "売上"), ("oi_yoy", "営利"), ("ni_yoy", "純利")]:
                v = result.get(key)
                if v:
                    try:
                        fv = float(v)
                        sign = "+" if fv > 0 else ""
                        emoji = "🔥" if fv > 30 else "⚠️" if fv < -10 else ""
                        parts.append(f"{label}{sign}{fv:.1f}%{emoji}")
                    except Exception:
                        pass
            if result.get("forecast_change"):
                emoji = "📈" if result["forecast_change"] == "上方修正" else "📉"
                parts.append(f"{emoji}{result['forecast_change']}")
            if result.get("div_change"):
                parts.append(f"💰{result['div_change']}")
            if result.get("guidance_label"):
                parts.append(f"来期{result['guidance_label']}")
            result["summary"] = " / ".join(parts) if parts else ""
            return result
        except Exception:
            return result

    def _evaluate_earnings_v2(e, pdf_data=None, pts_change=None, close_price=None):
        """決算データ（EDINET DB + 株探）＋PTS反応から総合5段階評価"""
        score = 3
        comments = []
        has_data = False

        # 株探データ（pdf_data）から業績前年比を評価
        if pdf_data and isinstance(pdf_data, dict):
            for key, label, weight in [
                ("revenue_yoy", "売上", 0.6), ("oi_yoy", "営利", 1.0), ("ni_yoy", "純利", 0.8)
            ]:
                v = pdf_data.get(key)
                if v is not None:
                    try:
                        fv = float(v)
                        has_data = True
                        if fv > 30: score += weight; comments.append(f"{label}+{fv:.0f}%🔥")
                        elif fv > 10: score += weight * 0.5; comments.append(f"{label}+{fv:.0f}%")
                        elif fv > 0: comments.append(f"{label}+{fv:.0f}%")
                        elif fv > -10: comments.append(f"{label}{fv:.0f}%")
                        else: score -= weight; comments.append(f"{label}{fv:.0f}%⚠️")
                    except Exception:
                        pass
            if pdf_data.get("forecast_change") == "上方修正":
                score += 0.5; comments.append("上方修正📈"); has_data = True
            elif pdf_data.get("forecast_change") == "下方修正":
                score -= 0.5; comments.append("下方修正📉"); has_data = True
            # 配当変動
            dc = pdf_data.get("div_change")
            if dc:
                if "増配" in dc:
                    score += 0.3; comments.append(f"💰{dc}")
                elif "減配" in dc:
                    score -= 0.3; comments.append(f"💰{dc}")
                elif "復配" in dc:
                    score += 0.4; comments.append(f"💰{dc}")
            # 配当利回り
            dy = pdf_data.get("div_yield")
            if dy:
                comments.append(f"利回り{dy}%")
            # ガイダンス（来期見通し）
            gl = pdf_data.get("guidance_label")
            if gl:
                gd = pdf_data.get("guidance_detail", "")
                comments.append(f"来期{gl}({gd})" if gd else f"来期{gl}")
                # ガイダンスでスコア微調整
                if "強気" in gl: score += 0.3
                elif "弱気" in gl: score -= 0.3
            # PER情報
            per_str = pdf_data.get("per", "")
            if per_str and per_str not in ("－", ""):
                try:
                    per_v = float(per_str)
                    if per_v < 10: score += 0.3; comments.append(f"PER{per_v:.1f}倍")
                    elif per_v > 40: score -= 0.2; comments.append(f"PER{per_v:.0f}倍")
                except Exception:
                    pass

        # EDINET DBデータ（補完）
        if e and isinstance(e, dict) and not has_data:
            for key_prefix, label, weight in [
                ("revenue", "売上", 0.6), ("operating_income", "営利", 1.0), ("net_income", "純利", 0.8)
            ]:
                yoy = None
                for suffix in ["_yoy", "_change", "_yoy_pct", "_growth"]:
                    v = e.get(f"{key_prefix}{suffix}")
                    if v is not None:
                        yoy = float(v)
                        if abs(yoy) < 5 and abs(yoy) > 0:
                            yoy = yoy * 100
                        break
                if yoy is not None:
                    has_data = True
                    if yoy > 30: score += weight; comments.append(f"{label}+{yoy:.0f}%🔥")
                    elif yoy > 10: score += weight * 0.5; comments.append(f"{label}+{yoy:.0f}%")
                    elif yoy > 0: comments.append(f"{label}+{yoy:.0f}%")
                    elif yoy > -10: comments.append(f"{label}{yoy:.0f}%")
                    else: score -= weight; comments.append(f"{label}{yoy:.0f}%⚠️")
            for k in ["forecast_revision", "guidance_change", "forecast_net_income_change"]:
                v = e.get(k)
                if v is not None:
                    fv = float(v)
                    if fv > 5: score += 0.5; comments.append("上方修正📈")
                    elif fv < -5: score -= 0.5; comments.append("下方修正📉")
                    break

        # PTS反応（市場の評価） — 異常値はスキップ
        if pts_change is not None and abs(pts_change) <= 30:
            if pts_change > 5: score += 0.5; comments.append(f"PTS+{pts_change:.1f}%🚀")
            elif pts_change > 2: score += 0.3; comments.append(f"PTS+{pts_change:.1f}%↑")
            elif pts_change < -5: score -= 0.5; comments.append(f"PTS{pts_change:.1f}%💥")
            elif pts_change < -2: score -= 0.3; comments.append(f"PTS{pts_change:.1f}%↓")

        score = max(1, min(5, round(score)))
        stars = "⭐" * score + "☆" * (5 - score)
        comment = " / ".join(comments) if comments else "データ取得中..."
        return score, stars, comment

    # UI
    kessan_date = st.date_input("決算日", value=datetime.now().date(), key="kessan_date")
    date_str = kessan_date.strftime("%Y%m%d")

    # --- 決算キャッシュ（JSONファイル + session_state、1日有効）---
    _KESSAN_CACHE_FILE = "kessan_cache.json"

    def _load_kessan_cache(target_date):
        """JSONファイルからキャッシュを読み込み"""
        try:
            with open(_KESSAN_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("date") == target_date:
                return cache.get("evaluated", [])
        except Exception:
            pass
        return []

    def _save_kessan_cache(target_date, evaluated):
        """キャッシュをJSONファイルに保存"""
        try:
            # price_dataの非シリアライズ可能な値を除去
            safe = []
            for e in evaluated:
                entry = {k: v for k, v in e.items() if k != "pdf_data"}
                # pdf_dataからシリアライズ可能なキーのみ保持
                pdf = e.get("pdf_data")
                if pdf and isinstance(pdf, dict):
                    entry["pdf_data"] = {k: v for k, v in pdf.items()
                                         if isinstance(v, (str, int, float, bool, type(None)))}
                else:
                    entry["pdf_data"] = None
                safe.append(entry)
            with open(_KESSAN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"date": target_date, "evaluated": safe}, f, ensure_ascii=False)
        except Exception:
            pass

    # session_stateにもキャッシュを保持
    if "kessan_evaluated" not in st.session_state:
        st.session_state["kessan_evaluated"] = []
        st.session_state["kessan_date_cached"] = ""

    # 日付が変わったらリセット
    if st.session_state.get("kessan_date_cached") != date_str:
        cached = _load_kessan_cache(date_str)
        st.session_state["kessan_evaluated"] = cached
        st.session_state["kessan_date_cached"] = date_str

    kc1, kc2, kc3 = st.columns([1, 1, 1])
    with kc1:
        fetch_btn = st.button("📰 決算発表銘柄を取得", key="kessan_fetch", use_container_width=True)
    with kc2:
        use_edb = st.checkbox("EDINET DBで詳細評価", value=False, key="kessan_edb_v2")
    with kc3:
        clear_cache = st.button("🗑️ キャッシュクリア", key="kessan_clear")

    if clear_cache:
        st.session_state["kessan_evaluated"] = []
        st.session_state["kessan_date_cached"] = ""
        try:
            os.remove(_KESSAN_CACHE_FILE)
        except Exception:
            pass
        st.success("キャッシュをクリアしました")
        st.rerun()

    if fetch_btn:
        with st.spinner("決算発表銘柄を取得中..."):
            kessan_list = _fetch_kessan_news(date_str)

        if not kessan_list:
            st.info(f"{kessan_date.strftime('%Y/%m/%d')} の決算発表銘柄が見つかりませんでした。株探の決算速報ページを確認してください。")
        else:
            # コードがない銘柄はSTOCK_DICTから逆引き
            for k in kessan_list:
                if "code" not in k or not k.get("code"):
                    name = k.get("name", "")
                    for ticker, sname in STOCK_DICT.items():
                        if name and name in sname:
                            k["code"] = ticker.replace(".T", "")
                            break

            # キャッシュ済み銘柄を特定し、新規のみ取得
            cached_eval = st.session_state.get("kessan_evaluated", [])
            cached_codes = {e.get("code") for e in cached_eval if e.get("code")}
            new_items = [k for k in kessan_list if k.get("code") and k["code"] not in cached_codes]
            new_no_code = [k for k in kessan_list if not k.get("code")]

            if cached_eval:
                st.info(f"💾 キャッシュ: {len(cached_codes)}社取得済み / 新規: {len(new_items)}社を追加取得")
            else:
                st.success(f"📰 {len(kessan_list)}社の決算発表を検出")

            # EDINET DB評価 + 株探解析 + PTS取得（並列処理で高速化）
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _fetch_one_kessan(k, use_edb_flag, edb_key):
                """1銘柄分の決算データを並列取得（銘柄内も3リクエスト並列）"""
                code = k.get("code")
                if not code:
                    return {**k, "score": 3, "stars": "⭐⭐⭐", "comment": k.get("headline", ""),
                            "price_data": {}, "pdf_data": None}
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with _TPE(max_workers=3) as inner:
                    f_edb = inner.submit(_fetch_earnings_detail_v2, code, edb_key) if (use_edb_flag and edb_key) else None
                    f_pdf = inner.submit(_parse_kessan_pdf_from_tdnet, code)
                    f_pts = inner.submit(_fetch_pts_price, code)
                    e = f_edb.result() if f_edb else None
                    pdf_data = f_pdf.result()
                    price_data = f_pts.result() or {}
                pts_change = price_data.get("pts_change")
                close_price = price_data.get("close") or price_data.get("current")
                sc, stars, comment = _evaluate_earnings_v2(e, pdf_data, pts_change, close_price)
                return {**k, "score": sc, "stars": stars, "comment": comment,
                        "price_data": price_data, "pdf_data": pdf_data}

            newly_evaluated = []
            if new_items:
                progress = st.progress(0, text=f"新規{len(new_items)}社を評価中（並列処理）...")
                done_count = 0
                total_count = len(new_items)

                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {
                        executor.submit(_fetch_one_kessan, k, use_edb, _edb_key_kessan): k
                        for k in new_items[:30]
                    }
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            result["_is_new"] = True
                            newly_evaluated.append(result)
                        except Exception:
                            k = futures[future]
                            newly_evaluated.append({**k, "score": 3, "stars": "⭐⭐⭐",
                                                    "comment": "取得エラー", "_is_new": True,
                                                    "price_data": {}, "pdf_data": None})
                        done_count += 1
                        progress.progress(done_count / max(total_count, 1))
                progress.empty()

            # コードなし銘柄
            for k in new_no_code:
                if not any(e.get("name") == k.get("name") for e in cached_eval):
                    newly_evaluated.append({**k, "score": 3, "stars": "⭐⭐⭐",
                                            "comment": k.get("headline", ""), "_is_new": True,
                                            "price_data": {}, "pdf_data": None})

            # キャッシュとマージ
            evaluated = list(cached_eval) + newly_evaluated
            evaluated.sort(key=lambda x: x.get("score", 3), reverse=True)

            # キャッシュ更新
            st.session_state["kessan_evaluated"] = evaluated
            st.session_state["kessan_date_cached"] = date_str
            _save_kessan_cache(date_str, evaluated)

    # --- キャッシュからの表示（ボタンを押さなくてもキャッシュがあれば表示）---
    evaluated = st.session_state.get("kessan_evaluated", [])
    if evaluated:
        st.caption(f"💾 {len(evaluated)}社の決算データを表示中（キャッシュ済み）")

        # 好決算ハイライト
        good = [e for e in evaluated if e.get("score", 3) >= 4]
        if good:
            st.markdown("### 🏆 好決算企業")
            for e in good:
                color = "#00c853" if e["score"] == 5 else "#2196f3"
                url = e.get("url", f"https://kabutan.jp/stock/?code={e.get('code','')}")
                pd_ = e.get("price_data") or {}
                price_html = ""
                if pd_.get("is_market_open") and pd_.get("current"):
                    cc = pd_.get("current_change")
                    if cc is not None:
                        pc_color = "#00c853" if cc > 0 else "#f44336" if cc < 0 else "#666"
                        arrow = "🚀" if cc > 5 else "↑" if cc > 0 else "💥" if cc < -5 else "↓" if cc < 0 else ""
                        price_html = f'<span style="color:{pc_color};font-weight:bold;margin-left:10px;">現在値 ¥{pd_["current"]:,.1f} ({cc:+.1f}%){arrow}</span>'
                    else:
                        price_html = f'<span style="margin-left:10px;">現在値 ¥{pd_["current"]:,.1f}</span>'
                if pd_.get("pts"):
                    pc = pd_.get("pts_change")
                    if pc is not None:
                        pts_color = "#00c853" if pc > 0 else "#f44336" if pc < 0 else "#666"
                        arrow = "🚀" if pc > 5 else "↑" if pc > 0 else "💥" if pc < -5 else "↓" if pc < 0 else ""
                        price_html += f'<span style="color:{pts_color};font-weight:bold;margin-left:10px;">PTS ¥{pd_["pts"]:,.1f} ({pc:+.1f}%){arrow}</span>'
                    else:
                        price_html += f'<span style="margin-left:10px;">PTS ¥{pd_["pts"]:,.1f}</span>'
                pdf = e.get("pdf_data")
                code_val = e.get("code", "")
                new_badge = '<span style="background:#ff6d00;color:#fff;font-size:0.65rem;padding:1px 5px;border-radius:4px;margin-right:4px;vertical-align:middle;">NEW</span>' if e.get("_is_new") else ""
                link_parts = []
                if pdf and pdf.get("pdf_url"):
                    link_parts.append(f'<a href="{pdf["pdf_url"]}" target="_blank" style="font-size:0.8rem;margin-left:8px;">📄決算PDF</a>')
                if pdf and pdf.get("tdnet_url"):
                    link_parts.append(f'<a href="{pdf["tdnet_url"]}" target="_blank" style="font-size:0.8rem;margin-left:4px;">📋TDNet</a>')
                if pdf and pdf.get("finance_url"):
                    link_parts.append(f'<a href="{pdf["finance_url"]}" target="_blank" style="font-size:0.8rem;margin-left:4px;">📊決算詳細</a>')
                elif code_val:
                    link_parts.append(f'<a href="https://kabutan.jp/stock/finance?code={code_val}" target="_blank" style="font-size:0.8rem;margin-left:4px;">📊決算詳細</a>')
                pdf_links_html = " ".join(link_parts)
                st.markdown(
                    f'<div style="padding:10px 14px;margin-bottom:6px;border-radius:8px;'
                    f'border-left:5px solid {color};background:rgba(0,200,83,0.08);">'
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
                    f'<span style="font-size:1rem;">{e["stars"]}</span>'
                    f'{new_badge}'
                    f'<a href="{url}" target="_blank" style="color:inherit;text-decoration:none;font-weight:bold;">'
                    f'{e.get("code","")} {e["name"]}</a>{price_html}{pdf_links_html}'
                    f'</div>'
                    f'<div style="margin-top:3px;font-size:0.85rem;opacity:0.8;">{e["comment"]}</div>'
                    f'</div>', unsafe_allow_html=True
                )
            st.markdown("---")

        # 全銘柄テーブル
        st.markdown("### 📋 全決算発表一覧")
        tbl_rows = []
        for e in evaluated:
            pd_ = e.get("price_data") or {}
            current_str = ""
            if pd_.get("current"):
                cc = pd_.get("current_change")
                if cc is not None:
                    arrow = "🚀" if cc > 5 else "↑" if cc > 0 else "💥" if cc < -5 else "↓" if cc < 0 else ""
                    current_str = f"¥{pd_['current']:,.1f}({cc:+.1f}%){arrow}"
                else:
                    current_str = f"¥{pd_['current']:,.1f}"
            pts_str = ""
            if pd_.get("pts"):
                pc = pd_.get("pts_change")
                if pc is not None:
                    arrow = "🚀" if pc > 5 else "↑" if pc > 0 else "💥" if pc < -5 else "↓" if pc < 0 else ""
                    pts_str = f"¥{pd_['pts']:,.1f}({pc:+.1f}%){arrow}"
                else:
                    pts_str = f"¥{pd_['pts']:,.1f}"
            pdf = e.get("pdf_data")
            div_str = ""
            if pdf:
                dy = pdf.get("div_yield")
                dc = pdf.get("div_change", "")
                if dy:
                    div_str = f"{dy}%"
                if dc:
                    div_str += f" {dc}" if div_str else dc
            guidance_str = ""
            if pdf and pdf.get("guidance_label"):
                guidance_str = pdf["guidance_label"]
                gd = pdf.get("guidance_detail", "")
                if gd:
                    guidance_str += f" {gd}"
            per_str = ""
            if pdf and pdf.get("per") and pdf["per"] not in ("－", ""):
                per_str = f"{pdf['per']}倍"
            doc_link = ""
            code_val = e.get("code", "")
            if pdf and pdf.get("pdf_url"):
                doc_link = pdf["pdf_url"]
            elif pdf and pdf.get("finance_url"):
                doc_link = pdf["finance_url"]
            elif code_val:
                doc_link = f"https://kabutan.jp/stock/finance?code={code_val}"
            tbl_rows.append({
                "評価": e.get("stars", "⭐⭐⭐"),
                "コード": e.get("code", ""),
                "銘柄名": ("🆕 " if e.get("_is_new") else "") + e["name"],
                "現在値": current_str,
                "PTS": pts_str,
                "PER": per_str,
                "配当利回り": div_str,
                "来期見通し": guidance_str,
                "決算ポイント": e.get("headline", ""),
                "コメント": e["comment"],
                "決算資料": doc_link,
            })
        df_kessan = pd.DataFrame(tbl_rows)
        st.dataframe(df_kessan, use_container_width=True, hide_index=True,
                     column_config={
                         "決算資料": st.column_config.LinkColumn("決算資料", display_text="📄開く"),
                     },
                     height=min(len(tbl_rows)*35+40, 800))

        # 要注意決算
        bad = [e for e in evaluated if e.get("score", 3) <= 2]
        if bad:
            st.markdown("### ⚠️ 要注意決算")
            for e in bad:
                url = e.get("url", f"https://kabutan.jp/stock/?code={e.get('code','')}")
                pdf = e.get("pdf_data")
                code_val = e.get("code", "")
                link_parts = []
                if pdf and pdf.get("pdf_url"):
                    link_parts.append(f'<a href="{pdf["pdf_url"]}" target="_blank" style="font-size:0.8rem;margin-left:8px;">📄PDF</a>')
                if code_val:
                    link_parts.append(f'<a href="https://kabutan.jp/stock/finance?code={code_val}" target="_blank" style="font-size:0.8rem;margin-left:4px;">📊決算詳細</a>')
                doc_links = " ".join(link_parts)
                st.markdown(
                    f'<div style="padding:8px 12px;margin-bottom:4px;border-radius:6px;'
                    f'border-left:4px solid #f44336;background:rgba(244,67,54,0.06);">'
                    f'<a href="{url}" target="_blank" style="color:inherit;text-decoration:none;">'
                    f'{e.get("code","")} {e["name"]}</a> — {e["comment"]}{doc_links}'
                    f'</div>', unsafe_allow_html=True
                )
