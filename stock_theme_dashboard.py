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
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

warnings.filterwarnings("ignore")

# =====================================================================
# ML モメンタムスコア: LightGBM による学習型モメンタム予測
# =====================================================================
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

_ML_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
os.makedirs(_ML_MODEL_DIR, exist_ok=True)

# ML要因分析の特徴量を日次でディスクキャッシュするディレクトリ
_ML_FEAT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_feature_cache")
os.makedirs(_ML_FEAT_CACHE_DIR, exist_ok=True)


def _ml_build_features(hist_df):
    """ヒストリカルデータから特徴量マトリクスを構築（1銘柄分）
    
    Parameters:
        hist_df: OHLCV DataFrame (index=Date)
    Returns:
        features DataFrame (index=Date, columns=特徴量名)
    """
    if hist_df is None or len(hist_df) < 80:
        return None

    close = hist_df["Close"].astype(float)
    high = hist_df["High"].astype(float) if "High" in hist_df.columns else close
    low = hist_df["Low"].astype(float) if "Low" in hist_df.columns else close
    volume = hist_df["Volume"].astype(float) if "Volume" in hist_df.columns else pd.Series(0, index=close.index)

    ret = close.pct_change()
    feat = pd.DataFrame(index=close.index)

    # リターン系
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_20d"] = close.pct_change(20)
    feat["ret_60d"] = close.pct_change(60)

    # ボラティリティ
    feat["vol_5d"] = ret.rolling(5).std()
    feat["vol_20d"] = ret.rolling(20).std()
    feat["vol_ratio"] = feat["vol_5d"] / feat["vol_20d"].replace(0, np.nan)

    # 出来高変化率
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    feat["volume_ratio"] = vol_ma5 / vol_ma20.replace(0, np.nan)

    # RSI (14日)
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # 移動平均乖離率
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    feat["dev_ma25"] = (close - ma25) / ma25 * 100
    feat["dev_ma75"] = (close - ma75) / ma75 * 100

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = macd_line - signal_line

    # ATR (Average True Range) 正規化
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs()
    }).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat["atr_norm"] = atr14 / close * 100

    # ボリンジャーバンド位置
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_position"] = (close - bb_ma) / (2 * bb_std.replace(0, np.nan))

    # 高値・安値からの位置
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    feat["price_position_20d"] = (close - low_20) / (high_20 - low_20).replace(0, np.nan)

    return feat.dropna()


def _ml_build_target(hist_df, forward_days=5):
    """将来N日リターンをターゲットとして構築
    
    Returns:
        Series of forward returns (regression target)
    """
    if hist_df is None or len(hist_df) < forward_days + 1:
        return None
    close = hist_df["Close"].astype(float)
    # 将来5日リターン（%）
    fwd_ret = close.shift(-forward_days) / close - 1
    return fwd_ret * 100  # パーセンテージ


def _ml_train_model(all_features_list, all_targets_list):
    """全銘柄の特徴量をプールして学習（ウォークフォワード方式）
    
    Parameters:
        all_features_list: list of DataFrames (各銘柄の特徴量)
        all_targets_list: list of Series (各銘柄のターゲット)
    Returns:
        trained LightGBM model or None
    """
    if not _HAS_LGBM:
        return None

    # 全銘柄の特徴量を結合
    X_list = []
    y_list = []
    for feat, tgt in zip(all_features_list, all_targets_list):
        if feat is None or tgt is None:
            continue
        # 共通indexでalign
        common_idx = feat.index.intersection(tgt.dropna().index)
        if len(common_idx) < 30:
            continue
        X_list.append(feat.loc[common_idx])
        y_list.append(tgt.loc[common_idx])

    if not X_list:
        return None

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    # NaN除去
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        return None

    # ウォークフォワード: 直近20%はバリデーション
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    return model


def _ml_get_model_path():
    """今日の日付に基づくモデルパスを返す"""
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(_ML_MODEL_DIR, f"momentum_lgbm_{today}.txt")


def _ml_load_or_train(hist_dict):
    """キャッシュされたモデルをロード、なければ学習して保存
    
    Parameters:
        hist_dict: dict of {ticker: DataFrame(OHLCV)} — 全ウォッチリスト銘柄のヒストリカルデータ
    Returns:
        (model, feature_names) or (None, None)
    """
    if not _HAS_LGBM:
        return None, None

    model_path = _ml_get_model_path()

    # 今日のモデルが既にあればロード
    if os.path.exists(model_path):
        try:
            model = lgb.Booster(model_file=model_path)
            # 特徴量名はモデルに保存されている
            return model, model.feature_name()
        except Exception:
            pass

    # 学習データ構築
    all_features = []
    all_targets = []
    for ticker, hist in hist_dict.items():
        feat = _ml_build_features(hist)
        tgt = _ml_build_target(hist)
        if feat is not None and tgt is not None:
            all_features.append(feat)
            all_targets.append(tgt)

    if not all_features:
        return None, None

    model = _ml_train_model(all_features, all_targets)
    if model is None:
        return None, None

    # モデル保存
    try:
        model.save_model(model_path)
        # 古いモデルファイルを削除（7日以上前）
        for f in os.listdir(_ML_MODEL_DIR):
            fpath = os.path.join(_ML_MODEL_DIR, f)
            if f.startswith("momentum_lgbm_") and f.endswith(".txt"):
                try:
                    file_date = datetime.strptime(f.replace("momentum_lgbm_", "").replace(".txt", ""), "%Y%m%d")
                    if (datetime.now() - file_date).days > 7:
                        os.remove(fpath)
                except Exception:
                    pass
    except Exception:
        pass

    return model, model.feature_name()


def _ml_predict_momentum(model, feature_names, hist_df):
    """学習済みモデルで直近の特徴量からモメンタムスコアを予測
    
    Returns:
        float: 予測される将来5日リターン（%）、予測不可の場合None
    """
    if model is None or hist_df is None:
        return None

    feat = _ml_build_features(hist_df)
    if feat is None or len(feat) == 0:
        return None

    # 直近1行の特徴量で予測
    latest = feat.iloc[[-1]]

    # 特徴量名の整合チェック
    missing = set(feature_names) - set(latest.columns)
    if missing:
        for col in missing:
            latest[col] = 0
    latest = latest[feature_names]

    if latest.isna().any(axis=1).iloc[0]:
        return None

    try:
        pred = model.predict(latest)[0]
        return round(float(pred), 3)
    except Exception:
        return None


def calc_ml_momentum_scores(hist_dict):
    """全ウォッチリスト銘柄のMLモメンタムスコアを一括計算
    
    Parameters:
        hist_dict: dict of {code: DataFrame(OHLCV)} — コード→ヒストリカルデータ
    Returns:
        dict of {code: ml_score} (予測不可の場合はNone)
        model_info: dict with training metadata
    """
    if not _HAS_LGBM or not hist_dict:
        return {}, {}

    model, feature_names = _ml_load_or_train(hist_dict)
    if model is None:
        return {}, {}

    scores = {}
    for code, hist in hist_dict.items():
        scores[code] = _ml_predict_momentum(model, feature_names, hist)

    # モデル情報
    model_info = {
        "n_features": len(feature_names) if feature_names else 0,
        "n_trees": model.num_trees() if model else 0,
        "feature_importance": {},
    }
    try:
        imp = model.feature_importance(importance_type="gain")
        top_idx = np.argsort(imp)[::-1][:5]
        model_info["feature_importance"] = {
            feature_names[i]: round(float(imp[i]), 1) for i in top_idx
        }
    except Exception:
        pass

    return scores, model_info

# =====================================================================

st.set_page_config(page_title="テーマ別資金フロー分析", page_icon="🎯", layout="wide")

# =====================================================================
# テーマ定義: テーマ名 → 代表銘柄(5社) + 対応する米国ETF + 概要
# =====================================================================
THEMES = {
    "蓄電池": {
        "stocks": {"6752.T": "パナソニック", "6762.T": "TDK", "6981.T": "村田製作所",
                   "6674.T": "GSユアサ", "6810.T": "マクセル",
                   "6619.T": "ダブル・スコープ", "4080.T": "田中化学研究所",
                   "4109.T": "ステラケミファ", "6996.T": "ニチコン", "5765.T": "五洋インテックス"},
        "us_etf": "QCLN", "desc": "EV・再エネ普及に伴う蓄電需要拡大", "kabutan": "蓄電池",
    },
    "防衛": {
        "stocks": {"7011.T": "三菱重工業", "7012.T": "川崎重工業", "7013.T": "IHI",
                   "5631.T": "日本製鋼所", "7721.T": "東京計器",
                   "6208.T": "石川製作所", "4274.T": "細谷火工", "7224.T": "新明和工業",
                   "7408.T": "ジャムコ", "6503.T": "三菱電機"},
        "us_etf": "ITA", "desc": "地政学リスク・防衛費増額", "kabutan": "防衛",
    },
    "レアアース": {
        "stocks": {"5713.T": "住友金属鉱山", "5714.T": "DOWAホールディングス",
                   "5706.T": "三井金属鉱業", "5711.T": "三菱マテリアル", "5727.T": "東邦チタニウム",
                   "5726.T": "大阪チタニウムテクノロジーズ", "5541.T": "大平洋金属",
                   "5715.T": "古河機械金属", "5801.T": "古河電気工業", "5707.T": "東邦亜鉛"},
        "us_etf": "REMX", "desc": "資源ナショナリズム・サプライチェーン再構築", "kabutan": "レアアース",
    },
    "地方銀行": {
        "stocks": {"7186.T": "コンコルディアFG", "8354.T": "ふくおかFG",
                   "7167.T": "めぶきFG", "8331.T": "千葉銀行", "8355.T": "静岡銀行",
                   "8368.T": "百五銀行", "8377.T": "ほくほくFG", "8385.T": "伊予銀行",
                   "8418.T": "山口FG", "8360.T": "山梨中央銀行"},
        "us_etf": "KRE", "desc": "金利上昇・地銀再編期待", "kabutan": "地方銀行",
    },
    "TOPIXコア30": {
        "stocks": {"7203.T": "トヨタ自動車", "8306.T": "三菱UFJFG", "6758.T": "ソニーG",
                   "6861.T": "キーエンス", "9984.T": "ソフトバンクG",
                   "6501.T": "日立製作所", "8316.T": "三井住友FG", "9432.T": "NTT",
                   "4063.T": "信越化学工業", "6902.T": "デンソー"},
        "us_etf": "SPY", "desc": "大型株への資金回帰・安定志向", "kabutan": "TOPIXコア30",
    },
    "半導体": {
        "stocks": {"8035.T": "東京エレクトロン", "6920.T": "レーザーテック",
                   "6857.T": "アドバンテスト", "6723.T": "ルネサスエレクトロニクス", "6146.T": "ディスコ",
                   "6526.T": "ソシオネクスト", "3436.T": "SUMCO",
                   "6963.T": "ローム", "6967.T": "新光電気工業", "4063.T": "信越化学工業"},
        "us_etf": "SOXX", "desc": "AI需要・半導体サイクル回復", "kabutan": "半導体",
    },
    "ペロブスカイト太陽電池": {
        "stocks": {"4005.T": "住友化学", "4183.T": "三井化学", "5019.T": "出光興産",
                   "6988.T": "日東電工", "4204.T": "積水化学工業",
                   "3891.T": "ニッポン高度紙工業", "4042.T": "東ソー",
                   "5020.T": "ENEOSホールディングス", "4188.T": "三菱ケミカルG", "6701.T": "NEC"},
        "us_etf": "TAN", "desc": "次世代太陽電池の実用化進展", "kabutan": "ペロブスカイト太陽電池",
    },
    "JPX日経400": {
        "stocks": {"8058.T": "三菱商事", "8031.T": "三井物産", "9432.T": "NTT",
                   "4063.T": "信越化学工業", "6501.T": "日立製作所",
                   "6098.T": "リクルートHD", "4568.T": "第一三共",
                   "6367.T": "ダイキン工業", "7741.T": "HOYA", "4519.T": "中外製薬"},
        "us_etf": "SPY", "desc": "資本効率重視・ガバナンス改革", "kabutan": "JPX日経400",
    },
    "肥料": {
        "stocks": {"4208.T": "UBE", "4401.T": "ADEKA", "4996.T": "クミアイ化学工業",
                   "4997.T": "日本農薬", "4955.T": "アグロカネショウ",
                   "4506.T": "住友ファーマ", "4631.T": "DIC",
                   "4042.T": "東ソー", "4021.T": "日産化学", "4452.T": "花王"},
        "us_etf": "MOO", "desc": "食料安全保障・農業DX", "kabutan": "肥料",
    },
    "石炭": {
        "stocks": {"1518.T": "三井松島HD", "1514.T": "住石HD",
                   "5765.T": "五洋インテックス", "8031.T": "三井物産", "8002.T": "丸紅",
                   "8001.T": "伊藤忠商事", "5020.T": "ENEOSホールディングス",
                   "9517.T": "イーレックス", "1605.T": "INPEX", "5021.T": "コスモエネルギーHD"},
        "us_etf": "XLE", "desc": "エネルギー安全保障・石炭価格高騰", "kabutan": "石炭",
    },
    "海水淡水化": {
        "stocks": {"6370.T": "栗田工業", "6368.T": "オルガノ", "6361.T": "荏原製作所",
                   "6366.T": "千代田化工建設", "6013.T": "タクミナ",
                   "6326.T": "クボタ", "6473.T": "ジェイテクト",
                   "6501.T": "日立製作所", "1963.T": "日揮HD", "6502.T": "東芝"},
        "us_etf": "PHO", "desc": "水資源不足・インフラ需要", "kabutan": "水ビジネス",
    },
    "宇宙開発": {
        "stocks": {"7012.T": "川崎重工業", "7011.T": "三菱重工業", "6208.T": "石川製作所",
                   "9412.T": "スカパーJSAT", "4274.T": "細谷火工",
                   "7224.T": "新明和工業", "6503.T": "三菱電機",
                   "7013.T": "IHI", "9432.T": "NTT", "6701.T": "NEC"},
        "us_etf": "UFO", "desc": "宇宙ビジネス商業化・安全保障", "kabutan": "宇宙開発関連",
    },
    "人工知能": {
        "stocks": {"6758.T": "ソニーG", "6861.T": "キーエンス", "9984.T": "ソフトバンクG",
                   "4686.T": "ジャストシステム", "3993.T": "PKSHA Technology",
                   "4382.T": "HEROZ", "3655.T": "ブレインパッド",
                   "6701.T": "NEC", "6702.T": "富士通", "9613.T": "NTTデータG"},
        "us_etf": "BOTZ", "desc": "生成AI普及・企業DX加速", "kabutan": "人工知能",
    },
    "データセンター": {
        "stocks": {"3042.T": "セキュアヴェイル", "3778.T": "さくらインターネット",
                   "9434.T": "ソフトバンク", "4812.T": "ISID", "3844.T": "コムチュア",
                   "3762.T": "テクマトリックス", "4684.T": "オービック",
                   "9432.T": "NTT", "3626.T": "TIS", "4307.T": "野村総合研究所"},
        "us_etf": "VPN", "desc": "クラウド需要・AI計算基盤拡大", "kabutan": "データセンター",
    },
    "フィジカルAI": {
        "stocks": {"6506.T": "安川電機", "6954.T": "ファナック", "7735.T": "SCREENホールディングス",
                   "6861.T": "キーエンス", "6645.T": "オムロン",
                   "6324.T": "ハーモニック・ドライブ", "6268.T": "ナブテスコ",
                   "6473.T": "ジェイテクト", "7272.T": "ヤマハ発動機", "6902.T": "デンソー"},
        "us_etf": "ROBO", "desc": "ロボティクス・自動化のAI融合", "kabutan": "ロボット",
    },
    "半導体製造装置": {
        "stocks": {"8035.T": "東京エレクトロン", "7735.T": "SCREENホールディングス",
                   "6146.T": "ディスコ", "7729.T": "東京精密", "6125.T": "岡本工作機械製作所",
                   "6857.T": "アドバンテスト", "6728.T": "アルバック",
                   "7731.T": "ニコン", "6920.T": "レーザーテック", "6526.T": "ソシオネクスト"},
        "us_etf": "SOXX", "desc": "先端プロセス投資・EUV関連", "kabutan": "半導体製造装置",
    },
    "再生可能エネルギー": {
        "stocks": {"9519.T": "レノバ", "9517.T": "イーレックス", "1407.T": "ウエストHD",
                   "9514.T": "エフオン", "3856.T": "Abalance",
                   "9522.T": "リニューアブル・ジャパン", "2151.T": "タケエイ",
                   "1407.T": "ウエストHD", "6255.T": "エヌ・ピー・シー", "5765.T": "五洋インテックス"},
        "us_etf": "ICLN", "desc": "脱炭素政策・GX推進", "kabutan": "太陽光発電関連",
    },
    "銀行": {
        "stocks": {"8306.T": "三菱UFJFG", "8316.T": "三井住友FG", "8411.T": "みずほFG",
                   "8309.T": "三井住友トラスト", "7182.T": "ゆうちょ銀行",
                   "8308.T": "りそなHD", "8473.T": "SBIホールディングス",
                   "8601.T": "大和証券G", "8604.T": "野村HD", "8750.T": "第一生命HD"},
        "us_etf": "XLF", "desc": "金利正常化・利ざや改善", "kabutan": "銀行",
    },
    "SaaS": {
        "stocks": {"4478.T": "フリー", "4443.T": "Sansan", "4776.T": "サイボウズ",
                   "3923.T": "ラクス", "4475.T": "HENNGE",
                   "4485.T": "JTOWER", "7046.T": "TDSE",
                   "4684.T": "オービック", "4307.T": "野村総合研究所", "3626.T": "TIS"},
        "us_etf": "WCLD", "desc": "企業DX・サブスクリプション成長", "kabutan": "SaaS",
    },
    "核融合発電": {
        "stocks": {"6501.T": "日立製作所", "6502.T": "東芝", "5631.T": "日本製鋼所",
                   "6366.T": "千代田化工建設", "5727.T": "東邦チタニウム",
                   "5726.T": "大阪チタニウムテクノロジーズ", "5801.T": "古河電気工業",
                   "6503.T": "三菱電機", "7011.T": "三菱重工業", "1963.T": "日揮HD"},
        "us_etf": "NLR", "desc": "次世代エネルギー・核融合実用化期待", "kabutan": "核融合発電",
    },
    "水素": {
        "stocks": {"6902.T": "デンソー", "7203.T": "トヨタ自動車", "8088.T": "岩谷産業",
                   "6366.T": "千代田化工建設", "5020.T": "ENEOSホールディングス",
                   "1963.T": "日揮HD", "6361.T": "荏原製作所",
                   "5019.T": "出光興産", "7012.T": "川崎重工業", "9531.T": "東京ガス"},
        "us_etf": "ICLN", "desc": "水素社会実現・グリーン水素", "kabutan": "水素",
    },
    "総合商社": {
        "stocks": {"8058.T": "三菱商事", "8031.T": "三井物産", "8001.T": "伊藤忠商事",
                   "8053.T": "住友商事", "8002.T": "丸紅",
                   "8015.T": "豊田通商", "2768.T": "双日",
                   "8020.T": "兼松", "3038.T": "神戸物産", "7466.T": "SPK"},
        "us_etf": "SPY", "desc": "バフェット効果・資源高・株主還元", "kabutan": "総合商社",
    },
    "インバウンド": {
        "stocks": {"9603.T": "エイチ・アイ・エス", "6191.T": "エアトリ",
                   "9024.T": "西武HD", "2809.T": "キユーピー", "7832.T": "バンダイナムコHD",
                   "4661.T": "オリエンタルランド", "9202.T": "ANA HD",
                   "9201.T": "日本航空", "3099.T": "三越伊勢丹HD", "2670.T": "ABCマート"},
        "us_etf": "JETS", "desc": "訪日外国人増加・円安恩恵", "kabutan": "インバウンド",
    },
    "海運": {
        "stocks": {"9101.T": "日本郵船", "9104.T": "商船三井", "9107.T": "川崎汽船",
                   "9110.T": "NSユナイテッド海運", "9119.T": "飯野海運",
                   "9115.T": "明治海運", "9171.T": "栗林商船",
                   "9173.T": "東海汽船", "9179.T": "川崎近海汽船", "9130.T": "共栄タンカー"},
        "us_etf": "SEA", "desc": "海運市況・コンテナ運賃動向", "kabutan": "海運",
    },
    "鉄道": {
        "stocks": {"9020.T": "JR東日本", "9021.T": "JR西日本", "9022.T": "JR東海",
                   "9005.T": "東急", "9009.T": "京成電鉄",
                   "9001.T": "東武鉄道", "9006.T": "京浜急行電鉄",
                   "9007.T": "小田急電鉄", "9008.T": "京王電鉄", "9042.T": "阪急阪神HD"},
        "us_etf": "XLI", "desc": "インバウンド・不動産開発・運賃改定", "kabutan": "鉄道関連",
    },
    "不動産": {
        "stocks": {"8801.T": "三井不動産", "8802.T": "三菱地所", "8830.T": "住友不動産",
                   "3289.T": "東急不動産HD", "8804.T": "東京建物",
                   "8803.T": "平和不動産", "3231.T": "野村不動産HD",
                   "8905.T": "イオンモール", "3291.T": "飯田グループHD", "8934.T": "サンフロンティア不動産"},
        "us_etf": "IYR", "desc": "金利動向・オフィス回帰・再開発", "kabutan": "不動産関連",
    },
    "建設": {
        "stocks": {"1801.T": "大成建設", "1802.T": "大林組", "1812.T": "鹿島建設",
                   "1803.T": "清水建設", "1808.T": "長谷工コーポレーション",
                   "1925.T": "大和ハウス工業", "1928.T": "積水ハウス",
                   "1878.T": "大東建託", "1860.T": "戸田建設", "1893.T": "五洋建設"},
        "us_etf": "XLI", "desc": "国土強靭化・再開発・半導体工場建設", "kabutan": "建設",
    },
    "光デバイス": {
        "stocks": {"6146.T": "ディスコ", "6920.T": "レーザーテック", "6962.T": "大真空",
                   "6976.T": "太陽誘電", "6728.T": "アルバック",
                   "6965.T": "浜松ホトニクス", "6925.T": "ウシオ電機",
                   "5765.T": "五洋インテックス", "6804.T": "ホシデン", "6967.T": "新光電気工業"},
        "us_etf": "SOXX", "desc": "光半導体・光通信・シリコンフォトニクス", "kabutan": "光デバイス",
    },
    "電力": {
        "stocks": {"9501.T": "東京電力HD", "9502.T": "中部電力", "9503.T": "関西電力",
                   "9504.T": "中国電力", "9505.T": "北陸電力",
                   "9506.T": "東北電力", "9507.T": "四国電力",
                   "9508.T": "九州電力", "9509.T": "北海道電力", "9513.T": "J-POWER"},
        "us_etf": "XLU", "desc": "原発再稼働・データセンター電力需要・電力料金改定", "kabutan": "電力会社",
    },
    "造船": {
        "stocks": {"7013.T": "IHI", "7011.T": "三菱重工業", "7012.T": "川崎重工業",
                   "7014.T": "名村造船所", "7003.T": "三井E&S",
                   "7004.T": "日立造船", "7007.T": "佐世保重工業",
                   "7018.T": "内海造船", "6203.T": "豊和工業", "5765.T": "五洋インテックス"},
        "us_etf": "XLI", "desc": "LNG船需要・防衛艦艇・老朽船更新サイクル", "kabutan": "造船",
    },
    "非鉄金属": {
        "stocks": {"5713.T": "住友金属鉱山", "5711.T": "三菱マテリアル", "5706.T": "三井金属鉱業",
                   "5714.T": "DOWAホールディングス", "5801.T": "古河電気工業",
                   "5802.T": "住友電気工業", "5803.T": "フジクラ",
                   "5707.T": "東邦亜鉛", "5715.T": "古河機械金属", "5741.T": "UACJ"},
        "us_etf": "XME", "desc": "銅価格上昇・電線需要・EV素材", "kabutan": "非鉄",
    },
    "ゲーム": {
        "stocks": {"7974.T": "任天堂", "9766.T": "コナミグループ", "9684.T": "スクウェア・エニックスHD",
                   "7832.T": "バンダイナムコHD", "3659.T": "ネクソン",
                   "2121.T": "MIXI", "3765.T": "ガンホー・オンライン",
                   "3668.T": "コロプラ", "9697.T": "カプコン", "4751.T": "サイバーエージェント"},
        "us_etf": "HERO", "desc": "ゲーム市場拡大・IP活用・eスポーツ", "kabutan": "ゲーム",
    },
    "エヌビディア関連": {
        "stocks": {"6857.T": "アドバンテスト", "8035.T": "東京エレクトロン", "6146.T": "ディスコ",
                   "6920.T": "レーザーテック", "6526.T": "ソシオネクスト",
                   "6723.T": "ルネサスエレクトロニクス", "3436.T": "SUMCO",
                   "6728.T": "アルバック", "7735.T": "SCREENホールディングス", "6963.T": "ローム"},
        "us_etf": "SOXX", "desc": "NVIDIA決算連動・GPU需要・AI半導体サプライチェーン", "kabutan": "エヌビディア関連",
    },
    "仮想通貨": {
        "stocks": {"8473.T": "SBIホールディングス", "8698.T": "マネックスグループ",
                   "3807.T": "フィスコ", "2315.T": "CAICA DIGITAL",
                   "3696.T": "セレス", "4475.T": "HENNGE",
                   "3825.T": "リミックスポイント", "1580.T": "日経平均ベア",
                   "2330.T": "フォーサイド", "7177.T": "GMOフィナンシャルHD"},
        "us_etf": "BITO", "desc": "暗号資産市場・ビットコインETF・Web3", "kabutan": "暗号資産（税制・規制）",
    },
    "金": {
        "stocks": {"1328.T": "金価格連動ETF", "5713.T": "住友金属鉱山",
                   "1540.T": "純金上場信託", "7456.T": "松田産業",
                   "5711.T": "三菱マテリアル", "5714.T": "DOWAホールディングス",
                   "1326.T": "SPDRゴールド", "7593.T": "VTホールディングス",
                   "5706.T": "三井金属鉱業", "1672.T": "金ETF"},
        "us_etf": "GLD", "desc": "金価格上昇・安全資産需要・中央銀行買い", "kabutan": "金",
    },
}

# =====================================================================
# みんかぶ テーマランキング取得
# =====================================================================
_MINKABU_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://minkabu.jp/theme/",
}

# みんかぶで取得可能なテーマランキング種別
MINKABU_THEME_RANKINGS = {
    "急上昇": "https://minkabu.jp/theme/rise_ranking",
    "人気": "https://minkabu.jp/theme/popular_ranking",
}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_minkabu_theme_ranking(kind="急上昇"):
    """みんかぶのテーマランキングを取得してDataFrameで返す。

    返す列: 順位, テーマ, 変動率(%), 関連銘柄数, 代表銘柄コード, 代表銘柄名,
            テーマURL, ランキング種別
    取得失敗時は空のDataFrameを返す。
    """
    import re as _re

    url = MINKABU_THEME_RANKINGS.get(kind)
    if not url:
        return pd.DataFrame()

    try:
        resp = requests.get(url, headers=_MINKABU_HEADERS, timeout=20)
        if resp.status_code != 200:
            return pd.DataFrame()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return pd.DataFrame()

        table = tables[0]
        rows = table.find_all("tr")
        records = []
        for tr in rows[1:]:
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            cells = [td.get_text(" ", strip=True) for td in tds]

            # テーマ名・URL
            theme_a = tr.find("a", href=_re.compile(r"^/theme/"))
            if not theme_a:
                continue
            theme_name = theme_a.get_text(strip=True)
            theme_url = "https://minkabu.jp" + theme_a["href"]

            # 順位（先頭セルの数字）
            rank = None
            for c in cells:
                m = _re.match(r"^\s*(\d+)\s*$", c)
                if m:
                    rank = int(m.group(1))
                    break

            # 変動率（末尾の +x.xx% / -x.xx%）
            change = None
            for c in cells:
                m = _re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", c)
                if m:
                    change = float(m.group(1))
            # 代表銘柄
            stock_a = tr.find("a", href=_re.compile(r"^/stock/"))
            rep_code = None
            rep_name = None
            if stock_a:
                rep_code = stock_a["href"].split("/stock/")[-1].strip()
                rep_text = stock_a.get_text(" ", strip=True)
                # "3132 マクニカＨＤ" のような表記から銘柄名を抽出
                m = _re.match(r"^\s*([0-9A-Za-z]+)\s+(.*)$", rep_text)
                rep_name = m.group(2).strip() if m else rep_text

            # 関連銘柄数（"全166社"）
            n_stocks = None
            for c in cells:
                m = _re.search(r"全\s*(\d+)\s*社", c)
                if m:
                    n_stocks = int(m.group(1))
                    break

            if theme_name and change is not None:
                records.append({
                    "順位": rank,
                    "テーマ": theme_name,
                    "変動率(%)": change,
                    "関連銘柄数": n_stocks,
                    "代表銘柄コード": rep_code,
                    "代表銘柄名": rep_name,
                    "テーマURL": theme_url,
                    "ランキング種別": kind,
                })

        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_minkabu_all_rankings():
    """みんかぶの全テーマランキング種別を取得し結合したDataFrameを返す。"""
    frames = []
    for kind in MINKABU_THEME_RANKINGS:
        df = fetch_minkabu_theme_ranking(kind)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_minkabu_rep_momentum(codes_tuple):
    """みんかぶテーマの代表銘柄について、値動き指標を一括取得する。

    返す: {code: {"週次": %, "月次": %, "高値比": %, "RSI": float, "出来高比": float}}
    高値比 = 52週高値に対する現在値の位置(0〜100、100が高値)。
    新形式コード(例 285A)もyfinanceで取得可能。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(code):
        try:
            hist = yf.Ticker(f"{code}.T").history(period="1y")
            if hist.empty or len(hist) < 20:
                return code, None
            close = hist["Close"]
            last = float(close.iloc[-1])

            ret_1w = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else None
            ret_1m = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None

            hi_52 = float(close.max())
            lo_52 = float(close.min())
            pos_52 = (last - lo_52) / (hi_52 - lo_52) * 100 if hi_52 > lo_52 else None

            # RSI(14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = None
            if loss.iloc[-1] is not None and not pd.isna(loss.iloc[-1]):
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
                rsi = 100 - (100 / (1 + rs))

            # 出来高比（直近5日平均 / 20日平均）
            vol = hist["Volume"]
            vol_ratio = None
            if len(vol) >= 20:
                v5 = vol.iloc[-5:].mean()
                v20 = vol.iloc[-20:].mean()
                vol_ratio = float(v5 / v20) if v20 > 0 else None

            return code, {
                "週次": ret_1w, "月次": ret_1m, "高値比": pos_52,
                "RSI": rsi, "出来高比": vol_ratio,
            }
        except Exception:
            return code, None

    result = {}
    codes = [c for c in codes_tuple if c]
    if not codes:
        return result
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_one, c) for c in codes]
        for f in as_completed(futs):
            code, data = f.result()
            if data is not None:
                result[code] = data
    return result


def build_minkabu_expectation(mk_df):
    """みんかぶランキング＋代表銘柄の値動きから『上昇期待スコア』を構築する。

    スコア構成（0〜100）:
      - 勢い（週次・月次モメンタム）        最大45点
      - 当日変動率                          最大15点
      - 押し目余地（52週高値からの距離）    最大15点（高値圏すぎるテーマは減点）
      - 注目度（人気/急上昇ランクインと順位）最大25点
    返す: テーマ単位に集約したDataFrame
    """
    if mk_df is None or mk_df.empty:
        return pd.DataFrame()

    # テーマ単位に集約（人気/急上昇の両方に出るテーマは1行にまとめる）
    agg = {}
    for _, r in mk_df.iterrows():
        t = r["テーマ"]
        if t not in agg:
            agg[t] = {
                "テーマ": t,
                "代表銘柄コード": r.get("代表銘柄コード"),
                "代表銘柄名": r.get("代表銘柄名"),
                "関連銘柄数": r.get("関連銘柄数"),
                "テーマURL": r.get("テーマURL"),
                "当日変動率": r.get("変動率(%)"),
                "人気順位": None,
                "急上昇順位": None,
            }
        if r.get("ランキング種別") == "人気":
            agg[t]["人気順位"] = r.get("順位")
        elif r.get("ランキング種別") == "急上昇":
            agg[t]["急上昇順位"] = r.get("順位")

    rows = list(agg.values())
    codes = tuple({row["代表銘柄コード"] for row in rows if row.get("代表銘柄コード")})
    mom = fetch_minkabu_rep_momentum(codes)

    for row in rows:
        m = mom.get(row.get("代表銘柄コード"), {}) or {}
        row["週次(%)"] = m.get("週次")
        row["月次(%)"] = m.get("月次")
        row["高値比"] = m.get("高値比")
        row["RSI"] = m.get("RSI")
        row["出来高比"] = m.get("出来高比")

        score = 0.0

        # 勢い（週次・月次）: 最大45点
        w = row["週次(%)"]
        mn = row["月次(%)"]
        if w is not None:
            score += max(0.0, min(25.0, (w + 5) / 20 * 25))  # -5%→0点, +15%→満点
        if mn is not None:
            score += max(0.0, min(20.0, (mn + 10) / 40 * 20))  # -10%→0点, +30%→満点

        # 当日変動率: 最大15点
        d = row["当日変動率"]
        if d is not None:
            score += max(0.0, min(15.0, (d + 1) / 4 * 15))  # -1%→0点, +3%→満点

        # 押し目余地（52週高値比）: 最大15点。40〜80%が理想、過熱は減点
        pos = row["高値比"]
        if pos is not None:
            if pos <= 80:
                score += max(0.0, min(15.0, pos / 80 * 15))
            else:
                score += max(0.0, 15.0 - (pos - 80) / 20 * 15)  # 高値圏は減点

        # 注目度: 最大25点（人気・急上昇ランクイン、上位ほど高い）
        attn = 0.0
        if row["人気順位"] is not None:
            attn += max(0.0, (21 - row["人気順位"]) / 20 * 12)
        if row["急上昇順位"] is not None:
            attn += max(0.0, (21 - row["急上昇順位"]) / 20 * 13)
        score += min(25.0, attn)

        row["期待スコア"] = round(score, 1)

        # 評価ラベル
        if score >= 70:
            row["評価"] = "🔥 最有力"
        elif score >= 55:
            row["評価"] = "⭐ 有力"
        elif score >= 40:
            row["評価"] = "○ 注目"
        else:
            row["評価"] = "・様子見"

    df = pd.DataFrame(rows).sort_values("期待スコア", ascending=False).reset_index(drop=True)
    return df


# =====================================================================
# 上昇初動検出（まだ上がっていないが、これから上がりそうな銘柄）
# =====================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_early_stage_signals(codes_tuple, names_map=None):
    """各銘柄について『上昇初動』の予兆指標を算出し、初動スコアと寄与理由を返す。

    初動の考え方: まだ本格上昇していない（過熱していない）が、
    出来高増・下値切り上げ・MA収束上抜け・押し目完了などの
    トレンド転換の予兆が揃っている状態を高く評価する。

    返す: DataFrame（初動スコア降順）。列に各寄与要素と『主な根拠』を含む。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    names_map = names_map or {}

    def _one(code):
        try:
            hist = yf.Ticker(f"{code}.T").history(period="1y")
            if hist.empty or len(hist) < 60:
                return code, None
            close = hist["Close"]
            vol = hist["Volume"]
            last = float(close.iloc[-1])

            # リターン
            ret_1w = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0.0
            ret_1m = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
            ret_3m = (last / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else 0.0

            # 移動平均
            ma5 = close.rolling(5).mean()
            ma25 = close.rolling(25).mean()
            ma75 = close.rolling(75).mean() if len(close) >= 75 else close.rolling(len(close)).mean()
            ma5_last, ma25_last = float(ma5.iloc[-1]), float(ma25.iloc[-1])
            ma75_last = float(ma75.iloc[-1])

            # 出来高比（直近5日平均 / 25日平均）
            v5 = float(vol.iloc[-5:].mean())
            v25 = float(vol.iloc[-25:].mean())
            vol_ratio = v5 / v25 if v25 > 0 else 1.0

            # RSI(14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 50.0
            if not pd.isna(loss.iloc[-1]):
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
                rsi = float(100 - (100 / (1 + rs)))

            # 52週レンジ内の位置
            hi_52 = float(close.max())
            lo_52 = float(close.min())
            pos_52 = (last - lo_52) / (hi_52 - lo_52) * 100 if hi_52 > lo_52 else 50.0

            # 下値切り上げ判定（直近10日安値 > その前10日安値）
            low = hist["Low"]
            recent_low = float(low.iloc[-10:].min())
            prev_low = float(low.iloc[-20:-10].min()) if len(low) >= 20 else recent_low
            higher_low = recent_low > prev_low

            # ボラティリティ収縮（直近20日ATR比 < 過去60日ATR比）＝エネルギー蓄積
            tr = pd.concat([
                hist["High"] - hist["Low"],
                (hist["High"] - close.shift(1)).abs(),
                (hist["Low"] - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr20 = float(tr.iloc[-20:].mean())
            atr60 = float(tr.iloc[-60:].mean())
            squeeze = atr20 < atr60 * 0.9  # ボラ収縮

            # MA5 の傾き（直近5日で上向き）
            ma5_slope_up = ma5.iloc[-1] > ma5.iloc[-4] if len(ma5) >= 4 else False

            # === 寄与要素ごとのスコア（各0〜満点、合計100） ===
            contribs = {}

            # 1. 未過熱（まだ上がっていない）: 月次リターンが控えめなほど高い  最大22点
            #    月次 -3%〜+8% が理想。大きく上昇済みは初動ではない
            if ret_1m <= 8:
                c1 = max(0.0, min(22.0, (12 - abs(ret_1m - 2)) / 12 * 22))
            else:
                c1 = max(0.0, 22.0 - (ret_1m - 8) * 2)  # 上がりすぎは減点
            contribs["未過熱度"] = round(max(0.0, c1), 1)

            # 2. 出来高増加（資金流入の初期兆候）: 最大22点
            c2 = max(0.0, min(22.0, (vol_ratio - 1.0) / 1.0 * 22))  # 2倍で満点
            contribs["出来高増加"] = round(c2, 1)

            # 3. 下値切り上げ（底固め）: 最大16点
            contribs["下値切り上げ"] = 16.0 if higher_low else 0.0

            # 4. MA収束・上抜け（トレンド転換初期）: 最大18点
            c4 = 0.0
            if ma5_last >= ma25_last:  # 短期が中期を上抜け
                c4 += 10.0
            # MA25とMA75が接近（収束）
            if ma75_last > 0 and abs(ma25_last - ma75_last) / ma75_last < 0.05:
                c4 += 5.0
            if ma5_slope_up:
                c4 += 3.0
            contribs["MA転換"] = round(min(18.0, c4), 1)

            # 5. 押し目余地・位置（高値圏でない）: 最大12点
            #    52週位置 30〜65% が初動として理想
            if pos_52 <= 65:
                c5 = max(0.0, min(12.0, (65 - abs(pos_52 - 45)) / 65 * 12 + 4))
            else:
                c5 = max(0.0, 12.0 - (pos_52 - 65) / 35 * 12)
            contribs["位置余地"] = round(max(0.0, c5), 1)

            # 6. RSI中位帯（過熱でも底でもなく上昇余地）: 最大10点
            #    RSI 45〜60 が理想
            c6 = max(0.0, min(10.0, (15 - abs(rsi - 52)) / 15 * 10))
            contribs["RSI適正"] = round(max(0.0, c6), 1)

            # ボラ収縮ボーナス
            if squeeze:
                contribs["ボラ収縮"] = 5.0
            else:
                contribs["ボラ収縮"] = 0.0

            total = round(sum(contribs.values()), 1)

            # 上位寄与要素（理由用）
            top_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

            return code, {
                "銘柄名": names_map.get(code, code),
                "初動スコア": total,
                "寄与": contribs,
                "上位寄与": top_contribs,
                "週次(%)": round(ret_1w, 1),
                "月次(%)": round(ret_1m, 1),
                "3ヶ月(%)": round(ret_3m, 1),
                "出来高比": round(vol_ratio, 2),
                "RSI": round(rsi, 0),
                "52週位置": round(pos_52, 0),
                "現在値": round(last, 1),
            }
        except Exception:
            return code, None

    codes = [c for c in codes_tuple if c]
    results = {}
    if not codes:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_one, c) for c in codes]
        for f in as_completed(futs):
            code, data = f.result()
            if data is not None:
                results[code] = data

    rows = []
    for code, d in results.items():
        # 「まだ上がっていない」フィルタ: 月次+15%超 or 52週位置85超は初動でないので除外
        if d["月次(%)"] > 15 or d["52週位置"] > 85:
            continue
        # 理由テキスト（上位3寄与、0点は除く）
        reason_parts = []
        _labels = {
            "未過熱度": "まだ本格上昇前",
            "出来高増加": "出来高増加(資金流入)",
            "下値切り上げ": "下値切り上げ(底固め)",
            "MA転換": "移動平均が上向き転換",
            "位置余地": "高値まで上昇余地あり",
            "RSI適正": "RSI中位で上昇余地",
            "ボラ収縮": "値動き収縮(エネルギー蓄積)",
        }
        for name, val in d["上位寄与"]:
            if val > 0 and len(reason_parts) < 3:
                reason_parts.append(f"{_labels.get(name, name)}(+{val:.0f})")
        d["主な根拠"] = " / ".join(reason_parts) if reason_parts else "予兆弱い"
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("初動スコア", ascending=False).reset_index(drop=True)
    return df


# 上昇理由の分類カテゴリ（ポジティブ材料）とキーワード
_THEME_REASON_CATEGORIES = [
    ("🏆 業績・好決算", ["増益", "最高益", "過去最高", "上方修正", "好決算", "黒字転換",
                      "増収", "受注", "受注高", "想定超", "上振れ", "営業利益"]),
    ("💰 株主還元", ["増配", "復配", "自社株買い", "自己株式", "株主還元", "配当性向",
                  "記念配当", "増益増配"]),
    ("🤝 提携・M&A", ["提携", "資本提携", "業務提携", "買収", "TOB", "MBO", "合併",
                   "統合", "出資", "協業", "共同開発"]),
    ("📈 格上げ・目標株価", ["格上げ", "目標株価", "レーティング", "強気", "投資判断",
                        "買い推奨", "オーバーウエート", "アナリスト"]),
    ("🏭 新製品・受注", ["新製品", "新工場", "量産", "新技術", "開発成功", "実用化",
                     "採用", "供給", "増産", "設備投資"]),
    ("🏛️ 政策・国策", ["補助金", "政府", "国策", "規制緩和", "法改正", "予算", "経済対策",
                    "閣議決定", "支援", "減税", "GX", "経済安全保障"]),
    ("🌍 市況・需給", ["需要拡大", "市況改善", "価格上昇", "値上げ", "品薄", "供給不足",
                   "特需", "需給逼迫", "円安", "資源高"]),
    ("🚀 テーマ・思惑", ["思惑", "期待", "関連銘柄", "物色", "人気化", "急騰", "ストップ高",
                    "材料視", "注目"]),
]

_THEME_NEGATIVE_WORDS = ["下方修正", "減益", "減収", "赤字", "最終赤字", "業績悪化",
                         "格下げ", "減配", "無配", "不正", "リコール", "急落",
                         "暴落", "懸念", "下落", "売られ"]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_theme_uptrend_reason(theme_name):
    """テーマ名でGoogle Newsを検索し、上昇のポジティブ材料を分類・抽出する。

    返す: {
      "categories": [(カテゴリ名, ヒット件数, [見出し例...]), ...],  # 多い順
      "headlines": [(タイトル, リンク, 掲載日, 極性), ...],          # 直近順
      "summary": "主な材料の要約文",
      "pos": ポジ見出し数, "neg": ネガ見出し数,
    }
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime as _pdt

    result = {"categories": [], "headlines": [], "summary": "", "pos": 0, "neg": 0}
    try:
        q = quote(f"{theme_name} 株 関連 OR 急騰 OR 物色 OR 材料")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return result
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        if not items:
            return result

        cat_hits = {c[0]: [] for c in _THEME_REASON_CATEGORIES}
        headlines = []
        pos_total = 0
        neg_total = 0

        for item in items[:25]:
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            pub = item.findtext("pubDate", "").strip()
            if not title:
                continue
            try:
                pub_dt = _pdt(pub)
                pub_str = pub_dt.strftime("%m/%d")
            except Exception:
                pub_dt = None
                pub_str = ""

            # 直近45日以内に限定（古い材料は除外）
            if pub_dt is not None:
                try:
                    age_days = (datetime.now(pub_dt.tzinfo) - pub_dt).days
                    if age_days > 45:
                        continue
                except Exception:
                    pass

            polarity = 0
            matched_cat = None
            for cat_name, kws in _THEME_REASON_CATEGORIES:
                if any(kw in title for kw in kws):
                    cat_hits[cat_name].append(title)
                    polarity += 1
                    if matched_cat is None:
                        matched_cat = cat_name
            neg_hit = any(nw in title for nw in _THEME_NEGATIVE_WORDS)
            if neg_hit:
                polarity -= 1

            if polarity > 0:
                pos_total += 1
            elif polarity < 0:
                neg_total += 1

            headlines.append((title, link, pub_str, polarity))

        # カテゴリを件数の多い順に
        cats_sorted = sorted(
            [(name, len(hl), hl[:3]) for name, hl in cat_hits.items() if hl],
            key=lambda x: x[1], reverse=True
        )

        # 要約文を生成（上位カテゴリ名を並べる）
        if cats_sorted:
            top_names = [c[0].split(" ", 1)[-1] for c in cats_sorted[:3]]
            summary = "・".join(top_names) + " などの材料が観測されています"
        else:
            summary = "明確なポジティブ材料は抽出できませんでした（テーマ全体の地合い・需給要因の可能性）"

        result["categories"] = cats_sorted
        result["headlines"] = headlines[:8]
        result["summary"] = summary
        result["pos"] = pos_total
        result["neg"] = neg_total
        return result
    except Exception:
        return result


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


def _is_intraday_last_row(price_index):
    """価格データの最終行が『場中の未確定データ』かどうかを判定する。

    日本市場(JST)の大引けは15:00。最終行の日付が現在のJST営業日と同じで、
    かつ現在時刻が15:00(JST)より前なら、その行はまだ確定していない場中データ。
    """
    try:
        if len(price_index) == 0:
            return False
        from datetime import timezone, timedelta
        jst = timezone(timedelta(hours=9))
        now_jst = datetime.now(jst)
        last_date = pd.Timestamp(price_index[-1]).date()
        # 最終行が本日(JST)かつ 15:15(JST) より前なら未確定とみなす（余裕を見て15分）
        if last_date == now_jst.date() and (now_jst.hour < 15 or (now_jst.hour == 15 and now_jst.minute < 15)):
            return True
        return False
    except Exception:
        return False


def build_theme_index(period="6mo", include_intraday=False):
    """全テーマの等金額加重インデックス（日次リターン）を構築。

    include_intraday=False（既定）の場合、場中の未確定な最終行を除外し、
    確定した最新営業日ベースでリターンを算出する。これにより日次ランキングが
    『途中経過の値』でブレる問題を防ぐ。

    返り値: (theme_ret_df, theme_cum, meta)
      meta = {"intraday_dropped": bool, "last_date": Timestamp, "is_intraday": bool}
    """
    all_tickers = set()
    for info in THEMES.values():
        all_tickers.update(info["stocks"].keys())
    all_tickers = sorted(all_tickers)

    prices = fetch_prices(all_tickers, period=period)
    meta = {"intraday_dropped": False, "last_date": None, "is_intraday": False}
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), meta

    prices = prices.sort_index()

    # 場中の未確定最終行を検出
    is_intraday = _is_intraday_last_row(prices.index)
    meta["is_intraday"] = is_intraday
    if is_intraday:
        meta["intraday_dropped"] = not include_intraday
        if not include_intraday and len(prices) > 1:
            # 未確定の当日行を落として確定営業日ベースにする
            prices = prices.iloc[:-1]

    if len(prices) >= 1:
        meta["last_date"] = pd.Timestamp(prices.index[-1])

    # テーマ別等金額加重リターン
    # 各日で最低カバレッジ(半数以上の銘柄にデータあり)を満たさない日は
    # 少数銘柄に引っ張られてノイズになるため NaN 扱いにする
    theme_returns = {}
    for theme, info in THEMES.items():
        cols = [t for t in info["stocks"].keys() if t in prices.columns]
        if len(cols) == 0:
            continue
        ret = prices[cols].pct_change()
        valid_count = ret.notna().sum(axis=1)
        min_required = max(1, int(np.ceil(len(cols) * 0.5)))
        mean_ret = ret.mean(axis=1)
        mean_ret[valid_count < min_required] = np.nan
        theme_returns[theme] = mean_ret

    theme_ret_df = pd.DataFrame(theme_returns).dropna(how="all")
    theme_cum = (1 + theme_ret_df.fillna(0)).cumprod() - 1  # 累積リターン

    return theme_ret_df, theme_cum, meta



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
    "電力": ["東京電力", "中部電力", "関西電力", "中国電力", "北陸電力",
            "東北電力", "四国電力", "九州電力", "北海道電力", "J-POWER"],
    "造船": ["IHI", "三菱重工", "川崎重工", "名村造船", "三井E&S",
            "日立造船", "佐世保重工", "内海造船"],
    "非鉄金属": ["住友金属鉱山", "三菱マテリアル", "三井金属", "DOWA", "古河電工",
              "住友電工", "フジクラ", "東邦亜鉛", "UACJ"],
    "ゲーム": ["任天堂", "コナミ", "スクエニ", "バンダイナムコ", "ネクソン",
            "カプコン", "ガンホー", "コロプラ", "サイバーエージェント"],
    "エヌビディア関連": ["アドバンテスト", "東京エレクトロン", "ディスコ", "レーザーテック",
                      "ソシオネクスト", "ルネサス", "SUMCO", "SCREEN"],
    "仮想通貨": ["SBI", "マネックス", "フィスコ", "CAICA", "セレス",
              "リミックスポイント", "GMOフィナンシャル"],
    "金": ["住友金属鉱山", "三菱マテリアル", "DOWA", "松田産業", "三井金属"],
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
    "電力": ["電力", "電力会社", "原発", "原子力", "再稼働", "電気料金", "送配電",
            "電力需要", "電力自由化", "発電", "火力", "原発再稼働", "電力逼迫"],
    "造船": ["造船", "船舶", "LNG船", "コンテナ船", "タンカー", "艦艇", "護衛艦",
            "新造船", "ドック", "舶用", "バルカー", "受注残"],
    "非鉄金属": ["非鉄", "非鉄金属", "銅", "アルミ", "亜鉛", "ニッケル", "電線",
              "伸銅", "銅価格", "LME", "電力ケーブル", "データセンター電線"],
    "ゲーム": ["ゲーム", "任天堂", "Switch", "PlayStation", "eスポーツ", "ソシャゲ",
            "モバイルゲーム", "IP", "コンソール", "Steam"],
    "エヌビディア関連": ["エヌビディア", "NVIDIA", "GPU", "H100", "B200", "AI半導体",
                      "データセンターGPU", "CUDAコア", "Jensen", "GeForce"],
    "仮想通貨": ["仮想通貨", "暗号資産", "ビットコイン", "Bitcoin", "イーサリアム",
              "ブロックチェーン", "Web3", "NFT", "DeFi", "暗号資産ETF", "マイニング"],
    "金": ["金", "ゴールド", "金価格", "金鉱", "安全資産", "有事の金",
          "中央銀行", "金ETF", "金先物", "貴金属"],
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
    """日次/週次/月次リターンを算出。

    日次は『共通の最新確定営業日』のリターンを返す。テーマごとに最終有効日が
    ズレると別々の日を比較してしまうため、全テーマ共通の最終日で揃える。
    """
    if theme_ret_df.empty:
        return pd.Series(dtype=float)
    if period == "1d":
        if len(theme_ret_df) == 0:
            return pd.Series(dtype=float)
        # 最終行を基準にしつつ、その日にデータが無いテーマは
        # 直近の有効値ではなく欠損(NaN)として扱い、同一営業日で比較する
        last_row = theme_ret_df.iloc[-1]
        return last_row
    elif period == "1w":
        return theme_ret_df.iloc[-5:].sum(min_count=1) if len(theme_ret_df) >= 5 else theme_ret_df.sum(min_count=1)
    elif period == "1m":
        return theme_ret_df.iloc[-21:].sum(min_count=1) if len(theme_ret_df) >= 21 else theme_ret_df.sum(min_count=1)
    elif period == "3m":
        return theme_ret_df.iloc[-63:].sum(min_count=1) if len(theme_ret_df) >= 63 else theme_ret_df.sum(min_count=1)
    return theme_ret_df.sum(min_count=1)


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


def analyze_weekly_regime(close_df, lookback_weeks=52):
    """週足での「逆張り vs 順張り」レジーム判定。

    対象銘柄群（または指数）の週次終値から、以下を総合して
    どちらの戦略が機能しやすい相場かを判定する。

    判定根拠:
      1. 週次リターンのラグ1自己相関（時系列・クロスセクション平均）
         - 正 → 順張り(モメンタム)が効きやすい
         - 負 → 逆張り(ミーンリバージョン)が効きやすい
      2. 戦略バックテスト勝率: 「前週勝ち組を買う(順張り)」vs
         「前週負け組を買う(逆張り)」をクロスセクションで週次比較
      3. ボラティリティ・レジーム（高ボラは逆張り寄りになりやすい補足情報）

    Parameters
    ----------
    close_df : pd.DataFrame
        index=日付, columns=銘柄 の終値（日足でよい。内部で週足へ変換）
    lookback_weeks : int
        判定に使う直近週数

    Returns
    -------
    dict | None
        判定できない場合は None
    """
    if close_df is None or close_df.empty:
        return None

    # 日足 → 週次(金曜終値)へリサンプル
    try:
        wk = close_df.resample("W-FRI").last()
    except Exception:
        # indexがDatetimeでない場合は変換を試みる
        try:
            close_df = close_df.copy()
            close_df.index = pd.to_datetime(close_df.index)
            wk = close_df.resample("W-FRI").last()
        except Exception:
            return None

    wk = wk.dropna(how="all")
    if len(wk) < 8:
        return None

    # 週次リターン(%)
    wk_ret = wk.pct_change() * 100.0
    wk_ret = wk_ret.tail(lookback_weeks)

    # ---- 1. ラグ1自己相関 ----
    # (a) 時系列: 銘柄ごとに r_t と r_{t-1} の相関をとり平均
    ts_autocorrs = []
    for col in wk_ret.columns:
        s = wk_ret[col].dropna()
        if len(s) >= 6:
            ac = s.autocorr(lag=1)
            if ac is not None and not np.isnan(ac):
                ts_autocorrs.append(ac)
    ts_autocorr = float(np.mean(ts_autocorrs)) if ts_autocorrs else np.nan

    # (b) クロスセクション: 前週リターン順位と今週リターンの関係
    #     各週で corr(r_{t-1}[銘柄], r_t[銘柄]) を計算し平均
    cs_corrs = []
    cols = list(wk_ret.columns)
    if len(cols) >= 3:
        r = wk_ret[cols]
        for i in range(1, len(r)):
            prev = r.iloc[i - 1]
            cur = r.iloc[i]
            mask = prev.notna() & cur.notna()
            if mask.sum() >= 3:
                c = np.corrcoef(prev[mask].values, cur[mask].values)[0, 1]
                if not np.isnan(c):
                    cs_corrs.append(c)
    cs_autocorr = float(np.mean(cs_corrs)) if cs_corrs else np.nan

    # ---- 2. 戦略バックテスト（クロスセクション・週次リバランス） ----
    # 前週リターン中央値より上=勝ち組、下=負け組。
    # 順張り: 翌週、前週勝ち組の平均リターンを得る
    # 逆張り: 翌週、前週負け組の平均リターンを得る
    momo_returns, rev_returns = [], []
    if len(cols) >= 4:
        r = wk_ret[cols]
        for i in range(1, len(r)):
            prev = r.iloc[i - 1].dropna()
            cur = r.iloc[i]
            if len(prev) < 4:
                continue
            med = prev.median()
            winners = prev[prev > med].index
            losers = prev[prev < med].index
            w_cur = cur[winners].dropna()
            l_cur = cur[losers].dropna()
            if len(w_cur) > 0 and len(l_cur) > 0:
                momo_returns.append(float(w_cur.mean()))
                rev_returns.append(float(l_cur.mean()))

    n_weeks_bt = len(momo_returns)
    momo_avg = float(np.mean(momo_returns)) if momo_returns else np.nan
    rev_avg = float(np.mean(rev_returns)) if rev_returns else np.nan
    # 勝率: 順張りが逆張りを上回った週の割合
    momo_win_rate = (
        float(np.mean([m > rv for m, rv in zip(momo_returns, rev_returns)]) * 100)
        if n_weeks_bt > 0 else np.nan
    )

    # ---- 3. ボラティリティ・レジーム ----
    # 全銘柄平均の週次リターン標準偏差（直近13週 vs 全期間）
    port_ret = wk_ret.mean(axis=1).dropna()
    vol_recent = float(port_ret.tail(13).std()) if len(port_ret) >= 6 else np.nan
    vol_full = float(port_ret.std()) if len(port_ret) >= 6 else np.nan
    high_vol = (
        bool(vol_recent > vol_full * 1.15)
        if (not np.isnan(vol_recent) and not np.isnan(vol_full) and vol_full > 0)
        else False
    )

    # ---- 総合スコア ----
    # 正 → 順張り優勢 / 負 → 逆張り優勢
    score = 0.0
    weight_sum = 0.0
    if not np.isnan(ts_autocorr):
        score += np.tanh(ts_autocorr * 3) * 1.0
        weight_sum += 1.0
    if not np.isnan(cs_autocorr):
        score += np.tanh(cs_autocorr * 3) * 1.2
        weight_sum += 1.2
    if not np.isnan(momo_avg) and not np.isnan(rev_avg):
        # 順張り超過リターン(順張り-逆張り)を正規化
        diff = momo_avg - rev_avg
        score += np.tanh(diff / 2.0) * 1.5
        weight_sum += 1.5
    # 高ボラは逆張り方向にわずかにバイアス
    if high_vol:
        score -= 0.3
        weight_sum += 0.3

    norm_score = (score / weight_sum) if weight_sum > 0 else 0.0  # -1〜+1目安

    if norm_score >= 0.25:
        verdict = "順張り優勢"
        emoji = "📈"
        color = "#00c853"
        advice = "前週の勝ち組（上昇銘柄・強いテーマ）を買い増す順張りが機能しやすい地合い。トレンドフォロー向き。"
    elif norm_score <= -0.25:
        verdict = "逆張り優勢"
        emoji = "🔄"
        color = "#ff7043"
        advice = "前週の負け組（下落・売られすぎ銘柄）の反発を狙う逆張りが機能しやすい地合い。急落拾い・リバウンド狙い向き。"
    else:
        verdict = "中立（どちらも不安定）"
        emoji = "⚖️"
        color = "#ffb300"
        advice = "順張り・逆張りどちらも明確な優位がない地合い。ポジションを絞るか、個別材料・需給で選別推奨。"

    return {
        "verdict": verdict,
        "emoji": emoji,
        "color": color,
        "advice": advice,
        "score": norm_score,
        "ts_autocorr": ts_autocorr,
        "cs_autocorr": cs_autocorr,
        "momo_avg": momo_avg,
        "rev_avg": rev_avg,
        "momo_win_rate": momo_win_rate,
        "n_weeks_bt": n_weeks_bt,
        "n_weeks": len(wk_ret),
        "n_stocks": len(cols),
        "vol_recent": vol_recent,
        "vol_full": vol_full,
        "high_vol": high_vol,
    }


def analyze_entry_timing(ohlc_dict, lookback_days=120):
    """寄り(始値)買い vs 引け(終値)買い の勝率・平均リターンを比較する。

    日足OHLCから、場中ロングの代表的な3パターンを検証する:
      1. 寄り買い→当日引け売り (Open→Close, 日中/デイトレ)
      2. 引け買い→翌日引け売り (Close→翌Close, 翌日持ち越し)
      3. 引け買い→翌寄り売り   (Close→翌Open, オーバーナイトのみ)

    Parameters
    ----------
    ohlc_dict : dict[str, pd.DataFrame]
        {銘柄: OHLC DataFrame(Open/High/Low/Close列を含む)}
    lookback_days : int
        各銘柄で検証する直近営業日数

    Returns
    -------
    dict | None
    """
    if not ohlc_dict:
        return None

    intraday = []   # 当日 Open→Close (%)
    overnight = []  # 前日Close→当日Open (%)
    nextday = []    # 前日Close→当日Close (%)

    n_stocks_used = 0
    for _code, df in ohlc_dict.items():
        if df is None or df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        if not all(k in cols for k in ("open", "close")):
            continue
        o = pd.to_numeric(df[cols["open"]], errors="coerce")
        c = pd.to_numeric(df[cols["close"]], errors="coerce")
        o = o.tail(lookback_days)
        c = c.tail(lookback_days)
        if len(c) < 20:
            continue
        n_stocks_used += 1

        # 1. 寄り→引け（当日）
        intr = (c / o - 1.0) * 100.0
        intraday.extend(intr.dropna().tolist())

        # 2. 前日引け→当日寄り（オーバーナイト）
        on = (o / c.shift(1) - 1.0) * 100.0
        overnight.extend(on.dropna().tolist())

        # 3. 前日引け→当日引け（翌日持ち越し）
        nd = (c / c.shift(1) - 1.0) * 100.0
        nextday.extend(nd.dropna().tolist())

    if n_stocks_used == 0 or not intraday:
        return None

    def _stats(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return {"win_rate": np.nan, "avg": np.nan, "median": np.nan, "n": 0}
        return {
            "win_rate": float((a > 0).mean() * 100),
            "avg": float(a.mean()),
            "median": float(np.median(a)),
            "n": int(len(a)),
        }

    s_intraday = _stats(intraday)    # 寄り買い当日決済
    s_overnight = _stats(overnight)  # 引け買い→翌寄り
    s_nextday = _stats(nextday)      # 引け買い→翌引け

    # 寄り買い(当日Open→Close) と 引け買い(Close→翌Close) の優劣判定
    verdict = "判定不能"
    emoji = "⚖️"
    color = "#ffb300"
    advice = ""
    wr_open = s_intraday["win_rate"]
    wr_close = s_nextday["win_rate"]
    avg_open = s_intraday["avg"]
    avg_close = s_nextday["avg"]
    if not np.isnan(wr_open) and not np.isnan(wr_close):
        # 勝率と平均リターンの両方を見て総合判定
        open_score = (wr_open - 50) + avg_open * 10
        close_score = (wr_close - 50) + avg_close * 10
        if open_score - close_score > 3:
            verdict = "寄り(始値)買いが有利"
            emoji = "🌅"
            color = "#00c853"
            advice = ("日中（寄り→引け）の上昇が優勢。場中ロングは寄りで入り、当日の値幅を取る戦略が機能しやすい地合い。"
                      "引け持ち越しより日中回転が有利な傾向。")
        elif close_score - open_score > 3:
            verdict = "引け(終値)買いが有利"
            emoji = "🌙"
            color = "#42a5f5"
            advice = ("引けで買って翌日以降に持ち越す方が優勢。オーバーナイト〜翌日の上昇が日中上昇を上回る地合い。"
                      "寄り天・日中ダレの傾向があるため、引けエントリーが機能しやすい。")
        else:
            verdict = "寄り・引けで優位差は小さい"
            emoji = "⚖️"
            color = "#ffb300"
            advice = "寄り買いと引け買いで明確な優劣はなし。日中・持ち越しどちらも大きな偏りがない地合い。"

    return {
        "verdict": verdict,
        "emoji": emoji,
        "color": color,
        "advice": advice,
        "intraday": s_intraday,     # 寄り買い→当日引け
        "overnight": s_overnight,   # 引け買い→翌寄り
        "nextday": s_nextday,       # 引け買い→翌引け
        "n_stocks": n_stocks_used,
        "lookback_days": lookback_days,
    }


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
    include_intraday = st.toggle(
        "場中データを含める（当日ザラ場）", value=False,
        help="OFF（推奨）: 確定した終値ベースでランキング。"
             "ON: 当日ザラ場の途中値も含む（大引けまで変動しブレやすい）",
    )

# データ取得
with st.spinner("📊 株価データ取得中..."):
    theme_ret_df, theme_cum, price_meta = build_theme_index(
        period=analysis_period, include_intraday=include_intraday
    )

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18 = st.tabs([
    "📊 テーマランキング", "📈 累積リターン",
    "🔬 テーマ詳細", "🌏 日米リードラグ", "🎯 売買シグナル",
    "🔥 HOTニュース", "📋 ポートフォリオ", "🐉 ドラゴンキング",
    "📢 出来高急増", "📊 TOPIX新基準", "🏛️ PBR改革",
    "🦈 アクティビスト", "👁️ ウォッチリスト", "📰 決算速報",
    "📅 決算カレンダー", "🏭 業種別決算選別", "🤖 ML要因分析",
    "📄 適時開示"
])

with tab1:
    st.subheader("テーマ別リターンランキング")

    # ランキングの基準日・データ状態を明示（日次ランキングのズレ防止）
    _last_dt = price_meta.get("last_date")
    _last_str = pd.Timestamp(_last_dt).strftime("%Y/%m/%d") if _last_dt is not None else "—"
    if price_meta.get("is_intraday") and include_intraday:
        st.warning(
            f"⚠️ 日次リターンは**当日ザラ場（未確定）**の値を含みます（{_last_str}）。"
            "大引けまで変動するため順位は暫定です。確定値で見るにはサイドバーの"
            "『場中データを含める』をOFFにしてください。"
        )
    elif price_meta.get("intraday_dropped"):
        st.info(
            f"📅 日次ランキングは確定した最新営業日（**{_last_str}**）ベースです。"
            "当日ザラ場は未確定のため除外しています（サイドバーで切替可）。"
        )
    else:
        st.caption(f"📅 基準日: {_last_str}（確定終値ベース）")

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

    # ── みんかぶ テーマランキング（外部ソースで多角的に分析） ──
    st.divider()
    st.markdown("#### 🌐 みんかぶ テーマランキング（外部ソース）")
    st.caption(
        "みんかぶが集計する市場全体のテーマランキング。自前の登録テーマだけでなく、"
        "市場で実際に物色されているテーマを横断的に把握できます。"
        "（出典: [みんかぶ テーマ](https://minkabu.jp/theme/) ／ 約30分キャッシュ）"
    )

    mk_col1, mk_col2 = st.columns([1, 1])
    with mk_col1:
        mk_refresh = st.button("🔄 みんかぶランキングを更新", key="mk_refresh_btn")
    if mk_refresh:
        fetch_minkabu_theme_ranking.clear()
        fetch_minkabu_all_rankings.clear()

    with st.spinner("みんかぶ テーマランキング取得中..."):
        mk_df = fetch_minkabu_all_rankings()

    if mk_df is None or mk_df.empty:
        st.info("みんかぶのテーマランキングを取得できませんでした。時間をおいて『更新』を試してください。")
    else:
        # 登録テーマ（THEMES）と重なるテーマにマークを付ける
        _registered = set(THEMES.keys())
        mk_df = mk_df.copy()
        mk_df["登録済"] = mk_df["テーマ"].apply(lambda t: "✅" if t in _registered else "")

        mk_tab_rise, mk_tab_pop, mk_tab_cross, mk_tab_exp, mk_tab_early = st.tabs(
            ["📈 急上昇テーマ", "🔥 人気テーマ", "🔀 クロス分析",
             "🚀 上昇期待スコア", "🌱 上昇初動検出"]
        )

        def _render_minkabu_table(df_kind, sort_by_change=True):
            if df_kind.empty:
                st.write("データなし")
                return
            disp = df_kind.copy()
            if sort_by_change:
                disp = disp.sort_values("変動率(%)", ascending=False)
            disp = disp[[
                "順位", "テーマ", "変動率(%)", "関連銘柄数",
                "代表銘柄コード", "代表銘柄名", "登録済", "テーマURL"
            ]]
            st.dataframe(
                disp,
                use_container_width=True,
                hide_index=True,
                height=520,
                column_config={
                    "変動率(%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "テーマURL": st.column_config.LinkColumn("リンク", display_text="開く"),
                },
            )
            # 変動率の横棒グラフ
            plot_df = df_kind.sort_values("変動率(%)", ascending=True)
            fig_mk = px.bar(
                plot_df, x="変動率(%)", y="テーマ", orientation="h",
                color="変動率(%)", color_continuous_scale="RdYlGn",
                title="テーマ別 株価変動率",
            )
            fig_mk.update_layout(
                height=max(400, len(plot_df) * 26),
                template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_mk, use_container_width=True)

        with mk_tab_rise:
            _render_minkabu_table(mk_df[mk_df["ランキング種別"] == "急上昇"])

        with mk_tab_pop:
            _render_minkabu_table(mk_df[mk_df["ランキング種別"] == "人気"])

        with mk_tab_cross:
            st.markdown("##### 🔀 人気 × 急上昇 クロス分析")
            st.caption("両方のランキングに登場するテーマ = 注目度と勢いを兼ね備えたテーマ")
            rise = mk_df[mk_df["ランキング種別"] == "急上昇"][["テーマ", "順位", "変動率(%)"]]
            pop = mk_df[mk_df["ランキング種別"] == "人気"][["テーマ", "順位", "変動率(%)"]]
            merged = pd.merge(
                rise, pop, on="テーマ", how="inner",
                suffixes=("_急上昇", "_人気")
            )
            if merged.empty:
                st.info("両ランキングに共通するテーマはありませんでした。")
            else:
                merged["合計順位"] = merged["順位_急上昇"] + merged["順位_人気"]
                merged = merged.sort_values("合計順位")
                merged_disp = merged[[
                    "テーマ", "順位_人気", "順位_急上昇", "変動率(%)_急上昇"
                ]].rename(columns={"変動率(%)_急上昇": "変動率(%)"})
                st.dataframe(
                    merged_disp, use_container_width=True, hide_index=True,
                    column_config={
                        "変動率(%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )
                for _, r in merged.head(5).iterrows():
                    st.markdown(
                        f"- **{r['テーマ']}** — 人気{int(r['順位_人気'])}位 / "
                        f"急上昇{int(r['順位_急上昇'])}位 / "
                        f"変動率 {r['変動率(%)_急上昇']:+.2f}%"
                    )

        with mk_tab_exp:
            st.markdown("##### 🚀 上昇期待スコア（代表銘柄の値動き × 注目度）")
            st.caption(
                "みんかぶの当日変動率だけでなく、各テーマ代表銘柄の実際の値動き"
                "（週次・月次モメンタム、52週高値からの位置、出来高、RSI）と、"
                "人気・急上昇ランクインの注目度を合成した0〜100のスコアです。"
                "高スコアほど上昇パフォーマンスが期待しやすい状態を示します。"
            )
            with st.spinner("代表銘柄の値動きを取得してスコア算出中..."):
                exp_df = build_minkabu_expectation(mk_df)

            if exp_df is None or exp_df.empty:
                st.info("スコアを算出できませんでした。")
            else:
                # トップ3カード
                top3 = exp_df.head(3)
                ccols = st.columns(len(top3)) if len(top3) > 0 else []
                for ci, (_, r) in enumerate(top3.iterrows()):
                    with ccols[ci]:
                        w_str = f"{r['週次(%)']:+.1f}%" if pd.notna(r.get("週次(%)")) else "—"
                        st.metric(
                            f"{r['評価']} {r['テーマ']}",
                            f"{r['期待スコア']:.0f} 点",
                            help=f"代表: {r.get('代表銘柄名') or '—'}",
                        )
                        st.caption(f"週次 {w_str}")

                # 象限マップ: x=月次モメンタム, y=注目度(期待スコア), 色=当日変動率, サイズ=関連銘柄数
                plot_df = exp_df.copy()
                plot_df["月次(%)"] = pd.to_numeric(plot_df["月次(%)"], errors="coerce")
                plot_df["関連銘柄数_size"] = pd.to_numeric(
                    plot_df["関連銘柄数"], errors="coerce"
                ).fillna(10).clip(lower=5)
                plot_df = plot_df.dropna(subset=["月次(%)", "期待スコア"])

                if not plot_df.empty:
                    x_mid = float(plot_df["月次(%)"].median())
                    fig_exp = px.scatter(
                        plot_df, x="月次(%)", y="期待スコア",
                        size="関連銘柄数_size", text="テーマ",
                        color="当日変動率", color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        hover_data={
                            "週次(%)": ":+.1f", "高値比": ":.0f",
                            "RSI": ":.0f", "関連銘柄数_size": False,
                        },
                        title="テーマ別 上昇期待マップ（右上＝勢い強×注目度高）",
                    )
                    fig_exp.update_traces(textposition="top center", textfont_size=9)
                    # 基準線（右上ゾーンを強調）
                    fig_exp.add_hline(y=55, line_dash="dash", line_color="#888", opacity=0.5)
                    fig_exp.add_vline(x=x_mid, line_dash="dash", line_color="#888", opacity=0.5)
                    fig_exp.update_layout(
                        height=600, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)
                    st.caption(
                        "📍 右上ゾーン（月次プラス × 期待スコア55以上）= 勢いと注目を兼ね備えた本命候補。"
                        "色が緑なほど当日も買われています。バブルの大きさは関連銘柄数。"
                    )

                # スコアランキング表
                show_cols = [
                    "評価", "テーマ", "期待スコア", "当日変動率", "週次(%)", "月次(%)",
                    "高値比", "RSI", "関連銘柄数", "代表銘柄名", "テーマURL"
                ]
                show_cols = [c for c in show_cols if c in exp_df.columns]
                st.dataframe(
                    exp_df[show_cols],
                    use_container_width=True, hide_index=True, height=520,
                    column_config={
                        "期待スコア": st.column_config.ProgressColumn(
                            "期待スコア", min_value=0, max_value=100, format="%.0f"
                        ),
                        "当日変動率": st.column_config.NumberColumn(format="%+.2f%%"),
                        "週次(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                        "月次(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                        "高値比": st.column_config.NumberColumn(format="%.0f"),
                        "RSI": st.column_config.NumberColumn(format="%.0f"),
                        "テーマURL": st.column_config.LinkColumn("リンク", display_text="開く"),
                    },
                )
                st.caption(
                    "スコア内訳: 勢い(週次・月次)45点 + 当日変動率15点 + 押し目余地15点 + 注目度25点。"
                    "高値比は52週レンジ内の位置(100=高値)。RSIは70超で過熱、30未満で売られ過ぎの目安。"
                )

                # ── 上昇理由の分析（最新ニュースから材料抽出） ──
                st.markdown("---")
                st.markdown("##### 🔎 なぜ上昇している？ — 上位テーマの材料分析")
                st.caption(
                    "上位テーマの直近45日のニュース見出しから、ポジティブ材料を分類抽出します。"
                    "（出典: Google ニュース検索。実際の見出しとリンクを根拠として表示）"
                )
                _top_themes = exp_df.head(10)["テーマ"].tolist()
                default_pick = _top_themes[:3]
                picked = st.multiselect(
                    "材料を分析するテーマ（期待スコア上位10件から選択）",
                    _top_themes, default=default_pick, key="mk_reason_pick"
                )
                if st.button("📰 選択テーマの上昇理由を分析", key="mk_reason_btn"):
                    for th in picked:
                        _r = exp_df[exp_df["テーマ"] == th].iloc[0]
                        with st.spinner(f"「{th}」の材料を分析中..."):
                            reason = fetch_theme_uptrend_reason(th)

                        # ヘッダ行
                        _w = f"{_r['週次(%)']:+.1f}%" if pd.notna(_r.get("週次(%)")) else "—"
                        _m = f"{_r['月次(%)']:+.1f}%" if pd.notna(_r.get("月次(%)")) else "—"
                        with st.expander(
                            f"{_r['評価']} {th} ｜ スコア{_r['期待スコア']:.0f} ｜ "
                            f"週次{_w} 月次{_m}",
                            expanded=True,
                        ):
                            st.markdown(f"**📝 推定材料:** {reason['summary']}")
                            pos, neg = reason["pos"], reason["neg"]
                            if pos or neg:
                                st.caption(
                                    f"直近ニュースのポジティブ見出し {pos}件 / ネガティブ {neg}件"
                                )

                            # 材料カテゴリのバッジ
                            if reason["categories"]:
                                badge_html = ""
                                for cat_name, cnt, examples in reason["categories"]:
                                    badge_html += (
                                        f"<span style='display:inline-block;margin:2px 4px;"
                                        f"padding:3px 10px;background:#1b3a2b;color:#7CFC9E;"
                                        f"border-radius:10px;font-size:0.8rem;'>"
                                        f"{cat_name} ×{cnt}</span>"
                                    )
                                st.markdown(badge_html, unsafe_allow_html=True)

                                # 代表的な材料カテゴリの見出し例
                                top_cat = reason["categories"][0]
                                if top_cat[2]:
                                    st.markdown(f"**主な材料（{top_cat[0]}）:**")
                                    for ex in top_cat[2]:
                                        st.markdown(f"- {ex}")
                            else:
                                st.info(
                                    "明確な個別材料は検出されませんでした。"
                                    "テーマ全体の地合い・資金流入・需給要因による上昇の可能性があります。"
                                )

                            # 根拠ニュース見出し（リンク付き）
                            if reason["headlines"]:
                                st.markdown("**🗞️ 根拠ニュース（直近）:**")
                                for title, link, pub, pol in reason["headlines"][:6]:
                                    mark = "🟢" if pol > 0 else ("🔴" if pol < 0 else "⚪")
                                    date_str = f"<span style='opacity:0.6'>{pub}</span> " if pub else ""
                                    st.markdown(
                                        f"{mark} {date_str}"
                                        f"<a href='{link}' target='_blank' "
                                        f"style='color:#4da6ff;text-decoration:none;'>{title}</a>",
                                        unsafe_allow_html=True,
                                    )
                    st.caption(
                        "⚠️ ニュース見出しベースの自動分類です。投資判断の前に必ず一次情報（適時開示・決算短信）をご確認ください。"
                    )

        with mk_tab_early:
            st.markdown("##### 🌱 上昇初動検出 — まだ上がっていないが、これから上がりそうな銘柄")
            st.caption(
                "本格上昇前（未過熱）でありながら、出来高増・下値切り上げ・移動平均の上向き転換・"
                "値動き収縮などトレンド転換の予兆が揃った銘柄を検出します。"
                "既に大きく上昇済みの銘柄（月次+15%超・52週高値圏）は初動対象外として除外します。"
            )

            # 対象銘柄の選択
            early_scope = st.radio(
                "検出対象",
                ["注目テーマの銘柄（みんかぶ上位テーマ）", "登録テーマ全銘柄", "特定テーマ"],
                horizontal=True, key="early_scope",
            )

            early_code_name = {}  # {"6501": "日立製作所", ...}（.Tなし）
            if early_scope == "特定テーマ":
                _pick_theme = st.selectbox("テーマ選択", list(THEMES.keys()), key="early_theme_pick")
                for tk, nm in THEMES[_pick_theme]["stocks"].items():
                    early_code_name[tk.replace(".T", "")] = nm
            elif early_scope == "登録テーマ全銘柄":
                for _info in THEMES.values():
                    for tk, nm in _info["stocks"].items():
                        early_code_name[tk.replace(".T", "")] = nm
            else:
                # みんかぶ上位テーマの登録テーマ銘柄を対象にする
                _mk_themes = [t for t in mk_df["テーマ"].unique() if t in _registered]
                for t in _mk_themes:
                    for tk, nm in THEMES[t]["stocks"].items():
                        early_code_name[tk.replace(".T", "")] = nm
                # 加えて、みんかぶ代表銘柄も対象に含める
                for _, rr in mk_df.iterrows():
                    c = rr.get("代表銘柄コード")
                    if c:
                        early_code_name.setdefault(str(c), rr.get("代表銘柄名") or str(c))

            st.caption(f"検出対象: {len(early_code_name)}銘柄")

            if st.button("🔍 上昇初動を検出", key="early_run_btn"):
                with st.spinner("各銘柄の値動きを分析して初動スコアを算出中...（銘柄数により時間がかかります）"):
                    early_df = fetch_early_stage_signals(
                        tuple(sorted(early_code_name.keys())), early_code_name
                    )
                st.session_state["early_df"] = early_df

            early_df = st.session_state.get("early_df")
            if early_df is None:
                st.info("『上昇初動を検出』を押すと分析を開始します。")
            elif early_df.empty:
                st.warning("初動条件を満たす銘柄が見つかりませんでした。対象を広げて再試行してください。")
            else:
                st.success(f"✅ {len(early_df)}銘柄を初動候補として検出（スコア降順）")

                # 上位カード（寄与理由付き）
                topn = early_df.head(6)
                for i in range(0, len(topn), 2):
                    ccols = st.columns(2)
                    for j, (_, r) in enumerate(topn.iloc[i:i+2].iterrows()):
                        with ccols[j]:
                            rank = i + j + 1
                            st.markdown(
                                f"**{rank}. {r['銘柄名']}** "
                                f"<span style='color:#7CFC9E;font-weight:bold;'>初動スコア {r['初動スコア']:.0f}</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='font-size:0.82rem;opacity:0.85;'>"
                                f"📊 週次{r['週次(%)']:+.1f}% / 月次{r['月次(%)']:+.1f}% / "
                                f"出来高比{r['出来高比']:.1f}倍 / RSI{r['RSI']:.0f} / "
                                f"52週位置{r['52週位置']:.0f}%</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='font-size:0.82rem;color:#9ecbff;margin-top:2px;'>"
                                f"🧩 判定根拠: {r['主な根拠']}</div>",
                                unsafe_allow_html=True,
                            )

                # 寄与度の内訳を可視化（上位銘柄の積み上げ棒）
                st.markdown("###### 🧩 初動スコアの寄与度内訳（上位10銘柄）")
                contrib_keys = ["未過熱度", "出来高増加", "下値切り上げ",
                                "MA転換", "位置余地", "RSI適正", "ボラ収縮"]
                stack_rows = []
                for _, r in early_df.head(10).iterrows():
                    c = r["寄与"]
                    for k in contrib_keys:
                        stack_rows.append({
                            "銘柄": r["銘柄名"], "寄与要素": k, "スコア": c.get(k, 0)
                        })
                stack_df = pd.DataFrame(stack_rows)
                fig_stack = px.bar(
                    stack_df, x="スコア", y="銘柄", color="寄与要素",
                    orientation="h", title="初動スコアの寄与度内訳（何が効いているか）",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig_stack.update_layout(
                    height=max(400, len(early_df.head(10)) * 42),
                    template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(fig_stack, use_container_width=True)

                # 明細テーブル
                show_cols = ["銘柄名", "初動スコア", "主な根拠", "現在値",
                             "週次(%)", "月次(%)", "3ヶ月(%)", "出来高比", "RSI", "52週位置"]
                show_cols = [c for c in show_cols if c in early_df.columns]
                st.dataframe(
                    early_df[show_cols],
                    use_container_width=True, hide_index=True, height=460,
                    column_config={
                        "初動スコア": st.column_config.ProgressColumn(
                            "初動スコア", min_value=0, max_value=100, format="%.0f"
                        ),
                        "週次(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                        "月次(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                        "3ヶ月(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                        "出来高比": st.column_config.NumberColumn(format="%.2f"),
                        "RSI": st.column_config.NumberColumn(format="%.0f"),
                        "52週位置": st.column_config.NumberColumn(format="%.0f"),
                    },
                )
                st.caption(
                    "寄与要素: 未過熱度22 / 出来高増加22 / 下値切り上げ16 / MA転換18 / "
                    "位置余地12 / RSI適正10 / ボラ収縮5（満点合計105、実質100目安）。"
                    "『主な根拠』は各銘柄でスコアに最も効いた上位3要素です。"
                    "⚠️ 予兆分析であり上昇を保証しません。一次情報と併せてご判断ください。"
                )

        # 登録テーマと市場ランキングの重なりサマリ
        _overlap = sorted(set(mk_df["テーマ"]) & _registered)
        if _overlap:
            st.success(
                f"登録テーマのうち {len(_overlap)} 件がみんかぶランキングにランクイン中: "
                + "、".join(_overlap)
            )

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

    # =================================================================
    # 相関ペア抽出（連動テーマ / 逆相関テーマ）
    #   - 全期間相関と直近20日相関を併記し、日々の連動を意識しやすくする
    # =================================================================
    st.markdown("---")
    st.subheader("🔗 連動・逆相関テーマの抽出")
    st.caption(
        "過去の日次リターンから、値動きが似ているテーマ（連動）と逆に動くテーマ（逆相関）を抽出します。"
        "全期間相関に加え、直近20営業日の短期相関も併記し、足もとの連動状態を把握できます。"
    )

    if len(corr.columns) >= 2 and len(theme_ret_filtered) >= 10:
        # 直近20日（データが少なければ全期間）の短期相関
        _recent_n = min(20, len(theme_ret_filtered))
        _corr_recent = theme_ret_filtered.iloc[-_recent_n:].corr()

        # 上三角のペアを列挙（自己相関と重複を除外）
        _pairs = []
        _cols = list(corr.columns)
        for _i in range(len(_cols)):
            for _j in range(_i + 1, len(_cols)):
                _a, _b = _cols[_i], _cols[_j]
                _c_all = corr.iloc[_i, _j]
                _c_rec = _corr_recent.loc[_a, _b] if (_a in _corr_recent.index and _b in _corr_recent.columns) else float("nan")
                if pd.isna(_c_all):
                    continue
                _pairs.append({
                    "テーマA": _a, "テーマB": _b,
                    "全期間相関": round(float(_c_all), 2),
                    f"直近{_recent_n}日相関": round(float(_c_rec), 2) if pd.notna(_c_rec) else None,
                })
        _pairs_df = pd.DataFrame(_pairs)

        if not _pairs_df.empty:
            cL, cR = st.columns(2)

            # 連動（正相関が高い）トップ
            with cL:
                st.markdown("**🟢 連動しやすいテーマ（正相関トップ）**")
                _top_pos = _pairs_df.sort_values("全期間相関", ascending=False).head(10)
                st.dataframe(
                    _top_pos.style.format({
                        "全期間相関": "{:+.2f}", f"直近{_recent_n}日相関": "{:+.2f}"
                    }).background_gradient(subset=["全期間相関"], cmap="Greens", vmin=0, vmax=1),
                    use_container_width=True, hide_index=True)
                if not _top_pos.empty:
                    _t = _top_pos.iloc[0]
                    st.caption(
                        f"最も連動: **{_t['テーマA']} ↔ {_t['テーマB']}**（{_t['全期間相関']:+.2f}）。"
                        "一方が動くともう一方も同方向に動きやすい＝分散効果は小さい。")

            # 逆相関（負相関が強い）トップ
            with cR:
                st.markdown("**🔴 逆に動きやすいテーマ（逆相関トップ）**")
                _top_neg = _pairs_df.sort_values("全期間相関", ascending=True).head(10)
                st.dataframe(
                    _top_neg.style.format({
                        "全期間相関": "{:+.2f}", f"直近{_recent_n}日相関": "{:+.2f}"
                    }).background_gradient(subset=["全期間相関"], cmap="Reds_r", vmin=-1, vmax=0),
                    use_container_width=True, hide_index=True)
                if not _top_neg.empty:
                    _t = _top_neg.iloc[0]
                    st.caption(
                        f"最も逆相関: **{_t['テーマA']} ↔ {_t['テーマB']}**（{_t['全期間相関']:+.2f}）。"
                        "一方が上げるともう一方が下げやすい＝ヘッジ・分散に有効だった組み合わせ。")

            # --- 日々意識するためのサマリー（本日の連動状態） ---
            st.markdown("**📅 本日の連動状態（直近リターンの符号で判定）**")
            _last_ret = theme_ret_filtered.iloc[-1]
            _last_date = theme_ret_filtered.index[-1].strftime("%Y/%m/%d")
            # 強い相関/逆相関ペアについて、直近リターンが想定通りか逸脱かを表示
            _watch = []
            for _, _row in _pairs_df.iterrows():
                _ca = _row["全期間相関"]
                if abs(_ca) < 0.5:
                    continue  # 弱い相関は対象外
                _ra = _last_ret.get(_row["テーマA"])
                _rb = _last_ret.get(_row["テーマB"])
                if _ra is None or _rb is None or pd.isna(_ra) or pd.isna(_rb):
                    continue
                _same_dir = (_ra >= 0) == (_rb >= 0)
                if _ca >= 0.5:
                    # 正相関ペア: 同方向なら想定通り、逆方向なら「連動崩れ」
                    _status = "✅ 連動通り" if _same_dir else "⚠️ 連動崩れ（乖離）"
                    _kind = "正相関"
                else:
                    # 逆相関ペア: 逆方向なら想定通り、同方向なら「逆相関崩れ」
                    _status = "✅ 逆相関通り" if not _same_dir else "⚠️ 逆相関崩れ"
                    _kind = "逆相関"
                _watch.append({
                    "関係": _kind,
                    "テーマA": _row["テーマA"], "A当日(%)": round(_ra * 100, 2),
                    "テーマB": _row["テーマB"], "B当日(%)": round(_rb * 100, 2),
                    "全期間相関": _ca,
                    "本日の状態": _status,
                })
            if _watch:
                _watch_df = pd.DataFrame(_watch).sort_values(
                    "全期間相関", key=lambda s: s.abs(), ascending=False)
                st.caption(f"判定日: {_last_date}（直近営業日の各テーマ当日リターンの符号で判定）")
                st.dataframe(
                    _watch_df.style.format({
                        "A当日(%)": "{:+.2f}", "B当日(%)": "{:+.2f}", "全期間相関": "{:+.2f}"
                    }),
                    use_container_width=True, hide_index=True)
                st.caption(
                    "『連動崩れ』『逆相関崩れ』は、過去の関係から外れた動き＝"
                    "資金の潮目が変わった可能性のサイン。順張り・乖離の手がかりになります。")
            else:
                st.info("強い相関（|相関|≥0.5）のペアがないため、本日の連動状態は表示していません。")
    else:
        st.info("相関ペア抽出には2テーマ以上・10営業日以上のデータが必要です。")

    # =================================================================
    # 過去リターンのフィードバック（実績サマリー）
    # ※ すべて選択期間の過去データに基づく実績であり、将来を保証しない
    # =================================================================
    st.markdown("---")
    st.subheader("📊 過去リターン実績サマリー（テーマ別）")
    st.caption(
        f"分析期間: {theme_ret_filtered.index[0].strftime('%Y/%m/%d')} 〜 "
        f"{theme_ret_filtered.index[-1].strftime('%Y/%m/%d')}（{len(theme_ret_filtered)}営業日）。"
        "以下はすべて過去の実績統計です。将来の成績を保証するものではありません。"
    )

    _ann = 252  # 年率換算の営業日数
    _summary_rows = []
    for _th in theme_ret_filtered.columns:
        _r = theme_ret_filtered[_th].dropna()
        if len(_r) < 5:
            continue
        _cum_total = (1 + _r).prod() - 1
        _ann_ret = (1 + _r.mean()) ** _ann - 1
        _ann_vol = _r.std() * (_ann ** 0.5)
        _sharpe = (_ann_ret / _ann_vol) if _ann_vol > 0 else float("nan")
        # 最大ドローダウン
        _cum_curve = (1 + _r).cumprod()
        _dd = (_cum_curve / _cum_curve.cummax() - 1).min()
        _win = (_r > 0).mean() * 100
        _best = _r.max() * 100
        _worst = _r.min() * 100
        _summary_rows.append({
            "テーマ": _th,
            "累積リターン(%)": round(_cum_total * 100, 1),
            "年率リターン(%)": round(_ann_ret * 100, 1),
            "年率ボラ(%)": round(_ann_vol * 100, 1),
            "シャープレシオ": round(_sharpe, 2) if _sharpe == _sharpe else None,
            "最大DD(%)": round(_dd * 100, 1),
            "勝率(%)": round(_win, 1),
            "最大上昇日(%)": round(_best, 2),
            "最大下落日(%)": round(_worst, 2),
        })

    if _summary_rows:
        _summary_df = pd.DataFrame(_summary_rows).sort_values("累積リターン(%)", ascending=False)
        st.dataframe(
            _summary_df.style.format({
                "累積リターン(%)": "{:+.1f}", "年率リターン(%)": "{:+.1f}",
                "年率ボラ(%)": "{:.1f}", "シャープレシオ": "{:.2f}",
                "最大DD(%)": "{:.1f}", "勝率(%)": "{:.1f}",
                "最大上昇日(%)": "{:+.2f}", "最大下落日(%)": "{:+.2f}",
            }).background_gradient(subset=["累積リターン(%)", "シャープレシオ"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True
        )
        st.caption(
            "シャープレシオ=年率リターン÷年率ボラ（リスク対比の効率。高いほど良い）。"
            "最大DD=期間中の高値からの最大下落幅。勝率=日次リターンがプラスだった割合。"
        )

    # =================================================================
    # セクター（テーマ）別の過去の有利なエントリー傾向
    #   (A) 季節性: 曜日別・月別の平均リターン
    #   (B) 押し目: 20日移動平均からの乖離帯ごとの「その後5日リターン」
    # =================================================================
    st.markdown("---")
    st.subheader("⏱️ 過去データで有利だったエントリー傾向（テーマ別）")
    st.caption(
        "※ 過去の統計的傾向であり、将来の優位性を保証しません。相場環境の変化で傾向は失われ得ます。"
    )

    _entry_theme = st.selectbox(
        "テーマを選択", list(theme_ret_filtered.columns), key="entry_timing_theme")

    if _entry_theme:
        _er = theme_ret_filtered[_entry_theme].dropna()

        colA, colB = st.columns(2)

        # (A-1) 曜日別平均リターン
        with colA:
            st.markdown("**曜日別 平均リターン**")
            _dow_map = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}
            _dow = _er.groupby(_er.index.dayofweek).agg(["mean", "count"])
            _dow = _dow[_dow.index <= 4]
            if not _dow.empty:
                _dow_df = pd.DataFrame({
                    "曜日": [_dow_map.get(i, str(i)) for i in _dow.index],
                    "平均リターン(%)": (_dow["mean"] * 100).round(3).values,
                    "日数": _dow["count"].values,
                })
                _fig_dow = px.bar(
                    _dow_df, x="曜日", y="平均リターン(%)", text="平均リターン(%)",
                    color="平均リターン(%)", color_continuous_scale="RdYlGn",
                    title="曜日別の平均日次リターン")
                _fig_dow.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117", height=320, showlegend=False)
                st.plotly_chart(_fig_dow, use_container_width=True)
                _best_dow = _dow_df.loc[_dow_df["平均リターン(%)"].idxmax()]
                st.caption(
                    f"過去は **{_best_dow['曜日']}曜** の平均が最も高い"
                    f"（{_best_dow['平均リターン(%)']:+.3f}% / {int(_best_dow['日数'])}日）")

        # (A-2) 月別平均リターン
        with colB:
            st.markdown("**月別 平均リターン**")
            _mon = _er.groupby(_er.index.month).agg(["mean", "count"])
            if not _mon.empty:
                _mon_df = pd.DataFrame({
                    "月": [f"{i}月" for i in _mon.index],
                    "平均リターン(%)": (_mon["mean"] * 100).round(3).values,
                    "日数": _mon["count"].values,
                })
                _fig_mon = px.bar(
                    _mon_df, x="月", y="平均リターン(%)", text="平均リターン(%)",
                    color="平均リターン(%)", color_continuous_scale="RdYlGn",
                    title="月別の平均日次リターン")
                _fig_mon.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117", height=320, showlegend=False)
                st.plotly_chart(_fig_mon, use_container_width=True)
                _best_mon = _mon_df.loc[_mon_df["平均リターン(%)"].idxmax()]
                st.caption(
                    f"過去は **{_best_mon['月']}** の平均が最も高い"
                    f"（{_best_mon['平均リターン(%)']:+.3f}% / {int(_best_mon['日数'])}日）")

        # (B) 押し目分析: 20日MA乖離率の帯ごとに「その後5日リターン」を集計
        st.markdown("**押し目エントリー分析（20日移動平均からの乖離 → その後5日リターン）**")
        _cum_idx = (1 + _er).cumprod()  # 擬似価格（指数化）
        _ma20 = _cum_idx.rolling(20).mean()
        _dev = (_cum_idx - _ma20) / _ma20 * 100  # 乖離率(%)
        _fwd5 = _cum_idx.shift(-5) / _cum_idx - 1  # その後5日リターン
        _bt = pd.DataFrame({"乖離率": _dev, "その後5日": _fwd5 * 100}).dropna()

        if len(_bt) >= 20:
            _bins = [-100, -10, -5, -2, 2, 5, 10, 100]
            _labels = ["-10%以下", "-10〜-5%", "-5〜-2%", "-2〜+2%",
                       "+2〜+5%", "+5〜+10%", "+10%以上"]
            _bt["帯"] = pd.cut(_bt["乖離率"], bins=_bins, labels=_labels)
            _grp = _bt.groupby("帯", observed=True)["その後5日"].agg(["mean", "median", "count"])
            _grp = _grp[_grp["count"] >= 3]  # サンプル3件未満は除外
            if not _grp.empty:
                _grp_df = _grp.reset_index()
                _grp_df.columns = ["乖離帯", "平均その後5日(%)", "中央値(%)", "サンプル数"]
                _fig_pb = px.bar(
                    _grp_df, x="乖離帯", y="平均その後5日(%)", text="平均その後5日(%)",
                    color="平均その後5日(%)", color_continuous_scale="RdYlGn",
                    title="20日MA乖離帯ごとの『その後5日』平均リターン")
                _fig_pb.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117", height=360, showlegend=False)
                _fig_pb.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(_fig_pb, use_container_width=True)
                st.dataframe(
                    _grp_df.style.format({
                        "平均その後5日(%)": "{:+.2f}", "中央値(%)": "{:+.2f}",
                        "サンプル数": "{:.0f}"}),
                    use_container_width=True, hide_index=True)
                _best_band = _grp_df.loc[_grp_df["平均その後5日(%)"].idxmax()]
                st.caption(
                    f"過去データ上、乖離が **{_best_band['乖離帯']}** のときに買うと、"
                    f"その後5日の平均は **{_best_band['平均その後5日(%)']:+.2f}%** と最も高い傾向"
                    f"（サンプル{int(_best_band['サンプル数'])}件）。"
                    "マイナス乖離帯が高い＝『下げたところを拾うと戻りやすかった』ことを示します。"
                )
            else:
                st.info("サンプル数が不足しているため押し目分析を表示できません。分析期間を長くしてください。")
        else:
            st.info("押し目分析に十分なデータがありません。サイドバーで分析期間を長く設定してください。")

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
        # xlsバイナリをpandasで読む（calamine優先）
        import io
        _content = io.BytesIO(resp.content)
        try:
            df = pd.read_excel(_content, engine="calamine")
        except Exception:
            _content.seek(0)
            try:
                df = pd.read_excel(_content, engine="openpyxl")
            except Exception:
                _content.seek(0)
                df = pd.read_excel(_content)
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_consensus_data(ticker):
    """株探からコンセンサス予想データを取得（来期=FY2027予想含む）"""
    code = ticker.replace(".T", "")
    result = {
        "has_data": False,
        "current_fy": None,  # 今期予想
        "next_fy": None,     # 来期予想（27年度）
        "analyst_count": 0,
        "target_price": None,
        "target_price_high": None,
        "target_price_low": None,
        "rating": None,       # レーティング平均
        "rating_dist": {},    # 強気/やや強気/中立/やや弱気/弱気 の分布
        "consensus_trend": None,  # 上方修正/据え置き/下方修正
        "eps_consensus": None,
        "eps_company": None,
        "revenue_consensus": None,
        "op_profit_consensus": None,
        "next_fy_eps_consensus": None,
        "next_fy_revenue_consensus": None,
        "next_fy_op_profit_consensus": None,
        "next_fy_growth": None,  # 来期利益成長率予想
    }

    try:
        # 株探コンセンサスページ
        url = f"https://kabutan.jp/stock/consensus/?code={code}"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.encoding = resp.apparent_encoding
        if resp.status_code != 200:
            return result

        soup = BeautifulSoup(resp.text, "html.parser")

        # レーティング情報
        rating_section = soup.select_one("div.rating_box") or soup.select_one("div#consensus_rating")
        if rating_section:
            # レーティング平均値
            avg_elem = rating_section.select_one("span.rating_avg") or rating_section.select_one("td.rating_avg")
            if avg_elem:
                try:
                    result["rating"] = float(avg_elem.get_text(strip=True))
                except (ValueError, TypeError):
                    pass

        # 目標株価
        target_section = soup.select("table")
        for table in target_section:
            text = table.get_text()
            if "目標株価" in text or "ターゲット" in text:
                tds = table.select("td")
                for i, td in enumerate(tds):
                    t = td.get_text(strip=True)
                    if "コンセンサス" in t or "平均" in t:
                        # 次のtdが値
                        if i + 1 < len(tds):
                            try:
                                val = tds[i+1].get_text(strip=True).replace(",", "").replace("円", "")
                                result["target_price"] = float(val)
                            except (ValueError, TypeError):
                                pass
                    elif "高値" in t:
                        if i + 1 < len(tds):
                            try:
                                val = tds[i+1].get_text(strip=True).replace(",", "").replace("円", "")
                                result["target_price_high"] = float(val)
                            except (ValueError, TypeError):
                                pass
                    elif "安値" in t:
                        if i + 1 < len(tds):
                            try:
                                val = tds[i+1].get_text(strip=True).replace(",", "").replace("円", "")
                                result["target_price_low"] = float(val)
                            except (ValueError, TypeError):
                                pass
                break

        # コンセンサス予想テーブル（今期・来期）
        consensus_tables = soup.select("table.stock_kabuka_table") or soup.select("table")
        for table in consensus_tables:
            headers = [th.get_text(strip=True) for th in table.select("th")]
            rows = table.select("tr")

            # 「コンセンサス」「会社予想」を含むテーブルを探す
            table_text = table.get_text()
            if "コンセンサス" not in table_text:
                continue

            for row in rows:
                cells = row.select("td")
                row_header = row.select_one("th")
                if not row_header or not cells:
                    continue
                rh_text = row_header.get_text(strip=True)

                # 売上高
                if "売上" in rh_text:
                    for cell in cells:
                        ct = cell.get_text(strip=True).replace(",", "")
                        try:
                            val = float(ct)
                            if result["revenue_consensus"] is None:
                                result["revenue_consensus"] = val
                            else:
                                result["next_fy_revenue_consensus"] = val
                        except (ValueError, TypeError):
                            pass

                # 営業利益
                if "営業利益" in rh_text or "営業" in rh_text:
                    for cell in cells:
                        ct = cell.get_text(strip=True).replace(",", "")
                        try:
                            val = float(ct)
                            if result["op_profit_consensus"] is None:
                                result["op_profit_consensus"] = val
                            else:
                                result["next_fy_op_profit_consensus"] = val
                        except (ValueError, TypeError):
                            pass

                # EPS
                if "EPS" in rh_text or "一株益" in rh_text or "1株益" in rh_text:
                    for cell in cells:
                        ct = cell.get_text(strip=True).replace(",", "")
                        try:
                            val = float(ct)
                            if result["eps_consensus"] is None:
                                result["eps_consensus"] = val
                            else:
                                result["next_fy_eps_consensus"] = val
                        except (ValueError, TypeError):
                            pass

            result["has_data"] = True
            break

        # アナリスト数・レーティング分布
        # 株探のレーティング分布テーブル
        for table in target_section:
            text = table.get_text()
            if any(kw in text for kw in ["強気", "中立", "弱気"]):
                tds = table.select("td")
                dist = {}
                for i, td in enumerate(tds):
                    t = td.get_text(strip=True)
                    if t in ["強気", "やや強気", "中立", "やや弱気", "弱気"]:
                        if i + 1 < len(tds):
                            try:
                                count = int(tds[i+1].get_text(strip=True).replace("人", "").replace("社", ""))
                                dist[t] = count
                            except (ValueError, TypeError):
                                pass
                if dist:
                    result["rating_dist"] = dist
                    result["analyst_count"] = sum(dist.values())
                break

        # 来期成長率の計算
        if result["eps_consensus"] and result["next_fy_eps_consensus"] and result["eps_consensus"] > 0:
            result["next_fy_growth"] = (result["next_fy_eps_consensus"] - result["eps_consensus"]) / abs(result["eps_consensus"])

    except Exception:
        pass

    # yfinanceからも補完データ取得
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not result["target_price"]:
            result["target_price"] = info.get("targetMeanPrice")
        if not result["target_price_high"]:
            result["target_price_high"] = info.get("targetHighPrice")
        if not result["target_price_low"]:
            result["target_price_low"] = info.get("targetLowPrice")
        if not result["analyst_count"]:
            result["analyst_count"] = info.get("numberOfAnalystOpinions", 0)
        if not result["rating"]:
            result["rating"] = info.get("recommendationMean")

        # Forward EPS（来期予想の代替）
        forward_eps = info.get("forwardEps")
        trailing_eps = info.get("trailingEps")
        if forward_eps and not result["eps_consensus"]:
            result["eps_consensus"] = forward_eps
        if trailing_eps and forward_eps and trailing_eps > 0 and not result["next_fy_growth"]:
            result["next_fy_growth"] = (forward_eps - trailing_eps) / abs(trailing_eps)

        # earningsGrowth（来期成長率）
        eg = info.get("earningsGrowth")
        if eg is not None and result["next_fy_growth"] is None:
            result["next_fy_growth"] = eg

        # revenueGrowth
        rg = info.get("revenueGrowth")
        if rg is not None and result.get("revenue_growth") is None:
            result["revenue_growth"] = rg

        result["has_data"] = True
    except Exception:
        pass

    return result


def assess_earnings_crossing_risk(ticker, name, consensus_data, earnings_info, stock_df=None):
    """決算跨ぎリスクを総合評価し、判断材料を返す"""
    risk_score = 50  # 中立スタート（0=安全, 100=危険）
    factors = []
    recommendation = "中立"

    days_to_earn = None
    next_earn = earnings_info.get("next_earnings")
    if next_earn is not None:
        try:
            today = pd.Timestamp.now().normalize()
            ne = pd.Timestamp(next_earn).tz_localize(None) if next_earn.tzinfo else next_earn
            days_to_earn = (ne - today).days
        except Exception:
            pass

    if days_to_earn is None or days_to_earn < 0:
        return None  # 決算日不明 or 済み

    # --- リスク要因の評価 ---

    # 1. コンセンサス成長率（来期予想）
    next_growth = consensus_data.get("next_fy_growth")
    if next_growth is not None:
        if next_growth < -0.10:
            risk_score += 25
            factors.append(("🔴 来期減益予想", f"EPS成長率 {next_growth*100:+.1f}%（大幅減益）", "danger"))
        elif next_growth < -0.03:
            risk_score += 15
            factors.append(("🟠 来期微減益予想", f"EPS成長率 {next_growth*100:+.1f}%", "warning"))
        elif next_growth < 0.05:
            risk_score += 5
            factors.append(("⚪ 来期横ばい予想", f"EPS成長率 {next_growth*100:+.1f}%", "neutral"))
        elif next_growth < 0.15:
            risk_score -= 10
            factors.append(("🔵 来期増益予想", f"EPS成長率 {next_growth*100:+.1f}%", "positive"))
        else:
            risk_score -= 20
            factors.append(("🟢 来期大幅増益予想", f"EPS成長率 {next_growth*100:+.1f}%", "very_positive"))

    # 2. アナリストレーティング
    rating = consensus_data.get("rating")
    if rating is not None:
        if rating >= 4.0:
            risk_score += 20
            factors.append(("🔴 アナリスト弱気", f"レーティング {rating:.1f}/5.0（売り寄り）", "danger"))
        elif rating >= 3.5:
            risk_score += 10
            factors.append(("🟠 アナリストやや弱気", f"レーティング {rating:.1f}/5.0", "warning"))
        elif rating <= 2.0:
            risk_score -= 15
            factors.append(("🟢 アナリスト強気", f"レーティング {rating:.1f}/5.0（買い寄り）", "very_positive"))
        elif rating <= 2.5:
            risk_score -= 8
            factors.append(("🔵 アナリストやや強気", f"レーティング {rating:.1f}/5.0", "positive"))
        else:
            factors.append(("⚪ アナリスト中立", f"レーティング {rating:.1f}/5.0", "neutral"))

    # 3. レーティング分布（弱気比率）
    dist = consensus_data.get("rating_dist", {})
    if dist:
        total_analysts = sum(dist.values())
        bearish = dist.get("弱気", 0) + dist.get("やや弱気", 0)
        bullish = dist.get("強気", 0) + dist.get("やや強気", 0)
        if total_analysts > 0:
            bearish_ratio = bearish / total_analysts
            if bearish_ratio > 0.4:
                risk_score += 15
                factors.append(("🔴 弱気派多数", f"弱気{bearish}/{total_analysts}人 ({bearish_ratio*100:.0f}%)", "danger"))
            elif bearish_ratio > 0.25:
                risk_score += 8
                factors.append(("🟠 弱気派一定数", f"弱気{bearish}/{total_analysts}人", "warning"))
            elif bullish / total_analysts > 0.6:
                risk_score -= 10
                factors.append(("🟢 強気派多数", f"強気{bullish}/{total_analysts}人", "positive"))

    # 4. 目標株価 vs 現在株価
    target = consensus_data.get("target_price")
    if target and stock_df is not None and not stock_df.empty:
        current_price = float(stock_df["Close"].iloc[-1])
        if current_price > 0:
            upside = (target - current_price) / current_price
            if upside < -0.10:
                risk_score += 20
                factors.append(("🔴 目標株価下回り", f"目標¥{target:,.0f} vs 現在¥{current_price:,.0f} ({upside*100:+.1f}%)", "danger"))
            elif upside < 0:
                risk_score += 10
                factors.append(("🟠 目標株価近接", f"目標¥{target:,.0f} ({upside*100:+.1f}%)", "warning"))
            elif upside > 0.20:
                risk_score -= 15
                factors.append(("🟢 上値余地大", f"目標¥{target:,.0f} ({upside*100:+.1f}%)", "very_positive"))
            elif upside > 0.05:
                risk_score -= 5
                factors.append(("🔵 上値余地あり", f"目標¥{target:,.0f} ({upside*100:+.1f}%)", "positive"))

    # 5. 直前のモメンタム（決算前に既に上がりすぎていないか）
    if stock_df is not None and len(stock_df) >= 20:
        close = stock_df["Close"].astype(float)
        ret_20d = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        if ret_20d > 0.15:
            risk_score += 15
            factors.append(("🟠 決算前急騰", f"直近20日 {ret_20d*100:+.1f}%（期待織込み済みリスク）", "warning"))
        elif ret_20d > 0.08:
            risk_score += 5
            factors.append(("⚪ 決算前上昇", f"直近20日 {ret_20d*100:+.1f}%", "neutral"))
        elif ret_20d < -0.10:
            risk_score -= 5
            factors.append(("🔵 決算前調整済み", f"直近20日 {ret_20d*100:+.1f}%（悪材料織込み済み可能性）", "positive"))

    # 6. 業績スコア（既存の earnings_score）
    earn_score = earnings_info.get("earnings_score", 0)
    if earn_score < -20:
        risk_score += 15
        factors.append(("🔴 業績スコア低い", f"業績 {earn_score:+d}点", "danger"))
    elif earn_score < 0:
        risk_score += 5
        factors.append(("🟠 業績スコアやや低い", f"業績 {earn_score:+d}点", "warning"))
    elif earn_score > 30:
        risk_score -= 15
        factors.append(("🟢 業績スコア高い", f"業績 {earn_score:+d}点", "very_positive"))
    elif earn_score > 10:
        risk_score -= 5
        factors.append(("🔵 業績スコア良好", f"業績 {earn_score:+d}点", "positive"))

    # --- 総合判断 ---
    risk_score = max(0, min(100, risk_score))

    if risk_score >= 70:
        recommendation = "回避推奨 🚫"
        rec_color = "#f44336"
        rec_detail = "コンセンサスが弱気。決算前に一部利確を検討"
    elif risk_score >= 55:
        recommendation = "要注意 ⚠️"
        rec_color = "#ff9800"
        rec_detail = "リスク要因あり。ポジション縮小を検討"
    elif risk_score >= 40:
        recommendation = "中立 ➖"
        rec_color = "#9e9e9e"
        rec_detail = "明確な方向感なし。通常ポジションで可"
    elif risk_score >= 25:
        recommendation = "跨ぎ可 ✅"
        rec_color = "#4caf50"
        rec_detail = "コンセンサス良好。保有継続で問題なし"
    else:
        recommendation = "積極保有 🟢"
        rec_color = "#00c853"
        rec_detail = "強気コンセンサス。決算跨ぎで上振れ期待"

    return {
        "ticker": ticker,
        "name": name,
        "days_to_earn": days_to_earn,
        "next_earnings": next_earn,
        "risk_score": risk_score,
        "recommendation": recommendation,
        "rec_color": rec_color,
        "rec_detail": rec_detail,
        "factors": factors,
        "consensus": consensus_data,
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

            # --- ⚠️ 決算跨ぎリスク管理セクション ---
            st.markdown("### ⚠️ 決算跨ぎリスク管理")
            st.caption("決算発表が近い銘柄のコンセンサス予想・アナリスト評価から跨ぎ判断を支援。来期（27年度）予想が弱気な場合は警告表示。")

            # 決算が14日以内の銘柄を抽出
            upcoming_earnings = [r for r in results if r.get("days_to_earn") is not None and 0 <= r["days_to_earn"] <= 14]

            if upcoming_earnings:
                st.warning(f"⚡ {len(upcoming_earnings)}銘柄が14日以内に決算発表を控えています")

                with st.spinner("📊 コンセンサス予想データ取得中..."):
                    crossing_assessments = []
                    for r in upcoming_earnings:
                        ticker = r["ticker"]
                        name = r["name"]
                        ei = {"next_earnings": r["next_earnings"], "earnings_score": r["earn_score"], "earnings_detail": r["earn_detail"]}
                        consensus = fetch_consensus_data(ticker)
                        df = pf_data.get(ticker)
                        assessment = assess_earnings_crossing_risk(ticker, name, consensus, ei, df)
                        if assessment:
                            crossing_assessments.append(assessment)

                if crossing_assessments:
                    # リスクスコア順にソート（高リスク順）
                    crossing_assessments.sort(key=lambda x: x["risk_score"], reverse=True)

                    # サマリー
                    avoid_count = sum(1 for a in crossing_assessments if a["risk_score"] >= 70)
                    caution_count = sum(1 for a in crossing_assessments if 55 <= a["risk_score"] < 70)
                    ok_count = sum(1 for a in crossing_assessments if a["risk_score"] < 40)

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("🚫 回避推奨", f"{avoid_count}銘柄")
                    sc2.metric("⚠️ 要注意", f"{caution_count}銘柄")
                    sc3.metric("✅ 跨ぎ可", f"{ok_count}銘柄")

                    # 各銘柄の詳細
                    for a in crossing_assessments:
                        code = a["ticker"].replace(".T", "")
                        ne = a["next_earnings"]
                        earn_date_str = ne.strftime("%m/%d") if ne is not None else "不明"
                        dte = a["days_to_earn"]
                        cons = a["consensus"]

                        # リスクバーの色
                        if a["risk_score"] >= 70:
                            bar_color = "#f44336"
                            bg_border = "#f44336"
                        elif a["risk_score"] >= 55:
                            bar_color = "#ff9800"
                            bg_border = "#ff9800"
                        elif a["risk_score"] >= 40:
                            bar_color = "#9e9e9e"
                            bg_border = "#555"
                        else:
                            bar_color = "#4caf50"
                            bg_border = "#4caf50"

                        # コンセンサス情報のHTML
                        consensus_html = ""
                        if cons.get("next_fy_growth") is not None:
                            g = cons["next_fy_growth"]
                            g_color = "#00c853" if g > 0.05 else "#f44336" if g < -0.03 else "#ff9800"
                            consensus_html += f"<span style='color:{g_color};font-weight:bold;'>来期EPS成長: {g*100:+.1f}%</span> "
                        if cons.get("target_price"):
                            consensus_html += f"<span style='color:#aaa;'>目標株価: ¥{cons['target_price']:,.0f}</span> "
                        if cons.get("analyst_count"):
                            consensus_html += f"<span style='color:#888;'>({cons['analyst_count']}人)</span> "

                        # レーティング分布
                        dist = cons.get("rating_dist", {})
                        dist_html = ""
                        if dist:
                            parts = []
                            for label in ["強気", "やや強気", "中立", "やや弱気", "弱気"]:
                                if label in dist and dist[label] > 0:
                                    lbl_color = {"強気": "#00c853", "やや強気": "#4caf50", "中立": "#9e9e9e", "やや弱気": "#ff9800", "弱気": "#f44336"}.get(label, "#888")
                                    parts.append(f"<span style='color:{lbl_color};'>{label}:{dist[label]}</span>")
                            if parts:
                                dist_html = " ".join(parts)

                        # 判断要因リスト
                        factors_html = ""
                        for factor_label, factor_detail, factor_type in a["factors"]:
                            f_color = {"danger": "#f44336", "warning": "#ff9800", "neutral": "#9e9e9e", "positive": "#4caf50", "very_positive": "#00c853"}.get(factor_type, "#888")
                            factors_html += (
                                f"<div style='padding:2px 0;'>"
                                f"<span style='color:{f_color};font-size:0.78rem;'>{factor_label}</span>"
                                f" <span style='color:#aaa;font-size:0.72rem;'>{factor_detail}</span>"
                                f"</div>"
                            )

                        # カード表示
                        card_html = (
                            f"<div style='background:#1a1a2e;padding:14px 16px;border-radius:10px;"
                            f"margin-bottom:8px;border-left:5px solid {bg_border};'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                            f"<div>"
                            f"<a href='https://kabutan.jp/stock/consensus/?code={code}' target='_blank' "
                            f"style='color:#fff;text-decoration:none;font-weight:bold;font-size:1rem;'>"
                            f"{a['name']}</a>"
                            f" <span style='color:#888;font-size:0.8rem;'>({code})</span>"
                            f" <span style='background:#d50000;color:#fff;padding:2px 8px;border-radius:4px;"
                            f"font-size:0.75rem;font-weight:bold;margin-left:8px;'>📅 決算 {earn_date_str} ({dte}日後)</span>"
                            f"</div>"
                            f"<div style='text-align:right;'>"
                            f"<span style='color:{a['rec_color']};font-weight:bold;font-size:1.1rem;'>{a['recommendation']}</span>"
                            f"<div style='color:#888;font-size:0.7rem;'>リスクスコア: {a['risk_score']}/100</div>"
                            f"</div>"
                            f"</div>"
                            f"<div style='margin:8px 0;background:#333;border-radius:4px;height:6px;'>"
                            f"<div style='background:{bar_color};width:{a['risk_score']}%;height:100%;border-radius:4px;'></div>"
                            f"</div>"
                        )
                        if consensus_html:
                            card_html += f"<div style='margin:6px 0;font-size:0.82rem;'>{consensus_html}</div>"
                        if dist_html:
                            card_html += f"<div style='margin:4px 0;font-size:0.75rem;'>📊 レーティング分布: {dist_html}</div>"
                        card_html += (
                            f"<div style='margin-top:8px;padding:8px;background:#0d1117;border-radius:6px;'>"
                            f"<div style='color:#888;font-size:0.72rem;margin-bottom:4px;'>📋 判断要因:</div>"
                            f"{factors_html}"
                            f"</div>"
                            f"<div style='margin-top:6px;color:{a['rec_color']};font-size:0.8rem;'>"
                            f"💡 {a['rec_detail']}"
                            f"</div>"
                            f"<div style='margin-top:6px;'>"
                            f"<a href='https://kabutan.jp/stock/consensus/?code={code}' target='_blank' "
                            f"style='color:#4da6ff;font-size:0.72rem;text-decoration:none;margin-right:12px;'>📊 コンセンサス詳細</a>"
                            f"<a href='https://kabutan.jp/stock/finance/?code={code}' target='_blank' "
                            f"style='color:#4da6ff;font-size:0.72rem;text-decoration:none;margin-right:12px;'>📈 業績推移</a>"
                            f"<a href='https://shikiho.toyokeizai.net/stocks/{code}' target='_blank' "
                            f"style='color:#4da6ff;font-size:0.72rem;text-decoration:none;'>📖 四季報</a>"
                            f"</div>"
                            f"</div>"
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

                    # 全体サマリーテーブル
                    with st.expander("📋 決算跨ぎ判断サマリーテーブル", expanded=False):
                        summary_data = []
                        for a in crossing_assessments:
                            cons = a["consensus"]
                            summary_data.append({
                                "銘柄": a["name"],
                                "コード": a["ticker"].replace(".T", ""),
                                "決算日": a["next_earnings"].strftime("%m/%d") if a["next_earnings"] else "不明",
                                "残日数": a["days_to_earn"],
                                "リスク": a["risk_score"],
                                "判断": a["recommendation"],
                                "来期成長率": f"{cons.get('next_fy_growth', 0)*100:+.1f}%" if cons.get("next_fy_growth") is not None else "N/A",
                                "目標株価": f"¥{cons['target_price']:,.0f}" if cons.get("target_price") else "N/A",
                                "アナリスト数": cons.get("analyst_count", 0),
                            })
                        if summary_data:
                            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                else:
                    st.info("コンセンサスデータの取得に失敗しました。")
            else:
                # 14日以内に決算がない場合でも、30日以内を表示
                upcoming_30d = [r for r in results if r.get("days_to_earn") is not None and 0 <= r["days_to_earn"] <= 30]
                if upcoming_30d:
                    st.info(f"📅 14日以内に決算発表の銘柄はありません。30日以内: {len(upcoming_30d)}銘柄")
                    for r in sorted(upcoming_30d, key=lambda x: x["days_to_earn"]):
                        code = r["ticker"].replace(".T", "")
                        ne = r["next_earnings"]
                        earn_str = ne.strftime("%m/%d") if ne else "不明"
                        st.markdown(
                            f"- **{r['name']}** ({code}) — 決算 {earn_str} ({r['days_to_earn']}日後) "
                            f"[📊 コンセンサス](https://kabutan.jp/stock/consensus/?code={code})"
                        )
                else:
                    st.success("✅ 30日以内に決算発表を控えている銘柄はありません。")

            st.divider()

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
    "SOX(半導体)": "^SOX",
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

    # -----------------------------------------------------------------
    # 信用評価損益率セクション
    # -----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📉 信用評価損益率 — 底打ちシグナル")
    st.caption(
        "信用取引の評価損益率は市場センチメントの逆張り指標。"
        "**-10%〜-20%**の範囲は投げ売りクライマックスに近く、底打ちのシグナルになりやすい。"
        "-5%以上は過熱圏で売りシグナル。"
    )
    st.markdown(
        "**データ引用元:** "
        "[トレーダーズ・ウェブ 信用評価損益率](https://www.traders.co.jp/margin/transition/transition.asp) / "
        "[松井証券 ネットストック情報](https://www.matsui.co.jp/market/stock/netstock-info/) / "
        "[日本経済新聞 信用取引情報](https://www.nikkei.com/markets/kabu/shinyo/) / "
        "[kabutan 信用残](https://kabutan.jp/warning/?mode=2_6)"
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_margin_evaluation_loss():
        """信用評価損益率データを取得する。
        
        松井証券の信用評価損益率（週次更新）を取得。
        取得失敗時はサンプルデータを生成して表示。
        
        Returns:
            DataFrame with columns: ['date', 'margin_loss_rate']
            margin_loss_rate: マイナス値（例: -15.2 は -15.2%を意味）
        """
        import re as _re
        import json as _json
        from datetime import timezone as _tz

        # =========================================================
        # 最優先ソース: nikkei225jp.com（最も更新が速い。日次JSONを直接取得）
        #   dailyweek2.json: 各行 [ts(ms), N225, TOPIX, ..., [7]=信用評価損益率, [8]=信用倍率, ...]
        #   data_upd.json  : {"sinyou": {"asof": "YYYY-MM-DD", "at": "..."} , ...}
        # =========================================================
        try:
            _nk_base = "https://nikkei225jp.com"
            _nk_headers = dict(_MINKABU_HEADERS)
            _nk_headers["Referer"] = "https://nikkei225jp.com/data/sinyou.php"

            _r = requests.get(
                _nk_base + "/_data/_nfsDATA/data_DAY/dailyweek2.json",
                headers=_nk_headers, timeout=15)
            if _r.status_code == 200:
                # 中身は純JSONではなく "var DAILY = [ ... ];" 形式のJS。
                # 配列部分だけを取り出してパースする。
                _txt = _r.content.decode("utf-8", errors="replace")
                _m_arr = _re.search(r"=\s*(\[.*\])\s*;?\s*$", _txt, _re.DOTALL)
                _rows = _json.loads(_m_arr.group(1)) if _m_arr else _json.loads(_txt)
                _recs = []
                for _row in _rows:
                    if not isinstance(_row, list) or len(_row) <= 7:
                        continue
                    _ts = _row[0]
                    _val = _row[7]
                    # 評価損益率が入っている行のみ（空文字や非数値は除外）
                    if _val is None or _val == "":
                        continue
                    try:
                        _v = float(_val)
                    except (TypeError, ValueError):
                        continue
                    if not (-60 < _v < 20):  # 妥当範囲
                        continue
                    try:
                        _d = datetime.fromtimestamp(int(_ts) / 1000, tz=_tz.utc)
                    except Exception:
                        continue
                    _recs.append({"date": _d.strftime("%Y-%m-%d"), "margin_loss_rate": round(_v, 2)})

                if len(_recs) >= 5:
                    _df = pd.DataFrame(_recs)
                    _df["date"] = pd.to_datetime(_df["date"])
                    _df = _df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

                    # 更新基準日（asof）を取得してセッションに保存（鮮度警告に使用）
                    _asof = None
                    try:
                        _ru = requests.get(
                            _nk_base + "/_data/_nfsDATA/data/data_upd.json",
                            headers=_nk_headers, timeout=10)
                        if _ru.status_code == 200:
                            _utxt = _ru.content.decode("utf-8", errors="replace")
                            try:
                                _upd = _json.loads(_utxt)
                            except Exception:
                                _mu = _re.search(r"=\s*(\{.*\})\s*;?\s*$", _utxt, _re.DOTALL)
                                _upd = _json.loads(_mu.group(1)) if _mu else {}
                            _asof = (_upd.get("sinyou") or {}).get("asof")
                    except Exception:
                        pass
                    try:
                        st.session_state["margin_data_source"] = "nikkei225jp.com"
                        st.session_state["margin_data_asof"] = _asof
                    except Exception:
                        pass
                    return _df
        except Exception:
            pass  # 失敗したら従来ソースへフォールバック

        # 松井証券の信用評価損益率ページをスクレイピング
        urls_to_try = [
            "https://www.traders.co.jp/margin/transition/transition.asp",
            "https://www.matsui.co.jp/market/stock/netstock-info/",
        ]
        
        for url in urls_to_try:
            try:
                resp = requests.get(url, headers=_MINKABU_HEADERS, timeout=15)
                if resp.status_code != 200:
                    continue
                resp.encoding = resp.apparent_encoding or "utf-8"
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # テーブルからデータ抽出を試みる
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    records = []
                    for tr in rows[1:]:
                        tds = tr.find_all(["td", "th"])
                        cells = [td.get_text(strip=True) for td in tds]
                        # 日付と数値を探す
                        for i, cell in enumerate(cells):
                            m_date = _re.search(r"(\d{4})[/\-年](\d{1,2})[/\-月](\d{1,2})", cell)
                            if m_date:
                                date_str = f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}"
                                # 同行の他セルから損益率を探す
                                for j, c2 in enumerate(cells):
                                    if i == j:
                                        continue
                                    m_val = _re.search(r"([+\-]?\d+\.?\d*)", c2.replace(",", ""))
                                    if m_val:
                                        val = float(m_val.group(1))
                                        if -50 < val < 10:  # 妥当な範囲
                                            records.append({"date": date_str, "margin_loss_rate": val})
                                            break
                                break
                    if len(records) >= 5:
                        df = pd.DataFrame(records)
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.sort_values("date").reset_index(drop=True)
                        try:
                            st.session_state["margin_data_source"] = "トレーダーズ・ウェブ"
                            st.session_state["margin_data_asof"] = df["date"].iloc[-1].strftime("%Y-%m-%d")
                        except Exception:
                            pass
                        return df
            except Exception:
                continue
        
        # スクレイピング失敗時: 日経225の値動きから擬似的に信用評価損益率を推定
        # （実際の市場では、指数が下がると評価損益率も悪化する傾向がある）
        try:
            nk = yf.download("^N225", period="2y", progress=False)
            if nk.empty:
                return pd.DataFrame()
            if isinstance(nk.columns, pd.MultiIndex):
                nk.columns = nk.columns.get_level_values(0)
            close = nk["Close"].astype(float)
            
            # 移動平均からの乖離 + RSI を使って擬似評価損益率を生成
            ma75 = close.rolling(75).mean()
            deviation = (close - ma75) / ma75 * 100
            
            # 信用評価損益率の典型レンジ: -25% 〜 +5%
            # 移動平均乖離率をスケーリング（おおよその相関関係）
            margin_rate = deviation * 0.8 - 5  # 基準を-5%にオフセット
            margin_rate = margin_rate.clip(-30, 5)
            
            # 週次にリサンプル
            weekly = margin_rate.resample("W-Fri").last().dropna()
            
            df = pd.DataFrame({
                "date": weekly.index,
                "margin_loss_rate": weekly.values.round(2),
            }).reset_index(drop=True)
            try:
                st.session_state["margin_data_source"] = "擬似推定（日経225ベース）"
                st.session_state["margin_data_asof"] = None
            except Exception:
                pass
            return df
        except Exception:
            return pd.DataFrame()

    margin_df = fetch_margin_evaluation_loss()
    
    if not margin_df.empty and len(margin_df) >= 4:
        latest_rate = float(margin_df["margin_loss_rate"].iloc[-1])
        prev_rate = float(margin_df["margin_loss_rate"].iloc[-2])
        change = latest_rate - prev_rate
        
        # シグナル判定
        if -20 <= latest_rate <= -10:
            signal_label = "🟢 底打ちゾーン（買いシグナル）"
            signal_color = "#00c853"
            signal_detail = "評価損益率が-10%〜-20%の範囲に入りました。歴史的に底打ちのタイミングに近いことが多いです。"
        elif latest_rate < -20:
            signal_label = "🔵 投げ売りクライマックス（強い買いシグナル）"
            signal_color = "#2196f3"
            signal_detail = "評価損益率が-20%を超えて悪化。パニック売りの極致に近く、反転期待が高まるゾーンです。"
        elif latest_rate > -5:
            signal_label = "🔴 過熱圏（売りシグナル）"
            signal_color = "#d50000"
            signal_detail = "評価損益率が-5%以上に改善。楽観が広がり過熱の兆候。利益確定や新規売りのタイミング。"
        elif latest_rate > -10:
            signal_label = "🟠 警戒圏"
            signal_color = "#ff6d00"
            signal_detail = "評価損益率が-10%付近。過熱感が出始めている状態です。"
        else:
            signal_label = "⚪ 通常圏"
            signal_color = "#888888"
            signal_detail = "中立的な水準。明確なシグナルなし。"
        
        # メトリクス表示
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("最新 評価損益率", f"{latest_rate:.1f}%",
                   delta=f"{change:+.1f}%",
                   delta_color="inverse" if change < 0 else "normal")
        mc2.metric("シグナル", signal_label.split("（")[0])
        mc3.metric("データ日付", margin_df["date"].iloc[-1].strftime("%Y/%m/%d"))
        mc4.metric("データ点数", f"{len(margin_df)}週分")

        # --- データソース明示 & 鮮度警告 ---
        _m_src = st.session_state.get("margin_data_source", "不明")
        _m_asof = st.session_state.get("margin_data_asof")
        # 基準日（asof優先、無ければ最終データ日）からの経過日数を算出
        try:
            _ref_date = pd.to_datetime(_m_asof) if _m_asof else margin_df["date"].iloc[-1]
            _age_days = (datetime.now().date() - _ref_date.date()).days
        except Exception:
            _ref_date, _age_days = None, None

        _src_note = f"📡 データソース: **{_m_src}**"
        if _m_asof:
            _src_note += f"（更新基準日: {pd.to_datetime(_m_asof).strftime('%Y/%m/%d')}）"
        if _age_days is not None:
            _src_note += f" — 基準日から**{_age_days}日経過**"
        st.caption(_src_note)

        if _m_src.startswith("擬似"):
            st.warning(
                "⚠️ 実データの取得に失敗したため、日経225から推定した**擬似値**を表示しています。"
                "実際の信用評価損益率とは乖離する可能性があります。"
            )
        elif _age_days is not None and _age_days >= 10:
            st.warning(
                f"⚠️ このデータは基準日から{_age_days}日経過しています。"
                "信用評価損益率は週次公表＋タイムラグがあるため、"
                "**進行中の相場変動（直近の急落・急騰）はまだ反映されていない可能性**があります。"
                "直近の値動きとあわせて慎重に判断してください。"
            )
        
        # シグナル解説
        st.markdown(
            f"<div style='background:#1a1a2e;padding:12px 16px;border-radius:8px;"
            f"border-left:5px solid {signal_color};margin:8px 0;'>"
            f"<div style='color:{signal_color};font-weight:700;font-size:1rem;'>{signal_label}</div>"
            f"<div style='color:#ccc;font-size:0.85rem;margin-top:4px;'>{signal_detail}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # チャート描画
        fig_margin = go.Figure()
        
        # メインライン
        fig_margin.add_trace(go.Scatter(
            x=margin_df["date"],
            y=margin_df["margin_loss_rate"],
            mode="lines+markers",
            name="信用評価損益率",
            line=dict(color="#26a69a", width=2),
            marker=dict(size=4),
        ))
        
        # 底打ちゾーン（-10% 〜 -20%）をハイライト
        fig_margin.add_hrect(
            y0=-20, y1=-10,
            fillcolor="rgba(76, 175, 80, 0.15)",
            line_width=0,
            annotation_text="底打ちゾーン (-10%〜-20%)",
            annotation_position="top left",
            annotation_font_color="#66bb6a",
        )
        
        # 投げ売りクライマックス（-20%以下）
        fig_margin.add_hrect(
            y0=-35, y1=-20,
            fillcolor="rgba(33, 150, 243, 0.1)",
            line_width=0,
            annotation_text="投げ売りクライマックス",
            annotation_position="bottom left",
            annotation_font_color="#42a5f5",
        )
        
        # 過熱圏（-5%以上）
        fig_margin.add_hrect(
            y0=-5, y1=5,
            fillcolor="rgba(244, 67, 54, 0.12)",
            line_width=0,
            annotation_text="過熱圏 (売りシグナル)",
            annotation_position="top right",
            annotation_font_color="#ef5350",
        )
        
        # 基準ライン
        fig_margin.add_hline(y=-10, line_dash="dash", line_color="#66bb6a", opacity=0.7)
        fig_margin.add_hline(y=-20, line_dash="dash", line_color="#42a5f5", opacity=0.7)
        fig_margin.add_hline(y=-5, line_dash="dash", line_color="#ef5350", opacity=0.7)
        fig_margin.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
        
        # 直近値をアノテーション
        fig_margin.add_annotation(
            x=margin_df["date"].iloc[-1],
            y=latest_rate,
            text=f"現在: {latest_rate:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=signal_color,
            font=dict(color=signal_color, size=12, family="Arial Black"),
            ax=40, ay=-30,
        )
        
        fig_margin.update_layout(
            title="信用評価損益率 推移（週次）",
            yaxis_title="評価損益率 (%)",
            xaxis_title="日付",
            height=450,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            hovermode="x unified",
            margin=dict(l=50, r=20, t=50, b=30),
            yaxis=dict(dtick=5),
        )
        st.plotly_chart(fig_margin, use_container_width=True)
        
        # 売買シグナル判定テーブル
        st.markdown("#### 📊 売買シグナル判定基準")
        signal_table = pd.DataFrame({
            "ゾーン": ["投げ売りクライマックス", "底打ちゾーン", "通常圏", "警戒圏", "過熱圏"],
            "評価損益率": ["< -20%", "-10% 〜 -20%", "-10% 〜 -20%以上", "-5% 〜 -10%", "> -5%"],
            "判定": ["🔵 強い買い", "🟢 買い", "⚪ 中立", "🟠 警戒（利確検討）", "🔴 売り"],
            "解説": [
                "パニック売りの極致。逆張り買いの好機（ただしナイフキャッチリスクあり）",
                "歴史的に底打ちしやすいゾーン。段階的な買い増しに適する",
                "中立的な水準。個別材料で判断",
                "楽観的ムード。利益確定を検討すべき水準",
                "過度な楽観。新規売りや全利確のタイミング",
            ],
        })
        st.dataframe(signal_table, use_container_width=True, hide_index=True)
        
        # 過去シグナル発生履歴
        st.markdown("#### 🕐 直近のシグナル発生履歴")
        signal_history = []
        for i in range(len(margin_df)):
            rate = float(margin_df["margin_loss_rate"].iloc[i])
            date = margin_df["date"].iloc[i]
            if -20 <= rate <= -10:
                signal_history.append({"日付": date.strftime("%Y/%m/%d"), "評価損益率": f"{rate:.1f}%", "シグナル": "🟢 底打ちゾーン"})
            elif rate < -20:
                signal_history.append({"日付": date.strftime("%Y/%m/%d"), "評価損益率": f"{rate:.1f}%", "シグナル": "🔵 投げ売りクライマックス"})
            elif rate > -5:
                signal_history.append({"日付": date.strftime("%Y/%m/%d"), "評価損益率": f"{rate:.1f}%", "シグナル": "🔴 過熱圏"})
        
        if signal_history:
            # 直近20件を表示
            hist_df_display = pd.DataFrame(signal_history[-20:][::-1])
            st.dataframe(hist_df_display, use_container_width=True, hide_index=True)
        else:
            st.info("直近期間にシグナル発生なし")
    else:
        st.warning("信用評価損益率データの取得に失敗しました。ネットワーク接続を確認してください。")

    st.caption("📖 ドラゴンキング理論: Sornette (2009) に基づく。パワーロー分布から逸脱する極端イベント（DK）を統計的に検出し、バブル崩壊の前兆を捉える手法。")

# =====================================================================
# Tab 9: 出来高急増ランキング
# =====================================================================

@st.cache_data(ttl=600)
def fetch_kabudragon_volume_ranking():
    """株ドラゴンから出来高急増ランキングを取得（エラー詳細付き）"""
    import re as _re
    url = "https://www.kabudragon.com/ranking/dekizou.html"
    _headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    }
    last_error = ""
    # リトライ（最大2回）
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=20, headers=_headers)
            resp.encoding = resp.apparent_encoding
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                continue
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
                    # 列数が8以上の行を対象（サイト構造変更に柔軟対応）
                    if len(tds) < 8 or len(tds) > 25:
                        continue
                    texts = [td.get_text(strip=True) for td in tds]
                    if not texts[0].isdigit():
                        continue
                    try:
                        # 列構成: 順位, コード, 名称, 市場, [日付], 取引値, 前日比, 前日比%, 急増率, 出来高, [高値, 安値]
                        offset = 4
                        if offset < len(texts) and _re.match(r"^\d+/\d+", texts[offset]):
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
                    except (IndexError, ValueError):
                        continue
                if rows:
                    break

            if rows:
                return rows[:50], date_str
            else:
                last_error = f"テーブル解析失敗（tables={len(soup.find_all('table'))}, content={len(resp.text)}bytes）"
                continue
        except requests.exceptions.Timeout:
            last_error = "タイムアウト（20秒）"
            continue
        except requests.exceptions.ConnectionError as e:
            last_error = f"接続エラー: {e}"
            continue
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
    # 全リトライ失敗時はエラーメッセージを返す
    return [], last_error


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
        err_detail = f"（{vol_date}）" if vol_date else ""
        st.warning(f"出来高急増ランキングの取得に失敗しました。{err_detail}")
        if vol_date:
            st.code(vol_date, language=None)
        st.info("💡 株ドラゴンのサイトが一時的にアクセスできない可能性があります。しばらく待ってからリロードしてください。")

# =====================================================================
# Tab 10: TOPIX新基準スクリーニング
# =====================================================================
_TOPIX_TRADED_VALUE_RATIO_THRESHOLD = 0.14
_TOPIX_CUMULATIVE_MCAP_PCT = 0.97
_TOPIX_CACHE_TTL = 86400

@st.cache_data(ttl=_TOPIX_CACHE_TTL, show_spinner=False)
def _topix_fetch_stock_list():
    """JPX上場銘柄一覧を取得（市場区分・業種付き）"""
    # JPXは配布ファイルを .xls → .xlsx に変更済み（旧.xlsは404）。
    # 新URLを優先し、失敗時は旧URLにフォールバック。
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    url_fallback = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        resp = requests.get(url_fallback, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return pd.DataFrame()
    # calamine エンジンで .xls/.xlsx 両対応
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
            _content2 = io.BytesIO(resp2.content)
            try:
                df = pd.read_excel(_content2, engine="calamine")
            except Exception:
                _content2.seek(0)
                df = pd.read_excel(_content2)
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

    # EDINET DB APIキー（.envから読み込み）
    _EDINETDB_API_KEY = os.environ.get("EDINETDB_API_KEY", "")

    def _edb_fetch(code):
        """EDINET DB REST API経由で企業情報+財務データ取得（キャッシュ付き）"""
        cached = _edb_load(code)
        if cached is not None:
            return cached
        if not _EDINETDB_API_KEY:
            return None
        # キャッシュになければ edinetdb.jp REST API から取得
        _headers = {"X-API-Key": _EDINETDB_API_KEY, "User-Agent": "Mozilla/5.0"}
        try:
            # 1. 証券コードで企業検索 → EDINETコード取得
            search_url = f"https://edinetdb.jp/v1/search?q={code}"
            sr = requests.get(search_url, timeout=15, headers=_headers)
            if sr.status_code != 200 or not sr.json():
                return None
            search_results = sr.json()
            # 検索結果から証券コードが一致するものを探す
            edinet_code = None
            for item in (search_results if isinstance(search_results, list) else [search_results]):
                sc = str(item.get("secCode", "") or item.get("sec_code", "")).replace(".0", "")
                if sc.startswith(code) or code in sc:
                    edinet_code = item.get("edinetCode") or item.get("edinet_code")
                    break
            if not edinet_code and isinstance(search_results, list) and len(search_results) == 1:
                edinet_code = search_results[0].get("edinetCode") or search_results[0].get("edinet_code")
            if not edinet_code:
                return None

            # 2. 企業詳細 + 財務時系列 + 指標を並列取得
            from concurrent.futures import ThreadPoolExecutor as _EdbPool
            def _get(url):
                r = requests.get(url, timeout=15, headers=_headers)
                return r.json() if r.status_code == 200 else None

            with _EdbPool(max_workers=3) as pool:
                f_company = pool.submit(_get, f"https://edinetdb.jp/v1/companies/{edinet_code}")
                f_fin = pool.submit(_get, f"https://edinetdb.jp/v1/companies/{edinet_code}/financials")
                f_ratios = pool.submit(_get, f"https://edinetdb.jp/v1/companies/{edinet_code}/ratios")

            company = f_company.result() or {}
            financials = f_fin.result()
            ratios = f_ratios.result()

            # 3. 統合データ構築
            data = {**company}
            if financials:
                data["financials"] = financials if isinstance(financials, list) else financials.get("data", financials.get("financials", []))
            if ratios:
                data["ratios"] = ratios if isinstance(ratios, list) else ratios.get("data", ratios.get("ratios", []))

            if data:
                _edb_save(code, data)
                return data
        except Exception:
            pass
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

    with ThreadPoolExecutor(max_workers=min(len(codes), 10)) as pool:
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
    """ウォッチリスト用: 株価＋テクニカル＋指数相関（5分キャッシュ）— 一括取得で高速化"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    codes = list(codes_tuple)
    tickers = [f"{c}.T" for c in codes]

    # ── 一括ダウンロード（1y + 5y）で個別API呼び出しを削減 ──
    _bulk_hist_1y = {}
    _bulk_hist_5y = {}
    try:
        # 1年分の株価を一括取得（OHLCV）
        _all_tickers = tickers + ["1306.T", "^N225"]
        df_1y = yf.download(_all_tickers, period="1y", progress=False, threads=True, group_by="ticker")
        if not df_1y.empty:
            for t in _all_tickers:
                try:
                    if isinstance(df_1y.columns, pd.MultiIndex) and t in df_1y.columns.get_level_values(0):
                        sub = df_1y[t].dropna(how="all")
                        if len(sub) > 0:
                            _bulk_hist_1y[t] = sub
                    elif len(_all_tickers) == 1:
                        _bulk_hist_1y[t] = df_1y.dropna(how="all")
                except Exception:
                    pass
    except Exception:
        pass

    try:
        # 5年分の株価を一括取得（年足陽線判定用）
        df_5y = yf.download(tickers, period="5y", progress=False, threads=True, group_by="ticker")
        if not df_5y.empty:
            for t in tickers:
                try:
                    if isinstance(df_5y.columns, pd.MultiIndex) and t in df_5y.columns.get_level_values(0):
                        sub = df_5y[t].dropna(how="all")
                        if len(sub) > 0:
                            _bulk_hist_5y[t] = sub
                    elif len(tickers) == 1:
                        _bulk_hist_5y[t] = df_5y.dropna(how="all")
                except Exception:
                    pass
    except Exception:
        pass

    # TOPIX・日経225の日次リターン（一括取得済みデータから抽出）
    _idx_returns = {}
    for idx_ticker, idx_name in [("1306.T", "TOPIX"), ("^N225", "日経225")]:
        ih = _bulk_hist_1y.get(idx_ticker)
        if ih is not None and len(ih) > 5 and "Close" in ih.columns:
            _idx_returns[idx_name] = ih["Close"].pct_change().dropna()

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
            # 一括取得済みデータを使用（個別API呼び出し不要）
            hist = _bulk_hist_1y.get(ticker)
            if hist is None:
                # フォールバック: 一括取得に失敗した銘柄のみ個別取得
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

        # 指数との相関・ベータ値
        res["corr_topix"] = None
        res["corr_n225"] = None
        res["beta_topix"] = None
        res["beta_n225"] = None
        if res.get("hist") is not None and len(res["hist"]) > 10:
            stock_ret = res["hist"]["Close"].pct_change().dropna()
            for idx_name, corr_key, beta_key in [("TOPIX", "corr_topix", "beta_topix"), ("日経225", "corr_n225", "beta_n225")]:
                idx_ret = _idx_returns.get(idx_name)
                if idx_ret is not None:
                    try:
                        merged = pd.DataFrame({"s": stock_ret, "i": idx_ret}).dropna()
                        if len(merged) > 10:
                            res[corr_key] = round(float(merged["s"].corr(merged["i"])), 2)
                            # ベータ値 = Cov(stock, index) / Var(index)
                            cov = merged["s"].cov(merged["i"])
                            var_idx = merged["i"].var()
                            if var_idx > 0:
                                res[beta_key] = round(float(cov / var_idx), 2)
                    except Exception:
                        pass

        # 年足陽線判定（直近3〜5年分）— 一括取得済みデータ使用
        res["yearly_bullish_count"] = 0
        res["yearly_bullish_label"] = ""
        try:
            hist_5y = _bulk_hist_5y.get(f"{code}.T")
            if hist_5y is None:
                # フォールバック: 一括取得に失敗した銘柄のみ個別取得
                hist_5y = yf.Ticker(f"{code}.T").history(period="5y")
            if hist_5y is not None and len(hist_5y) > 200:
                # 年ごとにグループ化して始値・終値を取得
                hist_5y.index = pd.to_datetime(hist_5y.index)
                yearly = hist_5y.groupby(hist_5y.index.year).agg(
                    Open=("Open", "first"),
                    Close=("Close", "last")
                )
                # 陽線判定（終値 > 始値）
                yearly["bullish"] = yearly["Close"] > yearly["Open"]
                # 直近の年は未確定なので除外（今年のデータ）
                current_year = pd.Timestamp.now().year
                completed_years = yearly[yearly.index < current_year]
                if len(completed_years) >= 2:
                    bullish_count = int(completed_years["bullish"].sum())
                    consecutive = 0
                    # 直近から連続陽線をカウント
                    for yr in sorted(completed_years.index, reverse=True):
                        if completed_years.loc[yr, "bullish"]:
                            consecutive += 1
                        else:
                            break
                    res["yearly_bullish_count"] = bullish_count
                    res["yearly_consecutive"] = consecutive
                    total_years = len(completed_years)
                    if consecutive >= 3:
                        res["yearly_bullish_label"] = f"🟢{consecutive}年連続陽線"
                    elif consecutive >= 2:
                        res["yearly_bullish_label"] = f"🔵{consecutive}年連続陽線"
                    elif bullish_count >= 2:
                        res["yearly_bullish_label"] = f"⚪{bullish_count}/{total_years}年陽線"
                    else:
                        res["yearly_bullish_label"] = f"🔴{bullish_count}/{total_years}年陽線"
        except Exception:
            pass

        return code, res

    with ThreadPoolExecutor(max_workers=min(len(codes), 10)) as pool:
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

        # 3. 一括削除（チェックボックス式）
        st.markdown("---")
        st.markdown("##### 🗑️ 銘柄を削除（チェックして削除）")
        if wl:
            del_targets = []
            # 3列でチェックボックス表示
            del_cols_ui = st.columns(3)
            for idx, (code, info) in enumerate(wl.items()):
                col = del_cols_ui[idx % 3]
                with col:
                    name = info.get("name", code)
                    if st.checkbox(f"{code} {name}", key=f"wl_del_chk_{code}"):
                        del_targets.append(code)

            if del_targets:
                st.warning(f"⚠️ {len(del_targets)}銘柄を削除します: {', '.join(del_targets)}")
                if st.button(f"❌ チェックした{len(del_targets)}銘柄を削除", key="wl_bulk_del", type="primary", use_container_width=True):
                    for c in del_targets:
                        wl.pop(c, None)
                    save_watchlist(wl)
                    st.success(f"{len(del_targets)}銘柄を削除しました")
                    st.rerun()
            else:
                st.caption("削除する銘柄にチェックを入れてください")

    if not wl:
        st.info("ウォッチリストに銘柄を追加してください。")
    else:
        codes = list(wl.keys())
        with st.spinner("ウォッチリスト分析中..."):
            fund_data = _wl_fetch_fundamental(tuple(codes))
            price_data = _wl_fetch_prices(tuple(codes))
            act_map = _wl_check_activist(tuple(codes))

            # ── MLモメンタムスコア計算 ──
            _ml_hist_dict = {}
            for c in codes:
                h = price_data.get(c, {}).get("hist")
                if h is not None and len(h) >= 80:
                    _ml_hist_dict[c] = h
            ml_scores, ml_model_info = calc_ml_momentum_scores(_ml_hist_dict)

        # マージ（price優先、fundで補完）+ 利回り計算
        wl_data = {}
        for c in codes:
            f = fund_data.get(c, {})
            p = price_data.get(c, {})
            d = {**f, **p, "code": c, "ticker": f"{c}.T"}
            # MLモメンタムスコアを追加
            d["ml_momentum"] = ml_scores.get(c)
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

        # ── 期待値スコア算出（売買シグナル+上位足+ファンダ+MLモメンタム総合） ──
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
            # 年足陽線ボーナス
            yb = d.get("yearly_bullish_count", 0)
            yc = d.get("yearly_consecutive", 0)
            if yc >= 3: s += 12
            elif yc >= 2: s += 8
            elif yb >= 2: s += 4
            elif yb == 0: s -= 5
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
            # MLモメンタムボーナス（予測将来リターンを期待値に反映）
            ml_mom = d.get("ml_momentum")
            if ml_mom is not None:
                # 予測リターン（%）をスコアに変換: ±5%程度が典型的
                s += np.clip(ml_mom * 3, -15, 15)
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

            # 💎🌟⭐ 優良銘柄マーク判定（段階式）
            # 条件: ①2年連続陽線 ②月足上昇 ③週足上昇 ④利回り2.5%以上 ⑤好業績
            _yc = d.get("yearly_consecutive", 0)
            _yb = d.get("yearly_bullish_count", 0)
            _mt = d.get("monthly_trend", "")
            _wt = d.get("weekly_trend", "")
            _dy = d.get("div_yield") or 0
            _eps = d.get("eps_cagr")
            _biz = d.get("biz_trend", "")
            _roe = d.get("roe") or 0

            # 各条件の判定
            _cond_yearly = _yc >= 1 or _yb >= 2  # 1年以上連続陽線 or 5年中2年以上陽線
            _cond_monthly = ("⬆️" in _mt or "↗️" in _mt)
            _cond_weekly = ("⬆️" in _wt or "↗️" in _wt)
            _cond_yield = _dy >= 2.5
            _cond_biz = ((_eps is not None and _eps > 0) or ("増" in _biz) or ("好" in _biz)
                         or _roe >= 10)

            _cond_count = sum([_cond_yearly, _cond_monthly, _cond_weekly, _cond_yield, _cond_biz])

            if _cond_count >= 5:
                gem_mark = "💎"
            elif _cond_count >= 4:
                gem_mark = "🌟"
            elif _cond_count >= 3:
                gem_mark = "⭐"
            else:
                gem_mark = ""
            d["_gem_mark"] = gem_mark
            d["_cond_count"] = _cond_count
            d["_cond_detail"] = (
                f"{'✅' if _cond_yearly else '❌'}年足 "
                f"{'✅' if _cond_monthly else '❌'}月足 "
                f"{'✅' if _cond_weekly else '❌'}週足 "
                f"{'✅' if _cond_yield else '❌'}利回{_dy:.1f}% "
                f"{'✅' if _cond_biz else '❌'}業績"
            )

            # 数値変換ヘルパー: 文字列(スクレイピング由来)や"－"等が混ざっても安全にfloat化する
            def _num(v):
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return float(v) if not (isinstance(v, float) and pd.isna(v)) else None
                try:
                    s = str(v).replace("倍", "").replace("%", "").replace(",", "").strip()
                    if s in ("", "—", "－", "-", "N/A", "nan", "None"):
                        return None
                    return float(s)
                except (TypeError, ValueError):
                    return None

            _per = _num(d.get("per"))
            _per3y = _num(d.get("per_3y"))
            _eps_cagr = _num(d.get("eps_cagr"))
            _roe = _num(d.get("roe"))
            _div_y = _num(d.get("div_yield"))
            _div_y3 = _num(d.get("div_yield_3y"))

            rows.append({
                "💎": gem_mark,
                "TOPIX": ('🟢追加' if wl.get(c,{}).get('topix_2026')=='TOPIX新規追加' else ('🔴除外' if wl.get(c,{}).get('topix_2026')=='TOPIX移行措置(除外)' else '')),
                "コード": c,
                "銘柄": d.get("name", wl[c].get("name", ""))[:8],
                "決算": ne_display,
                "業績": d.get("biz_trend", "—"),
                "年足": d.get("yearly_bullish_label", ""),
                "月足": d.get("monthly_trend", "—"),
                "週足": d.get("weekly_trend", "—"),
                "75MA": d.get("ma75_signal", ""),
                "株価": f"¥{d['last_price']:,.0f}" if d.get("last_price") else "—",
                "前日比": f"{d['daily_change_pct']:+.1f}%" if d.get("daily_change_pct") is not None else "—",
                "出来高": f"{vol_mark}×{d['vol_ratio']:.1f}" if d.get("vol_ratio") else "",
                "シグナル": d.get("signal", "—"),
                "期待値": expect,
                "ML予測": f"{d['ml_momentum']:+.2f}%" if d.get("ml_momentum") is not None else "—",
                "MACD": d.get("dj_signal", "—"),
                "押し目": wave_sig,
                "利回": f"{_div_y:.1f}%" if _div_y else "",
                "優待": yutai_val,
                "3Y利回": f"{_div_y3:.1f}%" if _div_y3 else "",
                "PER": f"{_per:.0f}" if _per else "",
                "PEG": f"{_per / _eps_cagr:.1f}" if _per and _eps_cagr and _eps_cagr > 0 else "",
                "3YPER": f"{_per3y:.0f}" if _per3y else "",
                "EPS成長": f"{_eps_cagr:+.0f}%" if _eps_cagr is not None else "",
                "ROE": f"{_roe:.0f}%" if _roe else "",
                "ACT": act_str,
                "β_T": f"{d['beta_topix']:.2f}" if d.get("beta_topix") is not None else "",
                "ρ_T": f"{d['corr_topix']:.2f}" if d.get("corr_topix") is not None else "",
                "β_N": f"{d['beta_n225']:.2f}" if d.get("beta_n225") is not None else "",
                "ρ_N": f"{d['corr_n225']:.2f}" if d.get("corr_n225") is not None else "",
                "テーマ": theme_str,
            })

        # 💎🌟⭐マーク銘柄数を表示
        gem5 = sum(1 for r in rows if r["💎"] == "💎")
        gem4 = sum(1 for r in rows if r["💎"] == "🌟")
        gem3 = sum(1 for r in rows if r["💎"] == "⭐")
        if gem5 + gem4 + gem3 > 0:
            parts = []
            if gem5: parts.append(f"💎完璧 {gem5}件")
            if gem4: parts.append(f"🌟優良 {gem4}件")
            if gem3: parts.append(f"⭐注目 {gem3}件")
            st.success(f"{' / '.join(parts)} — 年足陽線+月足週足↑+利回り+好業績（5条件中3つ以上）")

        df_wl = pd.DataFrame(rows)
        st.dataframe(df_wl, use_container_width=True, hide_index=True, height=min(len(rows)*38+40, 600),
                      column_config={
                          "期待値": st.column_config.ProgressColumn(
                              "期待値", min_value=-100, max_value=100, format="%d"
                          ),
                      })

        # ── MLモメンタムモデル情報 ──
        if ml_model_info and ml_model_info.get("n_features"):
            with st.expander("🤖 MLモメンタムモデル情報", expanded=False):
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("特徴量数", ml_model_info["n_features"])
                    st.metric("決定木数", ml_model_info["n_trees"])
                with mcol2:
                    st.markdown("**重要特徴量 Top5:**")
                    for fname, imp in ml_model_info.get("feature_importance", {}).items():
                        # 特徴量名を日本語で表示
                        _feat_jp = {
                            "ret_5d": "5日リターン", "ret_10d": "10日リターン",
                            "ret_20d": "20日リターン", "ret_60d": "60日リターン",
                            "vol_5d": "5日ボラ", "vol_20d": "20日ボラ", "vol_ratio": "ボラ比",
                            "volume_ratio": "出来高比", "rsi_14": "RSI(14)",
                            "dev_ma25": "MA25乖離", "dev_ma75": "MA75乖離",
                            "macd_hist": "MACDヒスト", "atr_norm": "ATR%",
                            "bb_position": "BB位置", "price_position_20d": "20日価格位置",
                        }
                        jp_name = _feat_jp.get(fname, fname)
                        st.text(f"  {jp_name}: {imp:.0f}")
                st.caption("💡 モデルは毎日自動再学習され、直近のマーケット状態に適応します。「ML予測」列は将来5営業日の予測リターン(%)です。")

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
            gem = f"{d.get('_gem_mark', '')} " if d.get("_gem_mark") else ""

            # 数値変換ヘルパー: 文字列(スクレイピング由来)や"－"等が混ざっても安全にfloat化する
            def _num_d(v):
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return float(v) if not (isinstance(v, float) and pd.isna(v)) else None
                try:
                    s = str(v).replace("倍", "").replace("%", "").replace(",", "").strip()
                    if s in ("", "—", "－", "-", "N/A", "nan", "None"):
                        return None
                    return float(s)
                except (TypeError, ValueError):
                    return None

            _d_per = _num_d(d.get("per"))
            _d_per3y = _num_d(d.get("per_3y"))
            _d_eps = _num_d(d.get("eps_cagr"))
            _d_roe = _num_d(d.get("roe"))
            _d_divy = _num_d(d.get("div_yield"))

            # PEGレシオ算出（PER ÷ EPS成長率）
            _peg = None
            if _d_per and _d_eps and _d_eps > 0:
                _peg = _d_per / _d_eps

            with st.expander(f"{gem}{mt}{wt} {('🟢TOPIX追加 ' if wl.get(c,{}).get('topix_2026')=='TOPIX新規追加' else ('🔴TOPIX除外 ' if wl.get(c,{}).get('topix_2026')=='TOPIX移行措置(除外)' else ''))}{'**'+ma75s+'** ' if ma75s else ''}**{c} {d.get('name','')}** {sig} 期待値:{expect} {dc_s}", expanded=False):
                # シグナルスコア・期待値・PEGを先頭に表示
                s1, s2, s3 = st.columns(3)
                s1.metric("シグナルスコア", f"{sig_score:+d}" if isinstance(sig_score, int) else f"{sig_score}")
                s2.metric("期待値スコア", f"{expect:+d}")
                s3.metric("PEGレシオ", f"{_peg:.2f}" if _peg is not None else "—",
                          help="PER÷EPS成長率。1.0未満で割安、1.0〜2.0で妥当、2.0超で割高の目安")

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("株価", f"¥{d['last_price']:,.0f}" if d.get("last_price") else "—", delta=f"{dc:+.2f}%" if dc else None)
                c2.metric("PER→3Y", f"{_d_per:.0f}→{_d_per3y:.0f}" if _d_per and _d_per3y else f"{_d_per:.0f}" if _d_per else "—")
                c3.metric("ROE", f"{_d_roe:.1f}%" if _d_roe else "—")
                c4.metric("利回り", f"{_d_divy:.2f}%" if _d_divy else "—")
                c5.metric("EPS成長", f"{_d_eps:+.1f}%" if _d_eps is not None else "—")
                c6.metric("MACD", d.get("dj_signal", "—"))

                # 年足陽線情報
                ybl = d.get("yearly_bullish_label", "")
                if ybl:
                    st.caption(f"📅 年足: {ybl}")
                # 優良銘柄条件の達成状況
                cond_detail = d.get("_cond_detail", "")
                if cond_detail and d.get("_gem_mark"):
                    st.caption(f"{d['_gem_mark']} 条件({d.get('_cond_count',0)}/5): {cond_detail}")

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

                # 指数連動性
                _has_idx = any(d.get(k) is not None for k in ["beta_topix", "corr_topix", "beta_n225", "corr_n225"])
                if _has_idx:
                    idx_parts = []
                    if d.get("beta_topix") is not None:
                        idx_parts.append(f"β(TOPIX)={d['beta_topix']:.2f}")
                    if d.get("corr_topix") is not None:
                        idx_parts.append(f"ρ(TOPIX)={d['corr_topix']:.2f}")
                    if d.get("beta_n225") is not None:
                        idx_parts.append(f"β(日経)={d['beta_n225']:.2f}")
                    if d.get("corr_n225") is not None:
                        idx_parts.append(f"ρ(日経)={d['corr_n225']:.2f}")
                    st.caption(f"📊 指数連動性: {' / '.join(idx_parts)}")

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

        # =====================================================================
        # ⚡ 決算 × 株価の乖離シグナル（好決算なのに売られた等を検出 + 相関度判定）
        # =====================================================================
        st.markdown("### ⚡ 決算 × 株価の乖離シグナル")
        st.caption(
            "決算の良し悪し（スコア）と発表後の株価反応（翌営業日リターン＞当日＞PTSの順で採用）を突き合わせ、"
            "乖離した銘柄に売買シグナルを付与します。相関度は『この相場が決算に素直か逆行か』を示します。"
        )

        import yfinance as _yf_sig

        @st.cache_data(ttl=1800, show_spinner=False)
        def _fetch_next_day_return(code, kessan_date_str):
            """決算日(kessan_date_str: YYYYMMDD)の翌営業日リターン(%)をyfinanceで取得。
            発表日終値→翌営業日終値の変化率。取得不可なら None。"""
            try:
                import pandas as _pd
                base = _pd.Timestamp(f"{kessan_date_str[:4]}-{kessan_date_str[4:6]}-{kessan_date_str[6:8]}")
                start = (base - _pd.Timedelta(days=4)).strftime("%Y-%m-%d")
                end = (base + _pd.Timedelta(days=8)).strftime("%Y-%m-%d")
                hist = _yf_sig.download(f"{code}.T", start=start, end=end, progress=False, threads=False)
                if hist is None or hist.empty or "Close" not in hist:
                    return None
                closes = hist["Close"].dropna()
                if hasattr(closes, "columns"):
                    closes = closes.iloc[:, 0]
                # 発表日(当日)以前で最も近い営業日のインデックスを探す
                idx = None
                for i, ts in enumerate(closes.index):
                    if ts.normalize() <= base.normalize():
                        idx = i
                if idx is None or idx + 1 >= len(closes):
                    return None
                base_close = float(closes.iloc[idx])
                next_close = float(closes.iloc[idx + 1])
                if base_close <= 0:
                    return None
                return round((next_close - base_close) / base_close * 100, 2)
            except Exception:
                return None

        # 株価反応(%)を各銘柄で決定: 翌営業日リターン > 当日前日比 > PTS の優先順
        _sig_rows = []
        for _e in evaluated:
            _pdp = _e.get("price_data") or {}
            _code = _e.get("code", "")
            _reaction = None
            _reaction_src = ""
            # 1) 翌営業日リターン（過去日付の決算なら正確）
            _nd = _fetch_next_day_return(_code, date_str) if _code and _code[:1].isdigit() else None
            if _nd is not None:
                _reaction = _nd; _reaction_src = "翌日"
            # 2) 当日前日比
            elif _pdp.get("current_change") is not None:
                _reaction = float(_pdp["current_change"]); _reaction_src = "当日"
            # 3) PTS
            elif _pdp.get("pts_change") is not None:
                _reaction = float(_pdp["pts_change"]); _reaction_src = "PTS"
            if _reaction is None:
                continue
            _score = _e.get("score", 3)
            # 決算の方向: score 4-5=good(+1), 3=neutral(0), 1-2=bad(-1)
            _earn_dir = 1 if _score >= 4 else (-1 if _score <= 2 else 0)
            _react_dir = 1 if _reaction > 1.0 else (-1 if _reaction < -1.0 else 0)
            _sig_rows.append({
                "code": _code, "name": _e.get("name", ""), "score": _score,
                "reaction": _reaction, "reaction_src": _reaction_src,
                "earn_dir": _earn_dir, "react_dir": _react_dir,
                "url": _e.get("url", f"https://kabutan.jp/stock/?code={_code}"),
            })

        if not _sig_rows:
            st.info("株価反応データが取得できる銘柄がありませんでした（場中・PTS・翌日リターンのいずれも未取得）。")
        else:
            # ---- 相関度（この相場の素直さ） ----
            # 決算方向と反応方向が両方0でないペアで一致率を計算
            _paired = [r for r in _sig_rows if r["earn_dir"] != 0 and r["react_dir"] != 0]
            _agree = sum(1 for r in _paired if r["earn_dir"] == r["react_dir"])
            _n_paired = len(_paired)
            if _n_paired >= 3:
                _agree_pct = _agree / _n_paired * 100
                # 相関係数（決算スコアを中心化した値 × 反応）で連続的に評価
                import numpy as _np_sig
                _xs = _np_sig.array([r["score"] - 3 for r in _paired], dtype=float)
                _ys = _np_sig.array([r["reaction"] for r in _paired], dtype=float)
                _corr = None
                if _xs.std() > 0 and _ys.std() > 0:
                    _corr = float(_np_sig.corrcoef(_xs, _ys)[0, 1])
                if _agree_pct >= 65:
                    _mood = "🟢 素直な相場（決算の良し悪しに株価が順当に反応）"
                elif _agree_pct <= 40:
                    _mood = "🔴 逆行相場（好決算売り・悪決算買いが多い＝セル・ザ・ファクト傾向）"
                else:
                    _mood = "🟡 まちまち（決算と株価反応の関係が弱い）"
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("反応方向の一致率", f"{_agree_pct:.0f}%", help="決算の良し悪しと株価の上下が一致した銘柄の割合")
                _mc2.metric("相関係数", f"{_corr:+.2f}" if _corr is not None else "—",
                            help="決算スコアと株価反応リターンのピアソン相関。+1に近いほど素直、-1に近いほど逆行")
                _mc3.metric("対象", f"{_n_paired}社", help="決算・株価反応ともに方向が明確な銘柄数")
                st.markdown(f"**地合い判定: {_mood}**")
            else:
                st.caption(f"相関判定には方向が明確な銘柄が3社以上必要です（現在{_n_paired}社）。")

            # ---- 乖離シグナル銘柄の抽出 ----
            # 好決算なのに売られた（押し目買い候補）
            _good_sold = [r for r in _sig_rows if r["earn_dir"] == 1 and r["reaction"] < -1.0]
            _good_sold.sort(key=lambda r: r["reaction"])
            # 悪決算なのに買われた（戻り売り/天井警戒）
            _bad_bought = [r for r in _sig_rows if r["earn_dir"] == -1 and r["reaction"] > 1.0]
            _bad_bought.sort(key=lambda r: r["reaction"], reverse=True)
            # 素直な好決算×買われ（順張り継続候補）
            _good_bought = [r for r in _sig_rows if r["earn_dir"] == 1 and r["reaction"] > 1.0]
            _good_bought.sort(key=lambda r: r["reaction"], reverse=True)

            def _render_sig_card(r, border, bg, badge_text, badge_bg, action):
                _rc = "#00c853" if r["reaction"] > 0 else "#f44336" if r["reaction"] < 0 else "#888"
                _arrow = "🚀" if r["reaction"] > 5 else "↑" if r["reaction"] > 0 else "💥" if r["reaction"] < -5 else "↓" if r["reaction"] < 0 else ""
                st.markdown(
                    f'<div style="padding:10px 14px;margin-bottom:6px;border-radius:8px;'
                    f'border-left:5px solid {border};background:{bg};">'
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
                    f'<span style="background:{badge_bg};color:#fff;font-size:0.7rem;font-weight:bold;'
                    f'padding:2px 8px;border-radius:4px;">{badge_text}</span>'
                    f'<a href="{r["url"]}" target="_blank" style="color:inherit;text-decoration:none;font-weight:bold;">'
                    f'{r["code"]} {r["name"]}</a>'
                    f'<span style="font-size:0.85rem;">決算{"⭐"*r["score"]}</span>'
                    f'<span style="color:{_rc};font-weight:bold;margin-left:4px;">'
                    f'{r["reaction_src"]}反応 {r["reaction"]:+.1f}%{_arrow}</span>'
                    f'</div>'
                    f'<div style="margin-top:3px;font-size:0.82rem;opacity:0.85;">{action}</div>'
                    f'</div>', unsafe_allow_html=True
                )

            if _good_sold:
                st.markdown("#### 🔵 好決算なのに売られた（押し目買い候補）")
                st.caption("決算は良好だが株価は下落。期待先行の利益確定売り（セル・ザ・ファクト）の可能性。地合いが素直なら反発を狙える。")
                for r in _good_sold:
                    _act = "🔵 買いシグナル: 好決算×逆行下落。相関が高い相場では押し目買い妙味。下げ止まり確認を推奨。"
                    _render_sig_card(r, "#2196f3", "rgba(33,150,243,0.10)", "好決算×売られ", "#2196f3", _act)

            if _bad_bought:
                st.markdown("#### 🟠 悪決算なのに買われた（戻り売り・警戒）")
                st.caption("決算は不調だが株価は上昇。悪材料出尽くし or 需給要因の可能性。持続性は低いことが多い。")
                for r in _bad_bought:
                    _act = "🟠 売り/警戒シグナル: 悪決算×逆行上昇。戻り売り・利確候補。飛びつき買いは避ける。"
                    _render_sig_card(r, "#ff9800", "rgba(255,152,0,0.10)", "悪決算×買われ", "#ff9800", _act)

            if _good_bought:
                with st.expander(f"🟢 素直な好決算×買われ（順張り継続候補 {len(_good_bought)}社）", expanded=False):
                    for r in _good_bought:
                        _act = "🟢 順張りシグナル: 決算も株価も好反応。トレンド継続を想定。押し目待ちか打診買い。"
                        _render_sig_card(r, "#00c853", "rgba(0,200,83,0.08)", "好決算×買われ", "#00c853", _act)

            if not (_good_sold or _bad_bought or _good_bought):
                st.info("明確な乖離シグナル銘柄は検出されませんでした。")

        st.markdown("---")

        # =====================================================================
        # 🎲 決算マタギ適性診断（過去の決算ギャップ統計からマタギ可否を判定）
        # =====================================================================
        st.markdown("### 🎲 決算マタギ適性診断")
        st.caption(
            "過去の決算発表ごとの『翌営業日ギャップ（発表日終値→翌営業日終値）』を集計し、"
            "この銘柄が決算をまたいで保有してよいタイプかを過去実績から判定します。"
        )

        import yfinance as _yf_mg
        import numpy as _np_mg
        import pandas as _pd_mg

        @st.cache_data(ttl=86400, show_spinner=False)
        def _fetch_earnings_gaps(code, lookback=8):
            """過去の決算日ごとの翌営業日ギャップ(%)を取得。
            戻り値: {"gaps":[{date,gap}], "error":str or None}"""
            try:
                tk = _yf_mg.Ticker(f"{code}.T")
                # 過去の決算日を取得
                edf = None
                try:
                    edf = tk.get_earnings_dates(limit=24)
                except Exception:
                    edf = None
                if edf is None or len(edf) == 0:
                    return {"gaps": [], "error": "決算日データが取得できませんでした（yfinance未対応銘柄の可能性）"}
                now = _pd_mg.Timestamp.now(tz=edf.index.tz) if edf.index.tz is not None else _pd_mg.Timestamp.now()
                past_dates = sorted([d for d in edf.index if d <= now], reverse=True)[:lookback]
                if not past_dates:
                    return {"gaps": [], "error": "過去の決算日が見つかりませんでした"}
                # 価格履歴（最古の決算日の少し前〜現在）
                oldest = min(past_dates)
                start = (oldest - _pd_mg.Timedelta(days=6)).strftime("%Y-%m-%d")
                hist = _yf_mg.download(f"{code}.T", start=start, progress=False, threads=False)
                if hist is None or hist.empty or "Close" not in hist:
                    return {"gaps": [], "error": "株価履歴が取得できませんでした"}
                closes = hist["Close"].dropna()
                if hasattr(closes, "columns"):
                    closes = closes.iloc[:, 0]
                closes.index = _pd_mg.to_datetime(closes.index).tz_localize(None)
                gaps = []
                for d in sorted(past_dates):
                    dd = _pd_mg.Timestamp(d).tz_localize(None).normalize()
                    # 発表日(以前で最も近い営業日)と翌営業日
                    before = closes[closes.index <= dd]
                    after = closes[closes.index > dd]
                    if len(before) == 0 or len(after) == 0:
                        continue
                    base_close = float(before.iloc[-1])
                    next_close = float(after.iloc[0])
                    if base_close <= 0:
                        continue
                    gap = round((next_close - base_close) / base_close * 100, 2)
                    gaps.append({"date": dd.strftime("%Y-%m-%d"), "gap": gap})
                return {"gaps": gaps, "error": None}
            except Exception as _ex:
                return {"gaps": [], "error": f"取得エラー: {_ex}"}

        def _judge_matagi(gaps):
            """ギャップ統計からマタギ適性を判定。"""
            arr = _np_mg.array([g["gap"] for g in gaps], dtype=float)
            n = len(arr)
            mean = float(arr.mean())
            median = float(_np_mg.median(arr))
            std = float(arr.std(ddof=1)) if n >= 2 else 0.0
            win = int((arr > 0).sum())
            win_rate = win / n * 100
            worst = float(arr.min())
            best = float(arr.max())
            big_drop = int((arr <= -5).sum())   # 5%超の急落回数
            # --- 適性スコア（0=非推奨, 100=跨ぎ向き）---
            s = 50.0
            s += (win_rate - 50) * 0.6          # 勝率
            s += mean * 3.0                      # 平均ギャップ
            s -= std * 1.5                       # ボラティリティ（高いほど減点）
            s += worst * 1.2                     # 最大下落（大きいマイナスほど減点）
            s -= big_drop / n * 30               # 急落頻度
            s = max(0, min(100, s))
            if s >= 65:
                verdict = "🟢 マタギ向き"
                detail = "過去実績では決算で上がりやすく、下落リスクも限定的。跨いで保有する妙味あり。"
            elif s >= 50:
                verdict = "🔵 条件付き可"
                detail = "ややプラス寄り。ポジションを抑えめにすれば跨ぎ可。"
            elif s >= 35:
                verdict = "🟡 中立〜慎重"
                detail = "方向感が乏しくボラも相応。無理に跨ぐ必要は薄い。"
            else:
                verdict = "🔴 マタギ非推奨"
                detail = "過去実績で下落・急落が目立つ、またはギャンブル性が高い。決算前にポジション整理を推奨。"
            return {
                "n": n, "mean": mean, "median": median, "std": std,
                "win_rate": win_rate, "worst": worst, "best": best,
                "big_drop": big_drop, "score": round(s), "verdict": verdict, "detail": detail,
            }

        # 診断対象の候補（評価済み銘柄 + 手入力）
        _eval_codes = [e.get("code") for e in evaluated if e.get("code") and e.get("code")[:1].isdigit()]
        _mg_c1, _mg_c2 = st.columns([3, 1])
        with _mg_c1:
            _mg_pick = st.selectbox(
                "診断する銘柄（当日決算発表銘柄から選択）",
                options=["（手入力）"] + [f"{e.get('code')} {e.get('name','')}" for e in evaluated if e.get("code") and e.get("code")[:1].isdigit()],
                key="matagi_pick",
            )
        with _mg_c2:
            _mg_manual = st.text_input("コード手入力", value="", key="matagi_manual", placeholder="例: 7203")

        _mg_code = ""
        if _mg_manual.strip():
            _mg_code = _mg_manual.strip()
        elif _mg_pick and _mg_pick != "（手入力）":
            _mg_code = _mg_pick.split(" ")[0]

        if st.button("🎲 マタギ適性を診断", key="matagi_run"):
            if not _mg_code:
                st.warning("銘柄を選択するかコードを入力してください。")
            else:
                with st.spinner(f"{_mg_code} の過去決算ギャップを集計中..."):
                    _mg_res = _fetch_earnings_gaps(_mg_code)
                if _mg_res["error"]:
                    st.error(_mg_res["error"])
                elif len(_mg_res["gaps"]) < 3:
                    st.warning(f"集計できた決算回数が{len(_mg_res['gaps'])}回のみで、判定には不十分です（3回以上必要）。")
                else:
                    _j = _judge_matagi(_mg_res["gaps"])
                    _vc = "#00c853" if _j["score"] >= 65 else "#2196f3" if _j["score"] >= 50 else "#ff9800" if _j["score"] >= 35 else "#f44336"
                    st.markdown(
                        f'<div style="padding:12px 16px;border-radius:8px;border-left:6px solid {_vc};'
                        f'background:rgba(128,128,128,0.08);margin-bottom:8px;">'
                        f'<div style="font-size:1.1rem;font-weight:bold;color:{_vc};">{_j["verdict"]}（適性スコア {_j["score"]}/100）</div>'
                        f'<div style="font-size:0.9rem;margin-top:4px;opacity:0.9;">{_j["detail"]}</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                    _k1, _k2, _k3, _k4 = st.columns(4)
                    _k1.metric("勝率（翌日上昇）", f"{_j['win_rate']:.0f}%", help=f"過去{_j['n']}回の決算のうち翌営業日が上昇した割合")
                    _k2.metric("平均ギャップ", f"{_j['mean']:+.1f}%", help="翌営業日リターンの平均")
                    _k3.metric("最大下落", f"{_j['worst']:+.1f}%", help="過去の決算翌日で最も下げた回")
                    _k4.metric("ボラティリティ", f"{_j['std']:.1f}%", help="ギャップの標準偏差。大きいほど予測困難")
                    _k5, _k6 = st.columns(2)
                    _k5.metric("最大上昇", f"{_j['best']:+.1f}%")
                    _k6.metric("5%超の急落回数", f"{_j['big_drop']}/{_j['n']}回")
                    # 明細
                    _df_mg = _pd_mg.DataFrame(_mg_res["gaps"])
                    _df_mg = _df_mg.rename(columns={"date": "決算日", "gap": "翌日ギャップ(%)"})
                    _df_mg = _df_mg.iloc[::-1].reset_index(drop=True)
                    st.dataframe(
                        _df_mg, use_container_width=True, hide_index=True,
                        column_config={"翌日ギャップ(%)": st.column_config.NumberColumn("翌日ギャップ(%)", format="%+.2f")},
                    )
                    st.caption(
                        "※ 翌営業日ギャップ＝決算発表日の終値→翌営業日終値の変化率。"
                        "yfinanceの決算日データに基づくため、場中発表/引け後発表の差や過去データの欠落で誤差が出る場合があります。"
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


# =====================================================================
# Tab 15: 決算カレンダー（当日〜明後日）
# =====================================================================
with tab15:
    st.subheader("📅 決算カレンダー — 当日〜明後日の決算発表予定")
    st.caption("株探の決算発表スケジュールから、当日・明日・明後日に決算発表が予定されている銘柄を一覧表示します。")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _fetch_earnings_schedule(date_str):
        """株探のニュース記事・決算速報ページから決算予定銘柄を取得"""
        import re as _re
        rows = []
        existing_codes = set()
        total_companies = 0

        # ソース1: kabutan カテゴリ9（決算発表予定ニュース記事）から取得
        # 当日〜翌日の決算予定が記事として公開される
        try:
            cat_url = "https://kabutan.jp/news/marketnews/?category=9"
            resp = requests.get(cat_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # 対象日付の記事リンクを探す
                target_date = datetime.strptime(date_str, "%Y%m%d")
                target_month = target_date.month
                target_day = target_date.day
                target_pattern = f"{target_month}月{target_day}日"

                links = soup.find_all("a", href=True)
                article_url = None
                for a in links:
                    text = a.get_text(strip=True)
                    href = a.get("href", "")
                    if "決算発表予定" in text and target_pattern in text and "/news/" in href:
                        article_url = href if href.startswith("http") else f"https://kabutan.jp{href}"
                        # 記事タイトルから総社数を抽出
                        m = _re.search(r"(\d+)社", text)
                        if m:
                            total_companies = int(m.group(1))
                        break

                if article_url:
                    resp2 = requests.get(article_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                    resp2.encoding = resp2.apparent_encoding
                    if resp2.status_code == 200:
                        soup2 = BeautifulSoup(resp2.text, "html.parser")
                        article = soup2.select_one("div#newstext") or soup2.select_one("div.body_text")
                        if article:
                            text = article.get_text()
                            for line in text.split("\n"):
                                line = line.strip()
                                m = _re.match(r"[★○●■◆\s]*<(\d{4})>\s*(.+?)\s*\[(.+?)\](.*?)$", line)
                                if m:
                                    code = m.group(1)
                                    if code in existing_codes:
                                        continue
                                    name = m.group(2).strip()
                                    market = m.group(3).strip()
                                    forecast = m.group(4).strip()
                                    # 注目マーク
                                    attention = ""
                                    if "★" in line[:5]:
                                        attention = "★注目"
                                    elif "○" in line[:5]:
                                        attention = "○増益"
                                    elif "●" in line[:5]:
                                        attention = "●減益"
                                    existing_codes.add(code)
                                    rows.append({
                                        "コード": code,
                                        "銘柄名": name,
                                        "市場": market,
                                        "予想": forecast,
                                        "注目": attention,
                                        "URL": f"https://kabutan.jp/stock/?code={code}",
                                    })
        except Exception:
            pass

        # ソース2: kabutan mode=4_1 全ページ（当日の決算発表銘柄）
        # 当日分のみ有効（翌日以降は同じデータが返る）
        target_date_obj = datetime.strptime(date_str, "%Y%m%d").date()
        if target_date_obj == datetime.now().date():
            try:
                for page in range(1, 30):
                    url = f"https://kabutan.jp/warning/?mode=4_1&page={page}"
                    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                    resp.encoding = resp.apparent_encoding
                    if resp.status_code != 200:
                        break
                    soup = BeautifulSoup(resp.text, "html.parser")
                    tables = soup.select("table.stock_table")
                    if not tables:
                        break
                    page_count = 0
                    trs = tables[0].find_all("tr")
                    for tr in trs:
                        tds = tr.find_all("td")
                        ths = tr.find_all("th")
                        if len(tds) < 4:
                            continue
                        code = tds[0].get_text(strip=True)
                        if not _re.match(r"^[0-9]{4}$", code):
                            continue
                        if code in existing_codes:
                            continue
                        name = ""
                        for th in ths:
                            t = th.get_text(strip=True)
                            if t and len(t) >= 2 and not _re.match(r"^\d", t):
                                name = t
                                break
                        if not name:
                            name = STOCK_DICT.get(f"{code}.T", code)
                        market = tds[1].get_text(strip=True) if len(tds) > 1 else ""
                        headline = tds[3].get_text(strip=True) if len(tds) > 3 else ""
                        existing_codes.add(code)
                        page_count += 1
                        rows.append({
                            "コード": code,
                            "銘柄名": name,
                            "市場": market,
                            "予想": headline,
                            "注目": "",
                            "URL": f"https://kabutan.jp/stock/?code={code}",
                        })
                    if page_count == 0:
                        break
            except Exception:
                pass
            if not total_companies:
                total_companies = len(rows)

        return rows, total_companies

    # --- UI ---
    today = datetime.now().date()
    dates = [today, today + timedelta(days=1), today + timedelta(days=2)]
    date_labels = ["📌 当日", "📆 明日", "📆 明後日"]

    cal_btn = st.button("🔍 決算予定を取得", key="cal_fetch", use_container_width=False)

    if cal_btn:
        all_results = {}
        with st.spinner("決算予定を取得中..."):
            for d, label in zip(dates, date_labels):
                d_str = d.strftime("%Y%m%d")
                schedule, total = _fetch_earnings_schedule(d_str)
                all_results[label] = {"date": d, "data": schedule, "total": total}
        st.session_state["earnings_calendar"] = all_results

    # 表示
    cal_data = st.session_state.get("earnings_calendar", {})
    if cal_data:
        total_count = sum(v.get("total", len(v["data"])) for v in cal_data.values())
        st.success(f"📅 合計 {total_count} 社の決算発表予定を取得しました")

        for label, info in cal_data.items():
            d = info["date"]
            data = info["data"]
            total = info.get("total", len(data))
            weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][d.weekday()]
            total_str = f"{total}社予定" if total > len(data) else f"{len(data)}社"
            header = f"{label} — {d.strftime('%Y/%m/%d')}（{weekday_jp}）  [{total_str} / 取得{len(data)}社]"

            with st.expander(header, expanded=(label == "📌 当日")):
                if not data:
                    st.info("この日の決算発表予定はありません（土日祝の可能性、または記事未公開）")
                else:
                    if total > len(data):
                        st.caption(f"📝 全{total}社中、注目{len(data)}社を表示（株探ニュース記事より）")
                    df_cal = pd.DataFrame(data)
                    # ポートフォリオ・ウォッチリスト銘柄をハイライト
                    portfolio_codes = set()
                    watchlist_codes = set()
                    try:
                        pf = load_portfolio()
                        portfolio_codes = {s.get("code", "").replace(".T", "") for s in pf}
                    except Exception:
                        pass
                    try:
                        wl = load_watchlist()
                        watchlist_codes = {s.get("code", "").replace(".T", "") for s in wl}
                    except Exception:
                        pass

                    # マーク列を追加
                    marks = []
                    for _, row in df_cal.iterrows():
                        code = row["コード"]
                        m = ""
                        if code in portfolio_codes:
                            m += "📋"
                        if code in watchlist_codes:
                            m += "👁️"
                        marks.append(m)
                    df_cal.insert(0, "保有", marks)

                    # 市場でフィルタ
                    markets_available = sorted(df_cal["市場"].unique().tolist())
                    if len(markets_available) > 1:
                        selected_markets = st.multiselect(
                            "市場フィルタ",
                            markets_available,
                            default=markets_available,
                            key=f"cal_filter_{d.strftime('%Y%m%d')}"
                        )
                        df_cal = df_cal[df_cal["市場"].isin(selected_markets)]

                    st.dataframe(
                        df_cal,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "URL": st.column_config.LinkColumn("詳細", display_text="📊開く"),
                        },
                        height=min(len(df_cal) * 35 + 40, 600)
                    )

                    # ポートフォリオ/ウォッチリスト銘柄のハイライト
                    highlighted = df_cal[df_cal["保有"] != ""]
                    if len(highlighted) > 0:
                        st.markdown("#### ⭐ 保有・ウォッチ銘柄の決算予定")
                        for _, row in highlighted.iterrows():
                            icon = row["保有"]
                            st.markdown(
                                f"- {icon} **{row['コード']}** {row['銘柄名']} "
                                f"— {row['予想']}"
                            )
    else:
        st.info("「🔍 決算予定を取得」ボタンを押すと、当日〜明後日の決算発表予定銘柄を表示します。")


# =====================================================================
# Tab 16: 業種別決算選別
# =====================================================================

def _render_earnings_stock_card(r):
    """決算銘柄カードを描画するヘルパー関数"""
    code = r["code"]
    name = r["name"]
    headline = r["headline"]
    earn_color = r["earn_color"]
    classification = r["classification"]
    sector = r["sector"]
    change = r.get("change", "")

    # 決算評価・ガイダンス情報
    eval_info = r.get("eval_info", {})
    guidance_info = r.get("guidance_info", {})

    # 株価変動の色
    chg_color = "#888"
    if change:
        if "+" in change or "↑" in change:
            chg_color = "#00c853"
        elif "-" in change or "↓" in change or "▼" in change:
            chg_color = "#f44336"

    # テーマ推定
    ticker = f"{code}.T"
    themes = guess_themes_by_name(ticker, name)
    theme_html = ""
    if themes:
        for th in themes[:3]:
            theme_html += (
                f"<span style='display:inline-block;padding:1px 5px;margin:1px;"
                f"background:#1a237e;border-radius:6px;color:#7986cb;"
                f"font-size:0.65rem;'>{th}</span>"
            )

    # 決算評価の詳細HTML
    eval_html = ""
    if eval_info:
        parts = []
        # 売上
        rev = eval_info.get("revenue_result")
        rev_yoy = eval_info.get("revenue_yoy")
        if rev:
            rev_color = "#00c853" if rev_yoy and rev_yoy > 0 else "#f44336" if rev_yoy and rev_yoy < 0 else "#aaa"
            yoy_str = f"({rev_yoy:+.1f}%)" if rev_yoy is not None else ""
            parts.append(f"<span style='color:{rev_color};'>売上{rev}{yoy_str}</span>")
        # 営業利益
        op = eval_info.get("op_profit_result")
        op_yoy = eval_info.get("op_profit_yoy")
        if op:
            op_color = "#00c853" if op_yoy and op_yoy > 0 else "#f44336" if op_yoy and op_yoy < 0 else "#aaa"
            yoy_str = f"({op_yoy:+.1f}%)" if op_yoy is not None else ""
            parts.append(f"<span style='color:{op_color};'>営利{op}{yoy_str}</span>")
        # 純利益
        net = eval_info.get("net_profit_result")
        net_yoy = eval_info.get("net_profit_yoy")
        if net:
            net_color = "#00c853" if net_yoy and net_yoy > 0 else "#f44336" if net_yoy and net_yoy < 0 else "#aaa"
            yoy_str = f"({net_yoy:+.1f}%)" if net_yoy is not None else ""
            parts.append(f"<span style='color:{net_color};'>純利{net}{yoy_str}</span>")
        # 進捗率
        progress = eval_info.get("progress_rate")
        if progress is not None:
            prog_color = "#00c853" if progress >= 80 else "#ff9800" if progress >= 60 else "#f44336"
            parts.append(f"<span style='color:{prog_color};'>進捗{progress:.0f}%</span>")
        if parts:
            eval_html = f"<div style='margin-top:3px;font-size:0.72rem;'>{' / '.join(parts)}</div>"

    # 来期ガイダンスHTML
    guidance_html = ""
    if guidance_info:
        g_parts = []
        # 来期売上予想
        next_rev = guidance_info.get("next_revenue")
        next_rev_chg = guidance_info.get("next_revenue_change")
        if next_rev:
            chg_str = f"({next_rev_chg:+.1f}%)" if next_rev_chg is not None else ""
            g_color = "#00c853" if next_rev_chg and next_rev_chg > 0 else "#f44336" if next_rev_chg and next_rev_chg < 0 else "#aaa"
            g_parts.append(f"<span style='color:{g_color};'>売上{next_rev}{chg_str}</span>")
        # 来期営業利益予想
        next_op = guidance_info.get("next_op_profit")
        next_op_chg = guidance_info.get("next_op_profit_change")
        if next_op:
            chg_str = f"({next_op_chg:+.1f}%)" if next_op_chg is not None else ""
            g_color = "#00c853" if next_op_chg and next_op_chg > 0 else "#f44336" if next_op_chg and next_op_chg < 0 else "#aaa"
            g_parts.append(f"<span style='color:{g_color};'>営利{next_op}{chg_str}</span>")
        # 来期純利益予想
        next_net = guidance_info.get("next_net_profit")
        next_net_chg = guidance_info.get("next_net_profit_change")
        if next_net:
            chg_str = f"({next_net_chg:+.1f}%)" if next_net_chg is not None else ""
            g_color = "#00c853" if next_net_chg and next_net_chg > 0 else "#f44336" if next_net_chg and next_net_chg < 0 else "#aaa"
            g_parts.append(f"<span style='color:{g_color};'>純利{next_net}{chg_str}</span>")
        # 来期EPS
        next_eps = guidance_info.get("next_eps")
        if next_eps:
            g_parts.append(f"<span style='color:#aaa;'>EPS {next_eps}</span>")
        # 配当予想
        next_div = guidance_info.get("next_dividend")
        if next_div:
            g_parts.append(f"<span style='color:#aaa;'>配当{next_div}</span>")
        # ガイダンス総合判定
        guidance_label = guidance_info.get("guidance_label", "")
        guidance_label_color = guidance_info.get("guidance_label_color", "#888")

        if g_parts or guidance_label:
            label_html = f"<span style='color:{guidance_label_color};font-weight:bold;margin-right:6px;'>{guidance_label}</span>" if guidance_label else ""
            guidance_html = f"<div style='margin-top:2px;font-size:0.72rem;'>📋来期: {label_html}{' / '.join(g_parts)}</div>"

    st.markdown(
        f"<div style='background:#1a1a2e;padding:10px 12px;border-radius:6px;"
        f"margin-bottom:4px;border-left:4px solid {earn_color};'>"
        f"<div style='display:grid;grid-template-columns:180px 90px 1fr 100px;align-items:center;gap:8px;'>"
        f"<div>"
        f"<a href='https://kabutan.jp/stock/?code={code}' target='_blank' "
        f"style='color:#fff;text-decoration:none;font-weight:600;font-size:0.88rem;'>{name}</a>"
        f" <span style='color:#888;font-size:0.72rem;'>{code}</span>"
        f"<div style='color:#888;font-size:0.68rem;'>{sector}</div>"
        f"</div>"
        f"<div style='text-align:center;'>"
        f"<span style='color:{earn_color};font-weight:bold;font-size:0.82rem;'>{classification}</span>"
        f"</div>"
        f"<div style='color:#ccc;font-size:0.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
        f"{headline}"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"<span style='color:{chg_color};font-size:0.82rem;'>{change}</span>"
        f"</div>"
        f"</div>"
        + eval_html
        + guidance_html
        + (f"<div style='margin-top:3px;'>{theme_html}</div>" if theme_html else "")
        + f"</div>",
        unsafe_allow_html=True
    )


with tab16:
    st.subheader("🏭 業種別決算選別 — 決算シーズンの勝ち組・負け組を業種で仕分け")
    st.caption("直近の決算発表結果を業種別に分類し、好決算・悪決算を一目で把握。購入候補の選定に活用できます。")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _fetch_sector_earnings_data():
        """株探の決算速報から直近の決算発表銘柄を取得し、業種情報を付与"""
        import re as _re

        all_rows = []
        existing_codes = set()

        def _parse_table(soup):
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
                name = ""
                for th in ths:
                    t = th.get_text(strip=True)
                    if t and len(t) >= 2 and not _re.match(r"^\d", t):
                        name = t
                        break
                if not name:
                    name = STOCK_DICT.get(f"{code}.T", code)
                headline = tds[3].get_text(strip=True) if len(tds) > 3 else ""
                price_str = tds[4].get_text(strip=True) if len(tds) > 4 else ""
                change_str = tds[7].get_text(strip=True) if len(tds) > 7 else ""
                existing_codes.add(code)
                parsed.append({
                    "code": code, "name": name, "headline": headline,
                    "price": price_str, "change": change_str,
                })
            return parsed

        # 前日引け後の決算発表（mode=4_3）
        try:
            resp = requests.get("https://kabutan.jp/warning/?mode=4_3", timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code == 200:
                all_rows.extend(_parse_table(BeautifulSoup(resp.text, "html.parser")))
        except Exception:
            pass

        # 当日場中の決算発表（mode=4_2）
        try:
            resp = requests.get("https://kabutan.jp/warning/?mode=4_2", timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code == 200:
                all_rows.extend(_parse_table(BeautifulSoup(resp.text, "html.parser")))
        except Exception:
            pass

        # 前日場中の決算発表（mode=4_1 ページ1-5）
        try:
            for page in range(1, 6):
                url = f"https://kabutan.jp/warning/?mode=4_1&page={page}"
                resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                resp.encoding = resp.apparent_encoding
                if resp.status_code != 200:
                    break
                parsed = _parse_table(BeautifulSoup(resp.text, "html.parser"))
                if not parsed:
                    break
                all_rows.extend(parsed)
        except Exception:
            pass

        return all_rows

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_sector_map():
        """JPX上場銘柄一覧から証券コード→業種のマッピングを取得"""
        try:
            stock_list = _topix_fetch_stock_list()
            if stock_list is not None and not stock_list.empty:
                return dict(zip(stock_list["code"], stock_list["sector"]))
        except Exception:
            pass
        return {}

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_earnings_and_guidance(code):
        """株探の決算ページから今期実績と来期ガイダンスを取得"""
        import re as _re
        result = {
            "eval_info": {},
            "guidance_info": {},
        }

        try:
            # 株探の業績ページ
            url = f"https://kabutan.jp/stock/finance/?code={code}"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.encoding = resp.apparent_encoding
            if resp.status_code != 200:
                return result

            soup = BeautifulSoup(resp.text, "html.parser")

            # 業績テーブルを探す
            tables = soup.select("table")
            finance_table = None
            for table in tables:
                ths = [th.get_text(strip=True) for th in table.select("th")]
                # 「売上高」「営業益」「経常益」「最終益」を含むテーブル
                if any("売上" in h for h in ths) and any("営業" in h or "営利" in h for h in ths):
                    finance_table = table
                    break

            if finance_table:
                rows = finance_table.select("tr")
                # ヘッダー行を特定
                header_row = None
                data_rows = []
                for row in rows:
                    ths = row.select("th")
                    tds = row.select("td")
                    if ths and not tds:
                        header_row = [th.get_text(strip=True) for th in ths]
                    elif tds:
                        row_data = {}
                        # 最初のth or tdが期の情報
                        first_cell = row.select_one("th") or row.select_one("td")
                        if first_cell:
                            row_data["period"] = first_cell.get_text(strip=True)
                        cells = [td.get_text(strip=True) for td in tds]
                        row_data["cells"] = cells
                        data_rows.append(row_data)

                # 直近の実績行と予想行を探す
                actual_row = None
                forecast_row = None
                next_forecast_row = None

                for i, dr in enumerate(data_rows):
                    period = dr.get("period", "")
                    # 「予」がついている行は予想
                    if "予" in period:
                        if forecast_row is None:
                            forecast_row = dr
                        else:
                            next_forecast_row = dr
                    elif "実" in period or ("連" in period and "予" not in period):
                        actual_row = dr

                # 数値パース関数
                def _parse_num(s):
                    if not s or s in ("-", "—", "―", ""):
                        return None
                    s = s.replace(",", "").replace("百万", "").replace("億", "")
                    try:
                        return float(s)
                    except (ValueError, TypeError):
                        return None

                def _parse_pct(s):
                    if not s:
                        return None
                    m = _re.search(r"([+-]?\d+\.?\d*)", s.replace(",", ""))
                    if m:
                        return float(m.group(1))
                    return None

                # 実績データの抽出
                if actual_row and actual_row.get("cells"):
                    cells = actual_row["cells"]
                    if len(cells) >= 4:
                        result["eval_info"]["revenue_result"] = cells[0] if cells[0] not in ("-", "") else None
                        result["eval_info"]["op_profit_result"] = cells[1] if len(cells) > 1 and cells[1] not in ("-", "") else None
                        result["eval_info"]["net_profit_result"] = cells[3] if len(cells) > 3 and cells[3] not in ("-", "") else None

                    # 前年比を探す（テーブルに%表示がある場合）
                    for cell in cells:
                        if "%" in cell:
                            val = _parse_pct(cell)
                            if val is not None:
                                if "revenue_yoy" not in result["eval_info"]:
                                    result["eval_info"]["revenue_yoy"] = val
                                elif "op_profit_yoy" not in result["eval_info"]:
                                    result["eval_info"]["op_profit_yoy"] = val

                # 来期予想（ガイダンス）の抽出
                target_row = next_forecast_row or forecast_row
                if target_row and target_row.get("cells"):
                    cells = target_row["cells"]
                    if len(cells) >= 4:
                        result["guidance_info"]["next_revenue"] = cells[0] if cells[0] not in ("-", "", "―") else None
                        result["guidance_info"]["next_op_profit"] = cells[1] if len(cells) > 1 and cells[1] not in ("-", "", "―") else None
                        result["guidance_info"]["next_net_profit"] = cells[3] if len(cells) > 3 and cells[3] not in ("-", "", "―") else None

                    # EPS（通常は後ろの方のカラム）
                    if len(cells) >= 5:
                        eps_val = cells[4] if cells[4] not in ("-", "", "―") else None
                        if eps_val:
                            result["guidance_info"]["next_eps"] = eps_val

                    # 配当（最後のカラムの場合が多い）
                    if len(cells) >= 6:
                        div_val = cells[-1] if cells[-1] not in ("-", "", "―") else None
                        if div_val and _parse_num(div_val) is not None:
                            result["guidance_info"]["next_dividend"] = div_val

                # 来期の増減率を計算
                if actual_row and target_row:
                    actual_cells = actual_row.get("cells", [])
                    target_cells = target_row.get("cells", [])
                    # 営業利益の増減率
                    if len(actual_cells) > 1 and len(target_cells) > 1:
                        act_op = _parse_num(actual_cells[1])
                        tgt_op = _parse_num(target_cells[1])
                        if act_op and tgt_op and act_op != 0:
                            chg = (tgt_op - act_op) / abs(act_op) * 100
                            result["guidance_info"]["next_op_profit_change"] = round(chg, 1)
                    # 売上の増減率
                    if len(actual_cells) > 0 and len(target_cells) > 0:
                        act_rev = _parse_num(actual_cells[0])
                        tgt_rev = _parse_num(target_cells[0])
                        if act_rev and tgt_rev and act_rev != 0:
                            chg = (tgt_rev - act_rev) / abs(act_rev) * 100
                            result["guidance_info"]["next_revenue_change"] = round(chg, 1)
                    # 純利益の増減率
                    if len(actual_cells) > 3 and len(target_cells) > 3:
                        act_net = _parse_num(actual_cells[3])
                        tgt_net = _parse_num(target_cells[3])
                        if act_net and tgt_net and act_net != 0:
                            chg = (tgt_net - act_net) / abs(act_net) * 100
                            result["guidance_info"]["next_net_profit_change"] = round(chg, 1)

            # 進捗率の取得（四半期決算の場合）
            progress_section = soup.find(string=_re.compile(r"進捗率|進捗"))
            if progress_section:
                parent = progress_section.find_parent()
                if parent:
                    text = parent.get_text()
                    m = _re.search(r"(\d+\.?\d*)%", text)
                    if m:
                        result["eval_info"]["progress_rate"] = float(m.group(1))

            # ガイダンス総合判定ラベル
            gi = result["guidance_info"]
            op_chg = gi.get("next_op_profit_change")
            net_chg = gi.get("next_net_profit_change")
            # 営業利益ベースで判定、なければ純利益
            chg_val = op_chg if op_chg is not None else net_chg
            if chg_val is not None:
                if chg_val >= 20:
                    gi["guidance_label"] = "大幅増益↑↑"
                    gi["guidance_label_color"] = "#00c853"
                elif chg_val >= 5:
                    gi["guidance_label"] = "増益↑"
                    gi["guidance_label_color"] = "#4caf50"
                elif chg_val >= -5:
                    gi["guidance_label"] = "横ばい→"
                    gi["guidance_label_color"] = "#9e9e9e"
                elif chg_val >= -20:
                    gi["guidance_label"] = "減益↓"
                    gi["guidance_label_color"] = "#ff9800"
                else:
                    gi["guidance_label"] = "大幅減益↓↓"
                    gi["guidance_label_color"] = "#f44336"
            elif gi.get("next_revenue") or gi.get("next_op_profit"):
                gi["guidance_label"] = "予想あり"
                gi["guidance_label_color"] = "#888"
            else:
                gi["guidance_label"] = "未定"
                gi["guidance_label_color"] = "#666"

        except Exception:
            pass

        return result

    def _classify_earnings(headline):
        """決算ヘッドラインから好決算/悪決算を分類"""
        if not headline:
            return "不明", 0, "#888"

        score = 0
        # 好決算キーワード
        good_keywords = {
            "上方修正": 3, "増額": 3, "最高益": 4, "過去最高": 4,
            "大幅増益": 3, "増益": 2, "増収増益": 3, "増収": 1,
            "黒字転換": 3, "黒字化": 3, "復配": 2, "増配": 2,
            "特別配当": 2, "自社株買い": 1, "上振れ": 2,
            "好調": 2, "堅調": 1, "順調": 1, "想定上回": 2,
            "2桁増益": 3, "3桁増益": 4, "急伸": 2,
        }
        # 悪決算キーワード
        bad_keywords = {
            "下方修正": -3, "減額": -3, "赤字": -3, "赤字転落": -4,
            "大幅減益": -3, "減益": -2, "減収減益": -3, "減収": -1,
            "無配": -3, "減配": -2, "下振れ": -2,
            "低調": -2, "苦戦": -2, "悪化": -2, "想定下回": -2,
            "2桁減益": -3, "営業赤字": -3, "最終赤字": -3,
            "未定": -1, "据え置き": 0,
        }

        for kw, pts in good_keywords.items():
            if kw in headline:
                score += pts
        for kw, pts in bad_keywords.items():
            if kw in headline:
                score += pts

        # 変化率の数値を抽出
        import re as _re
        # 「営業利益XX%増」「経常利益XX%減」パターン
        inc_match = _re.findall(r"(\d+(?:\.\d+)?)%増", headline)
        dec_match = _re.findall(r"(\d+(?:\.\d+)?)%減", headline)
        for m in inc_match:
            val = float(m)
            if val >= 50:
                score += 3
            elif val >= 20:
                score += 2
            elif val >= 5:
                score += 1
        for m in dec_match:
            val = float(m)
            if val >= 50:
                score -= 3
            elif val >= 20:
                score -= 2
            elif val >= 5:
                score -= 1

        if score >= 3:
            return "好決算 🟢", score, "#00c853"
        elif score >= 1:
            return "やや好 🔵", score, "#2196f3"
        elif score == 0:
            return "中立 ⚪", score, "#9e9e9e"
        elif score >= -2:
            return "やや悪 🟠", score, "#ff9800"
        else:
            return "悪決算 🔴", score, "#f44336"

    @st.cache_data(ttl=600, show_spinner=False)
    def _fetch_stock_detail_for_selection(code):
        """購入検討用の追加データを取得（株価・指標）"""
        try:
            ticker = f"{code}.T"
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "per": info.get("trailingPE"),
                "pbr": info.get("priceToBook"),
                "div_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "forward_pe": info.get("forwardPE"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "recommendation": info.get("recommendationMean"),
                "target_price": info.get("targetMeanPrice"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            return {}

    # --- UI ---
    st.markdown("---")

    fetch_btn = st.button("🔍 直近の決算発表データを取得", key="sector_earn_fetch", use_container_width=False)

    if fetch_btn:
        with st.spinner("決算発表データ取得中..."):
            earnings_rows = _fetch_sector_earnings_data()
            sector_map = _get_sector_map()
        st.session_state["sector_earnings_data"] = earnings_rows
        st.session_state["sector_map"] = sector_map

    earnings_data = st.session_state.get("sector_earnings_data")
    sector_map = st.session_state.get("sector_map", {})

    if earnings_data:
        # 各銘柄に業種と決算評価を付与
        enriched = []
        for row in earnings_data:
            code = row["code"]
            sector = sector_map.get(code, "不明")
            classification, earn_score, earn_color = _classify_earnings(row["headline"])
            enriched.append({
                **row,
                "sector": sector,
                "classification": classification,
                "earn_score": earn_score,
                "earn_color": earn_color,
                "eval_info": {},
                "guidance_info": {},
            })

        # 決算詳細・来期ガイダンス取得オプション
        st.markdown("#### 📊 決算評価 & 来期ガイダンス")
        detail_col1, detail_col2 = st.columns([3, 1])
        with detail_col1:
            st.caption("各銘柄の今期決算実績と来期会社予想（ガイダンス）を取得します。取得には時間がかかる場合があります。")
        with detail_col2:
            fetch_guidance_btn = st.button("📋 決算詳細・ガイダンス取得", key="fetch_guidance_btn", use_container_width=True)

        if fetch_guidance_btn:
            progress_bar = st.progress(0, text="決算詳細・ガイダンス取得中...")
            guidance_cache = {}
            total = len(enriched)
            for i, row in enumerate(enriched):
                code = row["code"]
                if code not in guidance_cache:
                    detail = _fetch_earnings_and_guidance(code)
                    guidance_cache[code] = detail
                progress_bar.progress((i + 1) / total, text=f"取得中... {i+1}/{total} ({row['name']})")
                time.sleep(0.3)  # レート制限対策
            progress_bar.empty()
            st.session_state["guidance_cache"] = guidance_cache
            st.success(f"✅ {len(guidance_cache)}銘柄の決算詳細・ガイダンスを取得しました")

        # キャッシュからガイダンス情報を付与
        guidance_cache = st.session_state.get("guidance_cache", {})
        if guidance_cache:
            for row in enriched:
                code = row["code"]
                if code in guidance_cache:
                    detail = guidance_cache[code]
                    row["eval_info"] = detail.get("eval_info", {})
                    row["guidance_info"] = detail.get("guidance_info", {})

            # ガイダンスフィルタ追加
            st.markdown("##### 🔍 来期ガイダンスフィルタ")
            gf_col1, gf_col2 = st.columns(2)
            with gf_col1:
                guidance_filter = st.multiselect(
                    "来期見通しで絞り込み",
                    options=["大幅増益↑↑", "増益↑", "横ばい→", "減益↓", "大幅減益↓↓", "未定", "予想あり"],
                    default=[],
                    key="guidance_filter",
                    placeholder="全て表示（選択で絞り込み）"
                )
            with gf_col2:
                guidance_sort = st.checkbox("来期ガイダンスが良い順にソート", key="guidance_sort_check")

            if guidance_filter:
                enriched = [r for r in enriched if r.get("guidance_info", {}).get("guidance_label", "未定") in guidance_filter]

            if guidance_sort:
                def _guidance_sort_key(r):
                    gi = r.get("guidance_info", {})
                    chg = gi.get("next_op_profit_change") or gi.get("next_net_profit_change")
                    return chg if chg is not None else -9999
                enriched.sort(key=_guidance_sort_key, reverse=True)

        st.markdown("---")

        # フィルタUI
        st.markdown("#### 🔧 フィルタ設定")
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            # 業種フィルタ
            all_sectors = sorted(set(r["sector"] for r in enriched if r["sector"] != "不明"))
            selected_sectors = st.multiselect(
                "業種で絞り込み",
                options=all_sectors,
                default=[],
                key="sector_earn_filter",
                placeholder="全業種表示（選択で絞り込み）"
            )

        with fc2:
            # 決算評価フィルタ
            class_options = ["好決算 🟢", "やや好 🔵", "中立 ⚪", "やや悪 🟠", "悪決算 🔴", "不明"]
            selected_classes = st.multiselect(
                "決算評価で絞り込み",
                options=class_options,
                default=["好決算 🟢", "やや好 🔵"],
                key="class_earn_filter",
                placeholder="表示する評価を選択"
            )

        with fc3:
            sort_option = st.selectbox(
                "並び順",
                ["決算スコア（高い順）", "決算スコア（低い順）", "業種別", "コード順"],
                key="sector_sort"
            )

        # フィルタ適用
        filtered = enriched
        if selected_sectors:
            filtered = [r for r in filtered if r["sector"] in selected_sectors]
        if selected_classes:
            filtered = [r for r in filtered if r["classification"] in selected_classes]

        # ソート
        if sort_option == "決算スコア（高い順）":
            filtered.sort(key=lambda x: x["earn_score"], reverse=True)
        elif sort_option == "決算スコア（低い順）":
            filtered.sort(key=lambda x: x["earn_score"])
        elif sort_option == "業種別":
            filtered.sort(key=lambda x: (x["sector"], -x["earn_score"]))
        else:
            filtered.sort(key=lambda x: x["code"])

        # --- 業種別サマリー ---
        st.markdown("#### 📊 業種別 決算結果サマリー")
        st.caption("各業種の好決算/悪決算の比率を表示。業種全体の好不調が一目でわかります。")

        # 業種別集計
        sector_summary = {}
        for r in enriched:
            sec = r["sector"]
            if sec == "不明":
                continue
            if sec not in sector_summary:
                sector_summary[sec] = {"total": 0, "good": 0, "bad": 0, "neutral": 0, "score_sum": 0}
            sector_summary[sec]["total"] += 1
            sector_summary[sec]["score_sum"] += r["earn_score"]
            if r["earn_score"] >= 1:
                sector_summary[sec]["good"] += 1
            elif r["earn_score"] <= -1:
                sector_summary[sec]["bad"] += 1
            else:
                sector_summary[sec]["neutral"] += 1

        # 業種スコアでソート
        sorted_sectors = sorted(
            sector_summary.items(),
            key=lambda x: x[1]["score_sum"] / max(x[1]["total"], 1),
            reverse=True
        )

        if sorted_sectors:
            # 上位・下位の業種をハイライト
            top_sectors = sorted_sectors[:5]
            bottom_sectors = sorted_sectors[-5:] if len(sorted_sectors) > 5 else []

            ts_col1, ts_col2 = st.columns(2)
            with ts_col1:
                st.markdown("##### 🟢 好調業種 TOP5")
                for sec, data in top_sectors:
                    avg_score = data["score_sum"] / max(data["total"], 1)
                    good_ratio = data["good"] / max(data["total"], 1) * 100
                    bar_width = min(good_ratio, 100)
                    st.markdown(
                        f"<div style='background:#1a1a2e;padding:8px 12px;border-radius:6px;margin-bottom:4px;"
                        f"border-left:4px solid #00c853;'>"
                        f"<div style='display:flex;justify-content:space-between;'>"
                        f"<span style='color:#fff;font-weight:bold;'>{sec}</span>"
                        f"<span style='color:#00c853;font-size:0.85rem;'>好決算率 {good_ratio:.0f}% ({data['good']}/{data['total']}社)</span>"
                        f"</div>"
                        f"<div style='margin-top:4px;background:#333;border-radius:3px;height:4px;'>"
                        f"<div style='background:#00c853;width:{bar_width}%;height:100%;border-radius:3px;'></div>"
                        f"</div>"
                        f"<div style='color:#888;font-size:0.72rem;margin-top:2px;'>平均スコア: {avg_score:+.1f} / 好{data['good']} 中{data['neutral']} 悪{data['bad']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            with ts_col2:
                st.markdown("##### 🔴 不調業種 WORST5")
                for sec, data in reversed(bottom_sectors):
                    avg_score = data["score_sum"] / max(data["total"], 1)
                    bad_ratio = data["bad"] / max(data["total"], 1) * 100
                    bar_width = min(bad_ratio, 100)
                    st.markdown(
                        f"<div style='background:#1a1a2e;padding:8px 12px;border-radius:6px;margin-bottom:4px;"
                        f"border-left:4px solid #f44336;'>"
                        f"<div style='display:flex;justify-content:space-between;'>"
                        f"<span style='color:#fff;font-weight:bold;'>{sec}</span>"
                        f"<span style='color:#f44336;font-size:0.85rem;'>悪決算率 {bad_ratio:.0f}% ({data['bad']}/{data['total']}社)</span>"
                        f"</div>"
                        f"<div style='margin-top:4px;background:#333;border-radius:3px;height:4px;'>"
                        f"<div style='background:#f44336;width:{bar_width}%;height:100%;border-radius:3px;'></div>"
                        f"</div>"
                        f"<div style='color:#888;font-size:0.72rem;margin-top:2px;'>平均スコア: {avg_score:+.1f} / 好{data['good']} 中{data['neutral']} 悪{data['bad']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # 全業種テーブル
            with st.expander("📋 全業種サマリーテーブル", expanded=False):
                sector_table_data = []
                for sec, data in sorted_sectors:
                    avg_score = data["score_sum"] / max(data["total"], 1)
                    good_ratio = data["good"] / max(data["total"], 1) * 100
                    bad_ratio = data["bad"] / max(data["total"], 1) * 100
                    # 来期ガイダンス集計（キャッシュがある場合）
                    guidance_up = 0
                    guidance_down = 0
                    if guidance_cache:
                        for r in enriched:
                            if r["sector"] == sec:
                                gi = r.get("guidance_info", {})
                                gl = gi.get("guidance_label", "")
                                if "増益" in gl:
                                    guidance_up += 1
                                elif "減益" in gl:
                                    guidance_down += 1
                    row_data = {
                        "業種": sec,
                        "発表社数": data["total"],
                        "好決算": data["good"],
                        "中立": data["neutral"],
                        "悪決算": data["bad"],
                        "好決算率(%)": round(good_ratio, 1),
                        "悪決算率(%)": round(bad_ratio, 1),
                        "平均スコア": round(avg_score, 2),
                    }
                    if guidance_cache:
                        row_data["来期増益"] = guidance_up
                        row_data["来期減益"] = guidance_down
                    sector_table_data.append(row_data)
                if sector_table_data:
                    st.dataframe(
                        pd.DataFrame(sector_table_data),
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(sector_table_data) * 35 + 40, 600)
                    )

        st.markdown("---")

        # --- 個別銘柄リスト ---
        st.markdown(f"#### 📋 決算発表銘柄一覧（{len(filtered)}/{len(enriched)}件表示）")

        if not filtered:
            st.info("条件に合致する銘柄がありません。フィルタを調整してください。")
        else:
            # 業種別にグループ表示
            if sort_option == "業種別":
                current_sector = None
                for r in filtered:
                    if r["sector"] != current_sector:
                        current_sector = r["sector"]
                        sec_data = sector_summary.get(current_sector, {})
                        sec_good = sec_data.get("good", 0)
                        sec_total = sec_data.get("total", 0)
                        st.markdown(
                            f"<div style='background:#0d1117;padding:6px 12px;border-radius:4px;margin:12px 0 4px 0;'>"
                            f"<span style='color:#4da6ff;font-weight:bold;font-size:0.95rem;'>🏭 {current_sector}</span>"
                            f" <span style='color:#888;font-size:0.78rem;'>（{sec_total}社発表 / 好決算{sec_good}社）</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    _render_earnings_stock_card(r)
            else:
                for r in filtered:
                    _render_earnings_stock_card(r)

            # 購入候補選定セクション
            st.markdown("---")
            st.markdown("#### 🛒 購入候補の選定")
            st.caption("好決算銘柄から購入候補を選び、ウォッチリストまたはポートフォリオに追加できます。")

            # 好決算銘柄のみ抽出
            good_stocks = [r for r in filtered if r["earn_score"] >= 1]
            if good_stocks:
                candidate_options = [f"{r['name']}（{r['code']}）— {r['sector']} [{r['classification']}]" for r in good_stocks]
                selected_candidates = st.multiselect(
                    "購入候補を選択",
                    options=candidate_options,
                    key="purchase_candidates",
                    placeholder="好決算銘柄から選択..."
                )

                if selected_candidates:
                    add_dest = st.radio(
                        "追加先",
                        ["👁️ ウォッチリストに追加", "📋 ポートフォリオに追加"],
                        key="add_destination",
                        horizontal=True
                    )

                    if st.button("✅ 選択した銘柄を追加", key="add_candidates_btn"):
                        added_count = 0
                        for cand in selected_candidates:
                            # コードを抽出
                            import re as _re_add
                            m = _re_add.search(r"（(\d{4})）", cand)
                            if m:
                                code = m.group(1)
                                ticker = f"{code}.T"
                                name = STOCK_DICT.get(ticker, code)

                                if "ウォッチリスト" in add_dest:
                                    try:
                                        wl = st.session_state.get("watchlist", [])
                                        if not any(w.get("code") == ticker for w in wl):
                                            wl.append({"code": ticker, "name": name})
                                            st.session_state["watchlist"] = wl
                                            added_count += 1
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        pf = st.session_state.get("my_portfolio", {})
                                        if ticker not in pf:
                                            pf[ticker] = name
                                            st.session_state["my_portfolio"] = pf
                                            save_portfolio(pf)
                                            added_count += 1
                                    except Exception:
                                        pass

                        if added_count > 0:
                            st.success(f"✅ {added_count}銘柄を追加しました")
                            st.rerun()
                        else:
                            st.info("全て既に登録済みです")

                    # 選択銘柄の詳細情報
                    st.markdown("##### 📊 選択銘柄の詳細")
                    for cand in selected_candidates[:5]:  # 最大5銘柄まで詳細表示
                        import re as _re_det
                        m = _re_det.search(r"（(\d{4})）", cand)
                        if m:
                            code = m.group(1)
                            with st.spinner(f"{code} の詳細取得中..."):
                                detail = _fetch_stock_detail_for_selection(code)
                            if detail and detail.get("price"):
                                price = detail["price"]
                                per = detail.get("per")
                                pbr = detail.get("pbr")
                                div_y = detail.get("div_yield")
                                fwd_pe = detail.get("forward_pe")
                                target = detail.get("target_price")
                                rev_g = detail.get("revenue_growth")
                                earn_g = detail.get("earnings_growth")
                                rec = detail.get("recommendation")
                                h52 = detail.get("52w_high")
                                l52 = detail.get("52w_low")

                                # 52週高値からの位置
                                pos_52w = ""
                                if h52 and l52 and price:
                                    pos = (price - l52) / (h52 - l52) * 100 if (h52 - l52) > 0 else 50
                                    pos_52w = f"52週レンジ内 {pos:.0f}%"

                                metrics_parts = []
                                if per: metrics_parts.append(f"PER {per:.1f}倍")
                                if fwd_pe: metrics_parts.append(f"予想PER {fwd_pe:.1f}倍")
                                if pbr: metrics_parts.append(f"PBR {pbr:.2f}倍")
                                if div_y: metrics_parts.append(f"配当利回り {div_y*100:.2f}%")
                                if rev_g is not None: metrics_parts.append(f"売上成長 {rev_g*100:+.1f}%")
                                if earn_g is not None: metrics_parts.append(f"利益成長 {earn_g*100:+.1f}%")
                                if target: metrics_parts.append(f"目標株価 ¥{target:,.0f}")
                                if rec: metrics_parts.append(f"レーティング {rec:.1f}/5")
                                if pos_52w: metrics_parts.append(pos_52w)

                                name = STOCK_DICT.get(f"{code}.T", code)
                                st.markdown(
                                    f"<div style='background:#1a1a2e;padding:10px 14px;border-radius:8px;"
                                    f"margin-bottom:4px;border-left:4px solid #4da6ff;'>"
                                    f"<div style='color:#fff;font-weight:bold;'>"
                                    f"<a href='https://kabutan.jp/stock/?code={code}' target='_blank' "
                                    f"style='color:#fff;text-decoration:none;'>{name}</a>"
                                    f" <span style='color:#888;'>({code})</span>"
                                    f" <span style='color:#4da6ff;font-size:1.1rem;'>¥{price:,.0f}</span>"
                                    f"</div>"
                                    f"<div style='color:#aaa;font-size:0.78rem;margin-top:4px;'>"
                                    f"{' / '.join(metrics_parts)}"
                                    f"</div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
            else:
                st.info("フィルタ条件に合致する好決算銘柄がありません。フィルタを調整してください。")

    else:
        st.info("「🔍 直近の決算発表データを取得」ボタンを押すと、業種別の決算結果を分析します。")



# =====================================================================
# Tab 17: ML要因分析 — 機械学習による株価変動要因分析
# =====================================================================
with tab17:
    st.subheader("🤖 機械学習による株価変動要因分析")
    st.caption("PER・ROE・PEGレシオ・EPS成長率・出来高・テクニカル指標・ダウ理論を特徴量として、株価変動の要因をML（LightGBM）で分析します。セクター別PER相対評価に対応。")

    # --- セクター定義 ---
    SECTOR_MAP = {
        "情報通信": ["9984.T", "9432.T", "9434.T", "4686.T", "3993.T", "4478.T", "4443.T",
                    "4776.T", "3923.T", "4475.T", "3778.T", "4812.T", "3844.T", "3042.T"],
        "電気機器": ["6758.T", "6861.T", "6752.T", "6762.T", "6981.T", "8035.T", "6920.T",
                    "6857.T", "6723.T", "6146.T", "6506.T", "6954.T", "7735.T", "6645.T",
                    "6674.T", "6810.T", "6962.T", "6976.T", "6728.T", "6988.T"],
        "機械": ["7011.T", "7012.T", "7013.T", "6208.T", "6370.T", "6368.T", "6361.T",
                "6366.T", "6013.T", "5631.T", "7721.T", "6125.T", "7729.T"],
        "銀行": ["8306.T", "8316.T", "8411.T", "8309.T", "7182.T", "7186.T", "8354.T",
                "7167.T", "8331.T", "8355.T"],
        "商社": ["8058.T", "8031.T", "8001.T", "8053.T", "8002.T"],
        "輸送用機器": ["7203.T", "6902.T"],
        "化学": ["4005.T", "4183.T", "4063.T", "4208.T", "4401.T", "4204.T"],
        "鉱業・非鉄": ["5713.T", "5714.T", "5706.T", "5711.T", "5727.T", "5019.T", "5020.T",
                      "1518.T", "1514.T", "5765.T"],
        "不動産": ["8801.T", "8802.T", "8830.T", "3289.T", "8804.T"],
        "建設": ["1801.T", "1802.T", "1812.T", "1803.T", "1808.T"],
        "運輸": ["9020.T", "9021.T", "9022.T", "9005.T", "9009.T", "9101.T", "9104.T",
                "9107.T", "9110.T", "9119.T"],
        "サービス": ["9603.T", "6191.T", "9024.T", "9412.T", "4274.T"],
        "エネルギー": ["9519.T", "9517.T", "1407.T", "9514.T", "3856.T", "8088.T"],
        "食品": ["2809.T"],
        "その他": ["7832.T", "4996.T", "4997.T", "4955.T", "6501.T", "6502.T"],
    }

    # 逆引き: ticker → sector
    TICKER_SECTOR = {}
    for _sec, _tickers in SECTOR_MAP.items():
        for _t in _tickers:
            TICKER_SECTOR[_t] = _sec

    # --- マクロ経済データ取得関数 ---
    @st.cache_data(ttl=3600)
    def fetch_macro_data():
        """金利・為替・商品価格・景気指標のプロキシを取得"""
        macro = {}
        try:
            # 米国10年債利回り
            tnx = yf.download("^TNX", period="3mo", progress=False)
            if not tnx.empty:
                if isinstance(tnx.columns, pd.MultiIndex):
                    macro["US10Y"] = float(tnx["Close"].iloc[-1].iloc[0])
                else:
                    macro["US10Y"] = float(tnx["Close"].iloc[-1])
        except Exception:
            macro["US10Y"] = None

        try:
            # ドル円
            usdjpy = yf.download("JPY=X", period="3mo", progress=False)
            if not usdjpy.empty:
                if isinstance(usdjpy.columns, pd.MultiIndex):
                    macro["USDJPY"] = float(usdjpy["Close"].iloc[-1].iloc[0])
                    macro["USDJPY_chg_20d"] = float(
                        (usdjpy["Close"].iloc[-1].iloc[0] / usdjpy["Close"].iloc[-20].iloc[0] - 1) * 100
                    ) if len(usdjpy) >= 20 else 0
                else:
                    macro["USDJPY"] = float(usdjpy["Close"].iloc[-1])
                    macro["USDJPY_chg_20d"] = float(
                        (usdjpy["Close"].iloc[-1] / usdjpy["Close"].iloc[-20] - 1) * 100
                    ) if len(usdjpy) >= 20 else 0
        except Exception:
            macro["USDJPY"] = None
            macro["USDJPY_chg_20d"] = 0

        try:
            # 原油 (WTI)
            oil = yf.download("CL=F", period="3mo", progress=False)
            if not oil.empty:
                if isinstance(oil.columns, pd.MultiIndex):
                    macro["WTI"] = float(oil["Close"].iloc[-1].iloc[0])
                    macro["WTI_chg_20d"] = float(
                        (oil["Close"].iloc[-1].iloc[0] / oil["Close"].iloc[-20].iloc[0] - 1) * 100
                    ) if len(oil) >= 20 else 0
                else:
                    macro["WTI"] = float(oil["Close"].iloc[-1])
                    macro["WTI_chg_20d"] = float(
                        (oil["Close"].iloc[-1] / oil["Close"].iloc[-20] - 1) * 100
                    ) if len(oil) >= 20 else 0
        except Exception:
            macro["WTI"] = None
            macro["WTI_chg_20d"] = 0

        try:
            # 金価格
            gold = yf.download("GC=F", period="3mo", progress=False)
            if not gold.empty:
                if isinstance(gold.columns, pd.MultiIndex):
                    macro["GOLD"] = float(gold["Close"].iloc[-1].iloc[0])
                    macro["GOLD_chg_20d"] = float(
                        (gold["Close"].iloc[-1].iloc[0] / gold["Close"].iloc[-20].iloc[0] - 1) * 100
                    ) if len(gold) >= 20 else 0
                else:
                    macro["GOLD"] = float(gold["Close"].iloc[-1])
                    macro["GOLD_chg_20d"] = float(
                        (gold["Close"].iloc[-1] / gold["Close"].iloc[-20] - 1) * 100
                    ) if len(gold) >= 20 else 0
        except Exception:
            macro["GOLD"] = None
            macro["GOLD_chg_20d"] = 0

        return macro

    # --- ニュース感情分析関数 ---
    @st.cache_data(ttl=1800)
    def fetch_news_sentiment(ticker, name):
        """Google Newsから銘柄関連ニュースを取得し、簡易感情スコアを算出"""
        import xml.etree.ElementTree as ET
        try:
            search_term = name.replace("HD", "").replace("FG", "").strip()
            url = f"https://news.google.com/rss/search?q={quote(search_term)}+株価&hl=ja&gl=JP&ceid=JP:ja"
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return 0, 0

            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            if not items:
                return 0, 0

            # 簡易感情分析: ポジティブ/ネガティブキーワードカウント
            positive_words = ["上昇", "高値", "好調", "増益", "増収", "最高", "上方修正",
                            "買い", "期待", "成長", "回復", "改善", "好決算", "増配",
                            "自社株買い", "株主還元", "過去最高", "急騰", "反発"]
            negative_words = ["下落", "安値", "不調", "減益", "減収", "下方修正",
                            "売り", "懸念", "リスク", "悪化", "低迷", "赤字",
                            "減配", "急落", "暴落", "不振", "損失"]

            pos_count = 0
            neg_count = 0
            article_count = min(len(items), 10)

            for item in items[:10]:
                title = (item.findtext("title", "") + " " + item.findtext("description", "")).lower()
                for w in positive_words:
                    if w in title:
                        pos_count += 1
                for w in negative_words:
                    if w in title:
                        neg_count += 1

            total = pos_count + neg_count
            if total == 0:
                sentiment_score = 0
            else:
                sentiment_score = (pos_count - neg_count) / total * 100  # -100〜+100

            return sentiment_score, article_count
        except Exception:
            return 0, 0

    # --- 配当利回り算出ヘルパー（株式分割・単位ゆらぎを吸収） ---
    def _calc_div_yield_pct(info):
        """yfinanceのinfoから配当利回り(%)を堅牢に算出する。

        - 優先順: dividendRate(予想配当) → trailingAnnualDividendRate(実績配当) を株価で割る
        - yfinanceのdividendYieldは仕様変更で「%表記(2.5)」「比率(0.025)」が混在するため
          値域で自動判定し、二重に100倍しないようにする
        - 配当額/株価方式は株式分割後の調整済み株価・配当を使うため分割の影響を受けにくい
        - 0〜15%の常識的な範囲外はノイズとして除外
        """
        try:
            price = (info.get("previousClose") or info.get("currentPrice")
                     or info.get("regularMarketPrice"))

            # 方法1: 予想年間配当 / 株価
            if price and price > 0:
                div_rate = info.get("dividendRate")
                if div_rate and div_rate > 0:
                    yld = div_rate / price * 100
                    if 0 < yld < 15:
                        return round(yld, 2)

                # 方法2: 実績年間配当 / 株価
                trailing = info.get("trailingAnnualDividendRate")
                if trailing and trailing > 0:
                    yld = trailing / price * 100
                    if 0 < yld < 15:
                        return round(yld, 2)

            # 方法3: dividendYield（%表記か比率表記かを値域で判定）
            dy = info.get("dividendYield")
            if dy and dy > 0:
                # 1未満なら比率(0.025=2.5%)とみなし100倍、それ以外は既に%表記
                yld = dy * 100 if dy < 1 else dy
                if 0 < yld < 15:
                    return round(yld, 2)
        except Exception:
            pass
        return None

    # --- PEGレシオ算出ヘルパー（複数ソースのフォールバック付き） ---
    def _calc_peg_ratio_ml(info):
        """yfinanceのinfoからPEGレシオを堅牢に算出する（ML特徴量用）。

        日本株では info["pegRatio"] が欠損することが多いため、
        PER÷成長率(%) で自前計算するフォールバックを備える。
        成長率は 年次EPS → 四半期EPS → 予想EPS → 売上成長率(暫定) の順で採用。
        PEG=PER÷成長率(%)。減益・成長率欠損時は None を返す。
        """
        try:
            # 1. yfinance提供値（pegRatio / trailingPegRatio）
            for key in ("pegRatio", "trailingPegRatio"):
                v = info.get(key)
                if v is not None and -1000 < v < 1000:
                    return round(float(v), 2)

            # PER（trailing優先→forward→株価÷EPSで自前計算）
            per = info.get("trailingPE") or info.get("forwardPE")
            if not per or per <= 0:
                price = (info.get("currentPrice") or info.get("regularMarketPrice")
                         or info.get("previousClose"))
                eps = info.get("trailingEps") or info.get("forwardEps")
                if price and eps and eps > 0:
                    per = price / eps
            if not per or per <= 0:
                return None

            # 2. 年次EPS成長率
            eg = info.get("earningsGrowth")
            if eg is not None and eg > 0:
                return round(per / (eg * 100), 2)

            # 3. 四半期EPS成長率
            eqg = info.get("earningsQuarterlyGrowth")
            if eqg is not None and eqg > 0:
                return round(per / (eqg * 100), 2)

            # 4. 予想EPS成長率（trailing→forward）
            teps = info.get("trailingEps")
            feps = info.get("forwardEps")
            if teps and feps and teps > 0:
                g = (feps - teps) / teps
                if g > 0:
                    return round(per / (g * 100), 2)

            # 5. 売上成長率で代用（暫定。減益・EPS成長率欠損時の最終手段）
            rg = info.get("revenueGrowth")
            if rg is not None and rg > 0:
                return round(per / (rg * 100), 2)
        except Exception:
            pass
        return None

    # --- ML用データ取得関数（拡張版） ---
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_ml_features(tickers, period="6mo"):
        """各銘柄の包括的特徴量を取得（ファンダメンタル+テクニカル+マクロ+センチメント）

        当日分はディスクにキャッシュし、その日のうちは再取得しない。
        （メモリキャッシュはアプリ再起動で消えるため、ディスクにも保存する）
        """
        import hashlib as _hl

        # --- 当日ディスクキャッシュの確認 ---
        _today = datetime.now().strftime("%Y%m%d")
        _key = _hl.md5(f"{_today}|{period}|{','.join(sorted(tickers))}".encode()).hexdigest()[:16]
        _cache_path = os.path.join(_ML_FEAT_CACHE_DIR, f"mlfeat_{_today}_{_key}.parquet")
        if os.path.exists(_cache_path):
            try:
                return pd.read_parquet(_cache_path)
            except Exception:
                pass  # 壊れていたら取得し直す

        # 古いキャッシュ（前日以前）を掃除
        try:
            for _f in os.listdir(_ML_FEAT_CACHE_DIR):
                if _f.startswith("mlfeat_") and not _f.startswith(f"mlfeat_{_today}_"):
                    try:
                        os.remove(os.path.join(_ML_FEAT_CACHE_DIR, _f))
                    except Exception:
                        pass
        except Exception:
            pass

        # マクロデータを先に取得（全銘柄共通）
        macro = fetch_macro_data()

        # --- 1銘柄分の特徴量取得（並列実行のためループ本体を関数化） ---
        # 戻り値: ("ok", record_dict) / ("hist_short", None) / ("exception", err_str)
        # ※ 取得する項目・計算内容は直列版と完全に同一。精度には影響しない。
        def _fetch_one(ticker):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period=period)
                if hist.empty or len(hist) < 30:
                    return ("hist_short", None)

                # === 1. 財務諸表指標 ===
                per = info.get("trailingPE") or info.get("forwardPE")
                forward_pe = info.get("forwardPE")
                pbr = info.get("priceToBook")
                roe = info.get("returnOnEquity")
                roa = info.get("returnOnAssets")
                # 配当利回り(%)を堅牢に算出（配当額÷株価、異常値フィルタ付き）
                dividend_yield = _calc_div_yield_pct(info)
                peg = _calc_peg_ratio_ml(info)
                eps_growth = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
                revenue_growth = info.get("revenueGrowth")
                profit_margin = info.get("profitMargins")
                operating_margin = info.get("operatingMargins")
                debt_to_equity = info.get("debtToEquity")
                current_ratio = info.get("currentRatio")
                free_cashflow = info.get("freeCashflow")
                market_cap = info.get("marketCap")
                enterprise_value = info.get("enterpriseValue")
                ev_ebitda = info.get("enterpriseToEbitda")

                # === 2. 成長性指標 ===
                revenue_per_share = info.get("revenuePerShare")
                book_value = info.get("bookValue")
                earnings_growth_5y = info.get("earningsGrowth")  # 近似

                # === 3. セクター情報 ===
                sector = TICKER_SECTOR.get(ticker, "その他")

                # === 4. 出来高分析 ===
                vol_mean_5 = hist["Volume"].iloc[-5:].mean()
                vol_mean_20 = hist["Volume"].iloc[-20:].mean()
                vol_mean_60 = hist["Volume"].iloc[-60:].mean() if len(hist) >= 60 else hist["Volume"].mean()
                vol_ratio_20_60 = vol_mean_20 / vol_mean_60 if vol_mean_60 > 0 else 1.0
                vol_ratio_5_20 = vol_mean_5 / vol_mean_20 if vol_mean_20 > 0 else 1.0
                # 出来高急増検出（直近5日が20日平均の2倍以上）
                vol_spike = 1 if vol_ratio_5_20 > 2.0 else 0

                # === 5. テクニカル指標（チャート） ===
                close = hist["Close"]

                # 移動平均線（短期〜長期）
                ma5 = close.rolling(5).mean()
                ma13 = close.rolling(13).mean()
                ma20 = close.rolling(20).mean()
                ma26 = close.rolling(26).mean()
                ma60 = close.rolling(60).mean() if len(close) >= 60 else close.rolling(len(close)).mean()

                # 移動平均乖離率
                ma5_dev = (close.iloc[-1] - ma5.iloc[-1]) / ma5.iloc[-1] * 100 if ma5.iloc[-1] > 0 else 0
                ma13_dev = (close.iloc[-1] - ma13.iloc[-1]) / ma13.iloc[-1] * 100 if len(close) >= 13 and ma13.iloc[-1] > 0 else 0
                ma20_dev = (close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100 if ma20.iloc[-1] > 0 else 0
                ma26_dev = (close.iloc[-1] - ma26.iloc[-1]) / ma26.iloc[-1] * 100 if len(close) >= 26 and ma26.iloc[-1] > 0 else 0
                ma60_dev = (close.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1] * 100 if len(close) >= 60 and ma60.iloc[-1] > 0 else 0

                # ゴールデンクロス/デッドクロス判定
                gc_dc = 0
                if len(close) >= 26:
                    if ma5.iloc[-1] > ma26.iloc[-1] and ma5.iloc[-2] <= ma26.iloc[-2]:
                        gc_dc = 1  # ゴールデンクロス
                    elif ma5.iloc[-1] < ma26.iloc[-1] and ma5.iloc[-2] >= ma26.iloc[-2]:
                        gc_dc = -1  # デッドクロス

                # RSI (14日)
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
                rsi = 100 - (100 / (1 + rs))

                # MACD
                ema12 = close.ewm(span=12).mean()
                ema26_line = close.ewm(span=26).mean()
                macd_line = ema12.iloc[-1] - ema26_line.iloc[-1]
                macd_signal = (ema12 - ema26_line).ewm(span=9).mean().iloc[-1]
                macd_hist = macd_line - macd_signal

                # ボリンジャーバンド（20日, 2σ）
                bb_mid = close.rolling(20).mean()
                bb_std = close.rolling(20).std()
                bb_upper = bb_mid + 2 * bb_std
                bb_lower = bb_mid - 2 * bb_std
                bb_position = 0
                if bb_std.iloc[-1] > 0:
                    bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100

                # ATR（Average True Range）- ボラティリティ
                high = hist["High"]
                low = hist["Low"]
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean().iloc[-1]
                atr_ratio = atr_14 / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0

                # === 6. ダウ理論 ===
                highs = []
                lows = []
                window = 5
                for i in range(0, min(20, len(hist)), window):
                    segment = hist.iloc[-(i+window):-(i) if i > 0 else len(hist)]
                    if len(segment) > 0:
                        highs.append(segment["High"].max())
                        lows.append(segment["Low"].min())
                highs.reverse()
                lows.reverse()

                dow_score = 0
                if len(highs) >= 3:
                    if highs[-1] > highs[-2] and highs[-2] > highs[-3]:
                        dow_score += 1
                    if highs[-1] < highs[-2] and highs[-2] < highs[-3]:
                        dow_score -= 1
                if len(lows) >= 3:
                    if lows[-1] > lows[-2] and lows[-2] > lows[-3]:
                        dow_score += 1
                    if lows[-1] < lows[-2] and lows[-2] < lows[-3]:
                        dow_score -= 1

                # === 7. 需給バランス（Yahoo Financeから取得可能な範囲） ===
                short_ratio = info.get("shortRatio")  # 空売り比率
                shares_short = info.get("sharesShort")
                shares_outstanding = info.get("sharesOutstanding")
                held_by_institutions = info.get("heldPercentInstitutions")
                held_by_insiders = info.get("heldPercentInsiders")

                # === 8. ニュース感情分析 ===
                # 日本語名を優先: THEMES辞書 → Yahoo Finance
                _jp_name = None
                for _th_info in THEMES.values():
                    if ticker in _th_info["stocks"]:
                        _jp_name = _th_info["stocks"][ticker]
                        break
                name = _jp_name or info.get("shortName") or info.get("longName") or ticker
                sentiment_score, news_count = fetch_news_sentiment(ticker, name)

                # === 9. マクロ経済指標（全銘柄共通） ===
                us10y = macro.get("US10Y")
                usdjpy_chg = macro.get("USDJPY_chg_20d", 0)
                wti_chg = macro.get("WTI_chg_20d", 0)
                gold_chg = macro.get("GOLD_chg_20d", 0)

                # === 10. 52週高値/安値からの位置 ===
                high_52w = info.get("fiftyTwoWeekHigh")
                low_52w = info.get("fiftyTwoWeekLow")
                price_now = close.iloc[-1]
                pos_52w = None
                if high_52w and low_52w and (high_52w - low_52w) > 0:
                    pos_52w = (price_now - low_52w) / (high_52w - low_52w) * 100

                # === 目的変数: 直近20日リターン ===
                ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0

                return ("ok", {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    # 財務諸表・バリュエーション
                    "PER": per,
                    "予想PER": forward_pe,
                    "PBR": pbr,
                    "ROE": roe * 100 if roe else None,
                    "ROA": roa * 100 if roa else None,
                    "配当利回り": dividend_yield,
                    "PEG": peg,
                    "EPS成長率": eps_growth * 100 if eps_growth else None,
                    "売上成長率": revenue_growth * 100 if revenue_growth else None,
                    "営業利益率": operating_margin * 100 if operating_margin else None,
                    "純利益率": profit_margin * 100 if profit_margin else None,
                    "D/Eレシオ": debt_to_equity,
                    "流動比率": current_ratio,
                    "EV/EBITDA": ev_ebitda,
                    # 出来高・需給
                    "出来高比率_5_20": vol_ratio_5_20,
                    "出来高比率_20_60": vol_ratio_20_60,
                    "出来高急増": vol_spike,
                    "機関投資家保有率": held_by_institutions * 100 if held_by_institutions else None,
                    "空売り比率": short_ratio,
                    # テクニカル
                    "MA5乖離率": ma5_dev,
                    "MA13乖離率": ma13_dev,
                    "MA20乖離率": ma20_dev,
                    "MA26乖離率": ma26_dev,
                    "MA60乖離率": ma60_dev,
                    "GC_DC": gc_dc,
                    "RSI": rsi,
                    "MACD_hist": macd_hist,
                    "BB位置": bb_position,
                    "ATR比率": atr_ratio,
                    "ダウ理論スコア": dow_score,
                    "52週位置": pos_52w,
                    # センチメント
                    "ニュース感情": sentiment_score,
                    "ニュース件数": news_count,
                    # マクロ（全銘柄共通）
                    "米10年金利": us10y,
                    "ドル円変化率": usdjpy_chg,
                    "原油変化率": wti_chg,
                    "金変化率": gold_chg,
                    # 予測日の終値（ローリング学習で20日後実績を正しく計算するため）
                    "終値": float(price_now) if pd.notna(price_now) else None,
                    # 目的変数
                    "20日リターン": ret_20d,
                })
            except Exception as _e:
                return ("exception", f"{ticker}: {type(_e).__name__}: {_e}")

        # --- 並列取得（ネットワークI/O待ちをオーバーラップさせる） ---
        # 各銘柄の処理は独立しており、取得項目は直列版と同一。
        # 並列数は Yahoo Finance のレート制限に配慮して控えめに設定。
        records = []
        _skip_stats = {"total": len(tickers), "ok": 0, "hist_short": 0, "exception": 0,
                       "errors": []}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        _max_workers = min(8, max(1, len(tickers)))
        _progress = None
        _done = 0
        _total = len(tickers)
        _t0 = time.time()

        def _fmt_sec(s):
            """秒を「M分S秒」/「S秒」の見やすい表記に整形。"""
            s = int(round(s))
            if s >= 60:
                return f"{s // 60}分{s % 60}秒"
            return f"{s}秒"

        try:
            _progress = st.progress(0.0, text=f"銘柄データ取得中… 0/{_total}（残り時間を推定中…）")
        except Exception:
            _progress = None
        with ThreadPoolExecutor(max_workers=_max_workers) as _ex:
            _futures = {_ex.submit(_fetch_one, t): t for t in tickers}
            for _fut in as_completed(_futures):
                try:
                    _status, _payload = _fut.result()
                except Exception as _e:
                    _status, _payload = "exception", f"{_futures[_fut]}: {type(_e).__name__}: {_e}"
                if _status == "ok":
                    records.append(_payload)
                    _skip_stats["ok"] += 1
                elif _status == "hist_short":
                    _skip_stats["hist_short"] += 1
                else:  # exception
                    _skip_stats["exception"] += 1
                    if len(_skip_stats["errors"]) < 5 and _payload:
                        _skip_stats["errors"].append(_payload)
                _done += 1
                if _progress is not None:
                    try:
                        # 経過時間から残り時間(ETA)を推定
                        _elapsed = time.time() - _t0
                        _remain = _total - _done
                        if _done > 0 and _remain > 0:
                            _eta = _elapsed / _done * _remain
                            _eta_txt = f"残り約{_fmt_sec(_eta)}"
                        else:
                            _eta_txt = "まもなく完了"
                        _progress.progress(
                            _done / _total,
                            text=f"銘柄データ取得中… {_done}/{_total}（経過{_fmt_sec(_elapsed)} / {_eta_txt}）")
                    except Exception:
                        pass
        if _progress is not None:
            try:
                _progress.empty()
            except Exception:
                pass
        # 診断情報をセッションに残す（キャッシュされる戻り値は変えない）
        try:
            st.session_state["ml_fetch_diag"] = _skip_stats
        except Exception:
            pass
        result_df = pd.DataFrame(records)
        # 当日分をディスクに保存（次回以降は再取得せず即読み込み）
        if not result_df.empty:
            try:
                result_df.to_parquet(_cache_path, index=False)
            except Exception:
                pass
        return result_df

    def calc_sector_relative_per(df):
        """セクター平均PERとの相対値を計算"""
        df = df.copy()
        # PER/PBRが文字列の場合があるため数値に変換
        df["PER"] = pd.to_numeric(df["PER"], errors="coerce")
        df["PBR"] = pd.to_numeric(df["PBR"], errors="coerce")
        sector_avg_per = df.groupby("sector")["PER"].transform("median")
        df["PER_セクター相対"] = df["PER"] - sector_avg_per
        # PBRもセクター相対
        sector_avg_pbr = df.groupby("sector")["PBR"].transform("median")
        df["PBR_セクター相対"] = df["PBR"] - sector_avg_pbr
        return df, df.groupby("sector")["PER"].median().to_dict()

    # --- UI ---
    st.markdown("---")
    st.markdown("#### 📌 分析対象の選択")

    ml_target_option = st.radio(
        "分析対象",
        ["全テーマ銘柄", "テーマ内銘柄（選択テーマ）"],
        horizontal=True,
        key="ml_target_option"
    )

    if ml_target_option == "テーマ内銘柄（選択テーマ）":
        ml_theme = st.selectbox("テーマ選択", list(THEMES.keys()), key="ml_theme_select")
        ml_tickers = list(THEMES[ml_theme]["stocks"].keys())
        st.info(f"対象: {ml_theme} — {len(ml_tickers)}銘柄")
    else:
        ml_tickers = sorted(set(t for info in THEMES.values() for t in info["stocks"].keys()))
        st.info(f"対象: 全テーマ — {len(ml_tickers)}銘柄")

    # =====================================================================
    # 🔄🌅 相場レジーム判定（週足 逆張り/順張り ＋ 寄り/引けエントリー勝率）
    #   ※ 株価データのみで完結するため、ML分析ボタンの前に常時表示する
    # =====================================================================
    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_weekly_close(tickers_tuple, period="1y"):
        """複数銘柄の終値をまとめて取得し、日付×銘柄のDataFrameで返す。"""
        try:
            raw = yf.download(
                list(tickers_tuple), period=period,
                progress=False, group_by="ticker", auto_adjust=True
            )
        except Exception:
            return pd.DataFrame()
        closes = {}
        for t in tickers_tuple:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if (t, "Close") in raw.columns:
                        closes[t] = raw[(t, "Close")]
                    elif t in raw.columns.get_level_values(0):
                        closes[t] = raw[t]["Close"]
                else:
                    closes[t] = raw["Close"]
            except Exception:
                continue
        if not closes:
            return pd.DataFrame()
        out = pd.DataFrame(closes)
        out.index = pd.to_datetime(out.index)
        return out

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_index_close(symbol="^N225", period="1y"):
        """市場指数の終値を取得（週次換算用）。"""
        try:
            raw = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                s = raw[("Close", symbol)] if ("Close", symbol) in raw.columns else raw["Close"].iloc[:, 0]
            else:
                s = raw["Close"]
            df = pd.DataFrame({symbol: s})
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_ohlc_dict(tickers_tuple, period="6mo"):
        """複数銘柄の日足OHLCを取得し {銘柄: DataFrame(Open/High/Low/Close)} で返す。"""
        result = {}
        try:
            raw = yf.download(
                list(tickers_tuple), period=period,
                progress=False, group_by="ticker", auto_adjust=False
            )
        except Exception:
            return result
        for t in tickers_tuple:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.get_level_values(0):
                        sub = raw[t][["Open", "High", "Low", "Close"]].dropna(how="all")
                    else:
                        continue
                else:
                    sub = raw[["Open", "High", "Low", "Close"]].dropna(how="all")
                if not sub.empty:
                    result[t] = sub
            except Exception:
                continue
        return result

    with st.expander("🔄🌅 相場レジーム判定（逆張り/順張り・寄り/引けエントリー）を表示", expanded=True):
        # --- 週足レジーム判定 ---
        st.markdown("##### 🔄 週足レジーム判定 — 逆張り / 順張り どちらが機能しやすいか")
        st.caption(
            "週次リターンの自己相関と、『前週勝ち組を買う(順張り)』vs『前週負け組を買う(逆張り)』の"
            "バックテスト勝率から、今の相場でどちらの戦略が効きやすいかを判定します。"
            "自己相関が正なら順張り(モメンタム)、負なら逆張り(ミーンリバージョン)が機能しやすい地合いです。"
        )

        _regime_period = "2y" if analysis_period in ("2y",) else "1y"
        with st.spinner("📅 週足データを取得してレジーム判定中..."):
            _theme_close = _fetch_weekly_close(tuple(ml_tickers), period=_regime_period)
            _topix_close = _fetch_index_close("1306.T", period=_regime_period)  # TOPIX連動ETF
            if _topix_close.empty:
                _topix_close = _fetch_index_close("^N225", period=_regime_period)  # フォールバック:日経225

        _reg_theme = analyze_weekly_regime(_theme_close)
        _reg_market = analyze_weekly_regime(_topix_close)

        def _render_regime_card(title, reg, subtitle):
            if reg is None:
                st.warning(f"{title}: 週足データが不足しておりレジーム判定できませんでした。")
                return
            st.markdown(
                f"<div style='border-left:5px solid {reg['color']};"
                f"background:#1a1d23;padding:12px 16px;border-radius:6px;margin-bottom:8px;'>"
                f"<div style='color:#888;font-size:0.75rem;'>{title} — {subtitle}</div>"
                f"<div style='color:{reg['color']};font-size:1.3rem;font-weight:bold;'>"
                f"{reg['emoji']} {reg['verdict']}</div>"
                f"<div style='color:#ccc;font-size:0.8rem;margin-top:4px;'>スコア: "
                f"{reg['score']:+.2f} （+1に近いほど順張り / −1に近いほど逆張り）</div>"
                f"<div style='color:#ddd;font-size:0.82rem;margin-top:6px;'>{reg['advice']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            _m1, _m2, _m3, _m4 = st.columns(4)
            _ts = reg["ts_autocorr"]
            _cs = reg["cs_autocorr"]
            _m1.metric(
                "週次自己相関(時系列)",
                f"{_ts:+.2f}" if not np.isnan(_ts) else "—",
                help="正=順張り / 負=逆張り。銘柄ごとの週次リターンのラグ1自己相関の平均。"
            )
            _m2.metric(
                "自己相関(銘柄間)",
                f"{_cs:+.2f}" if not np.isnan(_cs) else "—",
                help="前週リターンと今週リターンの銘柄横断相関の平均。正=強い銘柄が続伸(順張り有利)。"
            )
            _wr = reg["momo_win_rate"]
            _m3.metric(
                "順張り勝率",
                f"{_wr:.0f}%" if not np.isnan(_wr) else "—",
                help="順張り(前週勝ち組買い)が逆張り(前週負け組買い)を上回った週の割合。"
                     f"50%超=順張り優位。検証週数={reg['n_weeks_bt']}週"
            )
            _m4.metric(
                "ボラ状態",
                "高ボラ" if reg["high_vol"] else "通常",
                help="直近13週の変動が全期間平均の1.15倍超なら高ボラ。高ボラ局面は逆張りが効きやすい傾向。"
            )
            _ma, _ra = reg["momo_avg"], reg["rev_avg"]
            if not np.isnan(_ma) and not np.isnan(_ra):
                st.caption(
                    f"戦略別 平均週次リターン — 順張り: {_ma:+.2f}% / 逆張り: {_ra:+.2f}% "
                    f"（直近{reg['n_weeks']}週・{reg['n_stocks']}銘柄で検証）"
                )

        _rc1, _rc2 = st.columns(2)
        with _rc1:
            _subtitle = ml_theme if ml_target_option == "テーマ内銘柄（選択テーマ）" else "全テーマ銘柄"
            _render_regime_card("📦 テーマ銘柄群", _reg_theme, _subtitle)
        with _rc2:
            _render_regime_card("🏛️ 市場全体(TOPIX/日経)", _reg_market, "地合い")

        if _reg_theme and _reg_market:
            if _reg_theme["verdict"] == _reg_market["verdict"]:
                st.success(
                    f"✅ テーマ・市場ともに「{_reg_theme['verdict']}」。"
                    "地合いが一致しており、判定の信頼度は相対的に高めです。"
                )
            else:
                st.info(
                    f"ℹ️ テーマは「{_reg_theme['verdict']}」、市場全体は「{_reg_market['verdict']}」と"
                    "ねじれています。テーマ固有の需給が働いている可能性があり、"
                    "市場全体の地合いよりテーマ内の戦略判定を優先すると実態に合いやすいです。"
                )

        st.caption(
            "⚠️ 週足レジームは過去データに基づく傾向判定であり、将来の優位を保証しません。"
            "急変時（決算・政策・地政学イベント）は自己相関が反転することがあります。"
        )

        # --- エントリータイミング: 寄り買い vs 引け買い ---
        st.markdown("---")
        st.markdown("##### 🌅🌙 場中ロングの入り口判定 — 寄り(始値) vs 引け(終値) どちらで買うべきか")
        st.caption(
            "日足OHLCから、代表的な保有パターンの勝率と平均リターンを集計します。"
            "寄り買い=当日の寄りで入り引けで決済（日中）。引け買い=引けで入り翌日以降に持ち越し。"
        )

        _timing_period = "1y" if analysis_period in ("1y", "2y") else "6mo"
        with st.spinner("🕐 日足OHLCを取得して寄り/引けの勝率を集計中..."):
            _ohlc = _fetch_ohlc_dict(tuple(ml_tickers), period=_timing_period)
        _lookback = 240 if _timing_period == "1y" else 120
        _timing = analyze_entry_timing(_ohlc, lookback_days=_lookback)

        if _timing is None:
            st.warning("日足OHLCが不足しており、寄り/引けの勝率判定ができませんでした。")
        else:
            _ti = _timing["intraday"]    # 寄り買い→当日引け
            _tn = _timing["nextday"]     # 引け買い→翌引け
            _to = _timing["overnight"]   # 引け買い→翌寄り

            st.markdown(
                f"<div style='border-left:5px solid {_timing['color']};"
                f"background:#1a1d23;padding:12px 16px;border-radius:6px;margin-bottom:10px;'>"
                f"<div style='color:{_timing['color']};font-size:1.3rem;font-weight:bold;'>"
                f"{_timing['emoji']} {_timing['verdict']}</div>"
                f"<div style='color:#ddd;font-size:0.82rem;margin-top:6px;'>{_timing['advice']}</div>"
                f"<div style='color:#888;font-size:0.72rem;margin-top:6px;'>"
                f"対象 {_timing['n_stocks']}銘柄 / 直近約{_timing['lookback_days']}営業日で集計</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            _t1, _t2, _t3 = st.columns(3)
            with _t1:
                st.markdown("**🌅 寄り買い→当日引け決済**")
                st.caption("日中（Open→Close）デイトレ型")
                st.metric("勝率", f"{_ti['win_rate']:.1f}%" if not np.isnan(_ti['win_rate']) else "—",
                          help=f"当日の始値で買い終値で売って上昇した割合。検証 {_ti['n']}サンプル")
                st.metric("平均リターン", f"{_ti['avg']:+.3f}%" if not np.isnan(_ti['avg']) else "—",
                          help=f"中央値 {_ti['median']:+.3f}%")
            with _t2:
                st.markdown("**🌙 引け買い→翌引け決済**")
                st.caption("翌日持ち越し（Close→翌Close）型")
                st.metric("勝率", f"{_tn['win_rate']:.1f}%" if not np.isnan(_tn['win_rate']) else "—",
                          help=f"引けで買い翌日の引けで売って上昇した割合。検証 {_tn['n']}サンプル")
                st.metric("平均リターン", f"{_tn['avg']:+.3f}%" if not np.isnan(_tn['avg']) else "—",
                          help=f"中央値 {_tn['median']:+.3f}%")
            with _t3:
                st.markdown("**🌃 引け買い→翌寄り決済**")
                st.caption("オーバーナイトのみ（Close→翌Open）")
                st.metric("勝率", f"{_to['win_rate']:.1f}%" if not np.isnan(_to['win_rate']) else "—",
                          help=f"引けで買い翌日の寄りで売って上昇した割合。検証 {_to['n']}サンプル")
                st.metric("平均リターン", f"{_to['avg']:+.3f}%" if not np.isnan(_to['avg']) else "—",
                          help=f"中央値 {_to['median']:+.3f}%")

            _timing_bar = pd.DataFrame({
                "パターン": ["寄り買い→当日引け", "引け買い→翌引け", "引け買い→翌寄り"],
                "勝率(%)": [_ti["win_rate"], _tn["win_rate"], _to["win_rate"]],
                "平均リターン(%)": [_ti["avg"], _tn["avg"], _to["avg"]],
            })
            _fig_timing = go.Figure()
            _fig_timing.add_trace(go.Bar(
                x=_timing_bar["パターン"], y=_timing_bar["勝率(%)"],
                name="勝率(%)", marker_color="#26a69a",
                text=[f"{v:.1f}%" for v in _timing_bar["勝率(%)"]], textposition="auto",
            ))
            _fig_timing.add_hline(y=50, line_dash="dash", line_color="#888",
                                  annotation_text="勝率50%", annotation_position="right")
            _fig_timing.update_layout(
                title="エントリータイミング別 勝率比較（50%超=上昇優勢）",
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=340, yaxis_title="勝率(%)", showlegend=False,
            )
            st.plotly_chart(_fig_timing, use_container_width=True)

            st.caption(
                "💡 読み方: 『寄り買い→当日引け』の勝率が高ければ日中に素直に上がりやすい（寄りエントリー有利）。"
                "『引け買い→翌引け』が高ければ持ち越しで上がりやすい（引けエントリー有利）。"
                "『引け買い→翌寄り』はギャップ（夜間の値動き）の傾向を示します。"
                " ※手数料・スリッページ・流動性は未考慮です。"
            )

    # 特徴量カテゴリ選択
    st.markdown("#### 🎛️ 使用する特徴量カテゴリ")
    feat_categories = st.multiselect(
        "分析に含める特徴量カテゴリ",
        ["財務・バリュエーション", "成長性", "出来高・需給", "テクニカル（短期MA）",
         "テクニカル（長期MA）", "テクニカル（オシレーター）", "ダウ理論",
         "ニュース感情", "マクロ経済"],
        default=["財務・バリュエーション", "成長性", "出来高・需給",
                 "テクニカル（短期MA）", "テクニカル（オシレーター）", "ダウ理論",
                 "ニュース感情", "マクロ経済"],
        key="ml_feat_categories"
    )

    # カテゴリ→特徴量マッピング
    CATEGORY_FEATURES = {
        "財務・バリュエーション": ["PER_セクター相対", "PBR_セクター相対", "予想PER",
                                "配当利回り", "PEG", "EV/EBITDA", "D/Eレシオ", "流動比率"],
        "成長性": ["ROE", "ROA", "EPS成長率", "売上成長率", "営業利益率", "純利益率"],
        "出来高・需給": ["出来高比率_5_20", "出来高比率_20_60", "出来高急増",
                       "機関投資家保有率", "空売り比率"],
        "テクニカル（短期MA）": ["MA5乖離率", "MA13乖離率", "MA20乖離率", "GC_DC"],
        "テクニカル（長期MA）": ["MA26乖離率", "MA60乖離率", "52週位置"],
        "テクニカル（オシレーター）": ["RSI", "MACD_hist", "BB位置", "ATR比率"],
        "ダウ理論": ["ダウ理論スコア"],
        "ニュース感情": ["ニュース感情", "ニュース件数"],
        "マクロ経済": ["米10年金利", "ドル円変化率", "原油変化率", "金変化率"],
    }

    # 当日ディスクキャッシュの有無を表示
    _today_str = datetime.now().strftime("%Y%m%d")
    import hashlib as _hl_ui
    _key_ui = _hl_ui.md5(
        f"{_today_str}|{analysis_period}|{','.join(sorted(ml_tickers))}".encode()
    ).hexdigest()[:16]
    _cache_path_ui = os.path.join(_ML_FEAT_CACHE_DIR, f"mlfeat_{_today_str}_{_key_ui}.parquet")
    _has_today_cache = os.path.exists(_cache_path_ui)

    cbtn1, cbtn2 = st.columns([3, 2])
    with cbtn1:
        run_ml = st.button("🚀 ML要因分析を実行", key="ml_run_btn")
    with cbtn2:
        force_refresh = st.checkbox("🔄 当日キャッシュを無視して再取得", value=False, key="ml_force_refresh")
    if _has_today_cache:
        st.caption("💾 本日取得済みのデータがキャッシュにあります。再実行しても即時表示されます（再取得は上のチェックをON）。")

    if run_ml:
        if force_refresh:
            # 当日ディスクキャッシュとメモリキャッシュをクリアして取り直す
            try:
                if os.path.exists(_cache_path_ui):
                    os.remove(_cache_path_ui)
            except Exception:
                pass
            try:
                fetch_ml_features.clear()
            except Exception:
                pass
        with st.spinner("📊 各銘柄のファンダメンタル・テクニカル・センチメントデータ取得中...（数分かかる場合があります）"):
            ml_df = fetch_ml_features(ml_tickers, period=analysis_period)

        if ml_df.empty or len(ml_df) < 5:
            st.error("十分なデータが取得できませんでした。銘柄数を増やすか、時間をおいて再試行してください。")
            # --- 原因診断の表示 ---
            _diag = st.session_state.get("ml_fetch_diag")
            if _diag is None:
                st.info(
                    "💡 データ取得関数が実行されず、当日キャッシュから空/不足の結果が返った可能性があります。"
                    "「🔄 当日キャッシュを無視して再取得」にチェックを入れて再実行してください。"
                )
            else:
                st.warning(
                    f"取得結果の内訳: 対象 {_diag['total']}銘柄 / "
                    f"成功 {_diag['ok']} / 履歴不足(30日未満) {_diag['hist_short']} / "
                    f"例外 {_diag['exception']}"
                )
                if _diag["ok"] == 0 and _diag["hist_short"] == _diag["total"]:
                    st.info(
                        "💡 全銘柄が『履歴不足』です。yfinanceのデータ取得が機能していない可能性が高いです。"
                        "ターミナルで `python -m pip install --upgrade yfinance curl_cffi` を実行してください。"
                    )
                if _diag["errors"]:
                    with st.expander("例外の詳細（先頭5件）"):
                        for _err in _diag["errors"]:
                            st.code(_err)
        else:
            # セクター相対PER/PBR計算
            ml_df, sector_per_medians = calc_sector_relative_per(ml_df)

            st.success(f"✅ {len(ml_df)}銘柄のデータ取得完了")

            # マクロ経済データ表示
            macro = fetch_macro_data()
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("米10年金利", f"{macro.get('US10Y', 'N/A')}%")
            mc2.metric("ドル円", f"¥{macro.get('USDJPY', 'N/A')}", f"{macro.get('USDJPY_chg_20d', 0):+.1f}%")
            mc3.metric("WTI原油", f"${macro.get('WTI', 'N/A')}", f"{macro.get('WTI_chg_20d', 0):+.1f}%")
            mc4.metric("金価格", f"${macro.get('GOLD', 'N/A')}", f"{macro.get('GOLD_chg_20d', 0):+.1f}%")

            # --- 特徴量テーブル表示 ---
            st.markdown("#### 📋 取得データ一覧（主要指標）")
            display_cols = ["ticker", "name", "sector", "PER", "PER_セクター相対", "PBR",
                          "ROE", "PEG", "EPS成長率", "売上成長率", "配当利回り",
                          "出来高比率_5_20", "RSI", "ニュース感情", "ダウ理論スコア", "20日リターン"]
            available_display = [c for c in display_cols if c in ml_df.columns]
            st.dataframe(
                ml_df[available_display].style.format({
                    "PER": "{:.1f}", "PER_セクター相対": "{:+.1f}", "PBR": "{:.2f}",
                    "ROE": "{:.1f}%", "PEG": "{:.2f}",
                    "EPS成長率": "{:+.1f}%", "売上成長率": "{:+.1f}%",
                    "配当利回り": "{:.2f}%", "出来高比率_5_20": "{:.2f}",
                    "RSI": "{:.1f}", "ニュース感情": "{:+.1f}",
                    "ダウ理論スコア": "{:+d}", "20日リターン": "{:+.2f}%"
                }, na_rep="-"),
                use_container_width=True, height=400
            )

            # --- セクター別PER水準表示 ---
            st.markdown("#### 🏭 セクター別PER中央値")
            per_df = pd.DataFrame(list(sector_per_medians.items()), columns=["セクター", "PER中央値"])
            per_df = per_df.dropna().sort_values("PER中央値")
            fig_per_sector = px.bar(per_df, x="セクター", y="PER中央値",
                                   color="PER中央値", color_continuous_scale="RdYlGn_r",
                                   title="セクター別PER中央値（低いほど割安傾向）")
            fig_per_sector.update_layout(template="plotly_dark",
                                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_per_sector, use_container_width=True)

            # --- 機械学習モデル構築 ---
            st.markdown("#### 🧠 LightGBM 要因分析（拡張版）")

            # 選択されたカテゴリから特徴量リストを構築
            feature_cols = []
            for cat in feat_categories:
                feature_cols.extend(CATEGORY_FEATURES.get(cat, []))
            # ml_dfに存在するカラムのみ使用
            feature_cols = [c for c in feature_cols if c in ml_df.columns]

            target_col = "20日リターン"

            # 欠損値処理: まず完全欠損行を除外、次に残りを中央値で補完
            ml_model_df = ml_df[feature_cols + [target_col, "ticker", "name", "sector"]].copy()
            # target_colが欠損の行は除外
            ml_model_df = ml_model_df.dropna(subset=[target_col])
            # 特徴量の欠損は中央値で補完
            for col in feature_cols:
                median_val = ml_model_df[col].median()
                ml_model_df[col] = ml_model_df[col].fillna(median_val)

            if len(ml_model_df) < 5:
                st.warning("有効データが5銘柄未満のため、ML分析にはデータが不足しています。全テーマ銘柄で再実行してください。")
            else:
                try:
                    from sklearn.ensemble import GradientBoostingRegressor
                    from sklearn.model_selection import cross_val_score
                    import sklearn

                    # LightGBMがインストールされていればそちらを使用
                    try:
                        import lightgbm as lgb
                        use_lgb = True
                    except ImportError:
                        use_lgb = False

                    X = ml_model_df[feature_cols].values
                    y = ml_model_df[target_col].values

                    if use_lgb:
                        model = lgb.LGBMRegressor(
                            n_estimators=100, max_depth=4, learning_rate=0.1,
                            num_leaves=15, min_child_samples=3, verbose=-1
                        )
                        model.fit(X, y)
                        importances = model.feature_importances_
                        model_name = "LightGBM"
                    else:
                        model = GradientBoostingRegressor(
                            n_estimators=100, max_depth=3, learning_rate=0.1,
                            min_samples_leaf=3
                        )
                        model.fit(X, y)
                        importances = model.feature_importances_
                        model_name = "GradientBoosting"

                    # クロスバリデーション（分割数は最低2、最大5。サンプル数不足でCVできない場合はスキップ）
                    _n = len(ml_model_df)
                    _cv = max(2, min(5, _n // 3))
                    if _n >= _cv * 2:
                        cv_scores = cross_val_score(model, X, y, cv=_cv, scoring="r2")
                    else:
                        import numpy as _np
                        cv_scores = _np.array([float("nan")])

                    st.markdown(f"**使用モデル**: {model_name}")
                    st.markdown(f"**学習データ**: {len(ml_model_df)}銘柄")
                    import numpy as _np2
                    if _np2.isnan(cv_scores).all():
                        st.markdown("**CV R²スコア**: —（サンプル数が少なくCV未実施）")
                    else:
                        st.markdown(f"**CV R²スコア**: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

                    # --- 特徴量重要度 ---
                    st.markdown("##### 📊 特徴量重要度（株価変動への寄与度）")
                    imp_df = pd.DataFrame({
                        "特徴量": feature_cols,
                        "重要度": importances
                    }).sort_values("重要度", ascending=True)

                    # 特徴量の日本語説明
                    feature_desc = {
                        "PER_セクター相対": "PER（セクター相対・低いほど割安）",
                        "PBR_セクター相対": "PBR（セクター相対・低いほど割安）",
                        "予想PER": "予想PER（フォワード）",
                        "配当利回り": "配当利回り",
                        "PEG": "PEGレシオ（低いほど成長対比割安）",
                        "EV/EBITDA": "EV/EBITDA倍率",
                        "D/Eレシオ": "負債資本倍率",
                        "流動比率": "流動比率（短期安全性）",
                        "ROE": "ROE（自己資本利益率）",
                        "ROA": "ROA（総資産利益率）",
                        "EPS成長率": "EPS成長率",
                        "売上成長率": "売上高成長率",
                        "営業利益率": "営業利益率",
                        "純利益率": "純利益率",
                        "出来高比率_5_20": "出来高（5日/20日比率）",
                        "出来高比率_20_60": "出来高（20日/60日比率）",
                        "出来高急増": "出来高急増フラグ",
                        "機関投資家保有率": "機関投資家保有率",
                        "空売り比率": "空売り比率",
                        "MA5乖離率": "5日MA乖離率",
                        "MA13乖離率": "13週MA乖離率",
                        "MA20乖離率": "20日MA乖離率",
                        "MA26乖離率": "26週MA乖離率",
                        "MA60乖離率": "60日MA乖離率",
                        "GC_DC": "ゴールデン/デッドクロス",
                        "52週位置": "52週レンジ内位置",
                        "RSI": "RSI（相対力指数）",
                        "MACD_hist": "MACDヒストグラム",
                        "BB位置": "ボリンジャーバンド位置",
                        "ATR比率": "ATR比率（ボラティリティ）",
                        "ダウ理論スコア": "ダウ理論（トレンド判定）",
                        "ニュース感情": "ニュース感情スコア",
                        "ニュース件数": "ニュース記事数",
                        "米10年金利": "米国10年債利回り",
                        "ドル円変化率": "ドル円20日変化率",
                        "原油変化率": "原油20日変化率",
                        "金変化率": "金価格20日変化率",
                    }
                    imp_df["説明"] = imp_df["特徴量"].map(feature_desc)

                    fig_imp = px.bar(imp_df, x="重要度", y="説明", orientation="h",
                                    color="重要度", color_continuous_scale="Viridis",
                                    title="特徴量重要度 — どの要因が株価変動に最も影響するか")
                    fig_imp.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        height=400, showlegend=False,
                        yaxis_title="", xaxis_title="重要度スコア"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # --- SHAP値分析（利用可能な場合） ---
                    st.markdown("##### 🔍 SHAP値分析（各銘柄の要因寄与度）")
                    try:
                        import shap
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X)

                        # SHAP summary（棒グラフ）
                        shap_imp = np.abs(shap_values).mean(axis=0)
                        shap_df = pd.DataFrame({
                            "特徴量": feature_cols,
                            "SHAP重要度": shap_imp
                        }).sort_values("SHAP重要度", ascending=True)
                        shap_df["説明"] = shap_df["特徴量"].map(feature_desc)

                        fig_shap = px.bar(shap_df, x="SHAP重要度", y="説明", orientation="h",
                                         color="SHAP重要度", color_continuous_scale="Plasma",
                                         title="SHAP値 — 各要因の平均的な影響度")
                        fig_shap.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            height=400, showlegend=False,
                            yaxis_title="", xaxis_title="平均|SHAP値|"
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                        # 個別銘柄のSHAP値表示
                        st.markdown("##### 📈 銘柄別 要因寄与度（SHAP値）")
                        shap_detail_df = pd.DataFrame(shap_values, columns=feature_cols)
                        shap_detail_df["ticker"] = ml_model_df["ticker"].values if "ticker" in ml_model_df.columns else range(len(shap_detail_df))

                        # ml_model_dfにtickerを紐づけ
                        ml_model_df_reset = ml_model_df.reset_index(drop=True)
                        ticker_names = ml_df.set_index("ticker")[["name"]].to_dict()["name"]

                        # SHAP Waterfall的な表示（上位/下位銘柄）
                        predictions = model.predict(X)
                        pred_df = pd.DataFrame({
                            "ticker": ml_model_df["ticker"].values,
                            "name": ml_model_df["name"].values,
                            "実績リターン": y,
                            "予測リターン": predictions,
                        })
                        pred_df = pred_df.sort_values("予測リターン", ascending=False)

                        col_top, col_bottom = st.columns(2)
                        with col_top:
                            st.markdown("**🟢 ML予測 上位銘柄（上昇期待）**")
                            for i, (_, row) in enumerate(pred_df.head(5).iterrows()):
                                # pred_dfの行番号からshap_valuesのインデックスを取得
                                ticker_list = ml_model_df["ticker"].tolist()
                                if row["ticker"] in ticker_list:
                                    pos = ticker_list.index(row["ticker"])
                                    sv = shap_values[pos]
                                    top_factors = sorted(zip(feature_cols, sv), key=lambda x: -x[1])[:3]
                                    factors_str = ", ".join([f"{feature_desc.get(f,f)}({v:+.2f})" for f,v in top_factors])
                                    st.markdown(
                                        f"**{row['name']}** ({row['ticker']}) "
                                        f"予測: {row['予測リターン']:+.1f}% / 実績: {row['実績リターン']:+.1f}%\n"
                                        f"  - 主要因: {factors_str}"
                                    )

                        with col_bottom:
                            st.markdown("**🔴 ML予測 下位銘柄（下落リスク）**")
                            for i, (_, row) in enumerate(pred_df.tail(5).iterrows()):
                                ticker_list = ml_model_df["ticker"].tolist()
                                if row["ticker"] in ticker_list:
                                    pos = ticker_list.index(row["ticker"])
                                    sv = shap_values[pos]
                                    bottom_factors = sorted(zip(feature_cols, sv), key=lambda x: x[1])[:3]
                                    factors_str = ", ".join([f"{feature_desc.get(f,f)}({v:+.2f})" for f,v in bottom_factors])
                                    st.markdown(
                                        f"**{row['name']}** ({row['ticker']}) "
                                        f"予測: {row['予測リターン']:+.1f}% / 実績: {row['実績リターン']:+.1f}%\n"
                                        f"  - 主要因: {factors_str}"
                                    )

                    except ImportError:
                        st.info("💡 SHAP値分析を有効にするには `pip install shap` を実行してください。特徴量重要度のみ表示しています。")

                    # --- 閾値ベースの 買い/売り/中立 シグナルサマリー ---
                    # 目的変数は「約1か月（20営業日）スパンのリターン」に対応。
                    # 中期目線（数週間〜数か月）の売買判断向けに、±5%を閾値とする。
                    st.markdown("---")
                    st.markdown("##### 🎯 売買シグナル判定（中期・約1か月スパン）")
                    st.caption(
                        "モデルは約1か月（20営業日）スケールのリターンを対象にしています。"
                        "数週間〜数か月の中期保有を前提に、予測リターン **+5%超=買い / −5%未満=売り / その間=中立** で区分しています。"
                        "目先数日のトレード用ではありません。"
                    )

                    _preds_sig = model.predict(X)
                    _buy_th, _sell_th = 5.0, -5.0

                    def _classify_signal(v):
                        if v >= _buy_th:
                            return "買い"
                        if v <= _sell_th:
                            return "売り"
                        return "中立"

                    sig_df = pd.DataFrame({
                        "ticker": ml_model_df["ticker"].values,
                        "name": ml_model_df["name"].values,
                        "sector": ml_model_df["sector"].values,
                        "予測リターン": _preds_sig,
                        "実績リターン": y,
                    })
                    sig_df["シグナル"] = sig_df["予測リターン"].apply(_classify_signal)

                    _buy_n = int((sig_df["シグナル"] == "買い").sum())
                    _sell_n = int((sig_df["シグナル"] == "売り").sum())
                    _neutral_n = int((sig_df["シグナル"] == "中立").sum())

                    _sc1, _sc2, _sc3 = st.columns(3)
                    _sc1.metric("🟢 買いシグナル", f"{_buy_n}銘柄", help="予測リターン +5%超")
                    _sc2.metric("⚪ 中立", f"{_neutral_n}銘柄", help="予測リターン −5%〜+5%")
                    _sc3.metric("🔴 売りシグナル", f"{_sell_n}銘柄", help="予測リターン −5%未満")

                    # 買い・売りそれぞれの一覧（予測リターン順）
                    _emoji = {"買い": "🟢", "売り": "🔴", "中立": "⚪"}

                    _buy_list = sig_df[sig_df["シグナル"] == "買い"].sort_values("予測リターン", ascending=False)
                    _sell_list = sig_df[sig_df["シグナル"] == "売り"].sort_values("予測リターン", ascending=True)

                    _lc, _rc = st.columns(2)
                    with _lc:
                        st.markdown("**🟢 買いシグナル一覧（予測リターン高い順）**")
                        if _buy_list.empty:
                            st.caption("該当なし（予測+5%超の銘柄はありません）")
                        else:
                            _show = _buy_list[["ticker", "name", "sector", "予測リターン", "実績リターン"]].copy()
                            st.dataframe(
                                _show.style.format({"予測リターン": "{:+.1f}%", "実績リターン": "{:+.1f}%"}),
                                use_container_width=True, hide_index=True, height=min(400, 60 + 35 * len(_show))
                            )
                    with _rc:
                        st.markdown("**🔴 売りシグナル一覧（予測リターン低い順）**")
                        if _sell_list.empty:
                            st.caption("該当なし（予測−5%未満の銘柄はありません）")
                        else:
                            _show = _sell_list[["ticker", "name", "sector", "予測リターン", "実績リターン"]].copy()
                            st.dataframe(
                                _show.style.format({"予測リターン": "{:+.1f}%", "実績リターン": "{:+.1f}%"}),
                                use_container_width=True, hide_index=True, height=min(400, 60 + 35 * len(_show))
                            )

                    st.caption(
                        "⚠️ シグナルはモデルの予測であり、将来の値動きを保証するものではありません。"
                        "ファンダメンタルズ・需給・マクロ環境とあわせて最終判断してください。"
                    )

                    # --- 散布図: 予測 vs 実績 ---
                    st.markdown("##### 📉 予測 vs 実績リターン")
                    predictions = model.predict(X)
                    scatter_df = pd.DataFrame({
                        "実績リターン(%)": y,
                        "予測リターン(%)": predictions,
                        "銘柄": ml_model_df["name"].values,
                        "セクター": ml_model_df["sector"].values,
                    })
                    fig_scatter = px.scatter(
                        scatter_df, x="実績リターン(%)", y="予測リターン(%)",
                        color="セクター", hover_name="銘柄",
                        title="ML予測 vs 実績リターン（対角線に近いほど予測精度が高い）"
                    )
                    fig_scatter.add_shape(type="line", x0=y.min(), y0=y.min(),
                                        x1=y.max(), y1=y.max(),
                                        line=dict(color="white", dash="dash"))
                    fig_scatter.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        height=500
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # --- 要因別の偏依存プロット（PDP） ---
                    st.markdown("##### 📊 偏依存プロット（各要因と株価リターンの関係）")
                    pdp_feature = st.selectbox(
                        "表示する特徴量",
                        feature_cols,
                        format_func=lambda x: feature_desc.get(x, x),
                        key="pdp_feature_select"
                    )

                    feat_idx = feature_cols.index(pdp_feature)
                    feat_values = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
                    pdp_values = []
                    for val in feat_values:
                        X_temp = X.copy()
                        X_temp[:, feat_idx] = val
                        pdp_values.append(model.predict(X_temp).mean())

                    fig_pdp = go.Figure()
                    fig_pdp.add_trace(go.Scatter(
                        x=feat_values, y=pdp_values,
                        mode="lines", line=dict(color="#00bcd4", width=3),
                        name=feature_desc.get(pdp_feature, pdp_feature)
                    ))
                    fig_pdp.update_layout(
                        title=f"偏依存プロット: {feature_desc.get(pdp_feature, pdp_feature)}",
                        xaxis_title=feature_desc.get(pdp_feature, pdp_feature),
                        yaxis_title="予測リターン(%)",
                        template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        height=350
                    )
                    st.plotly_chart(fig_pdp, use_container_width=True)

                    # --- 解釈サマリー ---
                    st.markdown("---")
                    st.markdown("#### 💡 分析結果の解釈ガイド")
                    st.markdown("""
| カテゴリ | 指標 | 意味 | 株価への影響 |
|----------|------|------|-------------|
| **財務** | PER（セクター相対） | セクター中央値との差 | 負=割安→上昇余地 |
| **財務** | PBR（セクター相対） | 純資産対比の割安度 | 1未満=解散価値割れ |
| **財務** | 配当利回り | 年間配当/株価 | 高いほど下値支え |
| **成長** | ROE | 自己資本利益率 | 高いほど資本効率良好 |
| **成長** | EPS成長率 | 1株利益の成長率 | 高いほど利益モメンタム |
| **成長** | 売上成長率 | トップライン成長 | 高いほど事業拡大 |
| **需給** | 出来高比率 | 短期/中期出来高比 | 1超=資金流入加速 |
| **需給** | 機関投資家保有率 | 機関の保有割合 | 高いほど安定需要 |
| **需給** | 空売り比率 | ショートポジション | 高いほど踏み上げ期待 |
| **テクニカル** | MA乖離率 | 移動平均からの乖離 | 正=上方乖離（過熱注意） |
| **テクニカル** | RSI | 相対力指数(0-100) | 70超=買われすぎ |
| **テクニカル** | MACD | トレンド転換シグナル | 正=上昇モメンタム |
| **テクニカル** | BB位置 | ボリンジャーバンド内位置 | 0-100%（50%=中央） |
| **テクニカル** | ダウ理論スコア | 高値安値の切り上げ/下げ | +2=強い上昇トレンド |
| **センチメント** | ニュース感情 | ニュースのポジネガ比率 | 正=好材料多い |
| **マクロ** | 米10年金利 | 米国長期金利 | 上昇=グロース株に逆風 |
| **マクロ** | ドル円変化率 | 為替変動 | 円安=輸出企業に追い風 |
| **マクロ** | 原油変化率 | エネルギーコスト | 上昇=エネ株↑、製造業↓ |
| **マクロ** | 金変化率 | リスクオフ指標 | 上昇=リスク回避ムード |
""")

                except ImportError as e:
                    st.error(f"必要なライブラリがインストールされていません: {e}")
                    st.code("pip install scikit-learn lightgbm shap", language="bash")
                except Exception as e:
                    # ImportError以外の例外（cross_val_scoreのcv不足・数値エラー等）を可視化
                    import traceback as _tb
                    st.error(f"ML分析の実行中にエラーが発生しました: {type(e).__name__}: {e}")
                    st.caption(
                        "学習データが少ない場合、クロスバリデーション(CV)の分割数が不足して失敗することがあります。"
                        "『全テーマ銘柄』で再実行するか、特徴量カテゴリを増やしてください。"
                    )
                    with st.expander("エラー詳細（トレースバック）"):
                        st.code(_tb.format_exc())

            # =============================================================
            # 🚀 初動検出 & ローリング学習（精度蓄積型）
            # =============================================================
            def _render_init_hit_rate():
                """初動スコアの20日後的中率を既存ログから集計して表示する。
                追加のネットワーク取得は行わない（prediction_log.jsonを読むだけ）。
                的中の定義: 選抜した銘柄の20日後リターン(actual_return)がプラスだった割合。
                """
                _mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
                _plog = os.path.join(_mdir, "prediction_log.json")
                if not os.path.exists(_plog):
                    return
                try:
                    with open(_plog, "r", encoding="utf-8") as f:
                        _log = json.load(f)
                except Exception:
                    return

                # init_score と actual_return が両方そろった検証済みエントリのみ対象
                rows = []
                for p in _log:
                    sc = p.get("init_score")
                    ar = p.get("actual_return")
                    if sc is None or ar is None:
                        continue
                    try:
                        sc = float(sc); ar = float(ar)
                    except (TypeError, ValueError):
                        continue
                    if np.isnan(sc) or np.isnan(ar):
                        continue
                    rows.append((sc, ar))

                if len(rows) < 10:
                    st.markdown("##### 🎯 初動スコアの的中率（蓄積中）")
                    st.info(
                        f"検証済みデータ {len(rows)}件（10件以上で的中率を表示）。\n\n"
                        "毎日実行して予測を記録し、20営業日経過すると実績が埋まって的中率が算出されます。"
                    )
                    return

                arr = np.array(rows, dtype=float)
                scores = arr[:, 0]
                rets = arr[:, 1]

                # ベースライン: 全検証銘柄の上昇率（初動スコアを問わない）
                base_up = float((rets > 0).mean() * 100)
                base_mean = float(rets.mean())

                st.markdown("##### 🎯 初動スコアの20日後的中率（実績ベース）")
                st.caption(
                    "「初動スコアで選んだ銘柄が20営業日後にプラスだった割合」を実測。"
                    "ベースライン（全銘柄の上昇率）を上回れば、初動スコアに選別効果があると言えます。"
                )

                # 閾値別の的中率テーブル
                thresholds = [5, 8, 10]
                trows = []
                for th in thresholds:
                    mask = scores >= th
                    n = int(mask.sum())
                    if n == 0:
                        trows.append({"閾値": f"{th}以上", "件数": 0,
                                      "上昇的中率": None, "平均リターン": None,
                                      "対ベースライン": None})
                        continue
                    up = float((rets[mask] > 0).mean() * 100)
                    mret = float(rets[mask].mean())
                    trows.append({
                        "閾値": f"{th}以上", "件数": n,
                        "上昇的中率": round(up, 1),
                        "平均リターン": round(mret, 2),
                        "対ベースライン": round(up - base_up, 1),
                    })

                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("検証済み総数", f"{len(rows)}件")
                mcol2.metric("ベースライン上昇率", f"{base_up:.1f}%")
                mcol3.metric("平均リターン(全体)", f"{base_mean:+.2f}%")

                tdf = pd.DataFrame(trows)
                st.dataframe(
                    tdf.style.format({
                        "上昇的中率": "{:.1f}%", "平均リターン": "{:+.2f}%",
                        "対ベースライン": "{:+.1f}pt",
                    }, na_rep="—"),
                    use_container_width=True, hide_index=True
                )

                # 判定コメント（スコア5以上の効果）
                s5 = next((r for r in trows if r["閾値"] == "5以上" and r["件数"] > 0), None)
                if s5 is not None and s5["対ベースライン"] is not None:
                    if s5["対ベースライン"] >= 5:
                        st.success(
                            f"✅ 初動スコア5以上はベースラインを {s5['対ベースライン']:+.1f}pt 上回っており、"
                            "選別効果が確認できます。"
                        )
                    elif s5["対ベースライン"] <= -5:
                        st.warning(
                            f"⚠️ 初動スコア5以上はベースラインを {s5['対ベースライン']:+.1f}pt 下回っています。"
                            "現状のスコアリングは選別に寄与していません（データ蓄積で変動する可能性あり）。"
                        )
                    else:
                        st.info(
                            "初動スコアとベースラインに大きな差はありません。"
                            "データが増えると傾向がはっきりします。"
                        )

            st.markdown("---")
            st.markdown("#### 🎯 上昇初動検出 — まだ上がっていないが、これから上がりそうな銘柄")
            st.caption("過去の上昇パターンを学習し、現在「初動」に入っている可能性が高い銘柄を抽出します")

            # --- 初動検出ロジック ---
            # 条件: まだ大きく上がっていない（直近20日リターンが控えめ）が、
            #        ファンダメンタル・テクニカルが上昇パターンに合致
            try:
                # 初動スコア計算
                init_scores = []
                for idx, row in ml_model_df.iterrows():
                    score = 0
                    reasons = []

                    # 条件1: まだ上がっていない（20日リターンが-5%〜+5%の範囲）
                    ret = row.get("20日リターン", 0)
                    if ret is None:
                        ret = 0
                    if -5 <= ret <= 5:
                        score += 2
                        reasons.append("株価まだ動いていない")
                    elif ret > 5:
                        score -= 2  # 既に上昇済み

                    # 条件2: ファンダメンタルが良好
                    per_rel = row.get("PER_セクター相対")
                    if per_rel is not None and per_rel < -3:
                        score += 2
                        reasons.append(f"PER割安(セクター比{per_rel:+.1f})")
                    roe_val = row.get("ROE")
                    if roe_val is not None and roe_val > 10:
                        score += 1
                        reasons.append(f"ROE良好({roe_val:.1f}%)")
                    eps_g = row.get("EPS成長率")
                    if eps_g is not None and eps_g > 10:
                        score += 2
                        reasons.append(f"EPS成長({eps_g:+.1f}%)")
                    rev_g = row.get("売上成長率") if "売上成長率" in ml_model_df.columns else None
                    if rev_g is not None and rev_g > 5:
                        score += 1
                        reasons.append(f"売上成長({rev_g:+.1f}%)")

                    # 条件3: 出来高が増加し始めている（初動サイン）
                    vol_5_20 = row.get("出来高比率_5_20") if "出来高比率_5_20" in ml_model_df.columns else None
                    if vol_5_20 is not None and vol_5_20 > 1.3:
                        score += 2
                        reasons.append(f"出来高増加({vol_5_20:.2f}x)")
                    vol_spike = row.get("出来高急増") if "出来高急増" in ml_model_df.columns else None
                    if vol_spike and vol_spike > 0:
                        score += 1
                        reasons.append("出来高急増検出")

                    # 条件4: テクニカルが上昇転換の初期段階
                    rsi_val = row.get("RSI")
                    if rsi_val is not None and 40 <= rsi_val <= 60:
                        score += 1
                        reasons.append(f"RSI中立圏({rsi_val:.0f})")
                    elif rsi_val is not None and rsi_val < 35:
                        score += 2
                        reasons.append(f"RSI売られすぎ圏({rsi_val:.0f})")

                    macd_h = row.get("MACD_hist")
                    if macd_h is not None and macd_h > 0 and macd_h < 0.5:
                        score += 1
                        reasons.append("MACD上昇転換直後")

                    gc_dc = row.get("GC_DC") if "GC_DC" in ml_model_df.columns else None
                    if gc_dc is not None and gc_dc == 1:
                        score += 2
                        reasons.append("ゴールデンクロス発生")

                    dow = row.get("ダウ理論スコア")
                    if dow is not None and dow >= 1:
                        score += 1
                        reasons.append(f"ダウ理論上昇({dow:+d})")

                    # 条件5: ニュース感情がポジティブ
                    sentiment = row.get("ニュース感情") if "ニュース感情" in ml_model_df.columns else None
                    if sentiment is not None and sentiment > 20:
                        score += 1
                        reasons.append(f"ニュース好感({sentiment:+.0f})")

                    # 条件6: 52週安値圏からの反発（底値圏）
                    pos_52w = row.get("52週位置") if "52週位置" in ml_model_df.columns else None
                    if pos_52w is not None and pos_52w < 30:
                        score += 2
                        reasons.append(f"52週安値圏({pos_52w:.0f}%)")
                    elif pos_52w is not None and 30 <= pos_52w <= 50:
                        score += 1
                        reasons.append(f"52週中間({pos_52w:.0f}%)")

                    # ML予測も加味（予測リターンが高い）
                    ml_pred = predictions[list(ml_model_df.index).index(idx)] if idx in ml_model_df.index else 0
                    if ml_pred > 3:
                        score += 2
                        reasons.append(f"ML予測上昇({ml_pred:+.1f}%)")
                    elif ml_pred > 1:
                        score += 1
                        reasons.append(f"ML予測やや上昇({ml_pred:+.1f}%)")

                    init_scores.append({
                        "ticker": row.get("ticker", ""),
                        "name": row.get("name", ""),
                        "sector": row.get("sector", ""),
                        "初動スコア": score,
                        "20日リターン": ret,
                        "ML予測": ml_pred,
                        "理由": " / ".join(reasons[:5]),
                    })

                init_df = pd.DataFrame(init_scores)
                # 初動スコアを ticker→score の辞書にして後段のログ記録で参照（追加取得なし）
                init_score_map = {r["ticker"]: int(r["初動スコア"]) for r in init_scores}
                st.session_state["_init_score_map"] = init_score_map
                if not init_df.empty:
                    init_df = init_df.sort_values("初動スコア", ascending=False)

                    # 上位銘柄表示
                    st.markdown("##### 🌟 初動検出スコア上位銘柄")
                    st.caption("スコアが高いほど「まだ上がっていないが上昇条件が揃っている」銘柄")

                    top_init = init_df[init_df["初動スコア"] >= 5].head(15)
                    if len(top_init) == 0:
                        top_init = init_df.head(10)
                        st.info("明確な初動パターン銘柄は少ないですが、相対的に条件が揃っている銘柄を表示します。")

                    for _, row in top_init.iterrows():
                        score_color = "#00c853" if row["初動スコア"] >= 8 else "#4caf50" if row["初動スコア"] >= 5 else "#ff9800"
                        ticker_code = row["ticker"].replace(".T", "")
                        ticker_val = row["ticker"]
                        name_val = row["name"]
                        sector_val = row["sector"]
                        ret_val = row["20日リターン"]
                        ml_val = row["ML予測"]
                        reason_val = row["理由"]
                        init_score_val = row["初動スコア"]
                        st.markdown(
                            f"<div style='background:#1a1a2e;padding:12px 16px;border-radius:10px;"
                            f"margin-bottom:6px;border-left:4px solid {score_color};'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                            f"<span style='color:#fff;font-weight:bold;font-size:1rem;'>"
                            f"<a href='https://kabutan.jp/stock/?code={ticker_code}' "
                            f"target='_blank' style='color:#fff;text-decoration:none;'>"
                            f"{name_val}</a>"
                            f" <span style='color:#888;'>({ticker_val})</span>"
                            f" <span style='color:#aaa;font-size:0.8rem;'>[{sector_val}]</span>"
                            f"</span>"
                            f"<span style='background:{score_color};color:#fff;padding:3px 10px;"
                            f"border-radius:12px;font-weight:bold;'>初動スコア {init_score_val}</span>"
                            f"</div>"
                            f"<div style='color:#ccc;font-size:0.82rem;margin-top:6px;'>"
                            f"20日リターン: {ret_val:+.1f}% / ML予測: {ml_val:+.1f}%"
                            f"</div>"
                            f"<div style='color:#90caf9;font-size:0.78rem;margin-top:4px;'>"
                            f"📋 {reason_val}"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                    # --- 初動スコア分布 ---
                    st.markdown("##### 📊 初動スコア分布")
                    fig_init = px.histogram(init_df, x="初動スコア", nbins=15,
                                           color_discrete_sequence=["#4fc3f7"],
                                           title="全銘柄の初動スコア分布")
                    fig_init.update_layout(template="plotly_dark",
                                         paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                         height=300)
                    st.plotly_chart(fig_init, use_container_width=True)

                    # --- 初動スコアの20日後的中率（既存ログを再利用・追加取得なし） ---
                    _render_init_hit_rate()

            except Exception as e:
                st.warning(f"初動検出でエラーが発生しました: {e}")

            # =============================================================
            # 📈 ローリング学習 & モデル蓄積（精度追跡）
            # =============================================================
            st.markdown("---")
            st.markdown("#### 📈 ローリング学習 — 日々精度が高まる仕組み")
            st.caption("""
            **仕組み**: 実行するたびに「今日の特徴量→20日後のリターン」を記録。
            蓄積データが増えるほど、モデルの学習データが充実し予測精度が向上します。
            """)

            # モデル保存ディレクトリ
            MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
            os.makedirs(MODEL_DIR, exist_ok=True)
            HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.json")
            PREDICTION_LOG = os.path.join(MODEL_DIR, "prediction_log.json")

            # --- 予測ログ保存 ---
            today_str = datetime.now().strftime("%Y-%m-%d")

            # 今日の予測を保存
            try:
                pred_log = []
                if os.path.exists(PREDICTION_LOG):
                    with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
                        pred_log = json.load(f)

                # 既存ログのサニタイズ: actual_return が NaN 文字列/数値なら None に戻して再計算対象にする
                for _p in pred_log:
                    _ar = _p.get("actual_return")
                    if _ar is not None:
                        try:
                            if np.isnan(float(_ar)):
                                _p["actual_return"] = None
                        except (TypeError, ValueError):
                            _p["actual_return"] = None

                # 今日の予測を追加（重複防止）
                today_entries = [p for p in pred_log if p.get("date") == today_str]
                if len(today_entries) == 0:
                    _recorded = 0
                    for i, (idx, row) in enumerate(ml_model_df.iterrows()):
                        # 予測値が NaN の銘柄は記録しない（学習データの汚染を防ぐ）
                        _pred = predictions[i]
                        if pd.isna(_pred):
                            continue
                        _entry_price = row.get("終値", None)
                        _tk = row.get("ticker", "")
                        # 初動スコアを同じログに1フィールドだけ追記（別ログを作らずサイズを抑える）
                        _init_map = st.session_state.get("_init_score_map", {})
                        pred_log.append({
                            "date": today_str,
                            "ticker": _tk,
                            "name": row.get("name", ""),
                            "predicted_return": float(_pred),
                            "init_score": _init_map.get(_tk),
                            "entry_price": float(_entry_price) if pd.notna(_entry_price) else None,
                            "features": {col: float(row[col]) if pd.notna(row[col]) else None
                                        for col in feature_cols},
                            "actual_return": None,  # 20営業日後に entry_price と比較して埋める
                        })
                        _recorded += 1

                    with open(PREDICTION_LOG, "w", encoding="utf-8") as f:
                        json.dump(pred_log, f, ensure_ascii=False, indent=2)
                    st.info(f"📝 {_recorded}銘柄の予測を記録しました（{today_str}）")

                # --- 過去の予測精度を検証（予測日から20営業日後の実績を正しく計算して埋める） ---
                # 予測日の終値(entry_price) → 現在終値 の変化率を「予測日起点の実績リターン」とする。
                # 現在終値は、その日 ml_df に含まれる銘柄はそこから、無ければ個別取得する。
                updated_count = 0
                _now_price_cache = {}
                # ml_df から現在終値を引けるように辞書化
                _mldf_price = {}
                if "終値" in ml_df.columns:
                    for _, _r in ml_df.iterrows():
                        if pd.notna(_r.get("終値")):
                            _mldf_price[_r["ticker"]] = float(_r["終値"])

                def _get_current_price(tk):
                    """現在終値を取得（ml_df優先→キャッシュ→yfinance個別）。失敗時None。"""
                    if tk in _mldf_price:
                        return _mldf_price[tk]
                    if tk in _now_price_cache:
                        return _now_price_cache[tk]
                    try:
                        _h = yf.Ticker(tk).history(period="5d")
                        if not _h.empty:
                            _p = float(_h["Close"].iloc[-1])
                            _now_price_cache[tk] = _p
                            return _p
                    except Exception:
                        pass
                    _now_price_cache[tk] = None
                    return None

                # 実際の営業日換算に近づけるため、20営業日≒28暦日を閾値にする
                for entry in pred_log:
                    if entry.get("actual_return") is not None:
                        continue
                    try:
                        entry_date = datetime.strptime(entry["date"], "%Y-%m-%d")
                    except Exception:
                        continue
                    days_elapsed = (datetime.now() - entry_date).days
                    if days_elapsed < 28:
                        continue
                    ticker = entry.get("ticker", "")
                    entry_price = entry.get("entry_price")
                    if not ticker or entry_price is None or entry_price <= 0:
                        continue
                    cur_price = _get_current_price(ticker)
                    if cur_price is None or cur_price <= 0:
                        continue
                    actual_ret = (cur_price / entry_price - 1) * 100
                    if pd.notna(actual_ret):
                        entry["actual_return"] = float(actual_ret)
                        updated_count += 1

                if updated_count > 0:
                    with open(PREDICTION_LOG, "w", encoding="utf-8") as f:
                        json.dump(pred_log, f, ensure_ascii=False, indent=2)
                    st.info(f"📊 {updated_count}件の過去予測に実績リターンを記録しました")

                # --- 精度追跡表示 ---
                # actual_return が有効な数値（None/NaN除外）のものだけを「検証済み」とする
                def _is_valid_num(v):
                    if v is None:
                        return False
                    try:
                        return not np.isnan(float(v))
                    except (TypeError, ValueError):
                        return False
                verified = [p for p in pred_log
                            if _is_valid_num(p.get("actual_return"))
                            and _is_valid_num(p.get("predicted_return"))]
                if len(verified) >= 5:
                    st.markdown("##### 📊 過去の予測精度（蓄積データ）")
                    v_pred = [p["predicted_return"] for p in verified]
                    v_actual = [p["actual_return"] for p in verified]

                    # 相関係数
                    corr = np.corrcoef(v_pred, v_actual)[0, 1]
                    # 方向一致率（上昇/下落の方向が合っているか）
                    direction_match = sum(1 for p, a in zip(v_pred, v_actual)
                                         if (p > 0 and a > 0) or (p <= 0 and a <= 0))
                    direction_rate = direction_match / len(verified) * 100

                    # MAE
                    mae = np.mean(np.abs(np.array(v_pred) - np.array(v_actual)))

                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("蓄積予測数", f"{len(verified)}件")
                    pc2.metric("予測-実績相関", f"{corr:.3f}")
                    pc3.metric("方向一致率", f"{direction_rate:.1f}%")
                    pc4.metric("平均絶対誤差", f"{mae:.2f}%")

                    # 精度推移グラフ（日付別）
                    dates_in_log = sorted(set(p["date"] for p in verified))
                    if len(dates_in_log) >= 2:
                        accuracy_by_date = []
                        for d in dates_in_log:
                            d_entries = [p for p in verified if p["date"] == d]
                            if len(d_entries) >= 3:
                                d_pred = [p["predicted_return"] for p in d_entries]
                                d_actual = [p["actual_return"] for p in d_entries]
                                d_corr = np.corrcoef(d_pred, d_actual)[0, 1] if len(d_pred) > 1 else 0
                                d_dir = sum(1 for p, a in zip(d_pred, d_actual)
                                           if (p > 0 and a > 0) or (p <= 0 and a <= 0)) / len(d_entries) * 100
                                accuracy_by_date.append({"日付": d, "相関": d_corr, "方向一致率": d_dir})

                        if accuracy_by_date:
                            acc_df = pd.DataFrame(accuracy_by_date)
                            fig_acc = go.Figure()
                            fig_acc.add_trace(go.Scatter(
                                x=acc_df["日付"], y=acc_df["方向一致率"],
                                mode="lines+markers", name="方向一致率(%)",
                                line=dict(color="#4caf50", width=2)
                            ))
                            fig_acc.add_trace(go.Scatter(
                                x=acc_df["日付"], y=acc_df["相関"] * 100,
                                mode="lines+markers", name="相関×100",
                                line=dict(color="#2196f3", width=2)
                            ))
                            fig_acc.add_hline(y=50, line_dash="dash", line_color="gray",
                                            annotation_text="ランダム(50%)")
                            fig_acc.update_layout(
                                title="予測精度の推移（日々の蓄積で精度向上を確認）",
                                template="plotly_dark",
                                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                height=350, yaxis_title="%"
                            )
                            st.plotly_chart(fig_acc, use_container_width=True)

                else:
                    remaining = len(pred_log) - len(verified)
                    st.info(
                        f"📊 蓄積データ: {len(pred_log)}件の予測記録 / "
                        f"検証済み: {len(verified)}件 / 検証待ち: {remaining}件\n\n"
                        f"20日以上経過すると実績リターンが記録され、精度追跡が始まります。"
                        f"**毎日実行するほど精度追跡データが蓄積されます。**"
                    )

                # --- 蓄積データで再学習（データが十分な場合） ---
                # verified のうち、actual_return が有効な数値（NaN/None除外）のものだけ対象にする
                _valid_verified = [
                    p for p in verified
                    if p.get("actual_return") is not None
                    and not (isinstance(p.get("actual_return"), float) and np.isnan(p["actual_return"]))
                    and p.get("features")
                ]
                if len(_valid_verified) >= 30:
                    st.markdown("##### 🔄 蓄積データによる再学習")
                    # 過去の予測ログから学習データを構築
                    retrain_X = []
                    retrain_y = []
                    for entry in _valid_verified:
                        feats = entry.get("features", {})
                        actual = entry.get("actual_return")
                        # 目的変数が数値でなければスキップ
                        try:
                            actual = float(actual)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(actual):
                            continue
                        # 特徴量: None は np.nan として扱い、後段でまとめて補完する
                        row_feats = []
                        for col in feature_cols:
                            v = feats.get(col, None)
                            try:
                                v = float(v) if v is not None else np.nan
                            except (TypeError, ValueError):
                                v = np.nan
                            row_feats.append(v)
                        retrain_X.append(row_feats)
                        retrain_y.append(actual)

                    retrain_X = np.array(retrain_X, dtype=float)
                    retrain_y = np.array(retrain_y, dtype=float)

                    # 目的変数に NaN が残る行を除外（保険）
                    _mask = ~np.isnan(retrain_y)
                    retrain_X = retrain_X[_mask]
                    retrain_y = retrain_y[_mask]

                    # 特徴量の NaN を列ごとの中央値で補完（全NaN列は0）
                    if retrain_X.size > 0:
                        col_median = np.nanmedian(retrain_X, axis=0)
                        col_median = np.where(np.isnan(col_median), 0.0, col_median)
                        _nan_idx = np.where(np.isnan(retrain_X))
                        retrain_X[_nan_idx] = np.take(col_median, _nan_idx[1])

                    if len(retrain_X) >= 30:
                        # 蓄積データで再学習
                        if use_lgb:
                            retrained_model = lgb.LGBMRegressor(
                                n_estimators=150, max_depth=5, learning_rate=0.08,
                                num_leaves=20, min_child_samples=3, verbose=-1
                            )
                        else:
                            retrained_model = GradientBoostingRegressor(
                                n_estimators=150, max_depth=4, learning_rate=0.08,
                                min_samples_leaf=3
                            )
                        retrained_model.fit(retrain_X, retrain_y)

                        # 再学習モデルで現在の銘柄を予測（保存用）
                        try:
                            retrained_pred = retrained_model.predict(X)
                        except Exception:
                            retrained_pred = None

                        # クロスバリデーション（分割数は最低2、サンプル不足ならスキップ）
                        _n_re = len(retrain_X)
                        _cv_re = max(2, min(5, _n_re // 5))
                        if _n_re >= _cv_re * 2:
                            retrain_cv = cross_val_score(retrained_model, retrain_X, retrain_y,
                                                         cv=_cv_re, scoring="r2")
                            retrain_cv_mean = float(retrain_cv.mean())
                            retrain_cv_std = float(retrain_cv.std())
                        else:
                            retrain_cv_mean = float("nan")
                            retrain_cv_std = float("nan")

                        # 学習済みモデルをディスクに保存（日々更新）
                        try:
                            import joblib
                            _model_path = os.path.join(MODEL_DIR, "rolling_model.joblib")
                            joblib.dump(
                                {"model": retrained_model, "feature_cols": feature_cols,
                                 "trained_at": today_str, "n_samples": int(_n_re),
                                 "cv_r2": retrain_cv_mean},
                                _model_path
                            )
                        except Exception:
                            pass

                        st.markdown(f"**蓄積学習データ**: {len(retrain_X)}件")
                        if np.isnan(retrain_cv_mean):
                            st.markdown("**再学習CV R²**: —（サンプル不足でCV未実施）")
                        else:
                            st.markdown(f"**再学習CV R²**: {retrain_cv_mean:.3f} (±{retrain_cv_std:.3f})")

                        # 元モデルのCVと比較（元がnanなら比較しない）
                        _base_cv = float(cv_scores.mean()) if not np.isnan(cv_scores).all() else float("nan")
                        if not np.isnan(_base_cv):
                            st.markdown(f"**元モデルCV R²**: {_base_cv:.3f}")
                        if not np.isnan(retrain_cv_mean) and not np.isnan(_base_cv):
                            improvement = retrain_cv_mean - _base_cv
                            if improvement > 0:
                                st.success(f"✅ 蓄積データにより精度が向上しています (+{improvement:.3f})")
                            else:
                                st.info("蓄積データがさらに増えると精度向上が期待できます")
                    else:
                        st.info(f"有効な蓄積データが不足しています（現在{len(retrain_X)}件 / 30件で再学習開始）。")

            except Exception as e:
                import traceback as _tb_roll
                st.warning(f"ローリング学習の記録でエラー: {type(e).__name__}: {e}")
                with st.expander("エラー詳細"):
                    st.code(_tb_roll.format_exc())


# =====================================================================
# Tab 18: 適時開示 — TDnet/JPXからリアルタイム取得
# =====================================================================
with tab18:
    st.subheader("📄 適時開示情報（リアルタイム）")
    st.caption("東証TDnet・JPXから適時開示をリアルタイム取得。決算・業績修正・自社株買い・増配・MBO等の重要開示を即座に確認できます。")

    # --- 適時開示データ取得関数 ---
    @st.cache_data(ttl=180)  # 3分キャッシュ
    def fetch_tdnet_disclosures():
        """JPX適時開示閲覧サービスからRSS/スクレイピングで開示情報を取得"""
        disclosures = []

        # 方法1: JPX適時開示RSSフィード
        try:
            import xml.etree.ElementTree as ET
            # JPXの適時開示RSSフィード
            rss_urls = [
                "https://www.release.tdnet.info/inbs/I_main_00.html",
            ]

            # Google News経由で適時開示ニュースを取得（より確実）
            search_queries = [
                "適時開示+決算+業績修正",
                "適時開示+自社株買い+増配+株主還元",
                "適時開示+MBO+TOB+公開買付",
                "適時開示+株式分割+新規上場",
            ]

            for query in search_queries:
                url = f"https://news.google.com/rss/search?q={quote(query)}&hl=ja&gl=JP&ceid=JP:ja"
                try:
                    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code != 200:
                        continue
                    root = ET.fromstring(resp.content)
                    for item in root.findall(".//item"):
                        title = item.findtext("title", "").strip()
                        link = item.findtext("link", "").strip()
                        pub_date = item.findtext("pubDate", "").strip()

                        if not title:
                            continue

                        # 日時パース
                        from email.utils import parsedate_to_datetime
                        dt_obj = None
                        time_str = ""
                        if pub_date:
                            try:
                                dt_obj = parsedate_to_datetime(pub_date)
                                time_str = dt_obj.strftime("%m/%d %H:%M")
                            except Exception:
                                time_str = pub_date[:16]

                        disclosures.append({
                            "title": title,
                            "link": link,
                            "time": time_str,
                            "dt": dt_obj,
                            "source": "Google News",
                        })
                except Exception:
                    continue

        except Exception:
            pass

        # 方法2: 株探の適時開示ページからスクレイピング
        try:
            kabutan_url = "https://kabutan.jp/disclosures/"
            resp = requests.get(kabutan_url, timeout=10,
                              headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # 開示一覧テーブルを解析
                rows = soup.select("table tr")
                for row in rows[:100]:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        time_cell = cells[0].get_text(strip=True)
                        code_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                        title_cell = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                        link_tag = cells[2].find("a") if len(cells) > 2 else None
                        link_href = "https://kabutan.jp" + link_tag["href"] if link_tag and link_tag.get("href") else ""

                        if title_cell:
                            disclosures.append({
                                "title": f"[{code_cell}] {title_cell}" if code_cell else title_cell,
                                "link": link_href,
                                "time": time_cell,
                                "dt": None,
                                "source": "株探",
                            })
        except Exception:
            pass

        # 方法3: JPX適時開示ページから直接取得
        try:
            jpx_url = "https://www.jpx.co.jp/listing/disclosure/index.html"
            resp = requests.get(jpx_url, timeout=10,
                              headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select(".component-normal-table tbody tr, .disclosure-list li")
                for item in items[:50]:
                    text = item.get_text(strip=True)
                    link_tag = item.find("a")
                    link_href = ""
                    if link_tag and link_tag.get("href"):
                        href = link_tag["href"]
                        link_href = href if href.startswith("http") else "https://www.jpx.co.jp" + href
                    if text and len(text) > 5:
                        disclosures.append({
                            "title": text[:200],
                            "link": link_href,
                            "time": "",
                            "dt": None,
                            "source": "JPX",
                        })
        except Exception:
            pass

        # 重複除去
        seen = set()
        unique_disclosures = []
        for d in disclosures:
            key = d["title"][:50]
            if key not in seen:
                seen.add(key)
                unique_disclosures.append(d)

        # 時刻でソート（aware/naive混在を回避）
        def _sort_key(x):
            dt = x.get("dt")
            if dt is None:
                return datetime(2000, 1, 1).timestamp()
            # タイムゾーン情報を統一してtimestampに変換
            try:
                return dt.timestamp()
            except Exception:
                return datetime(2000, 1, 1).timestamp()

        unique_disclosures.sort(key=_sort_key, reverse=True)
        return unique_disclosures

    # --- 開示カテゴリ分類関数 ---
    def classify_disclosure(title):
        """開示タイトルからカテゴリを推定"""
        categories = {
            "決算": ["決算", "四半期", "通期", "中間", "業績"],
            "業績修正": ["修正", "上方", "下方", "予想の修正", "業績予想"],
            "自社株買い": ["自己株式", "自社株", "取得"],
            "配当": ["配当", "増配", "減配", "復配", "特別配当"],
            "株主還元": ["株主還元", "株主優待", "優待"],
            "M&A・TOB": ["MBO", "TOB", "公開買付", "買収", "合併", "統合", "子会社化"],
            "株式分割": ["株式分割", "株式併合", "分割"],
            "新規事業": ["新規事業", "提携", "協業", "新製品", "受注"],
            "人事・ガバナンス": ["代表", "役員", "取締役", "社長", "ガバナンス"],
            "資金調達": ["増資", "新株", "社債", "転換社債", "CB"],
            "上場・廃止": ["上場", "廃止", "上場廃止", "IPO"],
            "その他": [],
        }

        for cat, keywords in categories.items():
            for kw in keywords:
                if kw in title:
                    return cat
        return "その他"

    def get_category_emoji(category):
        """カテゴリ別の絵文字"""
        emoji_map = {
            "決算": "📊", "業績修正": "📈", "自社株買い": "💰",
            "配当": "💎", "株主還元": "🎁", "M&A・TOB": "🤝",
            "株式分割": "✂️", "新規事業": "🚀", "人事・ガバナンス": "👔",
            "資金調達": "🏦", "上場・廃止": "🔔", "その他": "📋",
        }
        return emoji_map.get(category, "📋")

    def get_category_color(category):
        """カテゴリ別の色"""
        color_map = {
            "決算": "#2196f3", "業績修正": "#ff9800", "自社株買い": "#4caf50",
            "配当": "#9c27b0", "株主還元": "#e91e63", "M&A・TOB": "#f44336",
            "株式分割": "#00bcd4", "新規事業": "#8bc34a", "人事・ガバナンス": "#607d8b",
            "資金調達": "#795548", "上場・廃止": "#ff5722", "その他": "#9e9e9e",
        }
        return color_map.get(category, "#9e9e9e")

    # --- メインUI ---
    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        filter_categories = st.multiselect(
            "カテゴリフィルタ",
            ["決算", "業績修正", "自社株買い", "配当", "株主還元",
             "M&A・TOB", "株式分割", "新規事業", "人事・ガバナンス",
             "資金調達", "上場・廃止", "その他"],
            default=["決算", "業績修正", "自社株買い", "配当", "M&A・TOB", "株式分割"],
            key="disclosure_filter"
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 最新を取得", key="disclosure_refresh")

    # テーマ銘柄のみフィルタオプション
    theme_only = st.toggle("テーマ関連銘柄のみ表示", value=False, key="disclosure_theme_only")

    # データ取得
    with st.spinner("📄 適時開示情報を取得中..."):
        disclosures = fetch_tdnet_disclosures()

    if not disclosures:
        st.warning("適時開示情報を取得できませんでした。しばらく待ってから再試行してください。")
    else:
        # カテゴリ分類
        for d in disclosures:
            d["category"] = classify_disclosure(d["title"])

        # フィルタ適用
        filtered = [d for d in disclosures if d["category"] in filter_categories]

        # テーマ銘柄フィルタ
        if theme_only:
            all_theme_codes = set()
            for info in THEMES.values():
                for ticker in info["stocks"].keys():
                    code = ticker.replace(".T", "")
                    all_theme_codes.add(code)
            filtered = [d for d in filtered if any(code in d["title"] for code in all_theme_codes)]

        # カテゴリ別集計
        st.markdown("#### 📊 カテゴリ別開示件数")
        cat_counts = {}
        for d in disclosures:
            cat = d["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        cat_cols = st.columns(min(6, len(cat_counts)))
        for i, (cat, count) in enumerate(sorted(cat_counts.items(), key=lambda x: -x[1])):
            if i < len(cat_cols):
                with cat_cols[i % len(cat_cols)]:
                    emoji = get_category_emoji(cat)
                    st.metric(f"{emoji} {cat}", f"{count}件")

        # --- 開示一覧表示 ---
        st.markdown(f"#### 📄 開示一覧（{len(filtered)}件）")

        if not filtered:
            st.info("フィルタ条件に合致する開示がありません。カテゴリフィルタを調整してください。")
        else:
            for d in filtered[:50]:  # 最大50件表示
                cat = d["category"]
                emoji = get_category_emoji(cat)
                color = get_category_color(cat)
                title = d["title"]
                link = d["link"]
                time_str = d["time"]
                source = d["source"]

                # タイトルをリンク化
                if link:
                    title_html = f"<a href='{link}' target='_blank' style='color:#fff;text-decoration:none;border-bottom:1px dotted #666;'>{title}</a>"
                else:
                    title_html = title

                st.markdown(
                    f"<div style='background:#1a1a2e;padding:10px 14px;border-radius:8px;"
                    f"margin-bottom:4px;border-left:4px solid {color};'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                    f"<div style='flex:1;'>"
                    f"<span style='background:{color};color:#fff;padding:2px 8px;border-radius:10px;"
                    f"font-size:0.72rem;font-weight:bold;margin-right:8px;'>{emoji} {cat}</span>"
                    f"<span style='color:#fff;font-size:0.88rem;'>{title_html}</span>"
                    f"</div>"
                    f"<div style='color:#888;font-size:0.72rem;white-space:nowrap;margin-left:12px;'>"
                    f"{time_str}"
                    f"<br><span style='color:#666;'>{source}</span>"
                    f"</div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

        # --- 注目開示ハイライト ---
        st.markdown("---")
        st.markdown("#### 🔥 注目開示（株価インパクト大）")
        st.caption("業績上方修正・自社株買い・増配・MBO等、株価に大きなインパクトを与える可能性がある開示")

        high_impact_keywords = [
            "上方修正", "増配", "自社株買い", "自己株式の取得",
            "MBO", "TOB", "公開買付", "株式分割", "特別配当",
            "過去最高", "最高益", "復配", "買収", "合併",
        ]

        high_impact = []
        for d in disclosures:
            for kw in high_impact_keywords:
                if kw in d["title"]:
                    high_impact.append(d)
                    break

        if high_impact:
            for d in high_impact[:20]:
                cat = d["category"]
                emoji = get_category_emoji(cat)
                color = get_category_color(cat)
                title = d["title"]
                link = d["link"]
                time_str = d["time"]

                if link:
                    title_html = f"<a href='{link}' target='_blank' style='color:#fff;text-decoration:none;'>{title}</a>"
                else:
                    title_html = title

                st.markdown(
                    f"<div style='background:#1a1a2e;padding:12px 16px;border-radius:10px;"
                    f"margin-bottom:6px;border-left:4px solid #ff6f00;'>"
                    f"<div>"
                    f"<span style='font-size:1.1rem;margin-right:6px;'>⚡</span>"
                    f"<span style='background:{color};color:#fff;padding:2px 8px;border-radius:10px;"
                    f"font-size:0.72rem;font-weight:bold;margin-right:8px;'>{emoji} {cat}</span>"
                    f"<span style='color:#fff;font-size:0.92rem;font-weight:bold;'>{title_html}</span>"
                    f"</div>"
                    f"<div style='color:#888;font-size:0.75rem;margin-top:4px;'>"
                    f"⏰ {time_str} | {d['source']}"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("現時点でインパクトの大きい開示は検出されていません。")

        # --- テーマ関連の開示 ---
        st.markdown("---")
        st.markdown("#### 🎯 テーマ関連銘柄の開示")

        theme_related = []
        all_theme_stocks = {}
        for theme_name, info in THEMES.items():
            for ticker, name in info["stocks"].items():
                code = ticker.replace(".T", "")
                all_theme_stocks[code] = {"name": name, "theme": theme_name}

        for d in disclosures:
            for code, stock_info in all_theme_stocks.items():
                if code in d["title"]:
                    theme_related.append({
                        **d,
                        "matched_code": code,
                        "matched_name": stock_info["name"],
                        "matched_theme": stock_info["theme"],
                    })
                    break

        if theme_related:
            for d in theme_related[:30]:
                cat = d["category"]
                emoji = get_category_emoji(cat)
                color = get_category_color(cat)
                theme_name = d["matched_theme"]
                stock_name = d["matched_name"]
                title = d["title"]
                link = d["link"]
                time_str = d["time"]

                if link:
                    title_html = f"<a href='{link}' target='_blank' style='color:#fff;text-decoration:none;'>{title}</a>"
                else:
                    title_html = title

                st.markdown(
                    f"<div style='background:#1a1a2e;padding:10px 14px;border-radius:8px;"
                    f"margin-bottom:4px;border-left:4px solid #4da6ff;'>"
                    f"<div>"
                    f"<span style='background:#1565c0;color:#fff;padding:2px 8px;border-radius:10px;"
                    f"font-size:0.7rem;margin-right:6px;'>{theme_name}</span>"
                    f"<span style='color:#4da6ff;font-weight:bold;margin-right:6px;'>{stock_name}</span>"
                    f"<span style='background:{color};color:#fff;padding:2px 6px;border-radius:8px;"
                    f"font-size:0.68rem;margin-right:6px;'>{emoji}{cat}</span>"
                    f"<span style='color:#ddd;font-size:0.85rem;'>{title_html}</span>"
                    f"</div>"
                    f"<div style='color:#888;font-size:0.72rem;margin-top:3px;'>"
                    f"⏰ {time_str}"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("テーマ関連銘柄の開示は現時点で検出されていません。")

        # --- 外部リンク ---
        st.markdown("---")
        st.markdown("#### 🔗 適時開示 外部サイト")
        link_cols = st.columns(4)
        with link_cols[0]:
            st.markdown("[📄 TDnet適時開示](https://www.release.tdnet.info/inbs/I_main_00.html)")
        with link_cols[1]:
            st.markdown("[📊 株探 適時開示](https://kabutan.jp/disclosures/)")
        with link_cols[2]:
            st.markdown("[🏛️ JPX開示情報](https://www.jpx.co.jp/listing/disclosure/index.html)")
        with link_cols[3]:
            st.markdown("[📰 EDINET](https://disclosure2.edinet-fsa.go.jp/)")
