# 株分析ダッシュボード

日本株のテーマ別資金フロー分析、マーケット概況、TOPIXスクリーニング、
ドラゴンキング理論に基づくバブル検出を行うツール群です。

## セットアップ

### 1. Python のインストール

Mac の場合:
```bash
brew install python
```

### 2. 依存パッケージのインストール

```bash
cd stock_app
pip3 install -r requirements.txt
```

### 3. 環境変数の設定（任意）

```bash
cp .env.example .env
# .env を編集して EDINET_API_KEY を設定（決算分析機能で使用）
```

## アプリ一覧と起動方法

### メイン: テーマ別資金フロー分析 (stock_theme_dashboard.py)
28テーマの日本株資金フロー、モメンタム、日米リードラグ、売買シグナル、
ニュース連動分析、ポートフォリオ管理、決算速報など。

```bash
# Mac/Linux
./start_dashboard.sh

# Windows
株分析ダッシュボード.bat

# 手動起動
python3 -m streamlit run stock_theme_dashboard.py
```

### マーケット分析 (stock_dashboard.py)
日経225、TOPIX、ドル円、VIX、コモディティのリアルタイムチャート、
相関ヒートマップ、TDnet適時開示情報。

```bash
./start_market.sh
```

### TOPIXスクリーナー (topix_screener.py)
次世代TOPIX選定基準に基づく銘柄スクリーニング。

```bash
./start_topix.sh
```

### ドラゴンキング分析 (Colab/ローカル)
- `dragon_king_stock_analysis.py` - ドラゴンキング理論による暴落予兆分析
- `lppls_confidence_indicator.py` - LPPLS信頼度指標（ドットコムバブル風）
- `lppls_fast_multi_asset.py` - 高速LPPLS（複数資産対応）
- `lppls_sp500_analysis.py` - SP500 LPPLS分析

## ファイル構成

```
stock_app/
├── stock_theme_dashboard.py   # メインダッシュボード（7800行）
├── stock_dashboard.py         # マーケット分析
├── topix_screener.py          # TOPIXスクリーナー
├── dragon_king_stock_analysis.py  # ドラゴンキング分析
├── lppls_confidence_indicator.py  # LPPLS信頼度指標
├── lppls_fast_multi_asset.py      # 高速LPPLS
├── lppls_sp500_analysis.py        # SP500 LPPLS
├── lppls_analysis_guide.md        # LPPLS分析ガイド
├── my_portfolio.json          # ポートフォリオデータ
├── my_watchlist.json          # ウォッチリストデータ
├── stock_dict_cache.json      # 銘柄辞書キャッシュ
├── kessan_cache.json          # 決算キャッシュ
├── activist_cache.json        # アクティビスト情報キャッシュ
├── .streamlit/config.toml     # Streamlitテーマ設定
├── .env.example               # 環境変数テンプレート
├── .env                       # 環境変数（要設定）
├── requirements.txt           # 依存パッケージ
├── 株分析ダッシュボード.bat    # Windows起動用
├── start_dashboard.sh         # Mac/Linux起動用
├── start_market.sh            # マーケット分析起動用
├── start_topix.sh             # TOPIXスクリーナー起動用
└── .gitignore
```

## 注意事項

- 株価データは yfinance 経由で取得（インターネット接続必須）
- キャッシュファイル（*_cache.json）は自動生成されるため、初回は空でもOK
- ポートフォリオ・ウォッチリストは各自の保有銘柄に合わせて編集してください
