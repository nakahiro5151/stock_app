#!/bin/bash
# マーケット分析ダッシュボード起動スクリプト (Mac/Linux用)
cd "$(dirname "$0")"
open "http://localhost:8502" 2>/dev/null || xdg-open "http://localhost:8502" 2>/dev/null &
python3 -m streamlit run stock_dashboard.py --server.headless true --server.port 8502
