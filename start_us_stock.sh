#!/bin/bash
# 米国株投資分析ダッシュボード起動スクリプト
cd "$(dirname "$0")"
streamlit run us_stock_dashboard.py --server.port 8503
