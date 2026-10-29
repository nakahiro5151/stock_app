@echo off
cd /d "%~dp0"
streamlit run us_stock_dashboard.py --server.port 8503
pause
