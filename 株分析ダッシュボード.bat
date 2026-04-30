@echo off
cd /d "%~dp0"
start "" http://localhost:8501
python -m streamlit run stock_theme_dashboard.py --server.headless true
pause
