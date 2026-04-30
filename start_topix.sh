#!/bin/bash
# TOPIXスクリーナー起動スクリプト (Mac/Linux用)
cd "$(dirname "$0")"
open "http://localhost:8503" 2>/dev/null || xdg-open "http://localhost:8503" 2>/dev/null &
python3 -m streamlit run topix_screener.py --server.headless true --server.port 8503
