@echo off
title Football Edge Dashboard — Starting...
cd /d "c:\Users\gavin\.gemini\antigravity\scratch\betting-dashboard"
echo Starting Football Edge Dashboard...
echo.
start "" "http://localhost:8501"
py -3 -m streamlit run app.py --server.port 8501
