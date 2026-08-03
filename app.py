import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------
# PAGE CONFIG & CSS
# -----------------------------------------
st.set_page_config(
    page_title='Football Edge Finder — ML Engine',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded'
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Premium gradient text for headers */
.gradient-text {
    background: linear-gradient(90deg, #00d4aa 0%, #0077ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.sub-header-text {
    color: #888888;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

/* Styled metric containers */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 212, 170, 0.25);
    border-radius: 12px;
    padding: 15px 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 212, 170, 0.2);
    border-color: rgba(0, 212, 170, 0.6);
}

/* Verdict Badges */
.verdict-badge {
    padding: 10px 15px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.1rem;
    text-align: center;
    margin-top: 10px;
    color: white;
}
.verdict-positive {
    background: rgba(0, 212, 170, 0.2);
    border: 1px solid #00d4aa;
    box-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
}
.verdict-negative {
    background: rgba(255, 50, 50, 0.2);
    border: 1px solid #ff3232;
    box-shadow: 0 0 10px rgba(255, 50, 50, 0.3);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom buttons */
.stButton > button {
    background: linear-gradient(90deg, #00d4aa 0%, #00a080 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #00a080 0%, #00d4aa 100%);
    transform: scale(1.02);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------
# IMPORT ML INTERFACE MODULE
# -----------------------------------------
try:
    import app_ml
    importlib.reload(app_ml)
    render_ml_backtester_tab = app_ml.render_ml_backtester_tab
    render_ml_predictions_tab = app_ml.render_ml_predictions_tab
except Exception as e:
    st.error(f"Error loading Machine Learning engine modules: {e}")
    render_ml_backtester_tab = None
    render_ml_predictions_tab = None

# -----------------------------------------
# HEADER & NAVIGATION
# -----------------------------------------
st.markdown('<div class="gradient-text">⚽ Football Edge Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-text">Dual-Model Machine Learning Engine • Dixon-Coles Statistical Baseline + XGBoost Ensemble</div>', unsafe_allow_html=True)

tab_scanner, tab_backtester = st.tabs([
    "🎯 Live Fixture Scanner & Opportunities", 
    "🤖 Walk-Forward ML Backtester"
])

# =========================================
# TAB 1: LIVE FIXTURE SCANNER & OPPORTUNITIES
# =========================================
with tab_scanner:
    if render_ml_predictions_tab is not None:
        render_ml_predictions_tab()
    else:
        st.warning("Live ML Scanner module unavailable.")

# =========================================
# TAB 2: WALK-FORWARD ML BACKTESTER
# =========================================
with tab_backtester:
    if render_ml_backtester_tab is not None:
        render_ml_backtester_tab()
    else:
        st.warning("ML Backtester module unavailable.")

st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #666; font-size: 0.8rem;">
    For quantitative research and educational purposes only. Past performance does not guarantee future results.
</div>
""", unsafe_allow_html=True)
