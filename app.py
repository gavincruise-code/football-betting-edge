import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import backend modules
try:
    import data_utils
    from poisson_engine import (
        prob_over25_matrix, prob_draw, implied_probability, 
        has_edge, expected_value, score_matrix
    )
    from backtester import run_backtest, BetRecord, BacktestResult
except ImportError as e:
    st.error(f"Backend modules not found. Ensure poisson_engine, data_utils, backtester, and ev_calculator are in the same directory. Error: {e}")

# -----------------------------------------
# PAGE CONFIG & CSS
# -----------------------------------------
st.set_page_config(page_title='Football Edge Finder', page_icon='⚽', layout='wide')

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
    margin-bottom: 1rem;
}

/* Styled metric cards */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-radius: 12px;
    padding: 15px 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 212, 170, 0.15);
    border-color: rgba(0, 212, 170, 0.5);
}

/* Verdict Badges */
.verdict-badge {
    padding: 10px 15px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.2rem;
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

/* Hide default footer and hamburger menu if desired */
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
# HELPER COMPONENTS
# -----------------------------------------
def render_verdict_card(title, model_prob, implied_prob, edge, ev):
    has_positive_edge = edge > 0
    badge_class = "verdict-positive" if has_positive_edge else "verdict-negative"
    icon = "✅ +EV" if has_positive_edge else "❌ -EV"
    
    st.markdown(f"### {title}")
    st.metric("Model Probability", f"{model_prob*100:.1f}%")
    st.metric("Implied Probability", f"{implied_prob*100:.1f}%")
    st.metric("Edge", f"{edge*100:.1f}%", delta=f"{edge*100:.1f}%", delta_color="normal" if edge > 0 else "inverse")
    st.metric("EV (per £10)", f"£{ev:.2f}")
    
    st.markdown(f'<div class="verdict-badge {badge_class}">{icon}</div>', unsafe_allow_html=True)

import importlib
try:
    import app_ml
    importlib.reload(app_ml)
    render_ml_backtester_tab = app_ml.render_ml_backtester_tab
    render_ml_predictions_tab = app_ml.render_ml_predictions_tab
except Exception as e:
    render_ml_backtester_tab = None
    render_ml_predictions_tab = None

# -----------------------------------------
# MAIN APP STRUCTURE
# -----------------------------------------
st.markdown('<div class="gradient-text">⚽ Football Edge Finder</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Backtester", "🧮 Live EV Calculator", "🤖 ML Backtester", "🎯 ML Predictions"])

# =========================================
# TAB 1: BACKTESTER
# =========================================
with tab1:
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        data_source = st.radio("Data Source", ["Download from football-data.co.uk", "Upload CSV"])
        
        df = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Football Data CSV", type="csv")
            if uploaded_file:
                try:
                    df = data_utils.load_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        else:
            try:
                leagues = data_utils.LEAGUES
                countries = list(leagues.keys())
                country = st.selectbox("Country", countries)
                league_name = st.selectbox("League", list(leagues[country].keys()))
                league_code = leagues[country][league_name]
                
                seasons = data_utils.get_available_seasons()
                season = st.selectbox("Season", seasons)
                season_code = data_utils.season_to_code(season)
                
                if st.button("📥 Download Data"):
                    with st.spinner(f"Downloading {league_name} {season}..."):
                        try:
                            df = data_utils.download_league_data(league_code, season_code)
                            st.session_state['bt_df'] = df
                            st.success("Data loaded successfully!")
                        except Exception as e:
                            st.error(f"Failed to download data: {e}")
            except Exception as e:
                st.warning("data_utils functions not fully available for download. Please ensure backend is complete.")
        
        # Load from session state if available
        if df is None and 'bt_df' in st.session_state:
            df = st.session_state['bt_df']
            
        st.divider()
        edge_margin = st.slider("Edge Margin (%)", 1, 15, 5, 1) / 100.0
        min_signal = st.slider("Min Signal Strength", 1, 4, 2, 1)
        stake = st.number_input("Stake (£)", min_value=1.0, value=10.0, step=1.0)
        
        st.write("Markets to trade:")
        col_m1, col_m2 = st.columns(2)
        trade_over25 = col_m1.checkbox("Over 2.5", value=True)
        trade_draw = col_m2.checkbox("Draw", value=True)
        markets = []
        if trade_over25: markets.append('over25')
        if trade_draw: markets.append('draw')
        
        run_btn = st.button("🚀 Run Backtest", use_container_width=True)

    if run_btn:
        if df is None:
            st.warning("Please load data first (upload CSV or download).")
        else:
            with st.spinner("Running backtest..."):
                try:
                    result = run_backtest(
                        df, 
                        edge_margin=edge_margin, 
                        min_signal_strength=min_signal, 
                        stake=stake, 
                        markets=markets
                    )
                    
                    st.subheader(f"Backtest Results")
                    
                    # Metrics
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Matches Analyzed", result.matches_analyzed)
                    c2.metric("Bets Placed", result.total_bets)
                    c3.metric("Win Rate", f"{result.win_rate*100:.1f}%")
                    c4.metric("Total P/L", f"£{result.total_profit_loss:.2f}", delta=round(result.total_profit_loss, 2))
                    c5.metric("ROI", f"{result.roi*100:.1f}%", delta=round(result.roi*100, 2))
                    
                    st.divider()
                    
                    col_chart1, col_chart2 = st.columns([2, 1])
                    
                    with col_chart1:
                        # Bankroll Curve
                        fig_br = go.Figure()
                        fig_br.add_trace(go.Scatter(
                            x=list(range(len(result.bankroll_curve))),
                            y=result.bankroll_curve,
                            mode='lines',
                            line=dict(color='#00d4aa', width=3),
                            name='Bankroll',
                            fill='tozeroy',
                            fillcolor='rgba(0, 212, 170, 0.1)'
                        ))
                        fig_br.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                        fig_br.update_layout(
                            title="Cumulative P/L (£)",
                            template="plotly_dark",
                            margin=dict(l=20, r=20, t=40, b=20),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Bet Number",
                            yaxis_title="Profit / Loss (£)",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_br, use_container_width=True)
                        
                    with col_chart2:
                        # Monthly P/L
                        if result.monthly_pl:
                            months = list(result.monthly_pl.keys())
                            pls = list(result.monthly_pl.values())
                            colors = ['#00d4aa' if val >= 0 else '#ff3232' for val in pls]
                            
                            fig_m = go.Figure(data=[go.Bar(
                                x=months,
                                y=pls,
                                marker_color=colors
                            )])
                            fig_m.update_layout(
                                title="Monthly P/L",
                                template="plotly_dark",
                                margin=dict(l=20, r=20, t=40, b=20),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig_m, use_container_width=True)
                    
                    st.subheader("Bet Log")
                    if result.bet_log:
                        # Convert to DataFrame for display
                        log_df = pd.DataFrame([vars(b) for b in result.bet_log])
                        
                        # Format dataframe
                        def color_result(val):
                            color = '#00d4aa' if val == 'WIN' else '#ff3232'
                            return f'color: {color}; font-weight: bold'
                            
                        styled_df = log_df[['date', 'home_team', 'away_team', 'market', 'model_prob', 'implied_prob', 'edge', 'odds', 'result', 'profit_loss', 'actual_score']].style.map(color_result, subset=['result']).format({
                            'model_prob': '{:.1%}',
                            'implied_prob': '{:.1%}',
                            'edge': '{:.1%}',
                            'odds': '{:.2f}',
                            'profit_loss': '£{:.2f}'
                        })
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # Edge distribution
                        st.subheader("Edge Distribution")
                        fig_edge = px.histogram(
                            log_df, x="edge", nbins=20, 
                            title="Distribution of Edge % on Placed Bets",
                            color_discrete_sequence=['#00d4aa'],
                            template="plotly_dark"
                        )
                        fig_edge.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig_edge, use_container_width=True)
                        
                    else:
                        st.info("No bets matched the criteria.")

                except Exception as e:
                    st.error(f"Error during backtest: {e}")
    elif df is None:
        # Empty State
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(0,212,170,0.3);">
            <h2 style="color: #00d4aa; margin-bottom: 15px;">Welcome to the Backtester</h2>
            <p style="color: #aaa; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Evaluate your quantitative models against historical data. 
                Upload a CSV or download directly from football-data.co.uk using the sidebar controls to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================
# TAB 2: LIVE EV CALCULATOR
# =========================================
with tab2:
    st.header("🧮 Live Match Assessment")
    
    with st.container():
        c_in1, c_in2 = st.columns(2)
        with c_in1:
            home_team = st.text_input("Home Team", value="Arsenal")
            home_scored = st.number_input("Home Avg Goals Scored (Home)", value=1.5, step=0.1)
            home_conceded = st.number_input("Home Avg Goals Conceded (Home)", value=1.0, step=0.1)
        with c_in2:
            away_team = st.text_input("Away Team", value="Liverpool")
            away_scored = st.number_input("Away Avg Goals Scored (Away)", value=1.1, step=0.1)
            away_conceded = st.number_input("Away Avg Goals Conceded (Away)", value=1.4, step=0.1)
            
    st.markdown("### Current Market Odds")
    c_odds1, c_odds2, c_odds3, c_odds4 = st.columns(4)
    odds_o25 = c_odds1.number_input("Over 2.5 Odds", value=1.90, step=0.05)
    odds_u25 = c_odds2.number_input("Under 2.5 Odds", value=1.95, step=0.05)
    odds_draw = c_odds3.number_input("Draw Odds", value=3.40, step=0.05)
    margin_ev = c_odds4.slider("EV Edge Margin (%)", 1, 15, 5, 1) / 100.0

    st.divider()
    
    try:
        # Calculations
        lam_home = (home_scored + away_conceded) / 2.0
        lam_away = (away_scored + home_conceded) / 2.0
        
        prob_o25 = prob_over25_matrix(lam_home, lam_away)
        prob_u25 = 1 - prob_o25
        prob_x = prob_draw(lam_home, lam_away)
        
        imp_o25 = implied_probability(odds_o25) if odds_o25 > 1 else 0
        imp_u25 = implied_probability(odds_u25) if odds_u25 > 1 else 0
        imp_draw = implied_probability(odds_draw) if odds_draw > 1 else 0
        
        edge_o25 = prob_o25 - imp_o25
        edge_u25 = prob_u25 - imp_u25
        edge_draw = prob_x - imp_draw
        
        ev_o25 = expected_value(prob_o25, odds_o25, 10.0) if odds_o25 > 1 else 0
        ev_u25 = expected_value(prob_u25, odds_u25, 10.0) if odds_u25 > 1 else 0
        ev_draw = expected_value(prob_x, odds_draw, 10.0) if odds_draw > 1 else 0
        
        st.subheader("📊 Model Verdict")
        vc1, vc2, vc3 = st.columns(3)
        
        with vc1:
            render_verdict_card("Over 2.5 Goals", prob_o25, imp_o25, edge_o25, ev_o25)
        with vc2:
            render_verdict_card("Under 2.5 Goals", prob_u25, imp_u25, edge_u25, ev_u25)
        with vc3:
            render_verdict_card("Draw", prob_x, imp_draw, edge_draw, ev_draw)
            
        st.divider()
        
        st.subheader("🎯 Scoreline Matrix")
        
        sm = score_matrix(lam_home, lam_away, max_goals=6)
        
        # Prepare data for heatmap
        z_data = [[sm[i][j] * 100 for j in range(7)] for i in range(7)]
        text_data = [[f"{sm[i][j]*100:.1f}%" for j in range(7)] for i in range(7)]
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"Awy {i}" for i in range(7)],
            y=[f"Hom {i}" for i in range(7)],
            text=text_data,
            texttemplate="%{text}",
            colorscale=[[0, "#000000"], [1, "#00d4aa"]],
            showscale=False
        ))
        
        fig_hm.update_layout(
            title=f"Predicted Score Probabilities: {home_team} vs {away_team}",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_autorange='reversed'
        )
        
        c_hm, c_top = st.columns([3, 1])
        with c_hm:
            st.plotly_chart(fig_hm, use_container_width=True)
            
        with c_top:
            st.markdown("### Top 5 Likely Scores")
            scores_list = []
            for i in range(7):
                for j in range(7):
                    scores_list.append((i, j, sm[i][j]))
            
            scores_list.sort(key=lambda x: x[2], reverse=True)
            
            for rank, (h, a, p) in enumerate(scores_list[:5]):
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid #00d4aa;">
                    <strong>{h} - {a}</strong> 
                    <span style="float: right; color: #00d4aa;">{p*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error in calculations: {e}. Make sure poisson_engine is correctly implemented.")

# =========================================
# TAB 3: ML BACKTESTER
# =========================================
with tab3:
    if render_ml_backtester_tab is not None:
        render_ml_backtester_tab()
    else:
        st.warning("ML Backtester module unavailable.")

# =========================================
# TAB 4: ML PREDICTIONS
# =========================================
with tab4:
    if render_ml_predictions_tab is not None:
        render_ml_predictions_tab()
    else:
        st.warning("ML Predictions module unavailable.")

st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #666; font-size: 0.8rem;">
    For educational and research purposes only. Past performance does not guarantee future results.
</div>
""", unsafe_allow_html=True)

