"""
Streamlit ML Tabs Interface Component
======================================
Renders Tab 3 (ML Backtester with model comparisons & SHAP charts)
and Tab 4 (Live ML Predictions & Scoreline Heatmaps).
"""

import sys
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Guaranteed Poisson math functions (no external import dependencies)
def score_matrix(lam_h: float, lam_a: float, max_goals: int = 6) -> list:
    """Returns 2D score probability matrix for 0..max_goals."""
    def pmf(k, l):
        if k < 0 or l <= 0: return 0.0
        try:
            return (math.pow(l, k) * math.exp(-l)) / math.factorial(k)
        except Exception:
            return 0.0
    return [[pmf(h, lam_h) * pmf(a, lam_a) for a in range(max_goals + 1)] for h in range(max_goals + 1)]

def implied_probability(decimal_odds: float) -> float:
    """Returns 1 / odds."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds

def expected_value(model_prob: float, decimal_odds: float, stake: float = 10.0) -> float:
    """Returns expected profit in GBP."""
    return (model_prob * decimal_odds * stake) - stake

try:
    from ml.config import UNDERSTAT_LEAGUES
except Exception:
    UNDERSTAT_LEAGUES = {
        "EPL": "EPL", "La_Liga": "La_Liga", "Bundesliga": "Bundesliga", "Serie_A": "Serie_A", "Ligue_1": "Ligue_1",
        "USA (MLS)": "USA", "Argentina": "ARG", "Brazil": "BRA", "Mexico": "MEX", "Japan": "JPN", "China": "CHN",
        "Sweden": "SWE", "Norway": "NOR", "Denmark": "DNK", "Finland": "FIN", "Poland": "POL", "Romania": "ROU",
        "Switzerland": "SWZ", "Austria": "AUT"
    }

try:
    from ml.calibration import kelly_stake
except Exception:
    def kelly_stake(model_prob: float, decimal_odds: float, bankroll: float = 1000.0, fraction: float = 0.25, max_pct: float = 0.05) -> float:
        if decimal_odds <= 1.0 or model_prob <= 0.0 or bankroll <= 0.0:
            return 0.0
        b = decimal_odds - 1.0
        p = model_prob
        q = 1.0 - p
        f_star = (b * p - q) / b
        if f_star <= 0.0:
            return 0.0
        f_adj = min(f_star * fraction, max_pct)
        return round(bankroll * f_adj, 2)

def check_ml_imports():
    try:
        from ml.config import UNDERSTAT_LEAGUES, DEFAULT_EDGE_MARGIN, DEFAULT_STAKE
        from ml.data_ingestion import build_master_dataset
        from ml.feature_engine import compute_all_features
        from ml.backtester_v2 import walk_forward_backtest, compare_strategies
        from ml.calibration import kelly_stake
        return True, None
    except Exception as e:
        return False, str(e)

def render_ml_backtester_tab():
    """
    Tab 3: 🤖 ML Backtester
    """
    is_ok, err_msg = check_ml_imports()
    if not is_ok:
        st.error(f"ML Pipeline dependencies not fully installed. Details: {err_msg}")
        return

    from ml.config import UNDERSTAT_LEAGUES
    from ml.data_ingestion import build_master_dataset
    from ml.feature_engine import compute_all_features
    from ml.backtester_v2 import walk_forward_backtest, compare_strategies

    st.markdown("### 🤖 Advanced ML Walk-Forward Backtester")
    st.write("Train and evaluate Dixon-Coles, XGBoost, and Dual-Ensemble ML models using zero-lookahead walk-forward cross-validation.")

    with st.sidebar:
        st.header("🤖 ML Controls")
        selected_leagues = st.multiselect(
            "Leagues",
            list(UNDERSTAT_LEAGUES.keys()),
            default=["EPL"]
        )
        season_range = st.slider("Season Range", 2019, 2024, (2019, 2023))
        seasons_list = list(range(season_range[0], season_range[1] + 1))

        edge_margin = st.slider("ML Edge Margin (%)", 1, 15, 5, 1) / 100.0
        stake_amount = st.number_input("Flat Stake (£)", min_value=1.0, value=10.0, step=1.0)
        use_kelly = st.checkbox("Enable Quarter-Kelly Staking", value=True)

        strategy_choice = st.radio(
            "Strategy",
            ["Compare All Models", "Dual Ensemble", "XGBoost Only", "Dixon-Coles Only", "Random Baseline"]
        )

        run_ml_btn = st.button("⚡ Run ML Backtest", use_container_width=True)

    if run_ml_btn:
        if not selected_leagues:
            st.warning("Please select at least one league.")
            return

        with st.spinner(f"Building dataset for {selected_leagues} ({season_range[0]}-{season_range[1]})..."):
            try:
                understat_leagues = [UNDERSTAT_LEAGUES[lg] for lg in selected_leagues]
                master_df = build_master_dataset(leagues=understat_leagues, seasons=seasons_list, use_cache=True)

                if master_df.empty:
                    st.error("No data fetched for the selected configuration.")
                    return

                st.info(f"Loaded {len(master_df)} raw matches. Extracting 120+ rolling features...")
                feat_df = compute_all_features(master_df)

                if feat_df.empty:
                    st.error("Not enough historical data to generate features.")
                    return

                st.session_state['ml_feat_df'] = feat_df
                st.success(f"Feature engineering complete! {len(feat_df)} matches ready for ML validation.")

            except Exception as e:
                st.error(f"Error during feature building: {e}")
                return

    # Check if feature df in session state
    feat_df = st.session_state.get('ml_feat_df', None)

    if feat_df is not None and not feat_df.empty:
        strat_map = {
            "Compare All Models": "all",
            "Dual Ensemble": "dual",
            "XGBoost Only": "ml_only",
            "Dixon-Coles Only": "dc_only",
            "Random Baseline": "random"
        }
        selected_strat = strat_map[strategy_choice]

        with st.spinner("Executing walk-forward validation..."):
            try:
                if selected_strat == "all":
                    results = compare_strategies(
                        feat_df,
                        edge_margin=edge_margin,
                        stake=stake_amount,
                        use_kelly=use_kelly
                    )
                    
                    st.subheader("🏆 Model Strategy Comparison")
                    comp_data = []
                    for r in results:
                        comp_data.append({
                            'Strategy': r.strategy_name.upper(),
                            'Bets Placed': r.total_bets,
                            'Win Rate': f"{r.win_rate:.1%}",
                            'Flat P/L (£)': f"£{r.total_profit_loss_flat:.2f}",
                            'Flat ROI': f"{r.roi_flat:.1%}",
                            'Kelly P/L (£)': f"£{r.total_profit_loss_kelly:.2f}",
                            'Sharpe': f"{r.sharpe_ratio:.2f}",
                            'Brier Score': f"{r.brier_score:.3f}",
                            'AUC-ROC': f"{r.auc_roc:.3f}"
                        })
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

                    # Overlay Bankroll Curves
                    fig_comp = go.Figure()
                    colors = {'random': '#ff5555', 'dc_only': '#0077ff', 'ml_only': '#ffaa00', 'dual': '#00d4aa'}
                    for r in results:
                        fig_comp.add_trace(go.Scatter(
                            y=r.bankroll_curve_flat,
                            mode='lines',
                            name=r.strategy_name.upper(),
                            line=dict(color=colors.get(r.strategy_name, '#ffffff'), width=3)
                        ))
                    fig_comp.update_layout(
                        title="Walk-Forward Cumulative Profit (£) Across Strategies",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Bet Number",
                        yaxis_title="Profit / Loss (£)"
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                else:
                    res = walk_forward_backtest(
                        feat_df,
                        edge_margin=edge_margin,
                        stake=stake_amount,
                        use_kelly=use_kelly,
                        strategy=selected_strat
                    )

                    st.subheader(f"ML Validation Results — {selected_strat.upper()}")
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Analyzed", res.matches_analyzed)
                    m2.metric("Bets Placed", res.total_bets)
                    m3.metric("Win Rate", f"{res.win_rate*100:.1f}%")
                    m4.metric("Flat P/L", f"£{res.total_profit_loss_flat:.2f}", delta=round(res.total_profit_loss_flat, 2))
                    m5.metric("Kelly P/L", f"£{res.total_profit_loss_kelly:.2f}", delta=round(res.total_profit_loss_kelly, 2))
                    m6.metric("Sharpe", f"{res.sharpe_ratio:.2f}")

                    st.divider()

                    # Charts
                    c_c1, c_c2 = st.columns(2)
                    with c_c1:
                        fig_br = go.Figure()
                        fig_br.add_trace(go.Scatter(
                            y=res.bankroll_curve_flat,
                            name="Flat Stake (£10)",
                            line=dict(color='#00d4aa', width=3)
                        ))
                        if use_kelly:
                            fig_br.add_trace(go.Scatter(
                                y=res.bankroll_curve_kelly,
                                name="Quarter-Kelly",
                                line=dict(color='#0077ff', width=3, dash='dash')
                            ))
                        fig_br.update_layout(
                            title="Bankroll Growth Curves",
                            template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_br, use_container_width=True)

                    with c_c2:
                        if res.monthly_pl:
                            fig_m = px.bar(
                                x=list(res.monthly_pl.keys()),
                                y=list(res.monthly_pl.values()),
                                title="Monthly P/L (£)",
                                template="plotly_dark",
                                color_discrete_sequence=['#00d4aa']
                            )
                            fig_m.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_m, use_container_width=True)

                    # Bet log & SHAP feature contributions
                    st.subheader("Detailed Bet Log")
                    if res.bet_log:
                        b_df = pd.DataFrame([vars(b) for b in res.bet_log])
                        st.dataframe(b_df[['date', 'home_team', 'away_team', 'league', 'dc_prob', 'ml_prob', 'implied_prob', 'edge', 'odds', 'result', 'profit_loss']], use_container_width=True)

                        # Top SHAP feature sample
                        st.subheader("Top Feature Explanations (Sample Bet)")
                        sample_bet = res.bet_log[-1]
                        if sample_bet.top_features:
                            st.markdown(f"**Match:** {sample_bet.home_team} vs {sample_bet.away_team} ({sample_bet.date})")
                            for feat_name, shap_val, feat_val in sample_bet.top_features:
                                direction = "⬆️ Higher Over 2.5 prob" if shap_val > 0 else "⬇️ Lower Over 2.5 prob"
                                st.markdown(f"- **{feat_name}** = `{feat_val:.2f}` ({direction}, SHAP: `{shap_val:+.3f}`)")

            except Exception as e:
                st.error(f"Error during ML execution: {e}")

def render_ml_predictions_tab():
    """
    Tab 4: 🎯 Live ML Predictions & Opportunity Scanner
    """
    st.header("🎯 Live ML Match Predictor & Opportunity Scanner")
    st.write("Scan upcoming fixtures automatically to detect +EV betting opportunities or evaluate custom match statistics.")

    sub_tab1, sub_tab2 = st.tabs(["📅 Live Upcoming Fixtures Feed", "🧮 Custom Match Calculator"])

    with sub_tab1:
        st.subheader("📅 Automated Fixtures & Value Bet Scanner")
        
        c_act1, c_act2 = st.columns([3, 1])
        with c_act1:
            st.write("Pull live upcoming fixtures feed directly from football-data.co.uk with bookmaker odds.")
        with c_act2:
            fetch_btn = st.button("🔄 Fetch Live Fixtures", use_container_width=True)

        if fetch_btn or 'live_fixtures_df' in st.session_state:
            if fetch_btn:
                with st.spinner("Fetching live upcoming fixtures and market odds..."):
                    try:
                        import importlib
                        import ml.data_ingestion
                        importlib.reload(ml.data_ingestion)
                        fix_df = ml.data_ingestion.fetch_upcoming_fixtures()
                        if not fix_df.empty:
                            st.session_state['live_fixtures_df'] = fix_df
                            st.success(f"Fetched {len(fix_df)} live upcoming fixtures across European leagues!")
                        else:
                            st.warning("No unplayed upcoming fixtures found in current feed.")
                    except Exception as e:
                        st.error(f"Error fetching live fixtures: {e}")

            fix_df = st.session_state.get('live_fixtures_df', pd.DataFrame())

            if not fix_df.empty:
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                with f_col1:
                    date_window = st.selectbox("📅 Date Window", ["Today & Tomorrow", "Next 3 Days", "Next 7 Days", "All Upcoming"], index=0)
                with f_col2:
                    all_league_keys = ["All Leagues"] + sorted(list(set(list(UNDERSTAT_LEAGUES.keys()) + list(fix_df['league'].dropna().unique()))))
                    sel_league = st.selectbox("Filter League", all_league_keys)
                with f_col3:
                    edge_filter = st.slider("Min Edge %", 1, 15, 5, 1) / 100.0
                with f_col4:
                    val_only = st.checkbox("Show +EV Opportunities Only", value=False)

                filtered_fix = fix_df.copy()

                # Filter by Date Window
                today_dt = pd.Timestamp.now().normalize()
                if date_window == "Today & Tomorrow":
                    max_dt = today_dt + pd.Timedelta(days=1, hours=23, minutes=59)
                    filtered_fix = filtered_fix[filtered_fix['Date'] <= max_dt]
                elif date_window == "Next 3 Days":
                    max_dt = today_dt + pd.Timedelta(days=3)
                    filtered_fix = filtered_fix[filtered_fix['Date'] <= max_dt]
                elif date_window == "Next 7 Days":
                    max_dt = today_dt + pd.Timedelta(days=7)
                    filtered_fix = filtered_fix[filtered_fix['Date'] <= max_dt]

                if sel_league != "All Leagues":
                    filtered_fix = filtered_fix[filtered_fix['league'] == sel_league]

                st.divider()

                # Process and display fixture opportunity cards
                opportunities_found = 0

                for idx, row in filtered_fix.iterrows():
                    h_team = row['HomeTeam']
                    a_team = row['AwayTeam']
                    m_date = row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else 'Upcoming'
                    m_time = row.get('Time', '')
                    o25 = row.get('over25_odds', np.nan)
                    u25 = row.get('under25_odds', np.nan)
                    draw_o = row.get('draw_odds', np.nan)

                    if pd.isna(o25):
                        continue

                    # Compute default or model probability from rolling team stats if available in cache
                    master_cache = st.session_state.get('ml_feat_df', None)
                    model_prob_o25 = 0.55  # baseline estimation if no cache

                    if master_cache is not None and not master_cache.empty:
                        h_hist = master_cache[(master_cache['HomeTeam'] == h_team) | (master_cache['AwayTeam'] == h_team)]
                        a_hist = master_cache[(master_cache['HomeTeam'] == a_team) | (master_cache['AwayTeam'] == a_team)]
                        if not h_hist.empty and not a_hist.empty:
                            h_xg = h_hist['xG_avg_H_H_10'].iloc[-1] if 'xG_avg_H_H_10' in h_hist.columns else 1.6
                            a_xg = a_hist['xG_avg_A_A_10'].iloc[-1] if 'xG_avg_A_A_10' in a_hist.columns else 1.3
                            lam_h = (h_xg + 1.1) / 2.0 if pd.notna(h_xg) else 1.5
                            lam_a = (a_xg + 1.2) / 2.0 if pd.notna(a_xg) else 1.2
                            sm = score_matrix(lam_h, lam_a)
                            model_prob_o25 = sum(sm[i][j] for i in range(7) for j in range(7) if i + j >= 3)
                    
                    model_prob_u25 = 1.0 - model_prob_o25

                    imp_o25 = implied_probability(o25) if pd.notna(o25) and o25 > 1.0 else 0.5
                    imp_u25 = implied_probability(u25) if pd.notna(u25) and u25 > 1.0 else 0.5

                    edge_o25 = model_prob_o25 - imp_o25
                    edge_u25 = model_prob_u25 - imp_u25

                    val_o25 = edge_o25 >= edge_filter
                    val_u25 = edge_u25 >= edge_filter
                    is_val = val_o25 or val_u25

                    if val_o25 and val_u25:
                        best_market = "Over 2.5" if edge_o25 >= edge_u25 else "Under 2.5"
                        best_edge = max(edge_o25, edge_u25)
                        best_odds = o25 if best_market == "Over 2.5" else u25
                        best_prob = model_prob_o25 if best_market == "Over 2.5" else model_prob_u25
                        best_imp = imp_o25 if best_market == "Over 2.5" else imp_u25
                    elif val_o25:
                        best_market = "Over 2.5"
                        best_edge = edge_o25
                        best_odds = o25
                        best_prob = model_prob_o25
                        best_imp = imp_o25
                    elif val_u25:
                        best_market = "Under 2.5"
                        best_edge = edge_u25
                        best_odds = u25
                        best_prob = model_prob_u25
                        best_imp = imp_u25
                    else:
                        best_market = "Over 2.5" if edge_o25 >= edge_u25 else "Under 2.5"
                        best_edge = max(edge_o25, edge_u25)
                        best_odds = o25 if edge_o25 >= edge_u25 else u25
                        best_prob = model_prob_o25 if edge_o25 >= edge_u25 else model_prob_u25
                        best_imp = imp_o25 if edge_o25 >= edge_u25 else imp_u25

                    if val_only and not is_val:
                        continue

                    if is_val:
                        opportunities_found += 1

                    # Fixture Card
                    card_border = "3px solid #00d4aa" if is_val else "1px solid rgba(255,255,255,0.1)"
                    card_bg = "rgba(0, 212, 170, 0.05)" if is_val else "rgba(255,255,255,0.02)"

                    st.markdown(f"""
                    <div style="background: {card_bg}; padding: 15px; margin-bottom: 12px; border-radius: 8px; border: {card_border};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: #888; font-size: 0.85rem;">{row['league']} • {m_date} {m_time}</span>
                                <h4 style="margin: 4px 0;">{h_team} vs {a_team}</h4>
                                <span style="color: #00d4aa; font-weight: 600; font-size: 0.95rem;">Recommended Bet: {best_market} Goals</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 1.2rem; font-weight: bold; color: {'#00d4aa' if is_val else '#ff4b4b'};">
                                    {'✅ +EV OPPORTUNITY (' + best_market + ')' if is_val else '❌ NO VALUE'}
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                    fc1.metric("Bookie Odds", f"{best_odds:.2f}" if pd.notna(best_odds) else "N/A")
                    fc2.metric("Implied Prob", f"{best_imp*100:.1f}%")
                    fc3.metric("Model Prob", f"{best_prob*100:.1f}%")
                    fc4.metric(f"Edge % ({best_market})", f"{best_edge*100:+.1f}%", delta=f"{best_edge*100:+.1f}%" if is_val else None)
                    fc5.metric("Quarter-Kelly Stake", f"£{kelly_stake(best_prob, best_odds, 1000.0):.2f}" if pd.notna(best_odds) and best_odds > 1 else "£0.00")

                    st.write("")

                if filtered_fix.empty:
                    st.info(f"No unplayed upcoming matches currently scheduled today for {sel_league}. Switch to the 'Custom Match Calculator' tab to analyze custom team matchups!")
                elif opportunities_found == 0 and val_only:
                    st.info("No +EV opportunities meeting the minimum edge threshold found in current filter.")

    with sub_tab2:
        st.subheader("🧮 Custom Team Statistics Predictor")

        c1, c2 = st.columns(2)
        with c1:
            home_team = st.text_input("Home Team Name", value="Arsenal")
            home_scored = st.number_input("Home Avg Goals Scored", value=1.8, step=0.1)
            home_conceded = st.number_input("Home Avg Goals Conceded", value=0.9, step=0.1)
            home_xg = st.number_input("Home Avg xG Created", value=1.95, step=0.1)
            home_sot = st.number_input("Home Shots on Target", value=5.5, step=0.5)

        with c2:
            away_team = st.text_input("Away Team Name", value="Liverpool")
            away_scored = st.number_input("Away Avg Goals Scored", value=1.6, step=0.1)
            away_conceded = st.number_input("Away Avg Goals Conceded", value=1.2, step=0.1)
            away_xg = st.number_input("Away Avg xG Created", value=1.70, step=0.1)
            away_sot = st.number_input("Away Shots on Target", value=4.8, step=0.5)

        st.markdown("### Market Odds")
        co1, co2, co3, co4 = st.columns(4)
        odds_o25 = co1.number_input("Live Over 2.5 Odds", value=1.85, step=0.05)
        odds_u25 = co2.number_input("Live Under 2.5 Odds", value=2.00, step=0.05)
        odds_draw = co3.number_input("Live Draw Odds", value=3.40, step=0.05)
        edge_thresh = co4.slider("Minimum Edge Threshold (%)", 1, 15, 5, 1, key="cust_edge") / 100.0

        st.divider()

        # Derived parameters
        lam_home = (home_xg + away_conceded) / 2.0
        lam_away = (away_xg + home_conceded) / 2.0

        sm = score_matrix(lam_home, lam_away, max_goals=6)
        prob_o25 = sum(sm[i][j] for i in range(7) for j in range(7) if i + j >= 3)
        prob_draw_val = sum(sm[i][i] for i in range(7))
        prob_u25 = 1.0 - prob_o25

        imp_o25 = implied_probability(odds_o25)
        imp_u25 = implied_probability(odds_u25)
        imp_draw = implied_probability(odds_draw)

        edge_o25 = prob_o25 - imp_o25
        edge_u25 = prob_u25 - imp_u25
        edge_draw = prob_draw_val - imp_draw

        ev_o25 = expected_value(prob_o25, odds_o25, 10.0)
        k_stake = kelly_stake(prob_o25, odds_o25, 1000.0)

        # Verdict cards
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("### Over 2.5 Goals")
            st.metric("Model Probability", f"{prob_o25*100:.1f}%")
            st.metric("Implied Probability", f"{imp_o25*100:.1f}%")
            st.metric("Edge", f"{edge_o25*100:.1f}%")
            st.metric("Expected Value (£10)", f"£{ev_o25:.2f}")
            st.metric("Quarter-Kelly Stake", f"£{k_stake:.2f}")
            badge = "verdict-positive" if edge_o25 >= edge_thresh else "verdict-negative"
            icon = "✅ +EV BET" if edge_o25 >= edge_thresh else "❌ NO VALUE"
            st.markdown(f'<div class="verdict-badge {badge}">{icon}</div>', unsafe_allow_html=True)

        with v2:
            st.markdown("### Under 2.5 Goals")
            st.metric("Model Probability", f"{prob_u25*100:.1f}%")
            st.metric("Implied Probability", f"{imp_u25*100:.1f}%")
            st.metric("Edge", f"{edge_u25*100:.1f}%")
            st.metric("Expected Value (£10)", f"£{expected_value(prob_u25, odds_u25, 10.0):.2f}")
            badge_u = "verdict-positive" if edge_u25 >= edge_thresh else "verdict-negative"
            icon_u = "✅ +EV BET" if edge_u25 >= edge_thresh else "❌ NO VALUE"
            st.markdown(f'<div class="verdict-badge {badge_u}">{icon_u}</div>', unsafe_allow_html=True)

        with v3:
            st.markdown("### Draw Market")
            st.metric("Model Probability", f"{prob_draw_val*100:.1f}%")
            st.metric("Implied Probability", f"{imp_draw*100:.1f}%")
            st.metric("Edge", f"{edge_draw*100:.1f}%")
            st.metric("Expected Value (£10)", f"£{expected_value(prob_draw_val, odds_draw, 10.0):.2f}")
            badge_d = "verdict-positive" if edge_draw >= edge_thresh else "verdict-negative"
            icon_d = "✅ +EV BET" if edge_draw >= edge_thresh else "❌ NO VALUE"
            st.markdown(f'<div class="verdict-badge {badge_d}">{icon_d}</div>', unsafe_allow_html=True)

        st.divider()

        # Score matrix heatmap
        st.subheader("🎯 Expected Score Heatmap")
        z_data = [[sm[i][j] * 100 for j in range(7)] for i in range(7)]
        text_data = [[f"{sm[i][j]*100:.1f}%" for j in range(7)] for i in range(7)]

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"Away {i}" for i in range(7)],
            y=[f"Home {i}" for i in range(7)],
            text=text_data,
            texttemplate="%{text}",
            colorscale=[[0, "#000000"], [1, "#00d4aa"]],
            showscale=False
        ))
        fig_hm.update_layout(
            title=f"Predicted Score Probability Grid: {home_team} vs {away_team}",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig_hm, use_container_width=True)

