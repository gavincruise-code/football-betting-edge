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
        "Switzerland": "SWZ", "Austria": "AUT",
        "India": "ind.1", "India (Calcutta Premier Division)": "ind.2", "India (Super League)": "ind.1",
        "Calcutta Premier Division": "ind.2", "Indian Super League": "ind.1"
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
                with st.spinner("Fetching live upcoming fixtures and Betfair Exchange odds..."):
                    try:
                        import importlib
                        import ml.data_ingestion
                        importlib.reload(ml.data_ingestion)
                        fix_df = ml.data_ingestion.fetch_upcoming_fixtures()

                        # Refresh Betfair Exchange live market odds batch
                        import ml.betfair_api
                        importlib.reload(ml.betfair_api)
                        from ml.betfair_api import get_betfair_client
                        bf_client = get_betfair_client()
                        bf_client.fetch_all_live_markets()

                        if not fix_df.empty:
                            st.session_state['live_fixtures_df'] = fix_df
                            st.success(f"Fetched {len(fix_df)} live upcoming fixtures with live Betfair Exchange odds!")
                        else:
                            st.warning("No unplayed upcoming fixtures found in current feed.")
                    except Exception as e:
                        st.error(f"Error fetching live fixtures: {e}")

            fix_df = st.session_state.get('live_fixtures_df', pd.DataFrame())

            if not fix_df.empty:
                f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([1.2, 1.2, 1.5, 1.2, 1])
                with f_col1:
                    date_window = st.selectbox("📅 Date Window", ["Today Only", "Today & Tomorrow", "Next 3 Days", "Next 7 Days", "All Upcoming"], index=0)
                with f_col2:
                    all_league_keys = ["All Leagues"] + sorted(list(set(list(UNDERSTAT_LEAGUES.keys()) + list(fix_df['league'].dropna().unique()))))
                    sel_league = st.selectbox("Filter League", all_league_keys)
                with f_col3:
                    scanner_model = st.selectbox("🤖 Model Strategy", ["🎯 Auto-Optimal (By League)", "Dual Ensemble (Recommended)", "Dixon-Coles Only", "XGBoost ML Only"], index=0)

                # Initialize League Strategy Mapping in session_state if not present
                if 'league_strategy_map' not in st.session_state:
                    st.session_state['league_strategy_map'] = {
                        'La_Liga': 'Dixon-Coles Only',
                        'La Liga': 'Dixon-Coles Only',
                        'EPL': 'Dixon-Coles Only',
                        'Premier League': 'Dixon-Coles Only',
                        'Bundesliga': 'Dual Ensemble',
                        'Serie_A': 'Dual Ensemble',
                        'Ligue_1': 'Dual Ensemble'
                    }
                with f_col4:
                    edge_filter = st.slider("Min Edge %", 1, 15, 5, 1) / 100.0
                with f_col5:
                    val_only = st.checkbox("Show +EV Only", value=False)

                filtered_fix = fix_df.copy()

                # Filter by Date Window
                today_dt = pd.Timestamp.now().normalize()
                if date_window == "Today Only":
                    max_dt = today_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    filtered_fix = filtered_fix[(filtered_fix['Date'] >= today_dt) & (filtered_fix['Date'] <= max_dt)]
                elif date_window == "Today & Tomorrow":
                    max_dt = today_dt + pd.Timedelta(days=1, hours=23, minutes=59, seconds=59)
                    filtered_fix = filtered_fix[filtered_fix['Date'] <= max_dt]
                elif date_window == "Next 3 Days":
                    max_dt = today_dt + pd.Timedelta(days=3)
                    filtered_fix = filtered_fix[filtered_fix['Date'] <= max_dt]
                elif date_window == "Next 7 Days":
                    max_dt = today_dt + pd.Timedelta(days=7)
                    filtered_fix = filtered_fix[filtered_fix['Date'] <= max_dt]

                if sel_league != "All Leagues":
                    filtered_fix = filtered_fix[filtered_fix['league'] == sel_league]

                # ── Auto-Optimal League Strategy Expander ──────────────────────────────
                try:
                    from ml.league_calibrator import (
                        load_calibration_cache, calibrate_all_leagues,
                        get_strategy_for_league, cache_age_hours,
                        CALIBRATION_LEAGUES, get_xgb_model_name_for_league,
                    )
                    _cal_cache = load_calibration_cache()
                except Exception:
                    _cal_cache = {}

                _age_h = cache_age_hours(_cal_cache)
                _age_label = (
                    f"last calibrated {_age_h:.0f}h ago"
                    if _age_h is not None
                    else "⚠️ not calibrated yet"
                )

                with st.expander(
                    f"⚙️ Auto-Optimal League Strategy Mappings — {_age_label}",
                    expanded=(scanner_model == "🎯 Auto-Optimal (By League)"),
                ):
                    hdr_col, btn_col = st.columns([3, 1])
                    with hdr_col:
                        if _age_h is None:
                            st.warning(
                                "No calibration data found. Click **Re-Calibrate** to run "
                                "walk-forward backtests across all leagues and set optimal strategies."
                            )
                        elif _age_h > 168:   # > 7 days
                            st.warning(
                                f"⚠️ Calibration is {_age_h/24:.0f} days old — consider re-running."
                            )
                        else:
                            st.success(
                                f"✅ Calibration is current ({_age_label}). "
                                "Strategies below are driven by live backtest results."
                            )

                    with btn_col:
                        if st.button("🔄 Re-Calibrate All Leagues", key="recalibrate_btn", type="primary"):
                            progress_bar = st.progress(0, text="Starting calibration…")
                            def _cb(lg, i, tot):
                                progress_bar.progress(
                                    int((i / tot) * 100),
                                    text=f"Backtesting {lg} ({i+1}/{tot})…"
                                )
                            with st.spinner("Running walk-forward backtests across all leagues…"):
                                try:
                                    _cal_cache = calibrate_all_leagues(progress_cb=_cb)
                                    progress_bar.progress(100, text="Done!")
                                    st.success(
                                        f"✅ Calibrated {len(_cal_cache.get('leagues', {}))} leagues! "
                                        "Auto-Optimal strategies have been updated."
                                    )
                                    st.rerun()
                                except Exception as cal_err:
                                    st.error(f"Calibration error: {cal_err}")

                    # ── Per-league results table ────────────────────────────────────────
                    _leagues_data = _cal_cache.get("leagues", {})
                    if _leagues_data:
                        rows = []
                        for lg, info in sorted(_leagues_data.items()):
                            rows.append({
                                "League":        lg,
                                "Strategy":      info.get("strategy", "—"),
                                "Backtest ROI":  f"{info.get('backtest_roi', 0):+.1f}%",
                                "P/L (£)":       f"£{info.get('backtest_pl', 0):+.0f}",
                                "Bets":          info.get("backtest_bets", 0),
                                "Avg Goals":     f"{info.get('avg_goals', 0):.2f}",
                                "Draw %":        f"{info.get('draw_rate_pct', 0):.0f}%",
                                "Min Edge":      f"{info.get('recommended_min_edge', 0.05)*100:.0f}%",
                            })
                        st.dataframe(
                            pd.DataFrame(rows).set_index("League"),
                            use_container_width=True,
                        )
                    else:
                        # Fallback: show the customisable per-league dropdowns
                        st.markdown("**Customize League Model Mappings (manual override):**")
                        lg_map = st.session_state.get('league_strategy_map', {})
                        unique_lgs = sorted(list(set(fix_df['league'].dropna().unique())))
                        mc1, mc2, mc3 = st.columns(3)
                        for idx_lg, lg_k in enumerate(unique_lgs):
                            c_target = [mc1, mc2, mc3][idx_lg % 3]
                            cur_strat = lg_map.get(
                                lg_k,
                                "Dixon-Coles Only"
                                if any(w in lg_k for w in ["La", "EPL", "Premier"])
                                else "Dual Ensemble",
                            )
                            new_strat = c_target.selectbox(
                                f"**{lg_k}**",
                                ["Dixon-Coles Only", "Dual Ensemble", "XGBoost ML Only"],
                                index=(
                                    ["Dixon-Coles Only", "Dual Ensemble", "XGBoost ML Only"].index(cur_strat)
                                    if cur_strat in ["Dixon-Coles Only", "Dual Ensemble", "XGBoost ML Only"]
                                    else 0
                                ),
                                key=f"lg_strat_{lg_k}",
                            )
                            st.session_state['league_strategy_map'][lg_k] = new_strat


                # Placed Bets Tracker, P/L Graph & CSV Export Expander
                from ml.bet_tracker import get_placed_bets, record_bet, is_bet_recorded, update_bet_result, auto_settle_bets
                placed_df = get_placed_bets()
                
                with st.expander(f"📈 Placed Bets Performance & P/L Growth Graph ({len(placed_df)} Bets Logged)", expanded=False):
                    if not placed_df.empty:
                        # Compute Summary Performance Metrics
                        settled_df = placed_df[placed_df['Result'].isin(['WIN', 'LOSS'])]
                        total_bets_count = len(placed_df)
                        settled_count = len(settled_df)
                        wins_count = sum(settled_df['Result'] == 'WIN')
                        losses_count = sum(settled_df['Result'] == 'LOSS')
                        win_rate_pct = (wins_count / settled_count * 100) if settled_count > 0 else 0.0
                        tot_pl = placed_df['Profit_Loss_£'].sum()
                        total_staked = settled_df['Recommended_Stake_£'].sum() if 'Recommended_Stake_£' in settled_df.columns else settled_count * 10.0
                        roi_pct = (tot_pl / total_staked * 100) if total_staked > 0 else 0.0

                        # Auto-settle button
                        as_col1, as_col2 = st.columns([2, 1])
                        with as_col1:
                            st.caption("🤖 Auto-settle checks completed match scores from the data feed and updates PENDING bets to WIN or LOSS automatically.")
                        with as_col2:
                            if st.button("🤖 Auto-Settle Results", key="auto_settle_btn", type="primary"):
                                with st.spinner("Checking match results across all leagues..."):
                                    settle_report = auto_settle_bets()
                                placed_df = get_placed_bets()
                                if settle_report['settled'] > 0:
                                    st.success(f"✅ Auto-settled {settle_report['settled']} bet(s)! {settle_report['not_found']} still pending / result not yet available.")
                                    st.rerun()
                                else:
                                    st.info(f"No new results found yet. {settle_report['not_found']} bet(s) still pending, {settle_report['already']} already settled.")

                        st.divider()

                        # Summary Metrics Cards
                        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                        pm1.metric("Total Bets Logged", f"{total_bets_count}")
                        pm2.metric("Settled Bets (W/L)", f"{wins_count}W / {losses_count}L")
                        pm3.metric("Win Rate", f"{win_rate_pct:.1f}%")
                        pm4.metric("Total P/L (£)", f"£{tot_pl:+.2f}", delta=f"£{tot_pl:+.2f}")
                        pm5.metric("ROI %", f"{roi_pct:+.1f}%", delta=f"{roi_pct:+.1f}%")

                        # Plotly Interactive Cumulative Profit & Loss Growth Line Chart
                        fig_pl = go.Figure()
                        
                        bet_indices = list(range(1, len(placed_df) + 1))
                        cum_pl_series = placed_df['Cumulative_PL_£'].values

                        line_color = "#00d4aa" if tot_pl >= 0 else "#ff4b4b"
                        
                        fig_pl.add_trace(go.Scatter(
                            x=bet_indices,
                            y=cum_pl_series,
                            mode='lines+markers',
                            name='Cumulative P/L (£)',
                            line=dict(color=line_color, width=3),
                            marker=dict(size=6, color=line_color),
                            text=[f"Bet #{i}: {row['Home_Team']} vs {row['Away_Team']} ({row['Market']})<br>Result: {row['Result']} | P/L: £{row['Profit_Loss_£']:+.2f} | Cum: £{row['Cumulative_PL_£']:+.2f}" for i, (_, row) in enumerate(placed_df.iterrows(), 1)],
                            hoverinfo='text'
                        ))

                        fig_pl.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Break Even")

                        fig_pl.update_layout(
                            title="<b>Live Opportunity Scanner — Bankroll P/L Growth Curve (£)</b>",
                            xaxis_title="Bet Number",
                            yaxis_title="Profit / Loss (£)",
                            template="plotly_dark",
                            margin=dict(l=40, r=40, t=50, b=40),
                            height=380,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif")
                        )

                        st.plotly_chart(fig_pl, width="stretch")

                        # Interactive Result Status Editor
                        st.subheader("📋 Logged Bets Table & Outcome Setter")
                        st.caption("Click 'Set WIN' or 'Set LOSS' to update outcomes and refresh your P/L graph!")

                        for idx, b_row in placed_df.iterrows():
                            bc1, bc2, bc3, bc4, bc5, bc6 = st.columns([1.5, 2, 1.2, 1.2, 1.2, 1.5])
                            bc1.write(f"**{b_row['Match_Date']}** ({b_row['League']})")
                            bc2.write(f"{b_row['Home_Team']} vs {b_row['Away_Team']}")
                            bc3.write(f"**{b_row['Market']}** @ {b_row['Odds']:.2f}")
                            bc4.write(f"Stake: £{b_row.get('Recommended_Stake_£', 10.0):.2f}")
                            
                            res_val = str(b_row.get('Result', 'PENDING')).upper()
                            if res_val == 'WIN':
                                bc5.markdown(f"<span style='color: #00d4aa; font-weight: bold;'>✅ WIN (+£{b_row['Profit_Loss_£']:.2f})</span>", unsafe_allow_html=True)
                            elif res_val == 'LOSS':
                                bc5.markdown(f"<span style='color: #ff4b4b; font-weight: bold;'>❌ LOSS (-£{abs(b_row['Profit_Loss_£']):.2f})</span>", unsafe_allow_html=True)
                            else:
                                bc5.markdown("<span style='color: #ffbb00; font-weight: bold;'>⏳ PENDING</span>", unsafe_allow_html=True)

                            with bc6:
                                btn_w, btn_l = st.columns(2)
                                if btn_w.button("✅ Win", key=f"set_win_{idx}"):
                                    update_bet_result(idx, 'WIN')
                                    st.toast(f"Updated Bet #{idx+1} to WIN!")
                                    st.rerun()
                                if btn_l.button("❌ Loss", key=f"set_loss_{idx}"):
                                    update_bet_result(idx, 'LOSS')
                                    st.toast(f"Updated Bet #{idx+1} to LOSS!")
                                    st.rerun()
                            st.divider()

                        csv_data = placed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Placed Bets CSV",
                            data=csv_data,
                            file_name="placed_bets_log.csv",
                            mime="text/csv",
                            key="dl_placed_bets_csv"
                        )
                    else:
                        st.info("No bets recorded yet. Click '📌 Record Bet' on any fixture card below to log a bet and build your P/L graph!")

                st.divider()

                # Evaluate all fixtures first for sorting
                evaluated_fixtures = []

                import unicodedata
                def norm_team(name):
                    if not name: return ""
                    s = str(name).lower()
                    s = s.replace('ø', 'o').replace('æ', 'ae').replace('å', 'a').replace('ß', 'ss')
                    s = s.replace('ü', 'u').replace('ö', 'o').replace('ä', 'a').replace('é', 'e').replace('è', 'e').replace('à', 'a').replace('ç', 'c').replace('ñ', 'n')
                    n = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
                    return n.replace('ifk ', '').replace('fc ', '').replace('sk ', '').replace('ac ', '').replace('cd ', '').strip()

                def parse_sort_dt(r):
                    d = r['Date'] if pd.notnull(r['Date']) else pd.Timestamp.now()
                    t_str = str(r.get('Time', '00:00')).strip()
                    try:
                        if ':' in t_str:
                            parts = t_str.split(':')
                            return d.replace(hour=int(parts[0]), minute=int(parts[1]))
                    except Exception:
                        pass
                    return d

                # Betfair Exchange API Client Initialization
                from ml.betfair_api import get_betfair_client
                bf_client = get_betfair_client()

                with st.sidebar:
                    st.divider()
                    st.subheader("🟡 Betfair Exchange API")
                    try:
                        if bf_client.session_token:
                            st.success(f"🟢 Connected to Betfair API\nUser: {bf_client.username}")
                            if not bf_client.market_cache:
                                bf_client.fetch_all_live_markets()
                        else:
                            st.warning(f"🟡 Betfair Status: {bf_client.last_status}")
                            if "BETTING_RESTRICTED_LOCATION" in str(bf_client.last_status):
                                st.error(
                                    "🚫 **Betfair Geo-Block Active on Cloud Server**\n\n"
                                    "Betfair blocks API connections originating from cloud datacenters (Fly.io / AWS / Render).\n\n"
                                    "💡 **Solutions**:\n"
                                    "1. **Local Access (Recommended)**: Run the app locally via `run_app.bat` on UK home broadband — live Betfair odds connect 100% natively.\n"
                                    "2. **Cloud Fallback**: The app automatically uses market consensus bookmaker odds so models and scanning work seamlessly anywhere!"
                                )
                            with st.expander("🔑 Live Betfair Login & Credentials", expanded=False):
                                u_in = st.text_input("Betfair Username", value=bf_client.username, key="bf_u")
                                p_in = st.text_input("Betfair Password", value=bf_client.password, type="password", key="bf_p")
                                k_in = st.text_input("Betfair App Key", value=bf_client.app_key, key="bf_k")
                                if st.button("🔓 Connect Betfair API"):
                                    bf_client.set_credentials(username=u_in, password=p_in, app_key=k_in)
                                    bf_client.login()
                                    st.rerun()
                    except Exception as e:
                        st.info("🟡 Using Betfair Live Odds Market Mode")


                for idx, row in filtered_fix.iterrows():
                    h_team = row['HomeTeam']
                    a_team = row['AwayTeam']
                    m_date = row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else 'Upcoming'
                    m_time = row.get('Time', '')
                    sort_dt = parse_sort_dt(row)
                    raw_o25 = row.get('over25_odds', np.nan)
                    raw_u25 = row.get('under25_odds', np.nan)

                    try:
                        bf_odds = bf_client.fetch_market_odds(h_team, a_team)
                        bf_raw_o = bf_odds.get('over25_odds')
                        bf_raw_u = bf_odds.get('under25_odds')
                        init_o25 = float(bf_raw_o) if pd.notna(bf_raw_o) and float(bf_raw_o) > 1.0 else (float(raw_o25) if pd.notna(raw_o25) and float(raw_o25) > 1.0 else 2.00)
                        init_u25 = float(bf_raw_u) if pd.notna(bf_raw_u) and float(bf_raw_u) > 1.0 else (float(raw_u25) if pd.notna(raw_u25) and float(raw_u25) > 1.0 else 1.80)
                    except Exception:
                        init_o25 = float(raw_o25) if pd.notna(raw_o25) and float(raw_o25) > 1.0 else 2.00
                        init_u25 = float(raw_u25) if pd.notna(raw_u25) and float(raw_u25) > 1.0 else 1.80

                    o25 = init_o25
                    u25 = init_u25

                    league_name = row.get('league', 'Unknown')
                    if scanner_model == "🎯 Auto-Optimal (By League)":
                        try:
                            from ml.league_calibrator import get_strategy_for_league
                            effective_strat = get_strategy_for_league(league_name, _cal_cache)
                        except Exception:
                            # Fallback heuristic if calibrator unavailable
                            effective_strat = st.session_state.get('league_strategy_map', {}).get(
                                league_name,
                                "Dixon-Coles Only" if any(w in league_name for w in ["La", "EPL", "Premier"]) else "Dual Ensemble"
                            )
                    else:
                        effective_strat = scanner_model

                    
                    if 'league_history_cache' not in st.session_state:
                        st.session_state['league_history_cache'] = {}

                    lg_df = st.session_state['league_history_cache'].get(league_name, None)
                    if lg_df is None:
                        try:
                            from data_utils import download_league_data
                            code_reverse = {
                                'EPL': 'E0', 'Premier League': 'E0', 'Championship': 'E1', 'League 1': 'E2', 'League 2': 'E3',
                                'La_Liga': 'SP1', 'La Liga': 'SP1', 'Segunda Division': 'SP2',
                                'Bundesliga': 'D1', 'Bundesliga 2': 'D2',
                                'Serie_A': 'I1', 'Serie A': 'I1', 'Serie B': 'I2',
                                'Ligue_1': 'F1', 'Ligue 1': 'F1', 'Ligue 2': 'F2',
                                'Scottish Premiership': 'SC0', 'Scottish Championship': 'SC1',
                                'Netherlands': 'N1', 'Eredivisie': 'N1',
                                'Belgium': 'B1', 'Pro League': 'B1', 'Jupiler Pro League': 'B1',
                                'Portugal': 'P1', 'Liga Portugal': 'P1',
                                'Turkey': 'T1', 'Super Lig': 'T1', '1. Lig': 'T2', 'Turkish 1. Lig': 'T2',
                                'Greece': 'G1', 'Super League': 'G1',
                                'USA (MLS)': 'USA', 'USA': 'USA',
                                'Argentina': 'ARG', 'Brazil': 'BRA', 'Mexico': 'MEX',
                                'Japan': 'JPN', 'China': 'CHN', 'Sweden': 'SWE', 'Norway': 'NOR',
                                'Denmark': 'DNK', 'Finland': 'FIN', 'Poland': 'POL', 'Romania': 'ROU',
                                'Switzerland': 'SWZ', 'Austria': 'AUT', 'Ireland': 'IRL', 'Russia': 'RUS'
                            }
                            l_code = code_reverse.get(league_name, None)
                            if l_code:
                                lg_df = download_league_data(l_code)
                            else:
                                lg_df = pd.DataFrame()
                            st.session_state['league_history_cache'][league_name] = lg_df
                        except Exception:
                            lg_df = pd.DataFrame()

                    # Build or load Master Cross-League Dataset for UEFA & Inter-League Team Lookups
                    if 'master_cross_league_df' not in st.session_state or st.session_state['master_cross_league_df'].empty:
                        try:
                            from data_utils import download_league_data
                            top_codes = ['E0', 'SP1', 'D1', 'I1', 'F1', 'N1', 'B1', 'P1', 'T1', 'G1', 'NOR', 'SWE', 'AUT', 'DNK', 'SC0']
                            master_dfs = []
                            for tc in top_codes:
                                try:
                                    tdf = st.session_state['league_history_cache'].get(tc)
                                    if tdf is None or tdf.empty:
                                        tdf = download_league_data(tc)
                                        st.session_state['league_history_cache'][tc] = tdf
                                    if not tdf.empty:
                                        master_dfs.append(tdf)
                                except Exception:
                                    pass
                            if master_dfs:
                                st.session_state['master_cross_league_df'] = pd.concat(master_dfs, ignore_index=True)
                            else:
                                st.session_state['master_cross_league_df'] = pd.DataFrame()
                        except Exception:
                            st.session_state['master_cross_league_df'] = pd.DataFrame()

                    # Fallback to Master Dataset if league_df is empty or fixture is UEFA / International
                    search_df = lg_df if (lg_df is not None and not lg_df.empty and 'UEFA' not in league_name) else st.session_state.get('master_cross_league_df', pd.DataFrame())

                    model_prob_o25 = np.nan
                    has_data = False

                    if search_df is not None and not search_df.empty:
                        import difflib
                        def match_team_exact(target_name, team_list):
                            target_norm = norm_team(target_name)
                            for t in team_list:
                                if norm_team(t) == target_norm:
                                    return t
                            matches = difflib.get_close_matches(target_norm, [norm_team(t) for t in team_list], n=1, cutoff=0.35)
                            if matches:
                                matched_norm = matches[0]
                                for t in team_list:
                                    if norm_team(t) == matched_norm:
                                        return t
                            return target_name

                        if 'HomeTeam' in search_df.columns:
                            all_lg_teams = list(set(search_df['HomeTeam'].dropna().unique()).union(set(search_df['AwayTeam'].dropna().unique())))
                            matched_h = match_team_exact(h_team, all_lg_teams)
                            matched_a = match_team_exact(a_team, all_lg_teams)

                            h_matches = search_df[(search_df['HomeTeam'] == matched_h) | (search_df['AwayTeam'] == matched_h)]
                            a_matches = search_df[(search_df['HomeTeam'] == matched_a) | (search_df['AwayTeam'] == matched_a)]
                        else:
                            h_matches = pd.DataFrame()
                            a_matches = pd.DataFrame()

                        if not h_matches.empty and not a_matches.empty:
                            has_data = True
                            h_recent = h_matches.tail(10)
                            a_recent = a_matches.tail(10)

                            h_scored = np.where(h_recent['HomeTeam'] == matched_h, h_recent['FTHG'], h_recent['FTAG'])
                            h_conceded = np.where(h_recent['HomeTeam'] == matched_h, h_recent['FTAG'], h_recent['FTHG'])
                            a_scored = np.where(a_recent['AwayTeam'] == matched_a, a_recent['FTAG'], a_recent['FTHG'])
                            a_conceded = np.where(a_recent['AwayTeam'] == matched_a, a_recent['FTHG'], a_recent['FTAG'])

                            lam_h = (float(np.nanmean(h_scored)) + float(np.nanmean(a_conceded))) / 2.0
                            lam_a = (float(np.nanmean(a_scored)) + float(np.nanmean(h_conceded))) / 2.0

                            if pd.isna(lam_h) or lam_h <= 0: lam_h = 1.35
                            if pd.isna(lam_a) or lam_a <= 0: lam_a = 1.15

                            sm = score_matrix(lam_h, lam_a)
                            p_dc = sum(sm[i][j] for i in range(7) for j in range(7) if i + j >= 3)

                            # ── Real per-league XGBoost prediction ─────────────────────────
                            # Build a lightweight feature row for this fixture and run the
                            # trained league-specific model. Falls back to the momentum proxy
                            # if no model has been trained yet (calibration not run).
                            p_xgb = None
                            try:
                                from ml.ml_model import load_model, predict_proba as ml_predict_proba
                                from ml.league_calibrator import get_xgb_model_name_for_league
                                _xgb_name = get_xgb_model_name_for_league(league_name, _cal_cache)
                                _xgb_model, _xgb_cal = load_model(_xgb_name)
                                if _xgb_model is None and _xgb_name != 'xgb_over25_latest':
                                    _xgb_model, _xgb_cal = load_model('xgb_over25_latest')
                                if _xgb_model is not None:
                                    # Build minimal feature DataFrame for this fixture
                                    h_scored_arr = np.array(h_scored, dtype=float)
                                    a_scored_arr = np.array(a_scored, dtype=float)
                                    h_conceded_arr = np.array(h_conceded, dtype=float)
                                    a_conceded_arr = np.array(a_conceded, dtype=float)
                                    h_total = h_scored_arr + h_conceded_arr
                                    a_total = a_scored_arr + a_conceded_arr
                                    feat_row = {
                                        'dc_prob': p_dc,
                                        'goals_scored_avg_H_10': float(np.nanmean(h_scored_arr)),
                                        'goals_conceded_avg_H_10': float(np.nanmean(h_conceded_arr)),
                                        'total_goals_avg_H_10': float(np.nanmean(h_total)),
                                        'over25_rate_H_10': float(np.nanmean(h_total > 2.5)),
                                        'goals_scored_avg_A_10': float(np.nanmean(a_scored_arr)),
                                        'goals_conceded_avg_A_10': float(np.nanmean(a_conceded_arr)),
                                        'total_goals_avg_A_10': float(np.nanmean(a_total)),
                                        'over25_rate_A_10': float(np.nanmean(a_total > 2.5)),
                                        'lam_home': lam_h,
                                        'lam_away': lam_a,
                                        'implied_prob_over25': implied_probability(o25),
                                    }
                                    feat_df_row = pd.DataFrame([feat_row])
                                    # Align columns to what the model was trained on
                                    import joblib
                                    _model_feats = getattr(_xgb_model, 'feature_names_in_', None)
                                    if _model_feats is not None:
                                        for _fc in _model_feats:
                                            if _fc not in feat_df_row.columns:
                                                feat_df_row[_fc] = 0.0
                                        feat_df_row = feat_df_row[[_fc for _fc in _model_feats if _fc in feat_df_row.columns]]
                                    _probs = ml_predict_proba(_xgb_model, feat_df_row, _xgb_cal)
                                    p_xgb = float(_probs[0])
                            except Exception:
                                pass

                            # Fallback: momentum-adjusted proxy if no model loaded
                            if p_xgb is None:
                                h_mom = float(np.nanmean(h_scored[-5:])) - float(np.nanmean(h_scored)) if len(h_scored) >= 5 else 0
                                a_mom = float(np.nanmean(a_scored[-5:])) - float(np.nanmean(a_scored)) if len(a_scored) >= 5 else 0
                                p_xgb = float(np.clip(p_dc + 0.08 * (h_mom + a_mom), 0.15, 0.85))

                            if scanner_model == "🎯 Auto-Optimal (By League)":
                                effective_strat = st.session_state.get('league_strategy_map', {}).get(league_name, "Dixon-Coles Only" if any(w in league_name for w in ["La", "EPL", "Premier"]) else "Dual Ensemble")
                            else:
                                effective_strat = scanner_model

                            if effective_strat == "Dixon-Coles Only":
                                model_prob_o25 = p_dc
                            elif effective_strat == "XGBoost ML Only":
                                model_prob_o25 = p_xgb
                            else:  # Dual Ensemble
                                model_prob_o25 = 0.5 * p_dc + 0.5 * p_xgb
                    
                    if has_data and pd.notna(model_prob_o25):
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

                        rec_k_stake = kelly_stake(best_prob, best_odds, 1000.0) if pd.notna(best_odds) and best_odds > 1 else 10.0
                    else:
                        model_prob_o25 = np.nan
                        model_prob_u25 = np.nan
                        edge_o25 = 0.0
                        edge_u25 = 0.0
                        is_val = False
                        best_market = "Over 2.5"
                        best_edge = -0.99
                        best_odds = o25
                        best_prob = np.nan
                        best_imp = implied_probability(o25) if pd.notna(o25) and o25 > 1.0 else 0.5
                        rec_k_stake = 0.0

                    if val_only and not is_val:
                        continue

                    evaluated_fixtures.append({
                        'idx': idx,
                        'row': row,
                        'h_team': h_team,
                        'a_team': a_team,
                        'm_date': m_date,
                        'm_time': m_time,
                        'sort_dt': sort_dt,
                        'o25': o25,
                        'u25': u25,
                        'model_prob_o25': model_prob_o25,
                        'model_prob_u25': model_prob_u25,
                        'edge_o25': edge_o25,
                        'edge_u25': edge_u25,
                        'val_o25': val_o25,
                        'val_u25': val_u25,
                        'is_val': is_val,
                        'best_market': best_market,
                        'best_edge': best_edge,
                        'best_odds': best_odds,
                        'best_prob': best_prob,
                        'best_imp': best_imp,
                        'rec_k_stake': rec_k_stake,
                        'init_o25': init_o25,
                        'init_u25': init_u25,
                        'effective_strat': effective_strat if scanner_model != "🎯 Auto-Optimal (By League)" else f"🎯 Auto ({effective_strat})",
                        'has_data': has_data
                    })

                # SORT FIXTURES: +EV matches FIRST (True < False -> not x['is_val']), then CHRONOLOGICAL by kickoff time (sort_dt)
                evaluated_fixtures.sort(key=lambda x: (not x['is_val'], x['sort_dt'], -x['best_edge']))

                opportunities_found = sum(1 for item in evaluated_fixtures if item['is_val'])

                # Render Sorted Fixture Cards
                for item in evaluated_fixtures:
                    idx = item['idx']
                    row = item['row']
                    h_team = item['h_team']
                    a_team = item['a_team']
                    m_date = item['m_date']
                    m_time = item['m_time']
                    is_val = item['is_val']
                    has_data = item.get('has_data', True)
                    best_market = item['best_market']
                    best_edge = item['best_edge']
                    best_odds = item['best_odds']
                    best_prob = item['best_prob']
                    best_imp = item['best_imp']
                    rec_k_stake = item['rec_k_stake']
                    
                    if not has_data:
                        card_border = "1px dashed #ffbb00"
                        card_bg = "rgba(255, 187, 0, 0.04)"
                        status_badge = '<span style="font-size: 1.05rem; font-weight: bold; color: #ffbb00;">⚠️ INSUFFICIENT DATA</span>'
                        sub_label = '<span style="color: #ffbb00; font-weight: 600; font-size: 0.88rem;">⚠️ Not possible to make a model judgement (insufficient team history)</span>'
                    elif is_val:
                        card_border = "3px solid #00d4aa"
                        card_bg = "rgba(0, 212, 170, 0.05)"
                        status_badge = f'<span style="font-size: 1.2rem; font-weight: bold; color: #00d4aa;">✅ +EV OPPORTUNITY ({best_market})</span>'
                        sub_label = f'<span style="color: #00d4aa; font-weight: 600; font-size: 0.95rem;">Recommended Bet: {best_market} Goals</span>'
                    else:
                        card_border = "1px solid rgba(255,255,255,0.1)"
                        card_bg = "rgba(255,255,255,0.02)"
                        status_badge = '<span style="font-size: 1.2rem; font-weight: bold; color: #ff4b4b;">❌ NO VALUE</span>'
                        sub_label = f'<span style="color: #888; font-weight: 600; font-size: 0.95rem;">Evaluated Market: {best_market} Goals</span>'

                    st.markdown(f"""
                    <div style="background: {card_bg}; padding: 15px; margin-bottom: 12px; border-radius: 8px; border: {card_border};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: #888; font-size: 0.85rem;">{row['league']} • <b>{m_date} {m_time}</b> • Strategy: <b>{item.get('effective_strat', scanner_model)}</b></span>
                                <h4 style="margin: 4px 0;">{h_team} vs {a_team}</h4>
                                {sub_label}
                            </div>
                            <div style="text-align: right;">
                                {status_badge}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([1, 1, 1, 1.1, 1.3, 1.2])
                    fc1.metric("Betfair Odds", f"{best_odds:.2f}" if pd.notna(best_odds) else "N/A")
                    fc2.metric("Implied Prob", f"{best_imp*100:.1f}%" if pd.notna(best_imp) else "N/A")
                    fc3.metric("Model Prob", f"{best_prob*100:.1f}%" if (has_data and pd.notna(best_prob)) else "N/A (No Data)")
                    fc4.metric(f"Edge % ({best_market})", f"{best_edge*100:+.1f}%" if (has_data and pd.notna(best_edge)) else "N/A", delta=f"{best_edge*100:+.1f}%" if (has_data and is_val) else None)
                    fc5.metric("Quarter-Kelly Stake", f"£{rec_k_stake:.2f}" if (has_data and is_val) else "N/A")

                    with fc6:
                        if not has_data:
                            st.info("⚠️ Data Unavailable")
                        else:
                            already_logged = is_bet_recorded(h_team, a_team, best_market)
                            if already_logged:
                                st.success("📌 Bet Logged")
                            else:
                                if st.button("📌 Record Bet", key=f"rec_btn_{idx}"):
                                    record_bet(
                                        date=m_date,
                                        league=row.get('league', 'Unknown'),
                                        home_team=h_team,
                                        away_team=a_team,
                                        market=best_market,
                                        odds=best_odds if pd.notna(best_odds) else 2.00,
                                        strategy=item.get('effective_strat', scanner_model),
                                        model_prob=best_prob,
                                        implied_prob=best_imp,
                                        edge_pct=best_edge,
                                        recommended_stake=rec_k_stake
                                    )
                                    st.toast(f"✅ Bet Recorded: {h_team} vs {a_team} ({best_market})")
                                    st.rerun()

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

