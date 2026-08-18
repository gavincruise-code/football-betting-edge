# -*- coding: utf-8 -*-
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
                            # Fix #5: Clear history cache so form data refreshes on new fetch
                            st.session_state.pop('league_history_cache', None)
                            st.session_state.pop('master_cross_league_df', None)
                            st.session_state.pop('xgb_model_cache', None)
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
                    # Deduplicated league list: config keys + any extra leagues in the live feed
                    _known = list(UNDERSTAT_LEAGUES.keys())
                    _feed_leagues = list(fix_df['league'].dropna().unique())
                    _all_leagues = sorted(set(_known + _feed_leagues))
                    all_league_keys = ["All Leagues"] + _all_leagues
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

                # Fix #4: User bankroll input for correct Kelly stake sizing
                with st.sidebar:
                    st.divider()
                    st.subheader("💰 Bankroll Settings")
                    user_bankroll = st.number_input(
                        "My Current Bankroll (£)",
                        min_value=100.0, max_value=1_000_000.0,
                        value=float(st.session_state.get('user_bankroll', 1000.0)),
                        step=100.0, format="%.0f",
                        help="Quarter-Kelly stake recommendations scale with your actual bankroll.",
                        key="bankroll_input"
                    )
                    st.session_state['user_bankroll'] = user_bankroll

                    st.divider()
                    st.subheader("⚠️ Early-Season Settings")
                    _is_aug_sep = pd.Timestamp.now().month in (8, 9)
                    early_season_mode = st.checkbox(
                        "Early-Season Mode",
                        value=_is_aug_sep,
                        help=(
                            "Automatically enabled Aug–Sep. When on:\n"
                            "• Stake dampener: 50% of normal Kelly stake\n"
                            "• Under 2.5 minimum edge raised to 8%\n"
                            "(Teams lack defensive cohesion early in season)"
                        ),
                        key="early_season_mode"
                    )
                    if early_season_mode:
                        st.caption("🔻 Stakes halved · 📈 Under 2.5 edge ≥ 8%")

                    exclude_uefa_qual = st.checkbox(
                        "Exclude UEFA Qualifiers",
                        value=True,
                        help=(
                            "Removes CL/EL/ECL qualifying rounds.\n"
                            "Inter-league knockout football has high variance\n"
                            "and poor cross-league model calibration."
                        ),
                        key="exclude_uefa_qual"
                    )
                    if exclude_uefa_qual:
                        st.caption("🚫 CL/EL/ECL qualifying excluded")

                filtered_fix = fix_df.copy()
                n_fetched = len(filtered_fix)

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
                n_after_date = len(filtered_fix)

                if sel_league != "All Leagues":
                    filtered_fix = filtered_fix[filtered_fix['league'] == sel_league]

                # UEFA Qualifier filter: knock out CL/EL/ECL qualifying rounds
                _UEFA_QUAL_PATTERNS = [
                    'UEFA CL Qualifying', 'UEFA EL Qualifying', 'UEFA ECL Qualifying',
                    'Champions League Qualifying', 'Europa League Qualifying',
                    'Conference League Qualifying',
                ]
                if exclude_uefa_qual and 'league' in filtered_fix.columns:
                    _qual_mask = filtered_fix['league'].apply(
                        lambda lg: any(p.lower() in str(lg).lower() for p in ['qualifying', 'cl qual', 'el qual', 'ecl qual'])
                    )
                    filtered_fix = filtered_fix[~_qual_mask]
                n_after_league = len(filtered_fix)


                # Count kicked-off (for display only — actual skip happens in loop)
                now_ts = pd.Timestamp.now()
                def _has_real_time(row):
                    t = str(row.get('Time', '')).strip()
                    return bool(t) and t not in ('00:00', 'nan', 'None', '')
                n_kicked_off = sum(
                    1 for _, r in filtered_fix.iterrows()
                    if _has_real_time(r) and (
                        r['Date'].replace(
                            hour=int(str(r.get('Time','00:00')).split(':')[0]),
                            minute=int(str(r.get('Time','00:00')).split(':')[1])
                        ) if ':' in str(r.get('Time','')) else r['Date']
                    ) < now_ts
                )
                n_after_kickoff = n_after_league - n_kicked_off

                # Show breakdown
                with st.expander(f"📊 Fixture Filter Breakdown — {n_fetched} fetched → showing ~{n_after_kickoff}", expanded=False):
                    st.markdown(f"""
| Stage | Count | Dropped |
|---|---|---|
| 📥 Raw feed (all sources) | **{n_fetched}** | — |
| 📅 After date window (*{date_window}*) | **{n_after_date}** | {n_fetched - n_after_date} |
| 🏆 After league filter (*{sel_league}*) | **{n_after_league}** | {n_after_date - n_after_league} |
| ⏰ After kickoff filter (already started) | **{n_after_kickoff}** | {n_kicked_off} |
| 📈 Insufficient team history | *(evaluated per fixture)* | — |
| ✅ +EV filter active | **{"Yes — value bets only" if val_only else "No — showing all"}** | — |
                    """)


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
                # football-data.co.uk uses abbreviated team names that ESPN/Betfair don't.
                # This map translates common→fd names BEFORE fuzzy matching so the
                # difflib cutoff doesn't need to bridge completely different strings.
                TEAM_ALIASES = {
                    # ── Portugal ──────────────────────────────────────────────────────
                    'sporting cp': 'sp lisbon', 'sporting clube de portugal': 'sp lisbon',
                    'vitoria de guimaraes': 'guimaraes', 'vitoria sc': 'guimaraes',
                    'vitoria guimaraes': 'guimaraes',
                    'sporting braga': 'sp braga', 'sc braga': 'sp braga',
                    'cd nacional': 'nacional', 'cd santa clara': 'santa clara',
                    'rio ave fc': 'rio ave', 'fc famalicao': 'famalicao',
                    'fc arouca': 'arouca', 'gil vicente fc': 'gil vicente',
                    'moreirense fc': 'moreirense', 'gd estoril praia': 'estoril',
                    'belenenses': 'b sad', 'casa pia ac': 'casa pia',
                    # ── Spain ─────────────────────────────────────────────────────────
                    'athletic bilbao': 'ath bilbao', 'athletic club': 'ath bilbao',
                    'atletico madrid': 'ath madrid', 'atletico de madrid': 'ath madrid',
                    'real betis': 'betis', 'real betis balompie': 'betis',
                    'real valladolid': 'valladolid', 'rayo vallecano': 'vallecano',
                    'deportivo alaves': 'alaves', 'sd alaves': 'alaves',
                    'cd leganes': 'leganes', 'ud las palmas': 'las palmas',
                    'real sociedad': 'sociedad', 'ca osasuna': 'osasuna',
                    'ud almeria': 'almeria', 'girona fc': 'girona',
                    'getafe cf': 'getafe', 'celta vigo': 'celta',
                    'cadiz cf': 'cadiz', 'elche cf': 'elche',
                    'espanyol': 'espanol', 'rcd espanyol': 'espanol',
                    'granada cf': 'granada',
                    # ── Italy ─────────────────────────────────────────────────────────
                    'ac milan': 'milan', 'inter milan': 'inter',
                    'fc internazionale': 'inter', 'internazionale': 'inter',
                    'ss lazio': 'lazio', 'as roma': 'roma',
                    'acf fiorentina': 'fiorentina', 'ac fiorentina': 'fiorentina',
                    'hellas verona': 'verona', 'udinese calcio': 'udinese',
                    'us sassuolo': 'sassuolo', 'bologna fc': 'bologna',
                    'torino fc': 'torino', 'cagliari calcio': 'cagliari',
                    'spezia calcio': 'spezia', 'venezia fc': 'venezia',
                    'us lecce': 'lecce', 'us salernitana': 'salernitana',
                    'ac monza': 'monza', 'frosinone calcio': 'frosinone',
                    'empoli fc': 'empoli', 'us cremonese': 'cremonese',
                    # ── Germany ───────────────────────────────────────────────────────
                    'bayer leverkusen': 'leverkusen', 'bayer 04 leverkusen': 'leverkusen',
                    'borussia dortmund': 'dortmund', 'borussia m\'gladbach': 'mgladbach',
                    'borussia monchengladbach': 'mgladbach',
                    'rb leipzig': 'leipzig', 'rasenballsport leipzig': 'leipzig',
                    'sc freiburg': 'freiburg', 'vfb stuttgart': 'stuttgart',
                    'tsg hoffenheim': 'hoffenheim', 'tsg 1899 hoffenheim': 'hoffenheim',
                    'fc augsburg': 'augsburg', 'vfl wolfsburg': 'wolfsburg',
                    'vfl bochum': 'bochum', 'eintracht frankfurt': 'ein frankfurt',
                    'fc koln': 'koln', '1. fc koln': 'koln',
                    'hertha bsc': 'hertha', 'sv werder bremen': 'werder bremen',
                    'fc heidenheim': 'heidenheim', 'sv darmstadt 98': 'darmstadt',
                    # ── France ────────────────────────────────────────────────────────
                    'paris saint-germain': 'paris sg', 'psg': 'paris sg',
                    'olympique marseille': 'marseille', 'om': 'marseille',
                    'olympique lyonnais': 'lyon', 'ol': 'lyon',
                    'stade rennais': 'rennes', 'stade brestois': 'brest',
                    'as monaco': 'monaco', 'ogc nice': 'nice',
                    'rc lens': 'lens', 'losc lille': 'lille',
                    'montpellier hsc': 'montpellier', 'stade de reims': 'reims',
                    'clermont foot': 'clermont', 'fc nantes': 'nantes',
                    'toulouse fc': 'toulouse', 'rc strasbourg': 'strasbourg',
                    'havre ac': 'le havre', 'le havre ac': 'le havre',
                    'metz': 'metz', 'angers sco': 'angers',
                    # ── Netherlands ───────────────────────────────────────────────────
                    'psv eindhoven': 'psv', 'ajax amsterdam': 'ajax',
                    'feyenoord rotterdam': 'feyenoord', 'az alkmaar': 'az',
                    'fc utrecht': 'utrecht', 'sc heerenveen': 'heerenveen',
                    'vitesse arnhem': 'vitesse', 'nec nijmegen': 'nec',
                    # ── Turkey ────────────────────────────────────────────────────────
                    'galatasaray sk': 'galatasaray', 'galatasaray a.s.': 'galatasaray',
                    'besiktas jk': 'besiktas', 'besiktas': 'besiktas',
                    'fenerbahce sk': 'fenerbahce', 'trabzonspor ak': 'trabzonspor',
                    'sivasspor': 'sivasspor', 'konyaspor': 'konyaspor',
                    'gaziantep fk': 'gaziantep', 'adana demirspor': 'adana demirspor',
                    'antalyaspor': 'antalyaspor', 'kayserispor': 'kayserispor',
                    'hatayspor': 'hatayspor', 'kasimpasa': 'kasimpasa',
                    'alanyaspor': 'alanyaspor', 'istanbulspor': 'istanbulspor',
                    # ── Greece ────────────────────────────────────────────────────────
                    'panathinaikos fc': 'panathinaikos', 'olympiacos fc': 'olympiakos',
                    'olympiacos': 'olympiakos', 'aek athens': 'aek',
                    'paok fc': 'paok', 'aris thessaloniki': 'aris',
                    # ── England (Championship / League 1 / League 2 short names) ──────
                    'wolves': 'wolverhampton wanderers', 'wolverhampton': 'wolverhampton wanderers',
                    'blackburn': 'blackburn rovers', 'blackpool fc': 'blackpool',
                    'sheffield utd': 'sheffield united', 'sheff utd': 'sheffield united',
                    'sheffield wed': 'sheffield weds', 'sheff wed': 'sheffield weds',
                    'west brom': 'west bromwich', 'wba': 'west bromwich',
                    'norwich': 'norwich city', 'cardiff': 'cardiff city',
                    'hull': 'hull city', 'hull city fc': 'hull city',
                    'swansea': 'swansea city', 'stoke': 'stoke city',
                    'luton': 'luton town', 'ipswich': 'ipswich town',
                    'sunderland afc': 'sunderland', 'middlesbrough fc': 'middlesbrough',
                    'coventry': 'coventry city', 'qpr': 'queens park rangers',
                    'pne': 'preston', 'preston ne': 'preston',
                    'bristol city fc': 'bristol city', 'bristol rovers fc': 'bristol rovers',
                    'barnsley fc': 'barnsley', 'burton albion fc': 'burton',
                    'fleetwood town fc': 'fleetwood', 'oxford united fc': 'oxford',
                    'wigan athletic fc': 'wigan', 'bolton wanderers fc': 'bolton',
                    'plymouth argyle fc': 'plymouth', 'exeter city fc': 'exeter',
                    'portsmouth fc': 'portsmouth', 'charlton athletic fc': 'charlton',
                    # ── Scotland ──────────────────────────────────────────────────────
                    'celtic fc': 'celtic', 'rangers fc': 'rangers',
                    'heart of midlothian': 'hearts', 'hearts fc': 'hearts',
                    'hibernian fc': 'hibs', 'hibernian': 'hibs',
                    'aberdeen fc': 'aberdeen', 'dundee united fc': 'dundee utd',
                    'motherwell fc': 'motherwell', 'st mirren fc': 'st mirren',
                    'livingston fc': 'livingston', 'ross county fc': 'ross county',
                    'kilmarnock fc': 'kilmarnock', 'st johnstone fc': 'st johnstone',
                    # ── English Football: ESPN full display names → football-data.co.uk names ──
                    # ESPN uses full official names; football-data.co.uk uses shortened versions.
                    # Without these aliases every team with City/United/Rovers/Town/etc. fails
                    # the 0.75 fuzzy threshold and returns "Insufficient team history".
                    # Covers all 5 English tiers: EPL, Championship, League 1, League 2, Conference.

                    # EPL
                    'manchester city': 'man city',
                    'manchester united': 'man united',
                    'manchester utd': 'man united',
                    'brighton & hove albion': 'brighton',
                    'brighton and hove albion': 'brighton',
                    "nottingham forest": "nott'm forest",
                    "nottm forest": "nott'm forest",
                    'newcastle united': 'newcastle',
                    'newcastle utd': 'newcastle',
                    'tottenham hotspur': 'tottenham',
                    'spurs': 'tottenham',
                    'afc bournemouth': 'bournemouth',
                    'west ham united': 'west ham',
                    'wolverhampton wanderers': 'wolves',
                    'wolverhampton': 'wolves',
                    'leicester city': 'leicester',
                    'leeds united': 'leeds',
                    'ipswich town': 'ipswich',
                    'luton town': 'luton',
                    'sheffield united': 'sheffield utd',
                    'sheffield wednesday': 'sheffield weds',
                    'birmingham city': 'birmingham',
                    'blackburn rovers': 'blackburn',
                    'west bromwich albion': 'west brom',
                    'queens park rangers': 'qpr',
                    'coventry city': 'coventry',
                    'norwich city': 'norwich',
                    'cardiff city': 'cardiff',
                    'swansea city': 'swansea',
                    'stoke city': 'stoke city',
                    'hull city': 'hull',
                    'sunderland afc': 'sunderland',
                    'middlesbrough fc': 'middlesbrough',
                    'watford fc': 'watford',
                    'crystal palace fc': 'crystal palace',
                    'aston villa fc': 'aston villa',
                    'everton fc': 'everton',
                    'chelsea fc': 'chelsea',
                    'arsenal fc': 'arsenal',
                    'liverpool fc': 'liverpool',
                    'brentford fc': 'brentford',
                    'fulham fc': 'fulham',
                    'burnley fc': 'burnley',

                    # Championship (E1) — additional to EPL above
                    'derby county': 'derby',
                    'lincoln city': 'lincoln',
                    'charlton athletic': 'charlton',
                    'preston north end': 'preston',
                    'millwall fc': 'millwall',
                    'plymouth argyle': 'plymouth',
                    'oxford united': 'oxford',
                    'bristol city fc': 'bristol city',
                    'sheffield utd': 'sheffield utd',

                    # League 1 (E2)
                    'burton albion': 'burton',
                    'cambridge united': 'cambridge',
                    'doncaster rovers': 'doncaster',
                    'peterborough united': 'peterboro',
                    'stockport county': 'stockport',
                    'wycombe wanderers': 'wycombe',
                    'huddersfield town': 'huddersfield',
                    'mansfield town': 'mansfield',
                    'bradford city': 'bradford',
                    'fleetwood town fc': 'fleetwood',
                    'exeter city': 'exeter',
                    'mk dons': 'mk dons',
                    'afc wimbledon': 'afc wimbledon',

                    # League 2 (E3)
                    'accrington stanley': 'accrington',
                    'colchester united': 'colchester',
                    'crewe alexandra': 'crewe',
                    'grimsby town': 'grimsby',
                    'northampton town': 'northampton',
                    'oldham athletic': 'oldham',
                    'rotherham united': 'rotherham',
                    'salford city': 'salford',
                    'swindon town': 'swindon',
                    'tranmere rovers': 'tranmere',
                    'bristol rovers': 'bristol rvs',
                    'cheltenham town': 'cheltenham',
                    'newport county afc': 'newport county',

                    # Conference (EC)
                    'carlisle united': 'carlisle',
                    'fc halifax town': 'halifax',
                    'halifax town': 'halifax',
                    'harrogate town': 'harrogate',
                    'hartlepool united': 'hartlepool',
                    'kidderminster harriers': 'kidderminster',
                    'scunthorpe united': 'scunthorpe',
                    'solihull moors': 'solihull',
                    'southend united': 'southend',
                    'sutton united': 'sutton',
                    'yeovil town': 'yeovil',
                    'forest green rovers': 'forest green',
                    'aldershot town': 'aldershot',
                    'boston united': 'boston utd',
                    'afc fylde': 'fylde',
                    'barrow afc': 'barrow',
                    'york city': 'york',
                    'dagenham and redbridge': 'dag and red',
                    'dag and red': 'dag and red',
                    'ebbsfleet united': 'ebbsfleet',
                    'maidenhead united': 'maidenhead',
                }

                def norm_team(name):
                    if not name: return ""
                    s = str(name).lower().strip()
                    # Check alias map first (exact match on lowercased input)
                    if s in TEAM_ALIASES:
                        s = TEAM_ALIASES[s]
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


                # ── Debug counters — reset before loop ────────────────────────────
                _dbg_no_league  = 0  # dropped: no valid league
                _dbg_kicked_off = 0  # dropped: already kicked off
                _dbg_no_league_list  = []  # (HomeTeam, AwayTeam, raw_league)
                _dbg_kicked_off_list = []  # (HomeTeam, AwayTeam, time_str)

                for idx, row in filtered_fix.iterrows():
                    h_team = row['HomeTeam']
                    a_team = row['AwayTeam']
                    m_date = row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else 'Upcoming'
                    m_time = row.get('Time', '')
                    sort_dt = parse_sort_dt(row)
                    raw_o25 = row.get('over25_odds', np.nan)
                    raw_u25 = row.get('under25_odds', np.nan)

                    # Skip fixtures with no valid league — no team history will be found
                    _row_league = str(row.get('league', '') or '').strip()
                    if not _row_league or _row_league in ('nan', 'Unknown', 'None'):
                        _dbg_no_league += 1
                        _dbg_no_league_list.append((
                            str(h_team), str(a_team),
                            repr(row.get('league', '<missing>')),
                            m_date
                        ))
                        continue

                    # Skip fixtures that have already kicked off.
                    # Only filter when we have a real kickoff time (ESPN fixtures supply HH:MM).
                    # Football-data.co.uk fixtures have no Time column and default to midnight,
                    # so we only exclude when time is explicitly set and non-zero.
                    _time_str = str(m_time).strip()
                    _has_real_time = bool(_time_str) and _time_str not in ('00:00', 'nan', 'None', '')
                    if _has_real_time and sort_dt < pd.Timestamp.now():
                        _dbg_kicked_off += 1
                        _dbg_kicked_off_list.append((
                            str(h_team), str(a_team), _time_str, m_date
                        ))
                        continue

                    # Fix #1: Resolve real odds — NaN when no market exists (suppresses false edges)
                    try:
                        bf_odds = bf_client.fetch_market_odds(h_team, a_team)
                        bf_raw_o = bf_odds.get('over25_odds')
                        bf_raw_u = bf_odds.get('under25_odds')
                        bf_is_real = bf_odds.get('source') != 'no_live_odds'
                        # Prefer Betfair live price, fall back to feed CSV odds, else NaN
                        init_o25 = float(bf_raw_o) if (bf_is_real and pd.notna(bf_raw_o) and float(bf_raw_o) > 1.0) else (float(raw_o25) if pd.notna(raw_o25) and float(raw_o25) > 1.0 else float('nan'))
                        init_u25 = float(bf_raw_u) if (bf_is_real and pd.notna(bf_raw_u) and float(bf_raw_u) > 1.0) else (float(raw_u25) if pd.notna(raw_u25) and float(raw_u25) > 1.0 else float('nan'))
                    except Exception:
                        init_o25 = float(raw_o25) if pd.notna(raw_o25) and float(raw_o25) > 1.0 else float('nan')
                        init_u25 = float(raw_u25) if pd.notna(raw_u25) and float(raw_u25) > 1.0 else float('nan')

                    o25 = init_o25
                    u25 = init_u25

                    # has_live_odds: true if EITHER side of the market is present
                    has_o25 = pd.notna(o25) and o25 > 1.0
                    has_u25 = pd.notna(u25) and u25 > 1.0
                    has_live_odds = has_o25 or has_u25

                    # Derive the missing side from the available one
                    # (assumes a ~103% book, typical for Betfair over/under markets)
                    if has_live_odds and not has_o25 and has_u25:
                        # Only u25 present: imply o25 from it
                        impl_under = 1.0 / u25
                        impl_over  = max(1.03 - impl_under, 0.05)
                        o25 = round(1.0 / impl_over, 3)
                    elif has_live_odds and has_o25 and not has_u25:
                        # Only o25 present: imply u25 from it
                        impl_over  = 1.0 / o25
                        impl_under = max(1.03 - impl_over, 0.05)
                        u25 = round(1.0 / impl_under, 3)

                    # Fix #8: Overround sanity check — only when both sides present
                    # Suppress thin/stale markets with > 15% margin
                    if has_live_odds and pd.notna(o25) and o25 > 1.0 and pd.notna(u25) and u25 > 1.0:
                        overround = (1.0 / o25) + (1.0 / u25)
                        if overround > 1.15:
                            has_live_odds = False  # Market is distorted — treat as no-odds

                    league_name = str(row.get('league', '') or 'Unknown')
                    if scanner_model == "🎯 Auto-Optimal (By League)":
                        try:
                            from ml.league_calibrator import get_strategy_for_league
                            effective_strat = get_strategy_for_league(league_name, _cal_cache)
                        except Exception:
                            # Fallback heuristic if calibrator unavailable
                            _lg_str = str(league_name)
                            effective_strat = st.session_state.get('league_strategy_map', {}).get(
                                _lg_str,
                                "Dixon-Coles Only" if any(w in _lg_str for w in ["La", "EPL", "Premier"]) else "Dual Ensemble"
                            )
                    else:
                        effective_strat = scanner_model

                    
                    if 'league_history_cache' not in st.session_state:
                        st.session_state['league_history_cache'] = {}

                    lg_df = st.session_state['league_history_cache'].get(league_name, None)
                    if lg_df is None:
                        # ── Step 1: Try football-data.co.uk (free, no API key) ──────────
                        try:
                            from data_utils import download_league_data
                            from dotenv import load_dotenv
                            load_dotenv()
                            # Only codes that actually exist on football-data.co.uk
                            code_reverse = {
                                'EPL': 'E0', 'Premier League': 'E0', 'Championship': 'E1',
                                'League 1': 'E2', 'League 2': 'E3', 'Conference': 'EC',
                                'La_Liga': 'SP1', 'La Liga': 'SP1', 'Segunda Division': 'SP2',
                                'Bundesliga': 'D1', 'Bundesliga 2': 'D2',
                                'Serie_A': 'I1', 'Serie A': 'I1', 'Serie B': 'I2',
                                'Ligue_1': 'F1', 'Ligue 1': 'F1', 'Ligue 2': 'F2',
                                'Scottish Premiership': 'SC0', 'Scottish Championship': 'SC1',
                                'Scottish League 1': 'SC2', 'Scottish League 2': 'SC3',
                                'Netherlands': 'N1', 'Eredivisie': 'N1',
                                'Belgium': 'B1', 'Pro League': 'B1', 'Jupiler Pro League': 'B1',
                                'Portugal': 'P1', 'Liga Portugal': 'P1',
                                'Turkey': 'T1', 'Super Lig': 'T1',
                                'Greece': 'G1', 'Super League': 'G1',
                            }
                            l_code = code_reverse.get(league_name)
                            lg_df = download_league_data(l_code) if l_code else pd.DataFrame()
                        except Exception:
                            lg_df = pd.DataFrame()

                        # ── Step 2: Fall back to API-Football for everything else ────────
                        if lg_df.empty:
                            try:
                                from ml.api_football_client import fetch_league_history, get_league_id
                                if get_league_id(league_name) is not None:
                                    lg_df = fetch_league_history(league_name, min_matches=50)
                            except Exception:
                                lg_df = pd.DataFrame()

                        st.session_state['league_history_cache'][league_name] = lg_df


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

                    # CRITICAL: Only use the league's own data.
                    # The former cross-league fallback was silently running Chinese/Japanese teams
                    # against European data and producing completely spurious model outputs.
                    # If lg_df is empty → the fixture will correctly show "Insufficient Data".
                    search_df = lg_df if (lg_df is not None and not lg_df.empty) else pd.DataFrame()

                    model_prob_o25 = np.nan
                    has_data = False
                    # Fix #2: Initialise these before the data block to avoid NameError on no-data fixtures
                    val_o25 = False
                    val_u25 = False
                    is_val = False
                    has_no_odds = False  # Fix: must reset each iteration — else leaks from previous fixture

                    if search_df is not None and not search_df.empty:
                        import difflib
                        from ml.feature_engine import (
                            compute_rolling_goal_features,
                            compute_rolling_xg_features,
                            compute_rolling_shot_features,
                        )
                        try:
                            from ml.config import ROLLING_WINDOWS, CONGESTION_THRESHOLD_DAYS
                        except Exception:
                            ROLLING_WINDOWS = [10, 20, 38]
                            CONGESTION_THRESHOLD_DAYS = 4

                        # Fix #2: Sort by date so .tail() always returns the truly most-recent matches
                        search_df = search_df.sort_values('Date').reset_index(drop=True)

                        def match_team_exact(target_name, team_list):
                            target_norm = norm_team(target_name)
                            for t in team_list:
                                if norm_team(t) == target_norm:
                                    return t, False  # exact match

                            # Raised cutoff to 0.75 — 0.6 allows "nantes" → "angers" (both 6 chars,
                            # share 4 letters). Also require the first 3 normalised chars to match
                            # as a sanity check against short-name collisions.
                            close = difflib.get_close_matches(
                                target_norm, [norm_team(t) for t in team_list], n=1, cutoff=0.75
                            )
                            if close:
                                # Sanity: first 3 chars of matched name must equal first 3 of target
                                if target_norm[:3] == close[0][:3]:
                                    for t in team_list:
                                        if norm_team(t) == close[0]:
                                            return t, True  # fuzzy match — caller can warn user
                            return target_name, True  # no match found

                        if 'HomeTeam' in search_df.columns:
                            all_lg_teams = list(set(search_df['HomeTeam'].dropna().unique()).union(
                                                set(search_df['AwayTeam'].dropna().unique())))
                            matched_h, h_fuzzy = match_team_exact(h_team, all_lg_teams)
                            matched_a, a_fuzzy = match_team_exact(a_team, all_lg_teams)

                            # Fix #1 & #7: Venue-split histories — home team's HOME games, away team's AWAY games
                            h_home_hist = search_df[search_df['HomeTeam'] == matched_h]
                            h_all_hist  = search_df[(search_df['HomeTeam'] == matched_h) | (search_df['AwayTeam'] == matched_h)]
                            a_away_hist = search_df[search_df['AwayTeam'] == matched_a]
                            a_all_hist  = search_df[(search_df['HomeTeam'] == matched_a) | (search_df['AwayTeam'] == matched_a)]
                        else:
                            h_home_hist = h_all_hist = a_away_hist = a_all_hist = pd.DataFrame()

                        MIN_HIST = 5
                        if len(h_all_hist) >= MIN_HIST and len(a_all_hist) >= MIN_HIST:
                            has_data = True

                            # Fix #5: Use 20-match window for better signal vs noise ratio
                            WINDOW = 20

                            # Fix #1 & #7: λ from venue-split histories
                            # Home team: goals scored AT HOME; Away team: goals conceded AWAY
                            def _ven_sc_cc(hist, is_home):
                                """Extract scored/conceded arrays from venue-specific history."""
                                if hist.empty:
                                    return np.array([]), np.array([])
                                s = hist['FTHG' if is_home else 'FTAG'].dropna().values.astype(float)
                                c = hist['FTAG' if is_home else 'FTHG'].dropna().values.astype(float)
                                return s, c

                            h_sc, h_cc = _ven_sc_cc(h_home_hist.tail(WINDOW), True)
                            a_sc, a_cc = _ven_sc_cc(a_away_hist.tail(WINDOW), False)

                            # Fallback to mixed-venue if venue-split history too thin
                            if len(h_sc) < MIN_HIST:
                                _t = h_all_hist.tail(WINDOW)
                                h_sc = np.where(_t['HomeTeam'] == matched_h, _t['FTHG'], _t['FTAG']).astype(float)
                                h_cc = np.where(_t['HomeTeam'] == matched_h, _t['FTAG'], _t['FTHG']).astype(float)
                            if len(a_sc) < MIN_HIST:
                                _t = a_all_hist.tail(WINDOW)
                                a_sc = np.where(_t['AwayTeam'] == matched_a, _t['FTAG'], _t['FTHG']).astype(float)
                                a_cc = np.where(_t['AwayTeam'] == matched_a, _t['FTHG'], _t['FTAG']).astype(float)

                            lam_h = (float(np.nanmean(h_sc)) + float(np.nanmean(a_cc))) / 2.0
                            lam_a = (float(np.nanmean(a_sc)) + float(np.nanmean(h_cc))) / 2.0
                            if pd.isna(lam_h) or lam_h <= 0: lam_h = 1.35
                            if pd.isna(lam_a) or lam_a <= 0: lam_a = 1.15

                            sm = score_matrix(lam_h, lam_a)
                            p_dc = sum(sm[i][j] for i in range(7) for j in range(7) if i + j >= 3)

                            # Fix #3: Build the full ~120-feature set that XGBoost was trained on
                            feats = {}
                            for _w in ROLLING_WINDOWS:
                                # Home team — home venue
                                feats.update(compute_rolling_goal_features(h_home_hist, matched_h, 'H_H', _w))
                                feats.update(compute_rolling_xg_features(h_home_hist,  matched_h, 'H_H', _w))
                                feats.update(compute_rolling_shot_features(h_home_hist, matched_h, 'H_H', _w))
                                # Home team — all venues
                                feats.update(compute_rolling_goal_features(h_all_hist,  matched_h, 'H_All', _w))
                                feats.update(compute_rolling_xg_features(h_all_hist,   matched_h, 'H_All', _w))
                                feats.update(compute_rolling_shot_features(h_all_hist,  matched_h, 'H_All', _w))
                                # Away team — away venue
                                feats.update(compute_rolling_goal_features(a_away_hist, matched_a, 'A_A', _w))
                                feats.update(compute_rolling_xg_features(a_away_hist,  matched_a, 'A_A', _w))
                                feats.update(compute_rolling_shot_features(a_away_hist, matched_a, 'A_A', _w))
                                # Away team — all venues
                                feats.update(compute_rolling_goal_features(a_all_hist,  matched_a, 'A_All', _w))
                                feats.update(compute_rolling_xg_features(a_all_hist,   matched_a, 'A_All', _w))
                                feats.update(compute_rolling_shot_features(a_all_hist,  matched_a, 'A_All', _w))

                            # Trend features: short-term vs season-long trajectory
                            for _tp in ['H_H', 'H_All', 'A_A', 'A_All']:
                                _g10 = feats.get(f"goals_scored_avg_{_tp}_10", np.nan)
                                _g38 = feats.get(f"goals_scored_avg_{_tp}_38", np.nan)
                                feats[f"goals_trend_{_tp}"] = _g10 - _g38 if (pd.notna(_g10) and pd.notna(_g38)) else np.nan
                                _x10 = feats.get(f"xG_avg_{_tp}_10", np.nan)
                                _x38 = feats.get(f"xG_avg_{_tp}_38", np.nan)
                                feats[f"xG_trend_{_tp}"] = _x10 - _x38 if (pd.notna(_x10) and pd.notna(_x38)) else np.nan

                            # Head-to-head features (last 5 meetings)
                            _h2h = search_df[
                                ((search_df['HomeTeam'] == matched_h) & (search_df['AwayTeam'] == matched_a)) |
                                ((search_df['HomeTeam'] == matched_a) & (search_df['AwayTeam'] == matched_h))
                            ].tail(5)
                            if _h2h.empty:
                                feats.update({'h2h_over25_rate_5': np.nan, 'h2h_avg_goals_5': np.nan, 'h2h_matches_available': 0.0})
                            else:
                                _tot = _h2h['FTHG'] + _h2h['FTAG']
                                feats.update({'h2h_over25_rate_5': float(np.mean(_tot > 2.5)),
                                              'h2h_avg_goals_5':   float(np.mean(_tot)),
                                              'h2h_matches_available': float(len(_h2h))})

                            # Contextual features: rest days, congestion, month
                            _today = pd.Timestamp.now().normalize()
                            _h_last = h_all_hist.iloc[-1]['Date'] if not h_all_hist.empty else _today
                            _a_last = a_all_hist.iloc[-1]['Date'] if not a_all_hist.empty else _today
                            _days_h = max(0, (_today - _h_last).days)
                            _days_a = max(0, (_today - _a_last).days)
                            feats['days_since_last_home'] = float(_days_h)
                            feats['days_since_last_away'] = float(_days_a)
                            feats['is_congested_home'] = float(_days_h < CONGESTION_THRESHOLD_DAYS)
                            feats['is_congested_away'] = float(_days_a < CONGESTION_THRESHOLD_DAYS)
                            feats['month'] = float(_today.month)

                            # Market features
                            _ip_o = (1.0 / o25) if (has_live_odds and pd.notna(o25) and o25 > 1.0) else np.nan
                            _ip_u = (1.0 / u25) if (has_live_odds and pd.notna(u25) and u25 > 1.0) else np.nan
                            feats['implied_prob_over25'] = _ip_o
                            feats['implied_prob_under25'] = _ip_u
                            feats['overround'] = (_ip_o + _ip_u - 1.0) if (pd.notna(_ip_o) and pd.notna(_ip_u)) else np.nan
                            feats['odds_ratio_over25'] = (o25 / u25) if (has_live_odds and pd.notna(u25) and u25 > 0) else np.nan

                            # DC-derived features (some models may have been trained with these)
                            feats['dc_prob'] = p_dc
                            feats['lam_home'] = lam_h
                            feats['lam_away'] = lam_a

                            # ── XGBoost prediction using full feature set ──────────────────────
                            p_xgb = None
                            try:
                                from ml.ml_model import load_model, predict_proba as ml_predict_proba
                                from ml.league_calibrator import get_xgb_model_name_for_league
                                _xgb_name = get_xgb_model_name_for_league(league_name, _cal_cache)
                                if 'xgb_model_cache' not in st.session_state:
                                    st.session_state['xgb_model_cache'] = {}
                                if _xgb_name not in st.session_state['xgb_model_cache']:
                                    _m, _c = load_model(_xgb_name)
                                    if _m is None and _xgb_name != 'xgb_over25_latest':
                                        _m, _c = load_model('xgb_over25_latest')
                                    st.session_state['xgb_model_cache'][_xgb_name] = (_m, _c)
                                _xgb_model, _xgb_cal = st.session_state['xgb_model_cache'][_xgb_name]
                                if _xgb_model is not None:
                                    feat_df_row = pd.DataFrame([feats])
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

                            # Fallback: momentum proxy if no trained model available
                            if p_xgb is None:
                                _ha = h_all_hist.tail(10)
                                _aa = a_all_hist.tail(10)
                                _h_all_sc = np.where(_ha['HomeTeam'] == matched_h, _ha['FTHG'], _ha['FTAG']).astype(float) if not _ha.empty else np.array([1.35])
                                _a_all_sc = np.where(_aa['AwayTeam'] == matched_a, _aa['FTAG'], _aa['FTHG']).astype(float) if not _aa.empty else np.array([1.15])
                                _h_mom = (float(np.nanmean(_h_all_sc[-5:])) - float(np.nanmean(_h_all_sc))) if len(_h_all_sc) >= 5 else 0
                                _a_mom = (float(np.nanmean(_a_all_sc[-5:])) - float(np.nanmean(_a_all_sc))) if len(_a_all_sc) >= 5 else 0
                                p_xgb = float(np.clip(p_dc + 0.08 * (_h_mom + _a_mom), 0.15, 0.85))

                            if effective_strat == "Dixon-Coles Only":
                                model_prob_o25 = p_dc
                            elif effective_strat == "XGBoost ML Only":
                                model_prob_o25 = p_xgb
                            else:  # Dual Ensemble
                                model_prob_o25 = 0.5 * p_dc + 0.5 * p_xgb

                    
                    # Fix #1: Only compute edge when real live odds exist
                    if has_data and pd.notna(model_prob_o25) and has_live_odds:
                        model_prob_u25 = 1.0 - model_prob_o25

                        # FIX H4: Proportional devigging — remove bookmaker overround before
                        # computing edge. Raw 1/odds inflates implied probability by ~2.5–4%.
                        # With both sides available: P_fair = (1/odds_side) / (1/odds_O + 1/odds_U)
                        # This gives the true market consensus probability without the margin.
                        if pd.notna(o25) and o25 > 1.0 and pd.notna(u25) and u25 > 1.0:
                            _raw_o = 1.0 / o25
                            _raw_u = 1.0 / u25
                            _total = _raw_o + _raw_u  # overround (e.g. 1.05 = 5% margin)
                            imp_o25 = _raw_o / _total  # fair implied probability Over 2.5
                            imp_u25 = _raw_u / _total  # fair implied probability Under 2.5
                        else:
                            # Only one side available — fall back to raw implied probability
                            imp_o25 = (1.0 / o25) if pd.notna(o25) and o25 > 1.0 else np.nan
                            imp_u25 = (1.0 / u25) if pd.notna(u25) and u25 > 1.0 else np.nan

                        edge_o25 = (model_prob_o25 - imp_o25) if pd.notna(imp_o25) else np.nan
                        edge_u25 = (model_prob_u25 - imp_u25) if pd.notna(imp_u25) else np.nan

                        val_o25 = pd.notna(edge_o25) and edge_o25 >= edge_filter
                        # Early-season mode: raise Under 2.5 minimum edge to 8%
                        # to combat poor defensive cohesion in Aug/Sep gameweeks
                        _u25_edge_threshold = max(edge_filter, 0.08) if early_season_mode else edge_filter
                        val_u25 = pd.notna(edge_u25) and edge_u25 >= _u25_edge_threshold
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

                        # Fix #4: Use user-specified bankroll for Kelly stake sizing
                        rec_k_stake = kelly_stake(best_prob, best_odds, user_bankroll) if pd.notna(best_odds) and best_odds > 1 else 10.0
                        # Early-season dampener: halve stakes in Aug/Sep due to high
                        # parameter uncertainty (new squads, summer transfers, no cohesion)
                        if early_season_mode:
                            rec_k_stake = round(rec_k_stake * 0.50, 2)
                    else:
                        model_prob_u25 = 1.0 - model_prob_o25 if pd.notna(model_prob_o25) else np.nan
                        edge_o25 = np.nan
                        edge_u25 = np.nan
                        is_val = False
                        best_market = "Over 2.5"
                        best_edge = np.nan
                        best_odds = o25 if has_live_odds else np.nan
                        best_prob = model_prob_o25
                        best_imp = implied_probability(o25) if has_live_odds and pd.notna(o25) else np.nan
                        rec_k_stake = 0.0
                        # Distinguish: has model data but no live odds vs truly no team data
                        has_no_odds = has_data and not has_live_odds

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
                        'has_data': has_data,
                        'has_live_odds': has_live_odds,
                        'has_no_odds': locals().get('has_no_odds', False),
                        # Fix #6: Fuzzy match metadata for user warning in card
                        'matched_h': locals().get('matched_h', h_team),
                        'matched_a': locals().get('matched_a', a_team),
                        'h_fuzzy':   locals().get('h_fuzzy', False),
                        'a_fuzzy':   locals().get('a_fuzzy', False),
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

                # ── FIX H5: Portfolio Kelly — cap simultaneous kickoff-window exposure ────
                # Without this, 10 x £50 bets at 3pm Saturday = £500 at risk simultaneously
                # (50% of a £1,000 bankroll). Kelly is designed for sequential bets, not
                # concurrent ones — simultaneous bets need portfolio-level normalisation.
                #
                # Algorithm:
                #   1. Group +EV bets by their kickoff time slot (rounded to nearest 30 min)
                #   2. For each slot, sum raw Kelly stakes
                #   3. If total > MAX_MATCHDAY_EXPOSURE * bankroll, scale all down proportionally
                _MAX_WINDOW_EXPOSURE = 0.15   # max 15% of bankroll per kickoff window
                _max_window_stake    = user_bankroll * _MAX_WINDOW_EXPOSURE

                # Group +EV items by kickoff slot
                from collections import defaultdict
                _slot_groups = defaultdict(list)
                for _item in evaluated_fixtures:
                    if _item['is_val'] and _item['rec_k_stake'] > 0:
                        # Round kickoff to nearest 30-minute slot for grouping
                        try:
                            _dt = _item['sort_dt']
                            _slot = _dt.replace(minute=(_dt.minute // 30) * 30, second=0, microsecond=0)
                        except Exception:
                            _slot = 'unknown'
                        _slot_groups[_slot].append(_item)

                for _slot, _slot_items in _slot_groups.items():
                    _total_raw = sum(i['rec_k_stake'] for i in _slot_items)
                    if _total_raw > _max_window_stake and _total_raw > 0:
                        _scale = _max_window_stake / _total_raw
                        for _item in _slot_items:
                            _item['rec_k_stake'] = round(_item['rec_k_stake'] * _scale, 2)
                            _item['portfolio_scaled'] = True   # flag for UI badge
                # ─────────────────────────────────────────────────────────────────────────

                opportunities_found = sum(1 for item in evaluated_fixtures if item['is_val'])
                _dbg_no_data    = sum(1 for item in evaluated_fixtures if not item['has_data'])
                _dbg_no_odds    = sum(1 for item in evaluated_fixtures if item.get('has_no_odds', False))
                _dbg_no_ev      = sum(1 for item in evaluated_fixtures if item['has_data'] and not item['is_val'] and not item.get('has_no_odds', False))
                _dbg_val_only_skip = len(evaluated_fixtures) - opportunities_found if val_only else 0
                n_in_feed = len(filtered_fix)

                with st.expander(
                    f"📊 Fixture Breakdown — {n_in_feed} in window → {len(evaluated_fixtures)} evaluated → {opportunities_found} +EV",
                    expanded=False
                ):
                    st.markdown(f"""
| Stage | Count | Notes |
|---|---|---|
| 🗂️ In date/league window | **{n_in_feed}** | After date & league filters |
| 🚫 Dropped: no valid league | **{_dbg_no_league}** | League field blank or NaN in feed |
| ⏰ Dropped: already kicked off | **{_dbg_kicked_off}** | Kickoff time has passed (BST) |
| ✅ Evaluated by model | **{len(evaluated_fixtures)}** | Passed into scanner loop |
| — &nbsp; of which: Insufficient team history | **{_dbg_no_data}** | < 5 matches in historical data |
| — &nbsp; of which: Awaiting live odds | **{_dbg_no_odds}** | No Betfair price found |
| — &nbsp; of which: No edge detected | **{_dbg_no_ev}** | Model edge below {edge_filter*100:.0f}% threshold |
| — &nbsp; of which: **+EV Opportunities** | **{opportunities_found}** | Edge ≥ {edge_filter*100:.0f}% with live odds |
                    """)

                    if _dbg_no_league_list:
                        st.markdown("**🚫 Fixtures dropped — no valid league:**")
                        no_lg_df = pd.DataFrame(
                            _dbg_no_league_list,
                            columns=["Home", "Away", "Raw league value", "Date"]
                        )
                        st.dataframe(no_lg_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No fixtures dropped for missing league.")

                    if _dbg_kicked_off_list:
                        with st.expander(f"⏰ {_dbg_kicked_off} fixtures already kicked off (click to expand)", expanded=False):
                            ko_df = pd.DataFrame(
                                _dbg_kicked_off_list,
                                columns=["Home", "Away", "KO Time (BST)", "Date"]
                            )
                            st.dataframe(ko_df, use_container_width=True, hide_index=True)

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
                    has_no_odds = item.get('has_no_odds', False)
                    best_market = item['best_market']
                    best_edge = item['best_edge']
                    best_odds = item['best_odds']
                    best_prob = item['best_prob']
                    best_imp = item['best_imp']
                    rec_k_stake = item['rec_k_stake']
                    matched_h = item.get('matched_h', h_team)
                    matched_a = item.get('matched_a', a_team)
                    h_fuzzy   = item.get('h_fuzzy', False)
                    a_fuzzy   = item.get('a_fuzzy', False)

                    # Fix #6: Build fuzzy-match warning string shown under team name
                    fuzzy_parts = []
                    if h_fuzzy and matched_h != h_team:
                        fuzzy_parts.append(f"{h_team} → <i>{matched_h}</i>")
                    if a_fuzzy and matched_a != a_team:
                        fuzzy_parts.append(f"{a_team} → <i>{matched_a}</i>")
                    fuzzy_warning = (f'<div style="color:#e8a500;font-size:0.78rem;margin-top:2px;">⚠️ Fuzzy name match: {" &amp; ".join(fuzzy_parts)}</div>' if fuzzy_parts else "")

                    if not has_data:
                        card_border = "1px dashed #ffbb00"
                        card_bg = "rgba(255, 187, 0, 0.04)"
                        status_badge = '<div style="font-size: 1.05rem; font-weight: bold; color: #ffbb00;">&#x26A0;&#xFE0F; INSUFFICIENT DATA</div>'
                        sub_label = '<div style="color: #ffbb00; font-weight: 600; font-size: 0.88rem; margin-top: 4px;">&#x26A0;&#xFE0F; Not possible to make a model judgement (insufficient team history)</div>'
                    elif has_no_odds:
                        card_border = "1px dashed #4ea8de"
                        card_bg = "rgba(78, 168, 222, 0.04)"
                        status_badge = '<div style="font-size: 1.05rem; font-weight: bold; color: #4ea8de;">&#x1F535; NO LIVE ODDS</div>'
                        sub_label = (f'<div style="color: #4ea8de; font-weight: 600; font-size: 0.88rem; margin-top: 4px;">Model probability {best_market}: {best_prob*100:.1f}% &#8212; Awaiting live market odds</div>' if pd.notna(best_prob) else '<div style="color: #4ea8de; margin-top:4px;">Awaiting live market odds</div>')
                    elif is_val:
                        card_border = "3px solid #00d4aa"
                        card_bg = "rgba(0, 212, 170, 0.05)"
                        status_badge = f'<div style="font-size: 1.2rem; font-weight: bold; color: #00d4aa;">&#x2705; +EV OPPORTUNITY ({best_market})</div>'
                        sub_label = f'<div style="color: #00d4aa; font-weight: 600; font-size: 0.95rem; margin-top: 4px;">Recommended Bet: {best_market} Goals</div>'
                    else:
                        card_border = "1px solid rgba(255,255,255,0.1)"
                        card_bg = "rgba(255,255,255,0.02)"
                        status_badge = '<div style="font-size: 1.2rem; font-weight: bold; color: #ff4b4b;">&#x274C; NO VALUE</div>'
                        sub_label = f'<div style="color: #888; font-weight: 600; font-size: 0.95rem; margin-top: 4px;">Evaluated Market: {best_market} Goals</div>'

                    # Build the full HTML card as a single string first to avoid
                    # Python-Markdown escaping inline elements after block elements (h4)
                    _league_safe = str(row['league']).replace('&', '&amp;').replace('<', '&lt;')
                    _ht_safe = str(h_team).replace('&', '&amp;').replace('<', '&lt;')
                    _at_safe = str(a_team).replace('&', '&amp;').replace('<', '&lt;')
                    _strat_safe = str(item.get('effective_strat', scanner_model)).replace('&', '&amp;').replace('<', '&lt;')
                    _card_html = (
                        f'<div style="background:{card_bg};padding:15px;margin-bottom:12px;border-radius:8px;border:{card_border}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                        f'<div style="flex:1">'
                        f'<div style="color:#888;font-size:0.85rem">{_league_safe} &bull; <b>{m_date} {m_time}</b> &bull; Strategy: <b>{_strat_safe}</b></div>'
                        f'<div style="font-size:1.1rem;font-weight:700;margin:6px 0">{_ht_safe} vs {_at_safe}</div>'
                        f'{fuzzy_warning}'
                        f'{sub_label}'
                        f'</div>'
                        f'<div style="text-align:right;padding-left:12px;flex-shrink:0">'
                        f'{status_badge}'
                        f'</div>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(_card_html, unsafe_allow_html=True)

                    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([1, 1, 1, 1.1, 1.3, 1.2])
                    fc1.metric("Betfair Odds", f"{best_odds:.2f}" if pd.notna(best_odds) else "No Live Odds")
                    fc2.metric("Implied Prob", f"{best_imp*100:.1f}%" if pd.notna(best_imp) else "—")
                    fc3.metric("Model Prob", f"{best_prob*100:.1f}%" if (has_data and pd.notna(best_prob)) else "N/A (No Data)")
                    fc4.metric(f"Edge % ({best_market})", f"{best_edge*100:+.1f}%" if (has_data and pd.notna(best_edge) and not has_no_odds) else "—", delta=f"{best_edge*100:+.1f}%" if (has_data and is_val) else None)
                    _portfolio_scaled = item.get('portfolio_scaled', False)
                    _stake_delta = None
                    _stake_delta_color = "normal"
                    if has_data and is_val:
                        if early_season_mode and _portfolio_scaled:
                            _stake_delta = "⚠️ 50% dampened · 🔀 portfolio scaled"
                            _stake_delta_color = "off"
                        elif early_season_mode:
                            _stake_delta = "⚠️ 50% dampened"
                            _stake_delta_color = "off"
                        elif _portfolio_scaled:
                            _stake_delta = "🔀 portfolio scaled"
                            _stake_delta_color = "off"
                    fc5.metric(
                        "Quarter-Kelly Stake",
                        f"£{rec_k_stake:.2f}" if (has_data and is_val) else "—",
                        delta=_stake_delta,
                        delta_color=_stake_delta_color
                    )

                    with fc6:
                        if not has_data:
                            st.info("⚠️ Data Unavailable")
                        elif has_no_odds:
                            st.info("🔵 Awaiting Odds")
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

