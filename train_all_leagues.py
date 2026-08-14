# -*- coding: utf-8 -*-
"""
train_all_leagues.py
====================
Trains a separate XGBoost Over-2.5 model for every supported league and
updates the calibration cache with the model name.

Run from the betting-dashboard directory:
    py -3 train_all_leagues.py

Each league needs at least MIN_MATCHES finished matches in its history.
Leagues with too little data are skipped (global fallback model used instead).
"""

import os
import sys
import json
import logging
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
MIN_MATCHES   = 150   # skip a league if fewer than this many finished matches
MIN_TRAIN     = 100   # minimum rows for training split
MODELS_DIR    = Path("models")
CACHE_FILE    = Path("league_strategy_cache.json")
MODELS_DIR.mkdir(exist_ok=True)

# Football-data.co.uk code map (source A, free)
FDCO_CODES = {
    "EPL": "E0", "Premier League": "E0", "Championship": "E1",
    "League 1": "E2", "League 2": "E3", "Conference": "EC",
    "Scottish Premiership": "SC0", "Scottish Championship": "SC1",
    "Scottish League 1": "SC2", "Scottish League 2": "SC3",
    "La_Liga": "SP1", "La Liga": "SP1", "Segunda Division": "SP2",
    "Bundesliga": "D1", "Bundesliga 2": "D2",
    "Serie_A": "I1", "Serie A": "I1", "Serie B": "I2",
    "Ligue_1": "F1", "Ligue 1": "F1", "Ligue 2": "F2",
    "Eredivisie": "N1", "Netherlands": "N1",
    "Belgium": "B1", "Jupiler Pro League": "B1",
    "Portugal": "P1", "Liga Portugal": "P1",
    "Turkey": "T1", "Super Lig": "T1",
    "Greece": "G1", "Super League Greece": "G1",
}

# API-Football IDs (source B)
APIF_IDS = {
    # Asia
    "Japan": 98, "China": 169, "South Korea": 292,
    "India": 323, "Indian Super League": 323, "Saudi Arabia": 307,
    "Thailand": 296,
    # Americas
    "USA (MLS)": 253, "Brazil": 71, "Argentina": 128, "Mexico": 262,
    "Colombia": 239, "Chile": 265, "Uruguay": 268, "Ecuador": 240, "Peru": 281,
    # Africa
    "Egypt": 233, "Morocco": 200, "South Africa": 288,
    # Europe extras
    "Sweden": 113, "Norway": 103, "Denmark": 119, "Finland": 244,
    "Poland": 106, "Romania": 283, "Switzerland": 207, "Austria": 218,
    "Croatia": 210, "Russia": 235, "Serbia": 286, "Czech Republic": 345,
    "Ukraine": 333, "Slovakia": 332, "Hungary": 271, "Israel": 384,
    # Oceania
    "Australia": 188,
}

# ── data loading ─────────────────────────────────────────────────────────────

def load_fdco(league_code: str) -> pd.DataFrame:
    from data_utils import download_league_data
    try:
        df = download_league_data(league_code)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        log.warning(f"  FDCO fetch failed ({league_code}): {e}")
        return pd.DataFrame()


def load_apif(league_name: str) -> pd.DataFrame:
    from ml.api_football_client import fetch_league_history
    try:
        df = fetch_league_history(league_name, min_matches=MIN_MATCHES)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        log.warning(f"  API-Football fetch failed ({league_name}): {e}")
        return pd.DataFrame()


def get_league_data(league_name: str) -> pd.DataFrame:
    """Return normalised historical match DataFrame for a league."""
    df = pd.DataFrame()

    # Try football-data.co.uk first
    code = FDCO_CODES.get(league_name)
    if code:
        df = load_fdco(code)

    # Fall back to API-Football
    if df.empty and league_name in APIF_IDS:
        df = load_apif(league_name)

    if df.empty:
        return pd.DataFrame()

    # Normalise columns to minimum needed
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure integer goal columns
    for col in ("FTHG", "FTAG"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Ensure over25 label
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["over25"] = (df["total_goals"] > 2.5).astype(int)

    # Add league column for feature engine
    df["league"] = league_name
    return df


# ── feature engineering ───────────────────────────────────────────────────────

def build_feature_rows(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Build one feature row per completed match using only data available
    BEFORE that match (strict zero-lookahead).

    Features are simple rolling stats — we avoid calling compute_all_features_for_match
    here because it requires upcoming fixture context; instead we build the
    same rolling features the feature engine produces but from the results table.
    """
    rows = []
    teams = set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna())

    for idx in range(window * 2, len(df)):
        row = df.iloc[idx]
        hist = df.iloc[:idx]

        h = str(row["HomeTeam"])
        a = str(row["AwayTeam"])

        h_home = hist[hist["HomeTeam"] == h].tail(window)
        h_all  = hist[(hist["HomeTeam"] == h) | (hist["AwayTeam"] == h)].tail(window)
        a_away = hist[hist["AwayTeam"] == a].tail(window)
        a_all  = hist[(hist["HomeTeam"] == a) | (hist["AwayTeam"] == a)].tail(window)

        if len(h_all) < 3 or len(a_all) < 3:
            continue

        def goals_scored(sub, team):
            s = pd.concat([
                sub.loc[sub["HomeTeam"] == team, "FTHG"].rename("g"),
                sub.loc[sub["AwayTeam"] == team, "FTAG"].rename("g"),
            ])
            return float(s.mean()) if len(s) else 1.35

        def goals_conceded(sub, team):
            s = pd.concat([
                sub.loc[sub["HomeTeam"] == team, "FTAG"].rename("g"),
                sub.loc[sub["AwayTeam"] == team, "FTHG"].rename("g"),
            ])
            return float(s.mean()) if len(s) else 1.15

        lam_h = goals_scored(h_home if len(h_home) >= 3 else h_all, h)
        lam_a = goals_conceded(a_away if len(a_away) >= 3 else a_all, a)
        lam_a_scored = goals_scored(a_away if len(a_away) >= 3 else a_all, a)
        lam_h_conc   = goals_conceded(h_home if len(h_home) >= 3 else h_all, h)

        h_over_rate = float((h_all["total_goals"] > 2.5).mean())
        a_over_rate = float((a_all["total_goals"] > 2.5).mean())

        # H2H
        h2h = hist[
            ((hist["HomeTeam"] == h) & (hist["AwayTeam"] == a)) |
            ((hist["HomeTeam"] == a) & (hist["AwayTeam"] == h))
        ].tail(5)
        h2h_over_rate = float((h2h["total_goals"] > 2.5).mean()) if len(h2h) else 0.5

        # Simple Poisson prob (1 - CDF at 2)
        import math
        lam_total = lam_h + lam_a_scored
        poisson_over25 = 1.0 - sum(
            (lam_total**k * math.exp(-lam_total)) / math.factorial(k)
            for k in range(3)
        )

        feat = {
            "lam_home":          lam_h,
            "lam_away":          lam_a_scored,
            "lam_home_concede":  lam_h_conc,
            "lam_away_concede":  lam_a,
            "lam_total":         lam_total,
            "poisson_over25":    float(np.clip(poisson_over25, 0, 1)),
            "h_over25_rate":     h_over_rate,
            "a_over25_rate":     a_over_rate,
            "combined_over25":   (h_over_rate + a_over_rate) / 2,
            "h2h_over25_rate":   h2h_over_rate,
            "h_n_matches":       len(h_all),
            "a_n_matches":       len(a_all),
            "over25":            int(row["over25"]),
        }
        rows.append(feat)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── training ──────────────────────────────────────────────────────────────────

def train_league(league_name: str, df: pd.DataFrame) -> str | None:
    """
    Train an XGBoost model for one league. Returns the saved model name or None.
    """
    log.info(f"  Building features ({len(df)} matches)...")
    feat_df = build_feature_rows(df)

    if feat_df.empty or len(feat_df) < MIN_TRAIN:
        log.warning(f"  Skipped — only {len(feat_df)} feature rows (need {MIN_TRAIN})")
        return None

    y = feat_df["over25"].astype(int)
    X = feat_df.drop(columns=["over25"])

    # 80/20 chronological split
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    from sklearn.metrics import log_loss, roc_auc_score
    import xgboost as xgb

    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict_proba(X_val)[:, 1]
    ll  = log_loss(y_val, val_pred)
    auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.5
    over_rate = float(y.mean())

    log.info(f"  Trained: {len(X_train)} train / {len(X_val)} val  |  "
             f"LogLoss={ll:.4f}  AUC={auc:.3f}  Over25%={over_rate*100:.1f}%")

    # Sanitise league name for filename
    safe = (league_name.replace(" ", "_")
                        .replace("(", "").replace(")", "")
                        .replace("/", "_"))
    model_name = f"xgb_{safe}_over25"
    path = MODELS_DIR / f"{model_name}.joblib"

    import joblib
    joblib.dump({"model": model, "calibrator": None}, path)
    log.info(f"  Saved  -> models/{model_name}.joblib")

    return model_name, over_rate, ll, auc


# ── calibration cache update ──────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"leagues": {}, "generated_at": None}


def save_cache(cache: dict):
    from datetime import datetime
    cache["generated_at"] = datetime.utcnow().isoformat()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_leagues = list(dict.fromkeys(list(FDCO_CODES.keys()) + list(APIF_IDS.keys())))

    # Deduplicate canonical names (EPL covers Premier League etc.)
    canonical = {}
    seen_codes = set()
    for name in all_leagues:
        code = FDCO_CODES.get(name) or str(APIF_IDS.get(name, ""))
        if code and code not in seen_codes:
            canonical[name] = code
            seen_codes.add(code)

    log.info(f"Training models for {len(canonical)} unique leagues")
    log.info("=" * 60)

    cache = load_cache()
    results = []

    for i, (league_name, code) in enumerate(canonical.items(), 1):
        log.info(f"[{i}/{len(canonical)}]  {league_name}")

        df = get_league_data(league_name)
        if df.empty or len(df) < MIN_MATCHES:
            log.warning(f"  Skipped — {len(df)} matches (need {MIN_MATCHES})")
            results.append((league_name, "SKIPPED", len(df), None, None, None))
            continue

        try:
            result = train_league(league_name, df)
            if result is None:
                results.append((league_name, "NO_FEATURES", len(df), None, None, None))
                continue

            model_name, over_rate, ll, auc = result

            # Update calibration cache
            entry = cache.setdefault("leagues", {}).setdefault(league_name, {})
            entry["xgb_model_name"] = model_name
            entry["over25_base_rate"] = round(over_rate, 4)
            entry["xgb_val_logloss"]  = round(ll, 4)
            entry["xgb_val_auc"]      = round(auc, 4)
            entry["n_matches"]        = len(df)
            if league_name in FDCO_CODES:
                entry["fd_code"] = FDCO_CODES[league_name]

            results.append((league_name, "OK", len(df), model_name, ll, auc))

        except Exception as e:
            log.error(f"  FAILED: {e}")
            results.append((league_name, "ERROR", len(df), None, None, None))

    save_cache(cache)

    # Summary table
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    ok  = [r for r in results if r[1] == "OK"]
    skp = [r for r in results if r[1] != "OK"]

    for league, status, n, model, ll, auc in results:
        if status == "OK":
            log.info(f"  OK       {league:35} n={n:>4}  LL={ll:.4f}  AUC={auc:.3f}")
        else:
            log.info(f"  {status:<8} {league:35} n={n:>4}")

    log.info("")
    log.info(f"Trained: {len(ok)} leagues   Skipped: {len(skp)} leagues")
    log.info(f"Cache updated: {CACHE_FILE}")


if __name__ == "__main__":
    main()
