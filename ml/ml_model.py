"""
XGBoost Machine Learning Classifier Module
===========================================
Train gradient-boosted decision trees on engineered features and Dixon-Coles baselines.
Supports Optuna hyperparameter tuning, SHAP feature importance, and probability calibration.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
import xgboost as xgb
import optuna
import shap

from ml.config import (
    OPTUNA_N_TRIALS, XGBOOST_PARAM_SPACE, MODELS_DIR,
    MIN_MODEL_PROBABILITY, MAX_MODEL_PROBABILITY
)

logger = logging.getLogger(__name__)
# Suppress Optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

META_COLUMNS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'over25', 'over25_odds', 'under25_odds', 'draw_odds', 'league', 'season_year']

def prepare_training_data(
    feature_df: pd.DataFrame,
    target_col: str = 'over25',
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Separates input features and target from raw dataset.
    """
    if exclude_cols is None:
        exclude_cols = META_COLUMNS

    y = feature_df[target_col].astype(int)
    drop_cols = [c for c in exclude_cols if c in feature_df.columns]
    X = feature_df.drop(columns=drop_cols, errors='ignore')

    # Convert non-numeric columns if any
    X = X.select_dtypes(include=[np.number, bool])

    return X, y, list(X.columns)

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train XGBoost with Optuna hyperparameter tuning on validation log loss.
    """
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', *XGBOOST_PARAM_SPACE['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *XGBOOST_PARAM_SPACE['learning_rate'], log=True),
            'n_estimators': trial.suggest_int('n_estimators', *XGBOOST_PARAM_SPACE['n_estimators']),
            'min_child_weight': trial.suggest_int('min_child_weight', *XGBOOST_PARAM_SPACE['min_child_weight']),
            'subsample': trial.suggest_float('subsample', *XGBOOST_PARAM_SPACE['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *XGBOOST_PARAM_SPACE['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *XGBOOST_PARAM_SPACE['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *XGBOOST_PARAM_SPACE['reg_lambda']),
            'n_jobs': -1,
        }
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        preds = clf.predict_proba(X_val)[:, 1]
        
        from sklearn.metrics import log_loss
        return log_loss(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=-1)

    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    val_preds = final_model.predict_proba(X_val)[:, 1]
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    metrics = {
        'best_params': best_params,
        'val_log_loss': float(log_loss(y_val, val_preds)),
        'val_brier': float(brier_score_loss(y_val, val_preds)),
        'val_auc': float(roc_auc_score(y_val, val_preds)) if len(np.unique(y_val)) > 1 else 0.5,
    }

    return final_model, metrics

def train_with_defaults(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """
    Train XGBoost with default baseline parameters for fast iteration.
    Utilizes all local CPU cores (n_jobs=-1).
    """
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    clean_cols = [str(col).replace('[', '').replace(']', '').replace('>', '_over_').replace('<', '_under_') for col in X_train.columns]
    X_train_clean = X_train.copy()
    X_train_clean.columns = clean_cols
    model.fit(X_train_clean, y_train)
    return model

def predict_proba(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    calibrator=None,
) -> np.ndarray:
    """
    Returns probability of Over 2.5 goals, optionally calibrated and clipped.
    """
    clean_cols = [str(col).replace('[', '').replace(']', '').replace('>', '_over_').replace('<', '_under_') for col in X.columns]
    X_clean = X.copy()
    X_clean.columns = clean_cols
    raw_probs = model.predict_proba(X_clean)[:, 1]

    if calibrator is not None:
        try:
            probs = calibrator.predict(raw_probs)
        except Exception:
            probs = raw_probs
    else:
        probs = raw_probs

    return np.clip(probs, MIN_MODEL_PROBABILITY, MAX_MODEL_PROBABILITY)

def compute_shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculates SHAP values for model predictions on X.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        return shap_vals, list(X.columns)
    except Exception as e:
        logger.warning(f"Failed to calculate SHAP values: {e}")
        return np.zeros(X.shape), list(X.columns)

def get_top_shap_features(
    model: xgb.XGBClassifier,
    X_single: pd.DataFrame,
    n: int = 5,
) -> List[Tuple[str, float, float]]:
    """
    Gets top N SHAP features for a single match row.
    Returns list of (feature_name, shap_value, feature_value).
    """
    shap_vals, cols = compute_shap_values(model, X_single)
    if len(shap_vals) == 0:
        return []

    row_shap = shap_vals[0]
    row_val = X_single.iloc[0].values

    # Sort by absolute SHAP magnitude
    indices = np.argsort(np.abs(row_shap))[::-1][:n]
    result = []
    for idx in indices:
        result.append((cols[idx], float(row_shap[idx]), float(row_val[idx])))

    return result

def save_model(model: xgb.XGBClassifier, name: str, calibrator=None) -> str:
    """Saves model and calibrator to models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump({'model': model, 'calibrator': calibrator}, file_path)
    return file_path

def load_model(name: str) -> Tuple[Optional[xgb.XGBClassifier], Optional[Any]]:
    """Loads model and calibrator from models directory."""
    file_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if os.path.exists(file_path):
        data = joblib.load(file_path)
        return data.get('model'), data.get('calibrator')
    return None, None
