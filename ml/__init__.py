"""
ML Pipeline for Over 2.5 Goals Value Betting
=============================================
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

from ml.data_ingestion import build_master_dataset, fetch_upcoming_fixtures
from ml.feature_engine import compute_all_features
from ml.backtester_v2 import walk_forward_backtest, compare_strategies
from ml.calibration import kelly_stake
