"""Run calibrate_all_leagues() for all 57 leagues and print a summary."""
import logging, os
from dotenv import load_dotenv
load_dotenv()
os.environ["NUMBA_DISABLE_JIT"] = "1"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

from ml.league_calibrator import calibrate_all_leagues

print("Starting calibration for all 57 leagues...")
cache = calibrate_all_leagues(n_matches=300, edge_margin=0.05, stake=10.0)
leagues = cache.get("leagues", {})

print()
print(f"Calibrated {len(leagues)} leagues. Summary:")
hdr = f"{'League':<30} {'Strategy':<25} {'ROI':>8}  {'AvgG':>6}  {'Draw%':>6}  Model"
print(hdr)
print("-" * 105)
for lg, info in sorted(leagues.items()):
    row = (
        f"{lg:<30} "
        f"{info.get('strategy', '?'):<25} "
        f"{info.get('backtest_roi', 0):>+7.1f}%  "
        f"{info.get('avg_goals', 0):>6.2f}  "
        f"{info.get('draw_rate_pct', 0):>5.0f}%  "
        f"{info.get('xgb_model_name', 'none')}"
    )
    print(row)
