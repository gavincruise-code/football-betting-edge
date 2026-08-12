"""
Standalone calibration runner — called by Windows Task Scheduler weekly.
Run directly: py -3 calibrate_leagues.py
"""
import sys
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "calibration.log"),
            encoding="utf-8",
        ),
    ],
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Weekly auto-calibration starting — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logging.info("=" * 60)

    try:
        from ml.league_calibrator import calibrate_all_leagues
        result = calibrate_all_leagues(n_matches=300)
        leagues = result.get("leagues", {})
        logging.info("Calibration complete — %d leagues updated.", len(leagues))
        for lg, info in sorted(leagues.items(), key=lambda x: x[1]["backtest_roi"], reverse=True):
            logging.info(
                "  %-26s  %-22s  ROI %+.1f%%  avg_goals %.2f",
                lg, info["strategy"], info["backtest_roi"], info["avg_goals"],
            )
    except Exception as exc:
        logging.error("Calibration failed: %s", exc)
        sys.exit(1)

    logging.info("Done.")
