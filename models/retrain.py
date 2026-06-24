"""
Retrain pipeline: fetch latest data → backup models → train ensemble → validate.

Steps
─────
  1. Fetch latest N months of EUR/USD 1-min MIDPOINT data from IB (requires TWS)
  2. Back up current ensemble to models/saved/backup_<timestamp>/
  3. Train a fresh 3-member PPO ensemble on the new data
  4. Walk-forward validation — prints per-month Sharpe table
  5. Models are ready; restart the live trader to deploy them

Usage
─────
    venv/bin/python models/retrain.py
    venv/bin/python models/retrain.py --skip-fetch          # reuse existing CSV
    venv/bin/python models/retrain.py --months 6 --timesteps 2000000 --n-models 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = ROOT / "models" / "saved"
DATA_DIR  = ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Step 1 — fetch
# ---------------------------------------------------------------------------

def fetch_data(n_months: int) -> Path:
    from data.ib_client import run_pipeline
    logger.info("Step 1/4 — Fetching %d months of EUR/USD 1-min data from IB...", n_months)
    df = asyncio.run(run_pipeline(n_months=n_months))
    csvs = sorted(DATA_DIR.glob("EURUSD_1min_*M_*.csv"))
    latest = csvs[-1]
    logger.info("Data saved → %s (%d bars)", latest.name, len(df))
    return latest


# ---------------------------------------------------------------------------
# Step 2 — backup
# ---------------------------------------------------------------------------

def backup_models() -> Path | None:
    ensemble_zips = sorted(MODEL_DIR.glob("ppo_eurusd_ensemble_*_best.zip"))
    if not ensemble_zips:
        logger.info("No existing ensemble models to back up.")
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = MODEL_DIR / f"backup_{ts}"
    backup_dir.mkdir()
    for z in ensemble_zips:
        shutil.copy2(z, backup_dir / z.name)
    logger.info("Step 2/4 — Backed up %d model(s) → %s", len(ensemble_zips), backup_dir)
    return backup_dir


# ---------------------------------------------------------------------------
# Step 3 — train
# ---------------------------------------------------------------------------

def train(n_models: int, total_timesteps: int) -> None:
    from models.train import train_ensemble
    logger.info(
        "Step 3/4 — Training ensemble (%d members × %s steps)...",
        n_models, f"{total_timesteps:,}",
    )
    paths = train_ensemble(n_models=n_models, total_timesteps=total_timesteps)
    logger.info("Ensemble trained. Members: %s", [p.name for p in paths])


# ---------------------------------------------------------------------------
# Step 4 — validate
# ---------------------------------------------------------------------------

def validate() -> None:
    import pandas as pd
    from models.walk_forward import walk_forward, load_data

    logger.info("Step 4/4 — Walk-forward validation on new ensemble...")

    # Print the new 80/20 cutoff so the [IN]/[OUT] labels in walk_forward
    # can be interpreted correctly (walk_forward.py has a hardcoded cutoff
    # from the previous training run — labels may differ, but Sharpe is correct)
    df = load_data()
    split_idx = int(len(df) * 0.80)
    new_cutoff = df.index[split_idx]
    logger.info(
        "New 80/20 split: train ends %s | val starts %s",
        df.index[split_idx - 1].date(), new_cutoff.date(),
    )

    walk_forward(model_name="ensemble")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EUR/USD PPO ensemble retrain pipeline")
    parser.add_argument("--months",      type=int, default=6,
                        help="Months of history to fetch (default: 6)")
    parser.add_argument("--n-models",    type=int, default=3,
                        help="Ensemble members to train (default: 3)")
    parser.add_argument("--timesteps",   type=int, default=2_000_000,
                        help="Training steps per member (default: 2,000,000)")
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Skip data fetch and use the most recent existing CSV")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RETRAIN PIPELINE")
    logger.info("  months=%d | n_models=%d | timesteps=%s | skip_fetch=%s",
                args.months, args.n_models, f"{args.timesteps:,}", args.skip_fetch)
    logger.info("=" * 60)

    if not args.skip_fetch:
        fetch_data(args.months)
    else:
        csvs = sorted(DATA_DIR.glob("EURUSD_1min_*.csv"))
        if not csvs:
            logger.error("--skip-fetch set but no CSV found in %s", DATA_DIR)
            sys.exit(1)
        logger.info("Step 1/4 — Skipping fetch. Using: %s", csvs[-1].name)

    backup_models()
    train(n_models=args.n_models, total_timesteps=args.timesteps)
    validate()

    logger.info("=" * 60)
    logger.info("RETRAIN COMPLETE — restart live_trader.py to deploy new models.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
