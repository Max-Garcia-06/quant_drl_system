"""
Out-of-sample evaluation harness.

Loads the best or final trained PPO model, runs a deterministic episode on
the held-out validation set, and reports:

  - Total return
  - Annualised Sharpe ratio
  - Maximum drawdown
  - Calmar ratio  (annualised return / |max drawdown|)
  - Win rate      (% of steps with positive PnL)

Saves an equity-curve plot to models/saved/equity_curve.png.

Usage
─────
    venv/bin/python models/evaluate.py                      # uses best model + latest CSV
    venv/bin/python models/evaluate.py --model final        # force final model
    venv/bin/python models/evaluate.py --val-frac 0.20      # override val split fraction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR  = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "saved"

ANNUALISE = np.sqrt(252 * 24 * 60)   # 1-minute Forex annualisation factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_val_data(val_frac: float = 0.20) -> pd.DataFrame:
    csvs = sorted(DATA_DIR.glob("EURUSD_1min_*M_*.csv"))
    if not csvs:
        csvs = sorted(DATA_DIR.glob("EURUSD_1min_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No EURUSD CSV found in {DATA_DIR}")

    src = csvs[-1]
    logger.info("Loading data: %s", src)
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[["open", "high", "low", "close", "volume"]]

    split_idx = int(len(df) * (1.0 - val_frac))
    val_df = df.iloc[split_idx:]
    logger.info(
        "Validation set: %d bars | %s → %s",
        len(val_df), val_df.index[0].date(), val_df.index[-1].date(),
    )
    return val_df


def load_model(which: str = "best") -> PPO:
    """Load 'best' (by val Sharpe) or 'final' model."""
    candidates = {
        "best":  MODEL_DIR / "ppo_eurusd_best.zip",
        "final": MODEL_DIR / "ppo_eurusd_final.zip",
    }
    path = candidates.get(which, MODEL_DIR / f"{which}.zip")
    if not path.exists():
        # Fall back to the other one
        alt = candidates["final"] if which == "best" else candidates["best"]
        logger.warning("%s not found, falling back to %s", path, alt)
        path = alt
    logger.info("Loading model: %s", path)
    return PPO.load(str(path))


def run_episode(model: PPO, val_df: pd.DataFrame) -> dict:
    """Run one deterministic episode and return raw arrays."""
    env = TradingEnv(val_df)
    obs, _ = env.reset()

    step_returns, positions, rewards = [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_returns.append(info["delta_pnl"])
        positions.append(info["position"])
        rewards.append(reward)
        done = terminated or truncated

    return {
        "returns":   np.asarray(step_returns),
        "positions": np.asarray(positions),
        "rewards":   np.asarray(rewards),
    }


def compute_metrics(returns: np.ndarray) -> dict:
    cum        = np.cumprod(1.0 + returns)
    total_ret  = float(cum[-1] - 1.0)
    n          = len(returns)

    # Annualised return (CAGR-style)
    annual_ret = float(cum[-1] ** (ANNUALISE**2 / n) - 1.0)

    # Sharpe
    std = returns.std()
    sharpe = float(returns.mean() / std * ANNUALISE) if std > 1e-10 else 0.0

    # Max drawdown
    rolling_max = np.maximum.accumulate(cum)
    drawdowns   = (cum - rolling_max) / rolling_max
    max_dd      = float(drawdowns.min())

    # Calmar
    calmar = float(annual_ret / abs(max_dd)) if max_dd < -1e-6 else np.inf

    # Win rate
    win_rate = float((returns > 0).mean())

    return {
        "total_return":    total_ret,
        "annual_return":   annual_ret,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "calmar":          calmar,
        "win_rate":        win_rate,
        "n_steps":         n,
        "equity_curve":    cum,
    }


def plot_equity(cum: np.ndarray, positions: np.ndarray, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    steps = np.arange(len(cum))
    ax1.plot(steps, cum, linewidth=0.8, color="steelblue")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    ax1.set_ylabel("Equity (starting = 1.0)")
    ax1.set_title("Out-of-Sample Equity Curve — PPO EUR/USD")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(steps, positions, alpha=0.5, color="orange")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax2.set_ylabel("Position [-1, 1]")
    ax2.set_xlabel("Step (1-minute bars)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Equity curve saved → %s", out_path)


def evaluate(which: str = "best", val_frac: float = 0.20) -> dict:
    val_df  = load_val_data(val_frac)
    model   = load_model(which)

    logger.info("Running deterministic episode on %d val bars...", len(val_df))
    result  = run_episode(model, val_df)
    metrics = compute_metrics(result["returns"])

    SEP = "=" * 56
    print(f"\n{SEP}")
    print(f"  Out-of-Sample Evaluation  ({which} model)")
    print(SEP)
    print(f"  Bars evaluated   : {metrics['n_steps']:,}")
    print(f"  Total return     : {metrics['total_return']*100:+.2f}%")
    print(f"  Annual return    : {metrics['annual_return']*100:+.2f}%")
    print(f"  Sharpe (annual)  : {metrics['sharpe']:+.3f}")
    print(f"  Max drawdown     : {metrics['max_drawdown']*100:.2f}%")
    print(f"  Calmar ratio     : {metrics['calmar']:.3f}")
    print(f"  Win rate         : {metrics['win_rate']*100:.1f}%")
    print(SEP)

    plot_equity(
        metrics["equity_curve"],
        result["positions"],
        MODEL_DIR / "equity_curve.png",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="best",
                        help="'best', 'final', or path stem")
    parser.add_argument("--val-frac", type=float, default=0.20,
                        help="Fraction of data used as validation (default 0.20)")
    args = parser.parse_args()
    evaluate(which=args.model, val_frac=args.val_frac)
