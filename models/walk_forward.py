"""
Walk-forward validation.

Evaluates the trained model on each calendar month of the full dataset
independently. Reports Sharpe, max drawdown, Calmar, and win rate per window
so we can see whether the edge is consistent across market regimes or
concentrated in one lucky period.

Windows marked [IN] are inside the training set (Feb–May 26).
Windows marked [OUT] are out-of-sample (May 26–Jun 22).

Usage
─────
    venv/bin/python models/walk_forward.py
    venv/bin/python models/walk_forward.py --model final
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.trading_env import TradingEnv

logging.basicConfig(
    level=logging.WARNING,          # suppress per-step noise
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR  = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "saved"
ANNUALISE = np.sqrt(252 * 24 * 60)
TRAIN_CUTOFF = pd.Timestamp("2026-05-26", tz="UTC")   # 80/20 split boundary


class EnsembleModel:
    def __init__(self, models: list) -> None:
        self.models = models

    def predict(self, obs, deterministic: bool = True):
        actions = [m.predict(obs, deterministic=deterministic)[0] for m in self.models]
        return np.mean(actions, axis=0), None


def load_data() -> pd.DataFrame:
    csvs = sorted(DATA_DIR.glob("EURUSD_1min_*M_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No EURUSD CSV found in {DATA_DIR}")
    df = pd.read_csv(csvs[-1], index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def run_episode(model: PPO, window_df: pd.DataFrame) -> dict:
    env = TradingEnv(window_df)
    obs, _ = env.reset()
    returns, positions = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        returns.append(info["delta_pnl"])
        positions.append(info["position"])
        done = terminated or truncated
    return {"returns": np.asarray(returns), "positions": np.asarray(positions)}


def compute_metrics(returns: np.ndarray) -> dict:
    if len(returns) < 2:
        return {}
    cum       = np.cumprod(1.0 + returns)
    total_ret = float(cum[-1] - 1.0)
    n         = len(returns)
    ann_ret   = float(cum[-1] ** (ANNUALISE ** 2 / n) - 1.0)
    std       = returns.std()
    sharpe    = float(returns.mean() / std * ANNUALISE) if std > 1e-10 else 0.0
    roll_max  = np.maximum.accumulate(cum)
    max_dd    = float(((cum - roll_max) / roll_max).min())
    calmar    = float(ann_ret / abs(max_dd)) if max_dd < -1e-6 else np.inf
    win_rate  = float((returns > 0).mean())
    return {
        "n_steps":      n,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "calmar":       calmar,
        "win_rate":     win_rate,
    }


def walk_forward(model_name: str = "best") -> None:
    df = load_data()

    if model_name == "ensemble":
        paths = sorted(MODEL_DIR.glob("ppo_eurusd_ensemble_*_best.zip"))
        if not paths:
            raise FileNotFoundError("No ensemble members found. Run: train.py --ensemble")
        models = [PPO.load(str(p)) for p in paths]
        model = EnsembleModel(models)
        print(f"\nModel: ensemble ({len(models)} members)")
    else:
        path = MODEL_DIR / f"ppo_eurusd_{model_name}.zip"
        if not path.exists():
            alt = "final" if model_name == "best" else "best"
            print(f"  {path.name} not found — falling back to ppo_eurusd_{alt}.zip")
            path = MODEL_DIR / f"ppo_eurusd_{alt}.zip"
        model = PPO.load(str(path))
        print(f"\nModel: {path.name}")

    # Build monthly windows
    periods = df.groupby(df.index.to_period("M"))
    windows = []
    for period, group in periods:
        start = group.index[0]
        label = period.strftime("%b %Y")
        tag   = "[OUT]" if start >= TRAIN_CUTOFF else "[IN] "
        windows.append((label, tag, group))

    SEP = "=" * 76
    print(f"\n{SEP}")
    print(f"  Walk-Forward Validation — {len(windows)} monthly windows")
    print(f"  [IN]=in-sample (training set)  [OUT]=out-of-sample")
    print(SEP)
    print(f"  {'Window':<12} {'Tag':<7} {'Bars':>7} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'WinRate':>8} {'TotRet':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    all_sharpes = []
    for label, tag, window_df in windows:
        if len(window_df) <= TradingEnv._CURSOR_START + 10:
            print(f"  {label:<12} {tag:<7} {'SKIP — too few bars':>40}")
            continue
        try:
            result  = run_episode(model, window_df)
            metrics = compute_metrics(result["returns"])
        except Exception as exc:
            print(f"  {label:<12} {tag:<7} ERROR: {exc}")
            continue

        sharpe = metrics["sharpe"]
        all_sharpes.append(sharpe)
        calmar_str = f"{metrics['calmar']:>8.2f}" if np.isfinite(metrics["calmar"]) else "     inf"
        flag = " ✓" if sharpe > 0 else " ✗"
        print(
            f"  {label:<12} {tag:<7} {metrics['n_steps']:>7,} "
            f"{sharpe:>+8.3f}{flag} "
            f"{metrics['max_drawdown']*100:>7.2f}% "
            f"{calmar_str} "
            f"{metrics['win_rate']*100:>7.1f}% "
            f"{metrics['total_return']*100:>+7.2f}%"
        )

    print(SEP)
    if all_sharpes:
        pos = sum(1 for s in all_sharpes if s > 0)
        print(f"\n  Positive Sharpe windows : {pos}/{len(all_sharpes)}")
        print(f"  Mean Sharpe (all)       : {np.mean(all_sharpes):+.3f}")
        print(f"  Median Sharpe           : {np.median(all_sharpes):+.3f}")
        print(f"  Std of Sharpe           : {np.std(all_sharpes):.3f}")
        verdict = (
            "CONSISTENT EDGE — Sharpe positive in majority of windows."
            if pos >= len(all_sharpes) * 0.6
            else "REGIME-SPECIFIC — edge concentrated in few windows, proceed with caution."
        )
        print(f"\n  Verdict: {verdict}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best", choices=["best", "final", "ensemble"])
    args = parser.parse_args()
    walk_forward(model_name=args.model)
