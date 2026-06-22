"""
Validation script for TradingEnv.

Generates synthetic EUR/USD-like 1-min OHLCV data with a proper DatetimeIndex
(required for time features), instantiates the environment, and runs random
actions through a full episode to verify:
  - observation/action space shapes and dtypes
  - absence of NaN / Inf at every step
  - reward is a scalar float
  - terminated flag fires correctly
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from env.trading_env import TradingEnv


def make_synthetic_ohlcv(n_bars: int = 3_000, seed: int = 42) -> pd.DataFrame:
    """Synthetic EUR/USD GBM series with a UTC DatetimeIndex (required by TradingEnv)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.0001, n_bars)
    close = 1.1000 * np.cumprod(1.0 + log_returns)
    noise = rng.uniform(5e-5, 2e-4, n_bars)
    open_ = np.concatenate([[close[0]], close[:-1]])

    df = pd.DataFrame({
        "open":   open_,
        "high":   close + rng.uniform(0.0, noise, n_bars),
        "low":    close - rng.uniform(0.0, noise, n_bars),
        "close":  close,
        "volume": rng.integers(100, 1_000, n_bars).astype(float),
    }, index=pd.date_range("2025-01-06 00:00", periods=n_bars, freq="1min", tz="UTC"))

    return df


def validate(n_episodes: int = 2) -> None:
    SEP = "=" * 64

    print(SEP)
    print("  TradingEnv v2 — Dimension & Stability Validation")
    print(SEP)

    df = make_synthetic_ohlcv(n_bars=3_000)
    env = TradingEnv(df)

    expected_obs = (TradingEnv.OBS_DIM,)   # (551,)
    expected_act = (1,)

    assert env.observation_space.shape == expected_obs, env.observation_space.shape
    assert env.action_space.shape == expected_act

    print(f"\nState vector  : Box{expected_obs}  (float32)")
    print(
        f"  [0:{TradingEnv.WINDOW_SIZE * TradingEnv.N_BAR_FEATURES}]"
        f"  {TradingEnv.WINDOW_SIZE} bars × {TradingEnv.N_BAR_FEATURES} features"
        f"  {TradingEnv.FEAT_COLS}  — rolling {TradingEnv.NORM_WINDOW}-bar Z-score"
    )
    print(f"  [{TradingEnv.WINDOW_SIZE * TradingEnv.N_BAR_FEATURES}]  unrealised PnL (tanh-bounded)")
    print(f"\nAction space  : Box{expected_act}  ∈ [-1, 1]")
    print(f"NORM_WINDOW   : {TradingEnv.NORM_WINDOW} bars (causal rolling Z-score)")
    print(f"Cursor start  : {TradingEnv._CURSOR_START}  (first obs has full norm history)")

    all_rewards, all_pnl = [], []
    total_steps = 0

    t0 = time.perf_counter()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)

        assert obs.shape == expected_obs
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs)),  "NaN in reset obs"
        assert not np.any(np.isinf(obs)),  "Inf in reset obs"

        ep_rewards = []
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert obs.shape == expected_obs
            assert obs.dtype == np.float32
            assert isinstance(reward, float)
            assert not np.any(np.isnan(obs)),   f"NaN in obs at step {len(ep_rewards)}"
            assert not np.any(np.isinf(obs)),   f"Inf in obs at step {len(ep_rewards)}"
            assert not np.isnan(reward),        f"NaN reward at step {len(ep_rewards)}"

            ep_rewards.append(reward)
            all_pnl.append(info["unrealised_pnl"])

            if terminated or truncated:
                break

        all_rewards.extend(ep_rewards)
        total_steps += len(ep_rewards)
        print(f"\n  Episode {ep+1}: {len(ep_rewards)} steps | "
              f"cum PnL={info['unrealised_pnl']:+.6f} | "
              f"mean reward={np.mean(ep_rewards):+.6f}")

    elapsed = time.perf_counter() - t0

    print(f"\n{SEP}")
    print(f"  {total_steps} total steps in {elapsed:.2f}s "
          f"({total_steps/elapsed:,.0f} steps/sec)")
    arr = np.asarray(all_rewards)
    print(f"  Reward — min={arr.min():+.5f}  max={arr.max():+.5f}  "
          f"μ={arr.mean():+.6f}  σ={arr.std():.6f}")
    print(f"\n{SEP}")
    print("  All assertions passed — TradingEnv v2 is stable.")
    print(SEP)


if __name__ == "__main__":
    validate(n_episodes=2)
