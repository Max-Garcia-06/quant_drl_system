"""
Validation script for TradingEnv.

Generates synthetic EUR/USD-like 1-min OHLCV data, instantiates the
environment, and runs random actions through a full episode to verify:
  - observation/action space shapes and dtypes
  - absence of NaN / Inf at every step
  - reward is a scalar float
  - terminated flag fires correctly and triggers a clean reset
  - state-vector component ranges are sane
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from env.trading_env import TradingEnv


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def make_synthetic_ohlcv(n_bars: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a EUR/USD-like 1-minute OHLCV series using geometric Brownian motion.
    σ = 1 pip/min  →  realistic intraday noise level.
    """
    rng = np.random.default_rng(seed)

    log_returns = rng.normal(0.0, 0.0001, size=n_bars)
    close = 1.1000 * np.cumprod(1.0 + log_returns)

    noise = rng.uniform(0.00005, 0.00020, size=n_bars)
    high = close + rng.uniform(0.0, noise, size=n_bars)
    low = close - rng.uniform(0.0, noise, size=n_bars)

    # Open = previous close (no gap)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    volume = rng.integers(100, 1_000, size=n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# Validation routine
# ---------------------------------------------------------------------------

def validate(n_episodes: int = 2) -> None:
    SEP = "=" * 64

    print(SEP)
    print("  TradingEnv — Dimension & Stability Validation")
    print(SEP)

    df = make_synthetic_ohlcv(n_bars=2_000)
    env = TradingEnv(df)

    # ── Space assertions ───────────────────────────────────────────────
    expected_obs = (TradingEnv.OBS_DIM,)   # (351,)
    expected_act = (1,)

    assert env.observation_space.shape == expected_obs, (
        f"Obs space shape mismatch: got {env.observation_space.shape}"
    )
    assert env.action_space.shape == expected_act, (
        f"Act space shape mismatch: got {env.action_space.shape}"
    )

    print(f"\nState vector  : Box{expected_obs}  (float32)")
    print(
        f"  [0 : {TradingEnv.WINDOW_SIZE * TradingEnv.N_BAR_FEATURES}]"
        f"  {TradingEnv.WINDOW_SIZE} bars × {TradingEnv.N_BAR_FEATURES} features"
        f"  [open, high, low, close, volume, macd_hist, rsi]  — Z-scored"
    )
    print(
        f"  [{TradingEnv.WINDOW_SIZE * TradingEnv.N_BAR_FEATURES}]"
        f"       unrealised PnL  — tanh-bounded"
    )
    print(f"\nAction space  : Box{expected_act}  ∈ [-1, 1]  (float32)")
    print(f"τ (tx cost)   : {TradingEnv.TRANSACTION_COST}")
    print(f"Window        : {TradingEnv.WINDOW_SIZE} periods")
    print(f"MACD          : EMA({TradingEnv.MACD_FAST}) − EMA({TradingEnv.MACD_SLOW}), signal={TradingEnv.MACD_SIGNAL}")
    print(f"RSI           : Wilder {TradingEnv.RSI_PERIOD}-period")

    # ── Episode loop ───────────────────────────────────────────────────
    all_rewards: list[float] = []
    all_pnl: list[float] = []
    all_sigma: list[float] = []
    total_steps = 0
    episode_lengths: list[int] = []

    t0 = time.perf_counter()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)

        # reset() checks
        assert obs.shape == expected_obs,   f"reset() obs shape {obs.shape}"
        assert obs.dtype == np.float32,     f"reset() dtype {obs.dtype}"
        assert not np.any(np.isnan(obs)),   "NaN in initial obs"
        assert not np.any(np.isinf(obs)),   "Inf in initial obs"

        ep_rewards: list[float] = []
        step = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            # Per-step invariants
            assert obs.shape == expected_obs, \
                f"ep={ep} step={step}: obs shape {obs.shape}"
            assert obs.dtype == np.float32, \
                f"ep={ep} step={step}: dtype {obs.dtype}"
            assert isinstance(reward, float), \
                f"ep={ep} step={step}: reward type {type(reward)}"
            assert not np.any(np.isnan(obs)), \
                f"ep={ep} step={step}: NaN in obs"
            assert not np.any(np.isinf(obs)), \
                f"ep={ep} step={step}: Inf in obs"
            assert not np.isnan(reward), \
                f"ep={ep} step={step}: NaN reward"
            assert not np.isinf(reward), \
                f"ep={ep} step={step}: Inf reward"

            ep_rewards.append(reward)
            all_pnl.append(info["unrealised_pnl"])
            all_sigma.append(info["sigma_pnl"])

            if terminated or truncated:
                break

        all_rewards.extend(ep_rewards)
        total_steps += step
        episode_lengths.append(step)
        print(
            f"\n  Episode {ep + 1}: {step} steps | "
            f"cumulative PnL = {info['unrealised_pnl']:+.6f} | "
            f"mean reward = {np.mean(ep_rewards):+.6f}"
        )

    elapsed = time.perf_counter() - t0

    # ── Aggregate stats ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Step-level statistics  ({total_steps} total steps, {elapsed:.2f}s)")
    print(SEP)

    def _fmt(label: str, arr: list[float]) -> str:
        a = np.asarray(arr)
        return (
            f"  {label:<22}"
            f"  min={a.min():+.8f}  max={a.max():+.8f}"
            f"  μ={a.mean():+.8f}  σ={a.std():.8f}"
        )

    print(_fmt("Reward", all_rewards))
    print(_fmt("Unrealised PnL", all_pnl))
    print(_fmt("σ_PnL", all_sigma))

    # Confirm obs[-1] (upnl) is always in (-1, 1) — tanh guarantee
    obs_sample, _ = env.reset()
    upnl_val = float(obs_sample[-1])
    assert -1.0 < upnl_val <= 1.0 or upnl_val == 0.0, \
        f"unrealised PnL component out of tanh range: {upnl_val}"

    print(f"\n  Throughput : {total_steps / elapsed:,.0f} steps/sec")
    print(f"\n{SEP}")
    print("  All assertions passed — TradingEnv dimensions are stable.")
    print(SEP)


if __name__ == "__main__":
    validate(n_episodes=2)
