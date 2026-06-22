"""
PPO training pipeline for the DRL Forex trading system.

Architecture
    Policy     :  MlpPolicy — shared [128, 128] MLP trunk (actor + critic)
    ent_coef   :  0.01  — entropy bonus discourages premature policy collapse
    Timesteps  :  100,000 (configurable via CLI or import)
    Checkpoints:  models/saved/checkpoints/  every 10,000 steps
    Final save :  models/saved/ppo_eurusd_final.zip
    TensorBoard:  models/tensorboard/ppo_eurusd_<run>/
                  → metrics/rolling_sharpe
                  → metrics/cumulative_pnl
                  → episode/pnl
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "saved"
TENSORBOARD_DIR = ROOT / "models" / "tensorboard"

for _d in (MODEL_DIR, MODEL_DIR / "checkpoints", TENSORBOARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom TensorBoard callback
# ---------------------------------------------------------------------------

class SharpePnLCallback(BaseCallback):
    """
    Logs the rolling Sharpe ratio and cumulative PnL to TensorBoard.

    TensorBoard keys
    ----------------
    metrics/rolling_sharpe   Sharpe of the last `rolling_window` step-returns
                             (mean / std, unitless — multiply by sqrt(N) to annualise)
    metrics/cumulative_pnl   Gross PnL accumulated across all environment steps
    episode/pnl              Unrealised PnL at episode termination
    """

    def __init__(self, rolling_window: int = 500, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.rolling_window = rolling_window
        self._step_returns: List[float] = []
        self._cumulative_pnl: float = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))

        for info, done in zip(infos, dones):
            delta_pnl: float = float(info.get("delta_pnl", 0.0))

            self._cumulative_pnl += delta_pnl
            self._step_returns.append(delta_pnl)
            if len(self._step_returns) > self.rolling_window:
                self._step_returns.pop(0)

            # Episode boundary: log terminal unrealised PnL
            if done:
                self.logger.record("episode/pnl", float(info.get("unrealised_pnl", 0.0)))

        # Rolling Sharpe — only meaningful once the buffer has warmed up
        if len(self._step_returns) >= 2:
            arr = np.asarray(self._step_returns, dtype=np.float64)
            std = arr.std()
            if std > 1e-10:
                # Annualisation note: multiply by sqrt(252*24*60) ≈ 602 for 1-min Forex
                self.logger.record("metrics/rolling_sharpe", float(arr.mean() / std))

        self.logger.record("metrics/cumulative_pnl", self._cumulative_pnl)
        return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_or_generate_data(n_synthetic_bars: int = 10_000) -> pd.DataFrame:
    """
    Load the most recent EURUSD 1-min CSV from data/raw/.
    Falls back to synthetic GBM data if no CSV exists (e.g. TWS not connected).
    """
    csvs = sorted(DATA_DIR.glob("EURUSD_1min_*.csv"))
    if csvs:
        src = csvs[-1]
        logger.info("Loading historical data: %s", src)
        df = pd.read_csv(src, index_col=0, parse_dates=True)
        return df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)

    logger.warning(
        "No EURUSD CSV found in %s — using %d bars of synthetic GBM data.",
        DATA_DIR,
        n_synthetic_bars,
    )
    rng = np.random.default_rng(42)
    log_ret = rng.normal(0.0, 0.0001, n_synthetic_bars)
    close = 1.1000 * np.cumprod(1.0 + log_ret)
    noise = rng.uniform(5e-5, 2e-4, n_synthetic_bars)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open":   open_,
            "high":   close + rng.uniform(0.0, noise, n_synthetic_bars),
            "low":    close - rng.uniform(0.0, noise, n_synthetic_bars),
            "close":  close,
            "volume": rng.integers(100, 1_000, n_synthetic_bars).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(df: pd.DataFrame):
    """
    Returns an env constructor for DummyVecEnv.
    Monitor wrapping is required by SB3 for episode reward/length tracking.
    """
    def _init() -> Monitor:
        return Monitor(TradingEnv(df))
    return _init


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(env: DummyVecEnv) -> PPO:
    """
    Construct PPO with a shared [128, 128] MLP policy/value network.

    net_arch=[128, 128]
        Both the actor (pi) and critic (vf) share a 2-layer trunk of 128 units.
        Heads are added on top by SB3 automatically.

    Key hyperparameters
    -------------------
    ent_coef   = 0.01   encourages continued exploration; prevents the policy
                         from collapsing to a fixed allocation too early
    n_steps    = 2048   rollout buffer length; PPO updates after collecting
                         this many steps from the env
    batch_size = 64     mini-batch size for gradient steps (2048 / 64 = 32 batches)
    n_epochs   = 10     passes over the rollout buffer per update cycle
    clip_range = 0.2    PPO clipping parameter (standard default)
    """
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(TENSORBOARD_DIR),
        verbose=1,
    )
    logger.info(
        "PPO built | arch=%s | lr=%.0e | ent_coef=%.4f | n_steps=%d | batch=%d | epochs=%d",
        [128, 128],
        model.learning_rate,
        model.ent_coef,
        model.n_steps,
        model.batch_size,
        model.n_epochs,
    )
    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 100_000) -> Path:
    """
    Full training pipeline:
      1. Load data
      2. Wrap in VecEnv
      3. Build PPO
      4. Train with Sharpe/PnL callback + checkpoint callback
      5. Save final model

    Returns
    -------
    Path
        Path to the saved .zip (without the .zip extension, per SB3 convention).
    """
    df = load_or_generate_data()
    logger.info("Dataset: %d bars after feature construction warm-up", len(df))

    vec_env = DummyVecEnv([make_env(df)])

    # Export normalization stats so the live trader uses identical µ/σ to training.
    # Monitor wraps TradingEnv, so unwrap one level to reach _means/_stds.
    _trading_env: TradingEnv = vec_env.envs[0].env
    norm_stats_path = MODEL_DIR / "norm_stats.npz"
    np.savez(str(norm_stats_path), means=_trading_env._means, stds=_trading_env._stds)
    logger.info(
        "Normalization stats saved → %s  (means shape=%s)",
        norm_stats_path,
        _trading_env._means.shape,
    )

    model = build_model(vec_env)

    callbacks = [
        SharpePnLCallback(rolling_window=500),
        CheckpointCallback(
            save_freq=10_000,
            save_path=str(MODEL_DIR / "checkpoints"),
            name_prefix="ppo_eurusd",
            verbose=1,
        ),
    ]

    logger.info("Training PPO — %d timesteps", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_eurusd",
        progress_bar=True,
        reset_num_timesteps=True,
    )

    save_path = MODEL_DIR / "ppo_eurusd_final"
    model.save(str(save_path))
    logger.info("Final model saved → %s.zip", save_path)

    # Quick sanity check: reload and run one forward pass
    loaded = PPO.load(str(save_path), env=vec_env)
    obs = vec_env.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (1, 1), f"Unexpected action shape: {action.shape}"
    logger.info("Load + predict sanity check passed. Action shape: %s", action.shape)

    return save_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on EUR/USD TradingEnv")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total environment steps to train for (default: 100,000)",
    )
    args = parser.parse_args()
    train(total_timesteps=args.timesteps)
