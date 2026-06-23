"""
PPO training pipeline — v2

Changes vs v1
─────────────
  Data         : 6-month CSV with DatetimeIndex (required for time features)
  Normalisation: rolling 500-bar Z-score inside TradingEnv (no norm_stats.npz)
  Split        : last 20% of bars held out as validation (time-based)
  Callback     : ValidationSharpeCallback — logs out-of-sample Sharpe to
                 TensorBoard every eval_freq steps; saves best model by val Sharpe
  Timesteps    : default 1,000,000
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

DATA_DIR       = ROOT / "data" / "raw"
MODEL_DIR      = ROOT / "models" / "saved"
TENSORBOARD_DIR = ROOT / "models" / "tensorboard"

for _d in (MODEL_DIR, MODEL_DIR / "checkpoints", TENSORBOARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.80   # first 80% of bars → training


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class SharpePnLCallback(BaseCallback):
    """Logs rolling Sharpe and cumulative PnL from training env to TensorBoard."""

    def __init__(self, rolling_window: int = 500, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.rolling_window = rolling_window
        self._step_returns: List[float] = []
        self._cumulative_pnl: float = 0.0

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            delta = float(info.get("delta_pnl", 0.0))
            self._cumulative_pnl += delta
            self._step_returns.append(delta)
            if len(self._step_returns) > self.rolling_window:
                self._step_returns.pop(0)
            if done:
                self.logger.record("episode/pnl", float(info.get("unrealised_pnl", 0.0)))

        if len(self._step_returns) >= 2:
            arr = np.asarray(self._step_returns, dtype=np.float64)
            std = arr.std()
            if std > 1e-10:
                self.logger.record("metrics/rolling_sharpe", float(arr.mean() / std))
        self.logger.record("metrics/cumulative_pnl", self._cumulative_pnl)
        return True


class ValidationSharpeCallback(BaseCallback):
    """
    Runs a deterministic episode on the validation env every eval_freq steps.
    Logs out-of-sample Sharpe to TensorBoard and saves the best model.

    Annualised Sharpe uses sqrt(252 * 24 * 60) ≈ 602 for 1-minute Forex.
    """

    ANNUALISE = np.sqrt(252 * 24 * 60)

    def __init__(
        self,
        val_df: pd.DataFrame,
        eval_freq: int = 100_000,
        save_path: Path = MODEL_DIR,
        save_name: str = "ppo_eurusd_best",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._val_env    = TradingEnv(val_df)
        self.eval_freq   = eval_freq
        self.save_path   = save_path
        self.save_name   = save_name
        self._best_sharpe: float = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        obs, _ = self._val_env.reset()
        returns: List[float] = []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self._val_env.step(action)
            returns.append(info["delta_pnl"])
            done = terminated or truncated

        arr = np.asarray(returns)
        std = arr.std()
        sharpe = float(arr.mean() / std * self.ANNUALISE) if std > 1e-10 else 0.0
        cum_pnl = float(np.sum(returns))

        self.logger.record("val/sharpe_annualised", sharpe)
        self.logger.record("val/cumulative_pnl",    cum_pnl)

        if self.verbose:
            logger.info(
                "Validation @ %d steps | Sharpe=%.3f | cum_pnl=%+.4f",
                self.num_timesteps, sharpe, cum_pnl,
            )

        if sharpe > self._best_sharpe:
            self._best_sharpe = sharpe
            best_path = self.save_path / self.save_name
            self.model.save(str(best_path))
            if self.verbose:
                logger.info("  ↳ New best val Sharpe — model saved → %s.zip", best_path)

        return True


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the most recent multi-month EURUSD CSV; synthetic fallback if absent."""
    # Prefer multi-month files, then any EURUSD file
    csvs = sorted(DATA_DIR.glob("EURUSD_1min_*M_*.csv"))
    if not csvs:
        csvs = sorted(DATA_DIR.glob("EURUSD_1min_*.csv"))

    if csvs:
        src = csvs[-1]
        logger.info("Loading data: %s", src)
        df = pd.read_csv(src, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df[["open", "high", "low", "close", "volume"]]

    logger.warning("No CSV found — using synthetic data (not suitable for live trading).")
    n = 50_000
    rng  = np.random.default_rng(42)
    ret  = rng.normal(0.0, 0.0001, n)
    c    = 1.1000 * np.cumprod(1.0 + ret)
    noise = rng.uniform(5e-5, 2e-4, n)
    o    = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame(
        {"open": o, "high": c+noise, "low": c-noise, "close": c,
         "volume": rng.integers(100, 1_000, n).astype(float)},
        index=pd.date_range("2025-01-06", periods=n, freq="1min", tz="UTC"),
    )


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based 80/20 split — never shuffled, no lookahead."""
    split_idx = int(len(df) * TRAIN_FRAC)
    train_df = df.iloc[:split_idx]
    val_df   = df.iloc[split_idx:]
    logger.info(
        "Split: %d train bars (→ %s) | %d val bars (%s →)",
        len(train_df), train_df.index[-1].date(),
        len(val_df),   val_df.index[0].date(),
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(df: pd.DataFrame):
    def _init() -> Monitor:
        return Monitor(TradingEnv(df))
    return _init


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(env: DummyVecEnv) -> PPO:
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(TENSORBOARD_DIR),
        verbose=1,
    )
    logger.info(
        "PPO | arch=[128,128] | lr=%.0e | ent_coef=%.4f | target_kl=0.01 | OBS_DIM=%d",
        model.learning_rate, model.ent_coef, TradingEnv.OBS_DIM,
    )
    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 2_000_000) -> Path:
    df = load_data()
    logger.info("Total bars: %d", len(df))

    train_df, val_df = split_data(df)

    vec_env = DummyVecEnv([make_env(train_df)])
    model   = build_model(vec_env)

    callbacks = [
        SharpePnLCallback(rolling_window=500),
        ValidationSharpeCallback(
            val_df=val_df,
            eval_freq=100_000,
            save_path=MODEL_DIR,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=100_000,
            save_path=str(MODEL_DIR / "checkpoints"),
            name_prefix="ppo_eurusd",
            verbose=1,
        ),
    ]

    logger.info("Training PPO — %d timesteps", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_eurusd_v2",
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final_path = MODEL_DIR / "ppo_eurusd_final"
    model.save(str(final_path))
    logger.info("Final model → %s.zip", final_path)

    # Quick sanity check
    loaded = PPO.load(str(final_path), env=vec_env)
    obs = vec_env.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (1, 1), f"Unexpected action shape: {action.shape}"
    logger.info("Load + predict check passed.")

    return final_path


def train_ensemble(
    n_models: int = 3,
    total_timesteps: int = 2_000_000,
    seeds: list[int] | None = None,
) -> list[Path]:
    """Train n_models PPO agents with different seeds; save best of each."""
    if seeds is None:
        seeds = list(range(n_models))

    df = load_data()
    logger.info("Total bars: %d", len(df))
    train_df, val_df = split_data(df)

    paths = []
    for i, seed in enumerate(seeds):
        logger.info("=== Ensemble member %d/%d (seed=%d) ===", i + 1, n_models, seed)
        vec_env = DummyVecEnv([make_env(train_df)])
        model   = build_model(vec_env)
        model.set_random_seed(seed)

        save_name = f"ppo_eurusd_ensemble_{i}_best"
        callbacks = [
            SharpePnLCallback(rolling_window=500),
            ValidationSharpeCallback(
                val_df=val_df,
                eval_freq=100_000,
                save_path=MODEL_DIR,
                save_name=save_name,
                verbose=1,
            ),
        ]
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"ppo_eurusd_ensemble_{i}",
            progress_bar=True,
            reset_num_timesteps=True,
        )
        paths.append(MODEL_DIR / save_name)
        logger.info("Ensemble member %d done → %s.zip", i, save_name)

    logger.info("Ensemble training complete. Members: %s", [str(p) for p in paths])
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--ensemble",  action="store_true",
                        help="Train 3 models with different seeds")
    parser.add_argument("--n-models",  type=int, default=3)
    args = parser.parse_args()

    if args.ensemble:
        train_ensemble(n_models=args.n_models, total_timesteps=args.timesteps)
    else:
        train(total_timesteps=args.timesteps)
