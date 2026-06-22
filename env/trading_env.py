"""
Gymnasium MDP environment for DRL Forex trading.

Observation  (float32, shape (351,)):
    Flattened rolling window of 50 bars × 7 Z-score-normalised features
    [open, high, low, close, volume, macd_hist, rsi], plus a tanh-bounded
    unrealised-PnL scalar.

Action  (float32, shape (1,)):
    Target portfolio allocation ∈ [-1, 1].
    -1 = 100% short · 0 = flat · +1 = 100% long.

Reward (differential Sharpe with transaction cost penalty):
    R_t = ΔPnL_t / σ_PnL  −  τ · |a_t − a_{t-1}|
    where σ_PnL is an EMA of the running PnL standard deviation and τ = 0.0001.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    # --- Hyper-parameters (class-level so validate_env.py can read them) ---
    WINDOW_SIZE: int = 50

    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    RSI_PERIOD: int = 14

    # Features per bar: open, high, low, close, volume, macd_hist, rsi
    N_BAR_FEATURES: int = 7

    # OBS_DIM = 50 × 7 + 1 (unrealised PnL scalar) = 351
    OBS_DIM: int = WINDOW_SIZE * N_BAR_FEATURES + 1

    TRANSACTION_COST: float = 0.0001    # τ
    SHARPE_BETA: float = 0.01           # EMA decay for running Sharpe stats
    UPNL_SCALE: float = 10.0            # tanh scale for unrealised-PnL normalisation

    def __init__(
        self,
        df: pd.DataFrame,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        feature_df = self._build_features(df)
        _feat_cols = ["open", "high", "low", "close", "volume", "macd_hist", "rsi"]

        # Z-score params computed on the full dataset.
        # NOTE: global stats introduce slight lookahead bias; swap for a rolling
        # scaler (e.g. sklearn.preprocessing.StandardScaler fitted on train split)
        # before live deployment.
        raw: np.ndarray = feature_df[_feat_cols].values.astype(np.float64)
        self._means: np.ndarray = raw.mean(axis=0)
        self._stds: np.ndarray = raw.std(axis=0).clip(min=1e-8)

        # Pre-normalise the entire feature matrix once; sliced at every step.
        self._normed: np.ndarray = ((raw - self._means) / self._stds).astype(np.float32)
        self._close: np.ndarray = feature_df["close"].values.astype(np.float64)
        self._n_bars: int = len(self._close)

        if self._n_bars <= self.WINDOW_SIZE:
            raise ValueError(
                f"Dataset has {self._n_bars} bars after indicator warm-up, "
                f"but WINDOW_SIZE={self.WINDOW_SIZE} requires more."
            )

        # --- Gymnasium spaces ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state — properly initialised in reset()
        self._cursor: int = self.WINDOW_SIZE
        self._position: float = 0.0
        self._prev_action: float = 0.0
        self._unrealised_pnl: float = 0.0
        self._sharpe_mean: float = 0.0
        self._sharpe_m2: float = 1e-8   # running E[ΔPnL²]; seeded > 0 to avoid σ=0

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._cursor = self.WINDOW_SIZE
        self._position = 0.0
        self._prev_action = 0.0
        self._unrealised_pnl = 0.0
        self._sharpe_mean = 0.0
        self._sharpe_m2 = 1e-8

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Scalar target allocation; clip defensively even though action_space bounds it
        a: float = float(np.clip(action, -1.0, 1.0).flat[0])

        # ── Price return from t → t+1 ──────────────────────────────────
        prev_close: float = self._close[self._cursor]
        self._cursor += 1
        terminated: bool = self._cursor >= self._n_bars
        next_close: float = self._close[min(self._cursor, self._n_bars - 1)]

        price_return: float = (next_close - prev_close) / (prev_close + 1e-12)

        # ── PnL accrues from the PREVIOUS allocation (before rebalancing) ─
        delta_pnl: float = self._position * price_return
        self._unrealised_pnl += delta_pnl

        # ── Rebalance to new target ─────────────────────────────────────
        self._position = a

        # ── Differential Sharpe running statistics (EMA) ───────────────
        #   η_t  = (1-β)·η_{t-1} + β·ΔPnL_t        running mean
        #   M_t  = (1-β)·M_{t-1} + β·ΔPnL_t²       running E[ΔPnL²]
        #   σ_t  = √(M_t − η_t²)
        self._sharpe_mean = (
            (1.0 - self.SHARPE_BETA) * self._sharpe_mean
            + self.SHARPE_BETA * delta_pnl
        )
        self._sharpe_m2 = (
            (1.0 - self.SHARPE_BETA) * self._sharpe_m2
            + self.SHARPE_BETA * delta_pnl ** 2
        )
        sigma_pnl: float = float(
            np.sqrt(max(self._sharpe_m2 - self._sharpe_mean ** 2, 1e-8))
        )

        # ── Reward: R_t = ΔPnL_t/σ_PnL − τ·|a_t − a_{t-1}| ──────────
        tc_penalty: float = self.TRANSACTION_COST * abs(a - self._prev_action)
        reward: float = float(delta_pnl / sigma_pnl - tc_penalty)

        self._prev_action = a

        info: Dict[str, float] = {
            "price_return": price_return,
            "delta_pnl": delta_pnl,
            "sigma_pnl": sigma_pnl,
            "unrealised_pnl": self._unrealised_pnl,
            "position": self._position,
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Window slice: shape (WINDOW_SIZE, N_BAR_FEATURES)
        window: np.ndarray = self._normed[
            self._cursor - self.WINDOW_SIZE : self._cursor
        ]  # guaranteed valid because cursor ∈ [WINDOW_SIZE, n_bars]

        # Flatten → (WINDOW_SIZE * N_BAR_FEATURES,) = (350,)
        flat = window.flatten()

        # Unrealised PnL soft-bounded to (-1, 1):
        # UPNL_SCALE=10 keeps |tanh| < 0.97 for typical intraday cumulative returns
        upnl_normed = np.float32(np.tanh(self._unrealised_pnl * self.UPNL_SCALE))

        return np.append(flat, upnl_normed).astype(np.float32)

    @classmethod
    def _build_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MACD histogram and Wilder RSI and append to OHLCV columns.
        Uses EWM throughout — no hard NaN cutoffs except the leading diff() row.
        """
        out = df[["open", "high", "low", "close", "volume"]].copy().astype(float)

        # MACD histogram = (fast EMA − slow EMA) − signal-line EMA
        ema_fast = out["close"].ewm(span=cls.MACD_FAST, adjust=False).mean()
        ema_slow = out["close"].ewm(span=cls.MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=cls.MACD_SIGNAL, adjust=False).mean()
        out["macd_hist"] = macd_line - signal_line

        # Wilder RSI using EWM with α = 1/period (equivalent to Wilder smoothing)
        delta = out["close"].diff()
        avg_gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / cls.RSI_PERIOD, adjust=False
        ).mean()
        avg_loss = (-delta).clip(lower=0.0).ewm(
            alpha=1.0 / cls.RSI_PERIOD, adjust=False
        ).mean()
        rs = avg_gain / avg_loss.replace(0.0, 1e-8)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # Drop the single leading NaN row introduced by diff()
        out = out.dropna().reset_index(drop=True)
        return out
