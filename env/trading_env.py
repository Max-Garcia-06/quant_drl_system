"""
Gymnasium MDP environment for DRL Forex trading.

Key upgrades vs v1
──────────────────
  Features      : volume dropped; ATR + hour_sin/cos + dow_sin/cos added
                  N_BAR_FEATURES: 7 → 11  |  OBS_DIM: 351 → 551
  Normalisation : global Z-score → causal 500-bar rolling Z-score
                  (eliminates lookahead bias; adapts to regime shifts)
  Inference parity : _build_features() and _apply_rolling_norm() are
                  classmethods imported by the live trader — guarantees
                  bitwise identical feature computation at inference time.

Observation (float32, shape (551,)):
    Rolling window of 50 bars × 11 Z-score-normalised features, plus a
    tanh-bounded unrealised-PnL scalar.

Action (float32, shape (1,)):
    Target portfolio allocation ∈ [-1, 1].

Reward (differential Sharpe with transaction cost):
    R_t = ΔPnL_t / σ_PnL  −  τ · |a_t − a_{t-1}|
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    WINDOW_SIZE: int = 50
    NORM_WINDOW: int = 500      # rolling Z-score lookback

    MACD_FAST: int   = 12
    MACD_SLOW: int   = 26
    MACD_SIGNAL: int = 9
    RSI_PERIOD: int  = 14
    ATR_PERIOD: int  = 14

    # [open, high, low, close, macd_hist, rsi, atr, hour_sin, hour_cos, dow_sin, dow_cos]
    FEAT_COLS: List[str] = [
        "open", "high", "low", "close",
        "macd_hist", "rsi", "atr",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    N_BAR_FEATURES: int = len(FEAT_COLS)   # 11

    # 50 bars × 11 features + 1 unrealised-PnL scalar = 551
    OBS_DIM: int = WINDOW_SIZE * N_BAR_FEATURES + 1

    TRANSACTION_COST: float = 0.0001
    SHARPE_BETA: float       = 0.01
    UPNL_SCALE: float        = 10.0

    # Cursor starts here so the full NORM_WINDOW is available for every bar
    # in the first observation window.
    _CURSOR_START: int = NORM_WINDOW

    def __init__(
        self,
        df: pd.DataFrame,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        feature_df = self._build_features(df)
        self._normed: np.ndarray = self._apply_rolling_norm(feature_df)
        self._close: np.ndarray  = feature_df["close"].values.astype(np.float64)
        self._n_bars: int        = len(self._close)

        min_bars = self._CURSOR_START + 1
        if self._n_bars <= min_bars:
            raise ValueError(
                f"Dataset has {self._n_bars} bars after feature build; "
                f"need > {min_bars} (NORM_WINDOW={self.NORM_WINDOW} + buffer)."
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0),
            shape=(1,), dtype=np.float32,
        )

        self._cursor: int         = self._CURSOR_START
        self._position: float     = 0.0
        self._prev_action: float  = 0.0
        self._unrealised_pnl: float = 0.0
        self._sharpe_mean: float  = 0.0
        self._sharpe_m2: float    = 1e-8

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._cursor         = self._CURSOR_START
        self._position       = 0.0
        self._prev_action    = 0.0
        self._unrealised_pnl = 0.0
        self._sharpe_mean    = 0.0
        self._sharpe_m2      = 1e-8
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a: float = float(np.clip(action, -1.0, 1.0).flat[0])

        prev_close: float = self._close[self._cursor]
        self._cursor += 1
        terminated: bool  = self._cursor >= self._n_bars
        next_close: float = self._close[min(self._cursor, self._n_bars - 1)]

        price_return: float = (next_close - prev_close) / (prev_close + 1e-12)
        delta_pnl: float    = self._position * price_return
        self._unrealised_pnl += delta_pnl
        self._position = a

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

        tc_penalty: float = self.TRANSACTION_COST * abs(a - self._prev_action)
        reward: float     = float(delta_pnl / sigma_pnl - tc_penalty)
        self._prev_action = a

        info: Dict[str, float] = {
            "price_return":   price_return,
            "delta_pnl":      delta_pnl,
            "sigma_pnl":      sigma_pnl,
            "unrealised_pnl": self._unrealised_pnl,
            "position":       self._position,
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Feature engineering (classmethods — called identically by live trader)
    # ------------------------------------------------------------------

    @classmethod
    def _build_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from raw OHLCV data.

        Requires a DatetimeIndex (or datetime column "date" as index) to
        extract hour-of-day and day-of-week time features.

        Returns a DataFrame with columns = FEAT_COLS, same DatetimeIndex.
        """
        out = df[["open", "high", "low", "close"]].copy().astype(float)

        # ── MACD histogram ────────────────────────────────────────────
        ema_fast   = out["close"].ewm(span=cls.MACD_FAST,   adjust=False).mean()
        ema_slow   = out["close"].ewm(span=cls.MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal     = macd_line.ewm(span=cls.MACD_SIGNAL, adjust=False).mean()
        out["macd_hist"] = macd_line - signal

        # ── Wilder RSI ────────────────────────────────────────────────
        delta    = out["close"].diff()
        avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0/cls.RSI_PERIOD, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0.0).ewm(alpha=1.0/cls.RSI_PERIOD, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0.0, 1e-8)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # ── ATR (Wilder EWM of True Range) ───────────────────────────
        prev_close = out["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.ewm(span=cls.ATR_PERIOD, adjust=False).mean()

        # ── Cyclical time features (from DatetimeIndex) ───────────────
        idx = out.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have a DatetimeIndex for time feature extraction."
            )
        hour = idx.hour + idx.minute / 60.0
        dow  = idx.dayofweek.astype(float)          # 0=Mon … 4=Fri
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dow_sin"]  = np.sin(2 * np.pi * dow  / 5.0)
        out["dow_cos"]  = np.cos(2 * np.pi * dow  / 5.0)

        out = out.dropna()
        return out[cls.FEAT_COLS]

    @classmethod
    def _apply_rolling_norm(cls, feat_df: pd.DataFrame) -> np.ndarray:
        """
        Causal 500-bar rolling Z-score normalisation.

        Each bar t is normalised using the mean/std of bars [t-NORM_WINDOW : t],
        so there is zero lookahead bias.  NaN values (first NORM_WINDOW bars
        where the window is incomplete) are filled with 0.

        Returns float32 array of shape (n_bars, N_BAR_FEATURES).
        """
        roll_mean = feat_df.rolling(cls.NORM_WINDOW, min_periods=50).mean()
        roll_std  = (
            feat_df.rolling(cls.NORM_WINDOW, min_periods=50)
            .std()
            .clip(lower=1e-8)
        )
        normed = ((feat_df - roll_mean) / roll_std).fillna(0.0)
        return normed.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        window = self._normed[
            self._cursor - self.WINDOW_SIZE : self._cursor
        ].flatten()
        upnl = np.float32(np.tanh(self._unrealised_pnl * self.UPNL_SCALE))
        return np.append(window, upnl).astype(np.float32)
