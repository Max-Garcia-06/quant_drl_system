"""
Live execution bridge — v2

Changes vs v1
─────────────
  Features      : matches TradingEnv v2 — ATR, hour_sin/cos, dow_sin/cos;
                  volume dropped; N_BAR_FEATURES 7 → 11
  Normalisation : no norm_stats.npz; calls TradingEnv._build_features() and
                  TradingEnv._apply_rolling_norm() directly — guaranteed parity
  Warm-up       : WARMUP_BARS 200 → 650 (NORM_WINDOW + WINDOW_SIZE + buffer)
  Bar buffer    : stores dicts with 'datetime' key for time features

Flow (every 1-minute candle close)
───────────────────────────────────
  reqRealTimeBars (5s MIDPOINT)
    → accumulate 12 × 5s bars at wall-clock minute boundary
    → aggregate into 1-min OHLCV dict (with datetime)
    → TradingEnv._build_features() → _apply_rolling_norm()
    → last 50 rows → flatten → append tanh(unrealised_pnl) → (551,) obs
    → model.predict(obs, deterministic=True) → action ∈ [-1, 1]
    → delta = round(action * capital/price − current_units)
    → ib.whatIfOrderAsync()  or  ib.placeOrder()

CRITICAL: patchAsyncio() at module top — do not remove.
"""

from __future__ import annotations

import ib_insync.util
ib_insync.util.patchAsyncio()

import asyncio
import logging
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ib_insync import IB, Forex, MarketOrder, RealTimeBarList
from stable_baselines3 import PPO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from env.trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH  = ROOT / "models" / "saved" / "ppo_eurusd_best"   # single-model fallback
MODEL_DIR   = ROOT / "models" / "saved"


class EnsembleModel:
    """Averages actions from N independently-trained PPO members."""

    def __init__(self, models: list) -> None:
        self.models = models

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        actions = [m.predict(obs, deterministic=deterministic)[0] for m in self.models]
        return np.mean(actions, axis=0), None


def _load_model(model_path: Path) -> EnsembleModel | PPO:
    """Load ensemble if members exist, otherwise load single model."""
    ensemble_paths = sorted(MODEL_DIR.glob("ppo_eurusd_ensemble_*_best.zip"))
    if ensemble_paths:
        members = [PPO.load(str(p)) for p in ensemble_paths]
        logger.info("Loaded ensemble of %d members: %s",
                    len(members), [p.name for p in ensemble_paths])
        return EnsembleModel(members)

    mp = Path(str(model_path) + ".zip")
    if not mp.exists():
        alt = MODEL_DIR / "ppo_eurusd_final"
        logger.warning("%s not found — falling back to %s", mp, alt)
        model_path = alt
    logger.info("Loading single model: %s", model_path)
    return PPO.load(str(model_path))


class RegimeGuard:
    """
    Scales down position size when ATR diverges from warm-up baseline.

    During warm-up, records mean and std of ATR across the historical bars.
    On each candle, if current ATR exceeds mean + N_SIGMA*std, linearly
    reduces the risk scalar toward 0 (full reduction at mean + 4*std).
    """

    N_SIGMA: float = 2.5

    def __init__(self) -> None:
        self._atr_mean: float = 0.0
        self._atr_std:  float = 1.0
        self._calibrated: bool = False

    def calibrate(self, bars: list) -> None:
        atrs = []
        for b in bars:
            h, l, c_prev = b["high"], b["low"], None
            atrs.append(h - l)
        arr = np.array(atrs, dtype=np.float64)
        self._atr_mean = float(arr.mean())
        self._atr_std  = float(arr.std()) or 1e-8
        self._calibrated = True
        logger.info(
            "RegimeGuard calibrated | ATR mean=%.6f | std=%.6f | threshold=%.6f",
            self._atr_mean, self._atr_std,
            self._atr_mean + self.N_SIGMA * self._atr_std,
        )

    def risk_scalar(self, current_bar: dict) -> float:
        if not self._calibrated:
            return 1.0
        atr_now = current_bar["high"] - current_bar["low"]
        threshold = self._atr_mean + self.N_SIGMA * self._atr_std
        if atr_now <= threshold:
            return 1.0
        upper = self._atr_mean + 4.0 * self._atr_std
        scalar = float(np.clip(1.0 - (atr_now - threshold) / (upper - threshold + 1e-8), 0.0, 1.0))
        logger.warning(
            "REGIME GUARD | atr=%.6f | threshold=%.6f | scalar=%.3f",
            atr_now, threshold, scalar,
        )
        return scalar


class DrawdownGuard:
    """
    Halts trading when peak-to-trough session drawdown exceeds max_drawdown.
    Auto-resumes after cooldown_hours without manual intervention.
    """

    def __init__(self, max_drawdown: float = 0.03, cooldown_hours: float = 2.0) -> None:
        self._max_drawdown   = max_drawdown
        self._cooldown_secs  = cooldown_hours * 3600
        self._peak_equity:   float = 0.0
        self._halted_until:  Optional[datetime] = None
        self._initialized:   bool = False

    def update(self, equity: float) -> bool:
        """Update peak, check drawdown. Returns True only on halt transition."""
        if not self._initialized:
            self._peak_equity = equity
            self._initialized = True

        if self.is_halted():
            return False

        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = (self._peak_equity - equity) / (self._peak_equity + 1e-12)
        if drawdown > self._max_drawdown:
            from datetime import timedelta
            self._halted_until = datetime.now(tz=timezone.utc) + timedelta(
                seconds=self._cooldown_secs
            )
            logger.warning(
                "DRAWDOWN HALT | drawdown=%.2f%% > %.0f%% threshold | "
                "peak=%.2f | current=%.2f | resuming at %s",
                drawdown * 100, self._max_drawdown * 100,
                self._peak_equity, equity,
                self._halted_until.strftime("%H:%M:%S UTC"),
            )
            return True

        return False

    def is_halted(self) -> bool:
        if self._halted_until is None:
            return False
        if datetime.now(tz=timezone.utc) >= self._halted_until:
            logger.info("DRAWDOWN GUARD | cooldown expired — resuming trading.")
            self._halted_until = None
            return False
        return True


class LiveTrader:
    """
    Connects to TWS, subscribes to EUR.USD real-time bars, runs PPO inference
    on every 1-minute candle, and rebalances via whatIfOrder or placeOrder.
    """

    TWS_HOST: str = "127.0.0.1"
    TWS_PORT: int = 7497
    CLIENT_ID: int = 3

    # Need NORM_WINDOW + WINDOW_SIZE bars warm so rolling norm is stable
    WARMUP_BARS: int = TradingEnv.NORM_WINDOW + TradingEnv.WINDOW_SIZE + 100   # 650

    # Enough bars in buffer before first prediction
    MIN_PREDICT_BARS: int = TradingEnv.NORM_WINDOW + TradingEnv.WINDOW_SIZE    # 550

    MIN_ORDER_UNITS: int = 1_000   # IB IDEALPRO minimum EUR lot

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        capital: float   = 1_000.0,
        whatif_only: bool = True,
    ) -> None:
        self.capital     = capital
        self.whatif_only = whatif_only

        self.model = _load_model(model_path)

        self.ib       = IB()
        self.contract = Forex("EURUSD")

        # Rolling 1-minute bar buffer — stores dicts with open/high/low/close/datetime
        self._minute_bars: deque[Dict] = deque(maxlen=self.WARMUP_BARS + 100)

        # 5-second RT bar accumulation
        self._rt_bar_accum: List = []
        self._rt_bar_minute: Optional[datetime] = None

        # Position / PnL state
        self._current_action: float   = 0.0   # normalised allocation ∈ [-1, 1]
        self._current_units: float    = 0.0   # actual EUR units held
        self._unrealised_pnl: float   = 0.0
        self._prev_close: Optional[float] = None

        self._rt_bars: Optional[RealTimeBarList] = None
        self._last_bar_time: Optional[datetime] = None
        self._regime_guard    = RegimeGuard()
        self._drawdown_guard  = DrawdownGuard(max_drawdown=0.03, cooldown_hours=2.0)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> None:
        asyncio.run(self._run_async())

    def close_position(self) -> None:
        """Standalone: connect to TWS, close open position, disconnect."""
        asyncio.run(self._run_close_async())

    def dry_run(self, n_bars: int = 700) -> None:
        """Test the full inference pipeline without connecting to TWS."""
        logger.info("=== DRY RUN ===")

        rng   = np.random.default_rng(99)
        ret   = rng.normal(0.0, 0.0001, n_bars)
        close = 1.1000 * np.cumprod(1.0 + ret)
        noise = rng.uniform(5e-5, 2e-4, n_bars)
        opens = np.concatenate([[close[0]], close[:-1]])
        base_dt = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")

        self._minute_bars.clear()
        for i in range(n_bars):
            self._minute_bars.append({
                "open":     float(opens[i]),
                "high":     float(close[i] + noise[i]),
                "low":      float(close[i] - noise[i]),
                "close":    float(close[i]),
                "datetime": base_dt + pd.Timedelta(minutes=i),
            })

        self._prev_close = close[-2]
        state = self._build_state()

        if state is None:
            logger.error("State building failed (%d/%d bars).",
                         len(self._minute_bars), self.MIN_PREDICT_BARS)
            return

        logger.info("State shape=%s | min=%.4f | max=%.4f | mean=%.4f",
                    state.shape, state.min(), state.max(), state.mean())

        action_arr, _ = self.model.predict(state, deterministic=True)
        action = float(np.clip(action_arr.flat[0], -1.0, 1.0))

        price      = close[-1]
        max_units  = self.capital / price
        target     = action * max_units
        delta      = round(target - self._current_units)
        side       = "BUY" if delta >= 0 else "SELL"
        order      = MarketOrder(side, abs(delta))

        logger.info("action=%+.4f | price=%.5f | target=%+.1f EUR | Δ=%+d EUR",
                    action, price, target, delta)
        logger.info("Order → MarketOrder(%r, %d) | below IB min: %s",
                    order.action, int(order.totalQuantity),
                    abs(delta) < self.MIN_ORDER_UNITS)
        logger.info("=== DRY RUN COMPLETE ===")

    # -----------------------------------------------------------------------
    # Async internals
    # -----------------------------------------------------------------------

    async def _run_async(self) -> None:
        _BACKOFF = [30, 60, 120, 300]   # seconds between reconnect attempts
        attempt  = 0
        while True:
            try:
                await self._connect()
                mode = "WHATIF-ONLY" if self.whatif_only else "LIVE ORDERS"
                logger.info("Live trader running (%s). Ctrl+C to stop.", mode)
                attempt = 0
                await self._watch_connection()
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info("Shutdown requested — flattening position...")
                if not self.whatif_only:
                    await self._close_position()
                else:
                    logger.info("WhatIf-only mode — no real positions to close.")
                await self._disconnect()
                return
            except Exception as exc:
                logger.warning("Connection lost: %s — will reconnect.", exc)

            await self._disconnect()
            wait = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
            attempt += 1
            logger.info("Reconnecting in %ds (attempt %d)...", wait, attempt)
            await asyncio.sleep(wait)

    async def _watch_connection(self) -> None:
        """Poll every 60s; raise ConnectionError if no RT bar for 20 minutes."""
        STALE_SECS = 20 * 60
        while True:
            await asyncio.sleep(60)
            if self._last_bar_time is None:
                continue
            age = (datetime.now(tz=timezone.utc) - self._last_bar_time).total_seconds()
            if age > STALE_SECS:
                raise ConnectionError(
                    f"No RT bars for {age / 60:.0f} min — connection presumed dead."
                )

    async def _run_close_async(self) -> None:
        """Connect, sync, close open position, then disconnect. No trading loop."""
        try:
            await self.ib.connectAsync(self.TWS_HOST, self.TWS_PORT, clientId=self.CLIENT_ID)
            await self.ib.qualifyContractsAsync(self.contract)
            logger.info("Connected to TWS for position close.")
            await self._sync_position()
            await self._close_position()
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
            logger.info("Disconnected.")

    async def _connect(self) -> None:
        await self.ib.connectAsync(self.TWS_HOST, self.TWS_PORT, clientId=self.CLIENT_ID)
        await self.ib.qualifyContractsAsync(self.contract)
        logger.info("Contract qualified: %s", self.contract)
        self.ib.positionEvent += self._on_position_event
        await self._warm_up()
        await self._sync_position()
        self._rt_bars = self.ib.reqRealTimeBars(
            self.contract, barSize=5, whatToShow="MIDPOINT", useRTH=False,
        )
        self._rt_bars.updateEvent += self._on_rt_bar_update
        logger.info("Subscribed to EUR.USD 5s real-time bars.")

    async def _disconnect(self) -> None:
        if self._rt_bars:
            self.ib.cancelRealTimeBars(self._rt_bars)
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("Disconnected.")

    async def _warm_up(self) -> None:
        logger.info("Loading %d historical 1-min bars for warm-up...", self.WARMUP_BARS)
        self._minute_bars.clear()
        self._rt_bar_accum  = []
        self._rt_bar_minute = None
        self._prev_close    = None
        bars = await self.ib.reqHistoricalDataAsync(
            self.contract, endDateTime="",
            durationStr="2 D",
            barSizeSetting="1 min",
            whatToShow="MIDPOINT",
            useRTH=False, formatDate=1,
        )
        for bar in bars[-self.WARMUP_BARS:]:
            self._minute_bars.append({
                "open":     bar.open,
                "high":     bar.high,
                "low":      bar.low,
                "close":    bar.close,
                "datetime": pd.Timestamp(bar.date),
            })
        if self._minute_bars:
            self._prev_close = self._minute_bars[-1]["close"]
        self._regime_guard.calibrate(list(self._minute_bars))
        logger.info("Warm-up: %d bars loaded.", len(self._minute_bars))

    def _on_position_event(self, position) -> None:
        """Update _current_units from IB's confirmed position after every fill."""
        c = position.contract
        if getattr(c, "symbol", "") == "EUR" and getattr(c, "secType", "") == "CASH":
            prev = self._current_units
            self._current_units = float(position.position)
            if abs(self._current_units - prev) > 1:
                logger.info(
                    "Position update from IB: %+.0f → %+.0f EUR",
                    prev, self._current_units,
                )

    async def _sync_position(self) -> None:
        for pos in self.ib.positions():
            c = pos.contract
            if getattr(c, "symbol", "") == "EUR" and getattr(c, "secType", "") == "CASH":
                self._current_units = float(pos.position)
                if self._prev_close and self._prev_close > 0:
                    self._current_action = float(
                        np.clip(self._current_units / (self.capital / self._prev_close), -1.0, 1.0)
                    )
                logger.info("IB position synced: %+.0f EUR (action≈%.4f)",
                            self._current_units, self._current_action)
                return
        logger.info("No existing EUR.USD position — starting flat.")

    async def _close_position(self) -> None:
        """Flatten any open EUR.USD position and wait up to 30s for fill."""
        # Re-query actual IB position — local tracking may drift from fills
        await self._sync_position()

        if abs(self._current_units) < self.MIN_ORDER_UNITS:
            logger.info(
                "Position already flat (%.0f EUR) — nothing to close.",
                self._current_units,
            )
            return

        side  = "SELL" if self._current_units > 0 else "BUY"
        qty   = abs(round(self._current_units))
        order = MarketOrder(action=side, totalQuantity=qty, tif="DAY")
        logger.info("CLOSING POSITION: %s %.0f EUR...", side, qty)

        trade = self.ib.placeOrder(self.contract, order)
        for _ in range(30):
            await asyncio.sleep(1.0)
            if trade.orderStatus.status == "Filled":
                logger.info(
                    "Position closed | filled=%.0f @ %.5f",
                    trade.orderStatus.filled,
                    trade.orderStatus.avgFillPrice,
                )
                self._current_units = 0.0
                return
        logger.warning(
            "Close order not confirmed filled within 30s — "
            "check TWS manually. orderId=%d", trade.order.orderId,
        )

    # -----------------------------------------------------------------------
    # RT bar accumulation → 1-minute candle
    # -----------------------------------------------------------------------

    def _on_rt_bar_update(self, bars: RealTimeBarList, hasNewBar: bool) -> None:
        if not hasNewBar:
            return
        self._last_bar_time = datetime.now(tz=timezone.utc)
        latest    = bars[-1]
        bar_min   = latest.time.replace(second=0, microsecond=0, tzinfo=timezone.utc)

        if self._rt_bar_minute is None:
            self._rt_bar_minute = bar_min
            self._rt_bar_accum  = []

        if bar_min > self._rt_bar_minute:
            if self._rt_bar_accum:
                candle = self._aggregate_to_minute(self._rt_bar_accum, self._rt_bar_minute)
                self._minute_bars.append(candle)
                logger.debug("1-min candle | O=%.5f H=%.5f L=%.5f C=%.5f",
                             candle["open"], candle["high"], candle["low"], candle["close"])
                asyncio.ensure_future(self._rebalance(candle["close"]))
            self._rt_bar_minute = bar_min
            self._rt_bar_accum  = []

        self._rt_bar_accum.append(latest)

    @staticmethod
    def _aggregate_to_minute(rt_bars: List, minute_dt: datetime) -> Dict:
        return {
            "open":     rt_bars[0].open_,
            "high":     max(b.high   for b in rt_bars),
            "low":      min(b.low    for b in rt_bars),
            "close":    rt_bars[-1].close,
            "datetime": pd.Timestamp(minute_dt),
        }

    # -----------------------------------------------------------------------
    # State construction — calls TradingEnv classmethods for exact parity
    # -----------------------------------------------------------------------

    def _build_state(self) -> Optional[np.ndarray]:
        if len(self._minute_bars) < self.MIN_PREDICT_BARS:
            return None

        buf = list(self._minute_bars)

        # Reconstruct DataFrame with DatetimeIndex (required by _build_features)
        df = pd.DataFrame(buf)
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = -1.0   # Forex MIDPOINT has no volume; _build_features ignores it
        df.index = pd.to_datetime(df["datetime"], utc=True)
        df = df[["open", "high", "low", "close", "volume"]]

        # Replicate TradingEnv feature pipeline exactly
        feat_df = TradingEnv._build_features(df)
        normed  = TradingEnv._apply_rolling_norm(feat_df)   # (n_bars, 11)

        # Take last WINDOW_SIZE rows of normalised features
        window = normed[-TradingEnv.WINDOW_SIZE:]
        if len(window) < TradingEnv.WINDOW_SIZE:
            return None

        flat = window.flatten()
        upnl = np.float32(np.tanh(self._unrealised_pnl * TradingEnv.UPNL_SCALE))
        obs  = np.append(flat, upnl).astype(np.float32)

        assert obs.shape == (TradingEnv.OBS_DIM,), f"Shape mismatch: {obs.shape}"
        return obs

    # -----------------------------------------------------------------------
    # Inference + order execution
    # -----------------------------------------------------------------------

    async def _rebalance(self, close_price: float) -> None:
        # Update PnL from the position held during the closed candle
        if self._prev_close and self._prev_close > 0:
            ret = (close_price - self._prev_close) / self._prev_close
            self._unrealised_pnl += self._current_action * ret
        self._prev_close = close_price

        equity = self.capital + self._unrealised_pnl * self.capital
        halt_triggered = self._drawdown_guard.update(equity)
        if halt_triggered:
            if not self.whatif_only:
                await self._close_position()
            else:
                logger.info("WhatIf-only mode — no real positions to close on halt.")
            return

        if self._drawdown_guard.is_halted():
            logger.info("DRAWDOWN GUARD | halted — bars accumulating, inference suspended.")
            return

        state = self._build_state()
        if state is None:
            logger.info("Buffer warming up (%d/%d bars).",
                        len(self._minute_bars), self.MIN_PREDICT_BARS)
            return

        action_arr, _ = self.model.predict(state, deterministic=True)
        action = float(np.clip(action_arr.flat[0], -1.0, 1.0))

        max_units    = self.capital / close_price
        scalar       = self._regime_guard.risk_scalar(self._minute_bars[-1])
        target_units = action * max_units * scalar
        delta_units  = round(target_units - self._current_units)

        logger.info(
            "Candle close | price=%.5f | action=%+.4f | "
            "target=%+.1f EUR | current=%+.1f EUR | Δ=%+d EUR | upnl=%+.6f",
            close_price, action, target_units,
            self._current_units, delta_units, self._unrealised_pnl,
        )

        if abs(delta_units) < 1:
            self._current_action = action
            return

        if abs(delta_units) < self.MIN_ORDER_UNITS:
            logger.warning("|Δ|=%d EUR < IB min %d EUR — skipped.",
                           abs(delta_units), self.MIN_ORDER_UNITS)
            self._current_action = action
            return

        side  = "BUY" if delta_units > 0 else "SELL"
        order = MarketOrder(action=side, totalQuantity=abs(delta_units), tif="DAY")

        if self.whatif_only:
            await self._what_if(order)
        else:
            self._place_live(order, delta_units)

        self._current_action = action

    async def _what_if(self, order: MarketOrder) -> None:
        try:
            result = await self.ib.whatIfOrderAsync(self.contract, order)
            if not result:
                logger.warning("WhatIf %s %d EUR | no response from IB (order preset override)",
                               order.action, int(order.totalQuantity))
                return
            s = result[0] if isinstance(result, list) else result
            logger.info(
                "WhatIf %s %d EUR | init_margin_Δ=%s | maint_margin_Δ=%s | commission≈%s %s",
                order.action, int(order.totalQuantity),
                s.initMarginChange, s.maintMarginChange,
                s.commission, s.commissionCurrency,
            )
        except Exception as exc:
            logger.error("whatIfOrderAsync failed: %s", exc)

    def _place_live(self, order: MarketOrder, delta_units: int) -> None:
        trade = self.ib.placeOrder(self.contract, order)
        self._current_units += delta_units
        logger.info("Order placed: %s", trade)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EUR.USD PPO Live Trader v2")
    parser.add_argument("--model",   default=str(MODEL_PATH))
    parser.add_argument("--capital", type=float, default=1_000.0)
    parser.add_argument("--live",    action="store_true",
                        help="Send real orders — DANGEROUS, paper account only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--close",   action="store_true",
                        help="Close open position and exit (no trading loop)")
    args = parser.parse_args()

    trader = LiveTrader(
        model_path=Path(args.model),
        capital=args.capital,
        whatif_only=not args.live,
    )

    if args.close:
        trader.close_position()
    elif args.dry_run:
        trader.dry_run()
    else:
        trader.run()
