"""
Live execution bridge: ib_insync real-time bars → PPO inference → IB orders.

Flow (every 1-minute candle close)
───────────────────────────────────
  reqRealTimeBars (5s MIDPOINT) ─► accumulate 12 × 5s bars
  ─► aggregate into 1-min OHLCV ─► append to rolling bar buffer
  ─► replicate TradingEnv feature pipeline (MACD, RSI) + Z-score normalise
  ─► model.predict(state, deterministic=True)  →  action ∈ [-1, 1]
  ─► compute target EUR units from capital; delta vs current position
  ─► ib.whatIfOrderAsync()  (whatif_only=True, default)
     or ib.placeOrder()     (live / paper account)

CRITICAL: patchAsyncio() is called at module import — do not remove.
NEVER execute live trades without explicit human confirmation.
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

MODEL_PATH = ROOT / "models" / "saved" / "ppo_eurusd_final"
NORM_STATS_PATH = ROOT / "models" / "saved" / "norm_stats.npz"

# ── Feature columns must match TradingEnv exactly ──────────────────────────
_FEAT_COLS = ["open", "high", "low", "close", "volume", "macd_hist", "rsi"]


# ---------------------------------------------------------------------------
# LiveTrader
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Connects to TWS, subscribes to EUR.USD real-time bars, runs PPO inference
    on every 1-minute candle, and rebalances the position via whatIfOrder or
    placeOrder.

    Parameters
    ----------
    model_path      : Path to the SB3 PPO .zip (without extension).
    norm_stats_path : Path to norm_stats.npz produced by models/train.py.
    capital         : Notional capital limit in USD (default $1,000).
    whatif_only     : If True (default), use whatIfOrder — never touches money.
    """

    TWS_HOST: str = "127.0.0.1"
    TWS_PORT: int = 7497
    CLIENT_ID: int = 3            # distinct from data client (1) and TWS default (0)

    WARMUP_BARS: int = 200        # 1-min historical bars loaded at startup
    # Enough bars for a warm MACD (26+9=35) before the first 50-bar window
    MIN_PREDICT_BARS: int = TradingEnv.WINDOW_SIZE + 40

    # IB Forex minimum order (base currency). 1,000 EUR for EUR.USD on IDEALPRO.
    # With $1,000 capital, delta will usually fall below this minimum — the code
    # will log a skip rather than error. Raise capital or lower this for testing.
    MIN_ORDER_UNITS: int = 1_000

    # 5-second real-time bars to aggregate per 1-minute candle
    _RT_BARS_PER_MINUTE: int = 12

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        norm_stats_path: Path = NORM_STATS_PATH,
        capital: float = 1_000.0,
        whatif_only: bool = True,
    ) -> None:
        self.capital = capital
        self.whatif_only = whatif_only

        # ── PPO model ──────────────────────────────────────────────────
        logger.info("Loading PPO model from %s", model_path)
        self.model = PPO.load(str(model_path))

        # ── Normalization stats (must exactly match training env) ──────
        stats = np.load(str(norm_stats_path))
        self._means: np.ndarray = stats["means"]   # shape (7,)
        self._stds: np.ndarray = stats["stds"]     # shape (7,)
        logger.info(
            "Norm stats loaded | means=%s | stds=%s",
            np.round(self._means, 6),
            np.round(self._stds, 6),
        )

        # ── IB connection ──────────────────────────────────────────────
        self.ib = IB()
        self.contract = Forex("EURUSD")

        # ── Bar buffers ────────────────────────────────────────────────
        # 1-minute OHLCV bars (historical + live aggregated)
        self._minute_bars: deque[Dict] = deque(maxlen=self.WARMUP_BARS + 50)
        # Accumulate 5-second RT bars until a minute boundary is crossed
        self._rt_bar_accum: List = []
        self._rt_bar_minute: Optional[datetime] = None

        # ── Position & PnL state ───────────────────────────────────────
        # normalised allocation ∈ [-1, 1] — mirrors TradingEnv._position
        self._current_action: float = 0.0
        # actual EUR units held (+ = long, - = short)
        self._current_units: float = 0.0
        # running PnL in return-space (matches TradingEnv._unrealised_pnl)
        self._unrealised_pnl: float = 0.0
        self._prev_close: Optional[float] = None

        # Active RT bar subscription handle
        self._rt_bars: Optional[RealTimeBarList] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Synchronous entry point — runs the async event loop until Ctrl+C."""
        asyncio.run(self._run_async())

    def dry_run(self, n_bars: int = 200) -> None:
        """
        Test the full inference pipeline without connecting to TWS.

        Fills the bar buffer with synthetic GBM data, builds the state tensor,
        runs model.predict(), computes the order delta, and prints the
        MarketOrder object that *would* be sent to IB.
        """
        logger.info("=== DRY RUN (no TWS connection) ===")

        # Generate synthetic EUR/USD-like 1-min bars
        rng = np.random.default_rng(99)
        log_ret = rng.normal(0.0, 0.0001, n_bars)
        close = 1.1000 * np.cumprod(1.0 + log_ret)
        noise = rng.uniform(5e-5, 2e-4, n_bars)
        opens = np.concatenate([[close[0]], close[:-1]])

        self._minute_bars.clear()
        for i in range(n_bars):
            self._minute_bars.append({
                "open":   float(opens[i]),
                "high":   float(close[i] + noise[i]),
                "low":    float(close[i] - noise[i]),
                "close":  float(close[i]),
                "volume": float(rng.integers(100, 1_000)),
            })

        current_price = close[-1]
        self._prev_close = close[-2]

        state = self._build_state()
        if state is None:
            logger.error(
                "State building failed — buffer has %d bars, need %d.",
                len(self._minute_bars), self.MIN_PREDICT_BARS,
            )
            return

        logger.info(
            "State tensor | shape=%s | min=%.4f | max=%.4f | mean=%.4f",
            state.shape, state.min(), state.max(), state.mean(),
        )

        action_arr, _ = self.model.predict(state, deterministic=True)
        action = float(np.clip(action_arr.flat[0], -1.0, 1.0))

        max_units = self.capital / current_price
        target_units = action * max_units
        delta_units = round(target_units - self._current_units)
        side = "BUY" if delta_units >= 0 else "SELL"
        order = MarketOrder(side, abs(delta_units))

        logger.info(
            "Prediction | price=%.5f | action=%+.4f | "
            "target=%+.1f EUR | current=%+.1f EUR | delta=%+.1f EUR",
            current_price, action, target_units,
            self._current_units, delta_units,
        )
        logger.info(
            "Order object → MarketOrder(action='%s', totalQuantity=%d) | "
            "below IB min lot (%d EUR): %s",
            order.action, order.totalQuantity,
            self.MIN_ORDER_UNITS,
            abs(delta_units) < self.MIN_ORDER_UNITS,
        )

        logger.info(
            "WhatIf simulation skipped in dry_run — "
            "requires live TWS connection. State + order construction: OK."
        )
        logger.info("=== DRY RUN COMPLETE ===")

    # -----------------------------------------------------------------------
    # Async internals
    # -----------------------------------------------------------------------

    async def _run_async(self) -> None:
        try:
            await self._connect()
            mode = "WHATIF-ONLY (paper simulation)" if self.whatif_only else "LIVE ORDERS"
            logger.info("Live trader running in %s mode. Press Ctrl+C to stop.", mode)
            await asyncio.sleep(float("inf"))
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutdown signal received.")
        finally:
            await self._disconnect()

    async def _connect(self) -> None:
        logger.info("Connecting to TWS at %s:%d (clientId=%d)",
                    self.TWS_HOST, self.TWS_PORT, self.CLIENT_ID)
        await self.ib.connectAsync(self.TWS_HOST, self.TWS_PORT, clientId=self.CLIENT_ID)

        await self.ib.qualifyContractsAsync(self.contract)
        logger.info("Contract qualified: %s", self.contract)

        await self._warm_up()
        await self._sync_position()

        # Subscribe to 5-second MIDPOINT real-time bars
        self._rt_bars = self.ib.reqRealTimeBars(
            contract=self.contract,
            barSize=5,
            whatToShow="MIDPOINT",
            useRTH=False,
        )
        self._rt_bars.updateEvent += self._on_rt_bar_update
        logger.info("Subscribed to EUR.USD real-time bars (5s MIDPOINT).")

    async def _disconnect(self) -> None:
        if self._rt_bars is not None:
            self.ib.cancelRealTimeBars(self._rt_bars)
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("Disconnected from TWS.")

    async def _warm_up(self) -> None:
        """
        Load the last WARMUP_BARS of 1-minute historical data into the bar buffer.
        This ensures MACD and RSI are fully warmed up before the first live candle.
        """
        logger.info("Loading %d historical 1-min bars for warm-up...", self.WARMUP_BARS)
        bars = await self.ib.reqHistoricalDataAsync(
            self.contract,
            endDateTime="",
            durationStr=f"{self.WARMUP_BARS + 10} B",
            barSizeSetting="1 min",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
        )
        for bar in bars[-self.WARMUP_BARS:]:
            self._minute_bars.append({
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            })
        if self._minute_bars:
            self._prev_close = self._minute_bars[-1]["close"]
        logger.info("Warm-up complete: %d bars loaded.", len(self._minute_bars))

    async def _sync_position(self) -> None:
        """Reconcile internal position state with the actual IB account position."""
        positions = self.ib.positions()
        for pos in positions:
            c = pos.contract
            if getattr(c, "symbol", "") == "EUR" and getattr(c, "secType", "") == "CASH":
                self._current_units = float(pos.position)
                # Estimate normalised action from units (approximate; current price needed)
                if self._prev_close and self._prev_close > 0:
                    max_u = self.capital / self._prev_close
                    self._current_action = (
                        np.clip(self._current_units / max_u, -1.0, 1.0) if max_u > 0 else 0.0
                    )
                logger.info(
                    "IB position synced: %+.0f EUR (action≈%.4f)",
                    self._current_units, self._current_action,
                )
                return
        logger.info("No existing EUR.USD position — starting flat.")

    # -----------------------------------------------------------------------
    # Real-time bar accumulation → 1-minute candle
    # -----------------------------------------------------------------------

    def _on_rt_bar_update(self, bars: RealTimeBarList, hasNewBar: bool) -> None:
        """
        Fires on every 5-second bar update.  Accumulates bars until a wall-clock
        minute boundary is crossed, then closes the 1-minute candle and
        schedules the async rebalance coroutine.
        """
        if not hasNewBar:
            return

        latest = bars[-1]
        # bar.time is a datetime in the exchange timezone; normalise to UTC minute
        bar_minute = latest.time.replace(second=0, microsecond=0, tzinfo=timezone.utc)

        if self._rt_bar_minute is None:
            # First bar received — open first candle bucket
            self._rt_bar_minute = bar_minute
            self._rt_bar_accum = []

        if bar_minute > self._rt_bar_minute:
            # Minute boundary crossed — close the accumulated candle
            if self._rt_bar_accum:
                candle = self._aggregate_to_minute(self._rt_bar_accum)
                self._minute_bars.append(candle)
                logger.debug(
                    "1-min candle closed | O=%.5f H=%.5f L=%.5f C=%.5f V=%.0f",
                    candle["open"], candle["high"], candle["low"],
                    candle["close"], candle["volume"],
                )
                # Schedule async rebalance without blocking the IB event loop
                asyncio.ensure_future(self._rebalance(candle["close"]))

            # Reset accumulator for the new minute
            self._rt_bar_minute = bar_minute
            self._rt_bar_accum = []

        self._rt_bar_accum.append(latest)

    @staticmethod
    def _aggregate_to_minute(rt_bars: List) -> Dict:
        """Collapse a list of 5-second RealTimeBars into a single 1-minute OHLCV dict."""
        return {
            "open":   rt_bars[0].open_,
            "high":   max(b.high for b in rt_bars),
            "low":    min(b.low for b in rt_bars),
            "close":  rt_bars[-1].close,
            "volume": sum(b.volume for b in rt_bars),
        }

    # -----------------------------------------------------------------------
    # State construction (must replicate TradingEnv._build_features exactly)
    # -----------------------------------------------------------------------

    def _build_state(self) -> Optional[np.ndarray]:
        """
        Construct the (351,) observation vector from the minute-bar buffer.

        Steps
        -----
        1. Convert buffer to DataFrame.
        2. Compute MACD histogram and Wilder RSI using the same EWM parameters
           as TradingEnv._build_features().
        3. Z-score normalise with the saved training means/stds.
        4. Slice the last WINDOW_SIZE (50) rows.
        5. Flatten to (350,) and append tanh-bounded unrealised PnL → (351,).

        Returns None if the buffer is not yet warm enough.
        """
        if len(self._minute_bars) < self.MIN_PREDICT_BARS:
            return None

        df = pd.DataFrame(list(self._minute_bars))[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        # ── MACD histogram ────────────────────────────────────────────
        ema_fast = df["close"].ewm(span=TradingEnv.MACD_FAST, adjust=False).mean()
        ema_slow = df["close"].ewm(span=TradingEnv.MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=TradingEnv.MACD_SIGNAL, adjust=False).mean()
        df["macd_hist"] = macd_line - signal_line

        # ── Wilder RSI ────────────────────────────────────────────────
        delta = df["close"].diff()
        avg_gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / TradingEnv.RSI_PERIOD, adjust=False
        ).mean()
        avg_loss = (-delta).clip(lower=0.0).ewm(
            alpha=1.0 / TradingEnv.RSI_PERIOD, adjust=False
        ).mean()
        rs = avg_gain / avg_loss.replace(0.0, 1e-8)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        df = df.dropna()

        # Take the last WINDOW_SIZE rows as the observation window
        window = df[_FEAT_COLS].tail(TradingEnv.WINDOW_SIZE)
        if len(window) < TradingEnv.WINDOW_SIZE:
            return None

        # ── Z-score normalise with training stats ─────────────────────
        raw = window.values.astype(np.float64)
        normed = ((raw - self._means) / self._stds).astype(np.float32)

        # ── Assemble final observation ────────────────────────────────
        flat = normed.flatten()                                        # (350,)
        upnl = np.float32(np.tanh(self._unrealised_pnl * TradingEnv.UPNL_SCALE))
        obs = np.append(flat, upnl).astype(np.float32)                # (351,)

        assert obs.shape == (TradingEnv.OBS_DIM,), (
            f"State shape mismatch: got {obs.shape}, expected ({TradingEnv.OBS_DIM},)"
        )
        return obs

    # -----------------------------------------------------------------------
    # Inference + order execution
    # -----------------------------------------------------------------------

    async def _rebalance(self, close_price: float) -> None:
        """
        Called on every 1-minute candle close:
          1. Update unrealised PnL with position that was held during the candle.
          2. Build state tensor.
          3. Run model.predict() to get target allocation.
          4. Compute delta between target and current position in EUR units.
          5. Submit whatIfOrder (or live order if whatif_only=False).
        """
        # PnL accrues from the allocation held during the *previous* candle
        if self._prev_close is not None and self._prev_close > 0:
            price_return = (close_price - self._prev_close) / self._prev_close
            self._unrealised_pnl += self._current_action * price_return
        self._prev_close = close_price

        # Build observation
        state = self._build_state()
        if state is None:
            logger.info(
                "Buffer warming up (%d/%d bars) — skipping prediction.",
                len(self._minute_bars), self.MIN_PREDICT_BARS,
            )
            return

        # Inference
        action_arr, _ = self.model.predict(state, deterministic=True)
        action = float(np.clip(action_arr.flat[0], -1.0, 1.0))

        # ── Position sizing ────────────────────────────────────────────
        # max_units: how many EUR we can hold if fully allocated to one side
        max_units: float = self.capital / close_price
        target_units: float = action * max_units
        delta_units: int = round(target_units - self._current_units)

        logger.info(
            "Candle close | price=%.5f | action=%+.4f | "
            "target=%+.1f EUR | current=%+.1f EUR | Δ=%+d EUR | "
            "unrealised_pnl=%+.6f",
            close_price, action, target_units,
            self._current_units, delta_units,
            self._unrealised_pnl,
        )

        # Skip if the rebalance is below IB minimum lot or negligible
        if abs(delta_units) < 1:
            logger.info("Δ=0 EUR — position already at target, no order needed.")
            self._current_action = action
            return

        if abs(delta_units) < self.MIN_ORDER_UNITS:
            logger.warning(
                "|Δ|=%d EUR is below IB minimum lot (%d EUR). "
                "Order skipped — increase capital or reduce MIN_ORDER_UNITS.",
                abs(delta_units), self.MIN_ORDER_UNITS,
            )
            # Still update the logical action so PnL tracking stays consistent
            self._current_action = action
            return

        side = "BUY" if delta_units > 0 else "SELL"
        order = MarketOrder(action=side, totalQuantity=abs(delta_units))

        if self.whatif_only:
            await self._what_if(order)
        else:
            self._place_live(order, delta_units)

        self._current_action = action

    async def _what_if(self, order: MarketOrder) -> None:
        """
        Submit a whatIfOrder to IB — simulates margin/commission without
        executing the trade.  Safe to use on both paper and live accounts.
        """
        try:
            state = await self.ib.whatIfOrderAsync(self.contract, order)
            logger.info(
                "WhatIf %s %d EUR | "
                "init_margin_Δ=%s | maint_margin_Δ=%s | commission≈%s %s",
                order.action,
                int(order.totalQuantity),
                state.initMarginChange,
                state.maintMarginChange,
                state.commission,
                state.commissionCurrency,
            )
        except Exception as exc:
            logger.error("whatIfOrderAsync failed: %s", exc)

    def _place_live(self, order: MarketOrder, delta_units: int) -> None:
        """Place a real market order (paper or live account, NOT whatif)."""
        trade = self.ib.placeOrder(self.contract, order)
        self._current_units += delta_units
        logger.info("Order placed: %s", trade)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EUR.USD PPO Live Trader")
    parser.add_argument("--model",  default=str(MODEL_PATH),      help="Path to .zip model")
    parser.add_argument("--stats",  default=str(NORM_STATS_PATH), help="Path to norm_stats.npz")
    parser.add_argument("--capital", type=float, default=1_000.0, help="Notional capital (USD)")
    parser.add_argument("--live",   action="store_true",          help="Disable whatif — send real orders (DANGEROUS)")
    parser.add_argument("--dry-run", action="store_true",         help="Test pipeline without connecting to TWS")
    args = parser.parse_args()

    trader = LiveTrader(
        model_path=args.model,
        norm_stats_path=args.stats,
        capital=args.capital,
        whatif_only=not args.live,
    )

    if args.dry_run:
        trader.dry_run()
    else:
        trader.run()
