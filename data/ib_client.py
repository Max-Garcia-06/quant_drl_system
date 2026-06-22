"""
Asynchronous IB data pipeline for fetching historical OHLCV data via ib_insync.
Supports multi-month chunked requests to pull up to 1 year of 1-min Forex data.
"""

import ib_insync.util
ib_insync.util.patchAsyncio()

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from ib_insync import IB, Forex, BarData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class IBDataClient:
    """Connects to TWS and fetches historical OHLCV bar data via ib_insync."""

    TWS_HOST  = "127.0.0.1"
    TWS_PORT  = 7497
    CLIENT_ID = 1

    # Seconds to wait between monthly chunk requests to respect IB pacing limits
    CHUNK_PAUSE = 3.0

    def __init__(self) -> None:
        self.ib = IB()

    async def connect(self) -> None:
        await self.ib.connectAsync(
            host=self.TWS_HOST,
            port=self.TWS_PORT,
            clientId=self.CLIENT_ID,
        )
        logger.info(
            "Connected to TWS at %s:%s (clientId=%s)",
            self.TWS_HOST, self.TWS_PORT, self.CLIENT_ID,
        )

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from TWS.")

    async def fetch_eurusd_ohlcv(
        self,
        duration: str = "5 D",
        bar_size: str = "1 min",
        end_dt: str   = "",
    ) -> pd.DataFrame:
        """Single-request historical fetch (short durations — days/weeks)."""
        contract = Forex("EURUSD")
        await self.ib.qualifyContractsAsync(contract)
        logger.info("Contract qualified: %s", contract)

        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
        )
        if not bars:
            raise RuntimeError(
                "No bars returned. Verify TWS is running, market data "
                "subscriptions are active, and the contract is correct."
            )
        df = self._bars_to_dataframe(bars)
        logger.info("Fetched %d bars (%s → %s)", len(df), df.index[0], df.index[-1])
        return df

    async def fetch_eurusd_long_history(self, n_months: int = 6) -> pd.DataFrame:
        """
        Pull n_months of 1-minute EUR.USD MIDPOINT data by making sequential
        monthly requests.

        IB limits each historical data request; chunking by month stays well
        within those limits.  A 3-second pause between requests respects IB's
        pacing rules (max 60 requests per 10 minutes).

        Returns a deduplicated, chronologically sorted DataFrame with a
        timezone-aware DatetimeIndex.
        """
        contract = Forex("EURUSD")
        await self.ib.qualifyContractsAsync(contract)
        logger.info("Fetching %d months of 1-min EUR.USD data in chunks...", n_months)

        all_bars: list[BarData] = []
        end_dt: str = ""  # empty = now

        n_weeks = n_months * 4          # ~4 weeks per month
        for chunk in range(n_weeks):
            logger.info("  Chunk %d/%d (endDateTime=%r)...", chunk + 1, n_weeks, end_dt or "now")
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr="1 W",          # 1 week per request — stays within IB limits
                barSizeSetting="1 min",
                whatToShow="MIDPOINT",
                useRTH=False,
                formatDate=1,
                timeout=120,                # raise from default 60s
            )
            if not bars:
                logger.warning("  Chunk %d returned 0 bars — stopping early.", chunk + 1)
                break

            # Prepend so final list is chronologically ordered
            all_bars = list(bars) + all_bars

            # Next chunk ends just before the start of this one
            first_bar_dt = bars[0].date   # tz-aware datetime (formatDate=1)
            end_dt = first_bar_dt.strftime("%Y%m%d %H:%M:%S")
            logger.info(
                "  Got %d bars | oldest: %s | newest: %s",
                len(bars), bars[0].date, bars[-1].date,
            )

            if chunk < n_weeks - 1:
                await asyncio.sleep(self.CHUNK_PAUSE)

        if not all_bars:
            raise RuntimeError("No historical bars returned across all chunks.")

        df = self._bars_to_dataframe(all_bars)
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < before:
            logger.info("Removed %d duplicate bars at chunk boundaries.", before - len(df))

        logger.info(
            "Total: %d bars | %s → %s",
            len(df), df.index[0], df.index[-1],
        )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_dataframe(bars: list[BarData]) -> pd.DataFrame:
        records = [
            {
                "date":   bar.date,
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df


async def run_pipeline(n_months: int = 6, output_csv: Path | None = None) -> pd.DataFrame:
    """End-to-end coroutine: connect → fetch n_months → save → disconnect."""
    client = IBDataClient()
    try:
        await client.connect()
        df = await client.fetch_eurusd_long_history(n_months=n_months)

        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = OUTPUT_DIR / f"EURUSD_1min_{n_months}M_{timestamp}.csv"

        df.to_csv(output_csv)
        logger.info("Data saved → %s", output_csv)
        return df
    finally:
        client.disconnect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch")
    args = parser.parse_args()
    asyncio.run(run_pipeline(n_months=args.months))
