"""
Asynchronous IB data pipeline for fetching historical OHLCV data via ib_insync.
"""

import ib_insync.util
ib_insync.util.patchAsyncio()

import asyncio
import logging
from pathlib import Path
from datetime import datetime

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

    TWS_HOST = "127.0.0.1"
    TWS_PORT = 7497
    CLIENT_ID = 1

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
            self.TWS_HOST,
            self.TWS_PORT,
            self.CLIENT_ID,
        )

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from TWS.")

    async def fetch_eurusd_ohlcv(
        self,
        duration: str = "5 D",
        bar_size: str = "1 min",
    ) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV bars for EUR.USD spot Forex.

        Args:
            duration:  TWS duration string  (e.g. "5 D", "1 W").
            bar_size:  TWS bar size string  (e.g. "1 min", "5 mins").

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, barCount, average.
        """
        contract = Forex("EURUSD")
        await self.ib.qualifyContractsAsync(contract)
        logger.info("Contract qualified: %s", contract)

        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",          # empty → up to now
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT",    # Forex has no volume; MIDPOINT gives clean OHLC
            useRTH=False,             # include extended / 24-h Forex sessions
            formatDate=1,
        )

        if not bars:
            raise RuntimeError(
                "No bars returned. Verify TWS is running, market data subscriptions "
                "are active, and the contract is correct."
            )

        df = self._bars_to_dataframe(bars)
        logger.info("Fetched %d bars (%s → %s)", len(df), df.index[0], df.index[-1])
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_dataframe(bars: list[BarData]) -> pd.DataFrame:
        records = [
            {
                "date":     bar.date,
                "open":     bar.open,
                "high":     bar.high,
                "low":      bar.low,
                "close":    bar.close,
                "volume":   bar.volume,
                "barCount": bar.barCount,
                "average":  bar.average,
            }
            for bar in bars
        ]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df


async def run_pipeline(output_csv: Path | None = None) -> pd.DataFrame:
    """End-to-end coroutine: connect → fetch → save → disconnect."""
    client = IBDataClient()
    try:
        await client.connect()
        df = await client.fetch_eurusd_ohlcv(duration="5 D", bar_size="1 min")

        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = OUTPUT_DIR / f"EURUSD_1min_{timestamp}.csv"

        df.to_csv(output_csv)
        logger.info("Data saved → %s", output_csv)
        return df
    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
