"""
Bybit price source — historical OHLCV via REST API.
Uses Bybit V5 public API (no auth required for market data).
"""
import asyncio
from datetime import datetime, timezone

import aiohttp

from tatc.core.models import Candle

# Bybit V5 REST base
_BASE_URL = "https://api.bybit.com"
_KLINE_ENDPOINT = "/v5/market/kline"
_TICKER_ENDPOINT = "/v5/market/tickers"

# Max candles per request
_MAX_LIMIT = 200


def _to_ms(dt_str: str) -> int:
    """Convert 'YYYY-MM-DD' or ISO string to milliseconds timestamp."""
    if "T" in dt_str or " " in dt_str:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_candle(symbol: str, raw: list) -> Candle:
    """Parse Bybit kline response row: [startTime, open, high, low, close, volume, turnover]"""
    return Candle(
        symbol=symbol,
        timestamp=datetime.fromtimestamp(int(raw[0]) / 1000, tz=timezone.utc),
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
    )


class BybitPriceSource:
    """
    Fetches historical OHLCV candles from Bybit V5 REST API.
    Handles pagination automatically.
    """

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._owns_session and self._session:
            await self._session.close()

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> list[Candle]:
        """
        Fetch all candles for symbol between start and end.
        interval: '1' (1m), '5', '15', '60' (1h), '240' (4h), 'D', 'W'
        start/end: 'YYYY-MM-DD' or ISO format
        """
        start_ms = _to_ms(start)
        end_ms = _to_ms(end)
        all_candles: list[Candle] = []

        current_start = start_ms
        while current_start < end_ms:
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": interval,
                "start": current_start,
                "end": end_ms,
                "limit": _MAX_LIMIT,
            }

            async with self._session.get(
                _BASE_URL + _KLINE_ENDPOINT, params=params
            ) as resp:
                data = await resp.json()

            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

            rows = data["result"]["list"]
            if not rows:
                break

            candles = [_parse_candle(symbol, r) for r in rows]
            # Bybit returns newest first — reverse to chronological
            candles.sort(key=lambda c: c.timestamp)
            all_candles.extend(candles)

            # Move window forward
            last_ts = int(rows[0][0])  # rows[0] is newest before reverse
            if last_ts <= current_start:
                break
            current_start = last_ts + 1

            if len(rows) < _MAX_LIMIT:
                break

            await asyncio.sleep(0.05)  # gentle rate limiting

        # Deduplicate and sort
        seen = set()
        unique = []
        for c in sorted(all_candles, key=lambda c: c.timestamp):
            if c.timestamp not in seen:
                seen.add(c.timestamp)
                unique.append(c)

        return unique

    async def get_price(self, symbol: str) -> float:
        """Get last traded price for symbol."""
        params = {"category": "spot", "symbol": symbol}
        async with self._session.get(
            _BASE_URL + _TICKER_ENDPOINT, params=params
        ) as resp:
            data = await resp.json()

        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

        return float(data["result"]["list"][0]["lastPrice"])
