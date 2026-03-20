"""
CryptoCurrency.cv news source — free, no API key required.
662,000+ articles from 2017–2025 (CryptoPanic + 100 other sources).
GitHub: https://github.com/nirholas/free-crypto-news

Supports historical fetch by date + ticker for backtesting.
"""
import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

from tatc.core.models import NewsItem

_BASE_URL = "https://cryptocurrency.cv"
_ARCHIVE_ENDPOINT = "/api/archive"


def _parse_item(raw: dict, idx: int) -> NewsItem | None:
    """Parse a single article from the API response."""
    # Handle both possible field names
    title = raw.get("title") or raw.get("headline") or ""
    if not title:
        return None

    # Timestamp — try multiple fields
    ts_str = (raw.get("pubDate") or raw.get("published_at") or
              raw.get("created_at") or raw.get("date") or "")
    try:
        if ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            return None
    except ValueError:
        return None

    # Coin tickers
    coins = []
    for field in ("tickers", "currencies", "coins", "symbols"):
        val = raw.get(field)
        if isinstance(val, list):
            coins = [str(c).upper() if isinstance(c, str) else str(c.get("code", "")).upper()
                     for c in val if c]
            break
        elif isinstance(val, str) and val:
            coins = [val.upper()]
            break

    body = raw.get("body") or raw.get("content") or raw.get("summary") or ""
    source = raw.get("source") or raw.get("domain") or "cryptocurrency.cv"
    if isinstance(source, dict):
        source = source.get("title") or source.get("name") or "cryptocurrency.cv"
    url = raw.get("url") or raw.get("link") or ""
    item_id = str(raw.get("id") or raw.get("_id") or f"ccv-{idx}")

    return NewsItem(
        id=item_id,
        timestamp=ts,
        title=title,
        body=str(body)[:2000],
        source=str(source),
        url=str(url),
        coins=coins,
    )


class CryptoCurrencyCVSource:
    """
    Fetches historical news from cryptocurrency.cv (no API key needed).
    Dataset: 662k+ articles, 2017–2025.
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

    async def _fetch_month(
        self,
        year: int,
        month: int,
        ticker: str | None = None,
    ) -> list[NewsItem]:
        """Fetch all articles for a given month, optionally filtered by ticker."""
        params: dict = {"date": f"{year}-{month:02d}"}
        if ticker:
            params["ticker"] = ticker.upper()

        try:
            async with self._session.get(
                _BASE_URL + _ARCHIVE_ENDPOINT,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    print(f"  [ccv] HTTP {resp.status} for {params}")
                    return []
                data = await resp.json(content_type=None)
        except Exception as e:
            print(f"  [ccv] fetch error for {params}: {e}")
            return []

        # Response may be a list or dict with results key
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = (data.get("results") or data.get("articles") or
                    data.get("data") or data.get("items") or [])
        else:
            return []

        items = []
        for i, row in enumerate(rows):
            item = _parse_item(row, i)
            if item:
                items.append(item)

        return items

    async def fetch_historical(
        self,
        start: str,
        end: str,
        coins: list[str] | None = None,
    ) -> list[NewsItem]:
        """
        Fetch historical news between start and end ('YYYY-MM-DD').
        If coins given, fetches each ticker separately and merges.
        If coins is None or empty, fetches all news (no ticker filter).
        """
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        # Build list of (year, month) to fetch
        months = []
        y, m = start_dt.year, start_dt.month
        while (y, m) <= (end_dt.year, end_dt.month):
            months.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1

        tickers = coins if coins else [None]
        all_items: list[NewsItem] = []
        seen_ids: set[str] = set()

        for ticker in tickers:
            label = ticker or "all"
            print(f"  [ccv] fetching {label}: {len(months)} months...", end=" ", flush=True)
            for year, month in months:
                items = await self._fetch_month(year, month, ticker)
                for item in items:
                    if item.id not in seen_ids:
                        # Filter to exact date range
                        if start_dt <= item.timestamp <= end_dt:
                            seen_ids.add(item.id)
                            all_items.append(item)
                await asyncio.sleep(1.1)  # rate limit: 1 req/sec
            print(f"done ({len(all_items)} total so far)")

        all_items.sort(key=lambda x: x.timestamp)
        return all_items

    async def stream(self, poll_interval: float = 5.0) -> AsyncIterator[NewsItem]:
        """Live polling stream (for future use)."""
        seen_ids: set[str] = set()
        while True:
            try:
                params = {"limit": 50}
                async with self._session.get(
                    _BASE_URL + _ARCHIVE_ENDPOINT, params=params
                ) as resp:
                    data = await resp.json(content_type=None)
                rows = data if isinstance(data, list) else data.get("results", [])
                for i, row in enumerate(reversed(rows)):
                    item = _parse_item(row, i)
                    if item and item.id not in seen_ids:
                        seen_ids.add(item.id)
                        yield item
            except Exception as e:
                print(f"[ccv] stream error: {e}")
            await asyncio.sleep(poll_interval)
