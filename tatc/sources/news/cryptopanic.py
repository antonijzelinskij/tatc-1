"""
CryptoPanic news source.
Uses the CryptoPanic API which provides timestamped crypto news — ideal for backtesting.
API key is free at https://cryptopanic.com/developers/api/

For backtesting: fetch_historical() pages through the API collecting news in a date range.
For live: stream() polls the API every few seconds for new items.
"""
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

from tatc.core.models import NewsItem

_BASE_URL = "https://cryptopanic.com/api/v1/posts/"


def _parse_item(raw: dict) -> NewsItem:
    """Parse a single CryptoPanic post into NewsItem."""
    # Extract mentioned coin slugs (UPPER)
    coins = [c["code"].upper() for c in raw.get("currencies", [])]

    # published_at format: "2024-01-15T10:23:00Z"
    ts_str = raw.get("published_at", raw.get("created_at", ""))
    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

    title = raw.get("title", "")
    body = raw.get("body") or ""

    return NewsItem(
        id=str(raw["id"]),
        timestamp=ts,
        title=title,
        body=body,
        source=raw.get("source", {}).get("title", "cryptopanic"),
        url=raw.get("url", ""),
        coins=coins,
    )


class CryptoPanicSource:
    """
    Fetches news from CryptoPanic API.
    Supports both historical batch fetch (for backtesting) and live polling.
    """

    def __init__(self, api_key: str, session: aiohttp.ClientSession | None = None):
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._owns_session and self._session:
            await self._session.close()

    async def _fetch_page(self, params: dict) -> tuple[list[NewsItem], str | None]:
        """Fetch one page of results. Returns (items, next_url)."""
        params = {**params, "auth_token": self._api_key, "public": "true"}
        async with self._session.get(_BASE_URL, params=params) as resp:
            data = await resp.json()

        items = [_parse_item(r) for r in data.get("results", [])]
        next_url = data.get("next")  # pagination URL
        return items, next_url

    async def fetch_historical(
        self,
        start: str,
        end: str,
        coins: list[str] | None = None,
    ) -> list[NewsItem]:
        """
        Fetch historical news between start and end (YYYY-MM-DD).
        Pages through CryptoPanic API collecting all matching items.
        coins: filter by coin codes e.g. ["BTC", "ETH"]. None = all coins.
        """
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

        params: dict = {"kind": "news"}
        if coins:
            params["currencies"] = ",".join(c.lower() for c in coins)

        all_items: list[NewsItem] = []
        next_url: str | None = None
        seen_ids: set[str] = set()

        while True:
            if next_url:
                async with self._session.get(next_url) as resp:
                    data = await resp.json()
                items = [_parse_item(r) for r in data.get("results", [])]
                next_url = data.get("next")
            else:
                items, next_url = await self._fetch_page(params)

            if not items:
                break

            # Filter to date range and deduplicate
            in_range = []
            stop = False
            for item in items:
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)

                if item.timestamp > end_dt:
                    continue
                if item.timestamp < start_dt:
                    stop = True
                    break
                in_range.append(item)

            all_items.extend(in_range)

            if stop or not next_url:
                break

            await asyncio.sleep(0.2)  # respect rate limits

        all_items.sort(key=lambda x: x.timestamp)
        return all_items

    async def stream(self, poll_interval: float = 3.0) -> AsyncIterator[NewsItem]:
        """
        Live news stream — polls CryptoPanic every poll_interval seconds.
        Yields only new items (deduplicates by ID).
        """
        seen_ids: set[str] = set()

        while True:
            try:
                items, _ = await self._fetch_page({"kind": "news"})
                for item in reversed(items):  # oldest first
                    if item.id not in seen_ids:
                        seen_ids.add(item.id)
                        yield item
            except Exception as e:
                print(f"[CryptoPanic] stream error: {e}")

            await asyncio.sleep(poll_interval)
