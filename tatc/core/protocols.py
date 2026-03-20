"""
Core protocols (interfaces) for the trading bot.
All major components are defined as Protocol classes — swap any implementation
without touching the rest of the codebase.
"""
from typing import AsyncIterator, Protocol, runtime_checkable

from tatc.core.models import Candle, NewsItem, Signal


@runtime_checkable
class NewsSource(Protocol):
    """Produces news items — live stream or historical replay."""

    async def stream(self) -> AsyncIterator[NewsItem]:
        """Yield news items in chronological order."""
        ...

    async def fetch_historical(
        self,
        start: str,
        end: str,
        coins: list[str] | None = None,
    ) -> list[NewsItem]:
        """Fetch historical news for backtesting."""
        ...


@runtime_checkable
class PriceSource(Protocol):
    """Provides OHLCV candle data."""

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> list[Candle]:
        """Fetch historical candles. interval: '1', '5', '15', '60', 'D'"""
        ...

    async def get_price(self, symbol: str) -> float:
        """Get current last price."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """Analyzes news and produces a trading signal."""

    def analyze(self, news: NewsItem) -> Signal:
        """
        Given a news item, return a Signal (BUY/SELL/HOLD).
        Must be fast — called in the hot path.
        """
        ...
