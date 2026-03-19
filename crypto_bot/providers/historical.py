"""
Исторические провайдеры данных (CSV-файлы).

HistoricalNewsProvider — читает новости из CSV.
HistoricalMarketData  — читает тиковые/секундные цены из CSV.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from crypto_bot.models.domain import NewsItem
from crypto_bot.providers.base import BaseNewsProvider, BaseMarketData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# News provider
# ---------------------------------------------------------------------------

class HistoricalNewsProvider(BaseNewsProvider):
    """Читает новости из CSV-файла и выдаёт их в хронологическом порядке.

    Ожидаемые колонки CSV:
        timestamp   — ISO-8601 строка или unix-ts, UTC
        coin        — тикер монеты (BTC, ETH, …)
        headline    — текст новости
        source      — (опционально) источник
    """

    REQUIRED_COLUMNS = {"timestamp", "coin", "headline"}

    def __init__(self, filepath: str | Path, coins_filter: list[str] | None = None) -> None:
        self._filepath = Path(filepath)
        self._coins_filter = [c.upper() for c in coins_filter] if coins_filter else None
        self._df: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        logger.info("Loading news data from %s", self._filepath)
        df = pd.read_csv(self._filepath)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"News CSV missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["coin"] = df["coin"].str.upper()

        if self._coins_filter:
            df = df[df["coin"].isin(self._coins_filter)]

        self._df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d news items for coins: %s", len(self._df), self._coins_filter or "all")

    def get_news_stream(self) -> Iterator[NewsItem]:
        assert self._df is not None
        for _, row in self._df.iterrows():
            yield NewsItem(
                timestamp=row["timestamp"].to_pydatetime(),
                coin=row["coin"],
                headline=row["headline"],
                source=row.get("source", ""),
            )


# ---------------------------------------------------------------------------
# Market data provider
# ---------------------------------------------------------------------------

class HistoricalMarketData(BaseMarketData):
    """Читает тиковые/секундные данные цен из CSV и отвечает на запросы
    «какая была цена в момент T?».

    Ожидаемые колонки CSV:
        timestamp   — ISO-8601 строка, UTC
        coin        — тикер (BTC, ETH, …)
        price       — цена (close или last trade price)

    Метод get_price_at использует pd.merge_asof (nearest forward/backward)
    чтобы симулировать реалистичное исполнение:
    берём первую цену, доступную ПОСЛЕ execution_timestamp.
    """

    REQUIRED_COLUMNS = {"timestamp", "coin", "price"}

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)
        # coin -> отсортированный DataFrame с [timestamp, price]
        self._data: dict[str, pd.DataFrame] = {}
        self._load()

    def _load(self) -> None:
        logger.info("Loading price data from %s", self._filepath)
        df = pd.read_csv(self._filepath)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Prices CSV missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["coin"] = df["coin"].str.upper()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Разбиваем по монетам для быстрого O(log N) поиска
        for coin, group in df.groupby("coin"):
            self._data[coin] = group[["timestamp", "price"]].reset_index(drop=True)

        logger.info(
            "Loaded price data for coins: %s | total rows: %d",
            list(self._data.keys()), len(df),
        )

    def get_price_at(self, coin: str, timestamp: datetime) -> float | None:
        """Возвращает цену монеты coin в момент >= timestamp.

        Логика сдвига времени:
            1. Берём все тики >= execution_timestamp (T + latency).
            2. Возвращаем цену первого доступного тика.
            3. Если тиков нет — возвращаем None (нет данных для исполнения).

        Это реалистичная симуляция: мы не можем купить по цене,
        которая была ДО того, как наш ордер дошёл до биржи.
        """
        coin = coin.upper()
        df = self._data.get(coin)
        if df is None:
            logger.warning("No price data for coin: %s", coin)
            return None

        ts = pd.Timestamp(timestamp).tz_localize("UTC") if timestamp.tzinfo is None \
            else pd.Timestamp(timestamp)

        # Находим первый тик >= execution_timestamp (forward-fill)
        idx = df["timestamp"].searchsorted(ts, side="left")

        if idx >= len(df):
            logger.warning(
                "No price data at or after %s for %s (exhausted)", ts, coin
            )
            return None

        price = float(df.iloc[idx]["price"])
        actual_ts = df.iloc[idx]["timestamp"]
        logger.debug(
            "Price lookup: coin=%s | requested=%s | actual_tick=%s | price=%.4f",
            coin, ts, actual_ts, price,
        )
        return price
