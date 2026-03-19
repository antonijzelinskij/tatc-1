"""
Base abstractions для провайдеров данных.

Принцип Open/Closed: добавляем новый источник данных (биржевой WebSocket,
REST API, БД) — только создаём новый subclass, ядро не трогаем.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator

from crypto_bot.models.domain import NewsItem


# ---------------------------------------------------------------------------
# News provider
# ---------------------------------------------------------------------------

class BaseNewsProvider(ABC):
    """Абстрактный источник новостей.

    Контракт:
    - get_news_stream() — итерирует NewsItem в хронологическом порядке.
    """

    @abstractmethod
    def get_news_stream(self) -> Iterator[NewsItem]:
        """Генерирует новости одну за другой (lazy — не грузит всё в память)."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ---------------------------------------------------------------------------
# Market data provider
# ---------------------------------------------------------------------------

class BaseMarketData(ABC):
    """Абстрактный источник рыночных данных.

    Контракт:
    - get_price_at(coin, timestamp) — возвращает лучшую доступную цену
      для указанного coin в момент времени >= timestamp (метод «вперёд»,
      т.к. в реальности мы можем исполнить только по следующей цене).
    """

    @abstractmethod
    def get_price_at(self, coin: str, timestamp: datetime) -> float | None:
        """Возвращает цену монеты coin в момент timestamp.

        Returns:
            float — цена, или None если данные отсутствуют.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
