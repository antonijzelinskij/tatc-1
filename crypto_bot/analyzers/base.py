"""
Base abstraction для анализатора текста.

Принцип Liskov / Dependency Inversion:
    Core Engine работает только через BaseAnalyzer — никакой зависимости
    от конкретной модели. Можно переключить CryptoBERT → GPT → rule-based
    без изменения ядра.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from crypto_bot.models.domain import Signal, NewsItem


class BaseAnalyzer(ABC):
    """Анализирует текст новости и возвращает торговый сигнал."""

    @abstractmethod
    def analyze(self, news_item: NewsItem) -> Signal:
        """Принимает NewsItem, возвращает Signal с action + confidence.

        Реализация ОБЯЗАНА:
        - заполнить signal.total_latency_ms (inference + network)
        - заполнить signal.news_timestamp из news_item.timestamp
        - не обращаться к рыночным данным (чистая ответственность)
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
