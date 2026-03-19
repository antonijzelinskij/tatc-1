"""
Base abstraction для движка исполнения ордеров.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from crypto_bot.models.domain import BacktestReport, Signal, Trade


class BaseExecutionEngine(ABC):
    """Принимает сигнал и цену → создаёт/закрывает сделку."""

    @abstractmethod
    def execute(
        self,
        signal: Signal,
        coin: str,
        price: float,
    ) -> Trade | None:
        """Исполняет сигнал по заданной цене.

        Args:
            signal: Торговый сигнал от анализатора.
            coin:   Тикер монеты.
            price:  Цена исполнения (стакан в T + latency).

        Returns:
            Trade — если ордер открыт/закрыт, иначе None.
        """
        ...

    @abstractmethod
    def get_report(self) -> BacktestReport:
        """Возвращает итоговый отчёт по всем сделкам."""
        ...
