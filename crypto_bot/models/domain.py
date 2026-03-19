"""
Domain models
-------------
Чистые dataclass-объекты без бизнес-логики.
Все остальные модули импортируют типы отсюда.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Перечисления
# ---------------------------------------------------------------------------

class Action(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Входные данные
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    """Одна новость из исторического датасета."""
    timestamp: datetime   # UTC момент публикации новости
    coin: str             # тикер монеты (BTC, ETH, …)
    headline: str         # заголовок / текст новости
    source: str = ""


# ---------------------------------------------------------------------------
# Результат анализа
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Торговый сигнал, сгенерированный ML-моделью."""
    action: Action
    confidence: float         # уверенность модели [0.0, 1.0]
    raw_label: str            # оригинальный лейбл модели (Bullish / Bearish / Neutral)
    text: str                 # текст, который анализировался
    news_timestamp: datetime  # время исходной новости
    total_latency_ms: int     # inference_latency_ms + network_latency_ms

    @property
    def execution_timestamp(self) -> datetime:
        """Виртуальный момент исполнения ордера: T_news + total_latency."""
        from datetime import timedelta
        return self.news_timestamp + timedelta(milliseconds=self.total_latency_ms)


# ---------------------------------------------------------------------------
# Исполненная сделка
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Запись о виртуальной сделке в бэктесте."""
    coin: str
    action: Action
    news_timestamp: datetime    # T=0 — момент выхода новости
    execution_timestamp: datetime  # T=0 + total_latency_ms
    entry_price: float          # цена по расписанию (стакан в момент execution_timestamp)
    quantity: float             # количество монет
    confidence: float
    total_latency_ms: int
    pnl: float = 0.0            # заполняется при закрытии позиции
    exit_price: float | None = None
    exit_timestamp: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_price is None


# ---------------------------------------------------------------------------
# Итоговая статистика бэктеста
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    trades: list[Trade] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.winning_trades / self.total_trades

    def summary(self) -> str:
        return (
            f"{'='*55}\n"
            f"  BACKTEST REPORT\n"
            f"{'='*55}\n"
            f"  Total trades   : {self.total_trades}\n"
            f"  Winning trades : {self.winning_trades}\n"
            f"  Win rate       : {self.win_rate:.1%}\n"
            f"  Total PnL      : {self.total_pnl:+.2f} USD\n"
            f"  Initial capital: {self.initial_capital:.2f} USD\n"
            f"  Final capital  : {self.final_capital:.2f} USD\n"
            f"  Return         : {(self.final_capital/self.initial_capital - 1):.2%}\n"
            f"{'='*55}"
        )
