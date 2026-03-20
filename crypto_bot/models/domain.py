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

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        """Максимальная просадка капитала (как доля от пика)."""
        if not self.trades:
            return 0.0
        capital = self.initial_capital
        peak = capital
        max_dd = 0.0
        for t in self.trades:
            capital += t.pnl
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def avg_confidence(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.confidence for t in self.trades) / len(self.trades)

    @property
    def avg_hold_hours(self) -> float:
        """Среднее время удержания позиции в часах."""
        closed = [t for t in self.trades if t.exit_timestamp]
        if not closed:
            return 0.0
        durations = [
            (t.exit_timestamp - t.execution_timestamp).total_seconds() / 3600
            for t in closed
        ]
        return sum(durations) / len(durations)

    def per_coin_summary(self) -> dict[str, dict]:
        """Статистика по каждой монете."""
        result: dict[str, dict] = {}
        for t in self.trades:
            if t.coin not in result:
                result[t.coin] = {"trades": 0, "pnl": 0.0, "wins": 0}
            result[t.coin]["trades"] += 1
            result[t.coin]["pnl"] += t.pnl
            if t.pnl > 0:
                result[t.coin]["wins"] += 1
        for coin, stats in result.items():
            n = stats["trades"]
            stats["win_rate"] = stats["wins"] / n if n > 0 else 0.0
        return result

    def summary(self) -> str:
        ret = (self.final_capital / self.initial_capital - 1) if self.initial_capital else 0.0
        lines = [
            f"{'='*60}",
            f"  BACKTEST REPORT",
            f"{'='*60}",
            f"  Total trades    : {self.total_trades}",
            f"  Winning trades  : {self.winning_trades}",
            f"  Win rate        : {self.win_rate:.1%}",
            f"  Total PnL       : {self.total_pnl:+.2f} USD",
            f"  Avg PnL/trade   : {self.avg_pnl:+.2f} USD",
            f"  Max drawdown    : {self.max_drawdown:.1%}",
            f"  Avg confidence  : {self.avg_confidence:.3f}",
            f"  Avg hold time   : {self.avg_hold_hours:.1f} h",
            f"  Initial capital : {self.initial_capital:.2f} USD",
            f"  Final capital   : {self.final_capital:.2f} USD",
            f"  Return          : {ret:.2%}",
        ]
        per_coin = self.per_coin_summary()
        if per_coin:
            lines.append(f"  --- Per coin ---")
            for coin, stats in sorted(per_coin.items()):
                lines.append(
                    f"  {coin:<6} trades={stats['trades']:>4} "
                    f"pnl={stats['pnl']:>+8.2f} USD  "
                    f"win={stats['win_rate']:.0%}"
                )
        lines.append(f"{'='*60}")
        return "\n".join(lines)
