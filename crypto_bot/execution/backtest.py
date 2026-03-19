"""
BacktestExecutionEngine
-----------------------
Симулирует исполнение ордеров без реального выхода на биржу.

Стратегия управления позицией (MVP-уровень):
    - Одна открытая позиция на монету одновременно.
    - BUY  → открывает LONG позицию.
    - SELL → если есть открытый LONG — закрывает его (фиксирует PnL).
             если нет открытой позиции — сигнал игнорируется (no-short в MVP).
    - HOLD → ничего не делаем.

PnL считается как:
    (exit_price - entry_price) / entry_price * position_value
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from crypto_bot.execution.base import BaseExecutionEngine
from crypto_bot.models.domain import Action, BacktestReport, Signal, Trade

logger = logging.getLogger(__name__)


class BacktestExecutionEngine(BaseExecutionEngine):
    """Виртуальный исполнитель ордеров для бэктеста.

    Args:
        initial_capital:    Стартовый капитал в USD.
        position_size_pct:  Доля капитала на одну сделку [0, 1].
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        position_size_pct: float = 0.10,
    ) -> None:
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._position_size_pct = position_size_pct

        # coin → открытый Trade
        self._open_positions: dict[str, Trade] = {}
        self._closed_trades: list[Trade] = []

    # ------------------------------------------------------------------
    # BaseExecutionEngine interface
    # ------------------------------------------------------------------

    def execute(self, signal: Signal, coin: str, price: float) -> Trade | None:
        """Обрабатывает сигнал и при необходимости открывает/закрывает сделку."""

        if signal.action == Action.HOLD:
            logger.debug("HOLD signal for %s — skipping", coin)
            return None

        if signal.action == Action.BUY:
            return self._open_long(signal, coin, price)

        if signal.action == Action.SELL:
            return self._close_long(signal, coin, price)

        return None

    def get_report(self) -> BacktestReport:
        return BacktestReport(
            trades=list(self._closed_trades),
            initial_capital=self._initial_capital,
            final_capital=self._capital,
        )

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _open_long(self, signal: Signal, coin: str, price: float) -> Trade | None:
        if coin in self._open_positions:
            logger.debug(
                "Already in position for %s (entry=%.4f) — ignoring BUY",
                coin, self._open_positions[coin].entry_price,
            )
            return None

        position_value = self._capital * self._position_size_pct
        if position_value <= 0 or price <= 0:
            logger.warning("Cannot open position: capital=%.2f price=%.4f", self._capital, price)
            return None

        quantity = position_value / price

        trade = Trade(
            coin=coin,
            action=Action.BUY,
            news_timestamp=signal.news_timestamp,
            execution_timestamp=signal.execution_timestamp,
            entry_price=price,
            quantity=quantity,
            confidence=signal.confidence,
            total_latency_ms=signal.total_latency_ms,
        )
        self._open_positions[coin] = trade

        logger.info(
            "OPEN LONG | %s | price=%.4f | qty=%.6f | value=%.2f USD | "
            "latency=%dms | exec_ts=%s",
            coin, price, quantity, position_value,
            signal.total_latency_ms,
            signal.execution_timestamp.isoformat(),
        )
        return trade

    def _close_long(self, signal: Signal, coin: str, price: float) -> Trade | None:
        open_trade = self._open_positions.pop(coin, None)
        if open_trade is None:
            logger.debug("No open position for %s — ignoring SELL signal", coin)
            return None

        pnl = (price - open_trade.entry_price) * open_trade.quantity
        self._capital += pnl

        open_trade.exit_price = price
        open_trade.exit_timestamp = signal.execution_timestamp
        open_trade.pnl = pnl
        self._closed_trades.append(open_trade)

        logger.info(
            "CLOSE LONG | %s | entry=%.4f → exit=%.4f | PnL=%+.2f USD | "
            "capital=%.2f USD",
            coin, open_trade.entry_price, price, pnl, self._capital,
        )
        return open_trade

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Возвращает все закрытые сделки как DataFrame для анализа."""
        if not self._closed_trades:
            return pd.DataFrame()

        rows = []
        for t in self._closed_trades:
            rows.append({
                "coin": t.coin,
                "action": t.action.value,
                "news_timestamp": t.news_timestamp,
                "execution_timestamp": t.execution_timestamp,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "confidence": t.confidence,
                "total_latency_ms": t.total_latency_ms,
            })
        return pd.DataFrame(rows)

    @property
    def current_capital(self) -> float:
        return self._capital

    @property
    def open_positions(self) -> dict[str, Trade]:
        return dict(self._open_positions)
