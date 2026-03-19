"""
Тесты: crypto_bot/models/domain.py
"""
from datetime import datetime, timezone, timedelta

import pytest

from crypto_bot.models.domain import (
    Action, BacktestReport, NewsItem, Signal, Trade,
)


def utcnow() -> datetime:
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def _make(self, latency_ms: int = 150) -> Signal:
        return Signal(
            action=Action.BUY,
            confidence=0.85,
            raw_label="Bullish",
            text="Bitcoin surges to new ATH",
            news_timestamp=utcnow(),
            total_latency_ms=latency_ms,
        )

    def test_execution_timestamp_offset(self):
        signal = self._make(latency_ms=150)
        expected = utcnow() + timedelta(milliseconds=150)
        assert signal.execution_timestamp == expected

    def test_zero_latency(self):
        signal = self._make(latency_ms=0)
        assert signal.execution_timestamp == signal.news_timestamp

    def test_action_enum(self):
        for action in Action:
            s = Signal(
                action=action, confidence=0.7, raw_label="x",
                text="t", news_timestamp=utcnow(), total_latency_ms=0
            )
            assert s.action == action


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

class TestTrade:
    def _make(self, exit_price=None) -> Trade:
        ts = utcnow()
        return Trade(
            coin="BTC",
            action=Action.BUY,
            news_timestamp=ts,
            execution_timestamp=ts + timedelta(milliseconds=150),
            entry_price=40_000.0,
            quantity=0.025,
            confidence=0.85,
            total_latency_ms=150,
            exit_price=exit_price,
        )

    def test_is_open_when_no_exit(self):
        assert self._make(exit_price=None).is_open is True

    def test_is_closed_when_exit_set(self):
        assert self._make(exit_price=42_000.0).is_open is False


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------

class TestBacktestReport:
    def _trade(self, pnl: float) -> Trade:
        ts = utcnow()
        return Trade(
            coin="BTC", action=Action.BUY,
            news_timestamp=ts, execution_timestamp=ts,
            entry_price=1.0, quantity=1.0, confidence=0.8,
            total_latency_ms=100, pnl=pnl, exit_price=1.0 + pnl,
        )

    def test_empty_report(self):
        r = BacktestReport(initial_capital=10_000, final_capital=10_000)
        assert r.total_trades == 0
        assert r.win_rate == 0.0
        assert r.total_pnl == 0.0

    def test_total_pnl(self):
        r = BacktestReport(
            trades=[self._trade(100), self._trade(-50), self._trade(25)],
            initial_capital=10_000, final_capital=10_075,
        )
        assert r.total_pnl == pytest.approx(75.0)

    def test_win_rate(self):
        r = BacktestReport(
            trades=[self._trade(10), self._trade(-5), self._trade(20), self._trade(-3)],
            initial_capital=10_000, final_capital=10_022,
        )
        assert r.win_rate == pytest.approx(0.5)

    def test_winning_trades_count(self):
        r = BacktestReport(
            trades=[self._trade(10), self._trade(-5), self._trade(0)],
            initial_capital=10_000, final_capital=10_005,
        )
        assert r.winning_trades == 1  # только pnl > 0

    def test_summary_contains_key_fields(self):
        r = BacktestReport(
            trades=[self._trade(100)],
            initial_capital=10_000, final_capital=10_100,
        )
        summary = r.summary()
        assert "Total trades" in summary
        assert "Win rate" in summary
        assert "PnL" in summary
