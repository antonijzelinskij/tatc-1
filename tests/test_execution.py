"""
Тесты: BacktestExecutionEngine
"""
from datetime import datetime, timezone, timedelta

import pytest

from crypto_bot.execution.backtest import BacktestExecutionEngine
from crypto_bot.models.domain import Action, BacktestReport, Signal, Trade


def utc(hour: int = 12, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour, minute, 0, tzinfo=timezone.utc)


def make_signal(action: Action, coin: str = "BTC", ts: datetime | None = None,
                confidence: float = 0.85, latency_ms: int = 150) -> Signal:
    news_ts = ts or utc()
    return Signal(
        action=action,
        confidence=confidence,
        raw_label=action.value,
        text="test",
        news_timestamp=news_ts,
        total_latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Открытие позиции
# ---------------------------------------------------------------------------

class TestOpenLong:

    def test_buy_opens_position(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        trade = eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        assert trade is not None
        assert trade.action == Action.BUY
        assert trade.entry_price == pytest.approx(40_000.0)

    def test_buy_calculates_quantity(self):
        # 10% от 10_000 = 1_000; qty = 1_000 / 40_000 = 0.025
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        trade = eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        assert trade.quantity == pytest.approx(0.025)

    def test_duplicate_buy_ignored(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        second = eng.execute(make_signal(Action.BUY), "BTC", price=41_000.0)
        assert second is None
        assert len(eng.open_positions) == 1

    def test_buy_different_coins_independent(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        t = eng.execute(make_signal(Action.BUY), "ETH", price=2_500.0)
        assert t is not None
        assert len(eng.open_positions) == 2


# ---------------------------------------------------------------------------
# Закрытие позиции и PnL
# ---------------------------------------------------------------------------

class TestCloseLong:

    def test_sell_closes_position(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        trade = eng.execute(make_signal(Action.SELL), "BTC", price=42_000.0)
        assert trade is not None
        assert not trade.is_open

    def test_sell_without_position_returns_none(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        result = eng.execute(make_signal(Action.SELL), "BTC", price=40_000.0)
        assert result is None

    def test_profitable_trade_pnl(self):
        # entry=40_000, exit=42_000, qty=0.025 → pnl = (42000-40000)*0.025 = 50
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        trade = eng.execute(make_signal(Action.SELL), "BTC", price=42_000.0)
        assert trade.pnl == pytest.approx(50.0)

    def test_losing_trade_pnl(self):
        # entry=40_000, exit=38_000, qty=0.025 → pnl = (38000-40000)*0.025 = -50
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        trade = eng.execute(make_signal(Action.SELL), "BTC", price=38_000.0)
        assert trade.pnl == pytest.approx(-50.0)

    def test_capital_updates_after_trade(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        eng.execute(make_signal(Action.SELL), "BTC", price=42_000.0)
        assert eng.current_capital == pytest.approx(10_050.0)

    def test_position_removed_after_close(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        eng.execute(make_signal(Action.SELL), "BTC", price=42_000.0)
        assert "BTC" not in eng.open_positions


# ---------------------------------------------------------------------------
# HOLD
# ---------------------------------------------------------------------------

class TestHold:

    def test_hold_signal_returns_none(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        result = eng.execute(make_signal(Action.HOLD), "BTC", price=40_000.0)
        assert result is None

    def test_hold_does_not_change_capital(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.HOLD), "BTC", price=40_000.0)
        assert eng.current_capital == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:

    def test_empty_report(self):
        eng = BacktestExecutionEngine(initial_capital=10_000)
        report = eng.get_report()
        assert isinstance(report, BacktestReport)
        assert report.total_trades == 0
        assert report.final_capital == pytest.approx(10_000.0)

    def test_report_contains_closed_trades(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        eng.execute(make_signal(Action.SELL), "BTC", price=42_000.0)
        report = eng.get_report()
        assert report.total_trades == 1

    def test_trades_dataframe_columns(self):
        eng = BacktestExecutionEngine(initial_capital=10_000, position_size_pct=0.10)
        eng.execute(make_signal(Action.BUY), "BTC", price=40_000.0)
        eng.execute(make_signal(Action.SELL), "BTC", price=42_000.0)
        df = eng.get_trades_dataframe()
        assert set(df.columns) >= {"coin", "entry_price", "exit_price", "pnl", "total_latency_ms"}

    def test_empty_trades_dataframe(self):
        eng = BacktestExecutionEngine(initial_capital=10_000)
        df = eng.get_trades_dataframe()
        assert df.empty
