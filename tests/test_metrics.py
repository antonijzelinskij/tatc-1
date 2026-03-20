"""Unit tests for backtesting metrics computation."""
import pytest

from tatc.backtester.metrics import compute_metrics, BacktestMetrics
from tatc.core.models import Order, OrderSide, OrderStatus
from datetime import datetime, timezone


def _order(side: OrderSide, qty: float, price: float) -> Order:
    return Order(
        id="test",
        symbol="BTCUSDT",
        side=side,
        quantity=qty,
        price=price,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=OrderStatus.FILLED,
    )


def _long_trade(entry_price: float, exit_price: float, qty: float = 1.0):
    """(entry_order, exit_order) for a long trade."""
    return (
        _order(OrderSide.BUY, qty, entry_price),
        _order(OrderSide.SELL, qty, exit_price),
    )


def _short_trade(entry_price: float, exit_price: float, qty: float = 1.0):
    """(entry_order, exit_order) for a short trade."""
    return (
        _order(OrderSide.SELL, qty, entry_price),
        _order(OrderSide.BUY, qty, exit_price),
    )


class TestNoTrades:
    def test_empty_trades_returns_zeros(self):
        metrics = compute_metrics([], 10_000.0, [10_000.0])
        assert metrics.total_trades == 0
        assert metrics.total_pnl == 0
        assert metrics.win_rate == 0
        assert metrics.sharpe_ratio == 0
        assert metrics.max_drawdown == 0


class TestWinRate:
    def test_all_winning(self):
        trades = [_long_trade(100, 110), _long_trade(200, 220)]
        equity = [10_000.0, 10_100.0, 10_220.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.win_rate == pytest.approx(1.0)
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 0

    def test_all_losing(self):
        trades = [_long_trade(100, 90), _long_trade(200, 180)]
        equity = [10_000.0, 9_900.0, 9_700.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.win_rate == pytest.approx(0.0)
        assert metrics.losing_trades == 2

    def test_mixed_win_rate(self):
        trades = [_long_trade(100, 110), _long_trade(100, 90)]
        equity = [10_000.0, 10_100.0, 10_000.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.win_rate == pytest.approx(0.5)


class TestPnL:
    def test_long_pnl(self):
        # 1 BTC: buy at 50k, sell at 55k → +5000
        trades = [_long_trade(50_000, 55_000, qty=1.0)]
        metrics = compute_metrics(trades, 10_000.0, [10_000.0, 15_000.0])
        assert metrics.total_pnl == pytest.approx(5_000.0)

    def test_short_pnl(self):
        # 1 BTC: short at 50k, buy at 45k → +5000
        trades = [_short_trade(50_000, 45_000, qty=1.0)]
        metrics = compute_metrics(trades, 10_000.0, [10_000.0, 15_000.0])
        assert metrics.total_pnl == pytest.approx(5_000.0)

    def test_losing_long_pnl(self):
        trades = [_long_trade(50_000, 45_000, qty=1.0)]
        metrics = compute_metrics(trades, 10_000.0, [10_000.0, 5_000.0])
        assert metrics.total_pnl == pytest.approx(-5_000.0)

    def test_pnl_pct(self):
        trades = [_long_trade(100, 110, qty=100)]  # pnl = +1000
        metrics = compute_metrics(trades, 10_000.0, [10_000.0, 11_000.0])
        assert metrics.total_pnl_pct == pytest.approx(10.0)  # 1000/10000 * 100

    def test_avg_best_worst_trade(self):
        trades = [
            _long_trade(100, 120, qty=1),  # pnl = +20
            _long_trade(100, 80, qty=1),   # pnl = -20
            _long_trade(100, 130, qty=1),  # pnl = +30 (best)
        ]
        equity = [10_000.0, 10_020.0, 10_000.0, 10_030.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.best_trade_pnl == pytest.approx(30.0)
        assert metrics.worst_trade_pnl == pytest.approx(-20.0)
        assert metrics.avg_trade_pnl == pytest.approx(10.0)  # (20 - 20 + 30) / 3


class TestMaxDrawdown:
    def test_flat_equity_no_drawdown(self):
        trades = [_long_trade(100, 100, qty=1)]  # break-even
        equity = [10_000.0, 10_000.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.max_drawdown == 0.0

    def test_drawdown_computed(self):
        # Equity goes 10000 → 11000 → 9000 → 10500
        # Peak = 11000, trough = 9000, drawdown = -2000 (18.18%)
        trades = [_long_trade(100, 90)]
        equity = [10_000.0, 11_000.0, 9_000.0, 10_500.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.max_drawdown == pytest.approx(-2_000.0)
        assert metrics.max_drawdown_pct == pytest.approx(-18.18, rel=1e-2)

    def test_always_declining_equity(self):
        trades = [_long_trade(100, 50)]
        equity = [10_000.0, 9_500.0, 9_000.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.max_drawdown == pytest.approx(-1_000.0)
        assert metrics.max_drawdown_pct == pytest.approx(-10.0)


class TestSharpe:
    def test_zero_variance_returns_zero(self):
        # Constant equity → std = 0 → sharpe = 0
        trades = [_long_trade(100, 100)]
        equity = [10_000.0, 10_000.0, 10_000.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.sharpe_ratio == 0.0

    def test_positive_returns_positive_sharpe(self):
        trades = [_long_trade(100, 110)]
        equity = [10_000.0, 10_500.0, 11_000.0]
        metrics = compute_metrics(trades, 10_000.0, equity)
        assert metrics.sharpe_ratio > 0
