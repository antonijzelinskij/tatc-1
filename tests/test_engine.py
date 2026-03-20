"""Unit tests for BacktestEngine capital accounting and position logic."""
from datetime import datetime, timezone, timedelta

import pytest

from tatc.backtester.engine import BacktestConfig, BacktestEngine
from tatc.core.models import Candle, NewsItem, Signal, SignalType


def _dt(offset_hours: int = 0) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=offset_hours)


def _candle(symbol: str, hour: int, price: float) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=_dt(hour),
        open=price,
        high=price * 1.01,
        low=price * 0.99,
        close=price,
        volume=1000.0,
    )


def _news(title: str, hour: int, coins: list[str] | None = None) -> NewsItem:
    return NewsItem(
        id=f"news-{hour}",
        timestamp=_dt(hour),
        title=title,
        body="",
        source="test",
        url="https://example.com",
        coins=coins or ["BTC"],
    )


class AlwaysBuyStrategy:
    """Stub strategy: always BUY with confidence 1.0."""
    def analyze(self, news: NewsItem) -> Signal:
        symbol = f"{news.coins[0]}USDT" if news.coins else "BTCUSDT"
        return Signal(
            signal=SignalType.BUY,
            confidence=1.0,
            symbol=symbol,
            news=news,
            timestamp=news.timestamp,
        )


class AlwaysSellStrategy:
    """Stub strategy: always SELL with confidence 1.0."""
    def analyze(self, news: NewsItem) -> Signal:
        symbol = f"{news.coins[0]}USDT" if news.coins else "BTCUSDT"
        return Signal(
            signal=SignalType.SELL,
            confidence=1.0,
            symbol=symbol,
            news=news,
            timestamp=news.timestamp,
        )


class AlwaysHoldStrategy:
    """Stub strategy: always HOLD."""
    def analyze(self, news: NewsItem) -> Signal:
        return Signal(
            signal=SignalType.HOLD,
            confidence=0.0,
            symbol="BTCUSDT",
            news=news,
            timestamp=news.timestamp,
        )


class TestLongTrades:
    # Engine: position opens at candle[i], closes when held >= hold_candles.
    # candle[i] increments held to 1, candle[i+hold_candles-1] closes (held == hold_candles).
    # So with hold_candles=2: 2 candles → entry at [0], exit at [1].

    def test_profitable_long_increases_capital(self):
        """Buy at 50k, sell at 55k → profitable."""
        cfg = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=1.0,
            hold_candles=2,
            allow_short=False,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),  # entry at open=50k, held→1
                _candle("BTCUSDT", 2, 55_000.0),  # held→2 >= 2, exit at close=55k
            ]
        }
        news = [_news("Bitcoin surges!", 0)]
        metrics = engine.run(candles, news)

        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.total_pnl > 0

    def test_losing_long_decreases_capital(self):
        """Buy at 50k, sell at 45k → loss."""
        cfg = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=1.0,
            hold_candles=2,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),
                _candle("BTCUSDT", 2, 45_000.0),
            ]
        }
        metrics = engine.run(candles, [_news("Bitcoin pumps!", 0)])
        assert metrics.total_pnl < 0
        assert metrics.losing_trades == 1

    def test_capital_math_long(self):
        """Verify exact PnL after a long trade (no slippage)."""
        initial = 10_000.0
        cfg = BacktestConfig(
            initial_capital=initial,
            position_size_pct=1.0,
            hold_candles=2,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        entry_price = 50_000.0
        exit_price = 55_000.0
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, entry_price),
                _candle("BTCUSDT", 2, exit_price),
            ]
        }
        metrics = engine.run(candles, [_news("BTC!", 0)])
        qty = initial / entry_price
        expected_pnl = (exit_price - entry_price) * qty
        assert metrics.total_pnl == pytest.approx(expected_pnl, rel=1e-6)


class TestShortTrades:
    def test_short_requires_allow_short(self):
        """SELL signal with allow_short=False → no trade."""
        cfg = BacktestConfig(allow_short=False, slippage_pct=0.0)
        engine = BacktestEngine(AlwaysSellStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),
                _candle("BTCUSDT", 2, 45_000.0),
            ]
        }
        metrics = engine.run(candles, [_news("crash!", 0)])
        assert metrics.total_trades == 0

    def test_profitable_short_increases_capital(self):
        """Short at 50k, buy back at 45k → profitable."""
        cfg = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=1.0,
            hold_candles=2,
            allow_short=True,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysSellStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),
                _candle("BTCUSDT", 2, 45_000.0),
            ]
        }
        metrics = engine.run(candles, [_news("crash!", 0)])
        assert metrics.total_trades == 1
        assert metrics.total_pnl > 0

    def test_capital_math_short(self):
        """Verify capital is correctly returned after a short trade (margin + profit)."""
        initial = 10_000.0
        cfg = BacktestConfig(
            initial_capital=initial,
            position_size_pct=1.0,
            hold_candles=2,
            allow_short=True,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysSellStrategy(), cfg)
        entry_price = 50_000.0
        exit_price = 45_000.0
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, entry_price),
                _candle("BTCUSDT", 2, exit_price),
            ]
        }
        metrics = engine.run(candles, [_news("crash!", 0)])
        qty = initial / entry_price
        # Short PnL: (entry - exit) * qty (price fell → profit)
        expected_pnl = (entry_price - exit_price) * qty
        assert metrics.total_pnl == pytest.approx(expected_pnl, rel=1e-6)

    def test_losing_short_reduces_capital(self):
        """Short at 50k, buy back at 55k → loss."""
        cfg = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=1.0,
            hold_candles=2,
            allow_short=True,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysSellStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),
                _candle("BTCUSDT", 2, 55_000.0),
            ]
        }
        metrics = engine.run(candles, [_news("crash!", 0)])
        assert metrics.total_pnl < 0


class TestPositionLogic:
    def test_no_trade_if_hold_signal(self):
        """HOLD signal → no position opened."""
        cfg = BacktestConfig()
        engine = BacktestEngine(AlwaysHoldStrategy(), cfg)
        candles = {"BTCUSDT": [_candle("BTCUSDT", 1, 50_000.0)]}
        metrics = engine.run(candles, [_news("meh", 0)])
        assert metrics.total_trades == 0

    def test_no_trade_if_no_candle_after_news(self):
        """News arrives after last candle → no position opened."""
        cfg = BacktestConfig()
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        # Candle is at hour 0, news at hour 1 (after candle)
        candles = {"BTCUSDT": [_candle("BTCUSDT", 0, 50_000.0)]}
        metrics = engine.run(candles, [_news("BTC pumps!", 1)])
        assert metrics.total_trades == 0

    def test_hold_candles_respected(self):
        """Position closes exactly after hold_candles candles."""
        cfg = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.1,
            hold_candles=2,
            slippage_pct=0.0,
        )
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),   # entry
                _candle("BTCUSDT", 2, 51_000.0),   # held (1)
                _candle("BTCUSDT", 3, 52_000.0),   # exit (held == 2)
                _candle("BTCUSDT", 4, 53_000.0),   # no position
            ]
        }
        metrics = engine.run(candles, [_news("BTC!", 0)])
        assert metrics.total_trades == 1

    def test_no_double_entry_same_symbol(self):
        """Two news events on same symbol → only one position opened."""
        cfg = BacktestConfig(hold_candles=10, slippage_pct=0.0)
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        candles = {
            "BTCUSDT": [
                _candle("BTCUSDT", 1, 50_000.0),
                _candle("BTCUSDT", 2, 51_000.0),
                _candle("BTCUSDT", 3, 52_000.0),
            ]
        }
        news = [_news("BTC!", 0), _news("BTC again!", 1)]
        metrics = engine.run(candles, news)
        # Only one trade because we were already in position on 2nd news
        assert metrics.total_trades == 1

    def test_no_symbol_data_skipped(self):
        """Signal for symbol not in candles → skipped gracefully."""
        cfg = BacktestConfig()
        engine = BacktestEngine(AlwaysBuyStrategy(), cfg)
        # candles only for ETH, but news mentions BTC
        candles = {"ETHUSDT": [_candle("ETHUSDT", 1, 3_000.0)]}
        metrics = engine.run(candles, [_news("BTC pumps!", 0, coins=["BTC"])])
        assert metrics.total_trades == 0
