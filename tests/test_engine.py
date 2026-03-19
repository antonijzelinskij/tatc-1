"""
Тесты: BacktestEngine (core event loop)

Ключевой фокус:
- Проверяем, что execution_timestamp = news_timestamp + total_latency_ms
- Проверяем, что движок запрашивает цену именно по execution_timestamp
- Проверяем фильтрацию монет
- Проверяем, что HOLD-сигналы не попадают в execution engine
"""
from datetime import datetime, timezone, timedelta
from typing import Iterator
from unittest.mock import MagicMock, call

import pytest

from crypto_bot.analyzers.base import BaseAnalyzer
from crypto_bot.core.engine import BacktestConfig, BacktestEngine
from crypto_bot.execution.base import BaseExecutionEngine
from crypto_bot.models.domain import Action, BacktestReport, NewsItem, Signal, Trade
from crypto_bot.providers.base import BaseMarketData, BaseNewsProvider


# ---------------------------------------------------------------------------
# Фабрики / фикстуры
# ---------------------------------------------------------------------------

def make_news(
    coin: str = "BTC",
    ts: datetime | None = None,
    headline: str = "test news",
) -> NewsItem:
    return NewsItem(
        timestamp=ts or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        coin=coin,
        headline=headline,
    )


def make_signal(
    action: Action = Action.BUY,
    news_item: NewsItem | None = None,
    latency_ms: int = 150,
) -> Signal:
    ni = news_item or make_news()
    return Signal(
        action=action,
        confidence=0.85,
        raw_label=action.value,
        text=ni.headline,
        news_timestamp=ni.timestamp,
        total_latency_ms=latency_ms,
    )


class StubNewsProvider(BaseNewsProvider):
    def __init__(self, items: list[NewsItem]):
        self._items = items

    def get_news_stream(self) -> Iterator[NewsItem]:
        yield from self._items


class StubMarketData(BaseMarketData):
    def __init__(self, price: float | None = 40_000.0):
        self._price = price
        self.calls: list[tuple] = []

    def get_price_at(self, coin: str, timestamp: datetime) -> float | None:
        self.calls.append((coin, timestamp))
        return self._price


class StubAnalyzer(BaseAnalyzer):
    def __init__(self, signal_factory):
        self._factory = signal_factory

    def analyze(self, news_item: NewsItem) -> Signal:
        return self._factory(news_item)


class StubExecutionEngine(BaseExecutionEngine):
    def __init__(self):
        self.calls: list[tuple] = []

    def execute(self, signal, coin, price) -> Trade | None:
        self.calls.append((signal, coin, price))
        return None

    def get_report(self) -> BacktestReport:
        return BacktestReport(initial_capital=10_000, final_capital=10_000)


def make_engine(
    news_items: list[NewsItem],
    analyzer: BaseAnalyzer,
    market_price: float | None = 40_000.0,
    coins: list[str] | None = None,
    execution: StubExecutionEngine | None = None,
) -> tuple[BacktestEngine, StubMarketData, StubExecutionEngine]:
    market = StubMarketData(price=market_price)
    exec_eng = execution or StubExecutionEngine()
    engine = BacktestEngine(
        config=BacktestConfig(coins=coins or ["BTC"]),
        news_provider=StubNewsProvider(news_items),
        market_data=market,
        analyzer=analyzer,
        execution_engine=exec_eng,
    )
    return engine, market, exec_eng


# ---------------------------------------------------------------------------
# Тесты симуляции задержки — САМЫЕ ВАЖНЫЕ
# ---------------------------------------------------------------------------

class TestLatencySimulation:

    def test_price_requested_at_execution_timestamp(self):
        """Движок должен запрашивать цену по T_news + total_latency_ms."""
        news_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        latency_ms = 150
        news = make_news(ts=news_ts)

        def factory(ni):
            return make_signal(Action.BUY, ni, latency_ms=latency_ms)

        engine, market, _ = make_engine([news], StubAnalyzer(factory))
        engine.run()

        assert len(market.calls) == 1
        _, requested_ts = market.calls[0]
        expected_ts = news_ts + timedelta(milliseconds=latency_ms)
        assert requested_ts == expected_ts

    def test_different_latencies_result_in_different_timestamps(self):
        """При разной задержке движок запрашивает разные timestamps."""
        news_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        news = make_news(ts=news_ts)

        latencies = [100, 500]
        idx = 0

        def factory(ni):
            nonlocal idx
            lat = latencies[idx % len(latencies)]
            idx += 1
            return make_signal(Action.BUY, ni, latency_ms=lat)

        news_items = [make_news(ts=news_ts), make_news(ts=news_ts)]
        engine, market, _ = make_engine(news_items, StubAnalyzer(factory))
        engine.run()

        ts_list = [ts for _, ts in market.calls]
        assert ts_list[0] == news_ts + timedelta(milliseconds=100)
        assert ts_list[1] == news_ts + timedelta(milliseconds=500)

    def test_execution_price_passed_to_engine(self):
        """Цена, полученная от market_data, должна попасть в execution_engine."""
        expected_price = 42_500.0
        news = make_news()

        def factory(ni):
            return make_signal(Action.BUY, ni, latency_ms=100)

        engine, _, exec_eng = make_engine(
            [news],
            StubAnalyzer(factory),
            market_price=expected_price,
        )
        engine.run()

        assert len(exec_eng.calls) == 1
        _, _, price = exec_eng.calls[0]
        assert price == pytest.approx(expected_price)


# ---------------------------------------------------------------------------
# Фильтрация монет
# ---------------------------------------------------------------------------

class TestCoinFiltering:

    def test_unknown_coin_skipped(self):
        """Новость по монете не из конфига должна игнорироваться."""
        news = make_news(coin="SOL")

        def factory(ni):
            return make_signal(Action.BUY, ni)

        engine, market, exec_eng = make_engine(
            [news],
            StubAnalyzer(factory),
            coins=["BTC"],
        )
        engine.run()

        assert len(exec_eng.calls) == 0
        assert len(market.calls) == 0

    def test_matching_coin_processed(self):
        news = make_news(coin="BTC")

        def factory(ni):
            return make_signal(Action.BUY, ni)

        engine, _, exec_eng = make_engine([news], StubAnalyzer(factory), coins=["BTC"])
        engine.run()

        assert len(exec_eng.calls) == 1


# ---------------------------------------------------------------------------
# HOLD пропускает execution
# ---------------------------------------------------------------------------

class TestHoldSkipping:

    def test_hold_signal_skips_price_lookup(self):
        news = make_news()

        def factory(ni):
            return make_signal(Action.HOLD, ni)

        engine, market, exec_eng = make_engine([news], StubAnalyzer(factory))
        engine.run()

        assert len(market.calls) == 0
        assert len(exec_eng.calls) == 0


# ---------------------------------------------------------------------------
# Нет цены → пропускаем сделку
# ---------------------------------------------------------------------------

class TestNoPriceSkipping:

    def test_no_price_data_skips_execution(self):
        news = make_news()

        def factory(ni):
            return make_signal(Action.BUY, ni)

        engine, _, exec_eng = make_engine(
            [news],
            StubAnalyzer(factory),
            market_price=None,  # нет данных
        )
        engine.run()

        assert len(exec_eng.calls) == 0


# ---------------------------------------------------------------------------
# Интеграционный тест: несколько событий
# ---------------------------------------------------------------------------

class TestMultipleEvents:

    def test_multiple_news_items_processed(self):
        base_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        news_items = [
            make_news(ts=base_ts + timedelta(minutes=i))
            for i in range(5)
        ]
        actions = [Action.BUY, Action.HOLD, Action.SELL, Action.BUY, Action.HOLD]
        idx = 0

        def factory(ni):
            nonlocal idx
            a = actions[idx]
            idx += 1
            return make_signal(a, ni)

        engine, market, exec_eng = make_engine(news_items, StubAnalyzer(factory))
        engine.run()

        # BUY(0) + SELL(2) + BUY(3) = 3 обращения к рынку
        assert len(market.calls) == 3
        # То же число вызовов execution_engine
        assert len(exec_eng.calls) == 3

    def test_report_returned(self):
        engine, _, _ = make_engine([], StubAnalyzer(lambda ni: make_signal()))
        report = engine.run()
        assert isinstance(report, BacktestReport)
