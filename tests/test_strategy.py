"""Unit tests for VaderStrategy."""
from datetime import datetime, timezone

import pytest

from tatc.core.models import NewsItem, SignalType
from tatc.strategies.sentiment_vader import VaderStrategy


def _news(title: str, body: str = "", coins: list[str] | None = None) -> NewsItem:
    return NewsItem(
        id="test-1",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        title=title,
        body=body,
        source="test",
        url="https://example.com",
        coins=coins or [],
    )


@pytest.fixture
def strategy() -> VaderStrategy:
    return VaderStrategy()


class TestSignalTypes:
    def test_bullish_headline_gives_buy(self, strategy):
        news = _news("Bitcoin surges to new ATH! Massive rally underway")
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.BUY

    def test_bearish_headline_gives_sell(self, strategy):
        news = _news("Exchange hacked, rug pull confirmed, crash incoming")
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.SELL

    def test_neutral_headline_gives_hold(self, strategy):
        news = _news("Bitcoin trades sideways on Tuesday")
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.HOLD

    def test_etf_approval_gives_buy(self, strategy):
        news = _news("SEC approves Bitcoin spot ETF, massive inflows expected")
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.BUY

    def test_scam_gives_sell(self, strategy):
        news = _news("DeFi protocol rugpull: scam exposed, fraud confirmed")
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.SELL


class TestConfidence:
    def test_confidence_is_abs_compound(self, strategy):
        news = _news("Bitcoin surges to ATH!")
        signal = strategy.analyze(news)
        assert signal.confidence >= 0
        assert signal.confidence <= 1.0
        assert signal.confidence == pytest.approx(abs(signal.metadata["compound"]))

    def test_strong_news_has_high_confidence(self, strategy):
        strong = _news("Bitcoin surges 20% to new ATH after massive institutional inflows")
        weak = _news("Bitcoin slightly up today")
        s_strong = strategy.analyze(strong)
        s_weak = strategy.analyze(weak)
        assert s_strong.confidence > s_weak.confidence

    def test_confidence_in_metadata(self, strategy):
        news = _news("Ethereum hacked, millions stolen")
        signal = strategy.analyze(news)
        assert "compound" in signal.metadata
        assert "pos" in signal.metadata
        assert "neg" in signal.metadata


class TestSymbolResolution:
    def test_coin_in_news_maps_to_symbol(self, strategy):
        news = _news("Bitcoin surges to ATH", coins=["BTC"])
        signal = strategy.analyze(news)
        assert signal.symbol == "BTCUSDT"

    def test_eth_coin_maps_correctly(self, strategy):
        news = _news("Ethereum upgrade approved", coins=["ETH"])
        signal = strategy.analyze(news)
        assert signal.symbol == "ETHUSDT"

    def test_no_coin_uses_default(self, strategy):
        news = _news("Crypto market rallies", coins=[])
        signal = strategy.analyze(news)
        assert signal.symbol == "BTCUSDT"  # default

    def test_custom_default_symbol(self):
        strat = VaderStrategy(default_symbol="ETHUSDT")
        news = _news("Market rallies")
        signal = strat.analyze(news)
        assert signal.symbol == "ETHUSDT"


class TestThresholds:
    def test_custom_thresholds(self):
        strat = VaderStrategy(buy_threshold=0.8, sell_threshold=-0.8)
        # A moderately positive headline that would normally be BUY (>=0.3)
        # should become HOLD with a higher threshold
        news = _news("Bitcoin slightly recovering, positive outlook")
        signal_default = VaderStrategy().analyze(news)
        signal_strict = strat.analyze(news)
        # With strict thresholds, the signal may be HOLD even if compound > 0.3
        if signal_default.signal == SignalType.BUY and signal_default.confidence < 0.8:
            assert signal_strict.signal == SignalType.HOLD

    def test_analyze_many(self, strategy):
        items = [
            _news("Bitcoin surges to ATH"),
            _news("Exchange hacked"),
            _news("Market sideways"),
        ]
        signals = strategy.analyze_many(items)
        assert len(signals) == 3
        assert signals[0].signal == SignalType.BUY
        assert signals[1].signal == SignalType.SELL
