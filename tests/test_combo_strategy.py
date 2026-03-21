"""Unit tests for ComboStrategy."""
from datetime import datetime, timedelta, timezone

import pytest

from tatc.core.models import NewsItem, SignalType
from tatc.strategies.combo import ComboStrategy


def _news(
    title: str,
    body: str = "",
    coins: list[str] | None = None,
    ts: datetime | None = None,
) -> NewsItem:
    return NewsItem(
        id="test-1",
        timestamp=ts or datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        title=title,
        body=body,
        source="test",
        url="https://example.com",
        coins=coins or ["BTC"],
    )


@pytest.fixture
def strategy() -> ComboStrategy:
    return ComboStrategy()


# ── Signal correctness ────────────────────────────────────────────────────────

class TestSignalTypes:
    def test_strong_buy_keyword_triggers_buy(self, strategy):
        news = _news("Bitcoin ETF approved by SEC today", coins=["BTC"])
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.BUY

    def test_halving_triggers_buy(self, strategy):
        news = _news("Bitcoin halving confirmed — miners celebrate", coins=["BTC"])
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.BUY

    def test_hack_triggers_sell(self):
        s = ComboStrategy()
        news = _news("Exchange hacked — funds drained", coins=["BTC"])
        signal = s.analyze(news)
        assert signal.signal == SignalType.SELL

    def test_rug_pull_triggers_sell(self):
        s = ComboStrategy()
        news = _news("DeFi rug pull confirmed, exit scam exposed", coins=["BTC"])
        signal = s.analyze(news)
        assert signal.signal == SignalType.SELL

    def test_neutral_gives_hold(self, strategy):
        news = _news("Bitcoin trades sideways on Tuesday", coins=["BTC"])
        signal = strategy.analyze(news)
        assert signal.signal == SignalType.HOLD

    def test_historical_context_suppressed(self):
        """Article saying 'back in 2017 Bitcoin crashed' should not trigger SELL."""
        s = ComboStrategy()
        news = _news(
            "Remembering when Bitcoin crashed back in 2017 — lessons learned",
            coins=["BTC"],
        )
        signal = s.analyze(news)
        # Noise penalty should suppress the signal to HOLD
        assert signal.signal == SignalType.HOLD

    def test_speculative_prediction_suppressed(self):
        """Price prediction articles should not trigger strong signals."""
        s = ComboStrategy()
        news = _news(
            "Analysts predict Bitcoin could reach $200k price target by year end",
            coins=["BTC"],
        )
        signal = s.analyze(news)
        assert signal.signal == SignalType.HOLD


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_same_title_suppressed_within_window(self):
        s = ComboStrategy(dedup_window_minutes=60)
        t = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        title = "Bitcoin ETF approved by SEC today"
        n1 = _news(title, coins=["BTC"], ts=t)
        n2 = _news(title, coins=["BTC"], ts=t + timedelta(minutes=30))

        s1 = s.analyze(n1)
        s2 = s.analyze(n2)

        # First should trigger, second is duplicate
        assert s1.signal != SignalType.HOLD or s2.signal == SignalType.HOLD
        assert s2.metadata.get("reason") == "duplicate"

    def test_same_title_allowed_after_window(self):
        s = ComboStrategy(dedup_window_minutes=60)
        t = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        title = "Bitcoin ETF approved by SEC today"
        n1 = _news(title, coins=["BTC"], ts=t)
        n2 = _news(title, coins=["BTC"], ts=t + timedelta(minutes=90))

        s.analyze(n1)
        s2 = s.analyze(n2)
        assert s2.metadata.get("reason") != "duplicate"


# ── Cooldown ──────────────────────────────────────────────────────────────────

class TestCooldown:
    def test_coin_on_cooldown_after_signal(self):
        s = ComboStrategy(cooldown_minutes=30)
        t = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

        n1 = _news("Bitcoin halving confirmed today", coins=["BTC"], ts=t)
        n2 = _news("Bitcoin ETF approval imminent for BTC", coins=["BTC"],
                   ts=t + timedelta(minutes=15))

        s1 = s.analyze(n1)
        s2 = s.analyze(n2)

        if s1.signal != SignalType.HOLD:
            assert s2.metadata.get("reason") == "cooldown"

    def test_cooldown_expires(self):
        s = ComboStrategy(cooldown_minutes=30)
        t = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

        n1 = _news("Bitcoin halving confirmed today", coins=["BTC"], ts=t)
        n2 = _news("Bitcoin surges to ATH again breaking records", coins=["BTC"],
                   ts=t + timedelta(minutes=45))

        s.analyze(n1)
        s2 = s.analyze(n2)
        assert s2.metadata.get("reason") != "cooldown"


# ── Relevance scoring ─────────────────────────────────────────────────────────

class TestRelevance:
    def test_coin_in_title_high_relevance(self, strategy):
        news = _news("Bitcoin halving confirmed", coins=["BTC"])
        signal = strategy.analyze(news)
        assert signal.metadata.get("relevance", 0) >= 0.8

    def test_coin_full_name_in_title(self, strategy):
        news = _news("Ethereum mainnet upgrade live", coins=["ETH"])
        signal = strategy.analyze(news)
        assert signal.metadata.get("relevance", 0) >= 0.8

    def test_low_relevance_suppresses_signal(self):
        s = ComboStrategy(min_relevance=0.8)
        # Coin only in tags, not mentioned in text
        news = _news("Market update: stocks and commodities move higher", coins=["BTC"])
        signal = s.analyze(news)
        assert signal.signal == SignalType.HOLD
        assert "low_relevance" in signal.metadata.get("reason", "")


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_contains_key_fields(self, strategy):
        news = _news("Bitcoin ETF approval confirmed today by SEC", coins=["BTC"])
        signal = strategy.analyze(news)
        assert "compound" in signal.metadata
        assert "adjusted_compound" in signal.metadata
        assert "relevance" in signal.metadata
        assert "noise_mult" in signal.metadata
        assert "urgency_mult" in signal.metadata

    def test_strong_keyword_recorded_in_metadata(self, strategy):
        news = _news("Bitcoin halving confirmed by network", coins=["BTC"])
        signal = strategy.analyze(news)
        assert signal.metadata.get("strong_keyword") == "BUY"

    def test_urgency_boost_applied(self, strategy):
        urgent = _news("Breaking: Bitcoin ETF approved", coins=["BTC"])
        normal = _news("Bitcoin ETF approved today", coins=["BTC"])
        s_urgent = strategy.analyze(urgent)
        s_normal = strategy.analyze(normal)
        assert s_urgent.metadata.get("urgency_mult", 1.0) == pytest.approx(1.2)
        assert s_normal.metadata.get("urgency_mult", 1.0) == pytest.approx(1.0)


# ── analyze_many ──────────────────────────────────────────────────────────────

class TestAnalyzeMany:
    def test_returns_same_count(self, strategy):
        items = [
            _news("Bitcoin halving is here", coins=["BTC"]),
            _news("Exchange hacked, funds gone", coins=["ETH"]),
            _news("Market sideways today", coins=["BTC"]),
        ]
        signals = strategy.analyze_many(items)
        assert len(signals) == 3

    def test_order_matches_input(self, strategy):
        items = [
            _news("Bitcoin halving confirmed", coins=["BTC"],
                  ts=datetime(2024, 1, 1, 1, tzinfo=timezone.utc)),
            _news("Ethereum hacked exploit", coins=["ETH"],
                  ts=datetime(2024, 1, 1, 2, tzinfo=timezone.utc)),
        ]
        signals = strategy.analyze_many(items)
        assert signals[0].news.coins == ["BTC"]
        assert signals[1].news.coins == ["ETH"]
