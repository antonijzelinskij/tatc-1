"""
Тесты: CryptoBERTAnalyzer
Модель мокируется — тесты не грузят реальные веса.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto_bot.analyzers.cryptobert import CryptoBERTAnalyzer
from crypto_bot.models.domain import Action, NewsItem, Signal


def make_news(headline: str = "Bitcoin surges", coin: str = "BTC") -> NewsItem:
    return NewsItem(
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        coin=coin,
        headline=headline,
    )


def make_analyzer(
    pipeline_output: list[dict],
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.60,
) -> CryptoBERTAnalyzer:
    """Создаёт CryptoBERTAnalyzer с замоканным pipeline."""
    mock_pipeline = MagicMock(return_value=[pipeline_output])

    with patch.object(CryptoBERTAnalyzer, "_load_pipeline", return_value=mock_pipeline):
        analyzer = CryptoBERTAnalyzer(
            model_name="mock/model",
            inference_latency_ms=100,
            network_latency_ms=50,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
    return analyzer


# ---------------------------------------------------------------------------
# Маппинг лейблов
# ---------------------------------------------------------------------------

class TestLabelMapping:

    def test_bullish_maps_to_buy(self):
        analyzer = make_analyzer([
            {"label": "Bullish", "score": 0.90},
            {"label": "Neutral", "score": 0.06},
            {"label": "Bearish", "score": 0.04},
        ])
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.BUY

    def test_bearish_maps_to_sell(self):
        analyzer = make_analyzer([
            {"label": "Bearish", "score": 0.85},
            {"label": "Neutral", "score": 0.10},
            {"label": "Bullish", "score": 0.05},
        ])
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.SELL

    def test_neutral_maps_to_hold(self):
        analyzer = make_analyzer([
            {"label": "Neutral", "score": 0.80},
            {"label": "Bullish", "score": 0.12},
            {"label": "Bearish", "score": 0.08},
        ])
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.HOLD

    def test_unknown_label_defaults_to_hold(self):
        analyzer = make_analyzer([{"label": "UNKNOWN_LABEL", "score": 0.99}])
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.HOLD


# ---------------------------------------------------------------------------
# Пороги уверенности
# ---------------------------------------------------------------------------

class TestThresholds:

    def test_buy_below_threshold_becomes_hold(self):
        analyzer = make_analyzer(
            [{"label": "Bullish", "score": 0.55}],
            buy_threshold=0.60,
        )
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.HOLD

    def test_buy_above_threshold_stays_buy(self):
        analyzer = make_analyzer(
            [{"label": "Bullish", "score": 0.75}],
            buy_threshold=0.60,
        )
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.BUY

    def test_sell_below_threshold_becomes_hold(self):
        analyzer = make_analyzer(
            [{"label": "Bearish", "score": 0.50}],
            sell_threshold=0.60,
        )
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.HOLD

    def test_sell_at_exact_threshold_is_hold(self):
        # confidence < threshold → HOLD (строгое неравенство)
        analyzer = make_analyzer(
            [{"label": "Bearish", "score": 0.60}],
            sell_threshold=0.60,
        )
        # score == threshold → не проходит (< threshold is False for ==)
        signal = analyzer.analyze(make_news())
        assert signal.action == Action.SELL  # 0.60 >= 0.60 → SELL


# ---------------------------------------------------------------------------
# Симуляция задержки
# ---------------------------------------------------------------------------

class TestLatency:

    def test_total_latency_ms_in_signal(self):
        analyzer = make_analyzer([{"label": "Bullish", "score": 0.90}])
        signal = analyzer.analyze(make_news())
        # inference=100 + network=50 = 150
        assert signal.total_latency_ms == 150

    def test_execution_timestamp_offset(self):
        from datetime import timedelta
        analyzer = make_analyzer([{"label": "Bullish", "score": 0.90}])
        news = make_news()
        signal = analyzer.analyze(news)
        expected = news.timestamp + timedelta(milliseconds=150)
        assert signal.execution_timestamp == expected

    def test_real_inference_time_is_irrelevant(self):
        """Pipeline может работать сколько угодно — latency берётся из конфига."""
        import time

        slow_pipeline = MagicMock(side_effect=lambda t: (time.sleep(0), [[{"label": "Bullish", "score": 0.9}]])[1])
        with patch.object(CryptoBERTAnalyzer, "_load_pipeline", return_value=slow_pipeline):
            analyzer = CryptoBERTAnalyzer(
                model_name="mock",
                inference_latency_ms=100,
                network_latency_ms=50,
            )
        signal = analyzer.analyze(make_news())
        assert signal.total_latency_ms == 150  # не зависит от реального времени


# ---------------------------------------------------------------------------
# Возвращаемый Signal
# ---------------------------------------------------------------------------

class TestSignalFields:

    def test_signal_confidence_matches_model_score(self):
        analyzer = make_analyzer([
            {"label": "Bullish", "score": 0.87},
            {"label": "Neutral", "score": 0.10},
        ])
        signal = analyzer.analyze(make_news())
        assert signal.confidence == pytest.approx(0.87)

    def test_signal_raw_label_preserved(self):
        analyzer = make_analyzer([{"label": "Bullish", "score": 0.90}])
        signal = analyzer.analyze(make_news())
        assert signal.raw_label == "Bullish"

    def test_news_timestamp_preserved(self):
        analyzer = make_analyzer([{"label": "Bullish", "score": 0.90}])
        news = make_news()
        signal = analyzer.analyze(news)
        assert signal.news_timestamp == news.timestamp

    def test_text_truncated_to_512_chars(self):
        long_text = "A" * 1000
        analyzer = make_analyzer([{"label": "Neutral", "score": 0.80}])
        signal = analyzer.analyze(make_news(headline=long_text))
        assert len(signal.text) <= 512
