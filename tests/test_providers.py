"""
Тесты: HistoricalNewsProvider и HistoricalMarketData
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from crypto_bot.providers.historical import HistoricalMarketData, HistoricalNewsProvider
from crypto_bot.models.domain import NewsItem


# ---------------------------------------------------------------------------
# Helpers: генерируем минимальные CSV в tmp_path
# ---------------------------------------------------------------------------

def write_news_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_prices_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# HistoricalNewsProvider
# ---------------------------------------------------------------------------

class TestHistoricalNewsProvider:

    @pytest.fixture
    def news_csv(self, tmp_path):
        return write_news_csv(tmp_path / "news.csv", [
            {"timestamp": "2024-01-01T00:00:00+00:00", "coin": "BTC", "headline": "BTC hits ATH", "source": "CoinDesk"},
            {"timestamp": "2024-01-01T01:00:00+00:00", "coin": "ETH", "headline": "ETH staking up", "source": "Reuters"},
            {"timestamp": "2024-01-01T02:00:00+00:00", "coin": "BTC", "headline": "BTC dumps hard", "source": "Bloomberg"},
        ])

    def test_stream_yields_news_items(self, news_csv):
        provider = HistoricalNewsProvider(news_csv)
        items = list(provider.get_news_stream())
        assert len(items) == 3
        assert all(isinstance(i, NewsItem) for i in items)

    def test_stream_is_chronological(self, news_csv):
        provider = HistoricalNewsProvider(news_csv)
        items = list(provider.get_news_stream())
        timestamps = [i.timestamp for i in items]
        assert timestamps == sorted(timestamps)

    def test_coin_filter(self, news_csv):
        provider = HistoricalNewsProvider(news_csv, coins_filter=["BTC"])
        items = list(provider.get_news_stream())
        assert len(items) == 2
        assert all(i.coin == "BTC" for i in items)

    def test_coin_filter_case_insensitive(self, news_csv):
        provider = HistoricalNewsProvider(news_csv, coins_filter=["btc"])
        items = list(provider.get_news_stream())
        assert len(items) == 2

    def test_fields_populated(self, news_csv):
        provider = HistoricalNewsProvider(news_csv)
        item = list(provider.get_news_stream())[0]
        assert item.coin == "BTC"
        assert item.headline == "BTC hits ATH"
        assert item.source == "CoinDesk"

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame([{"timestamp": "2024-01-01", "coin": "BTC"}]).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            HistoricalNewsProvider(p)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            HistoricalNewsProvider("/nonexistent/file.csv")


# ---------------------------------------------------------------------------
# HistoricalMarketData
# ---------------------------------------------------------------------------

class TestHistoricalMarketData:

    BASE_TS = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def prices_csv(self, tmp_path):
        rows = [
            {"timestamp": (self.BASE_TS + timedelta(seconds=i)).isoformat(),
             "coin": "BTC", "price": 40_000.0 + i * 10}
            for i in range(10)
        ]
        return write_prices_csv(tmp_path / "prices.csv", rows)

    def test_get_price_exact_timestamp(self, prices_csv):
        md = HistoricalMarketData(prices_csv)
        price = md.get_price_at("BTC", self.BASE_TS)
        assert price == pytest.approx(40_000.0)

    def test_get_price_forward_fill(self, prices_csv):
        """Запрашиваем время между тиками — должны получить следующий тик (≥ ts)."""
        md = HistoricalMarketData(prices_csv)
        # 500 мс после первого тика, второй тик через 1 сек → должны получить тик t+1
        ts = self.BASE_TS + timedelta(milliseconds=500)
        price = md.get_price_at("BTC", ts)
        assert price == pytest.approx(40_010.0)  # тик t+1 = 40_000 + 1*10

    def test_get_price_unknown_coin_returns_none(self, prices_csv):
        md = HistoricalMarketData(prices_csv)
        assert md.get_price_at("SOL", self.BASE_TS) is None

    def test_get_price_past_end_returns_none(self, prices_csv):
        md = HistoricalMarketData(prices_csv)
        ts = self.BASE_TS + timedelta(hours=1)  # за пределами данных
        assert md.get_price_at("BTC", ts) is None

    def test_coin_case_insensitive(self, prices_csv):
        md = HistoricalMarketData(prices_csv)
        assert md.get_price_at("btc", self.BASE_TS) is not None

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame([{"timestamp": "2024-01-01", "coin": "BTC"}]).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            HistoricalMarketData(p)

    def test_latency_simulation(self, prices_csv):
        """Ключевой тест: цена по T+latency должна отличаться от цены по T."""
        md = HistoricalMarketData(prices_csv)
        price_at_t0 = md.get_price_at("BTC", self.BASE_TS)
        price_at_t_plus_3s = md.get_price_at("BTC", self.BASE_TS + timedelta(seconds=3))
        assert price_at_t_plus_3s != price_at_t0
        assert price_at_t_plus_3s == pytest.approx(40_030.0)
