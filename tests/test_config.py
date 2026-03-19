"""
Тесты: crypto_bot/config/settings.py
"""
import textwrap
from pathlib import Path

import pytest

from crypto_bot.config.settings import (
    AppConfig, DataConfig, LoggingConfig, ModelConfig, TradingConfig,
    load_config,
)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        coins:
          - BTC
          - ETH
        data:
          news_file: "data/news.csv"
          prices_file: "data/prices.csv"
        model:
          name: "ElKulako/cryptobert"
          inference_latency_ms: 100
          network_latency_ms: 50
        trading:
          initial_capital: 10000.0
          position_size_pct: 0.10
          buy_threshold: 0.60
          sell_threshold: 0.60
        logging:
          level: "WARNING"
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def test_load_config_returns_app_config(config_file):
    cfg = load_config(config_file)
    assert isinstance(cfg, AppConfig)


def test_coins_parsed(config_file):
    cfg = load_config(config_file)
    assert cfg.coins == ["BTC", "ETH"]


def test_model_config(config_file):
    cfg = load_config(config_file)
    assert cfg.model.name == "ElKulako/cryptobert"
    assert cfg.model.inference_latency_ms == 100
    assert cfg.model.network_latency_ms == 50


def test_total_latency(config_file):
    cfg = load_config(config_file)
    assert cfg.model.total_latency_ms == 150  # 100 + 50


def test_trading_config(config_file):
    cfg = load_config(config_file)
    assert cfg.trading.initial_capital == 10_000.0
    assert cfg.trading.position_size_pct == pytest.approx(0.10)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_config_is_frozen(config_file):
    cfg = load_config(config_file)
    with pytest.raises((AttributeError, TypeError)):
        cfg.coins = ["SOL"]  # type: ignore[misc]
