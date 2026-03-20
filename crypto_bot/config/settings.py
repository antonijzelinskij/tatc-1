"""
Config & Settings
-----------------
Единственная точка входа для чтения конфигурации.
Используем dataclasses для type-safety и простоты доступа к полям.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Dataclasses — отражают структуру config.yaml
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    news_file: str
    prices_file: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    inference_latency_ms: int
    network_latency_ms: int

    @property
    def total_latency_ms(self) -> int:
        return self.inference_latency_ms + self.network_latency_ms


@dataclass(frozen=True)
class KeywordAnalyzerConfig:
    buy_threshold: float = 0.25
    sell_threshold: float = 0.25
    ticker_match_bonus: float = 0.10
    latency_ms: int = 10
    extra_bullish_words: tuple = field(default_factory=tuple)
    extra_bearish_words: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class EnsembleConfig:
    strategy: str = "weighted_vote"
    weights: dict = field(default_factory=lambda: {"cryptobert": 0.65, "keyword": 0.35})


@dataclass(frozen=True)
class TradingConfig:
    initial_capital: float
    position_size_pct: float
    buy_threshold: float
    sell_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class AppConfig:
    coins: list[str]
    data: DataConfig
    model: ModelConfig
    trading: TradingConfig
    logging: LoggingConfig
    analyzer: str = "ensemble"
    keyword_analyzer: KeywordAnalyzerConfig = field(default_factory=KeywordAnalyzerConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)


# ---------------------------------------------------------------------------
# Загрузчик
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Читает YAML-файл и возвращает типизированный AppConfig."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    # Keyword analyzer config (опционально)
    kw_raw = raw.get("keyword_analyzer", {})
    kw_cfg = KeywordAnalyzerConfig(
        buy_threshold=kw_raw.get("buy_threshold", 0.25),
        sell_threshold=kw_raw.get("sell_threshold", 0.25),
        ticker_match_bonus=kw_raw.get("ticker_match_bonus", 0.10),
        latency_ms=kw_raw.get("latency_ms", 10),
        extra_bullish_words=tuple(kw_raw.get("extra_bullish_words", [])),
        extra_bearish_words=tuple(kw_raw.get("extra_bearish_words", [])),
    )

    # Ensemble config (опционально)
    ens_raw = raw.get("ensemble", {})
    ens_cfg = EnsembleConfig(
        strategy=ens_raw.get("strategy", "weighted_vote"),
        weights=ens_raw.get("weights", {"cryptobert": 0.65, "keyword": 0.35}),
    )

    cfg = AppConfig(
        coins=raw["coins"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        trading=TradingConfig(**raw["trading"]),
        logging=LoggingConfig(**raw["logging"]),
        analyzer=raw.get("analyzer", "ensemble"),
        keyword_analyzer=kw_cfg,
        ensemble=ens_cfg,
    )

    # Настраиваем глобальный логгер сразу при загрузке конфига
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return cfg
