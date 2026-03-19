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

    cfg = AppConfig(
        coins=raw["coins"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        trading=TradingConfig(**raw["trading"]),
        logging=LoggingConfig(**raw["logging"]),
    )

    # Настраиваем глобальный логгер сразу при загрузке конфига
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return cfg
