"""
main.py — точка входа для запуска бэктеста.

Запуск:
    # 1. Установить зависимости
    pip install -r requirements.txt

    # 2. Сгенерировать тестовые данные
    python data/generate_sample_data.py

    # 3. Запустить бэктест
    python main.py
    python main.py --config config.yaml   # явно указать конфиг
    python main.py --debug                # режим подробного логирования
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH, если запускаем напрямую
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.analyzers.cryptobert import CryptoBERTAnalyzer
from crypto_bot.config.settings import load_config
from crypto_bot.core.engine import BacktestConfig, BacktestEngine
from crypto_bot.execution.backtest import BacktestExecutionEngine
from crypto_bot.providers.historical import HistoricalMarketData, HistoricalNewsProvider

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto News Bot — Backtester")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging (overrides config)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save trades CSV to this path (e.g. results/trades.csv)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Загружаем конфиг ────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Config loaded | coins=%s | model=%s | latency=%dms",
                cfg.coins, cfg.model.name, cfg.model.total_latency_ms)

    # ── 2. Проверяем наличие данных ────────────────────────────────────
    news_path   = Path(cfg.data.news_file)
    prices_path = Path(cfg.data.prices_file)

    if not news_path.exists() or not prices_path.exists():
        logger.error(
            "Data files not found.\n"
            "  news:   %s (exists=%s)\n"
            "  prices: %s (exists=%s)\n"
            "Run: python data/generate_sample_data.py",
            news_path, news_path.exists(),
            prices_path, prices_path.exists(),
        )
        sys.exit(1)

    # ── 3. Инициализируем компоненты ───────────────────────────────────
    logger.info("Initializing components...")

    # Провайдер новостей — фильтрует по монетам из конфига
    news_provider = HistoricalNewsProvider(
        filepath=news_path,
        coins_filter=cfg.coins,
    )

    # Провайдер рыночных данных
    market_data = HistoricalMarketData(filepath=prices_path)

    # ML-анализатор: загружает CryptoBERT при инициализации
    # total_latency = inference_latency_ms + network_latency_ms
    analyzer = CryptoBERTAnalyzer(
        model_name=cfg.model.name,
        inference_latency_ms=cfg.model.inference_latency_ms,
        network_latency_ms=cfg.model.network_latency_ms,
        buy_threshold=cfg.trading.buy_threshold,
        sell_threshold=cfg.trading.sell_threshold,
    )

    # Виртуальный движок исполнения
    execution_engine = BacktestExecutionEngine(
        initial_capital=cfg.trading.initial_capital,
        position_size_pct=cfg.trading.position_size_pct,
    )

    # ── 4. Запускаем событийный цикл ───────────────────────────────────
    engine = BacktestEngine(
        config=BacktestConfig(coins=cfg.coins),
        news_provider=news_provider,
        market_data=market_data,
        analyzer=analyzer,
        execution_engine=execution_engine,
    )

    logger.info("Running backtest...")
    report = engine.run()

    # ── 5. Выводим результаты ──────────────────────────────────────────
    print("\n" + report.summary())

    # Сохраняем CSV, если указан путь
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df = execution_engine.get_trades_dataframe()
        trades_df.to_csv(output_path, index=False)
        logger.info("Trades saved to %s (%d rows)", output_path, len(trades_df))


if __name__ == "__main__":
    main()
