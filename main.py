"""
main.py — точка входа для запуска бэктеста.

Запуск:
    # 1. Установить зависимости
    pip install -r requirements.txt

    # 2. [Опционально] Загрузить реальные данные
    python data/fetch_real_data.py

    # 3. Запустить бэктест
    python main.py
    python main.py --config config.yaml          # явно указать конфиг
    python main.py --analyzer keyword            # только keyword анализатор
    python main.py --analyzer cryptobert         # только CryptoBERT
    python main.py --analyzer ensemble           # оба (из конфига)
    python main.py --debug                       # подробное логирование
    python main.py --output results/trades.csv   # сохранить сделки
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH, если запускаем напрямую
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.config.settings import AppConfig, load_config
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
        "--analyzer", default=None,
        choices=["cryptobert", "keyword", "ensemble"],
        help="Override analyzer from config (cryptobert | keyword | ensemble)"
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


def build_analyzer(cfg: AppConfig, analyzer_override: str | None):
    """Создаёт нужный анализатор согласно конфигу (или override)."""
    analyzer_name = analyzer_override or cfg.analyzer

    if analyzer_name == "keyword":
        return _build_keyword_analyzer(cfg)
    if analyzer_name == "cryptobert":
        return _build_cryptobert_analyzer(cfg)
    if analyzer_name == "ensemble":
        return _build_ensemble_analyzer(cfg)

    raise ValueError(f"Unknown analyzer: '{analyzer_name}'. Use: cryptobert | keyword | ensemble")


def _build_keyword_analyzer(cfg: AppConfig):
    from crypto_bot.analyzers.keyword import KeywordAnalyzer, KeywordConfig

    kw = cfg.keyword_analyzer
    kw_cfg = KeywordConfig(
        buy_threshold=kw.buy_threshold,
        sell_threshold=kw.sell_threshold,
        ticker_match_bonus=kw.ticker_match_bonus,
        latency_ms=kw.latency_ms,
    )
    kw_cfg.bullish_words.extend(kw.extra_bullish_words)
    kw_cfg.bearish_words.extend(kw.extra_bearish_words)

    return KeywordAnalyzer(config=kw_cfg, tracked_coins=cfg.coins)


def _build_cryptobert_analyzer(cfg: AppConfig):
    from crypto_bot.analyzers.cryptobert import CryptoBERTAnalyzer

    return CryptoBERTAnalyzer(
        model_name=cfg.model.name,
        inference_latency_ms=cfg.model.inference_latency_ms,
        network_latency_ms=cfg.model.network_latency_ms,
        buy_threshold=cfg.trading.buy_threshold,
        sell_threshold=cfg.trading.sell_threshold,
    )


def _build_ensemble_analyzer(cfg: AppConfig):
    from crypto_bot.analyzers.ensemble import EnsembleAnalyzer, WeightedAnalyzer

    weights = cfg.ensemble.weights
    members = []

    if "cryptobert" in weights:
        members.append(WeightedAnalyzer(
            analyzer=_build_cryptobert_analyzer(cfg),
            weight=weights["cryptobert"],
            name="cryptobert",
        ))
    if "keyword" in weights:
        members.append(WeightedAnalyzer(
            analyzer=_build_keyword_analyzer(cfg),
            weight=weights["keyword"],
            name="keyword",
        ))

    if not members:
        raise ValueError("Ensemble has no members — check config.yaml ensemble.weights")

    return EnsembleAnalyzer(members=members, strategy=cfg.ensemble.strategy)


def main() -> None:
    args = parse_args()

    # ── 1. Загружаем конфиг ────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer_name = args.analyzer or cfg.analyzer
    logger.info(
        "Config loaded | coins=%s | analyzer=%s | model=%s | latency=%dms",
        cfg.coins, analyzer_name, cfg.model.name, cfg.model.total_latency_ms,
    )

    # ── 2. Проверяем наличие данных ────────────────────────────────────
    news_path   = Path(cfg.data.news_file)
    prices_path = Path(cfg.data.prices_file)

    if not news_path.exists() or not prices_path.exists():
        logger.error(
            "Data files not found.\n"
            "  news:   %s (exists=%s)\n"
            "  prices: %s (exists=%s)\n"
            "Run: python data/fetch_real_data.py  (real data)\n"
            " or: python data/generate_sample_data.py  (synthetic)",
            news_path, news_path.exists(),
            prices_path, prices_path.exists(),
        )
        sys.exit(1)

    # ── 3. Инициализируем компоненты ───────────────────────────────────
    logger.info("Initializing components | analyzer=%s ...", analyzer_name)

    news_provider = HistoricalNewsProvider(
        filepath=news_path,
        coins_filter=cfg.coins,
    )
    market_data = HistoricalMarketData(filepath=prices_path)
    analyzer = build_analyzer(cfg, args.analyzer)
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
