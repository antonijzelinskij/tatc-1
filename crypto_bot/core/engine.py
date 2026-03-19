"""
Core Backtest Engine
--------------------
Событийный цикл, объединяющий все компоненты.

Принцип единственной ответственности:
    Ядро только ОРКЕСТРИРУЕТ — оно не знает о формате CSV,
    не знает об архитектуре модели, не считает PnL напрямую.
    Каждый компонент заменяется независимо.

Ключевая логика сдвига времени:
    ┌─────────────────────────────────────────────────────────┐
    │  T=0             новость опубликована                   │
    │  T+inference_ms  модель вернула сигнал                  │
    │  T+total_ms      ордер дошёл до биржи (+ network_ms)    │
    │                                                         │
    │  execution_price = market_data.get_price_at(            │
    │      coin, T + inference_ms + network_ms                │
    │  )                                                      │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from crypto_bot.analyzers.base import BaseAnalyzer
from crypto_bot.execution.base import BaseExecutionEngine
from crypto_bot.models.domain import Action, BacktestReport
from crypto_bot.providers.base import BaseMarketData, BaseNewsProvider

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Минимальный набор параметров для запуска движка."""
    coins: list[str]


class BacktestEngine:
    """Событийный движок бэктеста.

    Принимает на вход абстракции (не конкретные классы):
        - news_provider   : источник исторических новостей
        - market_data     : источник исторических цен
        - analyzer        : ML-модель или любой другой анализатор
        - execution_engine: движок исполнения ордеров

    Использование:
        engine = BacktestEngine(cfg, news, prices, analyzer, executor)
        report = engine.run()
        print(report.summary())
    """

    def __init__(
        self,
        config: BacktestConfig,
        news_provider: BaseNewsProvider,
        market_data: BaseMarketData,
        analyzer: BaseAnalyzer,
        execution_engine: BaseExecutionEngine,
    ) -> None:
        self._config = config
        self._news_provider = news_provider
        self._market_data = market_data
        self._analyzer = analyzer
        self._execution_engine = execution_engine

        self._coins_set = {c.upper() for c in config.coins}

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def run(self) -> BacktestReport:
        """Запускает событийный цикл по всем историческим новостям.

        Returns:
            BacktestReport с итоговой статистикой.
        """
        logger.info(
            "Starting backtest | coins=%s | news=%r | market=%r | model=%r",
            self._config.coins,
            self._news_provider,
            self._market_data,
            self._analyzer,
        )

        processed = 0
        skipped_coin = 0
        skipped_price = 0
        trades_opened = 0

        for news_item in self._news_provider.get_news_stream():

            # ── Фильтр по монетам ──────────────────────────────────────
            if news_item.coin not in self._coins_set:
                skipped_coin += 1
                continue

            processed += 1
            logger.debug(
                "Event #%d | %s | %s | '%s...'",
                processed,
                news_item.timestamp.isoformat(),
                news_item.coin,
                news_item.headline[:80],
            )

            # ── Шаг 1: Получаем сигнал от ML-модели ───────────────────
            # Реальное время inference НЕ имеет значения для симуляции.
            # signal.total_latency_ms берётся из конфига.
            signal = self._analyzer.analyze(news_item)

            if signal.action == Action.HOLD:
                logger.debug("HOLD → skip execution")
                continue

            # ── Шаг 2: Запрашиваем цену на момент T + latency ─────────
            # Это центральная механика симуляции задержки:
            # execution_timestamp = news_timestamp + inference_ms + network_ms
            execution_price = self._market_data.get_price_at(
                coin=news_item.coin,
                timestamp=signal.execution_timestamp,   # ← T + latency
            )

            if execution_price is None:
                logger.warning(
                    "No price data at execution_ts=%s for %s — skipping trade",
                    signal.execution_timestamp.isoformat(),
                    news_item.coin,
                )
                skipped_price += 1
                continue

            # ── Шаг 3: Исполняем виртуальный ордер ────────────────────
            trade = self._execution_engine.execute(
                signal=signal,
                coin=news_item.coin,
                price=execution_price,
            )

            if trade is not None:
                trades_opened += 1

        # ------------------------------------------------------------------
        report = self._execution_engine.get_report()

        logger.info(
            "Backtest complete | events_processed=%d | skipped_coin=%d | "
            "skipped_price=%d | trades=%d",
            processed, skipped_coin, skipped_price, trades_opened,
        )
        return report
