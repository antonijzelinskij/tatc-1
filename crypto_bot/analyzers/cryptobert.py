"""
CryptoBERTAnalyzer
------------------
Реализация BaseAnalyzer на базе модели ElKulako/cryptobert
(FinBERT-style, fine-tuned на крипто-новостях).

Лейблы модели:
    "Bearish"  → SELL
    "Neutral"  → HOLD
    "Bullish"  → BUY

Симуляция задержки:
    total_latency_ms = inference_latency_ms + network_latency_ms
    — берётся из конфига, НЕ из реального времени выполнения.
    Это ключевое: реальный инференс может занять 500 мс на CPU,
    но в симулированном мире ордер исполняется ровно через
    total_latency_ms мс после выхода новости.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from crypto_bot.analyzers.base import BaseAnalyzer
from crypto_bot.models.domain import Action, NewsItem, Signal

if TYPE_CHECKING:
    from transformers import Pipeline

logger = logging.getLogger(__name__)

# Маппинг: лейбл CryptoBERT → Action
_LABEL_TO_ACTION: dict[str, Action] = {
    "Bullish": Action.BUY,
    "Bearish": Action.SELL,
    "Neutral": Action.HOLD,
    # Некоторые чекпоинты используют числовые лейблы — обрабатываем оба варианта
    "LABEL_0": Action.HOLD,
    "LABEL_1": Action.BUY,
    "LABEL_2": Action.SELL,
}

# CryptoBERT принимает не более 512 токенов; обрезаем на уровне символов
MAX_TEXT_CHARS = 512


class CryptoBERTAnalyzer(BaseAnalyzer):
    """Загружает CryptoBERT при инициализации и анализирует новости.

    Args:
        model_name:            HuggingFace model id (из конфига).
        inference_latency_ms:  Симулируемая задержка инференса (мс).
        network_latency_ms:    Симулируемая сетевая задержка (мс).
        buy_threshold:         Мин. уверенность для BUY-сигнала.
        sell_threshold:        Мин. уверенность для SELL-сигнала.
        device:                "cpu" | "cuda" | "mps" (None = авто).
    """

    def __init__(
        self,
        model_name: str,
        inference_latency_ms: int,
        network_latency_ms: int,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._inference_latency_ms = inference_latency_ms
        self._network_latency_ms = network_latency_ms
        self._total_latency_ms = inference_latency_ms + network_latency_ms
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

        self._pipeline: Pipeline = self._load_pipeline(device)

    # ------------------------------------------------------------------
    # Инициализация пайплайна
    # ------------------------------------------------------------------

    def _load_pipeline(self, device: str | None) -> "Pipeline":
        """Загружает transformers pipeline.

        Намеренно выносим в отдельный метод — упрощает unit-тесты
        (mock этого метода = не грузим реальную модель).
        """
        try:
            from transformers import pipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Установи зависимости: pip install transformers torch"
            ) from exc

        logger.info("Loading model: %s (device=%s)", self._model_name, device or "auto")

        kwargs: dict = {
            "task": "text-classification",
            "model": self._model_name,
            "top_k": None,          # возвращаем scores для всех лейблов
        }
        if device is not None:
            kwargs["device"] = device

        pipe = pipeline(**kwargs)
        logger.info("Model loaded successfully: %s", self._model_name)
        return pipe

    # ------------------------------------------------------------------
    # Основной метод анализа
    # ------------------------------------------------------------------

    def analyze(self, news_item: NewsItem) -> Signal:
        """Прогоняет текст новости через CryptoBERT.

        Временно й сдвиг:
            Возвращаемый Signal содержит total_latency_ms.
            Core Engine сам прибавит эту задержку к news_item.timestamp,
            чтобы получить execution_timestamp для запроса цены.

        Returns:
            Signal с action, confidence, raw_label и метаданными.
        """
        text = news_item.headline[:MAX_TEXT_CHARS]

        # ── Реальный инференс ──────────────────────────────────────────
        # Реальное время выполнения pipeline() нас НЕ интересует для
        # симуляции. Мы берём фиксированный inference_latency_ms из конфига.
        raw_results: list[dict] = self._pipeline(text)  # type: ignore[assignment]

        # pipeline с top_k=None возвращает [[{label, score}, ...]]
        # Нормализуем к плоскому списку
        if raw_results and isinstance(raw_results[0], list):
            scores: list[dict] = raw_results[0]
        else:
            scores = raw_results  # type: ignore[assignment]

        logger.debug("Raw model output for '%s...': %s", text[:60], scores)

        # ── Выбираем лейбл с максимальным score ───────────────────────
        best = max(scores, key=lambda x: x["score"])
        raw_label: str = best["label"]
        confidence: float = float(best["score"])

        # ── Маппинг лейбла в Action + порог уверенности ───────────────
        action = self._resolve_action(raw_label, confidence)

        signal = Signal(
            action=action,
            confidence=confidence,
            raw_label=raw_label,
            text=text,
            news_timestamp=news_item.timestamp,
            total_latency_ms=self._total_latency_ms,
        )

        logger.info(
            "Signal: coin=%s | action=%s | confidence=%.3f | label=%s | "
            "latency=%dms | exec_ts=%s",
            news_item.coin,
            action.value,
            confidence,
            raw_label,
            self._total_latency_ms,
            signal.execution_timestamp.isoformat(),
        )
        return signal

    def _resolve_action(self, raw_label: str, confidence: float) -> Action:
        """Применяет пороги уверенности к сырому лейблу модели."""
        mapped = _LABEL_TO_ACTION.get(raw_label, Action.HOLD)

        # Если уверенность ниже порога — понижаем до HOLD
        if mapped == Action.BUY and confidence < self._buy_threshold:
            logger.debug(
                "BUY confidence %.3f < threshold %.3f → downgraded to HOLD",
                confidence, self._buy_threshold,
            )
            return Action.HOLD
        if mapped == Action.SELL and confidence < self._sell_threshold:
            logger.debug(
                "SELL confidence %.3f < threshold %.3f → downgraded to HOLD",
                confidence, self._sell_threshold,
            )
            return Action.HOLD

        return mapped
