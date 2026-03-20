"""
EnsembleAnalyzer
----------------
Объединяет несколько анализаторов в один сигнал.

Стратегии:
    "weighted_vote"  — взвешенное голосование по score.
                       Финальный action = argmax(сумм весов * confidence).
    "majority"       — простое большинство (confidence = avg).
    "any_signal"     — BUY/SELL от любого анализатора перебивает HOLD
                       (агрессивный режим).
    "all_agree"      — сигнал только если все согласны (консервативный).

Веса задаются через конфиг:
    ensemble:
      strategy: weighted_vote
      analyzers:
        - name: cryptobert
          weight: 0.7
        - name: keyword
          weight: 0.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from crypto_bot.analyzers.base import BaseAnalyzer
from crypto_bot.models.domain import Action, NewsItem, Signal

logger = logging.getLogger(__name__)

VALID_STRATEGIES = {"weighted_vote", "majority", "any_signal", "all_agree"}


@dataclass
class WeightedAnalyzer:
    analyzer: BaseAnalyzer
    weight: float = 1.0
    name: str = ""


class EnsembleAnalyzer(BaseAnalyzer):
    """Запускает все дочерние анализаторы и объединяет их сигналы.

    Args:
        members:   список WeightedAnalyzer.
        strategy:  стратегия объединения.
    """

    def __init__(
        self,
        members: list[WeightedAnalyzer],
        strategy: str = "weighted_vote",
    ) -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {VALID_STRATEGIES}")
        if not members:
            raise ValueError("EnsembleAnalyzer requires at least one member")

        self._members = members
        self._strategy = strategy
        logger.info(
            "EnsembleAnalyzer | strategy=%s | members=[%s]",
            strategy,
            ", ".join(f"{m.name}(w={m.weight})" for m in members),
        )

    # ------------------------------------------------------------------

    def analyze(self, news_item: NewsItem) -> Signal:
        # Собираем сигналы от всех членов ансамбля
        signals: list[tuple[WeightedAnalyzer, Signal]] = []
        for member in self._members:
            sig = member.analyzer.analyze(news_item)
            signals.append((member, sig))
            logger.debug(
                "Ensemble member '%s': action=%s conf=%.3f",
                member.name, sig.action.value, sig.confidence,
            )

        action, confidence = self._combine(signals)

        # Latency = max из всех участников (параллельный запуск)
        max_latency = max(sig.total_latency_ms for _, sig in signals)

        # Строим сводную строку для raw_label
        raw_parts = [f"{m.name}={s.action.value}({s.confidence:.2f})" for m, s in signals]
        raw_label = f"[{self._strategy}] " + " | ".join(raw_parts)

        result = Signal(
            action=action,
            confidence=confidence,
            raw_label=raw_label,
            text=news_item.headline,
            news_timestamp=news_item.timestamp,
            total_latency_ms=max_latency,
        )

        logger.info(
            "Ensemble result: coin=%s | action=%s | confidence=%.3f | strategy=%s",
            news_item.coin, action.value, confidence, self._strategy,
        )
        return result

    # ------------------------------------------------------------------
    # Стратегии объединения
    # ------------------------------------------------------------------

    def _combine(
        self, signals: list[tuple[WeightedAnalyzer, Signal]]
    ) -> tuple[Action, float]:

        if self._strategy == "weighted_vote":
            return self._weighted_vote(signals)
        elif self._strategy == "majority":
            return self._majority(signals)
        elif self._strategy == "any_signal":
            return self._any_signal(signals)
        elif self._strategy == "all_agree":
            return self._all_agree(signals)
        raise RuntimeError(f"Unhandled strategy: {self._strategy}")

    def _weighted_vote(
        self, signals: list[tuple[WeightedAnalyzer, Signal]]
    ) -> tuple[Action, float]:
        """Взвешенное голосование: score[action] += weight * confidence."""
        scores: dict[Action, float] = {a: 0.0 for a in Action}
        total_weight = sum(m.weight for m, _ in signals)

        for member, sig in signals:
            scores[sig.action] += member.weight * sig.confidence

        # Нормализуем
        if total_weight > 0:
            for a in scores:
                scores[a] /= total_weight

        best_action = max(scores, key=lambda a: scores[a])
        return best_action, scores[best_action]

    def _majority(
        self, signals: list[tuple[WeightedAnalyzer, Signal]]
    ) -> tuple[Action, float]:
        """Простое большинство голосов."""
        counts: dict[Action, int] = {a: 0 for a in Action}
        for _, sig in signals:
            counts[sig.action] += 1

        best_action = max(counts, key=lambda a: counts[a])
        avg_confidence = sum(
            s.confidence for _, s in signals if s.action == best_action
        ) / max(counts[best_action], 1)
        return best_action, avg_confidence

    def _any_signal(
        self, signals: list[tuple[WeightedAnalyzer, Signal]]
    ) -> tuple[Action, float]:
        """Если хоть кто-то даёт BUY/SELL — берём самый уверенный из них."""
        non_hold = [
            (m, s) for m, s in signals if s.action != Action.HOLD
        ]
        if not non_hold:
            # Все HOLD
            avg_conf = sum(s.confidence for _, s in signals) / len(signals)
            return Action.HOLD, avg_conf

        # Самый уверенный ненейтральный сигнал
        best_member, best_sig = max(non_hold, key=lambda x: x[1].confidence)
        return best_sig.action, best_sig.confidence

    def _all_agree(
        self, signals: list[tuple[WeightedAnalyzer, Signal]]
    ) -> tuple[Action, float]:
        """Сигнал только если все анализаторы согласны."""
        actions = {s.action for _, s in signals}
        if len(actions) == 1:
            agreed_action = next(iter(actions))
            avg_conf = sum(s.confidence for _, s in signals) / len(signals)
            return agreed_action, avg_conf
        # Разногласие → HOLD
        avg_conf = sum(s.confidence for _, s in signals) / len(signals)
        return Action.HOLD, avg_conf
