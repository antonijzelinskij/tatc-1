"""
KeywordAnalyzer
---------------
Наш собственный анализатор: ищет в тексте новости тикеры и ключевые слова.

Логика:
    1. Находим монеты в тексте (aliases: BTC / Bitcoin, ETH / Ethereum, etc.)
    2. Считаем вес bullish/bearish слов
    3. Если итоговый score превышает пороги — выдаём BUY или SELL
    4. Confidence = нормализованный score (0.0–1.0)

Параметры подгоняются в config.yaml → keyword_analyzer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from crypto_bot.analyzers.base import BaseAnalyzer
from crypto_bot.models.domain import Action, NewsItem, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Дефолтные словари (можно переопределить через конфиг)
# ---------------------------------------------------------------------------

DEFAULT_BULLISH_WORDS: list[str] = [
    # рост
    "surge", "surges", "surging", "rally", "rallies", "rallying",
    "pump", "pumps", "pumping", "moon", "mooning", "breakout",
    "ath", "all-time high", "record high", "new high",
    # позитивные события
    "adoption", "launch", "launches", "partnership", "partnerships",
    "approval", "approved", "approves", "etf", "listing", "listed",
    "upgrade", "integration", "invest", "investment", "backing",
    "bullish", "positive", "growth", "gains", "gain", "rises", "rise",
    "recovery", "recover", "rebounds", "rebound", "soars", "soar",
    "milestone", "accumulate", "accumulation", "buy", "bought",
]

DEFAULT_BEARISH_WORDS: list[str] = [
    # падение
    "crash", "crashes", "crashing", "dump", "dumps", "dumping",
    "plunge", "plunges", "plunging", "drop", "drops", "dropping",
    "fall", "falls", "falling", "collapse", "collapses",
    "correction", "bear", "bearish", "bottom", "loss", "losses",
    "decline", "declines", "declining", "tumble", "tumbles",
    # негативные события
    "ban", "bans", "banned", "hack", "hacked", "hacker", "exploit",
    "fraud", "scam", "ponzi", "rug", "rug pull", "liquidation",
    "liquidated", "fine", "fined", "lawsuit", "investigation",
    "warning", "risk", "vulnerability", "breach", "stolen",
    "shutdown", "delist", "delisted", "regulation", "crackdown",
    "sell", "sold", "fear", "panic",
]

# Алиасы тикеров: символ → все варианты написания
DEFAULT_TICKER_ALIASES: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin", "xbt"],
    "ETH": ["eth", "ethereum", "ether"],
    "SOL": ["sol", "solana"],
    "BNB": ["bnb", "binance coin", "binance"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["ada", "cardano"],
    "DOGE": ["doge", "dogecoin"],
    "AVAX": ["avax", "avalanche"],
    "DOT": ["dot", "polkadot"],
    "MATIC": ["matic", "polygon"],
    "LINK": ["link", "chainlink"],
    "LTC": ["ltc", "litecoin"],
}


# ---------------------------------------------------------------------------
# Dataclass для конфигурации
# ---------------------------------------------------------------------------

@dataclass
class KeywordConfig:
    """Настройки keyword анализатора."""
    bullish_words: list[str] = field(default_factory=lambda: list(DEFAULT_BULLISH_WORDS))
    bearish_words: list[str] = field(default_factory=lambda: list(DEFAULT_BEARISH_WORDS))
    ticker_aliases: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_TICKER_ALIASES)
    )
    # Минимальный перевес bullish/bearish слов для сигнала (в штуках)
    # 1 = достаточно одного слова-перевеса (чувствительно)
    # 2 = нужно на 2 слова больше в одну сторону (консервативно)
    buy_threshold: float = 1
    sell_threshold: float = 1
    # Бонус за упоминание целевого тикера в тексте
    ticker_match_bonus: float = 0.1
    # Симулируемая задержка (мс)
    latency_ms: int = 10


# ---------------------------------------------------------------------------
# Анализатор
# ---------------------------------------------------------------------------

class KeywordAnalyzer(BaseAnalyzer):
    """Rule-based анализатор по ключевым словам и тикерам.

    Args:
        config: объект KeywordConfig (параметры из YAML).
        tracked_coins: список монет, которые мы отслеживаем (из основного конфига).
    """

    def __init__(self, config: KeywordConfig, tracked_coins: list[str] | None = None) -> None:
        self._cfg = config
        self._tracked_coins = [c.upper() for c in (tracked_coins or [])]

        # Компилируем regex-паттерны один раз
        self._bullish_re = self._compile_patterns(config.bullish_words)
        self._bearish_re = self._compile_patterns(config.bearish_words)
        self._ticker_re  = self._compile_ticker_patterns(config.ticker_aliases)

        logger.info(
            "KeywordAnalyzer ready | bullish_words=%d | bearish_words=%d | tickers=%d",
            len(config.bullish_words), len(config.bearish_words),
            len(config.ticker_aliases),
        )

    # ------------------------------------------------------------------
    # Компиляция паттернов
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_patterns(words: list[str]) -> re.Pattern:
        """Компилирует список слов в один regex (word boundaries)."""
        escaped = [re.escape(w) for w in sorted(words, key=len, reverse=True)]
        pattern = r"\b(?:" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    @staticmethod
    def _compile_ticker_patterns(
        aliases: dict[str, list[str]]
    ) -> dict[str, re.Pattern]:
        """Компилирует паттерны поиска тикеров по их алиасам."""
        result: dict[str, re.Pattern] = {}
        for ticker, alias_list in aliases.items():
            all_variants = [re.escape(v) for v in alias_list]
            # Добавляем сам тикер
            all_variants.insert(0, re.escape(ticker))
            pattern = r"\b(?:" + "|".join(all_variants) + r")\b"
            result[ticker] = re.compile(pattern, re.IGNORECASE)
        return result

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def analyze(self, news_item: NewsItem) -> Signal:
        text = news_item.headline
        text_lower = text.lower()

        # ── Считаем совпадения ─────────────────────────────────────────
        bullish_matches = self._bullish_re.findall(text_lower)
        bearish_matches = self._bearish_re.findall(text_lower)

        n_bull = len(bullish_matches)
        n_bear = len(bearish_matches)

        # ── Проверяем упоминание целевой монеты ───────────────────────
        coin = news_item.coin.upper()
        ticker_mentioned = self._coin_mentioned_in_text(coin, text)
        ticker_bonus = self._cfg.ticker_match_bonus if ticker_mentioned else 0.0

        # ── Вычисляем score ───────────────────────────────────────────
        # net = разница совпадений (абсолютная, не нормализованная)
        # Threshold — минимальное кол-во слов перевеса для сигнала
        net_score = n_bull - n_bear

        # Confidence: насколько однозначна картина
        # Если n_bull=3, n_bear=0 → conf=1.0. Если n_bull=2, n_bear=1 → conf=0.5
        total_hits = n_bull + n_bear
        if total_hits > 0:
            directional_conf = abs(net_score) / total_hits
        else:
            directional_conf = 0.0

        # Бонус за упоминание тикера увеличивает уверенность сигнала
        if ticker_mentioned:
            directional_conf = min(directional_conf + ticker_bonus, 1.0)

        # ── Принимаем решение ─────────────────────────────────────────
        # buy_threshold/sell_threshold — минимальный перевес в словах
        if net_score >= self._cfg.buy_threshold and n_bull > 0:
            action = Action.BUY
            confidence = max(directional_conf, 0.5)
        elif net_score <= -self._cfg.sell_threshold and n_bear > 0:
            action = Action.SELL
            confidence = max(directional_conf, 0.5)
        else:
            action = Action.HOLD
            confidence = 1.0 - directional_conf

        logger.debug(
            "Keyword | coin=%s | bull=%d(%s) bear=%d(%s) ticker=%s | "
            "net=%.3f → %s (conf=%.3f)",
            coin, n_bull, bullish_matches[:3], n_bear, bearish_matches[:3],
            ticker_mentioned, net_score, action.value, confidence,
        )

        return Signal(
            action=action,
            confidence=confidence,
            raw_label=f"bull={n_bull},bear={n_bear},net={net_score:+.3f}",
            text=text,
            news_timestamp=news_item.timestamp,
            total_latency_ms=self._cfg.latency_ms,
        )

    def _coin_mentioned_in_text(self, coin: str, text: str) -> bool:
        """Возвращает True если монета (или её алиасы) упомянута в тексте."""
        pattern = self._ticker_re.get(coin)
        if pattern is None:
            # Монета не в словаре алиасов — ищем просто тикер
            return bool(re.search(r"\b" + re.escape(coin) + r"\b", text, re.IGNORECASE))
        return bool(pattern.search(text))


# ---------------------------------------------------------------------------
# Фабричная функция для создания из конфига YAML
# ---------------------------------------------------------------------------

def keyword_analyzer_from_config(cfg_dict: dict, tracked_coins: list[str]) -> KeywordAnalyzer:
    """Создаёт KeywordAnalyzer из секции конфига keyword_analyzer."""
    kw_cfg = KeywordConfig(
        buy_threshold=cfg_dict.get("buy_threshold", 0.3),
        sell_threshold=cfg_dict.get("sell_threshold", 0.3),
        ticker_match_bonus=cfg_dict.get("ticker_match_bonus", 0.1),
        latency_ms=cfg_dict.get("latency_ms", 10),
    )
    # Можно расширять словари через конфиг
    extra_bullish = cfg_dict.get("extra_bullish_words", [])
    extra_bearish = cfg_dict.get("extra_bearish_words", [])
    kw_cfg.bullish_words.extend(extra_bullish)
    kw_cfg.bearish_words.extend(extra_bearish)

    return KeywordAnalyzer(config=kw_cfg, tracked_coins=tracked_coins)
