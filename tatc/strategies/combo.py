"""
Combo strategy: VADER + keyword filter + optional FinBERT voting.

Reduces false signals vs pure VADER through:
  1. Noise filter      — ignores historical/speculative context
  2. Relevance score   — coin must actually be the subject of the article
  3. Strong keywords   — direct triggers for major events (ETF, hack, halving...)
  4. Urgency boost     — "breaking", "confirmed" raises confidence
  5. Deduplication     — title hash + per-coin cooldown window
  6. FinBERT vote      — optional: requires VADER + FinBERT to agree

Usage:
  from tatc.strategies.combo import ComboStrategy
  strategy = ComboStrategy()                       # defaults
  strategy = ComboStrategy(use_finbert=True)       # with FinBERT voting
  strategy = ComboStrategy(cooldown_minutes=60)    # tighter cooldown
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tatc.core.models import NewsItem, Signal, SignalType
from tatc.strategies.sentiment_vader import CRYPTO_LEXICON

# ─── Keyword categories ───────────────────────────────────────────────────────

# Major actionable events — lower compound threshold applies when found
STRONG_BUY_KEYWORDS: frozenset[str] = frozenset({
    "halving", "halvening",
    "etf approved", "etf approval", "sec approves", "sec approved",
    "listed on binance", "listed on coinbase",
    "listing on binance", "listing on coinbase",
    "partnership", "acquisition", "acquires",
    "blackrock", "fidelity", "microstrategy",
    "all-time high", "ath",
    "major upgrade", "protocol upgrade", "mainnet launch",
    "institutional", "buys bitcoin", "adds bitcoin",
    "accumulating", "whale accumulation", "record inflows",
})

STRONG_SELL_KEYWORDS: frozenset[str] = frozenset({
    "hack", "hacked", "exploit", "exploited",
    "exit scam", "rug pull", "rugpull", "rug-pull",
    "sec charges", "sec sues", "doj charges",
    "bankruptcy", "insolvent", "insolvency",
    "fraud", "ponzi",
    "arrested", "arrest",
    "suspended withdrawals", "withheld withdrawals",
    "exchange collapse", "platform collapse",
    "massive selloff", "panic selling",
})

# Recent/live event markers — boost confidence
URGENCY_KEYWORDS: frozenset[str] = frozenset({
    "breaking", "just announced", "alert", "urgent",
    "confirmed", "developing", "just launched", "just listed",
    "now live", "today",
})

# Historical/speculative markers — reduce confidence
NOISE_KEYWORDS: frozenset[str] = frozenset({
    "remember when", "back in", "last year", "last cycle",
    "in 2017", "in 2018", "in 2019", "in 2020", "in 2021",
    "historically", "years ago", "months ago", "used to",
    "history of", "looking back",
    "price prediction", "price target",
    "analysts predict", "some analysts",
    "could reach", "might reach",
    "if bitcoin", "could happen",
})

# Coin full names → ticker for relevance scoring
_COIN_NAMES: dict[str, str] = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "cardano": "ADA", "dogecoin": "DOGE", "ripple": "XRP",
    "polkadot": "DOT", "chainlink": "LINK", "avalanche": "AVAX",
    "uniswap": "UNI", "litecoin": "LTC", "bnb": "BNB",
    "binance coin": "BNB", "tron": "TRX", "stellar": "XLM",
    "monero": "XMR", "matic": "MATIC", "polygon": "MATIC",
    "near protocol": "NEAR", "near": "NEAR",
    "algorand": "ALGO", "cosmos": "ATOM",
    "shiba inu": "SHIB", "pepe": "PEPE", "worldcoin": "WLD",
}

# Tuning constants
_STRONG_KEYWORD_MIN_COMPOUND = 0.15   # compound must cross this even with strong keyword
_URGENCY_BOOST = 1.2                  # multiply adjusted_compound by this
_NOISE_PENALTY = 0.4                  # multiply adjusted_compound by this
_FINBERT_OVERRIDE_THRESHOLD = 0.7    # skip finbert veto if |compound| > this


class ComboStrategy:
    """
    Combined VADER + keyword filter strategy.

    Args:
        buy_threshold:        VADER compound >= this → BUY (default 0.45)
        sell_threshold:       VADER compound <= this → SELL (default -0.45)
        min_relevance:        articles below this relevance score → HOLD (default 0.4)
        cooldown_minutes:     per-coin silence window after a signal (default 30)
        dedup_window_minutes: ignore re-occurrence of same title within N min (default 120)
        use_finbert:          require FinBERT to agree with VADER signal (default False)
        finbert_threshold:    min FinBERT confidence for vote (default 0.55)
        default_symbol:       fallback symbol when news.coins is empty
    """

    def __init__(
        self,
        buy_threshold: float = 0.45,
        sell_threshold: float = -0.45,
        min_relevance: float = 0.4,
        cooldown_minutes: int = 30,
        dedup_window_minutes: int = 120,
        use_finbert: bool = False,
        finbert_threshold: float = 0.55,
        default_symbol: str = "BTCUSDT",
    ):
        self._vader = SentimentIntensityAnalyzer()
        self._vader.lexicon.update(CRYPTO_LEXICON)

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_relevance = min_relevance
        self.cooldown_minutes = cooldown_minutes
        self.dedup_window_minutes = dedup_window_minutes
        self.default_symbol = default_symbol

        # Deduplication state (reset between runs)
        self._seen_hashes: dict[str, datetime] = {}
        self._coin_last_signal: dict[str, datetime] = {}

        # Optional FinBERT voter
        self._finbert: Optional[object] = None
        if use_finbert:
            from tatc.strategies.sentiment_finbert import FinBertStrategy
            self._finbert = FinBertStrategy(
                buy_threshold=finbert_threshold,
                sell_threshold=finbert_threshold,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _symbol_from_news(self, news: NewsItem) -> str:
        if news.coins:
            return f"{news.coins[0]}USDT"
        return self.default_symbol

    def _title_hash(self, title: str) -> str:
        return hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]

    def _is_duplicate(self, news: NewsItem) -> bool:
        h = self._title_hash(news.title)
        cutoff = timedelta(minutes=self.dedup_window_minutes)
        now = news.timestamp
        if h in self._seen_hashes and (now - self._seen_hashes[h]) < cutoff:
            return True
        self._seen_hashes[h] = now
        return False

    def _is_on_cooldown(self, symbol: str, news: NewsItem) -> bool:
        last = self._coin_last_signal.get(symbol)
        if last is None:
            return False
        return (news.timestamp - last) < timedelta(minutes=self.cooldown_minutes)

    def _mark_signal(self, symbol: str, ts: datetime) -> None:
        self._coin_last_signal[symbol] = ts

    def _relevance_score(self, news: NewsItem) -> float:
        """
        How much is this article actually about the coin(s)?
          1.0 — coin name/ticker in title
          0.8 — coin mentioned 3+ times in body
          0.6 — coin mentioned 1-2 times in body
          0.4 — coin only in tags (news.coins), not in text
          0.5 — no coin info at all
        """
        if not news.coins:
            return 0.5

        title_lower = news.title.lower()
        body_lower = (news.body or "").lower()
        full_text = f"{title_lower} {body_lower}"

        best = 0.0
        for coin in news.coins:
            # Build list of names to check: ticker + full name(s)
            names = [coin.lower()]
            for name, ticker in _COIN_NAMES.items():
                if ticker == coin.upper():
                    names.append(name)

            for name in names:
                if name in title_lower:
                    best = max(best, 1.0)
                    break
                count = full_text.count(name)
                if count >= 3:
                    best = max(best, 0.8)
                elif count >= 1:
                    best = max(best, 0.6)
            else:
                best = max(best, 0.4)

        return best

    def _noise_multiplier(self, text: str) -> float:
        for phrase in NOISE_KEYWORDS:
            if phrase in text:
                return _NOISE_PENALTY
        return 1.0

    def _urgency_multiplier(self, text: str) -> float:
        for phrase in URGENCY_KEYWORDS:
            if phrase in text:
                return _URGENCY_BOOST
        return 1.0

    def _strong_keyword_signal(self, text: str) -> Optional[SignalType]:
        for phrase in STRONG_BUY_KEYWORDS:
            if phrase in text:
                return SignalType.BUY
        for phrase in STRONG_SELL_KEYWORDS:
            if phrase in text:
                return SignalType.SELL
        return None

    def _make_hold(self, news: NewsItem, reason: str, compound: float = 0.0) -> Signal:
        return Signal(
            signal=SignalType.HOLD,
            confidence=0.0,
            symbol=self._symbol_from_news(news),
            news=news,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={"reason": reason, "compound": compound},
        )

    # ── Main analysis ─────────────────────────────────────────────────────────

    def analyze(self, news: NewsItem) -> Signal:
        symbol = self._symbol_from_news(news)
        text = f"{news.title} {news.body or ''}".lower()

        # 1. Skip duplicates
        if self._is_duplicate(news):
            return self._make_hold(news, "duplicate")

        # 2. Skip if coin on cooldown
        if self._is_on_cooldown(symbol, news):
            return self._make_hold(news, "cooldown")

        # 3. Relevance gate
        relevance = self._relevance_score(news)
        if relevance < self.min_relevance:
            return self._make_hold(news, f"low_relevance:{relevance:.2f}")

        # 4. VADER compound (title weighted 2x)
        vader_text = f"{news.title}. {news.title}. {news.body or ''}"
        scores = self._vader.polarity_scores(vader_text)
        compound = scores["compound"]

        # 5. Apply context multipliers
        noise_mult = self._noise_multiplier(text)
        urgency_mult = self._urgency_multiplier(text)
        relevance_mult = min(relevance, 1.0)  # score above 1.0 has no extra benefit

        adjusted = compound * noise_mult * urgency_mult * relevance_mult
        adjusted = max(-1.0, min(1.0, adjusted))  # clamp

        # 6. Determine signal type
        strong = self._strong_keyword_signal(text)

        if strong == SignalType.BUY and adjusted >= _STRONG_KEYWORD_MIN_COMPOUND:
            signal_type = SignalType.BUY
        elif strong == SignalType.SELL and adjusted <= -_STRONG_KEYWORD_MIN_COMPOUND:
            signal_type = SignalType.SELL
        elif adjusted >= self.buy_threshold:
            signal_type = SignalType.BUY
        elif adjusted <= self.sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # 7. FinBERT veto (optional)
        if signal_type != SignalType.HOLD and self._finbert is not None:
            fb = self._finbert.analyze(news)
            if fb.signal != signal_type and abs(adjusted) < _FINBERT_OVERRIDE_THRESHOLD:
                return self._make_hold(news, f"finbert_veto:{fb.signal.value}", compound)

        confidence = abs(adjusted)

        if signal_type != SignalType.HOLD:
            self._mark_signal(symbol, news.timestamp)

        return Signal(
            signal=signal_type,
            confidence=confidence,
            symbol=symbol,
            news=news,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={
                "compound": compound,
                "adjusted_compound": adjusted,
                "relevance": relevance,
                "noise_mult": noise_mult,
                "urgency_mult": urgency_mult,
                "strong_keyword": strong.value if strong else None,
                **scores,
            },
        )

    def reset(self) -> None:
        """Clear deduplication and cooldown state. Call before each backtest run."""
        self._seen_hashes.clear()
        self._coin_last_signal.clear()

    def analyze_many(self, news_items: list[NewsItem]) -> list[Signal]:
        """Analyze a list of news items sequentially (order matters for dedup/cooldown)."""
        return [self.analyze(item) for item in news_items]
