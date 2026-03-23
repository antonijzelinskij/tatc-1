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
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tatc.core.models import NewsItem, Signal, SignalType
from tatc.strategies.sentiment_vader import CRYPTO_LEXICON

# ─── Keyword categories ───────────────────────────────────────────────────────

# Major actionable events — lower compound threshold applies when found.
# NOTE: "ath"/"all-time high" removed — hitting ATH is often a local top (sell-the-news).
#       "approaching ath", "on track for ath" are bullish but too rare to keyword-match safely.
STRONG_BUY_KEYWORDS: frozenset[str] = frozenset({
    "halving", "halvening",
    "etf approved", "etf approval", "sec approves", "sec approved",
    "listed on binance", "listed on coinbase",
    "listing on binance", "listing on coinbase",
    "partnership", "acquisition", "acquires",
    "blackrock", "fidelity", "microstrategy",
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
    # Recovery/bounce patterns — reliably fail (data-driven: every bounce trade was a loss)
    "bounces back", "bounce back", "bounced back",
    "dip buyers", "dip buyer", "buying the dip",
    "finds support", "finding support", "found support",
    "recovery rally", "attempts recovery", "attempting recovery",
    "consolidation phase", "healthy pullback", "healthy correction",
    "back above", "reclaims",
    # Promotional/clickbait/presale — not actionable market signals
    "next crypto to explode", "is this the next", "early birds",
    "promises even greater", "promises huge", "huge returns",
    "best crypto to buy", "top crypto to buy", "altcoin to buy",
    "under the radar", "new crypto", "this crypto promises",
    "deposit dash", "promo code",
    "draw early", "seeking huge", "volatile dogecoin",
    "presale", "pre-sale", "airdrop", "x roi", "000x",
    "alternatives to", "benefits from the", "built on ethereum",
    # Bearish facts VADER mislabels as positive (real data finding)
    "blow-off top", "blowoff top",
    "downside target", "bearish target",
    "miners sell", "miners selling", "miner sell",
    "sharp decline in inflows", "inflows falling", "inflows decline",
    "etf hopes fade", "etf approval hopes fade",
    "wants to categorize", "security classification",
    "dethrones", "winning streak against",
    # Negative outlook phrases VADER fails to catch
    "remain dim", "remains dim", "odds slump", "approval odds",
    "warns bitcoin", "warns ethereum", "warns crypto",
    "will retrace", "deep retrace",
    "sec investigates", "sec investigation",
    "whale sells", "whale selling", "whale sold",
    "surpasses ethereum", "overtakes ethereum",
    "two rebounds", "massive bounce",
})

# "Buy the rumour, sell the news" patterns — event already priced in, price often reverses.
# When found in a BUY context, apply a strong penalty to prevent chasing the move.
# Data source: all 5 biggest losses were post-event confirmation articles.
SELL_THE_NEWS_KEYWORDS: frozenset[str] = frozenset({
    "profit taking", "profit-taking",
    "pulls it back", "pulls back", "pulled back",
    "sell the news", "selling the news",
    "correction after", "pullback after",
    "halving complete", "halving successful",
    "upgrade successful", "upgrade complete", "upgrade live",
    "etf approval rally", "post-etf", "post-halving",
    "euphoric after", "euphoria after",
    "first day trading",
    "sell-off after", "selloff after",
    "already surged", "already rallied", "already up",
    # New ATH = often local top; "approaching ATH" is fine but "hits/new ATH" is dangerous
    "new all-time high", "new ath", "hits ath", "hit ath", "reached ath",
    # Countdown phrases right before a known event — markets price it in, then dump
    "hours away", "days away",
    # Confirmed-bad events that sound neutral but are bearish
    "sec wants to classify", "sec classifies", "classified as security",
    "inflows down", "outflows spike", "etf outflows",
})

# Regex: article reports an already-completed % move — price clearly moved before we enter.
# Catches: "surges 20 percent", "up 80 percent", "gains 15%", "rallied 12 percent", etc.
_ALREADY_MOVED_RE = re.compile(
    r"\b(surges?|jumped?|rallied?|soared?|gained?|climbed?|up|rose|skyrocketed?)"
    r"\s+\d+\s*(%|percent)\b",
    re.IGNORECASE,
)

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
_SELL_THE_NEWS_PENALTY = 0.25         # heavy penalty: event already priced in
_FINBERT_OVERRIDE_THRESHOLD = 0.7    # skip finbert veto if |compound| > this


# Coins that are high-volatility and poorly predicted by news sentiment alone.
# Their threshold is boosted by this multiplier (e.g. 0.45 × 1.4 = 0.63 required).
_ALTCOIN_THRESHOLD_MULTIPLIER = 1.4
_ALTCOIN_COINS: frozenset[str] = frozenset({"SOL", "DOGE", "SHIB", "PEPE", "FLOKI",
                                             "BONK", "WIF", "MEME", "TURBO"})


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
        use_haiku:            enable Claude Haiku veto (default False)
        haiku_threshold:      min Haiku confidence to veto when it says opposite direction (default 0.6)
                              Note: HOLD vetoes always fire regardless of confidence.
        haiku_offline:        cache-only mode: skip API calls, only use cached results (default False)
        haiku_api_key:        Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        haiku_cache_dir:      directory to cache Haiku results (default: data/cache)
        altcoin_penalty:      if True, raise threshold for high-volatility altcoins (default True)
        default_symbol:       fallback symbol when news.coins is empty
    """

    def __init__(
        self,
        buy_threshold: float = 0.45,
        sell_threshold: float = -0.45,
        min_relevance: float = 0.75,
        cooldown_minutes: int = 30,
        dedup_window_minutes: int = 120,
        use_finbert: bool = False,
        finbert_threshold: float = 0.55,
        use_haiku: bool = False,
        haiku_threshold: float = 0.6,
        haiku_offline: bool = False,
        haiku_api_key: str | None = None,
        haiku_cache_dir: str = "data/cache",
        altcoin_penalty: bool = True,
        default_symbol: str = "BTCUSDT",
    ):
        self._vader = SentimentIntensityAnalyzer()
        self._vader.lexicon.update(CRYPTO_LEXICON)

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_relevance = min_relevance
        self.cooldown_minutes = cooldown_minutes
        self.dedup_window_minutes = dedup_window_minutes
        self.altcoin_penalty = altcoin_penalty
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

        # Optional Claude Haiku voter
        self._haiku: Optional[object] = None
        self._haiku_threshold = haiku_threshold
        if use_haiku:
            from tatc.strategies.sentiment_haiku import ClaudeHaikuStrategy
            mode = "offline/cache-only" if haiku_offline else "online"
            print(f"  [Haiku] Initializing Claude Haiku voter ({mode})...", flush=True)
            self._haiku = ClaudeHaikuStrategy(
                api_key=haiku_api_key,
                cache_dir=haiku_cache_dir,
                offline=haiku_offline,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _symbol_from_news(self, news: NewsItem) -> str:
        """
        Pick the coin that appears earliest in the article title.
        Falls back to coins[0] if none appear in the title.
        This prevents always trading BTC when BTC is merely listed alongside
        the true subject (e.g. "Solana DEX Volume Surpasses Ethereum [BTC,SOL,ETH]").
        """
        if not news.coins:
            return self.default_symbol
        title_lower = news.title.lower()
        best_coin = news.coins[0]
        best_pos: int | None = None
        for coin in news.coins:
            names = [coin.lower()]
            for name, ticker in _COIN_NAMES.items():
                if ticker == coin.upper():
                    names.append(name)
            for name in names:
                idx = title_lower.find(name)
                if idx != -1 and (best_pos is None or idx < best_pos):
                    best_pos = idx
                    best_coin = coin
        return f"{best_coin}USDT"

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
          1.0 — coin name/ticker is first crypto mentioned in title
          0.7 — coin name/ticker in title but other coins appear first
          0.8 — coin mentioned 3+ times in body
          0.6 — coin mentioned 1-2 times in body
          0.4 — coin only in tags, not in text
          0.5 — no coin info

        Penalty when another crypto name appears BEFORE the target coin in the title —
        catches "PEPE and WIF Rally as Bitcoin remains..." (BTC is secondary subject).
        """
        if not news.coins:
            return 0.5

        title_lower = news.title.lower()
        body_lower = (news.body or "").lower()
        full_text = f"{title_lower} {body_lower}"

        best = 0.0
        for coin in news.coins:
            names = [coin.lower()]
            for name, ticker in _COIN_NAMES.items():
                if ticker == coin.upper():
                    names.append(name)

            title_pos = None
            for name in names:
                idx = title_lower.find(name)
                if idx != -1:
                    if title_pos is None or idx < title_pos:
                        title_pos = idx

            if title_pos is not None:
                # Check if any OTHER known crypto name appears before our coin in the title
                other_before = False
                for other_name, other_ticker in _COIN_NAMES.items():
                    if other_ticker == coin.upper():
                        continue
                    other_idx = title_lower.find(other_name)
                    if other_idx != -1 and other_idx < title_pos:
                        other_before = True
                        break
                best = max(best, 0.7 if other_before else 1.0)
            else:
                count = full_text.count(coin.lower())
                for name in names[1:]:  # full names only in body check
                    count = max(count, full_text.count(name))
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

    def _sell_the_news_multiplier(self, text: str) -> float:
        """
        Detect 'buy the rumour, sell the news' patterns.
        Returns a heavy penalty multiplier when the article describes a completed
        event that the market has already reacted to.
        Checks both a keyword list and a regex for "surges/up X percent" patterns.
        """
        for phrase in SELL_THE_NEWS_KEYWORDS:
            if phrase in text:
                return _SELL_THE_NEWS_PENALTY
        # Regex: article reports an already-completed % gain
        if _ALREADY_MOVED_RE.search(text):
            return _SELL_THE_NEWS_PENALTY
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
        sell_news_mult = self._sell_the_news_multiplier(text)
        relevance_mult = min(relevance, 1.0)  # score above 1.0 has no extra benefit

        adjusted = compound * noise_mult * urgency_mult * sell_news_mult * relevance_mult
        adjusted = max(-1.0, min(1.0, adjusted))  # clamp

        # 6. Altcoin threshold boost — high-volatility coins require stronger signal
        coin = news.coins[0].upper() if news.coins else ""
        effective_buy_threshold = self.buy_threshold
        effective_sell_threshold = self.sell_threshold
        if self.altcoin_penalty and coin in _ALTCOIN_COINS:
            effective_buy_threshold = self.buy_threshold * _ALTCOIN_THRESHOLD_MULTIPLIER
            effective_sell_threshold = self.sell_threshold * _ALTCOIN_THRESHOLD_MULTIPLIER

        # 7. Determine signal type
        strong = self._strong_keyword_signal(text)

        # Strong keyword override is disabled when sell-the-news is detected
        # (e.g. "halving complete" has "halving" strong buy but is a sell-the-news)
        strong_allowed = sell_news_mult == 1.0

        if strong_allowed and strong == SignalType.BUY and adjusted >= _STRONG_KEYWORD_MIN_COMPOUND:
            signal_type = SignalType.BUY
        elif strong_allowed and strong == SignalType.SELL and adjusted <= -_STRONG_KEYWORD_MIN_COMPOUND:
            signal_type = SignalType.SELL
        elif adjusted >= effective_buy_threshold:
            signal_type = SignalType.BUY
        elif adjusted <= effective_sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # 7. FinBERT veto (optional)
        if signal_type != SignalType.HOLD and self._finbert is not None:
            fb = self._finbert.analyze(news)
            if fb.signal != signal_type and abs(adjusted) < _FINBERT_OVERRIDE_THRESHOLD:
                return self._make_hold(news, f"finbert_veto:{fb.signal.value}", compound)

        # 8. Claude Haiku veto (optional) — Haiku has domain knowledge VADER lacks.
        #    Unlike FinBERT, Haiku results are cached so repeated runs are free.
        #    Variant A: if Haiku says HOLD, skip the trade (no confidence gate —
        #    HOLD is naturally low-confidence; requiring 0.6 would suppress all vetoes).
        #    If Haiku says the opposite direction with high confidence, also veto.
        if signal_type != SignalType.HOLD and self._haiku is not None:
            hk = self._haiku.analyze(news)
            # confidence=-1 means "no cache hit in offline mode" → skip veto
            if hk.confidence >= 0:
                if hk.signal == SignalType.HOLD:
                    return self._make_hold(news, f"haiku_veto:HOLD:{hk.confidence:.2f}", compound)
                if hk.signal != signal_type and hk.confidence >= self._haiku_threshold:
                    return self._make_hold(news, f"haiku_veto:{hk.signal.value}:{hk.confidence:.2f}", compound)

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
                "sell_news_mult": sell_news_mult,
                "altcoin_penalty": coin in _ALTCOIN_COINS,
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
