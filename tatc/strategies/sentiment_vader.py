"""
VADER-based sentiment strategy.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
sentiment analyzer — no model loading, ~0ms latency, perfect for speed-critical trading.

Extended with a crypto-specific lexicon: VADER doesn't know that
"surges", "dumps", "ATH", "halving" etc. have financial meaning.

Signal logic:
  compound >= BUY_THRESHOLD  → BUY
  compound <= SELL_THRESHOLD → SELL
  otherwise                  → HOLD
"""
from datetime import datetime, timezone

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tatc.core.models import NewsItem, Signal, SignalType

# Default thresholds — tune via backtesting
_DEFAULT_BUY_THRESHOLD = 0.3
_DEFAULT_SELL_THRESHOLD = -0.3

# Crypto-specific sentiment scores (VADER uses -4 to +4 scale internally)
# Words VADER doesn't know or scores incorrectly in financial context
CRYPTO_LEXICON: dict[str, float] = {
    # Bullish
    "surges": 2.5, "surge": 2.5, "surging": 2.5,
    "pumps": 2.0, "pump": 1.5, "pumping": 2.0,
    "rallies": 2.5, "rally": 2.0, "rallying": 2.5,
    "bullish": 2.5, "bull": 1.5, "bulls": 1.5,
    "ath": 3.0, "all-time-high": 3.0,
    "breakout": 2.0, "breakthrough": 2.0,
    "halving": 1.5, "halvening": 1.5,
    "adoption": 1.5, "partnership": 1.5,
    "approval": 2.0, "approved": 2.5,
    "listing": 1.5, "listed": 1.5,
    "upgrade": 1.5, "launch": 1.0,
    "inflows": 2.0,
    "recovery": 1.5, "recovers": 1.5, "recovered": 1.5,
    "soars": 3.0, "soaring": 3.0,
    "skyrockets": 3.0,
    "mooning": 2.5,
    "accumulation": 1.5, "accumulate": 1.5,
    # Bearish
    "dumps": -2.5, "dump": -2.0, "dumping": -2.5,
    "crashes": -2.5, "crash": -2.5, "crashing": -2.5,
    "plunges": -2.5, "plunge": -2.5, "plunging": -2.5,
    "selloff": -2.0,
    "bearish": -2.5, "bear": -1.5, "bears": -1.5,
    "hack": -3.0, "hacked": -3.0, "exploit": -2.5, "exploited": -2.5,
    "scam": -3.0, "fraud": -3.0, "rug": -3.0, "rugpull": -3.0,
    "ban": -2.5, "banned": -2.5, "banning": -2.5,
    "lawsuit": -2.0, "sued": -2.0,
    "outflows": -2.0,
    "liquidations": -2.5, "liquidated": -2.5,
    "panic": -2.5, "fears": -2.0,
    "correction": -1.5,
    "downtrend": -2.0,
}


class VaderStrategy:
    """
    Analyzes news sentiment using VADER + crypto lexicon.
    Returns a trading signal (BUY/SELL/HOLD) with confidence score.

    Attributes:
        buy_threshold:  compound score >= this → BUY (range: 0 to 1)
        sell_threshold: compound score <= this → SELL (range: -1 to 0)
        default_symbol: symbol used when news.coins is empty
    """

    def __init__(
        self,
        buy_threshold: float = _DEFAULT_BUY_THRESHOLD,
        sell_threshold: float = _DEFAULT_SELL_THRESHOLD,
        default_symbol: str = "BTCUSDT",
    ):
        self._analyzer = SentimentIntensityAnalyzer()
        # Inject crypto lexicon into VADER
        self._analyzer.lexicon.update(CRYPTO_LEXICON)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.default_symbol = default_symbol

    def _text(self, news: NewsItem) -> str:
        """Combine title + body for analysis. Title weighted more."""
        if news.body:
            return f"{news.title}. {news.title}. {news.body}"
        return news.title

    def _symbol_from_news(self, news: NewsItem) -> str:
        """Pick trading symbol from news coins (first one + USDT) or default."""
        if news.coins:
            return f"{news.coins[0]}USDT"
        return self.default_symbol

    def analyze(self, news: NewsItem) -> Signal:
        """
        Run VADER sentiment on news title+body.
        Returns a Signal with type and confidence (abs of compound score).
        """
        scores = self._analyzer.polarity_scores(self._text(news))
        compound = scores["compound"]
        symbol = self._symbol_from_news(news)

        if compound >= self.buy_threshold:
            signal_type = SignalType.BUY
        elif compound <= self.sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        return Signal(
            signal=signal_type,
            confidence=abs(compound),
            symbol=symbol,
            news=news,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={"compound": compound, **scores},
        )

    def analyze_many(self, news_items: list[NewsItem]) -> list[Signal]:
        """Batch analyze a list of news items."""
        return [self.analyze(item) for item in news_items]
