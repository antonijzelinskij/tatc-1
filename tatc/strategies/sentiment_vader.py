"""
VADER-based sentiment strategy.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
sentiment analyzer — no model loading, ~0ms latency, perfect for speed-critical trading.

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


class VaderStrategy:
    """
    Analyzes news sentiment using VADER and returns a trading signal.

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
