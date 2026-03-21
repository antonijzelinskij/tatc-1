"""
FinBERT-based sentiment strategy.
FinBERT is a BERT model fine-tuned on financial texts (ProsusAI/finbert).
Understands financial context, negations, and nuance that VADER misses.

Signal logic:
  POSITIVE label → BUY
  NEGATIVE label → SELL
  NEUTRAL  label → HOLD

Confidence = model's probability score for the predicted label.

Performance note: uses batched inference (~200ms/batch on CPU, ~20ms on GPU).
Model is loaded once at init and reused across calls.
"""
from datetime import datetime, timezone

from tatc.core.models import NewsItem, Signal, SignalType

# FinBERT model from HuggingFace Hub
_FINBERT_MODEL = "ProsusAI/finbert"

# Default thresholds for buy/sell confidence
_DEFAULT_BUY_THRESHOLD = 0.6
_DEFAULT_SELL_THRESHOLD = 0.6

# Default batch size for inference
_DEFAULT_BATCH_SIZE = 32

# Max text length (FinBERT is BERT-based: 512 tokens)
_MAX_TEXT_LEN = 512


class FinBertStrategy:
    """
    Analyzes news sentiment using FinBERT (ProsusAI/finbert).
    Returns a trading signal (BUY/SELL/HOLD) with confidence score.

    Attributes:
        buy_threshold:   min confidence to emit BUY (positive label)
        sell_threshold:  min confidence to emit SELL (negative label)
        default_symbol:  symbol used when news.coins is empty
        batch_size:      items per inference batch
    """

    def __init__(
        self,
        buy_threshold: float = _DEFAULT_BUY_THRESHOLD,
        sell_threshold: float = _DEFAULT_SELL_THRESHOLD,
        default_symbol: str = "BTCUSDT",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: int = -1,  # -1 = CPU, 0 = first GPU
    ):
        from transformers import pipeline

        print(f"  [FinBERT] Loading model {_FINBERT_MODEL}...", flush=True)
        self._pipe = pipeline(
            "text-classification",
            model=_FINBERT_MODEL,
            tokenizer=_FINBERT_MODEL,
            device=device,
            truncation=True,
            max_length=_MAX_TEXT_LEN,
        )
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.default_symbol = default_symbol
        self.batch_size = batch_size
        print(f"  [FinBERT] Ready.", flush=True)

    def _text(self, news: NewsItem) -> str:
        """Combine title + body. Title is the most signal-rich part."""
        if news.body:
            # Keep it under token limit: title twice for emphasis + truncated body
            combined = f"{news.title}. {news.title}. {news.body}"
            return combined[:_MAX_TEXT_LEN * 4]  # rough char limit before tokenization
        return news.title

    def _symbol_from_news(self, news: NewsItem) -> str:
        if news.coins:
            return f"{news.coins[0]}USDT"
        return self.default_symbol

    def _label_to_signal(self, label: str, score: float) -> SignalType:
        label = label.upper()
        if label == "POSITIVE" and score >= self.buy_threshold:
            return SignalType.BUY
        if label == "NEGATIVE" and score >= self.sell_threshold:
            return SignalType.SELL
        return SignalType.HOLD

    def analyze(self, news: NewsItem) -> Signal:
        """Run FinBERT on a single news item."""
        result = self._pipe(self._text(news))[0]
        label = result["label"]
        score = result["score"]
        signal_type = self._label_to_signal(label, score)

        return Signal(
            signal=signal_type,
            confidence=score,
            symbol=self._symbol_from_news(news),
            news=news,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={"finbert_label": label, "finbert_score": score},
        )

    def analyze_many(self, news_items: list[NewsItem]) -> list[Signal]:
        """
        Batch analyze a list of news items.
        Uses batched inference for efficiency (~10x faster than one-by-one).
        """
        if not news_items:
            return []

        texts = [self._text(item) for item in news_items]
        total = len(texts)
        results = []

        print(f"  [FinBERT] Analyzing {total} articles in batches of {self.batch_size}...",
              flush=True)

        for start in range(0, total, self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            batch_results = self._pipe(batch_texts)
            results.extend(batch_results)

            done = min(start + self.batch_size, total)
            if done % (self.batch_size * 10) == 0 or done == total:
                print(f"  [FinBERT] {done}/{total} done", flush=True)

        signals = []
        for news, result in zip(news_items, results):
            label = result["label"]
            score = result["score"]
            signal_type = self._label_to_signal(label, score)
            signals.append(Signal(
                signal=signal_type,
                confidence=score,
                symbol=self._symbol_from_news(news),
                news=news,
                timestamp=datetime.now(tz=timezone.utc),
                metadata={"finbert_label": label, "finbert_score": score},
            ))

        return signals
