"""
Claude Haiku-based sentiment strategy.

Uses claude-haiku-4-5-20251001 to analyze crypto news articles and return
a trading signal with confidence. Unlike FinBERT, Haiku has deep knowledge
of crypto-specific events (halving, ETF approvals, exchange hacks, etc.).

Caching: results are stored in a local JSON cache keyed by article content
hash. Re-running the same backtest is free — no API calls repeated.

Usage:
    strategy = ClaudeHaikuStrategy()                        # standalone
    strategy = ClaudeHaikuStrategy(cache_dir="data/cache")  # custom cache
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from tatc.core.models import NewsItem, Signal, SignalType

_MODEL = "claude-haiku-4-5-20251001"
_CACHE_FILENAME = "haiku_signal_cache.json"
_MAX_BODY_CHARS = 800   # truncate body to keep tokens low

_SYSTEM_PROMPT = """\
You are a crypto trading signal classifier. Given a news article headline \
(and optional body), output ONLY a JSON object with these fields:
  "signal":     "BUY" | "SELL" | "HOLD"
  "confidence": float between 0.0 and 1.0
  "reason":     one short sentence (max 15 words)

Rules:
- BUY:  clearly bullish news that is NEW and not yet priced in
        (ETF approval, institutional buying, mainnet launch, halving, \
major upgrade, record inflows, partnership with major firm)
- SELL: clearly bearish news (hack/exploit, regulatory action, bankruptcy, \
major selloff, fraud, exchange collapse, SEC charges)
- HOLD: speculative/opinion, price predictions, already-moved ("surges 20%"), \
historical recap, minor news, unclear impact
- If the headline is about a price move that ALREADY happened → HOLD \
(the opportunity is gone)
- Confidence = how certain you are the market will react to this news
Output ONLY the JSON. No markdown, no explanation outside the JSON."""


class ClaudeHaikuStrategy:
    """
    Analyzes news with Claude Haiku via the Anthropic API.

    Args:
        api_key:      Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        cache_dir:    directory to store the result cache (default: data/cache)
        max_retries:  API call retries on transient errors
        default_symbol: fallback symbol when news.coins is empty
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/cache",
        max_retries: int = 3,
        default_symbol: str = "BTCUSDT",
    ):
        import anthropic
        from dotenv import load_dotenv  # noqa: F401 — optional, ignored if absent

        try:
            load_dotenv()
        except Exception:
            pass

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key= or set the env var."
            )

        self._client = anthropic.Anthropic(api_key=key)
        self.default_symbol = default_symbol
        self.max_retries = max_retries

        # Disk cache
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._cache_file = cache_path / _CACHE_FILENAME
        self._cache: dict[str, dict] = self._load_cache()

        self._calls = 0   # track API calls this session
        self._hits = 0    # cache hits this session
        self._min_interval = 1.5  # seconds between API calls (40/min < 50/min limit)
        self._last_call_ts = 0.0

    # ── Cache ──────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, dict]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text())
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self._cache_file.write_text(json.dumps(self._cache, ensure_ascii=False))

    def _article_key(self, news: NewsItem) -> str:
        text = f"{news.title}|{(news.body or '')[:_MAX_BODY_CHARS]}"
        return hashlib.md5(text.encode()).hexdigest()

    # ── Symbol ────────────────────────────────────────────────────────────

    def _symbol_from_news(self, news: NewsItem) -> str:
        if news.coins:
            return f"{news.coins[0]}USDT"
        return self.default_symbol

    # ── API call ──────────────────────────────────────────────────────────

    def _call_api(self, news: NewsItem) -> dict:
        """Call Haiku and return parsed JSON result. Retries on transient errors."""
        body_snippet = (news.body or "")[:_MAX_BODY_CHARS]
        user_msg = f"Headline: {news.title}"
        if body_snippet:
            user_msg += f"\n\nBody: {body_snippet}"
        if news.coins:
            user_msg += f"\n\nCoins mentioned: {', '.join(news.coins)}"

        # Rate-limit: ensure minimum interval between API calls
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_ts = time.time()

        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=_MODEL,
                    max_tokens=100,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                result = json.loads(raw)
                self._calls += 1
                return result
            except json.JSONDecodeError:
                # Haiku returned malformed JSON — treat as HOLD
                return {"signal": "HOLD", "confidence": 0.0, "reason": "parse_error"}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [Haiku] API error after {self.max_retries} tries: {e}")
                    return {"signal": "HOLD", "confidence": 0.0, "reason": "api_error"}

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(self, news: NewsItem) -> Signal:
        key = self._article_key(news)

        if key in self._cache:
            self._hits += 1
            result = self._cache[key]
        else:
            result = self._call_api(news)
            self._cache[key] = result
            # Persist every 50 new calls to avoid losing work on crash
            if self._calls % 50 == 0:
                self._save_cache()

        raw_signal = result.get("signal", "HOLD").upper()
        confidence = float(result.get("confidence", 0.0))

        signal_map = {
            "BUY": SignalType.BUY,
            "SELL": SignalType.SELL,
            "HOLD": SignalType.HOLD,
        }
        signal_type = signal_map.get(raw_signal, SignalType.HOLD)

        return Signal(
            signal=signal_type,
            confidence=confidence,
            symbol=self._symbol_from_news(news),
            news=news,
            timestamp=datetime.now(tz=timezone.utc),
            metadata={
                "haiku_signal": raw_signal,
                "haiku_confidence": confidence,
                "haiku_reason": result.get("reason", ""),
                "cache_hit": key in self._cache,
            },
        )

    def flush_cache(self) -> None:
        """Force-write cache to disk."""
        self._save_cache()

    def cache_stats(self) -> dict:
        return {
            "total_cached": len(self._cache),
            "api_calls_this_session": self._calls,
            "cache_hits_this_session": self._hits,
        }

    def analyze_many(self, news_items: list[NewsItem]) -> list[Signal]:
        results = []
        total = len(news_items)
        for i, item in enumerate(news_items):
            results.append(self.analyze(item))
            if (i + 1) % 100 == 0:
                stats = self.cache_stats()
                print(
                    f"  [Haiku] {i+1}/{total} "
                    f"(API calls: {stats['api_calls_this_session']}, "
                    f"cache hits: {stats['cache_hits_this_session']})"
                )
        self.flush_cache()
        return results
