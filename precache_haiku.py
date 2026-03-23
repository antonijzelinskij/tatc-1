#!/usr/bin/env python3
"""
Pre-populate the Haiku signal cache for all VADER BUY/SELL articles.

1. Loads cached news + runs combo strategy (no Haiku) to find VADER signals
2. Filters out articles already in the Haiku cache
3. Calls Haiku concurrently (async) up to the API rate limit
4. Saves progress every --save-every calls → safe to interrupt and resume

Speed depends on your API tier:
  Tier 1  (default)  : ~50 req/min  → ~75 min for 3551 articles
  Tier 2             : ~1000 req/min → ~4 min
  Custom             : use --rate N to override

Usage:
    python precache_haiku.py
    python precache_haiku.py --rate 1000 --concurrency 50   # Tier 2+
    python precache_haiku.py --dry-run                       # count only, no API calls
    python precache_haiku.py --coins BTC ETH --start 2024-01-01 --end 2024-04-01
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tatc.core.models import NewsItem, SignalType
from tatc.strategies.combo import ComboStrategy
from tatc.strategies.sentiment_haiku import _SYSTEM_PROMPT, _MAX_BODY_CHARS, _CACHE_FILENAME


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_news(cache_dir: Path, coins: list[str], start: str, end: str) -> list[NewsItem]:
    coin_str = "_".join(sorted(coins))
    path = cache_dir / f"news_{coin_str}_{start}_{end}.json"
    if not path.exists():
        sys.exit(f"News cache not found: {path}\nRun real_backtest.py first to populate the news cache.")
    data = json.loads(path.read_text())
    items = []
    for d in data:
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        items.append(NewsItem(
            id=d["id"], timestamp=ts, title=d["title"],
            body=d["body"], source=d["source"], url=d["url"], coins=d["coins"],
        ))
    return items


def _article_key(news: NewsItem) -> str:
    text = f"{news.title}|{(news.body or '')[:_MAX_BODY_CHARS]}"
    return hashlib.md5(text.encode()).hexdigest()


def _find_vader_signals(news_items: list[NewsItem]) -> list[NewsItem]:
    """Return articles where combo (no Haiku) generates BUY or SELL."""
    strategy = ComboStrategy()
    signals = strategy.analyze_many(news_items)
    return [s.news for s in signals if s.signal != SignalType.HOLD]


# ── async rate limiter ─────────────────────────────────────────────────────────

class _RateLimiter:
    """Global async rate limiter: allows at most `rate` calls per minute."""

    def __init__(self, rate_per_min: int):
        self._interval = 60.0 / rate_per_min
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


# ── async Haiku call ───────────────────────────────────────────────────────────

async def _call_haiku(client, news: NewsItem, semaphore: asyncio.Semaphore,
                      limiter: _RateLimiter) -> dict:
    user_msg = f"Headline: {news.title}"
    body = (news.body or "")[:_MAX_BODY_CHARS]
    if body:
        user_msg += f"\n\nBody: {body}"
    if news.coins:
        user_msg += f"\n\nCoins mentioned: {', '.join(news.coins)}"

    async with semaphore:
        await limiter.acquire()
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=100,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"signal": "HOLD", "confidence": 0.0, "reason": "parse_error"}
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() and attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                elif attempt < 2:
                    await asyncio.sleep(1)
                else:
                    return {"signal": "HOLD", "confidence": 0.0, "reason": f"api_error"}


# ── main async worker ──────────────────────────────────────────────────────────

async def run_precache(
    to_cache: list[NewsItem],
    cache: dict,
    cache_file: Path,
    api_key: str,
    rate_per_min: int,
    concurrency: int,
    save_every: int,
) -> None:
    import anthropic as _anthropic
    client = _anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    limiter = _RateLimiter(rate_per_min)

    total = len(to_cache)
    done = 0
    errors = 0
    start_ts = time.time()

    async def process(news: NewsItem) -> None:
        nonlocal done, errors
        key = _article_key(news)
        result = await _call_haiku(client, news, semaphore, limiter)
        cache[key] = result
        done += 1
        if "api_error" in result.get("reason", ""):
            errors += 1

        if done % save_every == 0 or done == total:
            cache_file.write_text(json.dumps(cache, ensure_ascii=False))
            elapsed = time.time() - start_ts
            rate_actual = done / (elapsed / 60) if elapsed > 0 else 0
            eta = (total - done) / rate_actual if rate_actual > 0 else 0
            bar = "█" * int(40 * done / total) + "░" * (40 - int(40 * done / total))
            print(
                f"\r  [{bar}] {done}/{total}  "
                f"{rate_actual:.0f}/min  "
                f"ETA {eta:.0f}min  "
                f"errors={errors}",
                end="", flush=True,
            )
            if done == total:
                print()  # newline at end

    await asyncio.gather(*[process(n) for n in to_cache])


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Pre-cache Haiku signals for VADER BUY/SELL articles")
    p.add_argument("--coins",       nargs="+", default=["BTC", "ETH", "SOL", "BNB"])
    p.add_argument("--start",       default="2024-01-01")
    p.add_argument("--end",         default="2024-04-01")
    p.add_argument("--cache-dir",   default="data/cache")
    p.add_argument("--rate",        type=int, default=45,
                   help="API requests per minute (default 45; set higher for Tier 2+)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max parallel in-flight requests (default 8)")
    p.add_argument("--save-every",  type=int, default=50,
                   help="Save cache to disk every N calls (default 50)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Count uncached articles and estimate time, no API calls")
    args = p.parse_args()

    coins = [c.upper() for c in args.coins]
    cache_dir = Path(args.cache_dir)

    # Load news
    print(f"Loading news cache for {coins} ({args.start} → {args.end})...")
    news_items = _load_news(cache_dir, coins, args.start, args.end)
    print(f"  → {len(news_items)} articles")

    # Find VADER BUY/SELL signals
    print("Running combo strategy to find VADER BUY/SELL signals...")
    vader_signals = _find_vader_signals(news_items)
    print(f"  → {len(vader_signals)} VADER BUY/SELL signals")

    # Load existing Haiku cache
    cache_file = cache_dir / _CACHE_FILENAME
    cache: dict = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
    print(f"  → {len(cache)} articles already cached")

    # Filter to uncached only
    to_cache = [n for n in vader_signals if _article_key(n) not in cache]
    print(f"  → {len(to_cache)} articles need caching")

    if not to_cache:
        print("\nAll VADER signals are already cached!")
        return

    # Estimate time
    est_min = len(to_cache) / args.rate
    print(f"\n  Rate : {args.rate} req/min  (concurrency={args.concurrency})")
    print(f"  ETA  : ~{est_min:.0f} min  ({est_min*60:.0f} sec)")

    if args.dry_run:
        print("\n[dry-run] No API calls made.")
        return

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set.")

    print(f"\nStarting... (Ctrl+C to stop — progress is saved every {args.save_every} calls)\n")
    try:
        asyncio.run(run_precache(
            to_cache=to_cache,
            cache=cache,
            cache_file=cache_file,
            api_key=api_key,
            rate_per_min=args.rate,
            concurrency=args.concurrency,
            save_every=args.save_every,
        ))
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved to cache.")
        return

    # Summary
    signals = [cache.get(_article_key(n), {}).get("signal", "?") for n in vader_signals]
    from collections import Counter
    counts = Counter(signals)
    print(f"\nCache summary for VADER signals:")
    for sig, cnt in sorted(counts.items()):
        print(f"  {sig}: {cnt}")
    print(f"Total cached: {len(cache)}")


if __name__ == "__main__":
    main()
