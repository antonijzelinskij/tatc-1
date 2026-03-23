#!/usr/bin/env python3
"""
pump_analysis.py — Find significant price moves and associated news.

Usage:
  python pump_analysis.py                          # BTC, 3%, 1h candles
  python pump_analysis.py --coins BTC ETH SOL      # multiple coins
  python pump_analysis.py --threshold 5 --window 1h 4h 24h
  python pump_analysis.py --dump                   # print all headlines per event
  python pump_analysis.py --start 2024-01-01 --end 2024-07-01
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

CACHE_DIR = "data/cache"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (reuse cached files produced by real_backtest.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_candles(coin: str, start: str, end: str) -> list[dict]:
    """Load hourly candles from cache. Returns list sorted by timestamp."""
    sym = f"{coin}USDT"
    fname = f"candles_{sym}_60_{start}_{end}.json"
    path = os.path.join(CACHE_DIR, fname)
    if not os.path.exists(path):
        # Try to find any file that covers the range
        for f in os.listdir(CACHE_DIR):
            if f.startswith(f"candles_{sym}_60_") and f.endswith(".json"):
                # crude check: filename dates contain our range
                parts = f.replace(".json", "").split("_")
                if len(parts) >= 5:
                    fs, fe = parts[-2], parts[-1]
                    if fs <= start and fe >= end:
                        path = os.path.join(CACHE_DIR, f)
                        break
        else:
            sys.exit(f"[ERROR] No candle cache found for {sym} {start}→{end}\n"
                     f"Run: python real_backtest.py --coins {coin} --start {start} --end {end}")
    with open(path) as f:
        candles = json.load(f)
    # Filter to requested range
    candles = [c for c in candles if start <= c["timestamp"][:10] < end]
    candles.sort(key=lambda c: c["timestamp"])
    return candles


def load_news(coins: list[str], start: str, end: str) -> list[dict]:
    """Load news from cache. Tries several filename patterns."""
    key = "_".join(sorted(coins))
    fname = f"news_{key}_{start}_{end}.json"
    path = os.path.join(CACHE_DIR, fname)
    if not os.path.exists(path):
        # Try broader cache files
        for f in sorted(os.listdir(CACHE_DIR)):
            if not f.startswith("news_") or not f.endswith(".json"):
                continue
            parts = f.replace(".json", "").split("_")
            # last two parts are dates
            try:
                fs, fe = parts[-2], parts[-1]
            except IndexError:
                continue
            if fs <= start and fe >= end:
                path = os.path.join(CACHE_DIR, f)
                break
        else:
            sys.exit(f"[ERROR] No news cache found for {coins} {start}→{end}\n"
                     f"Run: python real_backtest.py --coins {' '.join(coins)} --start {start} --end {end}")
    with open(path) as f:
        news = json.load(f)
    # Filter to range and relevant coins
    coin_set = set(coins)
    result = []
    for n in news:
        ts = n["timestamp"][:10]
        if ts < start or ts >= end:
            continue
        n_coins = set(n.get("coins", []))
        if not n_coins or n_coins & coin_set:
            result.append(n)
    result.sort(key=lambda n: n["timestamp"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Pump / dump detection
# ──────────────────────────────────────────────────────────────────────────────

def parse_window(w: str) -> int:
    """Convert '4h' or '240m' or '4' → number of 1h candles."""
    w = w.strip().lower()
    if w.endswith("h"):
        return int(w[:-1])
    if w.endswith("m"):
        return max(1, int(w[:-1]) // 60)
    return int(w)


def find_moves(candles: list[dict], threshold_pct: float, window_candles: int,
               direction: str = "both") -> list[dict]:
    """
    Find all windows of `window_candles` consecutive candles where
    cumulative return > threshold_pct (pump) or < -threshold_pct (dump).

    Returns list of events:
      { timestamp, open, close, pct_change, direction, candles_list }
    """
    events = []
    n = len(candles)
    threshold = threshold_pct / 100.0

    for i in range(n - window_candles + 1):
        window = candles[i : i + window_candles]
        open_price  = window[0]["open"]
        close_price = window[-1]["close"]
        high_price  = max(c["high"] for c in window)
        low_price   = min(c["low"] for c in window)
        pct = (close_price - open_price) / open_price

        is_pump = pct >= threshold
        is_dump = pct <= -threshold

        if (direction in ("both", "pump") and is_pump) or \
           (direction in ("both", "dump") and is_dump):
            events.append({
                "ts_start":   window[0]["timestamp"],
                "ts_end":     window[-1]["timestamp"],
                "open":       open_price,
                "close":      close_price,
                "high":       high_price,
                "low":        low_price,
                "pct_change": pct * 100,
                "direction":  "pump" if pct > 0 else "dump",
                "window_h":   window_candles,
            })

    # Deduplicate overlapping windows — keep the largest move per hour
    events.sort(key=lambda e: (e["ts_start"], -abs(e["pct_change"])))
    deduped = []
    last_end = None
    for e in events:
        if last_end is None or e["ts_start"] > last_end:
            deduped.append(e)
            last_end = e["ts_end"]
    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# News association
# ──────────────────────────────────────────────────────────────────────────────

def associate_news(events: list[dict], news: list[dict],
                   before_h: int = 4, after_h: int = 1) -> list[dict]:
    """
    For each event, collect news articles published in [start - before_h, end + after_h].
    Returns events enriched with 'news' list.
    """
    enriched = []
    for ev in events:
        ts_start = datetime.fromisoformat(ev["ts_start"])
        ts_end   = datetime.fromisoformat(ev["ts_end"])
        window_open  = ts_start - timedelta(hours=before_h)
        window_close = ts_end   + timedelta(hours=after_h)

        ev_news = []
        for n in news:
            nts = datetime.fromisoformat(n["timestamp"])
            if window_open <= nts <= window_close:
                ev_news.append(n)

        enriched.append({**ev, "news": ev_news,
                         "news_before": [n for n in ev_news
                                         if datetime.fromisoformat(n["timestamp"]) < ts_start],
                         "news_during": [n for n in ev_news
                                         if ts_start <= datetime.fromisoformat(n["timestamp"]) <= ts_end]})
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_stats(coin: str, events: list[dict], window_h: int, threshold: float):
    total      = len(events)
    pumps      = [e for e in events if e["direction"] == "pump"]
    dumps      = [e for e in events if e["direction"] == "dump"]
    with_news  = [e for e in events if e["news"]]
    before_only = [e for e in events if e["news_before"]]

    news_counts = [len(e["news"]) for e in events]
    before_counts = [len(e["news_before"]) for e in events]

    avg_news   = sum(news_counts) / total if total else 0
    avg_before = sum(before_counts) / total if total else 0
    pct_moves  = [e["pct_change"] for e in events]
    avg_pct    = sum(pct_moves) / total if total else 0

    print(f"\n{'═'*64}")
    print(f"  {coin}  |  window={window_h}h  threshold={threshold:+.0f}%")
    print(f"{'═'*64}")
    print(f"  Total events   : {total:4d}  (pumps={len(pumps)}, dumps={len(dumps)})")
    print(f"  Avg move       : {avg_pct:+.2f}%")
    print(f"  With any news  : {len(with_news):4d}  ({100*len(with_news)/total:.0f}%)")
    print(f"  With pre-news  : {len(before_only):4d}  ({100*len(before_only)/total:.0f}%)")
    print(f"  Avg news/event : {avg_news:.1f}  (pre-event: {avg_before:.1f})")

    # Distribution
    buckets = {0: 0, "1-3": 0, "4-10": 0, "11+": 0}
    for c in news_counts:
        if c == 0: buckets[0] += 1
        elif c <= 3: buckets["1-3"] += 1
        elif c <= 10: buckets["4-10"] += 1
        else: buckets["11+"] += 1
    print(f"\n  News count distribution (total window):")
    for k, v in buckets.items():
        bar = "█" * (v * 30 // max(total, 1))
        print(f"    {str(k):>4} articles : {v:3d}  {bar}")

    # Monthly breakdown
    monthly = defaultdict(list)
    for e in events:
        mo = e["ts_start"][:7]
        monthly[mo].append(e["pct_change"])
    print(f"\n  Monthly breakdown:")
    for mo, pcts in sorted(monthly.items()):
        avg = sum(pcts) / len(pcts)
        print(f"    {mo}  {len(pcts):3d} events  avg {avg:+.1f}%")


def print_event_details(events: list[dict], coin: str, max_events: int = 999):
    """Print each event with its headlines."""
    print(f"\n{'─'*64}")
    print(f"  EVENT DETAIL: {coin}  ({len(events)} events)")
    print(f"{'─'*64}")
    for i, ev in enumerate(events[:max_events]):
        ts = ev["ts_start"][:16]
        pct = ev["pct_change"]
        n_pre = len(ev["news_before"])
        n_dur = len(ev["news_during"])
        arrow = "▲" if ev["direction"] == "pump" else "▼"
        print(f"\n  [{i+1:3d}]  {arrow} {pct:+.2f}%  {ts}  "
              f"  pre={n_pre} during={n_dur}  "
              f"${ev['open']:,.0f}→${ev['close']:,.0f}")
        if ev["news_before"]:
            print("        PRE-EVENT news:")
            for n in ev["news_before"][:8]:
                nts = n["timestamp"][5:16]
                coins_tag = ",".join(n.get("coins", []))
                print(f"          {nts} [{coins_tag:12s}] {n['title'][:72]}")
        if ev["news_during"]:
            print("        DURING news:")
            for n in ev["news_during"][:4]:
                nts = n["timestamp"][5:16]
                coins_tag = ",".join(n.get("coins", []))
                print(f"          {nts} [{coins_tag:12s}] {n['title'][:72]}")


def print_top_headlines(events: list[dict], coin: str, direction: str = "pump"):
    """Show most common / representative headlines before big moves."""
    filtered = [e for e in events if e["direction"] == direction]
    # Collect all pre-event headlines
    all_before = []
    for e in filtered:
        for n in e["news_before"]:
            all_before.append((abs(e["pct_change"]), n["title"], n["timestamp"][:16]))
    # Sort by move size (biggest moves first)
    all_before.sort(reverse=True)
    dir_label = "PUMPS" if direction == "pump" else "DUMPS"
    print(f"\n{'─'*64}")
    print(f"  TOP PRE-EVENT HEADLINES for {coin} {dir_label} (sorted by move size)")
    print(f"{'─'*64}")
    seen = set()
    count = 0
    for pct, title, ts in all_before:
        if title in seen:
            continue
        seen.add(title)
        print(f"  {pct:+.1f}%  {ts}  {title[:70]}")
        count += 1
        if count >= 30:
            break


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Analyze price pumps/dumps and associated news.")
    p.add_argument("--coins",     nargs="+", default=["BTC"],
                   help="Coins to analyze (default: BTC)")
    p.add_argument("--start",     default="2024-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default="2024-07-01", help="End date YYYY-MM-DD")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="Minimum price move %% to flag as event (default: 3.0)")
    p.add_argument("--windows",   nargs="+", default=["1h", "4h"],
                   help="Candle windows to check (default: 1h 4h)")
    p.add_argument("--before",    type=int, default=4,
                   help="Hours of news to look at BEFORE event (default: 4)")
    p.add_argument("--after",     type=int, default=1,
                   help="Hours of news to look at DURING/AFTER event (default: 1)")
    p.add_argument("--direction", choices=["both", "pump", "dump"], default="both",
                   help="Which direction to analyze (default: both)")
    p.add_argument("--dump",      action="store_true",
                   help="Print all events with their headlines")
    p.add_argument("--top",       action="store_true",
                   help="Print top 30 pre-event headlines per coin/direction")
    args = p.parse_args()

    coins = [c.upper() for c in args.coins]
    windows = [parse_window(w) for w in args.windows]

    print(f"\n{'═'*64}")
    print(f"  PUMP/DUMP ANALYSIS")
    print(f"  Period    : {args.start} → {args.end}")
    print(f"  Coins     : {', '.join(coins)}")
    print(f"  Threshold : >{args.threshold:.1f}% per window")
    print(f"  Windows   : {args.windows}")
    print(f"  News window: -{args.before}h … +{args.after}h around event")
    print(f"{'═'*64}")

    # Load news once (covers all coins)
    print("\nLoading news...")
    news = load_news(coins, args.start, args.end)
    print(f"  → {len(news)} news items")

    for coin in coins:
        print(f"\nLoading candles: {coin}...")
        candles = load_candles(coin, args.start, args.end)
        print(f"  → {len(candles)} hourly candles")

        # Filter news to this coin
        coin_news = [n for n in news if coin in n.get("coins", [])]
        print(f"  → {len(coin_news)} coin-specific news")

        for w in windows:
            events = find_moves(candles, args.threshold, w, args.direction)
            events = associate_news(events, coin_news, args.before, args.after)
            print_stats(coin, events, w, args.threshold)

            if args.dump:
                print_event_details(events, coin)
            elif args.top:
                if args.direction in ("both", "pump"):
                    print_top_headlines(events, coin, "pump")
                if args.direction in ("both", "dump"):
                    print_top_headlines(events, coin, "dump")


if __name__ == "__main__":
    main()
