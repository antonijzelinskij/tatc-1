"""
Full-year backtest: 150+ real 2024 crypto events + realistic synthetic prices.
No internet needed. Covers BTC, ETH, SOL, DOGE, PEPE — Jan–Sep 2024.

Usage:
  python local_backtest.py
  python local_backtest.py --coins BTC --hold 3 --position-size 0.2
  python local_backtest.py --allow-short --min-confidence 0.5
"""
import argparse
import random
from datetime import datetime, timedelta, timezone

from tatc.backtester.engine import BacktestConfig, BacktestEngine
from tatc.core.models import Candle, NewsItem
from tatc.strategies.sentiment_vader import VaderStrategy
from tatc.strategies.combo import ComboStrategy
from data.news_2024 import NEWS_2024

random.seed(99)

# ──────────────────────────────────────────────
# Realistic 2024 price anchors (approximate)
# ──────────────────────────────────────────────
BTC_ANCHORS = [
    ("2024-01-01", 42_100), ("2024-01-11", 46_500), ("2024-01-17", 41_200),
    ("2024-02-01", 43_200), ("2024-02-12", 50_200), ("2024-02-26", 57_200),
    ("2024-03-05", 69_200), ("2024-03-13", 64_500), ("2024-03-22", 66_800),
    ("2024-04-01", 71_300), ("2024-04-08", 73_800), ("2024-04-13", 62_000),
    ("2024-04-20", 64_100), ("2024-04-22", 59_500), ("2024-05-01", 60_200),
    ("2024-05-08", 62_800), ("2024-05-22", 66_900), ("2024-05-27", 70_200),
    ("2024-06-10", 69_800), ("2024-06-20", 64_200), ("2024-06-25", 60_800),
    ("2024-07-01", 62_500), ("2024-07-05", 57_500), ("2024-07-13", 63_800),
    ("2024-07-16", 65_400), ("2024-07-18", 67_800),
    ("2024-08-01", 64_200), ("2024-08-05", 49_300), ("2024-08-07", 56_000),
    ("2024-08-15", 58_500), ("2024-08-23", 62_000),
    ("2024-09-01", 57_800), ("2024-09-06", 54_200), ("2024-09-18", 62_400),
    ("2024-09-30", 63_200),
]

# Other coins relative to BTC or independent
COIN_PROFILES = {
    "ETHUSDT":  {"ratio": 0.048,   "volatility": 1.3},
    "SOLUSDT":  {"base_jan": 100,  "volatility": 2.2, "end_sep": 155},
    "DOGEUSDT": {"base_jan": 0.088,"volatility": 2.8, "end_sep": 0.12},
    "PEPEUSDT": {"base_jan": 0.0000015, "volatility": 4.0, "end_sep": 0.000009},
}


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def interpolate_btc(anchors: list) -> list[tuple[datetime, float]]:
    """Generate hourly BTC prices from anchors with noise."""
    result = []
    parsed = [(parse_date(d), p) for d, p in anchors]
    for i in range(len(parsed) - 1):
        t0, p0 = parsed[i]
        t1, p1 = parsed[i + 1]
        hours = max(1, int((t1 - t0).total_seconds() / 3600))
        for h in range(hours):
            frac = h / hours
            price = p0 + (p1 - p0) * frac
            noise = random.gauss(0, price * 0.004)
            result.append((t0 + timedelta(hours=h), price + noise))
    return result


def make_btc_candles(prices: list) -> list[Candle]:
    candles = []
    for ts, close in prices:
        spread = close * random.uniform(0.002, 0.006)
        open_ = close + random.gauss(0, spread * 0.5)
        high = max(open_, close) + abs(random.gauss(0, spread * 0.8))
        low = min(open_, close) - abs(random.gauss(0, spread * 0.8))
        vol = random.uniform(500, 5000)
        candles.append(Candle("BTCUSDT", ts,
                               max(open_, 1), max(high, 1), max(low, 1), max(close, 1), vol))
    return candles


def make_derived_candles(symbol: str, btc_prices: list, profile: dict) -> list[Candle]:
    """Generate candles for alt coins derived from BTC with extra volatility."""
    candles = []
    if "ratio" in profile:
        # ETH: tracks BTC with ratio + extra noise
        ratio = profile["ratio"]
        vol_mult = profile["volatility"]
        for ts, btc_price in btc_prices:
            base = btc_price * ratio
            noise = random.gauss(0, base * 0.005 * vol_mult)
            close = max(base + noise, 0.001)
            spread = close * random.uniform(0.003, 0.008)
            open_ = close + random.gauss(0, spread * 0.5)
            high = max(open_, close) + abs(random.gauss(0, spread))
            low = min(open_, close) - abs(random.gauss(0, spread))
            candles.append(Candle(symbol, ts,
                                   max(open_, 0.0001), max(high, 0.0001),
                                   max(low, 0.0001), close, random.uniform(100, 10000)))
    else:
        # Independent coin: interpolate from start to end with BTC correlation
        base_jan = profile["base_jan"]
        end_sep = profile["end_sep"]
        vol_mult = profile["volatility"]
        n = len(btc_prices)
        for i, (ts, btc_price) in enumerate(btc_prices):
            frac = i / max(n - 1, 1)
            trend = base_jan + (end_sep - base_jan) * frac
            # Add BTC correlation (30%)
            btc_change = (btc_price - BTC_ANCHORS[0][1]) / BTC_ANCHORS[0][1]
            btc_contrib = trend * btc_change * 0.3
            noise = random.gauss(0, trend * 0.006 * vol_mult)
            close = max(trend + btc_contrib + noise, base_jan * 0.1)
            spread = close * random.uniform(0.003, 0.01)
            open_ = close + random.gauss(0, spread * 0.5)
            high = max(open_, close) + abs(random.gauss(0, spread))
            low = min(open_, close) - abs(random.gauss(0, spread))
            candles.append(Candle(symbol, ts,
                                   max(open_, 1e-10), max(high, 1e-10),
                                   max(low, 1e-10), close, random.uniform(1e6, 1e9)))
    return candles


def build_news_items(raw: list, coins_filter: list[str]) -> list[NewsItem]:
    items = []
    for i, (date_str, coins_str, title) in enumerate(raw):
        coins = coins_str.split()
        # Filter: keep if any of the requested coins is mentioned
        if coins_filter and not any(c in coins_filter for c in coins):
            continue
        dt = parse_date(date_str).replace(hour=9 + (i % 8), minute=(i * 11) % 60)
        items.append(NewsItem(
            id=str(i), timestamp=dt, title=title, body="",
            source="real_events_2024", url="", coins=coins,
        ))
    return items


def run(args):
    coins = [c.upper() for c in args.coins]
    symbols = [f"{c}USDT" for c in coins]

    print("=" * 60)
    print("  TATC Backtest — Real 2024 Events, Synthetic Prices")
    print(f"  Strategy: {args.strategy.upper()}")
    print(f"  Coins   : {', '.join(coins)}")
    print(f"  Period  : Jan 2024 → Sep 2024")
    print(f"  Capital : ${args.capital:,.0f} | Position: {args.position_size*100:.0f}%")
    print(f"  Hold    : {args.hold} candles (1h) = ~{args.hold}h")
    print(f"  Min conf: {args.min_confidence}")
    print(f"  Short   : {'YES' if args.allow_short else 'NO'}")
    print("=" * 60)

    # Build prices
    btc_prices = interpolate_btc(BTC_ANCHORS)
    print(f"\n[1/3] Building price data...")
    candles: dict = {}
    for sym in symbols:
        if sym == "BTCUSDT":
            candles[sym] = make_btc_candles(btc_prices)
        elif sym in COIN_PROFILES:
            candles[sym] = make_derived_candles(sym, btc_prices, COIN_PROFILES[sym])
        else:
            print(f"  WARNING: no price profile for {sym}, skipping")
            continue
        c = candles[sym]
        change = (c[-1].close - c[0].close) / c[0].close * 100
        print(f"  {sym}: {len(c)} candles | "
              f"${c[0].close:.4g} → ${c[-1].close:.4g} ({change:+.1f}%)")

    # Build news
    print(f"\n[2/3] Loading news events...")
    news_items = build_news_items(NEWS_2024, coins)
    print(f"  {len(news_items)} events for {coins}")

    # Sentiment preview
    if args.strategy == "combo":
        strategy = ComboStrategy(
            buy_threshold=args.buy_threshold if args.buy_threshold != 0.3 else 0.45,
            sell_threshold=args.sell_threshold if args.sell_threshold != -0.3 else -0.45,
            cooldown_minutes=args.cooldown,
            dedup_window_minutes=args.dedup_window,
        )
    else:
        strategy = VaderStrategy(
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    signals = strategy.analyze_many(news_items)
    n_buy  = sum(1 for s in signals if s.signal.value == "BUY")
    n_sell = sum(1 for s in signals if s.signal.value == "SELL")
    n_hold = sum(1 for s in signals if s.signal.value == "HOLD")
    print(f"  Sentiment: {n_buy} BUY / {n_sell} SELL / {n_hold} HOLD")

    # Run backtest
    config = BacktestConfig(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        hold_candles=args.hold,
        min_confidence=args.min_confidence,
        allow_short=args.allow_short,
        slippage_pct=0.001,
    )
    engine = BacktestEngine(strategy, config)

    print(f"\n[3/3] Running backtest engine...")
    metrics = engine.run(candles, news_items)
    print(metrics)

    # Top signals
    print("\nTop BUY signals:")
    buy_sigs = sorted([s for s in signals if s.signal.value == "BUY"],
                      key=lambda x: -x.confidence)
    for s in buy_sigs[:8]:
        print(f"  +{s.confidence:.2f} {s.news.timestamp.strftime('%m-%d')} "
              f"[{','.join(s.news.coins)}] {s.news.title[:58]}")

    print("\nTop SELL signals:")
    sell_sigs = sorted([s for s in signals if s.signal.value == "SELL"],
                       key=lambda x: -x.confidence)
    for s in sell_sigs[:8]:
        print(f"  -{s.confidence:.2f} {s.news.timestamp.strftime('%m-%d')} "
              f"[{','.join(s.news.coins)}] {s.news.title[:58]}")


def main():
    p = argparse.ArgumentParser(description="TATC Local Backtest — 2024 Real Events")
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "DOGE"])
    p.add_argument("--capital", type=float, default=10_000)
    p.add_argument("--position-size", type=float, default=0.1)
    p.add_argument("--hold", type=int, default=5,
                   help="Candles to hold (1 candle = 1 hour)")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--buy-threshold", type=float, default=0.3)
    p.add_argument("--sell-threshold", type=float, default=-0.3)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--strategy", default="vader",
                   choices=["vader", "combo"],
                   help="Sentiment strategy: vader (default) or combo (filtered)")
    p.add_argument("--cooldown", type=int, default=30,
                   help="[combo] Per-coin cooldown minutes (default: 30)")
    p.add_argument("--dedup-window", type=int, default=120,
                   help="[combo] Duplicate headline suppression window in minutes (default: 120)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
