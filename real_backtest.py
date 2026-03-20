"""
Real backtest: real news from cryptocurrency.cv + real prices from Bybit.
No API keys required.

Usage:
  python real_backtest.py                        # BTC+ETH, Jan-Mar 2024
  python real_backtest.py --start 2024-01-01 --end 2024-06-01 --coins BTC ETH SOL
  python real_backtest.py --coins BTC --start 2024-03-01 --end 2024-04-01
"""
import argparse
import asyncio

import aiohttp

from tatc.backtester.engine import BacktestConfig, BacktestEngine
from tatc.sources.news.cryptocurrencycv import CryptoCurrencyCVSource
from tatc.sources.price.bybit import BybitPriceSource
from tatc.strategies.sentiment_vader import VaderStrategy
from tatc import config as cfg


async def run(args: argparse.Namespace) -> None:
    coins = args.coins
    symbols = [f"{c}USDT" for c in coins]

    print("=" * 58)
    print(f"  TATC Real Backtest")
    print(f"  Period : {args.start} → {args.end}")
    print(f"  Coins  : {', '.join(coins)}")
    print(f"  Capital: ${args.capital:,.0f} | Position: {args.position_size*100:.0f}%")
    print(f"  Hold   : {args.hold_candles} candles ({args.interval}m)")
    print("=" * 58)

    async with aiohttp.ClientSession() as session:
        # 1. Fetch historical news
        print(f"\n[1/3] Fetching news from cryptocurrency.cv...")
        news_source = CryptoCurrencyCVSource(session)
        news_items = await news_source.fetch_historical(
            start=args.start,
            end=args.end,
            coins=coins,
        )
        print(f"  → {len(news_items)} news items loaded")

        if not news_items:
            print("\n  No news found. Try a different date range or coin.")
            return

        # Show sample
        print("\n  Sample headlines:")
        for item in news_items[:5]:
            print(f"    {item.timestamp.strftime('%Y-%m-%d %H:%M')} "
                  f"[{','.join(item.coins) or '?'}] {item.title[:60]}")

        # 2. Fetch price candles
        print(f"\n[2/3] Fetching {args.interval}m candles from Bybit...")
        price_source = BybitPriceSource(session)
        candles = {}
        for sym in symbols:
            print(f"  → {sym}...", end=" ", flush=True)
            try:
                sym_candles = await price_source.get_candles(
                    symbol=sym,
                    interval=args.interval,
                    start=args.start,
                    end=args.end,
                )
                if sym_candles:
                    candles[sym] = sym_candles
                    first, last = sym_candles[0], sym_candles[-1]
                    change = (last.close - first.close) / first.close * 100
                    print(f"{len(sym_candles)} candles, "
                          f"${first.close:.4g}→${last.close:.4g} ({change:+.1f}%)")
                else:
                    print("no data")
            except Exception as e:
                print(f"FAILED: {e}")

        if not candles:
            print("\n  No price data. Check your symbols or date range.")
            return

    # 3. Run backtest
    strategy = VaderStrategy(
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    bt_config = BacktestConfig(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        hold_candles=args.hold_candles,
        min_confidence=args.min_confidence,
        allow_short=args.allow_short,
        slippage_pct=0.001,
    )

    print(f"\n[3/3] Running backtest...")
    engine = BacktestEngine(strategy, bt_config)
    metrics = engine.run(candles, news_items)
    print(metrics)

    # Sentiment breakdown
    print("\nAll signals generated:")
    signals = strategy.analyze_many(news_items)
    buys  = [s for s in signals if s.signal.value == "BUY"]
    sells = [s for s in signals if s.signal.value == "SELL"]
    holds = [s for s in signals if s.signal.value == "HOLD"]
    print(f"  BUY: {len(buys)}  SELL: {len(sells)}  HOLD: {len(holds)}")

    print("\nTop BUY signals:")
    for s in sorted(buys, key=lambda x: -x.confidence)[:5]:
        print(f"  +{s.confidence:.2f} [{','.join(s.news.coins)}] {s.news.title[:60]}")
    print("Top SELL signals:")
    for s in sorted(sells, key=lambda x: -x.confidence)[:5]:
        print(f"  -{s.confidence:.2f} [{','.join(s.news.coins)}] {s.news.title[:60]}")


def main():
    p = argparse.ArgumentParser(description="TATC Real Backtest")
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH"],
                   help="Coin codes e.g. BTC ETH SOL DOGE")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-04-01")
    p.add_argument("--interval", default="60", help="Candle interval minutes: 1,5,15,60,240,D")
    p.add_argument("--capital", type=float, default=10_000)
    p.add_argument("--position-size", type=float, default=0.1)
    p.add_argument("--hold-candles", type=int, default=5)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--buy-threshold", type=float, default=0.3)
    p.add_argument("--sell-threshold", type=float, default=-0.3)
    p.add_argument("--allow-short", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
