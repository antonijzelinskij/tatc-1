"""
TATC — Trading Analysis Tool for Crypto
CLI entrypoint.

Usage:
  python main.py backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01
  python main.py backtest --start 2024-01-01 --end 2024-03-01   # all symbols
"""
import argparse
import asyncio
import sys

import aiohttp

from tatc import config
from tatc.backtester.engine import BacktestConfig, BacktestEngine
from tatc.sources.news.cryptopanic import CryptoPanicSource
from tatc.sources.price.bybit import BybitPriceSource
from tatc.strategies.sentiment_vader import VaderStrategy


async def run_backtest(args: argparse.Namespace) -> None:
    symbols = [args.symbol] if args.symbol else config.SYMBOLS[:10]  # default top 10
    coins = [s.replace("USDT", "") for s in symbols]

    print(f"[backtest] period: {args.start} → {args.end}")
    print(f"[backtest] symbols: {symbols}")
    print(f"[backtest] fetching news from CryptoPanic...")

    if not config.CRYPTOPANIC_API_KEY:
        print("ERROR: CRYPTOPANIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        # 1. Fetch historical news
        news_source = CryptoPanicSource(config.CRYPTOPANIC_API_KEY, session)
        news_items = await news_source.fetch_historical(
            start=args.start,
            end=args.end,
            coins=coins if not args.all_coins else None,
        )
        print(f"[backtest] loaded {len(news_items)} news items")

        # 2. Fetch historical candles for all symbols
        price_source = BybitPriceSource(session)
        candles: dict = {}
        for sym in symbols:
            print(f"[backtest] fetching candles for {sym}...")
            try:
                sym_candles = await price_source.get_candles(
                    symbol=sym,
                    interval=args.interval,
                    start=args.start,
                    end=args.end,
                )
                if sym_candles:
                    candles[sym] = sym_candles
                    print(f"  → {len(sym_candles)} candles")
            except Exception as e:
                print(f"  → SKIP ({e})")

    # 3. Run backtest
    strategy = VaderStrategy(
        buy_threshold=config.VADER_BUY_THRESHOLD,
        sell_threshold=config.VADER_SELL_THRESHOLD,
    )
    bt_config = BacktestConfig(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        hold_candles=args.hold_candles,
        min_confidence=args.min_confidence,
        allow_short=args.allow_short,
    )

    print(f"\n[backtest] running engine...")
    engine = BacktestEngine(strategy, bt_config)
    metrics = engine.run(candles, news_items)
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="TATC — Crypto News Trading Bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- backtest command ---
    bt = subparsers.add_parser("backtest", help="Run strategy on historical data")
    bt.add_argument("--symbol", default=None, help="Single symbol e.g. BTCUSDT (default: top 10)")
    bt.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    bt.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    bt.add_argument("--interval", default=config.DEFAULT_INTERVAL,
                    help="Candle interval: 1,5,15,60,240,D (default: 60)")
    bt.add_argument("--capital", type=float, default=config.DEFAULT_INITIAL_CAPITAL,
                    help="Initial capital in USD")
    bt.add_argument("--position-size", type=float, default=config.DEFAULT_POSITION_SIZE_PCT,
                    help="Fraction of capital per trade (default: 0.1)")
    bt.add_argument("--hold-candles", type=int, default=config.DEFAULT_HOLD_CANDLES,
                    help="Candles to hold before force-exit")
    bt.add_argument("--min-confidence", type=float, default=config.DEFAULT_MIN_CONFIDENCE,
                    help="Min signal confidence 0..1")
    bt.add_argument("--allow-short", action="store_true", help="Allow short selling")
    bt.add_argument("--all-coins", action="store_true", help="Fetch news for all coins")

    args = parser.parse_args()

    if args.command == "backtest":
        asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
