"""
Real backtest: real news + real prices from HuggingFace (no API keys needed).
Falls back to cryptocurrency.cv + Bybit if HuggingFace is unavailable.

News:   maryamfakhari/crypto-news-coindesk-2020-2025 (229k articles, 2019–2025)
Prices: adamzzzz/binance-klines-20240721 (Binance 1h OHLCV, 2020–2024-07)

Caching: data is saved to --cache-dir after first fetch.
         On subsequent runs, loaded from cache automatically.
         Use --no-cache to force re-download.

Usage:
  python real_backtest.py                                   # BTC+ETH, Jan-Mar 2024
  python real_backtest.py --coins BTC ETH SOL --start 2024-01-01 --end 2024-06-01
  python real_backtest.py --no-cache                        # ignore cache, re-download
  python real_backtest.py --allow-short --hold-candles 12
  python real_backtest.py --stop-loss 0.02 --take-profit 0.05  # 2% SL, 5% TP
  python real_backtest.py --fee 0.001                       # 0.1% fee per leg (default)
  python real_backtest.py --signal-window 4                 # aggregate news over 4h
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tatc.backtester.engine import BacktestConfig, BacktestEngine
from tatc.core.models import Candle, NewsItem
from tatc.strategies.sentiment_vader import VaderStrategy
from tatc.strategies.combo import ComboStrategy


# ──────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────

def _news_cache_path(cache_dir: Path, coins: list[str], start: str, end: str) -> Path:
    coin_str = "_".join(sorted(coins)) if coins else "all"
    return cache_dir / f"news_{coin_str}_{start}_{end}.json"


def _candles_cache_path(cache_dir: Path, symbol: str, interval: str, start: str, end: str) -> Path:
    return cache_dir / f"candles_{symbol}_{interval}_{start}_{end}.json"


def _save_news(path: Path, items: list[NewsItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "id": it.id, "timestamp": it.timestamp.isoformat(),
            "title": it.title, "body": it.body,
            "source": it.source, "url": it.url, "coins": it.coins,
        }
        for it in items
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_news(path: Path) -> list[NewsItem]:
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


def _save_candles(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "symbol": c.symbol, "timestamp": c.timestamp.isoformat(),
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        }
        for c in candles
    ]
    path.write_text(json.dumps(data))


def _load_candles(path: Path) -> list[Candle]:
    data = json.loads(path.read_text())
    candles = []
    for d in data:
        ts = datetime.fromisoformat(d["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        candles.append(Candle(
            symbol=d["symbol"], timestamp=ts,
            open=d["open"], high=d["high"], low=d["low"],
            close=d["close"], volume=d["volume"],
        ))
    return candles


# ──────────────────────────────────────────────────────────────
# Fetch helpers (HuggingFace primary, cryptocurrency.cv fallback)
# ──────────────────────────────────────────────────────────────

def fetch_news(coins: list[str], start: str, end: str) -> list[NewsItem]:
    """Fetch news from HuggingFace (primary) or cryptocurrency.cv (fallback)."""
    try:
        from tatc.sources.news.huggingface import fetch_historical
        items = fetch_historical(start=start, end=end, coins=coins)
        return items
    except Exception as e:
        print(f"  [hf-news] failed ({e}), falling back to cryptocurrency.cv")

    import asyncio
    import aiohttp
    from tatc.sources.news.cryptocurrencycv import CryptoCurrencyCVSource

    async def _fetch():
        async with aiohttp.ClientSession() as session:
            src = CryptoCurrencyCVSource(session)
            return await src.fetch_historical(start=start, end=end, coins=coins)

    return asyncio.run(_fetch())


def fetch_candles(symbol: str, interval: str, start: str, end: str) -> list[Candle]:
    """Fetch candles from HuggingFace (primary) or Bybit (fallback)."""
    try:
        from tatc.sources.price.huggingface import get_candles
        return get_candles(symbol=symbol, interval=interval, start=start, end=end)
    except Exception as e:
        print(f"  [hf-price] failed ({e}), falling back to Bybit")

    import asyncio
    import aiohttp
    from tatc.sources.price.bybit import BybitPriceSource

    async def _fetch():
        async with aiohttp.ClientSession() as session:
            src = BybitPriceSource(session)
            return await src.get_candles(symbol=symbol, interval=interval, start=start, end=end)

    return asyncio.run(_fetch())


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    coins = [c.upper() for c in args.coins]
    symbols = [f"{c}USDT" for c in coins]
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    print("=" * 58)
    print(f"  TATC Real Backtest")
    print(f"  Strategy: {args.strategy.upper()}")
    print(f"  Period : {args.start} → {args.end}")
    print(f"  Coins  : {', '.join(coins)}")
    print(f"  Capital: ${args.capital:,.0f} | Position: {args.position_size*100:.0f}%")
    print(f"  Hold   : {args.hold_candles} candles ({args.interval}m)")
    if args.leverage > 1.0:
        liq_pct = (1.0 / args.leverage - 0.005) * 100
        print(f"  Leverage: {args.leverage:.0f}x  (liq at ~{liq_pct:.1f}% adverse move)")
    if args.stop_loss:
        print(f"  SL/TP  : SL={args.stop_loss*100:.1f}%  TP={args.take_profit*100:.1f}%")
    print(f"  Fee    : {args.fee*100:.2f}% per leg on notional")
    if args.signal_window:
        print(f"  Window : {args.signal_window}h signal aggregation")
    print(f"  Cache  : {'OFF (--no-cache)' if args.no_cache else cache_dir}")
    print("=" * 58)

    # ── 1. News ──────────────────────────────────────────────
    print(f"\n[1/3] Fetching news...")
    news_path = _news_cache_path(cache_dir, coins, args.start, args.end)

    if use_cache and news_path.exists():
        news_items = _load_news(news_path)
        print(f"  → {len(news_items)} news items loaded from cache ({news_path.name})")
    else:
        news_items = fetch_news(coins=coins, start=args.start, end=args.end)
        print(f"  → {len(news_items)} news items fetched")
        if news_items and use_cache:
            _save_news(news_path, news_items)
            print(f"  → saved to cache: {news_path.name}")

    if not news_items:
        print("\n  No news found. Try a different date range or coin.")
        return

    print("\n  Sample headlines:")
    for item in news_items[:5]:
        print(f"    {item.timestamp.strftime('%Y-%m-%d %H:%M')} "
              f"[{','.join(item.coins) or '?'}] {item.title[:60]}")

    # ── 2. Prices ─────────────────────────────────────────────
    print(f"\n[2/3] Fetching {args.interval}m candles...")
    candles: dict[str, list[Candle]] = {}

    for sym in symbols:
        candles_path = _candles_cache_path(cache_dir, sym, args.interval, args.start, args.end)

        if use_cache and candles_path.exists():
            sym_candles = _load_candles(candles_path)
            first, last = sym_candles[0], sym_candles[-1]
            change = (last.close - first.close) / first.close * 100
            print(f"  → {sym}: {len(sym_candles)} candles from cache "
                  f"(${first.close:.4g}→${last.close:.4g}, {change:+.1f}%)")
        else:
            print(f"  → {sym}...", end=" ", flush=True)
            try:
                sym_candles = fetch_candles(
                    symbol=sym, interval=args.interval,
                    start=args.start, end=args.end,
                )
                if sym_candles:
                    first, last = sym_candles[0], sym_candles[-1]
                    change = (last.close - first.close) / first.close * 100
                    print(f"{len(sym_candles)} candles, "
                          f"${first.close:.4g}→${last.close:.4g} ({change:+.1f}%)")
                    if use_cache:
                        _save_candles(candles_path, sym_candles)
                        print(f"     saved to cache: {candles_path.name}")
                else:
                    print("no data")
                    continue
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if sym_candles:
            candles[sym] = sym_candles

    if not candles:
        print("\n  No price data. Check your symbols or date range.")
        return

    # ── 3. Backtest ───────────────────────────────────────────
    if args.strategy == "finbert":
        from tatc.strategies.sentiment_finbert import FinBertStrategy
        strategy = FinBertStrategy(
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    elif args.strategy == "combo":
        strategy = ComboStrategy(
            buy_threshold=args.buy_threshold if args.buy_threshold != 0.3 else 0.45,
            sell_threshold=args.sell_threshold if args.sell_threshold != -0.3 else -0.45,
            cooldown_minutes=args.cooldown,
            dedup_window_minutes=args.dedup_window,
            use_finbert=args.combo_finbert,
        )
    else:
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
        fee_pct=args.fee,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        signal_window_hours=args.signal_window,
        leverage=args.leverage,
        news_exit=args.news_exit,
        news_exit_min_confidence=args.news_exit_confidence,
        news_exit_only_loss=args.news_exit_only_loss,
    )

    print(f"\n[3/3] Running backtest...")
    engine = BacktestEngine(strategy, bt_config)
    metrics = engine.run(candles, news_items)
    print(metrics)

    # Sentiment breakdown (reset state so preview is independent of engine run)
    if hasattr(strategy, "reset"):
        strategy.reset()
    signals = strategy.analyze_many(news_items)
    buys  = [s for s in signals if s.signal.value == "BUY"]
    sells = [s for s in signals if s.signal.value == "SELL"]
    holds = [s for s in signals if s.signal.value == "HOLD"]
    print(f"\nSignals: BUY={len(buys)}  SELL={len(sells)}  HOLD={len(holds)}")

    print("\nTop BUY signals:")
    for s in sorted(buys, key=lambda x: -x.confidence)[:5]:
        print(f"  +{s.confidence:.2f} {s.news.timestamp.strftime('%Y-%m-%d')} "
              f"[{','.join(s.news.coins)}] {s.news.title[:60]}")
    print("Top SELL signals:")
    for s in sorted(sells, key=lambda x: -x.confidence)[:5]:
        print(f"  -{s.confidence:.2f} {s.news.timestamp.strftime('%Y-%m-%d')} "
              f"[{','.join(s.news.coins)}] {s.news.title[:60]}")

    # ── Trade analysis ────────────────────────────────────────────
    if metrics.trades:
        wins   = [t for t in metrics.trades if t.pnl > 0]
        losses = [t for t in metrics.trades if t.pnl <= 0]

        print(f"\n{'─'*72}")
        print(f"  LOSING TRADES  ({len(losses)} of {len(metrics.trades)})")
        print(f"{'─'*72}")
        for t in sorted(losses, key=lambda x: x.pnl)[:30]:
            meta = t.signal_meta
            sn   = meta.get("sell_news_mult", 1.0)
            nm   = meta.get("noise_mult", 1.0)
            flag = " [STN]" if sn < 1 else (" [NOISE]" if nm < 1 else "")
            print(f"  {t.pnl:>+8.2f}$  {t.entry_ts.strftime('%m-%d')}  "
                  f"{t.symbol:>8}  conf={t.signal_confidence:.2f}"
                  f"  adj={meta.get('adjusted_compound', 0):.2f}{flag}")
            print(f"             → {t.news_title[:72]}")

        print(f"\n{'─'*72}")
        print(f"  PER-COIN SUMMARY")
        print(f"{'─'*72}")
        by_coin: dict[str, list] = {}
        for t in metrics.trades:
            by_coin.setdefault(t.symbol, []).append(t)
        for sym, ct in sorted(by_coin.items()):
            w = sum(1 for t in ct if t.pnl > 0)
            tp = sum(t.pnl for t in ct)
            ac = sum(t.signal_confidence for t in ct) / len(ct)
            print(f"  {sym:>10}  {len(ct):>4} trades  W:{w} L:{len(ct)-w}"
                  f"  PnL:{tp:>+9.2f}$  avg_conf={ac:.2f}")


def main():
    p = argparse.ArgumentParser(description="TATC Real Backtest (HuggingFace data)")
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "BNB"],
                   help="Coin codes: BTC ETH SOL DOGE etc.")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end",   default="2024-04-01")
    p.add_argument("--interval", default="60",
                   help="Candle interval in minutes (only 60 supported for HF source)")
    p.add_argument("--capital",       type=float, default=10_000)
    p.add_argument("--position-size", type=float, default=0.1)
    p.add_argument("--hold-candles",  type=int,   default=5)
    p.add_argument("--min-confidence",type=float, default=0.3)
    p.add_argument("--buy-threshold", type=float, default=0.3)
    p.add_argument("--sell-threshold",type=float, default=-0.3)
    p.add_argument("--allow-short",    action="store_true")
    p.add_argument("--stop-loss",      type=float, default=0.0,
                   help="Stop-loss %% per trade, e.g. 0.02 = 2%% (0 = disabled)")
    p.add_argument("--take-profit",    type=float, default=0.0,
                   help="Take-profit %% per trade, e.g. 0.05 = 5%% (0 = disabled)")
    p.add_argument("--fee",            type=float, default=0.001,
                   help="Exchange fee per leg, default 0.001 = 0.1%%")
    p.add_argument("--signal-window",  type=int,   default=0,
                   help="Aggregate news over N hours before trading (0 = per-article)")
    p.add_argument("--cache-dir",      default="data/cache",
                   help="Directory for cached news+price data")
    p.add_argument("--no-cache",       action="store_true",
                   help="Ignore cache, re-download everything")
    p.add_argument("--strategy",       default="vader",
                   choices=["vader", "finbert", "combo"],
                   help="Sentiment strategy: vader (fast), finbert (accurate), combo (filtered)")
    p.add_argument("--cooldown",       type=int,   default=30,
                   help="[combo] Minutes of silence per coin after a signal (default: 30)")
    p.add_argument("--dedup-window",   type=int,   default=120,
                   help="[combo] Minutes to suppress duplicate headlines (default: 120)")
    p.add_argument("--combo-finbert",  action="store_true",
                   help="[combo] Also require FinBERT agreement (slower)")
    p.add_argument("--leverage",       type=float, default=1.0,
                   help="Futures leverage multiplier (1 = spot, 2/3/5/10 etc). "
                        "Enables liquidation at -(100/leverage)%% from entry.")
    p.add_argument("--news-exit",      action="store_true",
                   help="Exit LONG early when a SELL signal arrives during hold period")
    p.add_argument("--news-exit-confidence", type=float, default=0.0,
                   help="Min confidence for SELL signal to trigger early exit (0 = same as --min-confidence)")
    p.add_argument("--news-exit-only-loss", action="store_true",
                   help="Only exit early if the position is currently at a loss")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
