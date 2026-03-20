"""
Smoke test: synthetic candles (realistic BTC/ETH 2024 price action) + mock news.
Runs end-to-end without any network or API keys.
"""
import math
import random
from datetime import datetime, timedelta, timezone

from tatc.backtester.engine import BacktestConfig, BacktestEngine
from tatc.core.models import Candle, NewsItem
from tatc.strategies.sentiment_vader import VaderStrategy

# Seed for reproducibility
random.seed(42)


# --- Realistic BTC price anchors for 2024 (approximate) ---
BTC_ANCHORS = [
    ("2024-01-01", 42_000),
    ("2024-01-11", 46_000),   # ETF approved
    ("2024-01-17", 41_000),   # sell the news
    ("2024-02-12", 50_000),   # breakout
    ("2024-02-26", 57_000),
    ("2024-03-05", 69_000),   # ATH
    ("2024-03-11", 65_000),
    ("2024-04-01", 66_000),
    ("2024-04-20", 63_000),   # halving
    ("2024-04-22", 60_000),
    ("2024-05-23", 67_000),   # ETH ETF approval effect
    ("2024-06-01", 68_000),
    ("2024-06-20", 63_000),
    ("2024-07-01", 60_000),   # Mt Gox fears
    ("2024-07-16", 65_000),   # Trump rally
    ("2024-08-01", 58_000),
    ("2024-08-05", 49_000),   # global crash
    ("2024-08-15", 58_000),
    ("2024-09-01", 57_000),
]

# ETH roughly tracks BTC but at different levels
ETH_MULTIPLIER = 0.048    # ETH/BTC ratio ~0.048
DOGE_BASE = 0.08
SOL_BASE = 100.0


def parse_anchor_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def interpolate_prices(anchors: list[tuple[str, float]], interval_hours: int = 1) -> list[tuple[datetime, float]]:
    """Linear interpolation + gaussian noise between price anchors."""
    result = []
    parsed = [(parse_anchor_date(d), p) for d, p in anchors]

    for i in range(len(parsed) - 1):
        t0, p0 = parsed[i]
        t1, p1 = parsed[i + 1]
        hours = int((t1 - t0).total_seconds() / 3600)
        for h in range(hours):
            frac = h / hours
            price = p0 + (p1 - p0) * frac
            # Add realistic noise (~0.3% per hour)
            noise = random.gauss(0, price * 0.003)
            result.append((t0 + timedelta(hours=h), price + noise))

    return result


def make_candles(symbol: str, prices: list[tuple[datetime, float]]) -> list[Candle]:
    candles = []
    for ts, close in prices:
        spread = close * random.uniform(0.002, 0.008)
        open_ = close + random.gauss(0, spread * 0.3)
        high = max(open_, close) + abs(random.gauss(0, spread))
        low = min(open_, close) - abs(random.gauss(0, spread))
        volume = random.uniform(100, 2000)
        candles.append(Candle(
            symbol=symbol,
            timestamp=ts,
            open=max(open_, 0.001),
            high=max(high, 0.001),
            low=max(low, 0.001),
            close=max(close, 0.001),
            volume=volume,
        ))
    return candles


# --- Mock news items ---
MOCK_NEWS = [
    ("2024-01-11", "BTC", "Bitcoin ETF approved by SEC, massive inflows expected"),
    ("2024-01-12", "BTC", "Bitcoin ETF launches with record trading volume"),
    ("2024-01-17", "BTC", "Bitcoin crashes after ETF sell-the-news selloff"),
    ("2024-02-05", "BTC ETH", "Crypto market surges as Fed signals rate cuts ahead"),
    ("2024-02-12", "BTC", "Bitcoin breaks 50000 bullish momentum building"),
    ("2024-02-26", "BTC", "Bitcoin smashes through 57000 amid halving excitement"),
    ("2024-03-01", "ETH", "Ethereum ETF speculation drives ETH price surge"),
    ("2024-03-05", "BTC", "Bitcoin hits all-time high soars to 69000"),
    ("2024-03-11", "BTC", "Bitcoin correction dumps after hitting new ATH profit taking"),
    ("2024-03-15", "DOGE", "Dogecoin surges 40 percent massive pump following viral tweet"),
    ("2024-03-18", "DOGE", "Dogecoin crashes dumps hard after initial pump fades"),
    ("2024-04-01", "BTC", "Bitcoin halving approaches bullish sentiment grows"),
    ("2024-04-20", "BTC", "Bitcoin halving complete block reward cut to 3.125 BTC"),
    ("2024-04-22", "BTC ETH", "Post-halving selloff correction hits crypto markets"),
    ("2024-05-01", "ETH", "Ethereum Pectra upgrade approved bullish for ETH"),
    ("2024-05-10", "BTC", "Mt Gox creditor repayments create bearish selling pressure fears"),
    ("2024-05-15", "BTC ETH", "Crypto markets rally recovery as macro conditions improve"),
    ("2024-05-23", "ETH", "SEC approves spot Ethereum ETF massive approval catalyst"),
    ("2024-05-25", "ETH", "Ethereum surges rallies 20 percent on ETF approval news"),
    ("2024-06-01", "SOL", "Solana ecosystem booming bullish DeFi projects launching"),
    ("2024-06-15", "DOGE SHIB", "Meme coins crash dumps as speculative frenzy dies down"),
    ("2024-06-20", "BTC ETH SOL", "Crypto market panic crash bearish regulatory fears resurface"),
    ("2024-07-01", "BTC", "Mt Gox begins Bitcoin repayments bearish selling pressure"),
    ("2024-07-16", "BTC", "Bitcoin surges rallies bullish as political momentum grows"),
    ("2024-08-01", "BTC ETH", "Global market crash recession fears crypto tanks plunges"),
    ("2024-08-05", "BTC", "Bitcoin plunges crashes below 50000 in global risk-off selloff"),
    ("2024-08-15", "BTC ETH", "Crypto rallies recovers as recession fears ease bullish"),
]


def make_news_items(raw: list) -> list[NewsItem]:
    items = []
    for i, (date_str, coins_str, title) in enumerate(raw):
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=9, minute=(i * 7) % 60
        )
        coins = coins_str.split()
        items.append(NewsItem(
            id=str(i), timestamp=dt, title=title, body="",
            source="mock", url="", coins=coins,
        ))
    return items


def main():
    print("=" * 55)
    print("  TATC Backtest — Synthetic 2024 Price Data")
    print("  Period: 2024-01-01 → 2024-09-01 (1h candles)")
    print("=" * 55)

    # Generate synthetic candles
    btc_prices = interpolate_prices(BTC_ANCHORS)
    eth_prices = [(ts, p * ETH_MULTIPLIER) for ts, p in btc_prices]
    doge_prices = [(ts, DOGE_BASE * (1 + (p / 42000 - 1) * 1.5 + random.gauss(0, 0.02)))
                   for ts, p in btc_prices]
    sol_prices = [(ts, SOL_BASE * (1 + (p / 42000 - 1) * 2.0 + random.gauss(0, 0.03)))
                  for ts, p in btc_prices]

    candles = {
        "BTCUSDT": make_candles("BTCUSDT", btc_prices),
        "ETHUSDT": make_candles("ETHUSDT", eth_prices),
        "DOGEUSDT": make_candles("DOGEUSDT", doge_prices),
        "SOLUSDT": make_candles("SOLUSDT", sol_prices),
    }

    for sym, clist in candles.items():
        first, last = clist[0], clist[-1]
        change = (last.close - first.close) / first.close * 100
        print(f"  {sym}: {len(clist)} candles, "
              f"${first.close:.2f} → ${last.close:.2f} ({change:+.1f}%)")

    news_items = make_news_items(MOCK_NEWS)

    # --- Run backtest ---
    strategy = VaderStrategy(buy_threshold=0.3, sell_threshold=-0.3)

    print("\nSentiment analysis:")
    for item in news_items:
        sig = strategy.analyze(item)
        compound = sig.metadata.get("compound", 0)
        icon = "▲ BUY " if sig.signal.value == "BUY" else (
               "▼ SELL" if sig.signal.value == "SELL" else "─ HOLD")
        print(f"  {icon} [{compound:+.2f}] {item.title[:58]}")

    print(f"\n{'='*55}")
    print("  Running backtest (10k capital, 10% per trade, 5h hold)")
    print(f"{'='*55}")

    config = BacktestConfig(
        initial_capital=10_000.0,
        position_size_pct=0.1,
        hold_candles=5,
        min_confidence=0.2,
        allow_short=True,
        slippage_pct=0.001,
    )
    engine = BacktestEngine(strategy, config)
    metrics = engine.run(candles, news_items)
    print(metrics)


if __name__ == "__main__":
    main()
