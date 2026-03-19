"""
Генератор синтетических исторических данных для демонстрации бэктеста.

Создаёт два CSV-файла:
    data/sample_news.csv    — новости с метками монет и временем
    data/sample_prices.csv  — тиковые цены (1-секундный интервал)

Запуск:
    python data/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUTPUT_DIR = Path(__file__).parent

# Синтетические заголовки новостей: явно позитивные / негативные / нейтральные
BULLISH_HEADLINES = [
    "Bitcoin breaks all-time high as institutional demand surges",
    "Major bank announces crypto custody services for Bitcoin",
    "ETH 2.0 staking rewards hit record levels, attracting billions",
    "SEC approves spot Bitcoin ETF, market rallies",
    "MicroStrategy adds another 10,000 BTC to treasury reserves",
    "Ethereum smart contract volume reaches new milestone",
    "PayPal expands crypto payment options globally",
    "Bitcoin mining difficulty drops 10%, network becomes more profitable",
]

BEARISH_HEADLINES = [
    "China bans all cryptocurrency transactions and mining",
    "Major exchange files for bankruptcy amid liquidity crisis",
    "Bitcoin whale sells 50,000 BTC in single transaction",
    "Regulators freeze crypto assets at top-5 exchange",
    "Ethereum network suffers major exploit, $500M drained",
    "FTX collapse triggers industry-wide contagion fears",
    "SEC sues largest crypto exchange for securities violations",
    "Bitcoin miners forced to sell holdings at record pace",
]

NEUTRAL_HEADLINES = [
    "Bitcoin price consolidates near $40,000 support level",
    "Ethereum developers announce next testnet upgrade date",
    "Crypto market volume remains stable amid low volatility",
    "Bitcoin hash rate steady as miners await next difficulty adjustment",
    "DeFi protocols report normal activity levels this week",
]


def generate_news(
    start: str = "2024-01-01 00:00:00",
    n_events: int = 80,
    coins: list[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    coins = coins or ["BTC", "ETH"]

    # Случайные интервалы между новостями: от 5 до 120 минут
    intervals_min = rng.integers(5, 120, size=n_events)
    timestamps = pd.date_range(start=start, periods=1, freq="min").tolist()
    for delta in intervals_min[1:]:
        timestamps.append(timestamps[-1] + pd.Timedelta(minutes=int(delta)))

    all_headlines = BULLISH_HEADLINES + BEARISH_HEADLINES + NEUTRAL_HEADLINES
    headlines = rng.choice(all_headlines, size=n_events)
    selected_coins = rng.choice(coins, size=n_events)

    df = pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in timestamps],
        "coin": selected_coins,
        "headline": headlines,
        "source": rng.choice(["CoinDesk", "CryptoNews", "Bloomberg", "Reuters"], size=n_events),
    })
    return df


def generate_prices(
    news_df: pd.DataFrame,
    price_config: dict[str, float] | None = None,
    tick_interval_sec: int = 1,
    window_before_min: int = 1,
    window_after_min: int = 5,
) -> pd.DataFrame:
    """Генерирует тиковые цены вокруг каждого события новости.

    Для каждой новости генерируем окно цен [T-1min .. T+5min] с 1-сек тиками.
    Добавляем шум и небольшой тренд по направлению настроения новости.
    """
    rng = np.random.default_rng(SEED + 1)
    price_config = price_config or {"BTC": 40_000.0, "ETH": 2_500.0}

    rows = []
    last_prices: dict[str, float] = dict(price_config)

    for _, news_row in news_df.iterrows():
        coin = news_row["coin"]
        news_ts = pd.Timestamp(news_row["timestamp"])
        base_price = last_prices.get(coin, 1000.0)

        start_ts = news_ts - pd.Timedelta(minutes=window_before_min)
        end_ts   = news_ts + pd.Timedelta(minutes=window_after_min)
        ticks    = pd.date_range(start=start_ts, end=end_ts, freq=f"{tick_interval_sec}s")

        # Определяем тренд по тексту новости
        headline = news_row["headline"].lower()
        if any(w in headline for w in ["high", "approval", "record", "buy", "expand", "adds"]):
            trend = 0.0002  # +0.02% за тик в среднем
        elif any(w in headline for w in ["ban", "bankruptcy", "exploit", "sues", "collapse", "freeze"]):
            trend = -0.0002
        else:
            trend = 0.0

        price = base_price
        for ts in ticks:
            noise = rng.normal(0, 0.0005)
            price = price * (1 + trend + noise)
            price = max(price, 1.0)  # не уходим в ноль
            rows.append({"timestamp": ts.isoformat(), "coin": coin, "price": round(price, 4)})

        last_prices[coin] = price

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp", "coin"])
    df = df.sort_values(["coin", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating sample data...")

    news_df = generate_news(n_events=80)
    prices_df = generate_prices(news_df)

    news_path   = OUTPUT_DIR / "sample_news.csv"
    prices_path = OUTPUT_DIR / "sample_prices.csv"

    news_df.to_csv(news_path, index=False)
    prices_df.to_csv(prices_path, index=False)

    print(f"✓ News:   {news_path}   ({len(news_df)} rows)")
    print(f"✓ Prices: {prices_path} ({len(prices_df)} rows)")
    print("\nSample news:")
    print(news_df.head(5).to_string(index=False))
    print("\nSample prices:")
    print(prices_df.head(10).to_string(index=False))
