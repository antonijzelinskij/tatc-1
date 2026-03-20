"""
fetch_real_data.py
==================
Собирает РЕАЛЬНЫЕ данные из публичных источников без API-ключей.

Источники новостей:
  - CoinDesk RSS
  - CoinTelegraph RSS
  - CryptoSlate RSS
  - BeInCrypto RSS
  - Decrypt RSS
  - Bitcoin Magazine RSS
  - Investopedia Crypto RSS
  - The Block RSS
  - CryptoNews RSS
  - Bankless RSS

Источники цен:
  - Yahoo Finance (через yfinance), 1h-интервал, последние 730 дней

Результат:
  data/real_news.csv   — новости с реальными датами и источниками
  data/real_prices.csv — часовые цены BTC/ETH

Запуск:
  python data/fetch_real_data.py
"""

from __future__ import annotations

import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

# ─── параметры ──────────────────────────────────────────────────────────────

COINS = ["BTC", "ETH"]

# Ключевые слова для определения монеты из заголовка
COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "satoshi", "sats", "halving", "lightning network",
            "bitcoin etf", "spot etf", "microstrategy", "strategy"],
    "ETH": ["ethereum", "eth", "ether", "vitalik", "defi", "erc-20", "erc20",
            "dencun", "staking", "layer 2", "l2", "optimism", "arbitrum",
            "base chain", "solidity"],
}

# RSS-фиды крупных крипто-изданий
RSS_FEEDS: list[dict] = [
    {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "source": "CoinDesk",
    },
    {
        "url": "https://cointelegraph.com/rss",
        "source": "CoinTelegraph",
    },
    {
        "url": "https://cryptoslate.com/feed/",
        "source": "CryptoSlate",
    },
    {
        "url": "https://beincrypto.com/feed/",
        "source": "BeInCrypto",
    },
    {
        "url": "https://decrypt.co/feed",
        "source": "Decrypt",
    },
    {
        "url": "https://bitcoinmagazine.com/.rss/full/",
        "source": "BitcoinMagazine",
    },
    {
        "url": "https://www.theblock.co/rss.xml",
        "source": "TheBlock",
    },
    {
        "url": "https://cryptonews.com/news/feed/",
        "source": "CryptoNews",
    },
    {
        "url": "https://www.newsbtc.com/feed/",
        "source": "NewsBTC",
    },
    {
        "url": "https://ambcrypto.com/feed/",
        "source": "AMBCrypto",
    },
    {
        "url": "https://u.today/rss",
        "source": "U.Today",
    },
    {
        "url": "https://cryptopotato.com/feed/",
        "source": "CryptoPotato",
    },
]

# GDELT — историческое покрытие (бесплатно, без ключа)
# Запрос по ключевым словам за последние 3 месяца
GDELT_QUERIES: list[dict] = [
    {
        "query": "bitcoin cryptocurrency market",
        "coin": "BTC",
    },
    {
        "query": "ethereum blockchain crypto",
        "coin": "ETH",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CryptoBacktestBot/1.0; "
        "+https://github.com/crypto-backtest)"
    )
}

NEWS_OUT = Path(__file__).parent / "real_news.csv"
PRICES_OUT = Path(__file__).parent / "real_prices.csv"


# ─── вспомогательные функции ────────────────────────────────────────────────

def detect_coin(text: str) -> str | None:
    """Определяет монету по ключевым словам в тексте."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for coin, keywords in COIN_KEYWORDS.items():
        scores[coin] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else None


def parse_rfc2822(date_str: str | None) -> datetime | None:
    """Парсит RSS-дату в UTC datetime."""
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    except Exception:
        pass
    # Попытка вручную
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str.strip(), fmt).astimezone(timezone.utc)
        except Exception:
            pass
    return None


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


# ─── 1. Сбор новостей из RSS-фидов ─────────────────────────────────────────

def fetch_rss(feed: dict) -> list[dict]:
    url, source = feed["url"], feed["source"]
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] {source}: {e}")
        return []

    items = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        print(f"  [WARN] {source}: XML parse error")
        return []

    # Поддержка обоих форматов: RSS 2.0 и Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    rss_items = root.findall(".//item")
    atom_entries = root.findall(".//atom:entry", ns)
    all_entries = rss_items + atom_entries

    for entry in all_entries:
        # Заголовок
        title_el = entry.find("title")
        title = strip_html(title_el.text or "") if title_el is not None else ""
        if not title:
            continue

        # Дата
        date_el = (entry.find("pubDate")
                   or entry.find("dc:date", {"dc": "http://purl.org/dc/elements/1.1/"})
                   or entry.find("atom:published", ns)
                   or entry.find("atom:updated", ns))
        date_str = date_el.text if date_el is not None else None
        ts = parse_rfc2822(date_str)
        if ts is None:
            continue

        coin = detect_coin(title)
        if coin is None:
            continue

        items.append({
            "timestamp": ts.isoformat(),
            "coin": coin,
            "headline": title,
            "source": source,
        })

    print(f"  {source}: {len(items)} items")
    return items


def fetch_all_rss() -> list[dict]:
    print(f"\n[RSS] Fetching {len(RSS_FEEDS)} feeds...")
    all_items = []
    for feed in RSS_FEEDS:
        items = fetch_rss(feed)
        all_items.extend(items)
        time.sleep(0.3)  # вежливая задержка
    print(f"[RSS] Total: {len(all_items)} items")
    return all_items


# ─── 2. Сбор новостей из GDELT ──────────────────────────────────────────────

def fetch_gdelt(query_cfg: dict) -> list[dict]:
    """
    GDELT DOC 2.0 API — исторические новости.
    Документация: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    """
    from urllib.parse import urlencode
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query_cfg["query"],
        "mode": "artlist",
        "maxrecords": "250",
        "sort": "DateDesc",
        "format": "json",
    }
    url = f"{base}?{urlencode(params)}"
    coin = query_cfg["coin"]

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] GDELT ({coin}): {e}")
        return []

    articles = data.get("articles", []) if isinstance(data, dict) else []
    items = []
    for art in articles:
        title = art.get("title", "").strip()
        seendatetime = art.get("seendatetime", "")  # "20240115130000"
        if not title or not seendatetime:
            continue
        try:
            ts = datetime.strptime(seendatetime, "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        items.append({
            "timestamp": ts.isoformat(),
            "coin": coin,
            "headline": title,
            "source": "GDELT/" + art.get("domain", ""),
        })

    print(f"  GDELT ({coin}): {len(items)} items")
    return items


def fetch_all_gdelt() -> list[dict]:
    print(f"\n[GDELT] Querying {len(GDELT_QUERIES)} topics...")
    all_items = []
    for q in GDELT_QUERIES:
        items = fetch_gdelt(q)
        all_items.extend(items)
        time.sleep(1)
    print(f"[GDELT] Total: {len(all_items)} items")
    return all_items


# ─── 3. Сбор новостей через Wayback Machine ─────────────────────────────────

def fetch_wayback_snapshots(rss_url: str, source: str,
                             year_months: list[str]) -> list[dict]:
    """
    Загружает исторические RSS-снапшоты через Wayback Machine CDX API.
    year_months: список в формате ["202401", "202402", ...]
    """
    cdx_base = "http://web.archive.org/cdx/search/cdx"
    wb_base = "https://web.archive.org/web"
    items = []

    for ym in year_months:
        params = {
            "url": rss_url,
            "output": "json",
            "limit": "3",
            "from": ym + "01",
            "to": ym + "28",
            "filter": "statuscode:200",
            "fl": "timestamp,original",
        }
        try:
            resp = requests.get(
                cdx_base, params=params, headers=HEADERS, timeout=15
            )
            rows = resp.json()
        except Exception:
            continue

        if len(rows) < 2:
            continue

        for row in rows[1:2]:  # только первый снапшот за месяц
            snap_ts, orig_url = row
            snap_url = f"{wb_base}/{snap_ts}/{orig_url}"
            try:
                snap = requests.get(snap_url, headers=HEADERS, timeout=15)
                snap_items = _parse_rss_text(snap.text, source)
                items.extend(snap_items)
                time.sleep(0.5)
            except Exception:
                continue

    return items


def _parse_rss_text(xml_text: str, source: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for entry in root.findall(".//item"):
        title_el = entry.find("title")
        date_el = entry.find("pubDate")
        title = strip_html(title_el.text or "") if title_el is not None else ""
        if not title:
            continue
        ts = parse_rfc2822(date_el.text if date_el is not None else None)
        if ts is None:
            continue
        coin = detect_coin(title)
        if coin is None:
            continue
        items.append({
            "timestamp": ts.isoformat(),
            "coin": coin,
            "headline": title,
            "source": source,
        })
    return items


# Для исторического покрытия — скачиваем CoinDesk и CoinTelegraph за 2024
WAYBACK_FEEDS = [
    ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
    ("https://cointelegraph.com/rss", "CoinTelegraph"),
    ("https://cryptoslate.com/feed/", "CryptoSlate"),
]

# 2024 — год халвинга, отличный период для бэктеста
YEAR_MONTHS_2024 = [
    "202401", "202402", "202403", "202404", "202405", "202406",
    "202407", "202408", "202409", "202410", "202411", "202412",
]

# Последние месяцы 2025
YEAR_MONTHS_2025 = ["202501", "202502", "202503"]


def fetch_all_wayback() -> list[dict]:
    all_months = YEAR_MONTHS_2024 + YEAR_MONTHS_2025
    print(f"\n[Wayback] Fetching {len(WAYBACK_FEEDS)} feeds × {len(all_months)} months...")
    all_items = []
    for url, source in WAYBACK_FEEDS:
        items = fetch_wayback_snapshots(url, source, all_months)
        print(f"  {source}: {len(items)} items")
        all_items.extend(items)
    print(f"[Wayback] Total: {len(all_items)} items")
    return all_items


# ─── 4. Цены из Yahoo Finance ────────────────────────────────────────────────

def fetch_prices() -> pd.DataFrame:
    import yfinance as yf

    print(f"\n[yfinance] Downloading BTC/ETH hourly prices...")
    dfs = []

    for coin, ticker in [("BTC", "BTC-USD"), ("ETH", "ETH-USD")]:
        try:
            # 730 дней — максимум для часового интервала через yfinance
            df = yf.download(
                ticker,
                period="730d",
                interval="1h",
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print(f"  [WARN] {ticker}: empty data")
                continue
            df = df.reset_index()
            # Flatten multi-level columns if present
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            ts_col = "Datetime" if "Datetime" in df.columns else "Date"
            price_col = "Close"
            sub = df[[ts_col, price_col]].copy()
            sub.columns = ["timestamp", "price"]
            sub["coin"] = coin
            # Ensure UTC
            sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True)
            sub["timestamp"] = sub["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            dfs.append(sub)
            print(f"  {ticker}: {len(sub):,} rows  ({sub['timestamp'].iloc[0]} → {sub['timestamp'].iloc[-1]})")
        except Exception as e:
            print(f"  [WARN] {ticker}: {e}")

    if not dfs:
        raise RuntimeError("No price data fetched!")

    return pd.concat(dfs, ignore_index=True)[["timestamp", "coin", "price"]]


# ─── 5. Сборка итоговых датасетов ───────────────────────────────────────────

def build_news_df(all_items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_items)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp", "headline"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df = df[["timestamp", "coin", "headline", "source"]]
    return df


def main() -> None:
    print("=" * 60)
    print("  REAL DATA COLLECTION")
    print("=" * 60)

    # ── Цены ──────────────────────────────────────────────────────
    prices_df = fetch_prices()
    PRICES_OUT.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(PRICES_OUT, index=False)
    print(f"\n[Prices] Saved {len(prices_df):,} rows → {PRICES_OUT}")

    # ── Новости ───────────────────────────────────────────────────
    all_news: list[dict] = []

    # Источник 1: текущие RSS-фиды
    rss_items = fetch_all_rss()
    all_news.extend(rss_items)

    # Источник 2: GDELT (исторические, ~250 статей per coin)
    gdelt_items = fetch_all_gdelt()
    all_news.extend(gdelt_items)

    # Источник 3: Wayback Machine (2024+2025, потенциально тысячи)
    wayback_items = fetch_all_wayback()
    all_news.extend(wayback_items)

    news_df = build_news_df(all_news)

    if news_df.empty:
        print("[ERROR] No news collected!")
        sys.exit(1)

    news_df.to_csv(NEWS_OUT, index=False)

    # ── Статистика ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    print(f"  News total     : {len(news_df):,}")
    print(f"  News by coin   : {news_df['coin'].value_counts().to_dict()}")
    print(f"  News by source :")
    for src, cnt in news_df["source"].value_counts().head(15).items():
        print(f"    {src:<30} {cnt}")
    print(f"\n  Date range (news) : {news_df['timestamp'].min()} → {news_df['timestamp'].max()}")
    print(f"\n  Prices total   : {len(prices_df):,}")
    print(f"  Price date range  : {prices_df['timestamp'].min()} → {prices_df['timestamp'].max()}")
    print(f"\n  Saved to:")
    print(f"    News  : {NEWS_OUT}")
    print(f"    Prices: {PRICES_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
