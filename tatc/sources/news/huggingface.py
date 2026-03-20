"""
HuggingFace news source — CoinDesk/Decrypt/CryptoPolitain articles 2019–2025.
Dataset: maryamfakhari/crypto-news-coindesk-2020-2025 (229k articles, no auth required).

Categories field contains coin codes: BTC, ETH, SOL, etc.
"""
from datetime import datetime, timezone

from tatc.core.models import NewsItem

_DATASET_ID = "maryamfakhari/crypto-news-coindesk-2020-2025"

# Map coin code → strings that appear in categories field
_COIN_ALIASES: dict[str, list[str]] = {
    "BTC":  ["BTC", "BITCOIN"],
    "ETH":  ["ETH", "ETHEREUM"],
    "SOL":  ["SOL", "SOLANA"],
    "BNB":  ["BNB"],
    "XRP":  ["XRP", "RIPPLE"],
    "DOGE": ["DOGE", "DOGECOIN"],
    "ADA":  ["ADA", "CARDANO"],
    "AVAX": ["AVAX", "AVALANCHE"],
    "DOT":  ["DOT", "POLKADOT"],
    "MATIC":["MATIC", "POLYGON"],
    "LINK": ["LINK", "CHAINLINK"],
    "UNI":  ["UNI", "UNISWAP"],
    "LTC":  ["LTC", "LITECOIN"],
    "ATOM": ["ATOM", "COSMOS"],
    "NEAR": ["NEAR"],
    "ARB":  ["ARB", "ARBITRUM"],
    "OP":   ["OP", "OPTIMISM"],
    "INJ":  ["INJ", "INJECTIVE"],
    "SUI":  ["SUI"],
    "APT":  ["APT", "APTOS"],
}


def _parse_timestamp(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        # Format: "2024-01-15 10:30:00.000000"
        dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return None


def _row_coins(row: dict) -> list[str]:
    """Extract coin codes from categories + tags fields."""
    cats = (row.get("categories") or "").upper()
    tags = (row.get("tags") or "").upper()
    combined = cats + "|" + tags

    found = []
    for coin, aliases in _COIN_ALIASES.items():
        if any(alias in combined for alias in aliases):
            found.append(coin)
    return found


def fetch_historical(
    start: str,
    end: str,
    coins: list[str] | None = None,
) -> list[NewsItem]:
    """
    Load CoinDesk/Decrypt articles from HuggingFace dataset.
    Filters by date range and optionally by coin code.

    start/end: 'YYYY-MM-DD'
    coins: list of coin codes ['BTC', 'ETH', ...]
    """
    from datasets import load_dataset  # lazy import — heavy dependency

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    coins_upper = [c.upper() for c in coins] if coins else []
    coin_aliases_needed: set[str] = set()
    for coin in coins_upper:
        for alias in _COIN_ALIASES.get(coin, [coin]):
            coin_aliases_needed.add(alias)

    print(f"  [hf-news] loading {_DATASET_ID}...", end=" ", flush=True)
    ds = load_dataset(_DATASET_ID, split="train")
    print(f"{len(ds)} articles loaded")

    items: list[NewsItem] = []
    for i, row in enumerate(ds):
        ts = _parse_timestamp(row.get("published_on") or "")
        if ts is None:
            continue
        if not (start_dt <= ts <= end_dt):
            continue

        row_coins = _row_coins(row)

        # If coin filter specified, check intersection
        if coins_upper:
            if not any(c in coins_upper for c in row_coins):
                continue

        title = (row.get("title") or "").strip()
        if not title:
            continue

        body = (row.get("body") or "").strip()[:2000]
        source = row.get("source") or "coindesk"
        url = row.get("url") or ""
        item_id = str(row.get("id") or f"hf-{i}")

        items.append(NewsItem(
            id=item_id,
            timestamp=ts,
            title=title,
            body=body,
            source=source,
            url=url,
            coins=row_coins if row_coins else coins_upper,
        ))

    items.sort(key=lambda x: x.timestamp)
    return items
