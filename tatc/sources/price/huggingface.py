"""
HuggingFace price source — Binance spot 1h OHLCV data 2020–2024.
Dataset: adamzzzz/binance-klines-20240721 (covers BTCUSDT, ETHUSDT, SOLUSDT, etc.)
No API key required. Data up to 2024-07-20.

Reads Parquet files directly with symbol filter (efficient — no full scan).
"""
from datetime import datetime, timezone

from tatc.core.models import Candle

_DATASET_ID = "adamzzzz/binance-klines-20240721"
_PARQUET_FILES = [
    "data/spot.1h-00000-of-00003.parquet",
    "data/spot.1h-00001-of-00003.parquet",
    "data/spot.1h-00002-of-00003.parquet",
]

# Cache of which parquet files contain which symbols (populated on first use)
_symbol_file_cache: dict[str, str] = {}


def _hf_parquet_url(filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{_DATASET_ID}"
        f"/resolve/main/{filename}"
    )


def _find_parquet_for_symbol(symbol: str) -> list[str]:
    """Find which parquet file(s) contain data for the given symbol."""
    import pandas as pd

    if symbol in _symbol_file_cache:
        return [_symbol_file_cache[symbol]]

    from huggingface_hub import hf_hub_download

    for filename in _PARQUET_FILES:
        local = hf_hub_download(
            repo_id=_DATASET_ID,
            filename=filename,
            repo_type="dataset",
        )
        # Read only the symbol column to check presence
        symbols_in_file = pd.read_parquet(local, columns=["symbol"])["symbol"].unique()
        for sym in symbols_in_file:
            if sym not in _symbol_file_cache:
                _symbol_file_cache[sym] = local

    return [_symbol_file_cache[symbol]] if symbol in _symbol_file_cache else []


def get_candles(
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> list[Candle]:
    """
    Fetch OHLCV candles for symbol from HuggingFace Binance dataset.
    interval: only '60' (1h) is supported (maps to spot.1h parquet).
    start/end: 'YYYY-MM-DD'

    Data available: 2020-01-01 → 2024-07-20.
    """
    if interval not in ("60", "1h", "H"):
        raise ValueError(
            f"HuggingFace price source only supports 1h interval (got '{interval}'). "
            "Use interval='60'."
        )

    import pandas as pd
    from huggingface_hub import hf_hub_download

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"  [hf-price] loading {symbol} from {_DATASET_ID}...", end=" ", flush=True)

    all_dfs = []
    for filename in _PARQUET_FILES:
        local = hf_hub_download(
            repo_id=_DATASET_ID,
            filename=filename,
            repo_type="dataset",
        )
        df = pd.read_parquet(
            local,
            filters=[("symbol", "==", symbol)],
            columns=["index", "open_price", "high_price", "low_price", "close_price", "volume"],
        )
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print(f"no data for {symbol}")
        return []

    df = pd.concat(all_dfs).drop_duplicates(subset=["index"]).sort_values("index")

    # Normalize timestamps to UTC
    if df["index"].dt.tz is None:
        df["index"] = df["index"].dt.tz_localize("UTC")
    else:
        df["index"] = df["index"].dt.tz_convert("UTC")

    # Filter date range
    df = df[(df["index"] >= start_dt) & (df["index"] <= end_dt)]

    print(f"{len(df)} candles")

    candles = []
    for _, row in df.iterrows():
        ts = row["index"].to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        candles.append(Candle(
            symbol=symbol,
            timestamp=ts,
            open=float(row["open_price"]),
            high=float(row["high_price"]),
            low=float(row["low_price"]),
            close=float(row["close_price"]),
            volume=float(row["volume"]),
        ))

    return candles
