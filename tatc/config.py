"""
Central configuration. Override via environment variables or .env file.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY", "")
BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

# --- Coins to trade (Bybit spot symbols) ---
# Top coins + popular shitcoins. Extend freely.
SYMBOLS: list[str] = [
    # Majors
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    # Mid caps
    "NEARUSDT", "ATOMUSDT", "AAVEUSDT", "UNIUSDT", "LTCUSDT",
    "ETCUSDT", "FILUSDT", "VETUSDT", "ICPUSDT", "APTUSDT",
    # Shitcoins / memes
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT",
    "WIFUSDT", "MEMEUSDT", "TURBOUSDT", "BRETTUSDT", "MOGUSDT",
    # DeFi / L2
    "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
    "TIAUSDT", "JUPUSDT", "PYTHUSDT", "WUSDT", "STRKUSDT",
]

# Coin code → symbol mapping for news matching
# (CryptoPanic uses short codes like "BTC")
COIN_TO_SYMBOL: dict[str, str] = {
    sym.replace("USDT", ""): sym for sym in SYMBOLS
}

# --- Backtester defaults ---
DEFAULT_INTERVAL = "60"          # 1-hour candles
DEFAULT_HOLD_CANDLES = 5         # hold for 5 candles before force-exit
DEFAULT_POSITION_SIZE_PCT = 0.1  # 10% of capital per trade
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_MIN_CONFIDENCE = 0.2     # skip low-confidence signals

# --- Strategy defaults ---
VADER_BUY_THRESHOLD = 0.3
VADER_SELL_THRESHOLD = -0.3
