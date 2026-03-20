from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class NewsItem:
    id: str
    timestamp: datetime
    title: str
    body: str
    source: str
    url: str
    coins: list[str] = field(default_factory=list)  # e.g. ["BTC", "ETH"]


@dataclass
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    signal: SignalType
    confidence: float  # 0.0 to 1.0
    symbol: str
    news: NewsItem
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    signal: Optional[Signal] = None


@dataclass
class Position:
    symbol: str
    quantity: float        # positive = long
    avg_entry_price: float
    opened_at: datetime

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    def pnl(self, current_price: float) -> float:
        return (current_price - self.avg_entry_price) * self.quantity
