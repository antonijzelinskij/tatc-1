"""
Event-driven backtesting engine.

Flow:
  1. Load historical news + candles for the period
  2. Merge into a single timeline of events (news or candle)
  3. For each news event → run Strategy → generate Signal
  4. On BUY signal → open position at next candle's open price
  5. Exit after hold_candles candles (simple exit) or on SELL signal
  6. Track equity curve, compute metrics at end
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Union

from tatc.backtester.metrics import BacktestMetrics, compute_metrics
from tatc.core.models import (
    Candle,
    NewsItem,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    Signal,
    SignalType,
)
from tatc.core.protocols import Strategy


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0   # USD
    position_size_pct: float = 0.1      # fraction of capital per trade
    hold_candles: int = 5               # how many candles to hold before force-exit
    min_confidence: float = 0.0         # skip signals below this confidence
    allow_short: bool = False           # if True, SELL signals open short positions
    slippage_pct: float = 0.001         # 0.1% slippage on fills


# Timeline event: either a candle or a news item
Event = Union[tuple[str, Candle], tuple[str, NewsItem]]


class BacktestEngine:
    """
    Runs a strategy against historical news + price data.

    Usage:
        engine = BacktestEngine(strategy, config)
        metrics = await engine.run(candles_by_symbol, news_items)
        print(metrics)
    """

    def __init__(self, strategy: Strategy, config: BacktestConfig | None = None):
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(
        self,
        candles: dict[str, list[Candle]],   # symbol → sorted candles
        news_items: list[NewsItem],
    ) -> BacktestMetrics:
        """
        Run backtest synchronously (no async needed — data already loaded).

        candles: {symbol: [Candle, ...]} sorted by timestamp
        news_items: list of NewsItem sorted by timestamp
        """
        cfg = self.config
        capital = cfg.initial_capital
        equity_curve: list[float] = [capital]

        # Open positions per symbol: symbol → (position, entry_order, candles_held)
        open_positions: dict[str, tuple[Position, Order, int]] = {}
        completed_trades: list[tuple[Order, Order]] = []

        # Build per-symbol candle index for quick lookup
        candle_idx: dict[str, int] = {sym: 0 for sym in candles}

        # Build unified timeline: (timestamp, type, payload)
        timeline: list[tuple[datetime, str, object]] = []
        for item in news_items:
            timeline.append((item.timestamp, "news", item))
        for sym, clist in candles.items():
            for candle in clist:
                timeline.append((candle.timestamp, "candle", candle))
        timeline.sort(key=lambda x: x[0])

        for ts, etype, payload in timeline:
            if etype == "candle":
                candle: Candle = payload  # type: ignore
                sym = candle.symbol

                # Check if we need to exit open position for this symbol
                if sym in open_positions:
                    pos, entry_order, held = open_positions[sym]
                    held += 1

                    if held >= cfg.hold_candles:
                        # Force-exit at candle close
                        exit_price = candle.close * (1 - cfg.slippage_pct)
                        exit_order = Order(
                            id=str(uuid.uuid4()),
                            symbol=sym,
                            side=OrderSide.SELL if pos.is_long else OrderSide.BUY,
                            quantity=pos.quantity,
                            price=exit_price,
                            timestamp=candle.timestamp,
                            status=OrderStatus.FILLED,
                        )
                        pnl = pos.pnl(exit_price)
                        capital += pos.quantity * exit_price if pos.is_long else pnl
                        equity_curve.append(capital)
                        completed_trades.append((entry_order, exit_order))
                        del open_positions[sym]
                    else:
                        open_positions[sym] = (pos, entry_order, held)

            elif etype == "news":
                news: NewsItem = payload  # type: ignore

                # Generate signal
                signal: Signal = self.strategy.analyze(news)

                if signal.signal == SignalType.HOLD:
                    continue
                if signal.confidence < cfg.min_confidence:
                    continue

                sym = signal.symbol
                if sym not in candles:
                    continue  # no price data for this symbol

                # Find next available candle for this symbol after news timestamp
                clist = candles[sym]
                idx = candle_idx.get(sym, 0)
                while idx < len(clist) and clist[idx].timestamp <= ts:
                    idx += 1
                candle_idx[sym] = idx

                if idx >= len(clist):
                    continue  # no future candle available

                next_candle = clist[idx]

                # Skip if already in position for this symbol
                if sym in open_positions:
                    continue

                # Position sizing
                trade_capital = capital * cfg.position_size_pct
                if trade_capital <= 0:
                    continue

                entry_price = next_candle.open * (1 + cfg.slippage_pct)
                quantity = trade_capital / entry_price

                if signal.signal == SignalType.BUY:
                    side = OrderSide.BUY
                elif signal.signal == SignalType.SELL and cfg.allow_short:
                    side = OrderSide.SELL
                    entry_price = next_candle.open * (1 - cfg.slippage_pct)
                else:
                    continue

                entry_order = Order(
                    id=str(uuid.uuid4()),
                    symbol=sym,
                    side=side,
                    quantity=quantity,
                    price=entry_price,
                    timestamp=next_candle.timestamp,
                    status=OrderStatus.FILLED,
                    signal=signal,
                )
                position = Position(
                    symbol=sym,
                    quantity=quantity if side == OrderSide.BUY else -quantity,
                    avg_entry_price=entry_price,
                    opened_at=next_candle.timestamp,
                )
                capital -= trade_capital
                open_positions[sym] = (position, entry_order, 0)

        # Force-close remaining open positions at last candle price
        for sym, (pos, entry_order, _) in open_positions.items():
            clist = candles.get(sym, [])
            if not clist:
                continue
            last_price = clist[-1].close
            exit_price = last_price * (1 - cfg.slippage_pct)
            exit_order = Order(
                id=str(uuid.uuid4()),
                symbol=sym,
                side=OrderSide.SELL if pos.is_long else OrderSide.BUY,
                quantity=abs(pos.quantity),
                price=exit_price,
                timestamp=clist[-1].timestamp,
                status=OrderStatus.FILLED,
            )
            pnl = pos.pnl(exit_price)
            capital += abs(pos.quantity) * exit_price if pos.is_long else pnl
            equity_curve.append(capital)
            completed_trades.append((entry_order, exit_order))

        return compute_metrics(completed_trades, cfg.initial_capital, equity_curve)
