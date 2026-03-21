"""
Event-driven backtesting engine.

Flow:
  1. Load historical news + candles for the period
  2. Merge into a single timeline of events (news or candle)
  3. For each news event → run Strategy → generate Signal
  4. On BUY signal → open position at next candle's open price
  5. Exit after hold_candles candles, on SL/TP hit, or on SELL signal
  6. Track equity curve, compute metrics at end

New in v2:
  - fee_pct: trading fee applied on both entry and exit legs
  - stop_loss_pct / take_profit_pct: automatic exits on price triggers
  - signal_window_hours: aggregate news sentiment over a rolling window
    instead of reacting to each article individually
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
    position_size_pct: float = 0.1      # fraction of capital per trade (= margin fraction)
    hold_candles: int = 5               # candles to hold before force-exit (0 = no timeout)
    min_confidence: float = 0.0         # skip signals below this confidence
    allow_short: bool = False           # if True, SELL signals open short positions
    slippage_pct: float = 0.001         # 0.1% slippage on fills
    fee_pct: float = 0.001              # 0.1% fee per leg on full notional
    stop_loss_pct: float = 0.0          # 0 = disabled; price % move (e.g. 0.02 = 2% SL)
    take_profit_pct: float = 0.0        # 0 = disabled; price % move (e.g. 0.05 = 5% TP)
    signal_window_hours: int = 0        # 0 = per-news; >0 = aggregate over N hours
    leverage: float = 1.0               # 1x = spot; 2x/3x/5x/10x for futures
    maintenance_margin_pct: float = 0.005  # 0.5% maintenance margin (Binance standard)


# Timeline event: either a candle or a news item
Event = Union[tuple[str, Candle], tuple[str, NewsItem]]


class BacktestEngine:
    """
    Runs a strategy against historical news + price data.

    Usage:
        engine = BacktestEngine(strategy, config)
        metrics = engine.run(candles, news_items)
        print(metrics)
    """

    def __init__(self, strategy: Strategy, config: BacktestConfig | None = None):
        self.strategy = strategy
        self.config = config or BacktestConfig()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def run(
        self,
        candles: dict[str, list[Candle]],   # symbol → sorted candles
        news_items: list[NewsItem],
    ) -> BacktestMetrics:
        cfg = self.config

        # When windowing is active, pre-aggregate news into time buckets.
        # Only create buckets for coins we actually have price data for.
        if cfg.signal_window_hours > 0:
            tradeable_coins = {sym[:-4] for sym in candles}  # "BTCUSDT" → "BTC"
            effective_news = self._aggregate_news(
                news_items, cfg.signal_window_hours, tradeable_coins
            )
        else:
            effective_news = news_items

        capital = cfg.initial_capital
        equity_curve: list[float] = [capital]

        # Open positions: symbol → (position, entry_order, candles_held)
        open_positions: dict[str, tuple[Position, Order, int]] = {}
        completed_trades: list[tuple[Order, Order]] = []

        # Per-symbol candle pointer for "next candle after news" lookup
        candle_idx: dict[str, int] = {sym: 0 for sym in candles}

        # Build unified timeline
        timeline: list[tuple[datetime, str, object]] = []
        for item in effective_news:
            timeline.append((item.timestamp, "news", item))
        for sym, clist in candles.items():
            for candle in clist:
                timeline.append((candle.timestamp, "candle", candle))
        timeline.sort(key=lambda x: x[0])

        for ts, etype, payload in timeline:

            # ── Candle event ──────────────────────────────────────
            if etype == "candle":
                candle: Candle = payload  # type: ignore
                sym = candle.symbol

                if sym not in open_positions:
                    continue

                pos, entry_order, held = open_positions[sym]

                # Check liquidation first (catastrophic — entire margin lost)
                liq_price = self._liquidation_price(pos, cfg)
                if liq_price is not None and self._is_liquidated(pos, candle, liq_price):
                    capital, exit_order = self._close_position(
                        pos, entry_order, liq_price, candle.timestamp, capital, cfg,
                        liquidated=True,
                    )
                    equity_curve.append(capital)
                    completed_trades.append((entry_order, exit_order))
                    del open_positions[sym]
                    continue

                # Check SL / TP before incrementing hold counter
                exit_price, exit_reason = self._check_sl_tp(pos, candle, cfg)

                if exit_price is None:
                    held += 1
                    if cfg.hold_candles > 0 and held >= cfg.hold_candles:
                        exit_price = candle.close * (1 - cfg.slippage_pct)
                        exit_reason = "timeout"
                    else:
                        open_positions[sym] = (pos, entry_order, held)
                        continue

                # Close position
                capital, exit_order = self._close_position(
                    pos, entry_order, exit_price, candle.timestamp, capital, cfg
                )
                equity_curve.append(capital)
                completed_trades.append((entry_order, exit_order))
                del open_positions[sym]

            # ── News event ────────────────────────────────────────
            elif etype == "news":
                news: NewsItem = payload  # type: ignore
                signal: Signal = self.strategy.analyze(news)

                if signal.signal == SignalType.HOLD:
                    continue
                if signal.confidence < cfg.min_confidence:
                    continue

                sym = signal.symbol
                if sym not in candles:
                    continue

                # Skip if already in position
                if sym in open_positions:
                    continue

                # Find next candle after news timestamp
                clist = candles[sym]
                idx = candle_idx.get(sym, 0)
                while idx < len(clist) and clist[idx].timestamp <= ts:
                    idx += 1
                candle_idx[sym] = idx

                if idx >= len(clist):
                    continue

                next_candle = clist[idx]

                # Open position
                trade_capital = capital * cfg.position_size_pct
                if trade_capital <= 0:
                    continue

                if signal.signal == SignalType.BUY:
                    side = OrderSide.BUY
                    entry_price = next_candle.open * (1 + cfg.slippage_pct)
                elif signal.signal == SignalType.SELL and cfg.allow_short:
                    side = OrderSide.SELL
                    entry_price = next_candle.open * (1 - cfg.slippage_pct)
                else:
                    continue

                notional = trade_capital * cfg.leverage
                quantity = notional / entry_price
                entry_fee = notional * cfg.fee_pct
                capital -= trade_capital + entry_fee  # deduct margin + fee on full notional

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
                open_positions[sym] = (position, entry_order, 0)

        # Force-close remaining open positions at last candle
        for sym, (pos, entry_order, _) in open_positions.items():
            clist = candles.get(sym, [])
            if not clist:
                continue
            last_candle = clist[-1]
            exit_price = last_candle.close * (1 - cfg.slippage_pct)
            capital, exit_order = self._close_position(
                pos, entry_order, exit_price, last_candle.timestamp, capital, cfg
            )
            equity_curve.append(capital)
            completed_trades.append((entry_order, exit_order))

        return compute_metrics(completed_trades, cfg.initial_capital, equity_curve, candles)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_sl_tp(
        pos: Position, candle: Candle, cfg: BacktestConfig
    ) -> tuple[float | None, str | None]:
        """
        Check if SL or TP was hit during this candle.
        SL takes priority when both are hit (pessimistic assumption).
        Returns (exit_price, reason) or (None, None).
        """
        if pos.is_long:
            sl_price = pos.avg_entry_price * (1 - cfg.stop_loss_pct) if cfg.stop_loss_pct > 0 else None
            tp_price = pos.avg_entry_price * (1 + cfg.take_profit_pct) if cfg.take_profit_pct > 0 else None
            sl_hit = sl_price is not None and candle.low <= sl_price
            tp_hit = tp_price is not None and candle.high >= tp_price
        else:
            sl_price = pos.avg_entry_price * (1 + cfg.stop_loss_pct) if cfg.stop_loss_pct > 0 else None
            tp_price = pos.avg_entry_price * (1 - cfg.take_profit_pct) if cfg.take_profit_pct > 0 else None
            sl_hit = sl_price is not None and candle.high >= sl_price
            tp_hit = tp_price is not None and candle.low <= tp_price

        if sl_hit:
            return sl_price, "stop_loss"  # type: ignore[return-value]
        if tp_hit:
            return tp_price, "take_profit"  # type: ignore[return-value]
        return None, None

    @staticmethod
    def _liquidation_price(pos: Position, cfg: BacktestConfig) -> float | None:
        """
        Compute the liquidation price for a leveraged position.
        Returns None for spot (leverage <= 1).

        Long  liq = entry * (1 - 1/leverage + maintenance_margin)
        Short liq = entry * (1 + 1/leverage - maintenance_margin)
        """
        if cfg.leverage <= 1.0:
            return None
        ep = pos.avg_entry_price
        mm = cfg.maintenance_margin_pct
        if pos.is_long:
            return ep * (1.0 - 1.0 / cfg.leverage + mm)
        else:
            return ep * (1.0 + 1.0 / cfg.leverage - mm)

    @staticmethod
    def _is_liquidated(pos: Position, candle: Candle, liq_price: float) -> bool:
        """Returns True if the candle's range touched the liquidation price."""
        if pos.is_long:
            return candle.low <= liq_price
        else:
            return candle.high >= liq_price

    @staticmethod
    def _close_position(
        pos: Position,
        entry_order: Order,
        exit_price: float,
        exit_ts: datetime,
        capital: float,
        cfg: BacktestConfig,
        liquidated: bool = False,
    ) -> tuple[float, Order]:
        """
        Close position, apply fee, return updated capital + exit order.

        On liquidation the entire margin is lost — the position's exit_value
        returns 0 to capital (margin was already deducted at open).
        """
        if liquidated:
            # Margin already deducted at open; exchange keeps everything
            return_amount = 0.0
            exit_fee = 0.0
        else:
            # Return = margin + PnL (not full notional, since only margin was deducted at open)
            # margin = notional / leverage = qty * entry / leverage
            margin = abs(pos.quantity) * pos.avg_entry_price / cfg.leverage
            if pos.is_long:
                raw_pnl = (exit_price - pos.avg_entry_price) * abs(pos.quantity)
            else:
                raw_pnl = (pos.avg_entry_price - exit_price) * abs(pos.quantity)
            exit_fee = abs(pos.quantity) * exit_price * cfg.fee_pct
            return_amount = margin + raw_pnl
        capital += return_amount - exit_fee

        exit_order = Order(
            id=str(uuid.uuid4()),
            symbol=pos.symbol,
            side=OrderSide.SELL if pos.is_long else OrderSide.BUY,
            quantity=abs(pos.quantity),
            price=exit_price,
            timestamp=exit_ts,
            status=OrderStatus.FILLED,
        )
        return capital, exit_order

    @staticmethod
    def _aggregate_news(
        news_items: list[NewsItem],
        window_hours: int,
        tradeable_coins: set[str] | None = None,
        max_titles_per_bucket: int = 15,
    ) -> list[NewsItem]:
        """
        Group news into time buckets of `window_hours` hours.
        Returns one synthetic NewsItem per non-empty (coin, bucket) pair,
        stamped at the bucket's end time, with up to `max_titles_per_bucket`
        titles concatenated for VADER analysis.

        tradeable_coins: if provided, only create buckets for these coins
                         (avoids processing coins with no price data).
        """
        if not news_items:
            return []

        window_secs = window_hours * 3600
        buckets: dict[tuple, list[NewsItem]] = {}

        for item in news_items:
            ts_epoch = item.timestamp.timestamp()
            bucket_start = int(ts_epoch // window_secs)
            coins = item.coins or ["__global__"]
            for coin in coins:
                if tradeable_coins and coin != "__global__" and coin not in tradeable_coins:
                    continue
                key = (coin, bucket_start)
                bucket = buckets.setdefault(key, [])
                if len(bucket) < max_titles_per_bucket:
                    bucket.append(item)

        result: list[NewsItem] = []
        for (coin, bucket_start), items in buckets.items():
            bucket_end_ts = datetime.fromtimestamp(
                (bucket_start + 1) * window_secs, tz=timezone.utc
            )
            combined_title = ". ".join(it.title for it in items)
            synthetic = NewsItem(
                id=f"agg_{coin}_{bucket_start}",
                timestamp=bucket_end_ts,
                title=combined_title,
                body="",
                source="aggregated",
                url="",
                coins=[coin] if coin != "__global__" else [],
            )
            result.append(synthetic)

        result.sort(key=lambda x: x.timestamp)
        return result
