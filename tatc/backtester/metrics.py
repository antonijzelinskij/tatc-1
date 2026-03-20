"""
Backtesting performance metrics.
"""
import math
from dataclasses import dataclass

from tatc.core.models import Order, OrderSide


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float          # 0..1
    total_pnl: float         # USD
    total_pnl_pct: float     # % of initial capital
    max_drawdown: float      # USD (negative)
    max_drawdown_pct: float  # % of peak equity
    sharpe_ratio: float      # annualized, risk-free=0
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float

    def __str__(self) -> str:
        return (
            f"\n{'='*40}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*40}\n"
            f"  Trades:        {self.total_trades} "
            f"(W:{self.winning_trades} L:{self.losing_trades})\n"
            f"  Win rate:      {self.win_rate*100:.1f}%\n"
            f"  Total PnL:     ${self.total_pnl:+.2f} ({self.total_pnl_pct:+.2f}%)\n"
            f"  Max drawdown:  ${self.max_drawdown:.2f} ({self.max_drawdown_pct:.2f}%)\n"
            f"  Sharpe ratio:  {self.sharpe_ratio:.3f}\n"
            f"  Avg trade:     ${self.avg_trade_pnl:+.2f}\n"
            f"  Best trade:    ${self.best_trade_pnl:+.2f}\n"
            f"  Worst trade:   ${self.worst_trade_pnl:+.2f}\n"
            f"{'='*40}"
        )


def compute_metrics(
    trades: list[tuple[Order, Order]],  # (buy_order, sell_order) pairs
    initial_capital: float,
    equity_curve: list[float],          # equity value at each step
) -> BacktestMetrics:
    """
    Compute performance metrics from completed round-trip trades.

    trades: list of (entry_order, exit_order) tuples
    equity_curve: list of portfolio equity values over time
    """
    if not trades:
        return BacktestMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, total_pnl_pct=0,
            max_drawdown=0, max_drawdown_pct=0, sharpe_ratio=0,
            avg_trade_pnl=0, best_trade_pnl=0, worst_trade_pnl=0,
        )

    pnls: list[float] = []
    for entry, exit_ in trades:
        if entry.side == OrderSide.BUY:
            pnl = (exit_.price - entry.price) * entry.quantity
        else:
            pnl = (entry.price - exit_.price) * entry.quantity
        pnls.append(pnl)

    total_pnl = sum(pnls)
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]

    # Max drawdown
    peak = equity_curve[0] if equity_curve else initial_capital
    max_dd = 0.0
    max_dd_pct = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = equity - peak
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    # Sharpe ratio (daily returns, annualized)
    sharpe = _sharpe(equity_curve)

    return BacktestMetrics(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=len(winning) / len(trades) if trades else 0,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl / initial_capital * 100,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe,
        avg_trade_pnl=total_pnl / len(trades) if trades else 0,
        best_trade_pnl=max(pnls) if pnls else 0,
        worst_trade_pnl=min(pnls) if pnls else 0,
    )


def _sharpe(equity_curve: list[float], periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from equity curve (risk-free = 0)."""
    if len(equity_curve) < 2:
        return 0.0
    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
    ]
    if not returns:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(variance)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(periods_per_year)
