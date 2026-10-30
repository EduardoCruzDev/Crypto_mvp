"""Institutional Backtesting Engine for Multi-Timeframe Fractal Model.

Simulates exchange execution with taker fees, dynamic leverage, trailing/bracket stops,
and computes institutional performance metrics (Sharpe, Sortino, Calmar, Max Drawdown).
"""

from pathlib import Path
import sys
import numpy as np
import polars as pl
import torch

# Ensure src modules are resolvable
sys.path.append(str(Path(__file__).resolve().parent))

from ai.model import CryptoLSTM
from utils.logger import get_logger

logger = get_logger("BacktestEngine")


def compute_quant_metrics(
    equity_curve: list[float],
    trade_pnls: list[float],
    initial_capital: float = 1000.0,
    candles_per_year: int = 105120,  # 5-minute bars per year (365 * 24 * 12)
) -> dict:
    """Compute institutional quantitative risk and return metrics."""
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])

    total_return = (equity[-1] - initial_capital) / initial_capital
    num_candles = len(equity)
    years = max(num_candles / candles_per_year, 0.001)

    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if (1.0 + total_return) > 0 else -1.0

    # Running maximum and drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe Ratio (annualized, Rf = 0)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = float((mean_ret / (std_ret + 1e-9)) * np.sqrt(candles_per_year))

    # Sortino Ratio (annualized, downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
    sortino = float((mean_ret / (downside_std + 1e-9)) * np.sqrt(candles_per_year))

    # Calmar Ratio
    calmar = float(cagr / (max_drawdown + 1e-9)) if max_drawdown > 0 else 0.0

    # Trade-level statistics
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    total_trades = len(trade_pnls)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / (gross_loss + 1e-9)) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    avg_trade = float(np.mean(trade_pnls)) if trade_pnls else 0.0

    return {
        "initial_capital": initial_capital,
        "final_capital": equity[-1],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "wins": len(wins),
        "losses": len(losses),
        "avg_trade_pnl": avg_trade,
    }


def main():
    logger.info("Initializing Institutional Fractal Backtesting Engine...")
    base_dir = Path(__file__).resolve().parent.parent

    data_path = base_dir / "data" / "historical" / "btc_mtf_backtest.parquet"
    model_path = base_dir / "data" / "models" / "lstm_mtf_mvp.pt"

    if not data_path.exists():
        logger.error(f"Fractal backtest dataset not found at: {data_path}")
        return

    df = pl.read_parquet(data_path)

    exclude_cols = ["open_time", "eval_time", "close", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    logger.info(f"Loaded {len(df):,} test bars with {len(feature_cols)} feature channels.")

    # Z-Score normalization per feature channel
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() if df[col].std() != 0 else 1.0
        df = df.with_columns(((pl.col(col) - mean) / std).alias(col))

    data_x = df.select(feature_cols).to_numpy()
    prices_close = df["close"].to_numpy()

    seq_length = 60
    X = [data_x[i : i + seq_length] for i in range(len(data_x) - seq_length)]
    X = np.array(X)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Inference device: {device.type.upper()}")

    model = CryptoLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    logger.info("Executing batched inference on Deep Learning Model...")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    batch_size = 1024
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            preds = model(batch).cpu().numpy().flatten()
            predictions.extend(preds)

        predictions = np.array(predictions) / 100.0

    # Backtest execution simulation parameters
    taker_fee = 0.0005        # 0.05% Binance taker commission
    min_threshold = 0.0017    # Alpha conviction threshold
    max_expected = 0.0100     # Return ceiling for max leverage
    stop_loss_pct = 0.02      # 2% protective stop loss

    capital = 1000.0
    equity_curve = [capital]
    trade_pnls = []

    in_position = False
    entry_price = 0.0
    position_side = None
    contracts = 0.0

    offset = seq_length

    for i in range(len(predictions)):
        current_price = prices_close[i + offset]
        pred = predictions[i]

        if in_position:
            shield_triggered = False
            close_price = current_price

            # Stop loss evaluation
            if position_side == "BUY" and current_price <= entry_price * (1 - stop_loss_pct):
                shield_triggered = True
            elif position_side == "SELL" and current_price >= entry_price * (1 + stop_loss_pct):
                shield_triggered = True

            # Directional alpha reversal
            decision_reversal = (
                (position_side == "BUY" and pred < -min_threshold)
                or (position_side == "SELL" and pred > min_threshold)
            )

            if shield_triggered or decision_reversal:
                gross_pnl = (
                    (close_price - entry_price) * contracts
                    if position_side == "BUY"
                    else (entry_price - close_price) * contracts
                )
                fee_cost = close_price * contracts * taker_fee
                net_pnl = gross_pnl - fee_cost

                capital += net_pnl
                trade_pnls.append(net_pnl)
                in_position = False

        # Open new position if alpha threshold is satisfied
        if not in_position and abs(pred) > min_threshold:
            strength = min(max((abs(pred) - min_threshold) / (max_expected - min_threshold), 0.0), 1.0)
            invest_amount = 150.0 + (strength * (400.0 - 150.0))
            leverage = int(1 + (strength * (4 - 1)))

            position_side = "BUY" if pred > 0 else "SELL"
            entry_price = current_price
            notional = invest_amount * leverage
            contracts = notional / entry_price

            entry_fee = notional * taker_fee
            capital -= entry_fee
            in_position = True

        equity_curve.append(capital)

    # Force close open position at final bar
    if in_position:
        close_price = prices_close[-1]
        gross_pnl = (
            (close_price - entry_price) * contracts
            if position_side == "BUY"
            else (entry_price - close_price) * contracts
        )
        final_net = gross_pnl - (close_price * contracts * taker_fee)
        capital += final_net
        trade_pnls.append(final_net)
        equity_curve.append(capital)

    metrics = compute_quant_metrics(equity_curve, trade_pnls, initial_capital=1000.0)

    print("\n" + "=" * 60)
    print("📈 FRACTAL QUANT INSTITUTIONAL BACKTEST REPORT")
    print("=" * 60)
    print(f"💰 Initial Capital:        ${metrics['initial_capital']:,.2f}")
    print(f"💵 Final Capital:          ${metrics['final_capital']:,.2f}")
    print(f"📊 Net Return (ROI):       {metrics['total_return_pct']:+.2f}%")
    print(f"📅 Annualized Return:      {metrics['cagr_pct']:+.2f}%")
    print("-" * 60)
    print(f"🛡️ Max Drawdown (MDD):     {metrics['max_drawdown_pct']:.2f}%")
    print(f"⚖️ Sharpe Ratio (Rf=0):    {metrics['sharpe_ratio']:.2f}")
    print(f"📉 Sortino Ratio:          {metrics['sortino_ratio']:.2f}")
    print(f"🎯 Calmar Ratio:           {metrics['calmar_ratio']:.2f}")
    print(f"💎 Profit Factor:          {metrics['profit_factor']:.2f}")
    print("-" * 60)
    print(f"🤝 Total Trades:           {metrics['total_trades']}")
    print(f"✅ Wins / ❌ Losses:       {metrics['wins']} / {metrics['losses']}")
    print(f"🎯 Win Rate:               {metrics['win_rate_pct']:.1f}%")
    print(f"🪙 Avg Trade PnL:          ${metrics['avg_trade_pnl']:+.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
