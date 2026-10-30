"""Main live asynchronous execution loop for FractalQuant.

Synchronizes with Binance 5-minute candle closures, constructs multi-timeframe tensors,
runs PyTorch LSTM inference, validates risk with RiskManager, and routes orders via CCXT.
"""

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
import ccxt.async_support as ccxt
import numpy as np
import polars as pl
import torch

# Ensure local modules resolve cleanly
sys.path.append(str(Path(__file__).resolve().parent))

from ai.live_features import build_live_tensor
from ai.model import CryptoLSTM
from database.db_manager import log_ai_prediction, log_kline, log_trade_execution
from execution.risk import RiskManager
from utils.config import config
from utils.logger import get_logger
from utils.telegram import send_telegram_alert

logger = get_logger("LiveExecutionEngine")


async def wait_for_next_candle(interval_seconds: int = 300) -> None:
    """Synchronize engine clock to the exact close of Binance 5-minute candles."""
    now = int(time.time())
    seconds_until_next = interval_seconds - (now % interval_seconds)
    next_time = datetime.fromtimestamp(now + seconds_until_next, tz=timezone.utc)
    logger.info(
        f"Awaiting candle close: {seconds_until_next}s remaining "
        f"(Target Sync: {next_time.strftime('%H:%M:%S')} UTC)"
    )
    await asyncio.sleep(seconds_until_next)
    await asyncio.sleep(2)  # Latency safety buffer for exchange candle finalization


async def fetch_live_data(exchange: ccxt.binance, symbol: str = config.binance.symbol) -> dict:
    """Concurrently download OHLCV candles for all 7 fractal timeframes."""
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    live_data = {}
    for tf in timeframes:
        klines = await exchange.fetch_ohlcv(symbol, tf, limit=1000)
        live_data[tf] = klines
    return live_data


async def execute_trade(
    exchange: ccxt.binance,
    side: str,
    amount: float,
    leverage: int,
    price: float,
    symbol: str = config.binance.symbol,
) -> dict:
    """Execute market order and set leverage on Binance Futures."""
    try:
        await exchange.fapiPrivate_post_leverage({
            "symbol": config.binance.settle_symbol,
            "leverage": leverage,
        })
    except Exception as e:
        logger.debug(f"Leverage adjustment notice: {e}")

    order = await exchange.create_order(
        symbol=symbol,
        type="market",
        side=side,
        amount=amount,
    )
    return order


async def main() -> None:
    logger.info("⚡ Starting FractalQuant Live Execution Engine...")
    base_dir = config.paths.base_dir
    model_path = config.paths.model_weights

    # Initialize exchange connection
    exchange = ccxt.binance({
        "apiKey": config.binance.api_key,
        "secret": config.binance.api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    if config.binance.env == "testnet":
        exchange.enable_demo_trading(True)
        logger.info("Connected to BINANCE DEMO TRADING (USD-M Paper Trading)")
    else:
        logger.warning("🚨 CONNECTED TO BINANCE MAINNET (Real Capital Live Trading)")

    # Initialize Deep Learning Model & Device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CryptoLSTM(input_size=35, hidden_size=64, num_layers=2).to(device)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded PyTorch LSTM model weights into {device.type.upper()}")
    else:
        logger.error(f"Model weights not found at: {model_path}")
        return

    model.eval()
    risk_manager = RiskManager()

    try:
        while True:
            await wait_for_next_candle()
            logger.info("Fetching real-time multi-horizon market snapshot...")
            live_data = await fetch_live_data(exchange)

            # Persist 1m candle to TimescaleDB for telemetry
            try:
                last_1m = live_data["1m"][-1]
                await log_kline(
                    last_1m[0],
                    config.binance.settle_symbol,
                    last_1m[1],
                    last_1m[2],
                    last_1m[3],
                    last_1m[4],
                    last_1m[5],
                )
            except Exception as e:
                logger.error(f"Error logging 1m kline: {e}")

            # Construct 35-dimensional multi-timeframe tensor
            try:
                X_live = build_live_tensor(live_data, base_dir)
            except Exception as e:
                logger.warning(f"Error building live tensor; skipping iteration: {e}")
                continue

            X_tensor = torch.tensor(X_live, dtype=torch.float32).to(device)

            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().flatten()[0] / 100.0

            current_price = float(live_data["5m"][-1][4])
            logger.info(
                f"Prediction (30m Expected Return): {pred * 100:+.4f}% | Current Price: ${current_price:,.2f}"
            )

            # Extract 5m OHLC for ATR computation
            klines_5m = np.array(live_data["5m"])
            highs_5m = klines_5m[:, 2].astype(float)
            lows_5m = klines_5m[:, 3].astype(float)
            closes_5m = klines_5m[:, 4].astype(float)
            atr_5m = risk_manager.compute_atr(highs_5m, lows_5m, closes_5m)

            # Evaluate order sizing and risk constraints
            sizing = risk_manager.evaluate_order(
                pred=pred,
                current_price=current_price,
                account_balance=1000.0,
                atr=atr_5m,
            )

            decision = sizing.decision

            if sizing.is_allowed:
                logger.info(
                    f"🎯 ALPHA SIGNAL APPROVED: {sizing.side.upper()} {sizing.contracts} contracts "
                    f"(Notional: ${sizing.notional:.2f}, Leverage: x{sizing.leverage}) | "
                    f"Stop: ${sizing.stop_loss:,.2f} | TP: ${sizing.take_profit:,.2f}"
                )
                try:
                    order = await execute_trade(
                        exchange=exchange,
                        side=sizing.side,
                        amount=sizing.contracts,
                        leverage=sizing.leverage,
                        price=current_price,
                    )
                    order_id = order.get("id", "N/A")
                    await log_trade_execution(
                        timestamp=int(time.time()),
                        order_id=order_id,
                        symbol=config.binance.settle_symbol,
                        side=sizing.side.upper(),
                        contracts=sizing.contracts,
                        leverage=sizing.leverage,
                        entry_price=current_price,
                    )
                    alert_msg = (
                        f"🚨 *NEW EXECUTION ({sizing.side.upper()})*\n"
                        f"💰 Asset: BTC/USDT\n"
                        f"💵 Price: ${current_price:,.2f}\n"
                        f"🎯 Prediction: {pred * 100:+.2f}%\n"
                        f"⚡ Capital: ${sizing.invest_amount:.2f} (x{sizing.leverage})\n"
                        f"🛡️ Stop Loss: ${sizing.stop_loss:,.2f} | TP: ${sizing.take_profit:,.2f}"
                    )
                    await send_telegram_alert(alert_msg)
                    logger.info(f"Order {order_id} filled; alert broadcasted.")
                except Exception as e:
                    logger.error(f"Order routing failed: {e}")
            else:
                logger.info(f"Signal filtered: {sizing.rejection_reason}")

            # Persist model inference telemetry to TimescaleDB
            await log_ai_prediction(
                timestamp=int(time.time()),
                symbol=config.binance.settle_symbol,
                prediction=pred,
                certainty_pct=abs(pred) * 100,
                decision=decision,
            )

    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Terminating live loop...")
    finally:
        await exchange.close()
        from database.db_manager import db
        await db.close()
        logger.info("Exchange and database connections closed gracefully.")


if __name__ == "__main__":
    asyncio.run(main())
