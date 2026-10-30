"""Asynchronous WebSocket market data ingestion for Binance USD-M Futures."""

import asyncio
import json
from typing import Callable, Coroutine, Dict, Any, Optional
import websockets

try:
    from src.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("BinanceWebSocket")


class BinanceWebSocketStream:
    """Manages resilient WebSocket connections to Binance USD-M Futures streaming endpoints."""

    BASE_WS_URL = "wss://fstream.binance.com/ws"

    def __init__(self, symbol: str = "btcusdt", timeframe: str = "5m"):
        self.symbol = symbol.lower()
        self.timeframe = timeframe
        self.stream_name = f"{self.symbol}@kline_{self.timeframe}"
        self.url = f"{self.BASE_WS_URL}/{self.stream_name}"
        self.is_running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    async def connect_and_listen(
        self,
        on_kline_close: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None,
        on_tick: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        """Stream market klines with automatic exponential backoff reconnection."""
        self.is_running = True
        logger.info(f"Connecting to Binance Futures WebSocket stream: {self.stream_name}")

        while self.is_running:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info(f"Connected successfully to {self.stream_name}")
                    self._reconnect_delay = 1.0  # Reset delay on successful connection

                    while self.is_running:
                        message = await ws.recv()
                        data = json.loads(message)

                        if "k" in data:
                            kline = data["k"]
                            is_closed = kline.get("x", False)

                            if on_tick:
                                await on_tick(kline)

                            if is_closed and on_kline_close:
                                logger.info(
                                    f"Candle closed [{self.timeframe}]: "
                                    f"Close=${float(kline['c']):,.2f} Volume={float(kline['v']):.2f}"
                                )
                                await on_kline_close(kline)

            except (websockets.ConnectionClosed, websockets.WebSocketException) as e:
                if not self.is_running:
                    break
                logger.warning(
                    f"WebSocket disconnected ({e}). Reconnecting in {self._reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

            except Exception as e:
                if not self.is_running:
                    break
                logger.error(f"Unexpected error in WebSocket loop: {e}. Retrying in 5s...")
                await asyncio.sleep(5.0)

    def stop(self) -> None:
        """Gracefully stop the WebSocket loop."""
        self.is_running = False
        logger.info("WebSocket stream shutdown requested.")
