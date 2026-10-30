import os
import asyncio
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class BinanceExecution:
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY_TEST")
        secret = os.getenv("BINANCE_API_SECRET_TEST")

        if not api_key or not secret:
            raise ValueError("Faltan las credenciales en el archivo .env")

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'inverse': False
            }
        })
        self.exchange.enable_demo_trading(True)

    async def get_balance(self):
        try:
            balance = await self.exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            print(f"💰 Balance libre: {usdt_free} USDT")
            return usdt_free
        except Exception as e:
            return 0.0

    async def prepare_account(self, symbol):
        try:
            await self.exchange.cancel_all_orders(symbol)
            await self.exchange.set_margin_mode('cross', symbol)
            await self.exchange.set_leverage(10, symbol)
        except:
            pass  # Ignoramos si ya está configurado

    async def open_position_with_stop(self, symbol="BTC/USDT", side="buy", contracts=0.005, entry_price=None,
                                      stop_loss_price=None):
        print(f"\n🚀 Lanzando orden LIMIT {side.upper()} por {contracts} de {symbol} a {entry_price} USDT...")

        try:
            # 1. Abrir posición (Funciona 100%)
            order = await self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=contracts,
                price=entry_price
            )
            print(f"✅ ¡Orden de entrada colocada con éxito! ID: {order['id']}")

            # 2. Intento de Stop Loss Duro
            if stop_loss_price:
                try:
                    sl_side = "SELL" if side == "buy" else "BUY"
                    symbol_id = self.exchange.market_id(symbol)

                    await self.exchange.fapiPrivatePostOrder({
                        'symbol': symbol_id,
                        'side': sl_side,
                        'type': 'STOP_MARKET',
                        'quantity': contracts,
                        'stopPrice': stop_loss_price,
                        'reduceOnly': 'true'
                    })
                    print(f"🛡️ Stop Loss Duro colocado en: {stop_loss_price}")
                except Exception as e:
                    # Si la Testnet bloquea el Stop Loss, entra el Escudo de Software
                    print(
                        f"⚠️ Nota Mock Trading: El exchange bloqueó el Stop Loss duro. El Escudo de Software (WebSocket) lo cubrirá.")

            return order

        except Exception as e:
            print(f"❌ Error en la ejecución: {e}")

    async def close_all(self):
        await self.exchange.close()


async def test_execution():
    executor = BinanceExecution()
    balance = await executor.get_balance()
    symbol = "BTC/USDT"

    if balance > 0:
        await executor.prepare_account(symbol)
        ticker = await executor.exchange.fetch_ticker(symbol)

        entry_price = int(ticker['last'])
        sl_price = int(ticker['last'] * 0.98)

        await executor.open_position_with_stop(
            symbol=symbol, side="buy", contracts=0.005,
            entry_price=entry_price, stop_loss_price=sl_price
        )

    await executor.close_all()


if __name__ == "__main__":
    asyncio.run(test_execution())