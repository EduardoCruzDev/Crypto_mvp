import asyncio
import ccxt.async_support as ccxt
import polars as pl
from datetime import datetime, timezone
from pathlib import Path


async def download_timeframe(exchange, symbol, timeframe, start_ts, end_ts, save_dir):
    file_path = save_dir / f"btc_{timeframe}_historical.parquet"
    current_ts = start_ts

    if file_path.exists():
        try:
            existing_df = pl.read_parquet(file_path)
            if not existing_df.is_empty():
                last_saved_ts = existing_df.select(pl.max("timestamp")).item()
                tf_ms = exchange.parse_timeframe(timeframe) * 1000
                current_ts = last_saved_ts + tf_ms
                resume_date = datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc)
                print(f"\n🔄 [BTC/USDT - {timeframe}] Archivo previo detectado con {len(existing_df)} velas.")
                print(f"▶️ Resumiendo desde: {resume_date.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            print(f"⚠️ Error leyendo archivo de {timeframe}, empezando de cero.")
    else:
        print(f"\n📥 [BTC/USDT - {timeframe}] Iniciando descarga masiva desde cero...")

    all_ohlcv = []

    try:
        while current_ts < end_ts:
            klines = None
            for attempt in range(5):
                try:
                    klines = await exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
                    break
                except Exception as e:
                    print(f"\n⚠️ Micro-corte ({timeframe} - Intento {attempt + 1}/5). Esperando 5s...")
                    await asyncio.sleep(5)

            if klines is None or not klines:
                break

            all_ohlcv.extend(klines)
            last_ts = klines[-1][0]
            tf_ms = exchange.parse_timeframe(timeframe) * 1000
            current_ts = last_ts + tf_ms

            current_date = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            print(
                f"\r✅ [{timeframe}] Descargado hasta: {current_date.strftime('%Y-%m-%d %H:%M:%S')} | Nuevas: {len(all_ohlcv)}",
                end="")

            await asyncio.sleep(0.5)

            if last_ts >= end_ts:
                break
    except KeyboardInterrupt:
        print(f"\n🛑 Detenido por el usuario en {timeframe}. Guardando...")

    print(f"\n⚙️ Procesando y guardando datos de {timeframe}...")
    if len(all_ohlcv) > 0:
        new_df = pl.DataFrame(
            all_ohlcv,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row"
        )
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            df = pl.concat([existing_df, new_df])
        else:
            df = new_df

        df = df.unique(subset=["timestamp"], maintain_order=True)
        df = df.filter(pl.col("timestamp") <= end_ts)
        df.write_parquet(file_path)
        print(f"🎉 ¡[{timeframe}] Guardado exitoso! Total: {len(df)} velas.")
    else:
        print(f"ℹ️ No se descargaron velas nuevas para {timeframe}.")


async def main():
    symbol = "BTC/USDT"

    # La suite institucional completa (Omitimos '1m' porque ya la tienes al 100%)
    timeframes = ["5m", "15m", "1h", "4h", "1d", "1w"]

    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 10, 8, tzinfo=timezone.utc)
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    save_dir = Path(__file__).resolve().parent.parent.parent / "data" / "historical"
    save_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance({'enableRateLimit': True})

    for tf in timeframes:
        await download_timeframe(exchange, symbol, tf, start_ts, end_ts, save_dir)

    await exchange.close()
    print("\n🏆 ¡Toda la estructura fractal ha sido extraída desde Binance!")


if __name__ == "__main__":
    asyncio.run(main())