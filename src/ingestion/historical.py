import ccxt
import polars as pl
import os
import time
from datetime import datetime


def download_raw_binance_data(symbol_id="BTCUSD_PERP", timeframes=None, days=7):
    """
    Descarga directamente desde la API cruda de Binance COIN-M para obtener
    columnas adicionales de comportamiento de mercado (Taker Volume, Trades).
    """
    if timeframes is None:
        timeframes = ["1m"]  # Fase 1: Solo 1 minuto. Luego añadiremos ["5m", "30m", "1h", "1d"]

    # Conexión directa a la API de futuros COIN-M (dapi)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

    now = exchange.milliseconds()
    start_time_global = now - (days * 24 * 60 * 60 * 1000)

    for tf in timeframes:
        print(f"\n--- Iniciando Fase: {tf} para {symbol_id} ---")
        since = start_time_global
        all_raw_klines = []

        while since < now:
            try:
                print(
                    f"[{tf}] Descargando desde {datetime.fromtimestamp(since / 1000).strftime('%Y-%m-%d %H:%M:%S')}...")

                # Usamos la API implícita para obtener las 12 columnas nativas de Binance
                # dapi = Binance COIN-M Futures
                klines = exchange.dapiPublicGetKlines({
                    'symbol': symbol_id,
                    'interval': tf,
                    'startTime': since,
                    'limit': 1000
                })

                if not klines:
                    break

                all_raw_klines.extend(klines)

                # Avanzar el timestamp al tiempo de cierre de la última vela obtenida
                since = klines[-1][6] + 1
                time.sleep(0.3)  # Respetar rate limits

            except Exception as e:
                print(f"Error en la descarga: {e}")
                time.sleep(2)

        if not all_raw_klines:
            print(f"❌ No se obtuvieron datos para {tf}")
            continue

        # Estructuramos las 12 columnas que nos da Binance
        schema = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "base_asset_volume", "number_of_trades",
            "taker_buy_volume", "taker_buy_base_asset_volume", "ignore"
        ]

        df = pl.DataFrame(all_raw_klines, schema=schema, orient="row")

        # Casting de tipos: Todo llega como texto desde el JSON de Binance
        float_cols = ["open", "high", "low", "close", "volume",
                      "base_asset_volume", "taker_buy_volume", "taker_buy_base_asset_volume"]

        df = df.with_columns([
            pl.col(c).cast(pl.Float64) for c in float_cols
        ]).with_columns([
            pl.col("open_time").cast(pl.Int64),
            pl.col("number_of_trades").cast(pl.Int64)
        ])

        # Limpieza: Convertir timestamp a fecha legible y descartar columnas inútiles
        df = df.with_columns(
            pl.from_epoch("open_time", time_unit="ms").alias("datetime")
        ).drop(["close_time", "ignore"])

        # Guardado
        os.makedirs("data/historical", exist_ok=True)
        filename = f"data/historical/{symbol_id}_{tf}.parquet"
        df.write_parquet(filename)
        print(f"✅ Guardado: {len(df)} velas de {tf} en '{filename}'")


if __name__ == "__main__":
    # Fase 1: Descargamos 1 minuto.
    # Pediremos 30 días para tener un dataset robusto para empezar a probar el modelo.
    download_raw_binance_data(symbol_id="BTCUSD_PERP", timeframes=["1m"], days=30)