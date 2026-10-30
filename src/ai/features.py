import polars as pl
import numpy as np


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Toma un DataFrame con OHLCV crudo y genera features institucionales.
    """
    # 1. Asegurar que las columnas base existan
    base_cols = ["open", "high", "low", "close", "volume"]
    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    # 2. Retornos y Momentum
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("return"),
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    ])

    # 3. Medias Móviles (Tendencia)
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=7).alias("sma_7"),
        pl.col("close").rolling_mean(window_size=21).alias("sma_21"),
        pl.col("volume").rolling_mean(window_size=14).alias("vol_14")
    ])

    # 4. Distancias a las Medias (Imbalances)
    df = df.with_columns([
        (pl.col("close") / pl.col("sma_7") - 1).alias("dist_sma_7"),
        (pl.col("close") / pl.col("sma_21") - 1).alias("dist_sma_21")
    ])

    # 5. Volatilidad (Riesgo)
    df = df.with_columns([
        pl.col("log_return").rolling_std(window_size=14).alias("volatility_14")
    ])

    # 6. RSI (Relative Strength Index) - Momentum
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(window_size=14)
    avg_loss = loss.rolling_mean(window_size=14)

    # Prevenir división por cero
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    df = df.with_columns([
        rsi.alias("rsi_14")
    ])

    # 7. RSI (Relative Strength Index) - Momentum
    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(window_size=14)
    avg_loss = loss.rolling_mean(window_size=14)

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    df = df.with_columns([
        rsi.alias("rsi_14")
    ])

    return df.drop_nulls()