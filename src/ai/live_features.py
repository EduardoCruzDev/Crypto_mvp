import polars as pl
import numpy as np
import json
from pathlib import Path


def load_scalers(base_dir):
    scaler_path = base_dir / "data" / "models" / "scalers.json"
    with open(scaler_path, "r") as f:
        return json.load(f)


def compute_indicators(df: pl.DataFrame, tf_name: str) -> pl.DataFrame:
    df = df.sort("timestamp")
    df = df.with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias(f"ret_{tf_name}"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias(f"spread_{tf_name}")
    ])
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=9).alias(f"sma9_{tf_name}"),
        pl.col("close").rolling_mean(window_size=21).alias(f"sma21_{tf_name}"),
        pl.col("close").rolling_std(window_size=20).alias(f"vol_{tf_name}")
    ])
    df = df.with_columns([
        ((pl.col("close") - pl.col(f"sma9_{tf_name}")) / pl.col(f"sma9_{tf_name}")).alias(f"dist_sma9_{tf_name}"),
        ((pl.col("close") - pl.col(f"sma21_{tf_name}")) / pl.col(f"sma21_{tf_name}")).alias(f"dist_sma21_{tf_name}")
    ])

    cols_to_keep = ["timestamp", f"ret_{tf_name}", f"spread_{tf_name}", f"vol_{tf_name}", f"dist_sma9_{tf_name}",
                    f"dist_sma21_{tf_name}"]
    return df.select(cols_to_keep)


def build_live_tensor(live_data_dict, base_dir, seq_length=60):
    tf_ms_map = {
        "1m": 60 * 1000, "5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000
    }

    data_dir = base_dir / "data" / "historical"
    processed_tfs = {}

    # 1. Fusionar memoria caché (Parquet) con el pulso en vivo (Binance)
    for tf, ms_duration in tf_ms_map.items():
        # Cargar cola histórica (150 velas es suficiente para calcular SMA21 y rellenar secuencias)
        df_hist = pl.read_parquet(data_dir / f"btc_{tf}_historical.parquet").tail(1000)

        # Convertir datos vivos a Polars
        df_live = pl.DataFrame(
            live_data_dict[tf],
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row"
        )

        # Unir, eliminar duplicados y recalcular indicadores
        df_combined = pl.concat([df_hist, df_live]).unique(subset=["timestamp"], maintain_order=True)
        df_features = compute_indicators(df_combined, tf)

        # Alinear al milisegundo de evaluación
        df_features = df_features.with_columns(
            (pl.col("timestamp") + ms_duration).alias("eval_time")
        ).drop("timestamp").sort("eval_time")

        processed_tfs[tf] = df_features

    # 2. Reconstruir la Matriz Fractal base (Anclada a 5m)
    df_master = processed_tfs["5m"].select(["eval_time"])

    for tf, df_features in processed_tfs.items():
        df_master = df_master.join_asof(df_features, on="eval_time", strategy="backward")

    df_master = df_master.drop_nulls()

    # Extraer exactamente los últimos 60 períodos para la IA
    if len(df_master) < seq_length:
        raise ValueError("No hay suficientes velas en vivo combinadas con la caché para formar un tensor.")

    df_master = df_master.tail(seq_length)

    # 3. Normalización estricta (Escalado institucional)
    scalers = load_scalers(base_dir)
    feature_cols = [c for c in df_master.columns if c != "eval_time"]

    for col in feature_cols:
        mean = scalers[col]["mean"]
        std = scalers[col]["std"]
        df_master = df_master.with_columns(((pl.col(col) - mean) / std).alias(col))

    # 4. Formatear Tensor (Batch=1, Seq=60, Features=35)
    tensor_matrix = df_master.select(feature_cols).to_numpy()
    final_tensor = np.expand_dims(tensor_matrix, axis=0)

    return final_tensor