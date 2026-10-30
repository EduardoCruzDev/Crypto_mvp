import polars as pl
from pathlib import Path


def compute_indicators(df: pl.DataFrame, tf_name: str) -> pl.DataFrame:
    df = df.sort("timestamp")

    # 1. Rendimientos y Spread (Fuerza del movimiento)
    df = df.with_columns([
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias(f"ret_{tf_name}"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias(f"spread_{tf_name}")
    ])

    # 2. Medias Móviles y Volatilidad (Usamos window_size fijo para Polars moderno)
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=9).alias(f"sma9_{tf_name}"),
        pl.col("close").rolling_mean(window_size=21).alias(f"sma21_{tf_name}"),
        pl.col("close").rolling_std(window_size=20).alias(f"vol_{tf_name}")
    ])

    # 3. Distancia normalizada a las medias (Para que la IA entienda tendencias sin importar el precio de BTC)
    df = df.with_columns([
        ((pl.col("close") - pl.col(f"sma9_{tf_name}")) / pl.col(f"sma9_{tf_name}")).alias(f"dist_sma9_{tf_name}"),
        ((pl.col("close") - pl.col(f"sma21_{tf_name}")) / pl.col(f"sma21_{tf_name}")).alias(f"dist_sma21_{tf_name}")
    ])

    # Conservar solo los indicadores, desechamos open/high/low genéricos para evitar duplicados en la fusión
    cols_to_keep = ["timestamp", f"ret_{tf_name}", f"spread_{tf_name}", f"vol_{tf_name}", f"dist_sma9_{tf_name}",
                    f"dist_sma21_{tf_name}"]
    return df.select(cols_to_keep)


def main():
    print("⏳ Iniciando la Forja de la Matriz Multi-Temporal Institucional...")
    base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "historical"

    # Duración en milisegundos de cada vela
    tf_ms_map = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000
    }

    # 1. PREPARAR LA LÍNEA DE TIEMPO MAESTRA (ANCLA: 5 MINUTOS)
    df_base = pl.read_parquet(base_dir / "btc_5m_historical.parquet").sort("timestamp")

    # El objetivo (Target): La IA deberá predecir el retorno acumulado de las próximas 6 velas (30 minutos en el futuro)
    df_base = df_base.with_columns([
        (pl.col("close").shift(-6).log() - pl.col("close").log()).alias("target")
    ])

    # Convertimos el "Open Time" de 5m al "Close Time" exacto.
    # Esta será la columna 'eval_time': El milisegundo exacto donde el bot tiene que tomar la decisión.
    df_master = df_base.select(["timestamp", "close", "target"]).with_columns(
        (pl.col("timestamp") + tf_ms_map["5m"]).alias("eval_time")
    ).rename({"timestamp": "open_time"})

    # 2. ITERAR E INYECTAR TODAS LAS DIMENSIONES
    for tf, ms_duration in tf_ms_map.items():
        print(f"🔗 Extrayendo e inyectando geometría fractal: {tf}...")
        df_tf = pl.read_parquet(base_dir / f"btc_{tf}_historical.parquet").sort("timestamp")

        # Calcular los indicadores
        df_features = compute_indicators(df_tf, tf)

        # 🛡️ ESCUDO ANTI-TRAMPAS INSTITUCIONAL:
        # Desplazamos el timestamp de cada vela a su milisegundo de CIERRE.
        # Ejemplo: La vela diaria del Lunes será sellada a las 00:00 del Martes.
        # Así la IA jamás podrá usar la información del Lunes a las 14:00, previniendo el suicidio en vivo.
        df_features = df_features.with_columns(
            (pl.col("timestamp") + ms_duration).alias("eval_time")
        ).drop("timestamp").sort("eval_time")

        # Inyectar a la Matriz Maestra usando Polars ASOF (Estrategia Backward)
        # Esto engancha la última vela superior CERRADA disponible justo antes de nuestro eval_time actual.
        df_master = df_master.join_asof(df_features, on="eval_time", strategy="backward")

    # 3. LIMPIEZA FINAL
    print("🧹 Recortando bordes sin madurar (Limpieza de ventanas nulas)...")
    df_master = df_master.drop_nulls()

    save_path = base_dir / "btc_mtf_master.parquet"
    df_master.write_parquet(save_path)

    print(f"\n✅ ¡Matriz Fractal completada!")
    print(f"📂 Archivo maestro creado: btc_mtf_master.parquet")
    print(f"📊 Total de velas de 5m consolidadas: {len(df_master)}")
    print(f"🧠 Cada vela contiene el contexto simultáneo de: {', '.join(tf_ms_map.keys())}")


if __name__ == "__main__":
    main()