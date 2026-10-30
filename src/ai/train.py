import torch
import torch.nn as nn
import polars as pl
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from model import CryptoLSTM


def load_and_split_data():
    print("📂 Cargando Matriz Fractal Multi-Temporal...")
    base_dir = Path(__file__).resolve().parent.parent.parent
    file_path = base_dir / "data" / "historical" / "btc_mtf_master.parquet"

    df = pl.read_parquet(file_path)

    # Fecha de corte estricta: 1 de Julio de 2026
    split_date = datetime(2026, 7, 1, tzinfo=timezone.utc)
    split_ts = int(split_date.timestamp() * 1000)

    # Separar conjuntos
    df_train = df.filter(pl.col("open_time") < split_ts)
    df_backtest = df.filter(pl.col("open_time") >= split_ts)

    # Aislar y guardar el set de backtesting
    backtest_path = base_dir / "data" / "historical" / "btc_mtf_backtest.parquet"
    df_backtest.write_parquet(backtest_path)

    print(f"📉 Set de Entrenamiento: {len(df_train)} super-velas de 5m")
    print(f"🔮 Set de Backtesting guardado en: btc_mtf_backtest.parquet ({len(df_backtest)} super-velas de 5m)")

    return df_train


def prepare_data(df, seq_length=60):
    print("⚙️ Preparando tensores y normalizando datos...")

    # Las columnas que NO son características para la red
    exclude_cols = ["open_time", "eval_time", "close", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"📊 Normalizando {len(feature_cols)} dimensiones fractales (Z-Score)...")
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() if df[col].std() != 0 else 1.0
        df = df.with_columns(((pl.col(col) - mean) / std).alias(col))

    data_x = df.select(feature_cols).to_numpy()
    # AMPLIFICAMOS LA SEÑAL X100
    data_y = df.select(["target"]).to_numpy() * 100.0

    # seq_length de 60 en velas de 5m = ¡5 horas de contexto histórico exacto por predicción!
    print(f"🔪 Cortando secuencias de {seq_length} velas (5 horas de contexto histórico)...")
    X, y = [], []
    for i in range(len(data_x) - seq_length):
        X.append(data_x[i:i + seq_length])
        y.append(data_y[i + seq_length])

    return np.array(X), np.array(y), len(feature_cols)


def main():
    df_train = load_and_split_data()
    X, y, num_features = prepare_data(df_train)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🧠 Entrenando IA Fractal Institucional en: {device}")
    print(f"📦 Dimensiones del Tensor (Velas, Secuencia, Indicadores): {X.shape}")

    # Mantener los datos en CPU y enviar lotes a MPS (Protección de RAM)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    # ACTIVAMOS EL CAOS (SHUFFLE=TRUE) PARA FORZAR EL APRENDIZAJE
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = CryptoLSTM(input_size=num_features, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print(f"🚀 Iniciando entrenamiento profundo ({epochs} Epochs | Batch Size: 512)...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Error (Loss): {epoch_loss / len(loader):.6f}")

    save_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Guardamos este modelo con un nombre nuevo para no pisar el anterior
    model_path = save_dir / "lstm_mtf_mvp.pt"

    torch.save(model.state_dict(), model_path)
    print(f"\n✅ ¡Cerebro Fractal entrenado y guardado exitosamente!")


if __name__ == "__main__":
    main()