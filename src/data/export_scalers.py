import polars as pl
import json
from pathlib import Path


def main():
    print("⏳ Extrayendo memoria matemática (Scalers) de la Matriz Fractal...")
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Leemos la matriz maestra con la que entrenamos
    master_path = base_dir / "data" / "historical" / "btc_mtf_master.parquet"
    if not master_path.exists():
        print("❌ No se encontró btc_mtf_master.parquet")
        return

    df = pl.read_parquet(master_path)

    exclude_cols = ["open_time", "eval_time", "close", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    scalers = {}

    print(f"⚙️ Calculando Media y Desviación Estándar para {len(feature_cols)} dimensiones...")
    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std() if df[col].std() != 0 else 1.0

        scalers[col] = {
            "mean": float(mean_val),
            "std": float(std_val)
        }

    save_dir = base_dir / "data" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "scalers.json"

    with open(save_path, "w") as f:
        json.dump(scalers, f, indent=4)

    print(f"✅ ¡Memoria exportada exitosamente!")
    print(f"📂 Archivo guardado en: {save_path}")


if __name__ == "__main__":
    main()