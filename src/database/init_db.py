import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


async def create_tables():
    print("🔌 Conectando a TimescaleDB...")
    conn = await asyncpg.connect(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        host=os.getenv('DB_HOST'),
        port = os.getenv('DB_PORT')
    )

    try:
        # 1. Tabla de Velas (Klines)
        print("🏗️ Creando tabla 'market_kline_1m'...")
        await conn.execute('''
                           CREATE TABLE IF NOT EXISTS market_kline_1m
                           (
                               time
                               TIMESTAMPTZ
                               NOT
                               NULL,
                               symbol
                               TEXT
                               NOT
                               NULL,
                               open
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               high
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               low
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               close
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               volume
                               DOUBLE
                               PRECISION
                               NOT
                               NULL
                           );
                           ''')
        # Convertir a Hipertabla (particiones de 1 día por defecto)
        await conn.execute("SELECT create_hypertable('market_kline_1m', 'time', if_not_exists => TRUE);")

        # 2. Tabla de Registros de la IA
        print("🏗️ Creando tabla 'ai_logs'...")
        await conn.execute('''
                           CREATE TABLE IF NOT EXISTS ai_logs
                           (
                               time
                               TIMESTAMPTZ
                               NOT
                               NULL,
                               symbol
                               TEXT
                               NOT
                               NULL,
                               prediction
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               certainty_pct
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               decision
                               TEXT
                               NOT
                               NULL
                           );
                           ''')
        await conn.execute("SELECT create_hypertable('ai_logs', 'time', if_not_exists => TRUE);")

        # 3. Tabla de Ejecuciones de Trading (No necesita ser hipertabla por ahora)
        print("🏗️ Creando tabla 'trade_executions'...")
        await conn.execute('''
                           CREATE TABLE IF NOT EXISTS trade_executions
                           (
                               id
                               SERIAL
                               PRIMARY
                               KEY,
                               time
                               TIMESTAMPTZ
                               NOT
                               NULL,
                               order_id
                               TEXT,
                               symbol
                               TEXT
                               NOT
                               NULL,
                               side
                               TEXT
                               NOT
                               NULL,
                               contracts
                               DOUBLE
                               PRECISION
                               NOT
                               NULL,
                               leverage
                               INTEGER
                               NOT
                               NULL,
                               entry_price
                               DOUBLE
                               PRECISION
                               NOT
                               NULL
                           );
                           ''')

        print("✅ ¡Base de datos inicializada correctamente con arquitectura de Hipertablas!")
    except Exception as e:
        print(f"❌ Error creando tablas: {e}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(create_tables())