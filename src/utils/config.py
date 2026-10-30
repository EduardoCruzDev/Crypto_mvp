"""Centralized application configuration with validation and typed settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class BinanceConfig:
    env: str = field(default_factory=lambda: os.getenv("BINANCE_ENV", "testnet").lower())
    api_key: str = field(
        default_factory=lambda: (
            os.getenv("BINANCE_API_KEY_TEST")
            if os.getenv("BINANCE_ENV", "testnet").lower() == "testnet"
            else os.getenv("BINANCE_API_KEY", "")
        )
    )
    api_secret: str = field(
        default_factory=lambda: (
            os.getenv("BINANCE_API_SECRET_TEST")
            if os.getenv("BINANCE_ENV", "testnet").lower() == "testnet"
            else os.getenv("BINANCE_API_SECRET", "")
        )
    )
    symbol: str = "BTC/USDT"
    settle_symbol: str = "BTCUSDT"
    timeframe: str = "5m"


@dataclass(frozen=True)
class DatabaseConfig:
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgres"))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "crypto_db"))
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class TelegramConfig:
    token: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    enabled: bool = field(
        default_factory=lambda: bool(os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
    )


@dataclass(frozen=True)
class RiskConfig:
    min_alpha_threshold: float = 0.0017
    max_expected_return: float = 0.0100
    base_capital: float = 150.0
    max_capital: float = 400.0
    min_leverage: int = 1
    max_leverage: int = 4
    max_account_drawdown_pct: float = 0.05
    atr_stop_loss_mult: float = 2.0
    atr_take_profit_mult: float = 3.0


@dataclass(frozen=True)
class PathConfig:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    historical_dir: Path = BASE_DIR / "data" / "historical"
    models_dir: Path = BASE_DIR / "data" / "models"
    model_weights: Path = BASE_DIR / "data" / "models" / "lstm_mtf_mvp.pt"
    scalers_path: Path = BASE_DIR / "data" / "models" / "scalers.json"


@dataclass(frozen=True)
class AppConfig:
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Global singleton instance
config = AppConfig()
