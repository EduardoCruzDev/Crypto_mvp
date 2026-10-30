"""Institutional Risk Management module for algorithmic execution.

Provides ATR-based dynamic stop-loss/take-profit, volatility-adjusted position sizing,
exposure controls, and account-level maximum drawdown circuit breakers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    from src.utils.config import config
    from src.utils.logger import get_logger
except ImportError:
    from utils.config import config
    from utils.logger import get_logger

logger = get_logger("RiskManager")


@dataclass
class PositionSizing:
    """Position sizing parameters calculated from model alpha and risk constraints."""
    side: str
    decision: str
    strength: float
    invest_amount: float
    leverage: int
    notional: float
    contracts: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    is_allowed: bool = True
    rejection_reason: Optional[str] = None


class RiskManager:
    """Manages risk limits, position sizing, volatility-adaptive exits, and circuit breakers."""

    def __init__(
        self,
        min_alpha_threshold: float = config.risk.min_alpha_threshold,
        max_expected_return: float = config.risk.max_expected_return,
        base_capital: float = config.risk.base_capital,
        max_capital: float = config.risk.max_capital,
        min_leverage: int = config.risk.min_leverage,
        max_leverage: int = config.risk.max_leverage,
        max_account_drawdown_pct: float = config.risk.max_account_drawdown_pct,
        atr_period: int = 14,
        atr_sl_mult: float = config.risk.atr_stop_loss_mult,
        atr_tp_mult: float = config.risk.atr_take_profit_mult,
    ):
        self.min_alpha_threshold = min_alpha_threshold
        self.max_expected_return = max_expected_return
        self.base_capital = base_capital
        self.max_capital = max_capital
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.max_account_drawdown_pct = max_account_drawdown_pct
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

        # State tracking
        self.peak_equity: float = 0.0
        self.circuit_breaker_tripped: bool = False

    def update_equity(self, current_equity: float) -> bool:
        """Update peak equity and check if max drawdown circuit breaker is tripped.

        Returns True if circuit breaker was triggered, False otherwise.
        """
        if current_equity <= 0:
            return self.circuit_breaker_tripped

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown >= self.max_account_drawdown_pct:
                if not self.circuit_breaker_tripped:
                    logger.critical(
                        f"🚨 CIRCUIT BREAKER TRIPPED! Current Drawdown: {drawdown * 100:.2f}% "
                        f"(Max Allowed: {self.max_account_drawdown_pct * 100:.2f}%). Halting trading."
                    )
                self.circuit_breaker_tripped = True
                return True

        return False

    def reset_circuit_breaker(self, new_equity: float) -> None:
        """Manually reset the circuit breaker and update peak equity."""
        self.circuit_breaker_tripped = False
        self.peak_equity = new_equity
        logger.info(f"Circuit breaker manually reset. New baseline equity: ${new_equity:,.2f}")

    def compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: Optional[int] = None,
    ) -> float:
        """Compute Wilder's Average True Range (ATR) from OHLC arrays."""
        p = period or self.atr_period
        if len(close) < p + 1:
            # Fallback spread if history is insufficient
            return float(np.mean(high[-len(close):] - low[-len(close):]))

        prev_close = close[:-1]
        curr_high = high[1:]
        curr_low = low[1:]

        tr1 = curr_high - curr_low
        tr2 = np.abs(curr_high - prev_close)
        tr3 = np.abs(curr_low - prev_close)

        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = float(np.mean(true_range[-p:]))
        return atr

    def calculate_atr_brackets(
        self,
        side: str,
        entry_price: float,
        atr: float,
    ) -> Tuple[float, float]:
        """Calculate dynamic Stop Loss and Take Profit prices based on ATR.

        Returns (stop_loss_price, take_profit_price).
        """
        if side.lower() == "buy":
            stop_loss = entry_price - (atr * self.atr_sl_mult)
            take_profit = entry_price + (atr * self.atr_tp_mult)
        elif side.lower() == "sell":
            stop_loss = entry_price + (atr * self.atr_sl_mult)
            take_profit = entry_price - (atr * self.atr_tp_mult)
        else:
            raise ValueError(f"Invalid side: {side}")

        return round(stop_loss, 2), round(take_profit, 2)

    def evaluate_order(
        self,
        pred: float,
        current_price: float,
        account_balance: float = 1000.0,
        atr: Optional[float] = None,
    ) -> PositionSizing:
        """Evaluate alpha prediction and compute volatility/conviction-adjusted position sizing."""
        # 1. Check circuit breaker status
        if self.circuit_breaker_tripped:
            return PositionSizing(
                side="NONE",
                decision="HALT",
                strength=0.0,
                invest_amount=0.0,
                leverage=1,
                notional=0.0,
                contracts=0.0,
                is_allowed=False,
                rejection_reason="Circuit breaker active: trading halted.",
            )

        # 2. Check alpha signal threshold
        abs_pred = abs(pred)
        if abs_pred <= self.min_alpha_threshold:
            return PositionSizing(
                side="NONE",
                decision="HOLD",
                strength=0.0,
                invest_amount=0.0,
                leverage=1,
                notional=0.0,
                contracts=0.0,
                is_allowed=False,
                rejection_reason=f"Alpha magnitude {abs_pred:.5f} below threshold {self.min_alpha_threshold:.5f}",
            )

        # 3. Calculate conviction strength [0.0, 1.0]
        side = "buy" if pred > 0 else "sell"
        strength = min(
            max((abs_pred - self.min_alpha_threshold) / (self.max_expected_return - self.min_alpha_threshold), 0.0),
            1.0,
        )

        # 4. Scale capital and leverage linearly with conviction
        invest_amount = self.base_capital + (strength * (self.max_capital - self.base_capital))
        leverage = int(self.min_leverage + round(strength * (self.max_leverage - self.min_leverage)))
        notional = invest_amount * leverage
        contracts = round(notional / current_price, 3)

        # 5. Balance check
        required_margin = notional / leverage
        if required_margin > account_balance:
            return PositionSizing(
                side=side,
                decision=side.upper(),
                strength=strength,
                invest_amount=invest_amount,
                leverage=leverage,
                notional=notional,
                contracts=contracts,
                is_allowed=False,
                rejection_reason=f"Insufficient balance: requires ${required_margin:.2f}, available ${account_balance:.2f}",
            )

        # 6. Dynamic ATR brackets if provided
        sl_price, tp_price = None, None
        if atr is not None and atr > 0:
            sl_price, tp_price = self.calculate_atr_brackets(side, current_price, atr)

        return PositionSizing(
            side=side,
            decision=side.upper(),
            strength=strength,
            invest_amount=invest_amount,
            leverage=leverage,
            notional=notional,
            contracts=contracts,
            stop_loss=sl_price,
            take_profit=tp_price,
            is_allowed=True,
        )
