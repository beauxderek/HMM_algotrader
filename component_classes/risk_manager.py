import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_position_size: float = 0.25, max_leverage: float = 3.0, max_portfolio_risk: float = 0.5):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_portfolio_risk = max_portfolio_risk

    def validate_allocation(self, allocation: Dict[str, Tuple[float, float]], portfolio_value: float, 
                          current_positions: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Validate and adjust allocations based on risk parameters"""
        adjusted_allocation = allocation.copy()
        total_leverage = sum(abs(lev) for _, (_, lev) in allocation.items())
        
        if total_leverage > self.max_leverage:
            scale_factor = self.max_leverage / total_leverage
            adjusted_allocation = {
                symbol: (alloc, lev * scale_factor)
                for symbol, (alloc, lev) in allocation.items()
            }
        
        total_allocation = sum(alloc for alloc, _ in adjusted_allocation.values())
        if total_allocation > self.max_portfolio_risk:
            scale = self.max_portfolio_risk / total_allocation
            adjusted_allocation = {
                sym: (alloc * scale, lev)
                for sym, (alloc, lev) in adjusted_allocation.items()
            }
        
        return adjusted_allocation