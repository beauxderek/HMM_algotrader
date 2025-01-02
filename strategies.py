from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import json
import os
from datetime import datetime
import logging
from numba import jit
from component_classes.data_manager import DataManager
from component_classes.hmm_manager import HMMManager
from component_classes.exceptions import ModelUpdateError, InsufficientDataError

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: dict, data_manager: DataManager):
        self.name = name
        self.parameters = parameters
        self.data_manager = data_manager
        self.current_positions: Dict[str, float] = {}
        self.last_update_time: Optional[datetime] = None

    @abstractmethod
    async def initialize(self):
        """Initialize strategy components"""
        pass

    @abstractmethod
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
        """Calculate trading signals for given data"""
        pass

    @abstractmethod
    def update_state(self, new_data: Dict[str, pd.DataFrame]) -> None:
        """Update strategy state with new data"""
        pass

class HMMTrendStrategy(Strategy):
    """HMM-based trend following strategy"""
    
    def __init__(self, parameters: dict, data_manager: DataManager):
        super().__init__("HMM Trend Following", parameters, data_manager)
        self.symbols = parameters.get('symbols', [])
        self.hmm_manager = HMMManager(self.symbols, self.data_manager)

    async def initialize(self):
        """Initialize strategy and HMM models"""
        try:
            await self.hmm_manager.initialize(self.parameters.get('timeframe', '10min'))
            logger.info(f"Initialized HMM models for {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to initialize HMM models: {str(e)}")
            raise

    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
        """Calculate trading signals based on HMM state probabilities"""
        signals = {}
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
                
            try:
                # Get latest returns
                returns = data[symbol]['returns'].values
                if len(returns) < self.hmm_manager.min_history:
                    logger.warning(f"Insufficient history for {symbol}")
                    continue

                # Update HMM probabilities with new data
                self.hmm_manager.update_probabilities(symbol, returns[-1])
                
                # Get current state probabilities
                bull_prob, bear_prob = self.hmm_manager.get_state_probabilities(symbol)
                
                # Calculate position and confidence
                position, confidence = self._calculate_position(bull_prob)
                signals[symbol] = (position, confidence)
                
            except Exception as e:
                logger.error(f"Error calculating signal for {symbol}: {str(e)}")
                continue
        
        return signals

    def _calculate_position(self, bull_prob: float) -> Tuple[float, float]:
        """Calculate position size and confidence based on probability"""
        threshold = self.parameters.get('probability_threshold', 0.7)
        
        if bull_prob > threshold:
            size = min(1.0, (bull_prob - threshold) / (1 - threshold))
            return size, bull_prob
        elif bull_prob < (1 - threshold):
            size = -min(1.0, (1 - threshold - bull_prob) / threshold)
            return size, 1 - bull_prob
        return 0.0, 0.0

    def update_state(self, new_data: Dict[str, pd.DataFrame]) -> None:
        """Update strategy state with new data"""
        for symbol, df in new_data.items():
            if symbol in self.symbols:
                try:
                    returns = df['returns'].values
                    self.hmm_manager.update_probabilities(symbol, returns[-1])
                except Exception as e:
                    logger.error(f"Error updating state for {symbol}: {str(e)}")
        self.last_update_time = datetime.now()

class PairsStrategy(Strategy):
    """Pairs trading strategy with HMM volatility filtering"""
    
    def __init__(self, parameters: dict, data_manager: DataManager):
        super().__init__("Pairs Trading", parameters, data_manager)
        self.pairs = parameters.get('pairs', [])
        # Get all unique symbols from pairs
        self.symbols = list(set([sym for pair in self.pairs for sym in pair]))
        self.hmm_manager = HMMManager(self.symbols, self.data_manager)
        
        # Initialize tracking containers
        self.spreads: Dict[Tuple[str, str], pd.Series] = {}
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}
        self.volatility_states: Dict[Tuple[str, str], int] = {}

    async def initialize(self):
        """Initialize strategy and HMM models"""
        try:
            await self.hmm_manager.initialize(self.parameters.get('timeframe', '10min'))
            logger.info(f"Initialized HMM models for {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to initialize HMM models: {str(e)}")
            raise

    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
        """Calculate trading signals for pairs"""
        signals = {}
        lookback = self.parameters.get('lookback_period', 20)
        entry_threshold = self.parameters.get('entry_zscore', 2.0)
        exit_threshold = self.parameters.get('exit_zscore', 0.5)
        
        for pair in self.pairs:
            stock1, stock2 = pair
            if stock1 not in data or stock2 not in data:
                continue
                
            try:
                # Get price and returns series
                price1 = data[stock1]['adj_close'].values
                price2 = data[stock2]['adj_close'].values
                returns1 = data[stock1]['returns'].values
                returns2 = data[stock2]['returns'].values
                
                if len(price1) < lookback:
                    continue

                # Calculate spread
                hedge_ratio = calculate_hedge_ratio(price1[-lookback:], price2[-lookback:])
                spread = price1 - hedge_ratio * price2
                zscore = calculate_zscore(spread, lookback)
                
                # Store values
                self.hedge_ratios[pair] = hedge_ratio
                self.spreads[pair] = pd.Series(spread, index=data[stock1].index)
                
                # Update HMM models with new returns
                self.hmm_manager.update_probabilities(stock1, returns1[-1])
                self.hmm_manager.update_probabilities(stock2, returns2[-1])
                
                # Get volatility states
                vol_state1 = self.hmm_manager.get_volatility_state(stock1)
                vol_state2 = self.hmm_manager.get_volatility_state(stock2)
                
                # Only trade if both stocks are in low volatility state
                if vol_state1 == 0 and vol_state2 == 0:
                    # Calculate positions
                    pos1, pos2, conf = self._calculate_pair_positions(
                        stock1, stock2,
                        zscore[-1],
                        hedge_ratio,
                        entry_threshold,
                        exit_threshold
                    )
                    
                    signals[stock1] = (pos1, conf)
                    signals[stock2] = (pos2, conf)
                else:
                    # Close positions in high volatility state
                    signals[stock1] = (0.0, 0.0)
                    signals[stock2] = (0.0, 0.0)
                
            except Exception as e:
                logger.error(f"Error calculating signals for pair {pair}: {str(e)}")
                continue
            
        return signals
    
    def _calculate_pair_positions(self, stock1: str, stock2: str, zscore: float,
                                hedge_ratio: float, entry_threshold: float,
                                exit_threshold: float) -> Tuple[float, float, float]:
        """Calculate positions for both stocks in pair"""
        if abs(zscore) > entry_threshold:
            size = min(1.0, abs(zscore) / (2 * entry_threshold))
            confidence = abs(zscore) / entry_threshold
            
            if zscore > 0:
                # Short stock1, long stock2
                return -size, size * hedge_ratio, confidence
            else:
                # Long stock1, short stock2
                return size, -size * hedge_ratio, confidence
                
        elif abs(zscore) < exit_threshold:
            return 0.0, 0.0, 0.0
        
        # Hold existing positions
        return (self.current_positions.get(stock1, 0.0),
                self.current_positions.get(stock2, 0.0),
                abs(zscore) / entry_threshold)

    def update_state(self, new_data: Dict[str, pd.DataFrame]) -> None:
        """Update strategy state with new data"""
        for pair in self.pairs:
            stock1, stock2 = pair
            if stock1 in new_data and stock2 in new_data:
                try:
                    returns1 = new_data[stock1]['returns'].values
                    returns2 = new_data[stock2]['returns'].values
                    
                    self.hmm_manager.update_probabilities(stock1, returns1[-1])
                    self.hmm_manager.update_probabilities(stock2, returns2[-1])
                    
                except Exception as e:
                    logger.error(f"Error updating state for pair {pair}: {str(e)}")
                    
        self.last_update_time = datetime.now()
        
@jit(nopython=True)
def calculate_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling z-score with Numba optimization"""
    zscore = np.zeros_like(spread)
    for i in range(window, len(spread)):
        window_slice = spread[i-window:i]
        if np.std(window_slice) != 0:
            zscore[i] = (spread[i] - np.mean(window_slice)) / np.std(window_slice)
    return zscore

@jit(nopython=True)
def calculate_hedge_ratio(price1: np.ndarray, price2: np.ndarray) -> float:
    """Calculate hedge ratio using linear regression with Numba"""
    x_mean = np.mean(price2)
    y_mean = np.mean(price1)
    numerator = np.sum((price2 - x_mean) * (price1 - y_mean))
    denominator = np.sum((price2 - x_mean) ** 2)
    return numerator / denominator if denominator != 0 else 1.0

@dataclass
class PairAnalysis:
    """Store pair analysis results"""
    symbol1: str
    symbol2: str
    correlation: float
    coint_t_stat: float
    coint_p_value: float
    half_life: float
    beta: float
    is_valid: bool

class PairsIdentifier:
    """Identifies tradeable pairs using correlation and cointegration analysis"""
    
    def __init__(self, min_correlation: float = 0.7, 
                 max_p_value: float = 0.05,
                 min_half_life: int = 1,
                 max_half_life: int = 30):
        self.min_correlation = min_correlation
        self.max_p_value = max_p_value
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.pairs_file = "valid_pairs.json"
        
    def analyze_potential_pairs(self, data: Dict[str, pd.DataFrame], 
                              potential_pairs: List[Tuple[str, str]]) -> List[PairAnalysis]:
        """Analyze potential pairs for trading suitability"""
        results = []
        
        for sym1, sym2 in potential_pairs:
            if sym1 not in data or sym2 not in data:
                continue
                
            try:
                # Get price series
                price1 = data[sym1]['adj_close']
                price2 = data[sym2]['adj_close']
                
                # Calculate correlation
                correlation = price1.corr(price2)
                
                if correlation < self.min_correlation:
                    continue
                
                # Test for cointegration
                coint_result = coint(price1, price2)
                t_stat, p_value, _ = coint_result
                
                if p_value > self.max_p_value:
                    continue
                
                # Calculate half-life
                half_life = self._calculate_half_life(price1, price2)
                
                if not (self.min_half_life <= half_life <= self.max_half_life):
                    continue
                
                # Calculate hedge ratio
                beta = calculate_hedge_ratio(
                    price1.values,
                    price2.values
                )
                
                results.append(PairAnalysis(
                    symbol1=sym1,
                    symbol2=sym2,
                    correlation=correlation,
                    coint_t_stat=t_stat,
                    coint_p_value=p_value,
                    half_life=half_life,
                    beta=beta,
                    is_valid=True
                ))
                
            except Exception as e:
                logger.error(f"Error analyzing pair {sym1}-{sym2}: {str(e)}")
                continue
        
        return results
    
    def _calculate_half_life(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate mean reversion half-life using OLS"""
        # Calculate spread
        spread = price1 - price2
        
        # Calculate lag-1 spread
        spread_lag = spread.shift(1)
        delta_spread = spread - spread_lag
        
        # Remove NaN values
        spread_lag = spread_lag.dropna()
        delta_spread = delta_spread.dropna()
        
        # Perform OLS regression
        beta = np.polyfit(spread_lag, delta_spread, 1)[0]
        
        # Calculate half-life
        if beta >= 0:
            return float('inf')
        return -np.log(2) / beta
    
    def save_valid_pairs(self, pairs: List[PairAnalysis]):
        """Save valid pairs to JSON file"""
        pairs_data = {
            'last_update': datetime.now().isoformat(),
            'pairs': [
                {
                    'symbol1': p.symbol1,
                    'symbol2': p.symbol2,
                    'correlation': p.correlation,
                    'half_life': p.half_life,
                    'beta': p.beta
                }
                for p in pairs
            ]
        }
        
        with open(self.pairs_file, 'w') as f:
            json.dump(pairs_data, f, indent=4)
            
    def load_valid_pairs(self) -> List[Tuple[str, str]]:
        """Load valid pairs from JSON file"""
        if not os.path.exists(self.pairs_file):
            return []
            
        with open(self.pairs_file, 'r') as f:
            data = json.load(f)
            return [(p['symbol1'], p['symbol2']) for p in data['pairs']]

def create_strategy(strategy_type: str, parameters: dict, data_manager: DataManager) -> Strategy:
    """Factory function to create strategy instances"""
    strategies = {
        'hmm_trend': HMMTrendStrategy,
        'pairs': PairsStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
        
    return strategies[strategy_type](parameters, data_manager)