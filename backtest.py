import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import pickle
import logging
from component_classes.hmm_manager import compute_forward_probabilities_numba
from component_classes.data_manager import DataManager
from component_classes.risk_manager import RiskManager
from component_classes.exceptions import InsufficientDataError
from config import TIMEFRAMES, DB_CONFIG
import report

logger = logging.getLogger(__name__)

class Backtest(ABC):
    """Abstract base class for backtesting strategies"""
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000,
                 timeframe: str = '10min'):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.data_manager = DataManager(DB_CONFIG, None)  # API key not needed for backtest
        self.risk_manager = RiskManager()
        
        # Track performance
        self.portfolio_value: List[float] = [initial_capital]
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
        
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Run backtest and return results"""
        pass
    
    @abstractmethod
    async def calculate_signals(self, data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> Dict[str, Tuple[float, float]]:
        """Calculate trading signals for given timestamp"""
        pass

    async def execute_trades(self, signals: Dict[str, Tuple[float, float]], 
                           prices: Dict[str, float],
                           timestamp: pd.Timestamp) -> None:
        """Execute trades based on signals"""
        # Get current portfolio value
        current_portfolio = self.portfolio_value[-1]
        
        # Validate allocations through risk manager
        allocations = self.risk_manager.validate_allocation(
            {sym: (size, conf) for sym, (size, conf) in signals.items()},
            current_portfolio,
            self.positions
        )
        
        for symbol, (target_alloc, _) in allocations.items():
            current_position = self.positions.get(symbol, 0)
            current_value = current_position * prices[symbol]
            target_value = current_portfolio * target_alloc
            
            if abs(target_value - current_value) > 100:  # Minimum trade size
                # Calculate trade size
                trade_value = target_value - current_value
                trade_shares = trade_value / prices[symbol]
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'shares': trade_shares,
                    'price': prices[symbol],
                    'value': trade_value,
                    'portfolio_value': current_portfolio
                })
                
                # Update position
                self.positions[symbol] = current_position + trade_shares
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        positions_value = sum(self.positions.get(sym, 0) * price 
                            for sym, price in prices.items())
        return positions_value
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        metrics = {
            'total_return': (self.portfolio_value[-1] / self.initial_capital) - 1,
            'annualized_return': self._calculate_annualized_return(returns),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_days = (pd.to_datetime(self.end_date) - 
                     pd.to_datetime(self.start_date)).days
        years = total_days / 365
        return (1 + returns.mean()) ** (252 * years) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        portfolio_series = pd.Series(self.portfolio_value)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = portfolio_series / rolling_max - 1
        return drawdowns.min()
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate of trades"""
        if not self.trades:
            return 0
        profitable_trades = sum(1 for trade in self.trades 
                              if trade['value'] > 0)
        return profitable_trades / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        gross_profit = sum(trade['value'] for trade in self.trades 
                          if trade['value'] > 0)
        gross_loss = abs(sum(trade['value'] for trade in self.trades 
                            if trade['value'] < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    @abstractmethod
    def generate_report(self, filename: str) -> None:
        """Generate backtest report"""
        pass
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate input data"""
        if not data:
            logger.error("No data provided for backtest")
            return False
            
        # Check for empty dataframes
        empty_symbols = [sym for sym, df in data.items() if df.empty]
        if empty_symbols:
            logger.error(f"Empty data for symbols: {empty_symbols}")
            return False
            
        # Check for required columns
        required_columns = {'open', 'high', 'low', 'close', 'adj_close', 'returns'}
        for sym, df in data.items():
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                logger.error(f"Missing columns for {sym}: {missing_cols}")
                return False
        
        return True

class TrendFollowingBacktest(Backtest):
    """HMM-based trend following strategy backtest"""
    
    def __init__(self, 
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000,
                 timeframe: str = '10min',
                 probability_threshold: float = 0.7):
        super().__init__(start_date, end_date, initial_capital, timeframe)
        self.symbols = symbols
        
        # Create strategy instance
        strategy_params = {
            'symbols': symbols,
            'probability_threshold': probability_threshold,
            'timeframe': timeframe
        }
        self.strategy = HMMTrendStrategy(strategy_params, self.data_manager)
        self.signals_history = {}
        
    async def calculate_signals(self, data: Dict[str, pd.DataFrame], 
                              timestamp: pd.Timestamp) -> Dict[str, Tuple[float, float]]:
        """Calculate trading signals using strategy instance"""
        # Get data up to current timestamp
        current_data = {sym: df.loc[:timestamp] for sym, df in data.items()}
        
        # Get signals from strategy
        signals = self.strategy.calculate_signals(current_data)
        
        # Store signals for reporting
        for symbol, (position, confidence) in signals.items():
            self.signals_history.setdefault(symbol, []).append({
                'timestamp': timestamp,
                'position': position,
                'confidence': confidence,
                'bull_prob': self.strategy.hmm_manager.get_state_probabilities(symbol)[0]
            })
        
        return signals
    
    async def run(self) -> Dict[str, Any]:
        """Run trend following backtest"""
        logger.info("Starting trend following backtest...")
        
        # Initialize strategy
        await self.strategy.initialize()
        
        # Fetch data for all symbols
        data = {}
        for symbol in self.symbols:
            df = await self.data_manager.get_market_data(
                symbol, self.start_date, self.end_date, self.timeframe
            )
            if not df.empty:
                data[symbol] = df
        
        if not self.validate_data(data):
            raise ValueError("Invalid data for backtest")
        
        # Get common timestamps across all symbols
        common_dates = sorted(set.intersection(*[set(df.index) for df in data.values()]))
        
        # Run backtest
        for timestamp in common_dates:
            # Get current prices
            prices = {sym: data[sym].loc[timestamp, 'adj_close'] 
                     for sym in self.symbols}
            
            # Calculate signals
            signals = await self.calculate_signals(data, timestamp)
            
            # Execute trades
            await self.execute_trades(signals, prices, timestamp)
            
            # Update portfolio value
            current_value = self.calculate_portfolio_value(prices)
            self.portfolio_value.append(current_value)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        return {
            'metrics': metrics,
            'portfolio_value': self.portfolio_value,
            'trades': self.trades,
            'signals_history': self.signals_history,
            'bull_probabilities': {sym: [s['bull_prob'] for s in signals] 
                                 for sym, signals in self.signals_history.items()}
        }
    
    def generate_report(self, filename: str) -> None:
        """Generate trend following backtest report"""
        report_data = {
            'strategy_type': 'trend_following',
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio_value[-1],
            'metrics': self.performance_metrics,
            'portfolio_values': self.portfolio_value,
            'trades': self.trades,
            'signals_history': self.signals_history
        }
        
        report.generate_trend_following_report(report_data, filename)

class PairsBacktest(Backtest):
    """Pairs trading strategy backtest"""
    
    def __init__(self,
                 pairs: List[Tuple[str, str]],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000,
                 timeframe: str = '10min',
                 lookback_period: int = 20,
                 entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5):
        super().__init__(start_date, end_date, initial_capital, timeframe)
        self.pairs = pairs
        
        # Create strategy instance
        strategy_params = {
            'pairs': pairs,
            'lookback_period': lookback_period,
            'entry_zscore': entry_zscore,
            'exit_zscore': exit_zscore,
            'timeframe': timeframe
        }
        self.strategy = PairsStrategy(strategy_params, self.data_manager)
        
        # Track pair-specific data
        self.signals_history = {}
        self.pair_returns = {}
        self.pair_metrics = {}
        
    async def calculate_signals(self, data: Dict[str, pd.DataFrame],
                              timestamp: pd.Timestamp) -> Dict[str, Tuple[float, float]]:
        """Calculate trading signals using strategy instance"""
        # Get data up to current timestamp
        current_data = {sym: df.loc[:timestamp] for sym, df in data.items()}
        
        # Get signals from strategy
        signals = self.strategy.calculate_signals(current_data)
        
        # Store signals for reporting
        for pair in self.pairs:
            stock1, stock2 = pair
            if stock1 in signals and stock2 in signals:
                self.signals_history.setdefault(pair, []).append({
                    'timestamp': timestamp,
                    'position1': signals[stock1][0],
                    'position2': signals[stock2][0],
                    'confidence': signals[stock1][1],
                    'zscore': self.strategy.get_current_zscore(pair),
                    'hedge_ratio': self.strategy.hedge_ratios.get(pair, 1.0)
                })
        
        return signals
    
    async def run(self) -> Dict[str, Any]:
        """Run pairs trading backtest"""
        logger.info("Starting pairs trading backtest...")
        
        # Initialize strategy
        await self.strategy.initialize()
        
        # Get all unique symbols from pairs
        symbols = list(set([sym for pair in self.pairs for sym in pair]))
        
        # Fetch data
        data = {}
        for symbol in symbols:
            df = await self.data_manager.get_market_data(
                symbol, self.start_date, self.end_date, self.timeframe
            )
            if not df.empty:
                data[symbol] = df
        
        if not self.validate_data(data):
            raise ValueError("Invalid data for backtest")
        
        # Get common timestamps
        common_dates = sorted(set.intersection(*[set(df.index) for df in data.values()]))
        
        # Run backtest
        for timestamp in common_dates:
            # Get current prices
            prices = {sym: data[sym].loc[timestamp, 'adj_close'] 
                     for sym in symbols}
            
            # Calculate signals
            signals = await self.calculate_signals(data, timestamp)
            
            # Execute trades
            await self.execute_trades(signals, prices, timestamp)
            
            # Update portfolio value
            current_value = self.calculate_portfolio_value(prices)
            self.portfolio_value.append(current_value)
            
            # Calculate pair-specific returns
            for pair in self.pairs:
                pair_return = self._calculate_pair_return(pair, prices)
                self.pair_returns.setdefault(pair, []).append(pair_return)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        # Calculate pair-specific metrics
        for pair in self.pairs:
            self.pair_metrics[pair] = self._calculate_pair_metrics(pair)
        
        return {
            'metrics': metrics,
            'pair_metrics': self.pair_metrics,
            'portfolio_value': self.portfolio_value,
            'trades': self.trades,
            'signals_history': self.signals_history,
            'pair_returns': self.pair_returns
        }
    
    def _calculate_pair_return(self, pair: Tuple[str, str], 
                             prices: Dict[str, float]) -> float:
        """Calculate return for a specific pair"""
        stock1, stock2 = pair
        pos1 = self.positions.get(stock1, 0)
        pos2 = self.positions.get(stock2, 0)
        return (pos1 * prices[stock1] + pos2 * prices[stock2])
    
    def _calculate_pair_metrics(self, pair: Tuple[str, str]) -> Dict[str, float]:
        """Calculate performance metrics for a specific pair"""
        returns = pd.Series(self.pair_returns[pair]).pct_change().dropna()
        
        return {
            'total_return': (returns + 1).prod() - 1,
            'annualized_return': self._calculate_annualized_return(returns),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown()
        }
    
    def generate_report(self, filename: str) -> None:
        """Generate pairs trading backtest report"""
        report_data = {
            'strategy_type': 'pairs_trading',
            'pairs': self.pairs,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio_value[-1],
            'metrics': self.performance_metrics,
            'pair_metrics': self.pair_metrics,
            'portfolio_values': self.portfolio_value,
            'trades': self.trades,
            'signals_history': self.signals_history,
            'pair_returns': self.pair_returns
        }
        
        report.generate_pairs_trading_report(report_data, filename)
        
def create_backtest(strategy_type: str, **kwargs) -> Backtest:
    """
    Factory function to create appropriate backtest instance
    
    Args:
        strategy_type: Type of strategy ('trend_following' or 'pairs_trading')
        **kwargs: Strategy-specific parameters
    
    Returns:
        Appropriate backtest instance
    """
    if strategy_type == 'trend_following':
        return TrendFollowingBacktest(
            symbols=kwargs.get('symbols', []),
            start_date=kwargs['start_date'],
            end_date=kwargs['end_date'],
            initial_capital=kwargs.get('initial_capital', 1000000),
            timeframe=kwargs.get('timeframe', '10min'),
            probability_threshold=kwargs.get('probability_threshold', 0.7)
        )
    elif strategy_type == 'pairs_trading':
        return PairsBacktest(
            pairs=kwargs.get('pairs', []),
            start_date=kwargs['start_date'],
            end_date=kwargs['end_date'],
            initial_capital=kwargs.get('initial_capital', 1000000),
            timeframe=kwargs.get('timeframe', '10min'),
            lookback_period=kwargs.get('lookback_period', 20),
            entry_zscore=kwargs.get('entry_zscore', 2.0),
            exit_zscore=kwargs.get('exit_zscore', 0.5)
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")