from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import defaultdict

class BarAggregator:
    """Aggregates trade data into bars"""
    def __init__(self, interval_minutes: int = 60):
        self.interval = timedelta(minutes=interval_minutes)
        self.current_bars: Dict[str, Dict] = defaultdict(dict)
        self.last_bar_time: Dict[str, datetime] = {}
        
    def update(self, symbol: str, price: float, size: int, timestamp: datetime) -> Optional[Dict]:
        """Update bar data and return completed bar if interval is finished"""
        # Initialize or get current bar
        if symbol not in self.current_bars or self._is_new_bar(symbol, timestamp):
            self._start_new_bar(symbol, timestamp)
        
        bar = self.current_bars[symbol]
        
        # Update bar data
        if 'open' not in bar:
            bar['open'] = price
        bar['high'] = max(bar.get('high', price), price)
        bar['low'] = min(bar.get('low', price), price)
        bar['close'] = price
        bar['volume'] = bar.get('volume', 0) + size
        
        # Check if bar is complete
        if self._is_bar_complete(symbol, timestamp):
            completed_bar = self.current_bars.pop(symbol)
            self.last_bar_time[symbol] = timestamp
            return {
                'symbol': symbol,
                'timestamp': self.last_bar_time[symbol],
                **completed_bar
            }
        return None

    def _is_new_bar(self, symbol: str, timestamp: datetime) -> bool:
        return (symbol not in self.last_bar_time or 
                timestamp - self.last_bar_time[symbol] >= self.interval)

    def _start_new_bar(self, symbol: str, timestamp: datetime):
        self.current_bars[symbol] = {}
        if symbol not in self.last_bar_time:
            self.last_bar_time[symbol] = timestamp
            
    def _is_bar_complete(self, symbol: str, timestamp: datetime) -> bool:
        return timestamp - self.last_bar_time[symbol] >= self.interval
