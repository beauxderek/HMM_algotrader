from datetime import datetime, timedelta

# API Keys
POLYGON_API_KEY = 'kOyEETb1yEvkFGUPCQK8rswrBJPiHQEI'
ALPACA_API_KEY = 'PK8GPZ0JLLAO0TCZ7IR2'
ALPACA_SECRET_KEY = 'cgGTQmsJVZJN7dgeA2fJLenhtXqyV8TYPz5yovtH'
BASE_URL = 'https://paper-api.alpaca.markets'

# Timeframe Settings
TIMEFRAMES = {
    'default': '10min',
    'available': ['10min', 'hour', 'day'],
    'polygon_multiplier': {
        '10min': {'multiplier': 10, 'timespan': 'minute'},
        'hour': {'multiplier': 1, 'timespan': 'hour'},
        'day': {'multiplier': 1, 'timespan': 'day'}
    }
}

# Trading Securities
SECURITIES = [
    'SPY','QQQ','GOOG','JPM'
]

# Universe of potential pairs to analyze
PAIRS_UNIVERSE = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD'],
    'finance': ['JPM', 'GS', 'MS', 'BAC', 'C', 'WFC'],
    'energy': ['XOM', 'CVX', 'COP', 'EOG', 'PXD'],
    'retail': ['WMT', 'TGT', 'COST', 'AMZN'],
    'etf': ['SPY', 'QQQ', 'IWM', 'DIA']
}

# Pairs trading parameters
PAIRS_PARAMETERS = {
    'min_correlation': 0.7,
    'max_p_value': 0.05,
    'min_half_life': 1,
    'max_half_life': 30,
    'entry_zscore': 2.0,
    'exit_zscore': 0.5,
    'max_position_size': 0.2
}

# Date ranges
START_DATE = '2019-10-26'
TRAIN_END_DATE = '2023-07-01'
END_DATE = '2023-12-31'

# Strategy Settings
STRATEGY_TYPES = {
    'hmm_trend': {
        'name': 'HMM Trend Following',
        'n_components': 2,
        'min_history': 1000,
        'probability_threshold': 0.7
    },
    'pairs': {
        'name': 'Pairs Trading',
        'lookback_period': 20,
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
        'correlation_threshold': 0.7,
        'min_history': 1000
    }
}

# Live Trading Settings
LIVE_TRADING_SECURITIES = [
    'SPY','TQQQ','QQQ','UPRO','JPM','AAPL','V','DIA',
    'MSFT','GOOG','XOM','PLTR'
]

# Trading interval in hours
TRADING_INTERVAL = 1

# Database Configuration
DB_CONFIG = {
    'database': 'market_data.db',
    'polygon_tables': {
        '10min': 'polygon_10min',
        'hour': 'polygon_hourly',
        'day': 'polygon_daily'
    }
}

# Data Collection Settings
MAX_DAILY_API_CALLS = 5000
RATE_LIMIT_PAUSE = 12
MAX_RETRIES = 3
RETRY_DELAY = 5

def get_polygon_params(timeframe: str) -> dict:
    """Get Polygon API parameters for specified timeframe"""
    if timeframe not in TIMEFRAMES['available']:
        raise ValueError(f"Invalid timeframe. Must be one of {TIMEFRAMES['available']}")
    return TIMEFRAMES['polygon_multiplier'][timeframe]