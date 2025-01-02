class TradingError(Exception):
    """Base class for trading-related exceptions"""
    pass

class InsufficientDataError(TradingError):
    """Raised when there isn't enough data to make trading decisions"""
    pass

class ModelUpdateError(TradingError):
    """Raised when HMM model updates fail"""
    pass

class OrderExecutionError(TradingError):
    """Raised when order execution fails"""
    pass

class MarketDataError(TradingError):
    """Raised when there are issues with market data"""
    pass

class DatabaseConnectionError(TradingError):
    """Raised when issues occur with database connection"""
    pass

class WebSocketError(TradingError):
    """Raised when websocket connection encounters error"""
    pass
