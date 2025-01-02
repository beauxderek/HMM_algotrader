import asyncio
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
from component_classes.display_manager import DisplayManager
from component_classes.data_manager import DataManager
from component_classes.hmm_manager import HMMManager
from component_classes.risk_manager import RiskManager
from component_classes.exceptions import OrderExecutionError
from component_classes.websocket_manager import WebSocketManager
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.client import TradingClient
import json

import logging

from backtest import calculate_portfolio_allocation
from config import (
    LIVE_TRADING_SECURITIES, DB_CONFIG, 
    ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL,
    TRADING_INTERVAL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
class LiveTrader:
    def __init__(self, symbols: List[str], db_config: dict, alpaca_config: dict, 
                timeframe: str = 'hourly', trading_interval: int = 1, ignore_market_hours: bool = False):
        self.symbols = symbols
        self.timeframe = timeframe
        self.trading_interval = trading_interval
        self.display_manager = DisplayManager()
        self.ignore_market_hours = ignore_market_hours
        self.is_extended_hours_session = False
        
        # Initialize API clients
        self.trading_client = TradingClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key'],
            paper=True if 'paper' in alpaca_config['base_url'] else False
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key']
        )

        # Initialize components
        self.data_manager = DataManager(
            db_config['database'],
            source=db_config.get('source', 'yfinance'),
            timeframe=db_config.get('timeframe', 'hourly')
        )
        self.websocket_manager = WebSocketManager(
            alpaca_api_key=alpaca_config['api_key'],
            alpaca_secret_key=alpaca_config['secret_key'],
            symbols=symbols,
            data_manager=self.data_manager,
        )
        self.hmm_manager = HMMManager(symbols, self.data_manager)
        self.risk_manager = RiskManager()
        
        self.is_running = False
        self.last_trading_time = None

    async def initialize(self):
        """Initialize the trading system"""
        try:
            # Initialize the display manager with symbols first
            await self.display_manager.initialize_symbols(self.symbols)
            
            # Initialize empty portfolio
            initial_portfolio = {symbol: 0.0 for symbol in self.symbols}
            await self.display_manager.update_portfolio_allocations(initial_portfolio)
            
            # Initialize other components
            await self.websocket_manager.initialize()
            await self.hmm_manager.initialize(self.timeframe)
            
            # Get initial prices - FIXED request format
            try:
                for symbol in self.symbols:
                    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)  # Fixed parameter name
                    quote = self.data_client.get_stock_latest_quote(quote_request)
                    if quote and symbol in quote:
                        price = quote[symbol].ask_price
                        await self.display_manager.update_price(symbol, 'current', float(price))
                        await self.display_manager.update_price(symbol, 'open', float(price))
                        logger.info(f"Initial price for {symbol}: ${price:.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch initial prices: {str(e)}")
            
            # Pass display manager to websocket manager
            self.websocket_manager.display_manager = self.display_manager
            
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {str(e)}")
            raise

    def calculate_trade(self, bull_prob: float, bear_prob: float) -> Tuple[float, float]:
        """Determine the trade multiplier based on probabilities"""
        trade_multiplier = 0.0
        confidence_level = 0.0

        if bull_prob >= 0.8:
            confidence_level = bull_prob
            if bull_prob > 0.95:
                trade_multiplier = 1.0
            else:
                trade_multiplier = 1.0
        elif bear_prob > 0.9:
            confidence_level = bear_prob
            trade_multiplier = -1.0

        return trade_multiplier, confidence_level

    async def execute_trades(self, allocations: Dict[str, Tuple[float, float]]):
        """Execute trades based on calculated allocations"""
        try:
            account = self.trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
            
            positions = self.trading_client.get_all_positions()
            current_positions = {
                p.symbol: float(p.market_value)
                for p in positions
            }

            # Validate allocations through risk manager
            allocations = self.risk_manager.validate_allocation(
                allocations, portfolio_value, current_positions
            )

            for symbol, (allocation, leverage) in allocations.items():
                try:
                    current_value = current_positions.get(symbol, 0)
                    target_value = portfolio_value * allocation * abs(leverage)
                    
                    if abs(target_value - current_value) < 1:
                        continue

                    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    quote = self.data_client.get_stock_latest_quote(quote_request)
                    price = quote[symbol].ask_price
                    
                    shares_delta = int((target_value - current_value) / price)
                    
                    if shares_delta != 0:
                        order_data = MarketOrderRequest(
                            symbol=symbol,
                            qty=abs(shares_delta),
                            side=OrderSide.BUY if shares_delta > 0 else OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            extended_hours=self.is_extended_hours_session
                        )
                        
                        order = self.trading_client.submit_order(order_data)
                        logger.info(f"Executed trade for {symbol}: {shares_delta} shares at ${price:.2f}")

                except Exception as e:
                    logger.error(f"Failed to execute trade for {symbol}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Failed to execute trades: {str(e)}")
            raise OrderExecutionError(f"Failed to execute trades: {str(e)}")

    async def trading_iteration(self):
        try:
            logger.info("Beginning trading iteration...")
            
            allocations = {}
            selected_symbols = []
            probabilities = []
            
            for symbol in self.symbols:
                # Debug prints
                logger.info(f"Checking cache for {symbol}")
                if symbol in self.data_manager.cached_data:
                    logger.info(f"Found {len(self.data_manager.cached_data[symbol])} cached points for {symbol}")
                else:
                    logger.info(f"No cached data found for {symbol}")
                    
                cached_data = self.data_manager.cached_data.get(symbol)
                if cached_data is None:
                    logger.warning(f"No cached data available for {symbol}, fetching from database")
                    cached_data = await self.data_manager.get_historical_data(symbol, self.timeframe)
                    
                if cached_data is None or len(cached_data) < self.data_manager.min_history_points:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                latest_return = cached_data['returns'].iloc[-1]
                await self.hmm_manager.update_probabilities(symbol, latest_return)
                bull_prob, bear_prob = self.hmm_manager.get_state_probabilities(symbol)
                
                trade_multiplier, confidence = self.calculate_trade(bull_prob, bear_prob)
                
                if trade_multiplier != 0:
                    selected_symbols.append(symbol)
                    allocations[symbol] = (1.0, trade_multiplier)
                    probabilities.append((bull_prob, bear_prob))
                    logger.info(f"{symbol} - Bull: {bull_prob:.2f}, Bear: {bear_prob:.2f}, Multiplier: {trade_multiplier}")

            if selected_symbols:
                # Equal weight distribution among selected symbols
                weight = 1.0 / len(selected_symbols)
                final_allocations = {
                    symbol: (weight, alloc[1])
                    for symbol, alloc in allocations.items()
                }
                
                logger.info(f"Executing trades with allocations: {final_allocations}")
                await self.execute_trades(final_allocations)
            else:
                logger.info("No trades to execute")
                
            self.last_trading_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {str(e)}")
            raise

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        display_task = None
        
        try:
            await self.initialize()
            display_task = asyncio.create_task(self.display_manager.start())
            
            while self.is_running:
                try:
                    if not self.ignore_market_hours:
                        clock = self.trading_client.get_clock()
                        if not clock.is_open:
                            logger.info("Market is closed. Waiting...")
                            await asyncio.sleep(60)
                            continue

                    current_time = datetime.now()
                    
                    # Update portfolio display
                    try:
                        account = self.trading_client.get_account()
                        positions = self.trading_client.get_all_positions()
                        portfolio_value = float(account.portfolio_value)
                        
                        # Initialize all symbols to 0%
                        logger.info(f"Account portfolio value: ${portfolio_value}")
                        for position in positions:
                            logger.info(f"Position - Symbol: {position.symbol}, Market Value: ${float(position.market_value)}, Qty: {position.qty}")
                        
                        portfolio = {symbol: 0.0 for symbol in self.symbols}
                        
                        # Update with actual positions
                        for p in positions:
                            if p.symbol in self.symbols:
                                # Calculate actual percentage based on current position value
                                position_value = float(p.market_value)
                                percentage = (position_value / portfolio_value) * 100
                                portfolio[p.symbol] = percentage
                                logger.info(f"Calculated percentage for {p.symbol}: {percentage:.2f}%")

                        await self.display_manager.update_portfolio_allocations(portfolio)
                        
                    except Exception as e:
                        logger.error(f"Error updating portfolio display: {str(e)}")

                    # Check if it's time for next trade iteration
                    if (self.last_trading_time is None or
                        (current_time - self.last_trading_time).total_seconds() >= 
                        self.trading_interval * 3600):
                        
                        await self.trading_iteration()
                    
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Fatal error in trading system: {str(e)}")
            await self.emergency_shutdown()
        
        finally:
            self.is_running = False
            if display_task:
                await self.display_manager.stop()
                display_task.cancel()
                try:
                    await display_task
                except asyncio.CancelledError:
                    pass

    async def emergency_shutdown(self):
        """Handle emergency shutdown"""
        logger.critical("Initiating emergency shutdown")
        self.is_running = False
        
        try:
            await self.websocket_manager.close()
            
            try:
                self.trading_client.cancel_orders
                logger.info("Cancelled all pending orders")
                
                positions = self.trading_client.get_all_positions()
                for position in positions:
                    logger.info(f"Position at shutdown - {position.symbol}: {position.qty} shares at ${position.current_price}")
            except Exception as e:
                logger.error(f"Error during order/position handling: {str(e)}")
            
            await self._save_system_state()
            
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {str(e)}")

    async def _save_system_state(self):
        """Save the current system state"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'forward_probs': {
                    symbol: probs.tolist() if isinstance(probs, np.ndarray) else None
                    for symbol, probs in self.hmm_manager.forward_probs.items()
                },
                'last_trading_time': self.last_trading_time.isoformat() if self.last_trading_time else None
            }
            
            with open('trading_system_state.json', 'w') as f:
                json.dump(state, f)
            logger.info("Successfully saved system state")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {str(e)}")

if __name__ == "__main__":
    pass