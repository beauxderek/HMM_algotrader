import asyncio
import pandas as pd
from .bar_aggregator import BarAggregator
import logging
from typing import List, Dict, Set
from .data_manager import DataManager
from alpaca.trading.client import TradingClient
from datetime import datetime
import json
from .exceptions import WebSocketError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str, symbols: List[str], data_manager: DataManager):
        # Use paper trading URL
        self.ws_url = "wss://paper-api.alpaca.markets/stream"
        self.api_key = alpaca_api_key
        self.secret_key = alpaca_secret_key
        self.symbols = symbols
        self.data_manager = data_manager
        self.bar_aggregator = BarAggregator()
        self.connected = False
        self.ws = None
        self.last_trade_times: Dict[str, datetime] = {}
        self.missed_data_ranges: Set[tuple] = set()

    async def initialize(self):
        """Initialize websocket connection and handlers"""
        try:
            import websockets
            
            # Initialize database table
            await self._initialize_websocket_table()
            
            # Connect to Alpaca WebSocket
            self.ws = await websockets.connect(self.ws_url)
            
            # Send authentication message
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self.ws.send(json.dumps(auth_message))
            
            # Wait for auth response
            response = await self.ws.recv()
            r = json.loads(response)
            
            if r.get('stream') == 'authorization' and r.get('data', {}).get('status') == 'authorized':
                logger.info("Successfully authenticated WebSocket connection")
                
                # Subscribe to trade updates
                subscribe_message = {
                    "action": "listen",
                    "data": {
                        "streams": ["trade_updates"]
                    }
                }
                await self.ws.send(json.dumps(subscribe_message))
                
                # Start listening for messages
                self.connected = True
                asyncio.create_task(self._listen_for_messages())
                
            else:
                raise WebSocketError("Failed to authenticate WebSocket connection")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise WebSocketError(f"WebSocket initialization failed: {str(e)}")

    async def _initialize_websocket_table(self):
        """Initialize database table for websocket data"""
        try:
            with self.data_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS websocket_bars (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vwap REAL,
                    trade_count INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize websocket table: {str(e)}")
            raise

    async def _listen_for_messages(self):
        """Listen for incoming messages"""
        try:
            while self.connected:
                message = await self.ws.recv()
                data = json.loads(message)
                
                if data.get('stream') == 'trade_updates':
                    await self._handle_trade(data['data'])
                    
        except Exception as e:
            logger.error(f"Error in message listener: {str(e)}")
            self.connected = False

    async def _handle_trade(self, trade_data):
        """Process incoming trade data"""
        try:
            logger.info(f"Trade Update Received: {trade_data['event']}")
            
            # Log all order details
            order = trade_data['order']
            logger.info(f"""
    Order Details:
    Symbol: {order['symbol']}
    Status: {order['status']}
    Side: {order['side']}
    Qty: {order['qty']}
    Filled Qty: {order['filled_qty']}
    Type: {order['type']}
    """)

            # Only process filled or partially_filled orders
            if trade_data['event'] not in ['fill', 'partial_fill']:
                return
                
            symbol = order['symbol']
            timestamp = pd.Timestamp(trade_data['timestamp'])
            
            # For filled orders, use filled price and quantity
            if order['filled_avg_price'] is None or order['filled_qty'] is None:
                logger.warning(f"Incomplete fill data for {symbol}")
                return
                
            price = float(order['filled_avg_price'])
            qty = float(order['filled_qty'])
            
            logger.info(f"Processing filled trade - Symbol: {symbol}, Price: ${price:.2f}, Quantity: {qty}")
            
            # Create new data point
            new_data = pd.DataFrame({
                'timestamp': [timestamp],
                'open': [price],
                'high': [price],
                'low': [price],
                'close': [price],
                'adj_close': [price],
                'volume': [qty],
            }).set_index('timestamp')
            
            # Update trading data
            await self.data_manager.update_live_data(symbol, new_data)
            
            # Update display
            if hasattr(self, 'display_manager'):
                await self.display_manager.update_price(symbol, 'current', price)
                
                # Force a portfolio update
                try:
                    trading_client = TradingClient(
                        api_key=self.api_key,
                        secret_key=self.secret_key
                    )
                    positions = trading_client.get_all_positions()
                    portfolio_value = float(trading_client.get_account().portfolio_value)
                    
                    portfolio = {}
                    for p in positions:
                        market_value = float(p.market_value)
                        portfolio[p.symbol] = (market_value / portfolio_value) * 100
                        logger.info(f"Position update - {p.symbol}: ${market_value:.2f} ({portfolio[p.symbol]:.2f}%)")
                    
                    await self.display_manager.update_portfolio_allocations(portfolio)
                except Exception as e:
                    logger.error(f"Error updating portfolio after trade: {str(e)}")
            
            logger.info(f"Successfully processed fill for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing trade: {str(e)}")
            logger.error(f"Trade data that caused error: {trade_data}")
            
    async def _store_bar(self, bar_data: Dict):
        """Store completed bar and update trading data"""
        try:
            # Store in websocket_bars table
            with self.data_manager.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO websocket_bars (
                    symbol, timestamp, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    bar_data['symbol'],
                    bar_data['timestamp'],
                    bar_data['open'],
                    bar_data['high'],
                    bar_data['low'],
                    bar_data['close'],
                    bar_data['volume']
                ))
                conn.commit()

            # Create DataFrame for trading updates
            df = pd.DataFrame([{
                'timestamp': bar_data['timestamp'],
                'adj_close': bar_data['close'],
                'returns': (bar_data['close'] - bar_data['open']) / bar_data['open']
            }])
            df.set_index('timestamp', inplace=True)

            # Update trading data
            await self.data_manager.update_live_data(bar_data['symbol'], df)
            
        except Exception as e:
            logger.error(f"Failed to store bar data: {str(e)}")
            raise

    async def _handle_data_gap(self):
        """Handle missing data during disconnection using Alpaca REST API"""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            # Initialize historical client
            historical_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            for symbol in self.symbols:
                if symbol in self.last_trade_times:
                    last_time = self.last_trade_times[symbol]
                    current_time = datetime.now()
                    
                    if (current_time - last_time).total_seconds() > 60:
                        # Record missing data range
                        self.missed_data_ranges.add((symbol, last_time, current_time))
                        
                        # Create request for missing data
                        request = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=TimeFrame.Minute,
                            start=last_time,
                            end=current_time
                        )
                        
                        # Fetch missing bars
                        bars = historical_client.get_stock_bars(request)
                        
                        # Process each bar
                        for bar in bars[symbol]:
                            await self._store_bar({
                                'symbol': symbol,
                                'timestamp': bar.timestamp,
                                'open': bar.open,
                                'high': bar.high,
                                'low': bar.low,
                                'close': bar.close,
                                'volume': bar.volume
                            })
                        
                        logger.info(f"Recovered {len(bars[symbol])} bars for {symbol}")
                        
        except Exception as e:
            logger.error(f"Failed to handle data gap: {str(e)}")

    async def close(self):
        """Close the websocket connection"""
        try:
            if self.connected and self.ws:
                # Unsubscribe from updates
                unsubscribe_message = {
                    "action": "listen",
                    "data": {
                        "streams": []
                    }
                }
                await self.ws.send(json.dumps(unsubscribe_message))
                
                # Close connection
                await self.ws.close()
                self.connected = False
                logger.info("Closed WebSocket connection")
        except Exception as e:
            logger.error(f"Error closing WebSocket connection: {str(e)}")