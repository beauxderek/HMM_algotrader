import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from urllib3 import PoolManager
import sqlite3
import logging
import asyncio
from config import get_polygon_params, DB_CONFIG, MAX_DAILY_API_CALLS, RATE_LIMIT_PAUSE
from typing import Dict, List, Tuple, Optional, Set
import queue
from threading import Lock
from contextlib import contextmanager
from .exceptions import DatabaseConnectionError, MarketDataError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """Manages database connections"""
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(2):  # Start with two connections
            self._add_connection()

    def _add_connection(self):
        """Create new database connection"""
        if self.active_connections < self.max_connections:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            
            self.pool.put(conn)
            with self.lock:
                self.active_connections += 1

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        connection = None
        try:
            connection = self.pool.get(timeout=5)
            yield connection
        except queue.Empty:
            with self.lock:
                if self.active_connections < self.max_connections:
                    self._add_connection()
                    connection = self.pool.get()
                    yield connection
                else:
                    raise DatabaseConnectionError("No available database connections")
        finally:
            if connection is not None:
                self.pool.put(connection)

    def close_all(self):
        """Close all connections"""
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()
        self.active_connections = 0

class PolygonDataFetcher:
    """Handles data fetching from Polygon.io API"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.http = PoolManager()
        self.requests_today = 0
        self.last_request_time = datetime.min
        
    def fetch_aggregates(self, symbol: str, start_date: str, end_date: str, 
                        timeframe: dict) -> pd.DataFrame:
        """Fetch aggregate bars from Polygon"""
        if self.requests_today >= MAX_DAILY_API_CALLS:
            wait_time = RATE_LIMIT_PAUSE - (datetime.now() - self.last_request_time).seconds
            if wait_time > 0:
                time.sleep(wait_time)
            self.requests_today = 0

        multiplier = timeframe['multiplier']
        timespan = timeframe['timespan']
        
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/{start_date}/{end_date}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
        )
        
        all_results = []
        
        while url:
            try:
                response = self.http.request("GET", url)
                data = json.loads(response.data.decode('utf-8'))
                
                self.requests_today += 1
                self.last_request_time = datetime.now()
                
                if "results" not in data:
                    logger.warning(f"No results for {symbol}: {data.get('message', '')}")
                    break
                    
                all_results.extend(data["results"])
                url = data.get("next_url")
                if url:
                    url = f"{url}&apiKey={self.api_key}"
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                break
                
        if not all_results:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'num_trades'
        })
        
        df = df.drop(['t'], axis=1)
        df = df.set_index('timestamp')
        
        return df

class DataManager:
    def __init__(self, db_config: dict, polygon_api_key: str):
        self.db_path = db_config['database']
        self.connection_pool = DatabaseConnectionPool(self.db_path)
        self.polygon_fetcher = PolygonDataFetcher(polygon_api_key)
        self.cached_data: Dict[str, pd.DataFrame] = {}
        self.last_update_time: Dict[str, datetime] = {}
        self.min_history_points = 100

    async def initialize(self):
        """Initialize database and validate connection"""
        try:
            self.initialize_database()  # This is synchronous
            with self.connection_pool.get_connection() as conn:
                conn.execute("SELECT 1")
            logger.info("Database initialization successful")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise DatabaseConnectionError("Failed to initialize database")

    def initialize_database(self):
        """Initialize database schema - synchronous operation"""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables for each timeframe
            for table_name in DB_CONFIG['polygon_tables'].values():
                cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    vwap REAL,
                    num_trades INTEGER,
                    returns REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
                ''')
                
                # Create index for faster queries
                cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timestamp 
                ON {table_name} (symbol, timestamp)
                ''')
            
            # Create live trading data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_market_data (
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                returns REAL,
                PRIMARY KEY (symbol, timestamp)
            )
            ''')
            
            # Create metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol TEXT,
                timeframe TEXT,
                last_update DATETIME,
                records_count INTEGER,
                avg_gap_seconds REAL,
                data_quality_score REAL,
                PRIMARY KEY (symbol, timeframe)
            )
            ''')
            
            conn.commit()

    async def fetch_and_store_data(self, symbol: str, start_date: str, end_date: str, 
                                 timeframe: str) -> pd.DataFrame:
        """Fetch and store data with proper validation and error handling"""
        try:
            # Input validation
            if not all([symbol, start_date, end_date, timeframe]):
                raise ValueError("Missing required parameters")
            
            if timeframe not in DB_CONFIG['polygon_tables']:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Get polygon parameters
            params = get_polygon_params(timeframe)
            if not params:
                raise ValueError(f"Invalid timeframe configuration: {timeframe}")

            # Fetch data with retries
            for attempt in range(3):
                try:
                    df = self.polygon_fetcher.fetch_aggregates(
                        symbol, start_date, end_date, params
                    )
                    if not df.empty:
                        break
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Retry {attempt + 1} for {symbol}")
                    await asyncio.sleep(1)

            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()

            # Validate and process data
            df = self._validate_market_data(df)
            
            # Store validated data
            await self._store_market_data(symbol, df, timeframe)
            
            # Update cache
            self.cached_data[symbol] = df
            self.last_update_time[symbol] = datetime.now()

            # Update metadata
            await self._update_metadata(symbol, timeframe, df)

            return df

        except Exception as e:
            logger.error(f"Error in fetch_and_store_data for {symbol}: {str(e)}")
            raise

    def _validate_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation"""
        # Check required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not all(col in df.columns for col in required_cols):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure proper types
        df = df.astype({
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64'
        })

        # Add derived columns
        df['adj_close'] = df['close']
        df['returns'] = df['adj_close'].pct_change()

        # Validate price relationships
        valid_prices = (
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close']) &
            (df['volume'] >= 0)
        )
        
        # Remove invalid rows
        invalid_rows = ~valid_prices
        if invalid_rows.any():
            logger.warning(f"Removing {invalid_rows.sum()} invalid rows")
            df = df[valid_prices]

        # Check for gaps
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=10)  # Adjust based on timeframe
        gaps = time_diff[time_diff > expected_diff]
        if not gaps.empty:
            logger.warning(f"Found {len(gaps)} data gaps")

        return df

    async def _store_market_data(self, symbol: str, df: pd.DataFrame, timeframe: str):
        """Store data with proper error handling and duplicate prevention"""
        if df.empty:
            return

        table_name = DB_CONFIG['polygon_tables'][timeframe]
        
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    # Prepare data for storage
                    df_to_store = df.reset_index()
                    df_to_store['symbol'] = symbol
                    
                    # Remove existing data in the date range
                    start_date = df_to_store['timestamp'].min()
                    end_date = df_to_store['timestamp'].max()
                    
                    conn.execute(f"""
                        DELETE FROM {table_name}
                        WHERE symbol = ?
                        AND timestamp BETWEEN ? AND ?
                    """, (symbol, start_date, end_date))
                    
                    # Store new data
                    df_to_store.to_sql(
                        table_name,
                        conn,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                    
                    conn.execute("COMMIT")
                    logger.info(f"Successfully stored {len(df)} records for {symbol}")
                    
                except Exception as e:
                    conn.execute("ROLLBACK")
                    raise
                    
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {str(e)}")
            raise DatabaseConnectionError(f"Failed to store data: {str(e)}")

    async def _update_metadata(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Update metadata for stored data"""
        try:
            with self.connection_pool.get_connection() as conn:
                # Calculate data quality metrics
                gaps = df.index.to_series().diff()
                avg_gap = gaps.mean().total_seconds()
                
                missing_data = df.isnull().sum().sum()
                total_points = len(df) * len(df.columns)
                quality_score = 1 - (missing_data / total_points)
                
                # Update metadata
                conn.execute('''
                INSERT OR REPLACE INTO data_metadata 
                (symbol, timeframe, last_update, records_count, avg_gap_seconds, data_quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 
                    timeframe, 
                    datetime.now().isoformat(),
                    len(df),
                    avg_gap,
                    quality_score
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")

    async def get_market_data(self, symbol: str, start_date: str, end_date: str, 
                            timeframe: str, use_cache: bool = True) -> pd.DataFrame:
        """Get market data with caching support"""
        try:
            # Check cache first if enabled
            if use_cache and symbol in self.cached_data:
                cached_df = self.cached_data[symbol]
                mask = (cached_df.index >= pd.Timestamp(start_date)) & \
                       (cached_df.index <= pd.Timestamp(end_date))
                if mask.any():
                    return cached_df[mask]

            # Fetch from database
            table_name = DB_CONFIG['polygon_tables'][timeframe]
            
            with self.connection_pool.get_connection() as conn:
                query = f"""
                SELECT * FROM {table_name}
                WHERE symbol = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, start_date, end_date),
                    parse_dates=['timestamp'],
                    index_col='timestamp'
                )

                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    return df

                # Update cache
                if use_cache:
                    self.cached_data[symbol] = df
                    self.last_update_time[symbol] = datetime.now()

                return df

        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            raise MarketDataError(f"Failed to retrieve data: {str(e)}")

    async def update_live_data(self, symbol: str, new_data: pd.DataFrame):
        """Update live trading data"""
        try:
            if symbol not in self.cached_data:
                historical_data = await self.get_market_data(
                    symbol,
                    (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d'),
                    '10min'  # Default to 10-minute data for live trading
                )
                self.cached_data[symbol] = historical_data
            
            # Update cache
            self.cached_data[symbol] = pd.concat([self.cached_data[symbol], new_data])
            self.cached_data[symbol] = self.cached_data[symbol].loc[~self.cached_data[symbol].index.duplicated(keep='last')]
            self.last_update_time[symbol] = datetime.now()
            
            # Store in live data table
            with self.connection_pool.get_connection() as conn:
                new_data_to_store = new_data.copy()
                new_data_to_store['symbol'] = symbol
                new_data_to_store.to_sql(
                    'live_market_data',
                    conn,
                    if_exists='append',
                    index=True
                )
                
        except Exception as e:
            logger.error(f"Error updating live data: {str(e)}")
            raise MarketDataError(f"Failed to update live data: {str(e)}")

    async def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get latest cached data for a symbol"""
        return self.cached_data.get(symbol)

    def get_data_quality_metrics(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """Get data quality metrics from metadata"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT records_count, avg_gap_seconds, data_quality_score
                    FROM data_metadata
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'records_count': row[0],
                        'avg_gap_seconds': row[1],
                        'data_quality_score': row[2]
                    }
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving data quality metrics: {str(e)}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from the database"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            with self.connection_pool.get_connection() as conn:
                for table_name in DB_CONFIG['polygon_tables'].values():
                    conn.execute(f"""
                        DELETE FROM {table_name}
                        WHERE timestamp < ?
                    """, (cutoff_date,))
                
                conn.execute("""
                    DELETE FROM live_market_data
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.connection_pool.close_all()
        except:
            pass

if __name__ == "__main__":
    dm = DataManager(DB_CONFIG, None)  # For testing purposes