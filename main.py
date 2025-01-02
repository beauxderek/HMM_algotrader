import asyncio
import pandas as pd
import json
import subprocess
import os
import itertools
import sys
import time
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca_trade_api.rest import REST
import logging
from component_classes.data_manager import DataManager
from backtest import create_backtest
from strategies import PairsIdentifier, PairAnalysis, PairsStrategy
from ascii import main_menu, hidden_money_model, data_collection, backtest_ascii
#from livetrader import LiveTrader
from config import (
    LIVE_TRADING_SECURITIES, SECURITIES, DB_CONFIG, 
    ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL,
    TRADING_INTERVAL, POLYGON_API_KEY, PAIRS_PARAMETERS,
    PAIRS_UNIVERSE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

alpaca_config = {
    'api_key': ALPACA_API_KEY,
    'secret_key': ALPACA_SECRET_KEY,
    'base_url': BASE_URL
}

def get_user_choice(prompt, options=None):
    """Get validated user input"""
    while True:
        choice = input(prompt).strip().lower()
        if options is None:
            return choice
        elif choice in options:
            return choice
        else:
            print(f"Invalid choice. Please choose from {', '.join(options)}.")

def loading_screen(loading_screen_index):
    """Display loading animation"""
    ellipsis_ = itertools.cycle(['', '.', '..', '...'])
    message = "Loading, please wait"

    if loading_screen_index == 1:
        end_time = time.time() + 4
        while time.time() < end_time:
            sys.stdout.write(f"\r{message}{next(ellipsis_)} ")
            sys.stdout.flush()
            time.sleep(0.2)
        
        sys.stdout.write("\rLoading complete.           \n")
        sys.stdout.flush()

    if loading_screen_index == 4:
        print(hidden_money_model)
        end_time = time.time() + 4
        while time.time() < end_time:
            sys.stdout.write(f"\r{message}{next(ellipsis_)} ")
            sys.stdout.flush()
            time.sleep(0.2)
        
        sys.stdout.write("\rLoading complete.           \n")
        sys.stdout.flush()    

async def collect_historic_data():
    """Collect historical data using Polygon API"""
    print(data_collection)  # ASCII art display
    loading_screen(1)
    
    symbol_choice = get_user_choice(
        "Use configured symbols or input manually? (config/manual): ",
        ['config', 'manual']
    )
    
    if symbol_choice == 'manual':
        symbols_input = input("Enter symbols separated by commas (e.g., SPY,QQQ,AAPL): ")
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
    else:
        symbols = SECURITIES
    
    timeframe = get_user_choice(
        "Select data timeframe (10min/hour/day): ",
        ['10min', 'hour', 'day']
    )
    
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD) or 'now' for current date: ")
    
    if end_date.lower() == 'now':
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data_manager = DataManager(DB_CONFIG, POLYGON_API_KEY)
        # Initialize database synchronously first
        data_manager.initialize_database()
        
        for symbol in symbols:
            print(f"\nCollecting data for {symbol}...")
            df = await data_manager.fetch_and_store_data(
                symbol, start_date, end_date, timeframe
            )
            
            if df is None or df.empty:
                print(f"No data collected for {symbol}")
            else:
                print(f"Successfully collected {len(df)} records for {symbol}")
        
        print("\nData collection completed successfully!")
        input("\nPress Enter to return to main menu...")
        
    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")
        print(f"An error occurred while collecting data: {str(e)}")
        input("\nPress Enter to return to main menu...")

async def verify_stored_data():
    """Verify and optionally export stored data"""
    try:
        data_manager = DataManager(DB_CONFIG, POLYGON_API_KEY)
        
        print("\nWhat would you like to do?")
        print("1. Verify stored data")
        print("2. Export data to CSV")
        
        action = get_user_choice("Choice (1/2): ", ['1', '2'])
        
        if action == '2':
            await export_data(data_manager)
        else:
            await verify_data(data_manager)
            
    except Exception as e:
        logger.error(f"Error accessing database: {str(e)}")
        print(f"An error occurred while accessing the database: {str(e)}")

async def export_data(data_manager):
    """Export data to CSV files"""
    price_data_dir = "Price Data"
    if not os.path.exists(price_data_dir):
        os.makedirs(price_data_dir)
        print(f"\nCreated directory: {price_data_dir}")
    
    symbol_choice = get_user_choice(
        "\nUse configured symbols or input manually? (config/manual): ",
        ['config', 'manual']
    )
    
    if symbol_choice == 'manual':
        symbols_input = input("Enter symbols separated by commas (e.g., SPY,QQQ,AAPL): ")
        export_symbols = [s.strip().upper() for s in symbols_input.split(',')]
    else:
        export_symbols = SECURITIES
    
    timeframe = get_user_choice(
        "Select timeframe (10min/hour/day): ",
        ['10min', 'hour', 'day']
    )
    
    for symbol in export_symbols:
        try:
            # Await the async get_market_data call
            data = await data_manager.get_market_data(
                symbol,
                "1900-01-01",
                datetime.now().strftime('%Y-%m-%d'),
                timeframe
            )
            
            if data.empty:
                print(f"No data found for {symbol}")
                continue
            
            filename = f"{symbol}_{timeframe}.csv"
            filepath = os.path.join(price_data_dir, filename)
            data.to_csv(filepath)
            print(f"Exported {symbol} data to: {filepath}")
        except Exception as e:
            logger.error(f"Error exporting data for {symbol}: {str(e)}")
            print(f"Failed to export data for {symbol}: {str(e)}")

async def verify_data(data_manager):
    """Verify stored data quality"""
    while True:
        symbol = input("\nEnter symbol to verify (or 'exit' to return to main menu): ").upper()
        if symbol.lower() == 'exit':
            break
        
        print(f"\nAvailable timeframes for {symbol}:")
        print("1. 10-minute")
        print("2. Hourly")
        print("3. Daily")
        
        timeframe_choice = get_user_choice("Select timeframe (1/2/3): ", ['1', '2', '3'])
        timeframe_map = {'1': '10min', '2': 'hour', '3': 'day'}
        timeframe = timeframe_map[timeframe_choice]
        
        try:
            # Await the async get_market_data call
            data = await data_manager.get_market_data(
                symbol,
                "1900-01-01",
                datetime.now().strftime('%Y-%m-%d'),
                timeframe
            )
            
            if data.empty:
                print(f"No data found for {symbol}")
                continue
            
            print(f"\nData summary for {symbol} ({timeframe}):")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            print(f"Number of records: {len(data)}")
            print("\nPrice statistics:")
            print(f"Latest price: ${data['adj_close'].iloc[-1]:.2f}")
            print(f"High: ${data['high'].max():.2f}")
            print(f"Low: ${data['low'].min():.2f}")
            print(f"Average: ${data['adj_close'].mean():.2f}")
            
            missing_values = data.isnull().sum()
            if missing_values.any():
                print("\nMissing values:")
                for col, count in missing_values.items():
                    if count > 0:
                        print(f"{col}: {count}")
            
        except Exception as e:
            logger.error(f"Error verifying data for {symbol}: {str(e)}")
            print(f"An error occurred while verifying {symbol} data: {str(e)}")

async def identify_pairs():
    """Identify and save valid pairs for trading"""
    print()
    loading_screen(1)
    
    # Get user input for pairs universe
    print("\nAvailable sectors for pairs analysis:")
    for i, sector in enumerate(PAIRS_UNIVERSE.keys(), 1):
        print(f"{i}. {sector}")
    
    sectors = input("\nEnter sector numbers to analyze (comma-separated) or 'all': ")
    
    # Build list of potential pairs
    potential_pairs = []
    if sectors.lower() == 'all':
        symbols = [sym for syms in PAIRS_UNIVERSE.values() for sym in syms]
        potential_pairs = [(s1, s2) for i, s1 in enumerate(symbols) 
                          for s2 in symbols[i+1:]]
    else:
        selected_sectors = [list(PAIRS_UNIVERSE.keys())[int(i)-1] 
                          for i in sectors.split(',')]
        symbols = []
        for sector in selected_sectors:
            symbols.extend(PAIRS_UNIVERSE[sector])
        potential_pairs = [(s1, s2) for i, s1 in enumerate(symbols) 
                          for s2 in symbols[i+1:]]
    
    # Get data and analyze pairs
    data_manager = DataManager(DB_CONFIG, POLYGON_API_KEY)
    pairs_identifier = PairsIdentifier()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    data = {}
    for symbol in set([sym for pair in potential_pairs for sym in pair]):
        try:
            # Await the coroutine
            df = await data_manager.fetch_and_store_data(symbol, start_date, end_date, '10min')
            if df is not None and not df.empty:
                # Reindex the data frame to a common date range
                all_dates = pd.date_range(start=start_date, end=end_date, freq='10min')
                df = df.reindex(all_dates, fill_value=0)
                data[symbol] = df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            continue

    print("\nAnalyzing potential pairs...")
    results = pairs_identifier.analyze_potential_pairs(data, potential_pairs)
    
    if results:
        print("\nValid pairs identified:")
        for pair in results:
            print(f"\n{pair.symbol1} - {pair.symbol2}")
            print(f"Correlation: {pair.correlation:.3f}")
            print(f"Half-life: {pair.half_life:.1f} periods")
            print(f"Hedge ratio: {pair.beta:.3f}")
        
        pairs_identifier.save_valid_pairs(results)
        print("\nValid pairs saved successfully!")
    else:
        print("\nNo valid pairs identified with current parameters.")
    
    input("\nPress Enter to return to main menu...")

async def run_backtest_workflow():
    """Run backtest workflow with strategy selection"""
    print(backtest_ascii)  # ASCII art display
    
    # Get basic parameters
    start_date = get_user_choice("Enter start date (YYYY-MM-DD): ")
    end_date = get_user_choice("Enter end date (YYYY-MM-DD): ")
    timeframe = get_user_choice(
        "Select data timeframe (10min/hour/day): ",
        ['10min', 'hour', 'day']
    )
    
    # Strategy selection
    print("\nAvailable Strategies:")
    print("1. HMM Trend Following")
    print("2. Pairs Trading")
    strategy_choice = get_user_choice("Select strategy (1/2): ", ['1', '2'])
    
    try:
        if strategy_choice == '1':
            # Trend Following setup
            print("\nAvailable symbols:", ', '.join(SECURITIES))
            symbols_input = input("Enter symbols to trade (comma-separated) or 'all': ").strip()
            
            if symbols_input.lower() == 'all':
                selected_symbols = SECURITIES
            else:
                selected_symbols = [sym.strip().upper() for sym in symbols_input.split(',')]
                
            # Create backtest instance
            backtest = TrendFollowingBacktest(
                symbols=selected_symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                probability_threshold=STRATEGY_TYPES['hmm_trend']['probability_threshold']
            )
            
        else:
            # Pairs Trading setup
            try:
                with open('valid_pairs.json', 'r') as f:
                    pairs_data = json.load(f)
                    available_pairs = [(p['symbol1'], p['symbol2']) for p in pairs_data['pairs']]
            except FileNotFoundError:
                print("\nNo valid pairs found. Please run pairs identification first.")
                return
                
            print("\nAvailable pairs:")
            for i, (sym1, sym2) in enumerate(available_pairs, 1):
                print(f"{i}. {sym1}-{sym2}")
                
            pairs_input = input("\nEnter pair numbers to trade (comma-separated) or 'all': ").strip()
            
            if pairs_input.lower() == 'all':
                selected_pairs = available_pairs
            else:
                indices = [int(i.strip())-1 for i in pairs_input.split(',')]
                selected_pairs = [available_pairs[i] for i in indices]
            
            # Create backtest instance
            backtest = PairsBacktest(
                pairs=selected_pairs,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                lookback_period=STRATEGY_TYPES['pairs']['lookback_period'],
                entry_zscore=STRATEGY_TYPES['pairs']['entry_zscore'],
                exit_zscore=STRATEGY_TYPES['pairs']['exit_zscore']
            )
        
        # Run backtest
        print("\nInitializing backtest...")
        await backtest.strategy.initialize()
        
        print("\nRunning backtest...")
        results = await backtest.run()
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = "trend_following" if strategy_choice == '1' else "pairs_trading"
        report_filename = f"backtest_{strategy_name}_{timestamp}.pdf"
        
        # Generate report
        backtest.generate_report(report_filename)
        print(f"\nBacktest completed. Report saved as: {report_filename}")
        
        # Display key metrics
        print("\nKey Performance Metrics:")
        print(f"Total Return: {results['metrics']['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
        
        input("\nPress Enter to return to main menu...")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")
        print(f"\nError during backtest: {str(e)}")
        input("\nPress Enter to return to main menu...")

async def run_live_trading():
    print("not for now")
    """Initialize and run live trading"""
    '''try:
        print("\nInitializing Live Trading System...")
        
        # Strategy selection
        print("\nAvailable Strategies:")
        print("1. HMM Trend Following")
        print("2. Pairs Trading")
        strategy_choice = get_user_choice("Select strategy (1/2): ", ['1', '2'])
        
        if strategy_choice == '1':
            trading_symbols = LIVE_TRADING_SECURITIES
        else:
            try:
                # Continue run_live_trading function
                with open('valid_pairs.json', 'r') as f:
                    pairs_data = json.load(f)
                    available_pairs = [(p['symbol1'], p['symbol2']) for p in pairs_data['pairs']]
                    
                print("\nAvailable pairs:")
                for i, (sym1, sym2) in enumerate(available_pairs, 1):
                    print(f"{i}. {sym1}-{sym2}")
                
                pairs_input = input("\nEnter pair numbers to trade (comma-separated) or 'all': ").strip()
                
                if pairs_input.lower() == 'all':
                    selected_pairs = available_pairs
                else:
                    indices = [int(i.strip())-1 for i in pairs_input.split(',')]
                    selected_pairs = [available_pairs[i] for i in indices]
                    
                trading_symbols = list(set([sym for pair in selected_pairs for sym in pair]))
            
            except FileNotFoundError:
                print("\nNo valid pairs found. Please run pairs identification first.")
                return
        
        print(f"\nTrading Symbols: {', '.join(trading_symbols)}")
        
        timeframe = get_user_choice(
            "Select timeframe (10min/hour/day): ",
            ['10min', 'hour', 'day']
        )
        
        # Check data availability
        data_manager = DataManager(DB_CONFIG, POLYGON_API_KEY)
        missing_symbols = []
        
        for symbol in trading_symbols:
            data = data_manager.get_market_data(
                symbol,
                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d'),
                timeframe
            )
            if data.empty:
                missing_symbols.append(symbol)
        
        if missing_symbols:
            print(f"\nWarning: Missing data for symbols: {', '.join(missing_symbols)}")
            proceed = input("Do you want to proceed anyway? (yes/no): ").lower()
            if proceed != 'yes':
                print("Live trading cancelled.")
                return
        
        print(f"\nInitializing live trading system...")
        print(f"Trading Interval: {TRADING_INTERVAL} hour(s)")
        print("Using Alpaca API for market data streaming")
        
        # Create initial client to check market status
        temp_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True if 'paper' in BASE_URL else False
        )
        
        clock = temp_client.get_clock()
        ignore_market_hours = False
        
        if not clock.is_open:
            next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
            next_close = clock.next_close.strftime('%Y-%m-%d %H:%M:%S')
            
            print("\nWARNING: Market is currently closed!")
            print(f"Next market open: {next_open}")
            print(f"Next market close: {next_close}")
            
            after_hours_choice = input("\nWould you like to proceed with after-hours trading? (yes/no): ").lower()
            if after_hours_choice == 'yes':
                print("\nWARNING: Enabling after-hours trading. Be aware of:")
                print("- Lower liquidity may result in wider spreads")
                print("- Higher volatility and potential price gaps")
                print("- Some orders may not execute until regular market hours")
                
                confirm = input("\nDo you understand these risks and want to proceed? (yes/no): ").lower()
                if confirm == 'yes':
                    ignore_market_hours = True
                else:
                    print("Live trading cancelled.")
                    return
            else:
                print("Live trading cancelled.")
                return

        confirm = input("\nDo you want to start live trading? (yes/no): ").lower()
        if confirm != 'yes':
            print("Live trading cancelled.")
            return

        trader = LiveTrader(
            symbols=trading_symbols,
            strategy_type='pairs_trading' if strategy_choice == '2' else 'trend_following',
            strategy_params={'pairs': selected_pairs} if strategy_choice == '2' else {},
            db_config={**DB_CONFIG, 'timeframe': timeframe},
            alpaca_config=alpaca_config,
            trading_interval=TRADING_INTERVAL,
            ignore_market_hours=ignore_market_hours
        )
        
        try:
            print("\nStarting live trading system...")
            if ignore_market_hours:
                print("After-hours trading enabled")
            print("Press Ctrl+C to stop trading and initiate emergency shutdown")
            
            await trader.run()
        except KeyboardInterrupt:
            print("\nInitiating graceful shutdown...")
            await trader.emergency_shutdown()
        
    except Exception as e:
        logger.error(f"Error in live trading: {str(e)}")
        print(f"An error occurred during live trading: {str(e)}")'''

def main():
    while True:
        print(f"{main_menu}")
        print("\n=== Algorithmic Trading System ===")
        print("1. Collect Historic Data")
        print("2. Verify Stored Data")
        print("3. Identify Trading Pairs")
        print("4. Run Backtest")
        print("5. Live Trading")
        print("6. Exit")
        
        choice = get_user_choice("Select an option (1-6): ", ['1', '2', '3', '4', '5', '6'])
        
        if choice == '1':
            print(data_collection)
            loading_screen(1)
            asyncio.run(collect_historic_data())  # Changed this line
        elif choice == '2':
            print(data_collection)
            asyncio.run(verify_stored_data())  # Changed this line
        elif choice == '3':
            print()
            asyncio.run(identify_pairs())
        elif choice == '4':
            print(backtest_ascii)
            run_backtest_workflow()
        elif choice == '5':
            loading_screen(4)
            asyncio.run(run_live_trading())
        else:
            print("Exiting program...")
            break

if __name__ == "__main__":
    main()