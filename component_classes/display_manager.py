from typing import Dict, Optional, List
from datetime import datetime
import os
from tabulate import tabulate
import asyncio
import logging
from collections import defaultdict
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisplayManager:
    def __init__(self):
        self.price_data = defaultdict(lambda: {
            'portfolio_pct': 0.0,
            'open': 0.0,
            'current': 0.0,
            'high': float('-inf'),
            'low': float('inf'),
            'last_update': None
        })
        self.is_running = False
        self.update_lock = asyncio.Lock()
        self.previous_lines = 0
        self.start_line = 0

    async def _display_loop(self):
        """Main display loop"""
        # Print a separator line
        print("\n" + "="*50)
        print("=== Live Trading Status ===")
        print("="*50 + "\n")
        
        # Save where our display section starts
        self.start_line = self.previous_lines
        
        while self.is_running:
            # Move cursor to start of our section
            if self.previous_lines > 0:
                sys.stdout.write(f"\033[{self.previous_lines}A")
            
            # Generate the new table
            table = self._format_table()
            
            # Clear previous content line by line
            if self.previous_lines > 0:
                for _ in range(self.previous_lines):
                    sys.stdout.write("\033[2K")  # Clear line
                    sys.stdout.write("\033[1B")  # Move down
                sys.stdout.write(f"\033[{self.previous_lines}A")  # Back to top
            
            # Print new table
            print(table)
            print("\nPress Ctrl+C to stop trading\n")
            
            # Update line count
            self.previous_lines = len(table.split('\n')) + 2  # +2 for the Ctrl+C message and blank line
            
            # Flush output
            sys.stdout.flush()
            
            await asyncio.sleep(1)

    async def update_portfolio_allocations(self, portfolio_data: Dict[str, float]):
        """Update portfolio allocation percentages"""
        async with self.update_lock:
            for symbol, percentage in portfolio_data.items():
                self.price_data[symbol]['portfolio_pct'] = percentage
                # Add logging to verify updates
                logger.info(f"Updated portfolio allocation for {symbol}: {percentage:.2f}%")

    async def update_price(self, symbol: str, price_type: str, value: float):
        """Update price information for a symbol"""
        async with self.update_lock:
            if price_type == 'current':
                self.price_data[symbol]['current'] = value
                if self.price_data[symbol]['high'] == float('-inf'):
                    self.price_data[symbol]['high'] = value
                if self.price_data[symbol]['low'] == float('inf'):
                    self.price_data[symbol]['low'] = value
                self.price_data[symbol]['high'] = max(self.price_data[symbol]['high'], value)
                self.price_data[symbol]['low'] = min(self.price_data[symbol]['low'], value)
                self.price_data[symbol]['last_update'] = datetime.now()
            elif price_type == 'open':
                self.price_data[symbol]['open'] = value

    async def initialize_symbols(self, symbols: List[str]):
        """Initialize display with symbols"""
        async with self.update_lock:
            for symbol in symbols:
                if symbol not in self.price_data:
                    self.price_data[symbol] = {
                        'portfolio_pct': 0.0,
                        'open': 0.0,
                        'current': 0.0,
                        'high': float('-inf'),
                        'low': float('inf'),
                        'last_update': None
                    }

    def _save_cursor_position(self):
        """Save the current cursor position"""
        sys.stdout.write("\033[s")
        sys.stdout.flush()

    def _restore_cursor_position(self):
        """Restore the saved cursor position"""
        sys.stdout.write("\033[u")
        sys.stdout.flush()

    def _move_cursor_up(self, lines):
        """Move cursor up n lines"""
        sys.stdout.write(f"\033[{lines}A")
        sys.stdout.flush()

    def _clear_lines(self, num_lines):
        """Clear n lines from current cursor position"""
        for _ in range(num_lines):
            sys.stdout.write("\033[2K")  # Clear current line
            sys.stdout.write("\033[1A")  # Move up one line
        sys.stdout.write("\033[2K")  # Clear the last line
        sys.stdout.flush()

    def _format_table(self) -> str:
        """Format the price data into a table"""
        headers = ['Symbol', 'Portfolio %', 'Open', 'Current', 'High-Low', 'Last Update']
        rows = []
        
        for symbol, data in sorted(self.price_data.items()):
            # Calculate high-low range
            high_low = f"${data['high'] - data['low']:.2f}" if data['high'] != float('-inf') else "N/A"
            
            # Format last update time
            last_update = (data['last_update'].strftime('%H:%M:%S') 
                         if data['last_update'] else 'N/A')
            
            # Format price change
            current_price = data['current']
            open_price = data['open']
            price_change = ((current_price - open_price) / open_price * 100) if open_price != 0 else 0
            current_display = f"${current_price:.2f} ({price_change:+.1f}%)"
            
            row = [
                symbol,
                f"{data['portfolio_pct']:.1f}%",
                f"${data['open']:.2f}",
                current_display,
                high_low,
                last_update
            ]
            rows.append(row)
        
        return tabulate(
            rows,
            headers=headers,
            tablefmt='grid',
            numalign='right',
            stralign='center'
        )
            
    async def start(self):
        """Start the display"""
        self.is_running = True
        await self._display_loop()

    async def stop(self):
        """Stop the display"""
        self.is_running = False