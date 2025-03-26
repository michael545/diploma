import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

class DelayedDataApp(EClient, EWrapper):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.data = {}
        self.nextOrderId = None
        self.data_ready = threading.Event()
        self.errors = []
        self.debug_messages = []

    def error(self, reqId: int, errorCode: int, errorString: str, advanced: any = None) -> None:
        msg = f"Error {errorCode}: {errorString} (reqId: {reqId})"
        print(msg)
        self.errors.append(msg)

    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextOrderId = orderId
        print(f"Connected to TWS. Next order ID: {orderId}")

    def historicalData(self, reqId: int, bar: BarData) -> None:
        # Log each bar we receive
        bar_msg = f"Bar: {bar.date}, O:{bar.open}, H:{bar.high}, L:{bar.low}, C:{bar.close}, V:{bar.volume}"
        self.debug_messages.append(bar_msg)
        
        if reqId not in self.data:
            self.data[reqId] = []
            
        # Add the bar to our data list
        self.data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        print(f"Historical data complete. From {start} to {end}")
        
        # Convert our list to a DataFrame
        if reqId in self.data and self.data[reqId]:
            df = pd.DataFrame(self.data[reqId])
            
            # Convert date strings to datetime
            try:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Update our data dictionary with the DataFrame
                self.data[reqId] = df
                
                print(f"Processed {len(df)} bars into DataFrame")
            except Exception as e:
                print(f"Error processing data: {e}")
        else:
            print("No data received")
            
        # Signal that data retrieval is complete
        self.data_ready.set()

def get_historical_data(symbol="NVDA", days=30):
    """
    Get historical data for the specified symbol
    """
    app = DelayedDataApp()
    
    try:
        print(f"Connecting to TWS to get {days} days of historical data for {symbol}...")
        app.connect("127.0.0.1", 7496, clientId=100)  # Using a different client ID
        
        # Start the client thread
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()
        
        # Wait for connection
        connection_timeout = 10
        for i in range(connection_timeout):
            if isinstance(app.nextOrderId, int):
                break
            print(f"Waiting for connection... ({i+1}/{connection_timeout})")
            time.sleep(1)
        
        if not isinstance(app.nextOrderId, int):
            print("Failed to connect to TWS.")
            return None
        
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # Reset the data ready event
        app.data_ready.clear()
        
        # Request historical data with delayed data settings
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")
        
        # Make several attempts with different settings
        strategies = [
            # (duration, barSize, useRTH, whatToShow)
            (f"{days} D", "1 day", 1, "TRADES"),     # Daily bars
            (f"{days} D", "1 hour", 1, "TRADES"),    # Hourly bars
            (f"{days} D", "1 day", 1, "MIDPOINT"),   # Daily midpoint prices
            (f"{days} D", "1 day", 0, "TRADES"),     # Include non-RTH
            (f"{days} D", "1 hour", 0, "MIDPOINT")   # Hourly midpoint, non-RTH
        ]
        
        for i, (duration, bar_size, use_rth, what_to_show) in enumerate(strategies):
            if i > 0 and app.data and 1 in app.data and isinstance(app.data[1], pd.DataFrame) and not app.data[1].empty:
                # We already have data, skip additional attempts
                break
                
            # Clear previous data
            app.data = {}
            app.data_ready.clear()
            app.errors = []
            app.debug_messages = []
            
            print(f"\nAttempt {i+1}: Requesting {duration} of {bar_size} {what_to_show} data (useRTH={use_rth})...")
            
            app.reqHistoricalData(
                reqId=1,
                contract=contract,
                endDateTime=end_date_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,  # Try different data types
                useRTH=use_rth,           # Try with and without regular trading hours
                formatDate=1,             # yyyyMMdd HH:mm:ss
                keepUpToDate=False,
                chartOptions=[]
            )
            
            # Wait with timeout
            data_timeout = 15
            print(f"Waiting for data (timeout: {data_timeout} seconds)...")
            app.data_ready.wait(data_timeout)
            
            if 1 in app.data and isinstance(app.data[1], pd.DataFrame) and not app.data[1].empty:
                print(f"Success! Retrieved {len(app.data[1])} bars of data")
                print("\nFirst few rows:")
                print(app.data[1].head())
                
                # Show debug info
                print("\nDebug messages:")
                for msg in app.debug_messages[:5]:  # First 5 messages
                    print(f"- {msg}")
                if len(app.debug_messages) > 5:
                    print(f"... and {len(app.debug_messages)-5} more")
                
                # Save to CSV
                csv_filename = f"{symbol}_historical_{bar_size.replace(' ', '_')}.csv"
                app.data[1].to_csv(csv_filename)
                print(f"\nData saved to {csv_filename}")
                
                return app.data[1]
            else:
                print("No data received with these settings.")
                if app.errors:
                    print("Errors:")
                    for err in app.errors:
                        print(f"- {err}")
                        
                # Try again with different settings
                time.sleep(1)
        
        print("\nAll attempts failed to retrieve data.")
        print("Suggestions:")
        print("1. Check TWS logs for any error messages")
        print("2. Make sure delayed data is enabled in TWS")
        print("3. Try a more liquid symbol like SPY or AAPL")
        print("4. Try during regular market hours")
        print("5. Check if your account has market data subscription for this symbol")
        
        return None
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if 'app' in locals() and app.isConnected():
            app.disconnect()
            print("Disconnected from TWS")

if __name__ == "__main__":
    # Get user input
    symbol = input("Enter symbol (default: NVDA): ").strip() or "NVDA"
    try:
        days = int(input("Enter number of days of history (default: 30): ").strip() or "30")
    except ValueError:
        days = 30
        print("Invalid input, using default 30 days")
    
    # Get the data
    df = get_historical_data(symbol, days)
    
    if df is not None:
        # Calculate example indicators
        print("\nCalculating indicators...")
        # 20-day moving average
        df['MA20'] = df['close'].rolling(window=20).mean()
        # 50-day moving average
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        print(df.tail())