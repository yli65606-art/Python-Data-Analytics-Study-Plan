from ib_insync import IB, Stock, util
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nest_asyncio
import pytz 
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA # Only use this import
import datetime as dt

nest_asyncio.apply()

# 1. Redefine the Day Trading Strategy Class with fixed P&L
class DayTradingStrategyFixedPL(Strategy):
    n1 = 9
    n2 = 20
    take_profit_pct = 2  # 10%
    stop_loss_pct = 1    # 5%

    def init(self):
        close = self.data.Close
        # This is correct for using backtesting.test.SMA
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        current_time = self.data.index[-1].time()
        if current_time >= dt.time(15, 50, 0):
            if self.position:
                self.position.close()
            return
        # Use the crossover function on the indicator objects managed by Backtest
        if crossover(self.sma1, self.sma2):
            self.buy(tp=self.data.Close[-1] * (1 + self.take_profit_pct / 100),
                     sl=self.data.Close[-1] * (1 - self.stop_loss_pct / 100))
        elif crossover(self.sma2, self.sma1):
            if self.position:
                self.position.close()


# The following code shall not be modified
# 2. Define the data fetching function
def fetch_historical_data(ib, contract, duration, bar_size):
    ny_tz = pytz.timezone('America/New_York')
    end_date = datetime.now(ny_tz)
    all_bars = []
    target_start_date = end_date - timedelta(days=duration)
    print(f"Fetching data for {contract.symbol} back to {target_start_date.date()}...")
    while end_date > target_start_date:
        end_date_str = end_date.strftime("%Y%m%d %H:%M:%S America/New_York")
        bars = ib.reqHistoricalData(contract, endDateTime=end_date_str, durationStr='5 D', barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True, formatDate=1)
        if not bars: break
        all_bars.insert(0, bars) 
        end_date = bars[0].date # The attribute is on the list object itself
    df = pd.DataFrame([b for chunk in all_bars for b in chunk])
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert('America/New_York')
    df = df.set_index('date')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'BarCount']
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 3. Connection, Data Fetching, and Backtest Execution
ib = IB()
try:
    ib.connect('127.0.0.1', 7496, clientId=39) # Using Client ID 38
    print("Connected to IBKR")
    contract = Stock('SPY', 'ARCA', 'USD')
    data = fetch_historical_data(ib, contract, duration=90, bar_size='1 min')
    ib.disconnect()
    print("Disconnected from IBKR")
    print(f"Fetched {len(data)} bars of data.")
    
    # --- FIX: Removed conflicting pandas-ta calls ---
    # data.ta.ema(...) 
    # data.ta.rsi(...)
    
    data.dropna(inplace=True) 
    
    # Run the Backtest using the correct strategy class name
    bt = Backtest(data, DayTradingStrategyFixedPL, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot(filename='SPY_DayTrading_V6_FixedPLExits_Performance')

except ConnectionRefusedError:
    print("Connection failed. Is TWS/IB Gateway running and configured correctly?")
except Exception as e:
    print(f"An error occurred: {e}")





