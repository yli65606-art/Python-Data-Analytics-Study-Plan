# Bollinger Bands strategy （day trade）

from ib_insync import IB, Stock, util
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import nest_asyncio
import pytz 
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import datetime as dt 
import pandas_ta as ta

nest_asyncio.apply()

# 1. Define the Bollinger Bands Strategy Class
class BollingerBandsStrategy(Strategy):
    bb_period = 20
    bb_dev = 2 # Standard deviations

    def init(self):
        pass


    def next(self):
        current_close = self.data.Close[-1]

        # End of day exit (Day trading rule remains)
        current_time = self.data.index[-1].time()
        if current_time >= dt.time(15, 50, 0):
            if self.position: self.position.close(); return

        # Access the bands via the full appended column names
        upper_band = self.data['BBU_20_2.0_2.0'][-1]
        lower_band = self.data['BBL_20_2.0_2.0'][-1]

        # Exit existing long position if price crosses above the upper band (overbought)
        if self.position.is_long:
            if current_close > upper_band:
                 self.position.close()

        # Entry logic: Buy when price crosses below the lower band (oversold)
        else:
            if current_close < lower_band:
                # Buy signal: Go long with default TP/SL managed internally
                # (TP and SL percentage can be added here as arguments if desired)
                self.buy()

# 2. Define the data fetching function (no changes needed)
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
        end_date = bars[0].date
    df = pd.DataFrame([b for chunk in all_bars for b in chunk])
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert('America/New_York')
    df = df.set_index('date')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'BarCount']
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 3. Connection, Data Fetching, and Backtest Execution
ib = IB()
try:
    ib.connect('127.0.0.1', 7496, clientId=45) # Using Client ID 39
    print("Connected to IBKR")
    contract = Stock('SPY', 'ARCA', 'USD')
    data = fetch_historical_data(ib, contract, duration=90, bar_size='1 min')
    ib.disconnect()
    print("Disconnected from IBKR")
    print(f"Fetched {len(data)} bars of data.")
    
    # --- FIX: Calculate indicators using pandas-ta *before* backtest ---
    data.ta.bbands(length=20, std=2, append=True)
    # The column names generated are BBU_20_2, BBM_20_2, BBL_20_2
    data.dropna(inplace=True) 

    # Use the new Bollinger Bands Strategy class V2
    bt = Backtest(data, BollingerBandsStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot(filename='SPY_DayTrading_BollingerBandsV2_Performance')

except ConnectionRefusedError:
    print("Connection failed. Is TWS/IB Gateway running and configured correctly?")
except Exception as e:
    print(f"An error occurred: {e}")