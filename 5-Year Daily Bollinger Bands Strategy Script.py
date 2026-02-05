# 5-Year Daily Bollinger Bands Strategy Script

import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import datetime as dt 
import pandas_ta as ta
import matplotlib.pyplot as plt


# 1. Define the Bollinger Bands Strategy Class
class BollingerBandsDailyStrategy(Strategy):
    bb_period = 20
    bb_dev = 2 

    def init(self):
        pass

    def next(self):
        current_close = self.data.Close[-1]
        
        upper_band = self.data['BBU_20_2.0_2.0'][-1]
        lower_band = self.data['BBL_20_2.0_2.0'][-1]

        if self.position.is_long:
            if current_close > upper_band:
                 self.position.close()

        else:
            if current_close < lower_band:
                self.buy() 


# 2. Fetch the 5 years of data using yfinance
symbol = 'SPY'
start_date_str = '2020-12-22' 
end_date = dt.date.today() - dt.timedelta(days=1)
end_date_str = end_date.strftime('%Y-%m-%d')
initial_cash = 10000 # Store the initial cash value here

data = yf.download(symbol, start=start_date_str, end=end_date_str)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
data.columns = [col.capitalize() for col in data.columns]
data = data.dropna()

print(f"Fetched {len(data)} daily bars of data.")

# 3. Prepare Indicators and Run the Backtest
data.ta.bbands(length=20, std=2, append=True)
data.dropna(inplace=True) 

# Pass the initial cash value to the Backtest constructor
bt = Backtest(data, BollingerBandsDailyStrategy, cash=initial_cash, commission=.002, exclusive_orders=True)
stats = bt.run() 

print(stats)
# This generates the interactive HTML plot which has the comparison built-in
bt.plot(filename='SPY_Daily_BollingerBands_SpecificDates_Performance')


# 4. Generate a static matplotlib comparison plot
plt.figure(figsize=(12, 6))

# Use the hardcoded initial cash value
market_returns = (1 + data['Close'].pct_change()).cumprod() * initial_cash
strategy_equity = stats['_equity_curve']['Equity'] 

market_returns = market_returns.dropna()
strategy_equity = strategy_equity.dropna()

plt.plot(market_returns.index, market_returns.values.astype(float), label=f'{symbol} Buy & Hold', color='blue')
plt.plot(strategy_equity.index, strategy_equity.values.astype(float), label='Strategy Equity', color='red', linestyle='--')

plt.title(f'{symbol} Strategy vs Market Performance')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True)
plt.show()






