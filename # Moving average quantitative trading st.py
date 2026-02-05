# Moving average quantitative trading strategy

import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import datetime

# 1. Define the ATR Indicator Function
def ATR(data):
    high_low = data.High - data.Low
    high_close = np.abs(data.High - data.Close.shift())
    low_close = np.abs(data.Low - data.Close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(14).mean()

# 2. Define the improved Strategy Class for Day Trading
class DayTradingStrategy(Strategy):
    n1 = 20
    n2 = 50
    atr_multiplier_sl = 3.0
    risk_per_trade_pct = 0.01

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)
        self.atr = self.I(ATR, self.data)

    def next(self):
        current_time = self.data.index[-1].time()
        if current_time >= datetime.time(15, 50):
            if self.position:
                self.position.close()
            return

        if self.position:
            if crossover(self.sma2, self.sma1):
                 self.position.close()
        
        else:
            if crossover(self.sma1, self.sma2):
                current_atr_value = self.atr[-1]
                stop_loss_distance_dollars = current_atr_value * self.atr_multiplier_sl
                if stop_loss_distance_dollars == 0: return

                cash_available = self.equity
                position_size = (cash_available * self.risk_per_trade_pct) / stop_loss_distance_dollars
                position_size = int(position_size)

                if position_size > 0:
                    sl_price = self.data.Close[-1] - stop_loss_distance_dollars
                    self.buy(size=position_size, sl=sl_price)


# 3. Get your data (SPY) for a short period using minute data
data = yf.download('SPY', start='2024-10-01', end='2024-10-31', interval='1m')

# --- FIX IS HERE ---
# Explicitly reassign column names to ensure they are strings
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
# --- END FIX ---

data = data.dropna()

# 4. Instantiate the Backtest object
bt = Backtest(data, DayTradingStrategy,
              cash=10000,
              commission=.002,
              exclusive_orders=True)

# 5. Run the backtest
stats = bt.run()
print(stats)
bt.plot(filename='SPY_DayTrading_Adaptive_Performance')




