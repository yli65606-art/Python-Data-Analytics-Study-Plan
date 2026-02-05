# Simple EMA based self-designed trading strategy

# ==================================================
# EMA Swing Trading Strategy on SPY (Daily)
# ==================================================

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# ==================================================
# Strategy Definition
# ==================================================
class EMASwingStrategy(Strategy):
    ema_fast = 9
    ema_mid = 20
    ema_slow = 50
    stop_loss_pct = 2.0  # 2%

    def init(self):
        # Calculate EMAs manually within init() using pandas Series for compatibility
        self.ema9 = self.I(
            lambda x: pd.Series(x).ewm(span=self.ema_fast, adjust=False).mean(),
            self.data.Close
        )
        self.ema20 = self.I(
            lambda x: pd.Series(x).ewm(span=self.ema_mid, adjust=False).mean(),
            self.data.Close
        )
        self.ema50 = self.I(
            lambda x: pd.Series(x).ewm(span=self.ema_slow, adjust=False).mean(),
            self.data.Close
        )

    def next(self):
        price = self.data.Close[-1]

        # Exit: EMA9 crosses below EMA20
        # Use crossover() with indicator objects
        if self.position:
            if crossover(self.ema20, self.ema9):
                self.position.close()
            return

        # Entry: EMA9 crosses above EMA50
        # Use crossover() with indicator objects
        if crossover(self.ema9, self.ema50):
            self.buy(
                # --- FIX: Remove size=1.0 or use size=None for full capital ---
                # size=1.0, <-- REMOVE THIS LINE
                sl=price * (1 - self.stop_loss_pct / 100)
            )


# ==================================================
# Data Loading
# ==================================================
def load_data(symbol="SPY", start="2020-01-01"):
    df = yf.download(
        symbol,
        start=start,
        auto_adjust=False, # Required to keep Open, High, Low, Close, Volume
        progress=False
    )

    # Fix yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns are present and clean data
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    return df


# ==================================================
# Run Backtest
# ==================================================
initial_cash = 10_000 # Start with $10,000
data = load_data("SPY", start="2020-01-01")

bt = Backtest(
    data,
    EMASwingStrategy,
    cash=initial_cash,
    commission=0.001, # 0.1% commission per trade
    exclusive_orders=True,
    finalize_trades=True
)

stats = bt.run()

print("\n=== Strategy Summary ===")
print(stats[[
    'Return [%]',
    'CAGR [%]',
    'Sharpe Ratio',
    'Max. Drawdown [%]',
    '# Trades',
    'Win Rate [%]'
]])


# ==================================================
# Equity Curve vs Buy & Hold
# ==================================================

# Strategy equity curve
equity_curve = stats['_equity_curve']['Equity']
equity_curve.index = data.index[:len(equity_curve)]

# Buy & Hold curve calculation
market_curve = (data['Close'] / data['Close'].iloc) * initial_cash

# Align indices for plotting
market_curve = market_curve.reindex(equity_curve.index)

# Plot actual equity values
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label="EMA Strategy (Actual Equity)")
plt.plot(market_curve, label="Buy & Hold SPY (Actual Equity)")
plt.title(f"Performance Comparison (Base = ${initial_cash:,.0f})")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True)
plt.show()

# Also display the interactive HTML plot provided by the library
bt.plot(filename='SPY_EMA_Swing_Performance')







