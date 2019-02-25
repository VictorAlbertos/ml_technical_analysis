import pandas as pd
import matplotlib.pyplot as plt
from itertools import accumulate

from data_utils import set_date_as_index, compute_trend_based_on_trading_signal, compute_trading_decision, \
    compute_returns_long, compute_returns_short


def plot_trading_signal(df):
    plt.plot(df.close)

    buy_dates = df[df.trading_decision == 'buy']
    plt.plot_date(buy_dates.index, buy_dates.close, color='green')

    sell_dates = df[df.trading_decision == 'sell']
    plt.plot_date(sell_dates.index, sell_dates.close, color='red')

    plt.show()


def plot_returns(df):
    _, pct_long = compute_returns_long(df)
    print(f'long: {sum(pct_long) * 100}%')

    _, pct_short = compute_returns_short(df)
    print(f'short: {sum(pct_short) * 100}%')

    plt.title('Returns pct')
    plt.plot(pct_long, label='Long pct')
    plt.plot(pct_short, label='Short pct')
    plt.legend()
    plt.show()

    plt.title('Acc returns pct')
    plt.plot(list(accumulate(pct_long)), label='Long pct')
    plt.plot(list(accumulate(pct_short)), label='Short pct')
    plt.legend()
    plt.show()


symbol = 'S&P_500'
model = 'p_t_s_lstm'

df = pd.read_csv(f'datasets/{symbol}.csv')

df.trading_signal = df[model]
df.drop([model], inplace=True, axis=1)
df.dropna(subset=['trading_signal'], inplace=True)

set_date_as_index(df)
compute_trend_based_on_trading_signal(df)
compute_trading_decision(df)

plot_trading_signal(df)
plot_returns(df)
