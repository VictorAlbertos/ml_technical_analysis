import pandas as pd
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
from pyti.double_smoothed_stochastic import double_smoothed_stochastic
from pyti.relative_strength_index import relative_strength_index

from data_utils import compute_trading_decision, compute_trend_based_on_trading_signal, set_date_as_index


def rename_adj_close(df):
    df.close = df.adj_close
    df.drop(['adj_close'], axis=1, inplace=True)


def compute_simple_moving_average(df):
    df['sma'] = sma(df.close, 15)


def compute_moving_average_convergence_divergence(df):
    df['macd'] = moving_average_convergence_divergence(df.close, 12, 26)


def compute_stochastic_kd(df):
    df['stochastic_k'] = double_smoothed_stochastic(df.close, 14)
    df['stochastic_d'] = df.stochastic_k.rolling(window=3).mean().shift(1)


def compute_relative_strength_index(df):
    df['rsi'] = relative_strength_index(df.close, 14)


def compute_william_r(df):
    period = 14
    highest_price_period = df.high.rolling(window=period).max().shift(1)
    lowest_price_period = df.low.rolling(window=period).min().shift(1)
    df['william_r'] = (highest_price_period - df.close) / (highest_price_period - lowest_price_period) * 100


def compute_trend(df):
    period = 5
    df['up'] = (df.close > df.sma).rolling(period).sum().shift(1) == period
    df['down'] = (df.close < df.sma).rolling(period).sum().shift(1) == period

    def map_up_down(x):
        up = x[0]
        down = x[1]
        if up:
            return 'up'
        elif down:
            return 'down'
        else:
            return 'none'

    df['trend'] = df[['up', 'down']].apply(map_up_down, axis=1)
    df.drop(['up', 'down'], axis=1, inplace=True)


def compute_trading_signal(df):
    df_reversed = df.iloc[::-1]

    cp = df_reversed.close
    min_cp = df_reversed.close.rolling(5).min()
    max_cp = df_reversed.close.rolling(5).max()
    df['trading_signal'] = ((cp - min_cp) / (max_cp - min_cp)) * 0.5

    def apply_ts_factor_trend(x):
        ts = x[0]
        trend = x[1]
        if trend == 'up':
            return ts + 0.5
        else:
            return ts

    df['trading_signal'] = df[['trading_signal', 'trend']].apply(apply_ts_factor_trend, axis=1)


symbol = 'AAPL'
df = pd.read_csv(f'raw/{symbol}.csv')
df.dropna(inplace=True)

rename_adj_close(df)

set_date_as_index(df)

compute_simple_moving_average(df)

compute_moving_average_convergence_divergence(df)

compute_stochastic_kd(df)

compute_relative_strength_index(df)

compute_william_r(df)

compute_trend(df)

compute_trading_signal(df)

compute_trend_based_on_trading_signal(df)

compute_trading_decision(df)

df.dropna(subset=['stochastic_d', 'trading_signal'], inplace=True)
df.to_csv(f'datasets/{symbol}.csv')
