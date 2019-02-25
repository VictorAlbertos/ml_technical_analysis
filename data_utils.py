import pandas as pd


def compute_returns_long(df, bought_price=None):
    pct_long = []

    for index, row in df.iterrows():
        if row.trading_decision == 'buy' and bought_price is None:
            bought_price = row.close
        elif row.trading_decision == 'sell' and bought_price is not None:
            pct_long.append((row.close - bought_price) / bought_price)
            bought_price = None

    return bought_price, pct_long


def compute_returns_short(df, bought_price=None):
    pct_short = []

    for index, row in df.iterrows():
        if row.trading_decision == 'sell' and bought_price is None:
            bought_price = row.close
        elif row.trading_decision == 'buy' and bought_price is not None:
            pct_short.append((row.close - bought_price) / bought_price * -1)
            bought_price = None

    return bought_price, pct_short


def set_date_as_index(df):
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)


def compute_trend_based_on_trading_signal(df):
    trading_signal_mean = 0.5
    df['trend'] = df['trading_signal'].apply(lambda ts: 'up' if ts > trading_signal_mean else 'down')


def compute_trading_decision(df):
    df['buy'] = (df.trend == 'down') & (df.trend.shift(1) == 'up')
    df['sell'] = (df.trend == 'up') & (df.trend.shift(1) == 'down')

    def map_buy_sell(x):
        buy = x[0]
        sell = x[1]
        if buy:
            return 'buy'
        elif sell:
            return 'sell'
        else:
            return ''

    df['trading_decision'] = df[['buy', 'sell']].apply(map_buy_sell, axis=1)
    df.drop(['buy', 'sell'], axis=1, inplace=True)
