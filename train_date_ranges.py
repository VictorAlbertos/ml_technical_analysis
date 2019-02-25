import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from itertools import accumulate
from sklearn.metrics import *
from data_utils import set_date_as_index, compute_trend_based_on_trading_signal, compute_trading_decision, \
    compute_returns_long, compute_returns_short
from xgboost.sklearn import XGBRegressor


def build_model():
    return XGBRegressor(
        n_estimators=100,
        colsample_bytree=0.9,
        gamma=0,
        learning_rate=0.07,
        max_depth=7,
        min_child_weight=4,
        objective='reg:linear',
        reg_alpha=0.05,
        subsample=0.8
    )


def fit_and_predict_model(df_train, df_test, model):
    y_train = df_train.trading_signal
    y_test = df_test.trading_signal

    features = ['sma', 'macd', 'stochastic_k', 'stochastic_d', 'rsi', 'william_r']
    X_train = df_train[features]
    X_test = df_test[features]

    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)

    predictions = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, predictions)}")

    return predictions


symbol = 'S&P_500'

df = pd.read_csv(f'datasets/{symbol}.csv')
set_date_as_index(df)

start_date = df.loc['1975-01-01':].index[0]
end_date = df.loc['2015-01-01':].index[0]
current_date = start_date + relativedelta(months=12)

bought_price_long = None
bought_price_short = None
pcts_long = []
pcts_short = []

while current_date < end_date:
    start_date = current_date - relativedelta(months=2)
    df_train = df.loc[start_date:current_date - relativedelta(months=1) - relativedelta(days=1)]
    df_test = df.loc[current_date - relativedelta(months=1):current_date]

    df_test['trading_signal'] = fit_and_predict_model(df_train, df_test, build_model())

    compute_trend_based_on_trading_signal(df_test)
    compute_trading_decision(df_test)

    bought_price_long, pct_long = compute_returns_long(df_test, bought_price_long)
    pcts_long.append(sum(pct_long))

    bought_price_short, pct_short = compute_returns_short(df_test, bought_price_short)
    pcts_short.append(sum(pct_short))


    current_date = current_date + relativedelta(months=1)

print(f'long: {sum(pcts_long) * 100}%')
print(f'short: {sum(pcts_short) * 100}%')

plt.title('Acc returns pct')
plt.plot(list(accumulate(pcts_long)), label='Long pct')
plt.plot(list(accumulate(pcts_short)), label='Short pct')
plt.legend()
plt.show()