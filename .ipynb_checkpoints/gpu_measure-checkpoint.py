import cupy as cp
import numpy as np
import pandas as pd
import time
import cudf as cf
import cuml as ml
import rmm

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import rstl

from time import perf_counter
from cuml.ensemble import RandomForestRegressor
from cuml.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

def lowess_cp_inline(x, y, f=2. / 3., it=3, dl = 1.0):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = x.shape[0]
    r = cp.int(cp.ceil(f * n))
    h = cp.zeros(n)
    for i in range(n):
        h[i] = cp.sort(cp.abs(x - x[i]))[r]
    yest = cp.zeros(n)
    delta = cp.repeat(cp.array(dl), n)
    for iteration in range(it):
        for i in range(n):
            row = i
            weights = delta * cp.power(1 - cp.power(cp.clip(cp.abs((x[row, None] - x[None, :])/h[row]), 0.0, 1.0), 3.0), 3.0)
            b = cp.array(cp.array([cp.sum(weights * y), cp.sum(weights * y * x)]))
            A = cp.array(cp.array([cp.array([cp.sum(weights), cp.sum(weights * x)]), cp.array([cp.sum(weights * x), cp.sum(weights * x * x)])]))
            beta = cp.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = cp.float32(cf.Series(cp.abs(residuals)).median())
        delta = cp.clip(residuals / (6.0 * s), -1, 1)
        delta = cp.power((1 - cp.power(delta, 2.0)), 2)
    return yest

def IsSummer_func(x, out):
    for i, a in enumerate(x):
        if a >= 6 and a <= 8:
            out[i] = 1
        else:
            out[i] = 0
            
def heatint_season_func(x, y, out):
    for i, (a, e) in enumerate(zip(x, y)):
        if a < 13 and e == 1:
            out[i] = 1
        else:
            out[i] = 0
            
"""
Computes MAPE
"""
def mean_absolute_percentage_error(y_true: cp.array, y_pred: cp.array) -> cp.float32:
    return cp.mean(cp.abs((y_true - y_pred) / y_true)) * 100

"""
Computes SMAPE
"""
def symetric_mean_absolute_percentage_error(y_true: cp.array, y_pred: cp.array) -> cp.float32:
    return cp.mean(cp.abs((y_pred - y_true) / ((cp.abs(y_true) + cp.abs(y_pred))/2.0))) * 100

"""
Computes MAE, MSE, MAPE, SMAPE, R2
"""
def compute_metrics(df: cf.DataFrame) -> cf.Series:
    y_true, y_pred = df['y_true'].values, df['y_pred'].values
    return compute_metrics_raw(y_true, y_pred)

def compute_metrics_raw(y_true: cf.Series, y_pred: cf.Series) -> cf.Series:
    mae, mse, mape, smape = mean_absolute_error(y_true=y_true, y_pred=y_pred), mean_squared_error(y_true=y_true, y_pred=y_pred), mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred), symetric_mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    return cf.Series(cp.array([mae, mse, mape, smape]), index=['MAE', 'MSE', 'MAPE', 'SMAPE'])
#     return cf.DataFrame.from_records([{'MAE': mae, 'MSE': mse, 'MAPE': mape, 'SMAPE': smape}], index=[0])

def experiment_with_n_diff(n_diff=1):
    start_data = perf_counter()
    df = cf.read_csv('data/ppnet_metar_v7.csv',  sep=';', index_col=0)
    # TODO: Odkomentovat pro repeat datasetu 10x
    # df = cf.concat([df]*10, ignore_index=True)
    start = perf_counter()
    start_full = perf_counter()

    df = df[df.Year < 2019]
    df['TestSet'] = 0
    df.loc[df.Year == 2018, 'TestSet'] = 1
    X, y = df.drop('Consumption', axis=1), df.Consumption

    y_diff = y.diff(n_diff).dropna()
    y_diff_index = y_diff.index
    X['Consumption_diff_1'] = y_diff
    lags = cp.array([24, 25, 35, 36, 37, 47, 48, 49, 71,72,73, 95, 96, 97, 119, 120, 121, 143,144,145, 168, 167, 169])
    lags_arr = cp.arange(n_diff, 24)
    lags_con = cp.concatenate((lags, lags_arr))
    lags = lags_con

    for x in lags:
        X[f'Consumption_lag_{x}'] = y.shift(x)
        X[f'Temperature_lag_{x}'] = X['Temperature'].shift(x)
        X[f'Consumption_diff_1_lag_{x}'] = X['Consumption_diff_1'].shift(x)
        X[f'Humidity_lag_{x}'] = X['Humidity'].shift(x)
        X[f'Cena_lag_{x}'] = X['Cena_bfill'].shift(x)
        X[f'Windspeed_lag_{x}'] = X['Wind_speed'].shift(x)
        X[f'Pressure_lag_{x}'] = X['Pressure'].shift(x)

    X['Day_of_week_sin'] = cp.sin(2 * cp.pi * X['Day_of_week']/7.0)
    X['Day_of_week_cos'] = cp.cos(2 * cp.pi * X['Day_of_week']/7.0)
    X['Month_sin'] = cp.sin(2 * cp.pi * X['Month']/12.0)
    X['Month_cos'] = cp.cos(2 * cp.pi * X['Month']/12.0)
    X['Hour_sin'] = cp.sin(2 * cp.pi * X['Hour']/23.0)
    X['Hour_cos'] = cp.cos(2 * cp.pi * X['Hour']/23.0)
    X['Day_sin'] = cp.sin(2 * cp.pi * X['Day']/31.0)
    X['Day_cos'] = cp.cos(2 * cp.pi * X['Day']/31.0)

    cp.cuda.Stream.null.synchronize()
    end = perf_counter()
    print(f'Time before STL: {end - start}')
    start = perf_counter()

    x = cp.array([i for i in range(len(y_diff.values))])
    f = 0.01
    period = 24*7
    trend = lowess_cp_inline(x, y_diff.values, f, it=1, dl=0.01)
    detrended = y_diff.values - trend
    period_averages = cp.array([cp.mean(detrended[i::period]) for i in range(period)])
    period_averages -= cp.mean(period_averages)
    seasonal = cp.tile(period_averages, len(y_diff.values) // period + 1)[:len(y_diff.values)]
    resid = detrended - seasonal
    residual_trend = resid + trend

    X['Seasonal'] = cp.concatenate([cp.array([cp.nan]*n_diff), seasonal])
    X['Residual'] = cp.concatenate([cp.array([cp.nan]*n_diff), residual_trend])
    X['Trend'] = cp.concatenate([cp.array([cp.nan]*n_diff), trend])

    cp.cuda.Stream.null.synchronize()
    end = perf_counter()
    print(f'Time computing STL: {end - start}')
    start = perf_counter()

    for x in lags:
        X[f'Seasonal_lag_{x}'] = X['Seasonal'].shift(x)
        X[f'Residual_lag_{x}'] = X['Residual'].shift(x)
        X[f'Residual_lag_inverse_{x}'] = X['Residual'].shift(x)*-1

    X['IsSummer'] = 0
    IsSummer_final = X.apply_rows(IsSummer_func, incols={'Month':'x'}, outcols={'out': cp.int}, kwargs={}).out
    X['IsMiddleOfTheDay'] = 0
    X.loc[X.Hour.applymap(lambda x: x >= 13 and x <= 15), 'IsMiddleOfTheDay'] = 1 
    X['IsHeatingSeason'] = 1
    X.loc[X.Month.applymap(lambda x: x >= 6 and x <= 8), 'IsHeatingSeason'] = 0
    heat_final = X.apply_rows(heatint_season_func, incols={'Temperature':'x', 'IsHeatingSeason':'y'}, outcols={'out': cp.int}, kwargs={}).out
    X['IsHeatingSeason'] = heat_final
    X['IsWeekend'] = 0
    X.loc[X.Day_of_week.applymap(lambda x: x >= 6 and x <= 7), 'IsWeekend'] = 1
    X['IsStartOfGasTradeDay'] = 0
    X.loc[X.Hour == 6, 'IsStartOfGasTradeDay'] = 1

    y = X.Residual
    X_train, X_test, y_train, y_test = X[X.TestSet == 0], X[X.TestSet == 1], y[X.TestSet == 0], y[X.TestSet == 1]

    arr_drop_columns = cf.Series(['Cena',
     'Cena_bfill',
     'Clouds_high_m',
     'Clouds_high_text',
     'Clouds_low_m',
     'Clouds_low_text',
     'Clouds_medium_m',
     'Clouds_medium_text',
     'Consumption_diff_1',
     'Datetime.1',
     'Dewpoint',
     'Humidity',
     'IsMissing',
     'Mnozstvi',
     'Mnozstvi_bfill',
     'Phenomena',
     'Pressure',
     'Pressure2',
     'PreviousTemp_lag24',
     'PreviousTemp_lag48',
     'Recent_phenomena',
     'Residual',
     'Temperature',
     'TestSet',
     'Trend',
     'Visibility',
     'Wind_direction',
     'Wind_speed',
     'Year'])

    X_train_selected_features = X_train.drop(arr_drop_columns, axis=1)
    X_test_selected_features = X_test.drop(arr_drop_columns, axis=1)
    X_train_selected_features_nona = X_train_selected_features.dropna()
    X_test_selected_features_nona = X_test_selected_features.dropna()
    y_train_no_na = y_train.dropna()
    y_test_no_na = y_test.dropna()
    y_train_no_na = y_train_no_na[y_train_no_na.index.isin(X_train_selected_features_nona.index)]
    X_test_selected_features_nona = X_test_selected_features_nona[X_test_selected_features_nona.index.isin(y_test_no_na.index)]

    cp.cuda.Stream.null.synchronize()
    end = perf_counter()
    print(f'Time second phase preprocess: {end - start}')
    
    start = perf_counter()
    alg = RandomForestRegressor(n_estimators=5, accuracy_metric='mse', max_features=1.0, seed=14)
    alg.fit(X_train_selected_features_nona, y_train_no_na)
    y_pred = alg.predict(X_test_selected_features_nona)
    ps_y_pred = y_pred
    ps_y_pred.index = y_test_no_na.index
    y_pred_tmp = ps_y_pred + X[X.TestSet == 1].Seasonal
    df_tmp = (df.Consumption.shift(n_diff)[df.TestSet == 1] + y_pred_tmp)
    orig_data = df.loc[df.TestSet == 1, 'Consumption']
    df_res_hourly = cf.DataFrame({'y_true': orig_data, 'y_pred': df_tmp})
    cp.cuda.Stream.null.synchronize()
    end_full = perf_counter()
    print(f'Time for ML phase: {end_full - start}')
    print(f'Full pipeline: {end_full - start_full}')
    print(f'Full pipeline (with data load): {end_full - start_data}')

if __name__ == "__main__":
    for i in range(5):
        print(f"Experiment #{i}")
        experiment_with_n_diff(1)