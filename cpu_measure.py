import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from evaluation_metrics import compute_metrics, compute_metrics_csv, mean_absolute_percentage_error, symetric_mean_absolute_percentage_error

def lowess_inline(x, y, f=2. / 3., it=3, dl=1):
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
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    # w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    # w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.repeat(dl, n)
    for iteration in range(it):
        for i in range(n):
            row = i
            weights = delta * np.power(1 - np.power(np.clip(np.abs((x[row, None] - x[None, :])/h[row]), 0.0, 1.0), 3.0), 3.0)
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

def experiment_with_n_diff(n_diff=1):
    start_data = perf_counter()
    df = pd.read_csv('data/ppnet_metar_v7.csv',  sep=';', index_col=0)
    start = perf_counter()
    start_full = perf_counter()
    df = df[df.Year < 2019]
    df['TestSet'] = 0
    df.loc[df.Year == 2018, 'TestSet'] = 1
    X, y = df.drop('Consumption', axis=1), df.Consumption
    
    y_diff = y.diff(n_diff).dropna()
    y_diff_index = y_diff.index
    X.loc[:, 'Consumption_diff_1'] = y_diff
    lags = [24, 25, 35, 36, 37, 47, 48, 49, 71,72,73, 95, 96, 97, 119, 120, 121, 143,144,145, 168, 167, 169]
    for lag in range(n_diff, 24):
        lags.insert(0, lag)
        
    for x in lags:
        X.loc[:, f'Consumption_lag_{x}'] = y.shift(x)
        X.loc[:, f'Temperature_lag_{x}'] = X.loc[:, 'Temperature'].shift(x)
        X.loc[:, f'Consumption_diff_1_lag_{x}'] = X.loc[:, 'Consumption_diff_1'].shift(x)
        X.loc[:, f'Humidity_lag_{x}'] = X.loc[:, 'Humidity'].shift(x)
        X.loc[:, f'Cena_lag_{x}'] = X.loc[:, 'Cena_bfill'].shift(x)
        X.loc[:, f'Windspeed_lag_{x}'] = X['Wind_speed'].shift(x)
        X.loc[:, f'Pressure_lag_{x}'] = X['Pressure'].shift(x)
        
    X.loc[:, 'Day_of_week_sin'] = np.sin(2 * np.pi * X['Day_of_week']/X['Day_of_week'].max())
    X.loc[:, 'Day_of_week_cos'] = np.cos(2 * np.pi * X['Day_of_week']/X['Day_of_week'].max())
    X.loc[:, 'Month_sin'] = np.sin(2 * np.pi * X['Month']/X['Month'].max())
    X.loc[:, 'Month_cos'] = np.cos(2 * np.pi * X['Month']/X['Month'].max())
    X.loc[:, 'Hour_sin'] = np.sin(2 * np.pi * X['Hour']/X['Hour'].max())
    X.loc[:, 'Hour_cos'] = np.cos(2 * np.pi * X['Hour']/X['Hour'].max())
    X.loc[:, 'Day_sin'] = np.sin(2 * np.pi * X['Day']/X['Day'].max())
    X.loc[:, 'Day_cos'] = np.cos(2 * np.pi * X['Day']/X['Day'].max())
    
#     stl_decomp = rstl.STL(y_diff, freq=24*7, robust=False, s_window='periodic')
#     trend, seasonal, residual, weights = stl_decomp.trend, stl_decomp.seasonal, stl_decomp.remainder, stl_decomp.weights
#     residual_trend = residual + trend
    end = perf_counter()
    print(f'Time before STL: {end - start}')
    start = perf_counter()
    
    x = np.array([i for i in range(len(y_diff.values))])
    f = 0.01
    period = 24*7

    #TODO: Zmenit iterace na 3
    trend = lowess_inline(x, y_diff.values, f, it=1, dl=0.01)
    detrended = y_diff.values - trend
    period_averages = np.array([np.mean(detrended[i::period]) for i in range(period)])
    period_averages -= np.mean(period_averages)
    seasonal = np.tile(period_averages, len(y_diff.values) // period + 1)[:len(y_diff.values)]
    resid = detrended - seasonal
    residual_trend = resid + trend
    
    X.loc[:, f'Seasonal'] = pd.Series((np.concatenate([[np.nan]*n_diff, seasonal])), index=X.index)
    X.loc[:, f'Residual'] = pd.Series((np.concatenate([[np.nan]*n_diff, residual_trend])), index=X.index)
    X.loc[:, f'Trend'] = pd.Series((np.concatenate([[np.nan]*n_diff,trend])), index=X.index)
    
    end = perf_counter()
    print(f'Time computing STL: {end - start}')
    start = perf_counter()
    
    for x in lags:
        X.loc[:, f'Seasonal_lag_{x}'] = X['Seasonal'].shift(x)
        X.loc[:, f'Residual_lag_{x}'] = X['Residual'].shift(x)
        X.loc[:, f'Residual_lag_inverse_{x}'] = X['Residual'].shift(x)*-1
    X.index = pd.DatetimeIndex(X.index)
    X['IsSummer'] = 0
    X.loc[X.Month.between(6, 8), 'IsSummer'] = 1
    X['IsMiddleOfTheDay'] = 0
    X.loc[X.Hour.between(13, 15), 'IsMiddleOfTheDay'] = 1

    X['IsHeatingSeason'] = 1
    X.loc[X.Month.between(6, 8), 'IsHeatingSeason'] = 0
    heat_final = X.apply(lambda x: 1 if x['Temperature'] < 13 and x['IsHeatingSeason'] == 1 else 0, axis=1)
    X['IsHeatingSeason'] = heat_final

    X['IsWeekend'] = 0
    X.loc[X.Day_of_week.between(6, 7), 'IsWeekend'] = 1

    X['IsStartOfGasTradeDay'] = 0
    X.loc[X.Hour == 6, 'IsStartOfGasTradeDay'] = 1

    X['IsMonthStart'] = [1 if x else 0 for x in X.index.is_month_start]
    X['IsMonthEnd'] = [1 if x else 0 for x in X.index.is_month_end]
    X['Quarter'] = X.index.quarter
    y = X.Residual
        
    X_train, X_test, y_train, y_test = X[X.TestSet == 0], X[X.TestSet == 1], y[X.TestSet == 0], y[X.TestSet == 1]
    arr_drop_columns = ['Residual_diff_from_midnight', 'Year','Temperature', 'Pressure', 'Pressure2', 'Humidity', 'Wind_direction', 'Wind_speed', 'Phenomena', 'Recent_phenomena', 'Visibility', 'Dewpoint', 'Datetime.1', 'Clouds_low_text', 'Clouds_low_m', 'Clouds_medium_text', 'Clouds_medium_m', 'Clouds_high_text', 'Clouds_high_m', 'IsMissing', 'PreviousTemp_lag24', 'PreviousTemp_lag48', 'Trend', 'Mnozstvi', 'Mnozstvi_bfill', 'Cena', 'Cena_bfill', 'TestSet', 'Consumption_diff_1', 
       'Residual', 'Residual_lag_minus_1', 'Consumption_diff_1_lag_minus_1',
       'Residual_lag_minus_2', 'Consumption_diff_1_lag_minus_2',
       'Residual_lag_minus_3', 'Consumption_diff_1_lag_minus_3',
       'Residual_lag_minus_4', 'Consumption_diff_1_lag_minus_4',
       'Residual_lag_minus_5', 'Consumption_diff_1_lag_minus_5',
       'Residual_lag_minus_6', 'Consumption_diff_1_lag_minus_6',
       'Residual_lag_minus_7', 'Consumption_diff_1_lag_minus_7',
       'Residual_lag_minus_8', 'Consumption_diff_1_lag_minus_8',
       'Residual_lag_minus_9', 'Consumption_diff_1_lag_minus_9',
       'Residual_lag_minus_10', 'Consumption_diff_1_lag_minus_10',
       'Residual_lag_minus_11', 'Consumption_diff_1_lag_minus_11',
       'Residual_lag_minus_12', 'Consumption_diff_1_lag_minus_12',
       'Residual_lag_minus_13', 'Consumption_diff_1_lag_minus_13',
       'Residual_lag_minus_14', 'Consumption_diff_1_lag_minus_14',
       'Residual_lag_minus_15', 'Consumption_diff_1_lag_minus_15',
       'Residual_lag_minus_16', 'Consumption_diff_1_lag_minus_16',
       'Residual_lag_minus_17', 'Consumption_diff_1_lag_minus_17',
       'Residual_lag_minus_18', 'Consumption_diff_1_lag_minus_18',
       'Residual_lag_minus_19', 'Consumption_diff_1_lag_minus_19',
       'Residual_lag_minus_20', 'Consumption_diff_1_lag_minus_20',
       'Residual_lag_minus_21', 'Consumption_diff_1_lag_minus_21',
       'Residual_lag_minus_22', 'Consumption_diff_1_lag_minus_22',
       'Residual_lag_minus_23', 'Consumption_diff_1_lag_minus_23',
       'Residual_lag_minus_24', 'Consumption_diff_1_lag_minus_24', 'ResetSignal']
    X_train_selected_features = X_train.drop(arr_drop_columns, axis=1, errors='ignore')
    X_test_selected_features = X_test.drop(arr_drop_columns, axis=1, errors='ignore')
    
    X_train_selected_features_nona = X_train_selected_features.dropna()
    X_test_selected_features_nona = X_test_selected_features.dropna()
    
    y_train_no_na = y_train.dropna()
    y_test_no_na = y_test.dropna()
    y_train_no_na = y_train_no_na[y_train_no_na.index.isin(X_train_selected_features_nona.index)]
    X_test_selected_features_nona = X_test_selected_features_nona[X_test_selected_features_nona.index.isin(y_test_no_na.index)]
    
    end = perf_counter()
    print(f'Time second phase preprocess: {end - start}')
    start = perf_counter()
    
    
    alg = RandomForestRegressor(n_estimators=5, random_state=14)
    alg.fit(X_train_selected_features_nona, y_train_no_na)
    y_pred = alg.predict(X_test_selected_features_nona)
    ps_y_pred = y_pred
    # ps_y_pred.index = y_test_no_na.index
    y_pred_tmp = ps_y_pred + X[X.TestSet == 1].Seasonal
    df_tmp = (df.Consumption.shift(n_diff)[df.TestSet == 1] + y_pred_tmp)
    orig_data = df.loc[df.TestSet == 1, 'Consumption']
    df_res_hourly = pd.DataFrame({'y_true': orig_data, 'y_pred': df_tmp})
    end = perf_counter()
    print(f'Time for ML phase: {end - start}')
    end_full = perf_counter()
    print(f'Full pipeline: {end_full - start_full}')
    print(f'Full pipeline (with data load): {end_full - start_data}')
    print(compute_metrics(df_res_hourly))


if __name__ == "__main__":
    for i in range(5):
        print(f"Experiment #{i}")
        experiment_with_n_diff(1)