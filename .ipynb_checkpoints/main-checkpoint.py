import cupy as cp
import numpy as np
import pandas as pd
import time
import cudf as cf
import cuml as ml

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from evaluation_metrics import compute_metrics, compute_metrics_csv, mean_absolute_percentage_error, symetric_mean_absolute_percentage_error

from statsmodels.tsa.stattools import acf, pacf, ccf, ccovf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
import rstl

from time import perf_counter

def cuda_test_batch():
    print(cp.__version__)
    print(cf.__version__)
    print(ml.__version__)

    s = time.time()
    x_cpu = np.ones((1000,1000,1000))
    e = time.time()
    print(e - s)

    s = time.time()
    x_gpu = cp.ones((1000,1000,1000))
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print(e - s)
    
    ### Numpy and CPU
    s = time.time()
    x_cpu *= 5
    x_cpu *= x_cpu
    x_cpu += x_cpu
    e = time.time()
    print(e - s)

    ### CuPy and GPU
    s = time.time()
    x_gpu *= 5
    x_gpu *= x_gpu
    x_gpu += x_gpu
    cp.cuda.Stream.null.synchronize()
    e = time.time()
    print(e - s)

def fill_seasonal(month, day, hour, seasonal_dict):
    season = seasonal_dict[month]
    return season[(season.Day_of_week == day) & (season.Hour == hour)].Seasonal.values[0]

def experiment_with_n_diff(n_diff=24):
    alg = RandomForestRegressor(n_estimators=10, n_jobs=8, random_state=14)
    df = pd.read_csv('data/ppnet_metar_v7.csv',  sep=';', index_col=0)
    df = df[df.Year < 2019].copy()
    df['TestSet'] = 0
    df.loc[df.Year == 2018, 'TestSet'] = 1
    X, y = df.loc[:, df.columns != 'Consumption'], df.Consumption
    
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
    stl_decomp = rstl.STL(y_diff, freq=24*7, robust=False, s_window='periodic')
    trend, seasonal, residual, weights = stl_decomp.trend, stl_decomp.seasonal, stl_decomp.remainder, stl_decomp.weights
    residual_trend = residual + trend
    X.loc[:, f'Seasonal'] = pd.Series((np.concatenate([[np.nan]*n_diff, seasonal])), index=X.index)
    X.loc[:, f'Residual'] = pd.Series((np.concatenate([[np.nan]*n_diff, residual_trend])), index=X.index)
    X.loc[:, f'Trend'] = pd.Series((np.concatenate([[np.nan]*n_diff,trend])), index=X.index)
    
    seasonal_dict = {k:X[(X.TestSet == 0) & (X.Month == k)].groupby(['Day_of_week', 'Hour']).Seasonal.mean().reset_index() for k in  X.Month.value_counts().index.values}
    seasonal_testset = X[X.TestSet == 1].loc[:, ['Month', 'Day_of_week', 'Hour']].apply(lambda x: fill_seasonal(x[0], x[1], x[2], seasonal_dict), axis=1)
    X.loc[X.TestSet == 1, 'Seasonal'] = X[X.TestSet == 1].loc[:, ['Month', 'Day_of_week', 'Hour']].apply(lambda x: fill_seasonal(x[0], x[1], x[2], seasonal_dict), axis=1)
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
    
    resample_d_mean, resample_d_min, resample_d_max, resample_d_median, resample_d_std = X.Residual.resample('D').mean().shift(1), X.Residual.resample('D').min().shift(1), X.Residual.resample('D').max().shift(1), X.Residual.resample('D').median().shift(1), X.Residual.resample('D').std().shift(1)
    X['RollingDMean'] = X.apply(lambda x: resample_d_mean[x.name.date()], axis=1)
    X['RollingDMax'] = X.apply(lambda x: resample_d_min[x.name.date()], axis=1)
    X['RollingDMin'] = X.apply(lambda x: resample_d_max[x.name.date()], axis=1)
    X['RollingDMedian'] = X.apply(lambda x: resample_d_median[x.name.date()], axis=1)
    X['RollingDStd'] = X.apply(lambda x: resample_d_std[x.name.date()], axis=1)
    
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
    
    alg.fit(X_train_selected_features_nona, y_train_no_na)
    y_pred = alg.predict(X_test_selected_features_nona)
    df_res_residual = pd.DataFrame({'y_true': y_test_no_na.values, 'y_pred': y_pred}, index=y_test_no_na.index)
    
    ps_y_pred = pd.Series(y_pred, index=y_test_no_na.index)
    y_pred_tmp = ps_y_pred + X[(X.index >= '2018-01-01 00:00:00') & (X.index <= '2018-12-31 23:00:00')].Seasonal
    df_tmp = (df.Consumption.shift(n_diff)[(df.index >= '2018-01-01 00:00:00') & (df.index <= '2018-12-31 23:00:00')] + y_pred_tmp)
    df_tmp.index = pd.DatetimeIndex(df_tmp.index)
    orig_data = df.loc[df.TestSet == 1, 'Consumption']
    orig_data.index = pd.DatetimeIndex(orig_data.index)
    df_res_hourly = pd.DataFrame({'y_true': orig_data, 'y_pred': df_tmp})
    
    # df_res_unicorn = create_unicorn_metric_df(df_res_hourly)
    df_res_daily = df_res_hourly.resample('D').sum()
    metrics = compute_metrics(df_res_hourly)
    print(metrics)
    print('All done')
    #return df_res_residual, df_res_hourly, df_res_daily, df_res_unicorn

def experiment_cpu(n_diff=1):
    df = pd.read_csv('data/ppnet_metar_v7.csv',  sep=';', index_col=0)
    # df = pd.concat([df]*50, ignore_index=True)
    start = perf_counter()
    print(df.shape)
    df = df[df.Year < 2019]
    df['TestSet'] = 0
    df.loc[df.Year == 2018, 'TestSet'] = 1
    X, y = df.drop('Consumption', axis=1), df.Consumption
    
    y_diff = y.diff(n_diff).dropna()
    y_diff_index = y_diff.index
    X['Consumption_diff_1'] = y_diff
    lags = np.array([24, 25, 35, 36, 37, 47, 48, 49, 71,72,73, 95, 96, 97, 119, 120, 121, 143,144,145, 168, 167, 169])
    lags_arr = np.arange(n_diff, 24)
    lags_con = np.concatenate((lags, lags_arr))
        
    for x in lags:
        X[f'Consumption_lag_{x}'] = y.shift(x)
        X[f'Temperature_lag_{x}'] = X.loc[:, 'Temperature'].shift(x)
        X[f'Consumption_diff_1_lag_{x}'] = X.loc[:, 'Consumption_diff_1'].shift(x)
        X[f'Humidity_lag_{x}'] = X.loc[:, 'Humidity'].shift(x)
        X[f'Cena_lag_{x}'] = X.loc[:, 'Cena_bfill'].shift(x)
        X[f'Windspeed_lag_{x}'] = X['Wind_speed'].shift(x)
        X[f'Pressure_lag_{x}'] = X['Pressure'].shift(x)

    X['Day_of_week_sin'] = np.sin(2 * np.pi * X['Day_of_week']/7.0)
    X['Day_of_week_cos'] = np.cos(2 * np.pi * X['Day_of_week']/7.0)
    X['Month_sin'] = np.sin(2 * np.pi * X['Month']/12.0)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month']/12.0)
    X['Hour_sin'] = np.sin(2 * np.pi * X['Hour']/23.0)
    X['Hour_cos'] = np.cos(2 * np.pi * X['Hour']/23.0)
    X['Day_sin'] = np.sin(2 * np.pi * X['Day']/31.0)
    X['Day_cos'] = np.cos(2 * np.pi * X['Day']/31.0)

    stl_decomp = rstl.STL(y_diff, freq=24*7, robust=False, s_window='periodic')
    trend, seasonal, residual, weights = stl_decomp.trend, stl_decomp.seasonal, stl_decomp.remainder, stl_decomp.weights
    residual_trend = residual + trend
    X.loc[:, f'Seasonal'] = pd.Series((np.concatenate([[np.nan]*n_diff, seasonal])), index=X.index)
    X.loc[:, f'Residual'] = pd.Series((np.concatenate([[np.nan]*n_diff, residual_trend])), index=X.index)
    X.loc[:, f'Trend'] = pd.Series((np.concatenate([[np.nan]*n_diff,trend])), index=X.index)

    seasonal_dict = {k:X[(X.TestSet == 0) & (X.Month == k)].groupby(['Day_of_week', 'Hour']).Seasonal.mean().reset_index() for k in  X.Month.value_counts().index.values}
    seasonal_arr = np.ndarray((12, 7, 23))
    tab = X[(X.TestSet == 0) & (X.Month == 1)].groupby(['Day_of_week', 'Hour']).Seasonal.mean().reset_index().values
    print(tab)
    print(tab.shape)
    tab_reshaped = tab.reshape(168, 3, 1)
    # print(tab.reshape(7, 24, -1))
    # seasonal_arr[0, :, :] = tab
    # print(seasonal_arr)
    # print(seasonal_arr.shape)
    # print(seasonal_arr[0, 0, 2])
    end = perf_counter()
    print(f'Difference: {end - start}')

def experiment_gpu(n_diff=1):
    # alg = RandomForestRegressor(n_estimators=10, n_jobs=8, random_state=14)
    df = cf.read_csv('data/ppnet_metar_v7.csv',  sep=';', index_col=0)
    # TODO: Odkomentovat pro repeat datasetu 10x
    # df = cf.concat([df]*10, ignore_index=True)
    start = perf_counter()
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
        
    for x in lags:
        X[f'Consumption_lag_{x}'] = y.shift(x)
        X[f'Temperature_lag_{x}'] = X['Temperature'].shift(x)
        X[f'Consumption_diff_1_lag_{x}'] = X['Consumption_diff_1'].shift(x)
        X[f'Humidity_lag_{x}'] = X['Humidity'].shift(x)
        X[f'Cena_lag_{x}'] = X['Cena_bfill'].shift(x)
        X[f'Windspeed_lag_{x}'] = X['Wind_speed'].shift(x)
        X[f'Pressure_lag_{x}'] = X['Pressure'].shift(x)
    # print(X['Day_of_week'].max())
    # print(X['Month'].max())
    # print(X['Hour'].max())
    # print(X['Day'].max())

    X['Day_of_week_sin'] = cp.sin(2 * cp.pi * X['Day_of_week']/7.0)
    X['Day_of_week_cos'] = cp.cos(2 * cp.pi * X['Day_of_week']/7.0)
    X['Month_sin'] = cp.sin(2 * cp.pi * X['Month']/12.0)
    X['Month_cos'] = cp.cos(2 * cp.pi * X['Month']/12.0)
    X['Hour_sin'] = cp.sin(2 * cp.pi * X['Hour']/23.0)
    X['Hour_cos'] = cp.cos(2 * cp.pi * X['Hour']/23.0)
    X['Day_sin'] = cp.sin(2 * cp.pi * X['Day']/31.0)
    X['Day_cos'] = cp.cos(2 * cp.pi * X['Day']/31.0)

    start_rstl = perf_counter()
    y_diff_np = cp.asnumpy(y_diff.values)
    stl_decomp = rstl.STL(y_diff_np, freq=24*7, robust=False, s_window='periodic')
    trend, seasonal, residual, weights = cp.array(stl_decomp.trend), cp.array(stl_decomp.seasonal), cp.array(stl_decomp.remainder), cp.array(stl_decomp.weights)
    residual_trend = residual + trend
    X[f'Seasonal'] = cf.Series((cp.concatenate([cp.array([cp.nan])*n_diff, seasonal])), index=X.index)
    X[f'Residual'] = cf.Series((cp.concatenate([cp.array([cp.nan])*n_diff, residual_trend])), index=X.index)
    X[f'Trend'] = cf.Series((cp.concatenate([cp.array([cp.nan])*n_diff,trend])), index=X.index)
    end_rstl = perf_counter()
    print(f'Difference RSTL: {end_rstl - start_rstl}')

    # seasonal_dict = {k:X[(X.TestSet == 0) & (X.Month == k)].groupby(['Day_of_week', 'Hour']).Seasonal.mean().reset_index() for k in  X.Month.value_counts().index.values}
    # print(seasonal_dict)

    seasonal_cparray = cp.ndarray((12, 7, 23))

    cp.cuda.Stream.null.synchronize()
    end = perf_counter()
    print(f'Difference: {end - start}')

if __name__ == "__main__":
    # df = cf.read_csv('data/ppnet_metar_v7.csv',  sep=';', index_col=0)
    # df = df[df.Year < 2019].copy()
    # df['TestSet'] = 0
    # df.loc[df.Year == 2018, 'TestSet'] = 1
    # print(df.head())
    experiment_cpu()
    # experiment_gpu()
    # cp.cuda.Stream.null.synchronize()