import cupy as cp
import numpy as np
import pandas as pd
import time

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

if __name__ == "__main__":
    print(cp.__version__)
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