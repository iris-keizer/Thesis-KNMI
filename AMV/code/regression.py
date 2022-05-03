"""
File containing the Python functions to perform the regression between AMV and the atmospheric contribution to sea level height at the Dutch coast.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
lagged_regression_obs.ipynb

"""

# Import necessary packages
import copy
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as linr


wind_labels = ['NearestPoint', 'Timmerman', 'Dangendorf']
AMV_names = ['HadISSTv2', 'ERSSTv5', 'COBE-SST2']



"""
Perform regression
------------------


"""





def regression(data_x, data_y, lag):
    
    # Create dataframe
    x_l = pd.DataFrame(data={'time': data_x.index, 'AMV':data_x.values})
    x_l = x_l.set_index('time')
    
        
    # Standardize x
    scaler = StandardScaler()
    x_l = copy.deepcopy(x_l)
    x_l.iloc[:,:] = scaler.fit_transform(x_l)
    
    
    # Execute lagged regression by shifting the AMV dataframe. 
    x_l.index = x_l.index + lag
    
    
    # Create data series of equal time span
    y = data_y[data_y.index.isin(x_l.index)]
    x = x_l[x_l.index.isin(y.index)]
    
    
    # Fit the regression model
    linear_regression = linr()
    fit = linear_regression.fit(x, y)
    r2 = linear_regression.score(x, y)
    intercept = linear_regression.intercept_
    coefs = linear_regression.coef_.tolist()[0]
    AMV_reg_timeseries = coefs*x_l['AMV']
        
    yhat = linear_regression.predict(x)
    mse = mean_squared_error(y, yhat) # Calculate insample mse (non-negative)
    rmse = np.sqrt(mse)
        
    
    
    return [rmse, r2, intercept, coefs], AMV_reg_timeseries



def lagged_regression(data_x, data_y):
    
    lags = np.arange(-20, 21)
    
    ts_lst1 = []
    r_lst1 = []
    
    for column_x in data_x:
        ts_lst0 = []
        r_lst0 = []
        
        for column_y in data_y:
            df_timeseries = pd.DataFrame(columns = lags)
            df_results = df_timeseries.copy()
            df_timeseries['time'] = data_x.index
            df_timeseries = df_timeseries.set_index('time')
            
            df_results['result'] = ['rmse', 'r$^2$', 'constant', 'coef']
            df_results = df_results.set_index('result')
            
            for lag in lags:
                
                results, timeseries = regression(data_x[column_x], data_y[column_y], lag)
                df_timeseries[lag] = timeseries
                df_results[lag] = results
                
            ts_lst0.append(df_timeseries)
            r_lst0.append(df_results)
        
        ts_lst1.append(pd.concat(ts_lst0, axis=1, keys=data_y.columns))
        r_lst1.append(pd.concat(r_lst0, axis=1, keys=data_y.columns))
        
    timeseries = pd.concat(ts_lst1, axis=1, keys=data_x.columns)
    results = pd.concat(r_lst1, axis=1, keys=data_x.columns)

    return results, timeseries




def lagged_regression_cmip6(data_x, data_y):
    
    
    lags = np.arange(0, 41)
    
    ts_lst1 = []
    r_lst1 = []
    
    for model in data_x.columns:
        ts_lst0 = []
        r_lst0 = []
        
        for wl in wind_labels:
            df_timeseries = pd.DataFrame(columns = lags)
            df_results = df_timeseries.copy()
            df_timeseries['time'] = data_x.index
            df_timeseries = df_timeseries.set_index('time')
            
            df_results['result'] = ['rmse', 'r$^2$', 'constant', 'coef']
            df_results = df_results.set_index('result')
            
            for lag in lags:
                
                results, timeseries = regression(data_x[model], data_y[wl, model], lag)
                df_timeseries[lag] = timeseries
                df_results[lag] = results
                
            ts_lst0.append(df_timeseries)
            r_lst0.append(df_results)
        
        ts_lst1.append(pd.concat(ts_lst0, axis=1, keys=wind_labels))
        r_lst1.append(pd.concat(r_lst0, axis=1, keys=wind_labels))
        
    timeseries = pd.concat(ts_lst1, axis=1, keys=data_x.columns)
    results = pd.concat(r_lst1, axis=1, keys=data_x.columns)

    return results, timeseries

