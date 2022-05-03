"""
File containing the Python functions to perform the regression between SST and the atmospheric contribution to sea level height at the Dutch coast.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
lagged_regression_obs.ipynb

"""

# Import necessary packages
import copy
import numpy as np
import xarray as xr
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as linr


wind_labels = ['NearestPoint', 'Timmerman', 'Dangendorf']



"""
Perform regression
------------------


"""





def regression(data_x, data_y, lag):
    
    # Create dataframe
    x_l = pd.DataFrame(data={'time': data_x.year.values, 'SST':data_x.values})
    x_l = x_l.set_index('time')
    x_l = x_l.dropna()
    
    if x_l['SST'].size == 0:
        return [np.nan, np.nan, np.nan, np.nan], xr.DataArray(data=data_x.values ,dims=["time"], coords=dict(time=data_x.year.values,),) 
    
    
    # Standardize x
    scaler = StandardScaler()
    x_l = copy.deepcopy(x_l)
    x_l.iloc[:,:] = scaler.fit_transform(x_l)
    
    
    # Execute lagged regression by shifting the SST dataframe. 
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
    timeseries = coefs*x_l['SST']
    SST_reg_timeseries = xr.DataArray(data=timeseries.values ,dims=["time"], coords=dict(time=timeseries.index.values,),) 
        
    
    yhat = linear_regression.predict(x)
    mse = mean_squared_error(y, yhat) # Calculate insample mse (non-negative)
    rmse = np.sqrt(mse)
        
    
    
    return [rmse, r2, intercept, coefs], SST_reg_timeseries






def lagged_regression(data_x, data_y):
    
    lags = [0]
    
    ts_lst_wl = []
    res_lst_wl = []
    for wl in wind_labels:
        
        ts_lst_lat = []
        res_lst_lat = []
        for lat in data_x.lat:
            
            ts_lst_lon = []
            res_lst_lon = []
            for lon in data_x.lon:
                
                ts_lst_lag = []
                rmse_lst = []
                r2_lst = []
                intercept_lst = []
                coefs_lst = []
                for lag in lags:
                    result, ts = regression(data_x.sel(lat=lat, lon=lon, drop = True), data_y[wl], lag)
                    ts_lst_lag.append(ts)
                    rmse_lst.append(result[0])
                    r2_lst.append(result[1])
                    intercept_lst.append(result[2])
                    coefs_lst.append(result[3])
                    
                ts_lst_lon.append(xr.concat(ts_lst_lag, dim=lags).rename({"concat_dim":"lag"}))
                
                res_lst_lon.append(xr.Dataset(data_vars = dict(
                rmse = (['lag'], rmse_lst),
                r2 = (['lag'], r2_lst),
                intercept = (['lag'], intercept_lst),
                reg_coef = (['lag'], coefs_lst)),
                                             coords = dict(lag=lags)))
                
                
            ts_lst_lat.append(xr.concat(ts_lst_lon, dim=data_x.lon))
            res_lst_lat.append(xr.concat(res_lst_lon, dim=data_x.lon))
                                   
        ts_lst_wl.append(xr.concat(ts_lst_lat, dim=data_x.lat))
        res_lst_wl.append(xr.concat(res_lst_lat, dim=data_x.lat))
                                   
    timeseries = xr.concat(ts_lst_wl, dim=wind_labels).rename({"concat_dim":"wind_model"})
    results = xr.concat(res_lst_wl, dim=wind_labels).rename({"concat_dim":"wind_model"})
                                   
    return results, timeseries




def lagged_regression_cmip6(data_x, data_y):
    
    
    lags = [0]
    
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