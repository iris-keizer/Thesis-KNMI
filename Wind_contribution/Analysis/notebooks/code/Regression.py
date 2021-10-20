"""
File containing the Python functions to perform a regression between sea level data and wind data 

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
nearby_wind_regression_obs_era5.ipynb 
nearby_wind_regression_cmip6_historical.ipynb 

"""


# Import necessary packages
import copy
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.linear_model import LinearRegression as linr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression



"""
Practical functions
-------------------


"""


def station_names(): 
    """
    Function to obtain tide gauge station names as list
    
    """
    return ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']






# Declare global variables
stations = station_names()



"""
REGRESSION FUNCTION
-------------------

"""


def regression_obs(wind_data, tg_data, model = 'NearestPoint'):
    """
    Function to perform the regression between the tide gauge data and observed wind data 
    
    For model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    """
    
    # Add trend column to wind dataframe
    trend_lst = copy.deepcopy(wind_data.index.tolist())
    trend_lst = [j - wind_data.index[0] for j in trend_lst]
    wind_data['trend'] =  trend_lst
    
    
    # Get names of all regression and wind regression coefficients
    regg_names, wind_names =  regression_names(model)
    
    
    # Create lists to save variables
    timeseries_lst = []
    coef_lst = []
    alpha_lst = []
    intercept_lst = []
    rmse_lst = []
    R2_wind_lst = []
    
    
    # Perform regression for each station
    for idx, station in enumerate(stations):
        y = tg_data[station]
        
        
        if model == 'NearestPoint':
            
            # Create x dataframe with timeseries used in the regression
            x = pd.DataFrame(data={'time': wind_data.index.tolist(), 
                                   'u$^2$' : wind_data[station, 'u$^2$'].tolist(), 
                                   'v$^2$' : wind_data[station, 'v$^2$'].tolist(), 
                                   'trend': wind_data['trend'].tolist()})
            x = x.set_index('time')
                
            # Define regression
            regression_ = linr()
            
        elif model == 'Timmerman':
            
            # Create x dataframe with timeseries used in the regression
            x = copy.deepcopy(wind_data)
            
            # Define regression
            tss = TimeSeriesSplit(n_splits=5)
            regression_ = LassoCV(alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.01], 
                                            cv=tss, max_iter=500000000)
            
            
        # Drop nan values
        x = x.dropna()
        y = y.dropna()
        
        
        # Standardize x
        scaler = StandardScaler()
        x = copy.deepcopy(x)
        x.iloc[:,:] = scaler.fit_transform(x)
        
        
        # Create copy such that regression result can be obtained for full timeseries
        x_timeseries = copy.deepcopy(x)
        
        
        # Create dataframes of equal time span
        y = y[y.index.isin(x.index)]
        x = x[x.index.isin(y.index)]
        
        
        # Fit the regression model and add results to lists
        if  model == 'Timmerman':
            fit = regression_.fit(x,y)
            alpha = regression_.alpha_
            alpha_lst.append(alpha)
            regression_ = Lasso(alpha)
        fit = regression_.fit(x,y)
        score = regression_.score(x,y) # R^2 for the whole regression including the trend
        intercept_lst.append(regression_.intercept_)
        coef_lst.append(regression_.coef_.tolist())
        f_statistic, p_values = f_regression(x, y)
    
        # Check significance
        significance_level = 95
        significance = significance_test(p_values[0], 1-significance_level/100)
    
        
        # Calculate mse
        yhat = regression_.predict(x)
        mse = mean_squared_error(y, yhat) # Calculate insample mse
        if significance == False:
            mse =  np.nan
            
            
        rmse_lst.append(np.sqrt(mse))
        
        # Obtain dataframe containing timeseries resulting from regression
        df = pd.DataFrame(data=dict(time=x_timeseries.index))
        df = df.set_index('time')
        
        for i in range(len(regg_names)):
            df[regg_names[i]] = coef_lst[-1][i] * x_timeseries[x_timeseries.columns[i]]
        df['total'] = df.sum(axis=1)
        df['wind total'] = df[wind_names].sum(axis=1)
        
        if model ==  'Timmerman':
            region_names = ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']
            for i in range(len(region_names)):
                df[region_names[i]] = df[[wind_names[2*i], wind_names[2*i+1]]].sum(axis=1)
            df['u2 total'] = df[[wind_names[0], wind_names[2], wind_names[4], wind_names[6], wind_names[8], wind_names[10]]].sum(axis=1)
            df['v2 total'] = df[[wind_names[1], wind_names[3], wind_names[5], wind_names[7], wind_names[9], wind_names[11]]].sum(axis=1)
        
        # Add dataframe to list 
        timeseries_lst.append(df)
        
        # Calculate R^2 for wind contribution to sea level height
        y_wind = df['Wind total']
        y_wind = y_wind[y_wind.index.isin(x.index)]
        if significance == False:
            R2_wind_lst.append(np.nan)
        else:
            R2_wind_lst.append(regression_.score(x,y_wind))
        
        
    # Create dataframe of timeseries
    timeseries_df = pd.concat(timeseries_lst, axis=1, keys = stations)
    
    
    
    # Create dataframe of coefficients
    
    # Transpose coef list
    numpy_array = np.array(coef_lst)
    transpose = numpy_array.T
    coef_lst_T = transpose.tolist()
    
    results_df = pd.DataFrame(data={'station': stations, 'R$^2$' : R2_wind_lst, 'RMSE': rmse_lst, 'constant' : intercept_lst})
    for i in range(len(coef_lst_T)):
        results_df[regg_names[i]] = coef_lst_T[i]
        
    results_df = results_df.set_index('station')
    
    return(results_df, timeseries_df)




def regression_cmip6(wind_data, tg_data, wind_model = 'NearestPoint'):
    """
    Function to perform the regression between the cmip6 sea level and wind data
    
    For model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    """

    
    # Add trend column to wind dataframe
    trend_lst = copy.deepcopy(wind_data.time.values)
    trend_lst = [j - wind_data.time.values[0] for j in trend_lst]
    
    
    # Get names of all regression and wind regression coefficients
    regg_names, wind_names =  regression_names(wind_model)
    
    
    # Only use models occuring in both datasets
    tg_data = tg_data.where(tg_data.model.isin(wind_data.model), drop=True)
    wind_data = wind_data.where(wind_data.model.isin(tg_data.model), drop=True)
    
    # Create lists to save datasets
    timeseries_lst1 = []
    reg_results_lst = []
    
    # Perform regression for each model
    for model in wind_data.model.values:

        # Create lists to save variables
        timeseries_lst = []
        coef_lst = []
        alpha_lst = []
        intercept_lst = []
        rmse_lst = []
        R2_wind_lst = []


        # Perform regression for each station
        for idx, station in enumerate(stations):
            y = pd.DataFrame(data={'time': tg_data.time.values,
                                   'zos': tg_data.zos.sel(model=model, station=station).values})
            
            y = y.set_index('time')
            
            if wind_model == 'NearestPoint':
                # Create x dataframe with timeseries used in the regression
                x = pd.DataFrame(data={'time': wind_data.time.values, 
                                       'u$^2$' : wind_data.u2.sel(model=model, station=station).values, 
                                       'v$^2$' : wind_data.v2.sel(model=model, station=station).values, 
                                       'trend': trend_lst})
                x = x.set_index('time')
                
                
                # Define regression
                regression_ = linr()

            elif wind_model == 'Timmerman':

                # Create x dataframe with timeseries used in the regression
                x = copy.deepcopy(wind_data) #ADJUST

                # Define regression
                tss = TimeSeriesSplit(n_splits=5)
                regression_ = LassoCV(alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.01], 
                                                cv=tss, max_iter=500000000)


            # Drop nan values
            x = x.dropna()
            y = y.dropna()

            
            # Standardize x
            scaler = StandardScaler()
            x = copy.deepcopy(x)
            x.iloc[:,:] = scaler.fit_transform(x)


            # Create copy such that regression result can be obtained for full timeseries
            x_timeseries = copy.deepcopy(x)

            
            # Create dataframes of equal time span
            y = y[y.index.isin(x.index)]
            x = x[x.index.isin(y.index)]
            
            
            # Fit the regression model and add results to lists
            if  wind_model == 'Timmerman':
                fit = regression_.fit(x, y.values.ravel())
                alpha = regression_.alpha_
                alpha_lst.append(alpha)
                regression_ = Lasso(alpha)
                
            fit = regression_.fit(x, y.values.ravel())
            score = regression_.score(x, y.values.ravel()) # R^2 for the whole regression including the trend
            intercept_lst.append(regression_.intercept_)
            coef_lst.append(regression_.coef_.tolist())
            f_statistic, p_values = f_regression(x, y.values.ravel())
            
            # Check significance
            significance_level = 95
            significance = significance_test(p_values[0], 1-significance_level/100)

            
            # Calculate rmse
            yhat = regression_.predict(x)
            mse = mean_squared_error(y, yhat) # Calculate insample mse
            if significance == False:
                mse =  np.nan
            rmse_lst.append(np.sqrt(mse))

            
            # Obtain dataframe containing timeseries resulting from regression
            df = pd.DataFrame(data=dict(time=x_timeseries.index))
            df = df.set_index('time')
            
            for i in range(len(regg_names)):
                
                df[regg_names[i]] = coef_lst[-1][i] * x_timeseries[x_timeseries.columns[i]]
        
            
            df['total'] = df.sum(axis=1)
            df['wind total'] = df[wind_names].sum(axis=1)
            
            
            if wind_model ==  'Timmerman':
                region_names = ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']
                for i in range(len(region_names)):
                    df[region_names[i]] = df[[wind_names[2*i], wind_names[2*i+1]]].sum(axis=1)
                df['u2 total'] = df[[wind_names[0], wind_names[2], wind_names[4], wind_names[6], wind_names[8], wind_names[10]]].sum(axis=1)
                df['v2 total'] = df[[wind_names[1], wind_names[3], wind_names[5], wind_names[7], wind_names[9], wind_names[11]]].sum(axis=1)
                
              
            # Create dataset
            if wind_model == 'NearestPoint':
                timeseries_lst.append(xr.Dataset(data_vars=dict(u2=(['time'],  df['u$^2$'].values),
                                                               v2=(['time'],  df['v$^2$'].values),
                                                               trend=(['time'],  df['trend'].values),
                                                               total=(['time'],  df['total'].values),
                                                               wind_total=(['time'],  df['wind total'].values),
                                                               time = df.index.values)))
            
            
            # Calculate R^2 for wind contribution to sea level height
            y_wind = df['wind total']
            y_wind = y_wind[y_wind.index.isin(x.index)]
            if significance == False:
                R2_wind_lst.append(np.nan)
            else:
                R2_wind_lst.append(regression_.score(x,y_wind.values.ravel()))


            
        # Transpose coef list
        numpy_array = np.array(coef_lst)
        transpose = numpy_array.T
        coef_lst_T = transpose.tolist()
        
        
        # Create dataset of regression results (R^2, rmse, regression coefficients)
        reg_results_lst.append(xr.Dataset(data_vars=dict(r2=(['station'], R2_wind_lst),
                                                        rmse=(['station'], rmse_lst),
                                                        constant=(['station'], intercept_lst),
                                                        u2_coef=(['station'], coef_lst_T[0]),
                                                        v2_coef=(['station'], coef_lst_T[1]),
                                                        trend_coef=(['station'], coef_lst_T[2]),
                                                        station = stations)))
        
                                      
        # Put all station datasets in one dataset
        timeseries_lst1.append(xr.concat(timeseries_lst, dim=stations).rename({"concat_dim":"station"}))          
    
                                      
    # Put all model datasets in one dataset
    timeseries_dataset = xr.concat(timeseries_lst1, dim=wind_data.model.values).rename({"concat_dim":"model"})                           
    results_dataset = xr.concat(reg_results_lst, dim=wind_data.model.values).rename({"concat_dim":"model"})
                                      
    return(results_dataset, timeseries_dataset)









"""
USEFULL FUNCTIONS
-------------------

"""

def regression_names(model):
    
    if model == 'NearestPoint':
        regg_names = ['u$^2$', 'v$^2$', 'trend']
        wind_names = ['u$^2$', 'v$^2$']
        
    elif model == 'Timmerman':
        regg_names = ['Channel u$^2$', 'Channel v$^2$', 'South u$^2$',  'South v$^2$',  'Mid-West u$^2$', 'Mid-West v$^2$',  
                      'Mid-East u$^2$', 'Mid-East v$^2$', 'North-West u$^2$', 'North-West v$^2$', 'North-East u$^2$', 'North-East v$^2$', 'Trend']
        wind_names = ['Channel u$^2$', 'Channel v$^2$', 'South u$^2$',  'South v$^2$',  'Mid-West u$^2$', 'Mid-West v$^2$',  
                      'Mid-East u$^2$', 'Mid-East v$^2$', 'North-West u$^2$', 'North-West v$^2$', 'North-East u$^2$', 'North-East v$^2$']
            
    return regg_names, wind_names



def significance_test(p, alpha):
    if p < alpha: return True
    else: return False