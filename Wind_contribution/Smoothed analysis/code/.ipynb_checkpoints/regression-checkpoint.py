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
import pickle

import numpy as np
import xarray as xr
import pandas as pd

from sklearn.linear_model import LinearRegression as linr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, LassoCV, RidgeCV


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



"""
Practical functions
-------------------


"""


def station_names(): 
    """
    Function to obtain tide gauge station names as list
    
    """
    return ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']


def timmerman_region_names(): 
    """
    Function to obtain timmerman region names as list
    
    """
    return ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']


def save_nc_data(data, folder, variable, name): 
    """
    Function to save data as NETCDF4 file
    
    For folder choose ['observations', 'cmip6'], 
    for variable choose ['Wind', 'SLH', 'Pressure', 'SST', 'Regression results']
    
    """
    data.to_netcdf(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.nc", mode='w')

    
def save_csv_data(data, folder, variable, name): 
    """
    Function to save data as .csv file
    
    For folder choose ['observations', 'cmip6'], 
    for variable choose ['Wind', 'SLH', 'Pressure', 'SST', 'Regression results']
    
    """
    data.to_csv(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.csv")

    
    
# Declare global variables
stations = station_names()
regions = timmerman_region_names()
significance_level = 95
alphas = [0.03582961338564776, 0.07386858364808699, 0.09391780126679188, 0.13034639992652514, 
          0.21143711191718467, 0.0657308019386455, 0.02740434989586052]






"""
REGRESSION FUNCTION
-------------------


"""


def regression_obs(wind_data, tg_data, wind_model = 'NearestPoint', data_type = 'era5'):
    """
    Function to perform the regression between the tide gauge data and observed wind data 
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    """
    
    # Create list for the trend
    trend_lst = copy.deepcopy(wind_data.index.tolist())
    trend_lst = [j - wind_data.index[0] for j in trend_lst]
    wind_data['trend'] =  trend_lst
    
    
    # Get names of all regression and wind regression coefficients
    regg_names, wind_names =  regression_names(wind_model)
    
    
    # Create lists to save variables
    timeseries_lst = []
    coef_lst = []
    alpha_lst = []
    intercept_lst = []
    rmse_lst = []
    R2_total_lst = []
    R2_wind_lst = []
    R2_u2_lst = []
    R2_v2_lst = []
    
    # Create dataframe for significance
    signif_df = pd.DataFrame(data=dict(reggression_contributor=regg_names+['total', 'wind total'])) # Create dataframe 
    signif_df = signif_df.set_index('reggression_contributor')
    
    
    # Perform regression for each station
    for idx, station in enumerate(stations):
        y = tg_data[station]
        
        
        if wind_model == 'NearestPoint':
            
            # Create x dataframe with timeseries used in the regression
            x = pd.DataFrame(data={'time': wind_data.index.tolist(), 
                                   'u$^2$' : wind_data[station, 'u$^2$'].tolist(), 
                                   'v$^2$' : wind_data[station, 'v$^2$'].tolist(), 
                                   'trend': wind_data['trend'].tolist()})
            x = x.set_index('time')
                
            # Define regression
            regression_ = linr()
            
        elif wind_model == 'Timmerman':

            # Create x dataframe with timeseries used in the regression
            x = copy.deepcopy(wind_data)

        elif wind_model == 'Dangendorf':

            # Create x dataframe with timeseries used in the regression
            x = copy.deepcopy(wind_data)

            # Define regression
            regression_ = linr()
            
            
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
            
            # Define regression
            tss = TimeSeriesSplit(n_splits=5)
            regression_ = LassoCV(cv=tss, max_iter=10**(9))
            
            fit = regression_.fit(x,y)
            alpha = regression_.alpha_
            alpha_lst.append(alpha)
            regression_ = Lasso(alpha)
            
            
        fit = regression_.fit(x,y)
        score = regression_.score(x,y) # R^2 for the whole regression including the trend
        intercept_lst.append(regression_.intercept_)
        coef_lst.append(regression_.coef_.tolist())
    
        
        # Calculate mse
        yhat = regression_.predict(x)
        mse = mean_squared_error(y, yhat) # Calculate insample mse
        rmse_lst.append(np.sqrt(mse))
        
        # Obtain dataframe containing timeseries resulting from regression
        df = pd.DataFrame(data=dict(time=x_timeseries.index))
        df = df.set_index('time')
        
        for i in range(len(regg_names)):
            df[regg_names[i]] = coef_lst[-1][i] * x_timeseries[x_timeseries.columns[i]]
            
        df['total'] = df.sum(axis=1)
        df['wind total'] = df[wind_names].sum(axis=1)
        
        if wind_model ==  'Timmerman':
            region_names = regions[:-1]
            for i in range(len(region_names)):
                df[region_names[i]] = df[[wind_names[2*i], wind_names[2*i+1]]].sum(axis=1)
            df['u$^2$ total'] = df[[wind_names[0], wind_names[2], wind_names[4], wind_names[6], 
                                 wind_names[8], wind_names[10]]].sum(axis=1)
            df['v$^2$ total'] = df[[wind_names[1], wind_names[3], wind_names[5], wind_names[7], 
                                 wind_names[9], wind_names[11]]].sum(axis=1)
        # Add dataframe to list 
        timeseries_lst.append(df)
        
        
        # Calculate R^2 values
        R2_total_lst.append(regression_.score(x, y.values.ravel())) # R^2 for the whole regression including the trend
        R2_wind_lst.append(R2_var(df, y, 'wind total', regression_))
        if wind_model == 'NearestPoint':
            R2_u2_lst.append(R2_var(df, y, 'u$^2$', regression_))
            R2_v2_lst.append(R2_var(df, y, 'v$^2$', regression_))
        elif wind_model == 'Timmerman':
            R2_u2_lst.append(R2_var(df, y, 'u$^2$ total', regression_))
            R2_v2_lst.append(R2_var(df, y, 'v$^2$ total', regression_))
        elif wind_model == 'Dangendorf':
            R2_u2_lst.append(R2_var(df, y, 'Negative corr region', regression_))
            R2_v2_lst.append(R2_var(df, y, 'Positive corr region', regression_))
        
        
        
        # Check significance for each regression contributor
        f_statistic, p_values = f_regression(x, y)
        
        sign_lst = []
        for i, p_value in enumerate(p_values):
            sign_lst.append(significance_test(p_value, 1-significance_level/100))
        
        
        # Check significance for total regression and wind contribution
        x_total = df['total'].to_frame()[df['total'].to_frame().index.isin(y.index)]
        x_wind_total = df['wind total'].to_frame()[df['wind total'].to_frame().index.isin(y.index)]
        
        f_statistic, p_values = f_regression(x_total, y)
        sign_lst.append(significance_test(p_values[0], 1-significance_level/100))
        
        f_statistic, p_values = f_regression(x_wind_total, y)
        sign_lst.append(significance_test(p_values[0], 1-significance_level/100))

        signif_df[station] = sign_lst # Add to significance dataframe
                    
                

        
    # Create dataframe of timeseries
    timeseries_df = pd.concat(timeseries_lst, axis=1, keys = stations)
    
    
    
    # Create dataframe of coefficients
    
    # Transpose coef list
    numpy_array = np.array(coef_lst)
    transpose = numpy_array.T
    coef_lst_T = transpose.tolist()
    
    if wind_model == 'NearestPoint' or wind_model == 'Timmerman':
        results_df = pd.DataFrame(data={'station': stations, 
                                        'R$^2$' : R2_total_lst,
                                        'R$^2_{wind}$' : R2_wind_lst,
                                        'R$^2_{u^2}$' : R2_u2_lst,
                                        'R$^2_{v^2}$' : R2_v2_lst,
                                        'RMSE': rmse_lst, 
                                        'constant' : intercept_lst})
    elif wind_model == 'Dangendorf':
        results_df = pd.DataFrame(data={'station': stations, 
                                        'R$^2$' : R2_total_lst,
                                        'R$^2_{wind}$' : R2_wind_lst,
                                        'R$^2_{neg}$' : R2_u2_lst,
                                        'R$^2_{pos}$' : R2_v2_lst,
                                        'RMSE': rmse_lst, 
                                        'constant' : intercept_lst})
                                                            
    for i in range(len(coef_lst_T)):
        results_df[regg_names[i]] = coef_lst_T[i]
        
    results_df = results_df.set_index('station')
        
        
    # Save the dataframes
    save_csv_data(results_df, 'observations', 'Regression results', f'results_{wind_model}_{data_type}')
    save_csv_data(timeseries_df, 'observations', 'Regression results', f'timeseries_{wind_model}_{data_type}')
        
    
    return(results_df, timeseries_df, signif_df)









def regression_cmip6(wind_data, zos, wind_model = 'NearestPoint', data_type = 'historical'):
    """
    Function to perform the regression between the cmip6 sea level and wind data
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    """
    
    # Perform regression with zos data untill 1980, thereafter zos accelerates
    zos = zos.where(zos.time <= 1980, drop=True)
    
    
    # Add trend column to wind dataframe
    trend_lst = copy.deepcopy(wind_data.time.values)
    trend_lst = [j - wind_data.time.values[0] for j in trend_lst]
    
    
    # Get names of all regression and wind regression coefficients
    regg_names, wind_names =  regression_names(wind_model)
    
    
    # Create lists to save datasets
    timeseries_lst1 = []
    reg_results_lst = []
    signif_ds_lst = []
    
    
    # Create dataframe for scales of standardization
    scalers = {}
    
    
    # Perform regression for each model
    for model in wind_data.model.values:

        # Create lists to save variables
        timeseries_lst = []
        coef_lst = []
        alpha_lst = []
        intercept_lst = []
        rmse_lst = []
        R2_total_lst = []
        R2_wind_lst = []
        R2_u2_lst = []
        R2_v2_lst = []
        sign_total = []
        sign_wind_total = []
        sign_lst = []
    
        
         # Create dataframe for significance
        signif_df = pd.DataFrame(data=dict(reggression_contributor=regg_names+['total', 'wind total'])) # Create dataframe 
        signif_df = signif_df.set_index('reggression_contributor')
        
        
        # Perform regression for each station
        for idx, station in enumerate(stations):
            y = pd.DataFrame(data={'time': zos.time.values,
                                   'zos': zos.zos.sel(model=model, station=station).values})
            
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
                dfs = []
                for region in wind_data.tim_region.values:
                    dfs.append(wind_data.sel(model=model, 
                                             tim_region=region, drop=True).to_dataframe())

                x = pd.concat(dfs, axis=1, keys=wind_data.tim_region.values)

                x['trend'] = trend_lst

                # Define regression
                tss = TimeSeriesSplit(n_splits=5)
                regression_ = LassoCV(alphas=alphas, cv=tss, max_iter=10**(9))
                
                
                
            elif wind_model == 'Dangendorf':


                # Create x dataframe with timeseries used in the regression
                x = pd.DataFrame(data={'time': wind_data.time.values, 
                                        'Negative corr region' : wind_data['Negative corr region'].sel(model=model).values, 
                                        'Positive corr region' : wind_data['Positive corr region'].sel(model=model).values, 
                                        'trend': trend_lst})
                x = x.set_index('time')


                # Define regression
                regression_ = linr()
                
                
            # Drop nan values
            x = x.dropna()
            y = y.dropna()

            
            # Standardize x and save scaler for projections
            scaler = StandardScaler()
            if station == 'Average':
                scalers[model] = scaler.fit(x)
            
            x = copy.deepcopy(x)
            x.iloc[:,:] = scaler.fit_transform(x)


            # Create copy such that regression result can be obtained for full time series
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
                #regression_ = linr()
                
                
            fit = regression_.fit(x, y.values.ravel())
            intercept_lst.append(regression_.intercept_)
            coef_lst.append(regression_.coef_.tolist())
            
            
            
            # Calculate rmse
            yhat = regression_.predict(x)
            mse = mean_squared_error(y, yhat) # Calculate insample mse
            rmse_lst.append(np.sqrt(mse))
            
            # Obtain dataframe containing timeseries resulting from regression
            df = pd.DataFrame(data=dict(time=x_timeseries.index))
            df = df.set_index('time')
            
            for i in range(len(regg_names)):
                
                df[regg_names[i]] = coef_lst[-1][i] * x_timeseries[x_timeseries.columns[i]]
        
            
            df['total'] = df.sum(axis=1)
            df['wind total'] = df[wind_names].sum(axis=1)
            
            
            if wind_model ==  'Timmerman':
                for i in range(len(regions)):
                    df[regions[i]] = df[[wind_names[2*i], wind_names[2*i+1]]].sum(axis=1)
                df['u$^2$ total'] = df[[wind_names[0], wind_names[2], wind_names[4], wind_names[6], 
                                     wind_names[8], wind_names[10]]].sum(axis=1)
                df['v$^2$ total'] = df[[wind_names[1], wind_names[3], wind_names[5], wind_names[7], 
                                     wind_names[9], wind_names[11]]].sum(axis=1)
                
            
            # Create dataset
            if wind_model == 'NearestPoint':
                timeseries_lst.append(xr.Dataset(data_vars=dict(u2=(['time'],  df['u$^2$'].values),
                                                               v2=(['time'],  df['v$^2$'].values),
                                                               trend=(['time'],  df['trend'].values),
                                                               total=(['time'],  df['total'].values),
                                                               wind_total=(['time'],  df['wind total'].values),
                                                               time = df.index.values)))

            elif wind_model == 'Timmerman':
                
                timeseries_lst.append(xr.Dataset(data_vars=dict(channel_u2=(['time'], df[regg_names[0]].values),
                                                                channel_v2=(['time'], df[regg_names[1]].values),
                                                                south_u2=(['time'], df[regg_names[2]].values),
                                                                south_v2=(['time'], df[regg_names[3]].values),
                                                                midwest_u2=(['time'], df[regg_names[4]].values),
                                                                midwest_v2=(['time'], df[regg_names[5]].values),
                                                                mideast_u2=(['time'], df[regg_names[6]].values),
                                                                mideast_v2=(['time'], df[regg_names[7]].values),
                                                                northwest_u2=(['time'], df[regg_names[8]].values),
                                                                northwest_v2=(['time'], df[regg_names[9]].values),
                                                                northeast_u2=(['time'], df[regg_names[10]].values),
                                                                northeast_v2=(['time'], df[regg_names[11]].values),
                                                                trend=(['time'], df[regg_names[12]].values),
                                                                total=(['time'], df['total'].values),
                                                                wind_total=(['time'], df['wind total'].values),
                                                                channel=(['time'], df[regions[0]].values),
                                                                south=(['time'], df[regions[1]].values),
                                                                midwest=(['time'], df[regions[2]].values),
                                                                mideast=(['time'], df[regions[3]].values),
                                                                northwest=(['time'], df[regions[4]].values),
                                                                northeast=(['time'], df[regions[5]].values),
                                                                u2_total=(['time'], df['u$^2$ total'].values),
                                                                v2_total=(['time'], df['v$^2$ total'].values),
                                                                time = df.index.values)))
          
        
            elif wind_model == 'Dangendorf':
                timeseries_lst.append(xr.Dataset(data_vars=dict(neg_corr_region=(['time'],  df['Negative corr region'].values),
                                                               pos_corr_region=(['time'],  df['Positive corr region'].values),
                                                               trend=(['time'],  df['trend'].values),
                                                               total=(['time'],  df['total'].values),
                                                               wind_total=(['time'],  df['wind total'].values),
                                                               time = df.index.values)))
                
                
            
            
            # Calculate R^2 values
            R2_total_lst.append(regression_.score(x, y.values.ravel())) # R^2 for the whole regression including the trend
            R2_wind_lst.append(R2_var(df, y, 'wind total', regression_))
            if wind_model == 'NearestPoint':
                R2_u2_lst.append(R2_var(df, y, 'u$^2$', regression_))
                R2_v2_lst.append(R2_var(df, y, 'v$^2$', regression_))
            elif wind_model == 'Timmerman':
                R2_u2_lst.append(R2_var(df, y, 'u$^2$ total', regression_))
                R2_v2_lst.append(R2_var(df, y, 'v$^2$ total', regression_))
            elif wind_model == 'Dangendorf':
                R2_u2_lst.append(R2_var(df, y, 'Negative corr region', regression_))
                R2_v2_lst.append(R2_var(df, y, 'Positive corr region', regression_))
                    
                
            # Check significance for each regression contributor
            f_statistic, p_values = f_regression(x, y.values.ravel())

            sign_lst = []
            for i, p_value in enumerate(p_values):
                sign_lst.append(significance_test(p_value, 1-significance_level/100))


            # Check significance for total regression and wind contribution
            x_total = df['total'].to_frame()[df['total'].to_frame().index.isin(y.index)]
            x_wind_total = df['wind total'].to_frame()[df['wind total'].to_frame().index.isin(y.index)]

            f_statistic, p_values = f_regression(x_total, y.values.ravel())
            sign_lst.append(significance_test(p_values[0], 1-significance_level/100))
            f_statistic, p_values = f_regression(x_wind_total, y.values.ravel())
            sign_lst.append(significance_test(p_values[0], 1-significance_level/100))
            signif_df[station] = sign_lst # Add to significance dataframe
               
            

            
            
        # Transpose coef list
        numpy_array = np.array(coef_lst)
        transpose = numpy_array.T
        coef_lst_T = transpose.tolist()
        
        
        # Create dataset of regression results (R^2, R^2_wind, R^2_u2, R^2_v2, rmse, regression coefficients)
        if wind_model == 'NearestPoint':
            reg_results_lst.append(xr.Dataset(data_vars=dict(r2=(['station'], R2_total_lst),
                                                            r2_wind=(['station'], R2_wind_lst),
                                                            r2_u2=(['station'], R2_u2_lst),
                                                            r2_v2=(['station'], R2_v2_lst),
                                                            rmse=(['station'], rmse_lst),
                                                            constant=(['station'], intercept_lst),
                                                            u2=(['station'], coef_lst_T[0]),
                                                            v2=(['station'], coef_lst_T[1]),
                                                            trend=(['station'], coef_lst_T[2]),
                                                            station = stations)))
        elif wind_model == 'Timmerman':
            reg_results_lst.append(xr.Dataset(data_vars=dict(r2=(['station'], R2_total_lst),
                                                            r2_wind=(['station'], R2_wind_lst),
                                                            r2_u2=(['station'], R2_u2_lst),
                                                            r2_v2=(['station'], R2_v2_lst),
                                                            rmse=(['station'], rmse_lst),
                                                            constant=(['station'], intercept_lst),
                                                            channel_u2=(['station'], coef_lst_T[0]),
                                                            channel_v2=(['station'], coef_lst_T[1]),
                                                            south_u2=(['station'], coef_lst_T[2]),
                                                            south_v2=(['station'],coef_lst_T[3]),
                                                            midwest_u2=(['station'], coef_lst_T[4]),
                                                            midwest_v2=(['station'],coef_lst_T[5]),
                                                            mideast_u2=(['station'], coef_lst_T[6]),
                                                            mideast_v2=(['station'], coef_lst_T[7]),
                                                            northwest_u2=(['station'], coef_lst_T[8]),
                                                            northwest_v2=(['station'], coef_lst_T[9]),
                                                            northeast_u2=(['station'], coef_lst_T[10]),
                                                            northeast_v2=(['station'], coef_lst_T[11]),
                                                            trend=(['station'], coef_lst_T[12]),
                                                            station = stations)))
        elif wind_model == 'Dangendorf':
            reg_results_lst.append(xr.Dataset(data_vars=dict(r2=(['station'], R2_total_lst),
                                                            r2_wind=(['station'], R2_wind_lst),
                                                            r2_neg=(['station'], R2_u2_lst),
                                                            r2_pos=(['station'], R2_v2_lst),
                                                            rmse=(['station'], rmse_lst),
                                                            constant=(['station'], intercept_lst),
                                                            neg_corr_region=(['station'], coef_lst_T[0]),
                                                            pos_corr_region=(['station'], coef_lst_T[1]),
                                                            trend=(['station'], coef_lst_T[2]),
                                                            station = stations)))
            
        
                                 
        # Put all station datasets in one dataset
        timeseries_lst1.append(xr.concat(timeseries_lst, dim=stations).rename({"concat_dim":"station"}))          
        
        # Create dataset of significance
        signif_ds_lst.append(xr.Dataset.from_dataframe(signif_df.T).rename({'index':'station'}))
        
    # Put all model datasets in one dataset
    timeseries_dataset = xr.concat(timeseries_lst1, dim=wind_data.model.values).rename({"concat_dim":"model"})                           
    results_dataset = xr.concat(reg_results_lst, dim=wind_data.model.values).rename({"concat_dim":"model"})
    signif_dataset = xr.concat(signif_ds_lst, dim=wind_data.model.values).rename({"concat_dim":"model"})
        
    # Save the datasets
    save_nc_data(results_dataset, 'cmip6', 'Regression results', f'results_{wind_model}_{data_type}')
    save_nc_data(timeseries_dataset, 'cmip6', 'Regression results', f'timeseries_{wind_model}_{data_type}')
    save_nc_data(signif_dataset, 'cmip6', 'Regression results', f'significance_{wind_model}_{data_type}')    
        
    
    # Save the scalers
    file = open(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/scalers_{wind_model}.pkl', 'wb')
    pickle.dump(scalers, file)
    file.close()
        
    return(results_dataset, timeseries_dataset, signif_dataset)









"""
USEFULL FUNCTIONS
-------------------

"""

def regression_names(model):
    
    if model == 'NearestPoint':
        regg_names = ['u$^2$', 'v$^2$', 'trend']
        
    elif model == 'Timmerman':
        regg_names = ['Channel u$^2$', 'Channel v$^2$', 'South u$^2$',  'South v$^2$',  'Mid-West u$^2$', 
                      'Mid-West v$^2$', 'Mid-East u$^2$', 'Mid-East v$^2$', 'North-West u$^2$', 
                      'North-West v$^2$', 'North-East u$^2$', 'North-East v$^2$', 'trend']
        
    elif model == 'Dangendorf':
        regg_names = ['Negative corr region', 'Positive corr region', 'trend']
            
    return regg_names, regg_names[:-1]



def significance_test(p, alpha):
    if p < alpha: return True
    else: return False
    
    
def R2_var(df, y, var, regression_):
    """
    For var choose 
    for model = 'NearestPoint' ['wind total', 'u$^2$', 'v$^2$']
    for model = 'Timmerman'    ['wind total', 'u$^2$ total', 'v$^2$ total']
    for model = 'Dangndorf'    ['wind total', 'Negative corr region', 'Positive corr region']
    """
    
    x_wind = pd.DataFrame(data={'time': df.index.values, 
                                var : df[var].values})
    x_wind = x_wind.set_index('time')
    x_wind = x_wind[x_wind.index.isin(y.index)]
    
    #fit = regression_.fit(x_wind, y)
    score = regression_.score(x_wind,y)
    
    return score
    