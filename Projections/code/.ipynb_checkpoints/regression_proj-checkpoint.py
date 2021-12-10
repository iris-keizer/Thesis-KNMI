"""
File containing the Python functions to perform a regression between sea level data and wind data to prepare for the projection.
The difference with the other regression is that the trend is not used as a regressor.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
regression_proj.py

"""


# Import necessary packages
import copy
import pickle
import numpy as np
import xarray as xr
import pandas as pd

from scipy.signal import detrend

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.linear_model import LinearRegression as linr



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




def regression_names(model):
    
    if model == 'NearestPoint':
        regg_names = ['u$^2$', 'v$^2$']
        
    elif model == 'Timmerman':
        regg_names = ['Channel u$^2$', 'Channel v$^2$', 'South u$^2$',  'South v$^2$',  'Mid-West u$^2$', 
                      'Mid-West v$^2$', 'Mid-East u$^2$', 'Mid-East v$^2$', 'North-West u$^2$', 
                      'North-West v$^2$', 'North-East u$^2$', 'North-East v$^2$']
        
    elif model == 'Dangendorf':
        regg_names = ['Negative corr region', 'Positive corr region']
            
    return regg_names


def regression_names_trend(model):
    
    if model == 'NearestPoint':
        regg_names = ['u$^2$', 'v$^2$', 'trend']
        
    elif model == 'Timmerman':
        regg_names = ['Channel u$^2$', 'Channel v$^2$', 'South u$^2$',  'South v$^2$',  'Mid-West u$^2$', 
                      'Mid-West v$^2$', 'Mid-East u$^2$', 'Mid-East v$^2$', 'North-West u$^2$', 
                      'North-West v$^2$', 'North-East u$^2$', 'North-East v$^2$', 'trend']
        
    elif model == 'Dangendorf':
        regg_names = ['Negative corr region', 'Positive corr region', 'trend']
            
    return regg_names




def timmerman_region_names(): 
    """
    Function to obtain timmerman region names as list
    
    """
    return ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']



def save_csv_data(data, name): 
    """
    Function to save data as .csv file
    
    For folder choose ['observations', 'cmip6'], 
    for variable choose ['Wind', 'SLH', 'Pressure', 'SST', 'Regression results']
    
    """
    data.to_csv(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/Projections/{name}.csv")



    

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
    
    fit = regression_.fit(x_wind, y)
    score = regression_.score(x_wind,y)
    
    return score
    
    
    
def significance_test(p, alpha):
    if p < alpha: return True
    else: return False
    
    
    

# Declare global variables
stations = station_names()
regions = timmerman_region_names()
significance_level = 95
alphas = [0.03582961338564776, 0.07386858364808699, 0.09391780126679188, 0.13034639992652514, 
          0.21143711191718467, 0.0657308019386455, 0.02740434989586052]


models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'BCC-ESM1',
       'CAMS-CSM1-0', 'CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2',
       'CNRM-CM6-1', 'CNRM-ESM2-1', 'CanESM5', 'CanESM5-CanOE',
       'EC-Earth3', 'EC-Earth3-AerChem', 'EC-Earth3-CC', 'EC-Earth3-Veg',
       'EC-Earth3-Veg-LR', 'FGOALS-f3-L', 'GFDL-CM4', 'GFDL-ESM4',
       'GISS-E2-1-G', 'GISS-E2-1-H', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
       'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'MIROC6',
       'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
       'NESM3', 'NorCPM1', 'UKESM1-0-LL']



"""
REGRESSION FUNCTION
-------------------


"""


def regression_cmip6(wind_data, zos, wind_model = 'NearestPoint', data_type = 'historical'):
    """
    Function to perform the regression between the cmip6 sea level and wind data
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    """
    
    # Perform regression with zos data untill 1980, thereafter zos accelerates
    zos = zos.where(zos.time <= 1980, drop=True)
    
    # Get names of all regression and wind regression coefficients
    regg_names = regression_names(wind_model)
    
    
    # Create lists to save data variables 
    timeseries_dfs = []
    
    # Create dataframe for significance
    signif_df = pd.DataFrame(data=dict(reggression_contributor=regg_names+['total'])) # Create dataframe 
    signif_df = signif_df.set_index('reggression_contributor')
        
    # Create dataframe for scales of standardization
    scalers = {}
    
    # Create dataframe for results
    if wind_model == 'NearestPoint':
        variables = ['R$^2$', 'R$^2_{u^2}$', 'R$^2_{v^2}$', 'rmse', 'constant'] + regg_names
    elif wind_model == 'Timmerman':
        variables = ['R$^2$', 'R$^2_{u^2}$', 'R$^2_{v^2}$', 'rmse', 'constant'] + regg_names

    elif wind_model == 'Dangendorf':
        variables = ['R$^2$', 'R$^2_{neg}$', 'R$^2_{pos}$', 'rmse', 'constant'] + regg_names
        
    results_df = pd.DataFrame({'result':variables}) # dataframe to save regression results
    results_df = results_df.set_index('result')
    
    wc_df = pd.DataFrame({'time':wind_data.time.values}) # dataframe to save atmospheric contribution
    wc_df = wc_df.set_index('time')
    
    
    # Perform regression for each model
    for model in wind_data.model.values:

        signif_df[model] = ''
        results_df[model] = ''
        
        # Create detrended dataframe for the dependent variable
        y = pd.DataFrame(data={'time': zos.time.values,
                               'zos': detrend(zos.zos.sel(model=model).values)})
            
        y = y.set_index('time')
            
        if wind_model == 'NearestPoint':
                
            # Create non detrended dataframe with forcing parameters
            x_nd = pd.DataFrame(data={'time': wind_data.time.values, 
                                       'u$^2$' : wind_data.u2.sel(model=model).values, 
                                       'v$^2$' : wind_data.v2.sel(model=model).values})
            x_nd = x_nd.set_index('time')
                
                
            # Define regression
            regression_ = linr()

        elif wind_model == 'Timmerman':

            # Create non detrended dataframe with forcing parameters
            dfs = []
            dfs_nd = []
            for region in wind_data.tim_region.values:
                dfs.append(wind_data.sel(model=model, 
                                             tim_region=region, drop=True).to_dataframe())
                dfs.append(wind_data.sel(model=model, 
                                             tim_region=region, drop=True).to_dataframe())

            x_nd = pd.concat(dfs, axis=1, keys=wind_data.tim_region.values)


            # Define regression
            tss = TimeSeriesSplit(n_splits=5)
            regression_ = LassoCV(alphas=alphas, cv=tss, max_iter=10**(9))
                
                
            

        elif wind_model == 'Dangendorf':

            # Create non detrended dataframe with forcing parameters
            x_nd = pd.DataFrame(data={'time': wind_data.time.values, 
                                        'Negative corr region' : wind_data['Negative corr region'].sel(model=model).values, 
                                        'Positive corr region' : wind_data['Positive corr region'].sel(model=model).values})
            x_nd = x_nd.set_index('time')


            # Define regression
            regression_ = linr()
    
        
        # Drop nan values
        x_nd = x_nd.dropna()
        y = y.dropna()
        
        # Create detrended dataframe
        x = x_nd.apply(detrend)

        # Save scaler for projections and non-detrended dataframe
        scaler = StandardScaler()
        scalers[model] = scaler.fit(x)
            
        # Standardize the detrended and non-detrended dataframe using the same scale
        x = copy.deepcopy(x)
        x.iloc[:,:] = scaler.fit_transform(x)
        x_nd.iloc[:,:] = scalers[model].fit_transform(x_nd)

            
        # Create dataframes of equal time span
        y = y[y.index.isin(x.index)]
        x = x[x.index.isin(y.index)]
        
            
        # Fit the regression model and add results to lists
        if  wind_model == 'Timmerman':
            fit = regression_.fit(x, y.values.ravel())
            alpha = regression_.alpha_
            regression_ = Lasso(alpha)
                
                
        fit = regression_.fit(x, y.values.ravel())
        results_df[model]['constant'] = regression_.intercept_
        coefs = regression_.coef_.tolist()
        
        
        for i, reg_res in enumerate(regg_names):
            results_df[model][reg_res] = coefs[i]
            
        
        # Calculate rmse
        yhat = regression_.predict(x)
        mse = mean_squared_error(y, yhat) # Calculate insample mse
        results_df[model]['rmse'] = np.sqrt(mse)
            
            
        # Obtain dataframe containing timeseries resulting from regression
        df = pd.DataFrame(data=dict(time=x_nd.index))
        df = df.set_index('time')
        
            
        for i in range(len(regg_names)):
                
            df[regg_names[i]] = coefs[i] * x_nd[x_nd.columns[i]]
        
            
        df['total'] = df.sum(axis=1)
        wc_df[model] = df.sum(axis=1)
            
        if wind_model ==  'Timmerman':
            for i in range(len(regions)):
                df[regions[i]] = df[[regg_names[2*i], regg_names[2*i+1]]].sum(axis=1)
            df['u$^2$ total'] = df[[regg_names[0], regg_names[2], regg_names[4], regg_names[6], 
                                     regg_names[8], regg_names[10]]].sum(axis=1)
            df['v$^2$ total'] = df[[regg_names[1], regg_names[3], regg_names[5], regg_names[7], 
                                     regg_names[9], regg_names[11]]].sum(axis=1)
                
            
        timeseries_dfs.append(df)
        
            
        # Calculate R^2 values
        results_df[model]['R$^2$'] = regression_.score(x, y.values.ravel()) # R^2 for the whole regression
        if wind_model == 'NearestPoint':
            results_df[model]['R$^2_{u^2}$'] = R2_var(df, y, 'u$^2$', regression_)
            results_df[model]['R$^2_{v^2}$'] = R2_var(df, y, 'v$^2$', regression_)
        elif wind_model == 'Timmerman':
            results_df[model]['R$^2_{u^2}$'] = R2_var(df, y, 'u$^2$ total', regression_)
            results_df[model]['R$^2_{v^2}$'] = R2_var(df, y, 'v$^2$ total', regression_)
        elif wind_model == 'Dangendorf':
            results_df[model]['R$^2_{neg}$'] = R2_var(df, y, 'Negative corr region', regression_)
            results_df[model]['R$^2_{pos}$'] = R2_var(df, y, 'Positive corr region', regression_)
                    
                
        # Check significance for each regression contributor
        f_statistic, p_values = f_regression(x, y.values.ravel())

        for i, p_value in enumerate(p_values):
            signif_df[model][regg_names[i]] = significance_test(p_value, 1-significance_level/100)
            

        # Check significance for total regression
        x_total = df['total'].to_frame()[df['total'].to_frame().index.isin(y.index)]

        f_statistic, p_values = f_regression(x_total, y.values.ravel())
        signif_df[model]['total'] = significance_test(p_values[0], 1-significance_level/100)
        
               
    # Put all model datasets in one dataset
    timeseries_df = pd.concat(timeseries_dfs, axis=1, keys = models)   
    
    
    # Save data
    save_csv_data(results_df, f'{wind_model}_results')
    save_csv_data(timeseries_df, f'{wind_model}_timeseries')
    save_csv_data(wc_df, f'{wind_model}_wc_timeseries')
    save_csv_data(signif_df, f'{wind_model}_significance')
    
    # Save the scalers
    file = open(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/Projections/scalers_{wind_model}.pkl', 'wb')
    pickle.dump(scalers, file)
    file.close()

    
    
    return results_df, timeseries_df, signif_df

















def regression_cmip6_trend(wind_data, zos, wind_model = 'NearestPoint', data_type = 'historical'):
    """
    Function to perform the regression between the cmip6 sea level and wind data with trend as a forcing parameter in the regression
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    """
    
    # Perform regression with zos data untill 1980, thereafter zos accelerates
    zos = zos.where(zos.time <= 1980, drop=True)
    
    
    # Get names of all regression and wind regression coefficients
    regg_names = regression_names_trend(wind_model)
    wind_names = regg_names[:-1]
    
    
    # Create lists to save data variables 
    timeseries_dfs = []
    
    
    # Create dataframe for significance
    signif_df = pd.DataFrame(data=dict(reggression_contributor=regg_names+['total'])) # Create dataframe 
    signif_df = signif_df.set_index('reggression_contributor')
    
    
    # Create dataframe for scales of standardization
    scalers = {}
    
    
    # Create dataframe for results
    if wind_model == 'NearestPoint':
        variables = ['R$^2$', 'R$^2_{wind}$', 'R$^2_{u^2}$', 'R$^2_{v^2}$', 'rmse', 'constant'] + regg_names
    elif wind_model == 'Timmerman':
        variables = ['R$^2$', 'R$^2_{wind}$', 'R$^2_{u^2}$', 'R$^2_{v^2}$', 'rmse', 'constant'] + regg_names

    elif wind_model == 'Dangendorf':
        variables = ['R$^2$', 'R$^2_{wind}$', 'R$^2_{neg}$', 'R$^2_{pos}$', 'rmse', 'constant'] + regg_names
        
    results_df = pd.DataFrame({'result':variables}) # dataframe to save regression results
    results_df = results_df.set_index('result')
    
    wc_df = pd.DataFrame({'time':wind_data.time.values}) # dataframe to save atmospheric contribution
    wc_df = wc_df.set_index('time')
    
    
    # Perform regression for each model
    for model in wind_data.model.values:

        signif_df[model] = ''
        results_df[model] = ''
        
        # Create detrended dataframe for the dependent variable
        y = pd.DataFrame(data={'time': zos.time.values,
                               'zos': detrend(zos.zos.sel(model=model).values)})
            
        y = y.set_index('time')
            
        if wind_model == 'NearestPoint':
                
            # Create dataframe with forcing parameters
            x = pd.DataFrame(data={'time': wind_data.time.values, 
                                       'u$^2$' : wind_data.u2.sel(model=model).values, 
                                       'v$^2$' : wind_data.v2.sel(model=model).values,
                                        'trend': [j - wind_data.time.values[0] for j in wind_data.time.values]})
            x = x.set_index('time')
                
                
            # Define regression
            regression_ = linr()

        elif wind_model == 'Timmerman':

            # Create dataframe with forcing parameters
            dfs = []
            dfs_nd = []
            for region in wind_data.tim_region.values:
                dfs.append(wind_data.sel(model=model, 
                                             tim_region=region, drop=True).to_dataframe())
            
            x = pd.concat(dfs, axis=1, keys=wind_data.tim_region.values)
            x['trend'] = [j - wind_data.time.values[0] for j in wind_data.time.values]

            # Define regression
            tss = TimeSeriesSplit(n_splits=5)
            regression_ = LassoCV(alphas=alphas, cv=tss, max_iter=10**(9))
                
                
            

        elif wind_model == 'Dangendorf':

            # Create dataframe with forcing parameters
            x = pd.DataFrame(data={'time': wind_data.time.values, 
                                        'Negative corr region' : wind_data['Negative corr region'].sel(model=model).values, 
                                        'Positive corr region' : wind_data['Positive corr region'].sel(model=model).values,
                                          'trend': [j - wind_data.time.values[0] for j in wind_data.time.values]})
            x = x.set_index('time')


            # Define regression
            regression_ = linr()
    
        
        # Drop nan values
        x = x.dropna()
        y = y.dropna()
        

        # Save scaler for projections and non-detrended dataframe
        scaler = StandardScaler()
        scalers[model] = scaler.fit(x)
            
        # Standardize the detrended and non-detrended dataframe using the same scale
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
            regression_ = Lasso(alpha)
                
                
        fit = regression_.fit(x, y.values.ravel())
        results_df[model]['constant'] = regression_.intercept_
        coefs = regression_.coef_.tolist()
        
        
        for i, reg_res in enumerate(regg_names):
            results_df[model][reg_res] = coefs[i]
            
        
        # Calculate rmse
        yhat = regression_.predict(x)
        mse = mean_squared_error(y, yhat) # Calculate insample mse
        results_df[model]['rmse'] = np.sqrt(mse)
            
            
        # Obtain dataframe containing timeseries resulting from regression
        df = pd.DataFrame(data=dict(time=x_timeseries.index))
        df = df.set_index('time')
        
            
        for i in range(len(regg_names)):
                
            df[regg_names[i]] = coefs[i] * x_timeseries[x_timeseries.columns[i]]
        
            
        df['total'] = df.sum(axis=1)
        wc_df[model] = df[wind_names].sum(axis=1)
        df['wind total'] = df[wind_names].sum(axis=1)
            
        
        if wind_model ==  'Timmerman':
            for i in range(len(regions)):
                df[regions[i]] = df[[regg_names[2*i], regg_names[2*i+1]]].sum(axis=1)
            df['u$^2$ total'] = df[[regg_names[0], regg_names[2], regg_names[4], regg_names[6], 
                                     regg_names[8], regg_names[10]]].sum(axis=1)
            df['v$^2$ total'] = df[[regg_names[1], regg_names[3], regg_names[5], regg_names[7], 
                                     regg_names[9], regg_names[11]]].sum(axis=1)
                
            
        timeseries_dfs.append(df)
        
            
        # Calculate R^2 values
        results_df[model]['R$^2$'] = regression_.score(x, y.values.ravel()) # R^2 for the whole regression
        results_df[model]['R$^2_{wind}$'] = R2_var(df, y, 'wind total', regression_) # R^2 for atmospheric contribution
        if wind_model == 'NearestPoint':
            results_df[model]['R$^2_{u^2}$'] = R2_var(df, y, 'u$^2$', regression_)
            results_df[model]['R$^2_{v^2}$'] = R2_var(df, y, 'v$^2$', regression_)
        elif wind_model == 'Timmerman':
            results_df[model]['R$^2_{u^2}$'] = R2_var(df, y, 'u$^2$ total', regression_)
            results_df[model]['R$^2_{v^2}$'] = R2_var(df, y, 'v$^2$ total', regression_)
        elif wind_model == 'Dangendorf':
            results_df[model]['R$^2_{neg}$'] = R2_var(df, y, 'Negative corr region', regression_)
            results_df[model]['R$^2_{pos}$'] = R2_var(df, y, 'Positive corr region', regression_)
                    
                
        # Check significance for each regression contributor
        f_statistic, p_values = f_regression(x, y.values.ravel())

        for i, p_value in enumerate(p_values):
            signif_df[model][regg_names[i]] = significance_test(p_value, 1-significance_level/100)
            

        # Check significance for total regression
        x_total = df['total'].to_frame()[df['total'].to_frame().index.isin(y.index)]

        f_statistic, p_values = f_regression(x_total, y.values.ravel())
        signif_df[model]['total'] = significance_test(p_values[0], 1-significance_level/100)
        
               
    # Put all model datasets in one dataset
    timeseries_df = pd.concat(timeseries_dfs, axis=1, keys = models)   
    
    
    # Save data
    save_csv_data(results_df, f'{wind_model}_results')
    save_csv_data(timeseries_df, f'{wind_model}_timeseries')
    save_csv_data(wc_df, f'{wind_model}_wc_timeseries')
    save_csv_data(signif_df, f'{wind_model}_significance')
    
    # Save the scalers
    file = open(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/Projections/scalers_{wind_model}.pkl', 'wb')
    pickle.dump(scalers, file)
    file.close()

    
    
    return results_df, timeseries_df, signif_df


