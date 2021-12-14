"""
File containing the Python functions to import the correct data used for the regression between AMV and the atmospheric contribution to sea level height at the Dutch coast.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
lagged_regression_obs.ipynb

"""

# Import necessary packages

import pandas as pd
import xarray as xr

import statsmodels.api as sm


from scipy.signal import detrend

AMV_names = ['HadISSTv2', 'ERSSTv5', 'COBE-SST2']

"""
Practical functions
-------------------


"""




def df_smooth(df, window):
    df_lo = df.copy()
    
    for column in df:
        frac = window/df[column].values.size
        df_lo[column] = lowess(df[column].values, df.index.values, frac, return_sorted=False)
        
        
    return df_lo



# Declare global variables
wind_labels = ['NearestPoint', 'Timmerman', 'Dangendorf']

lowess = sm.nonparametric.lowess

"""
Import data
------------


"""

def import_obs_ac_slh_data(smoothed = False, window = 21):
    '''
    Function to import the atmospheric contribution to sea-level time series resulting from the regression between observational data
    
    '''
    
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Regression results/'
    
    
    for wl in wind_labels:
        
        # Import data results from regression
        df_tot_era5 = pd.read_csv(f'{path}timeseries_{wl}_era5.csv', header =[0,1], index_col = 0)
        df_tot_20cr = pd.read_csv(f'{path}timeseries_{wl}_20cr.csv', header =[0,1], index_col = 0)
        
        if wl == 'NearestPoint':
            
            # Create new dataframe only consisting of atmospheric contribution
            df_era5 = pd.DataFrame({'time' : df_tot_era5.index})
            df_era5 = df_era5.set_index('time')
            df_20cr = pd.DataFrame({'time' : df_tot_20cr.index})
            df_20cr = df_20cr.set_index('time')
        
        df_era5[wl] = df_tot_era5['Average', 'wind total']
        df_20cr[wl] = df_tot_20cr['Average', 'wind total']


    # detrend the dataframes
    df_era5 = df_era5.apply(detrend)
    df_20cr = df_20cr.apply(detrend)
    
    # apply lowess smoothing filter if smoothed = True
    if smoothed == True:
        df_era5 = df_smooth(df_era5, window)
        df_20cr = df_smooth(df_20cr, window)
    
    return df_era5, df_20cr




def import_AMV_data(smoothed = False, window = 21):
    '''
    Function to import the observational AMV data
    
    '''
    
    
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/AMV/Data/AMV/'
    
    # Deseasonalized and detrended
    AMV_ds_dt_had = xr.open_dataset(path + 'AMO_ds_dt_raw_had.nc') 
    AMV_ds_dt_ersst = xr.open_dataset(path + 'AMO_ds_dt_raw_ersst.nc') 
    AMV_ds_dt_cobe = xr.open_dataset(path + 'AMO_ds_dt_raw_cobe.nc') 

    # Change variable names
    AMV_ds_dt_had = AMV_ds_dt_had.rename({'__xarray_dataarray_variable__': 'SST index'})
    AMV_ds_dt_ersst = AMV_ds_dt_ersst.rename({'__xarray_dataarray_variable__': 'SST index'})
    AMV_ds_dt_cobe = AMV_ds_dt_cobe.rename({'__xarray_dataarray_variable__': 'SST index'})

    # Obtain annual averages 
    AMV_ds_dt_had = AMV_ds_dt_had.groupby('time.year').mean('time')
    AMV_ds_dt_ersst = AMV_ds_dt_ersst.groupby('time.year').mean('time')
    AMV_ds_dt_cobe = AMV_ds_dt_cobe.groupby('time.year').mean('time')
    
    # Change all time coordinate names to time (instead of year)
    AMV_had = AMV_ds_dt_had.rename({'year': 'time'})
    AMV_ersst = AMV_ds_dt_ersst.rename({'year': 'time'})
    AMV_cobe = AMV_ds_dt_cobe.rename({'year': 'time'})
    
    lst = [AMV_had['SST index'].to_pandas(), AMV_ersst['SST index'].to_pandas(), AMV_cobe['SST index'].to_pandas()]
    
    AMV_data = pd.concat(lst, keys = AMV_names, axis=1)
    
    if smoothed == True:
        AMV_data = df_smooth(AMV_data, window)
    
    return AMV_data