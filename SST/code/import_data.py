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


# Declare global variables
wind_labels = ['NearestPoint', 'Timmerman', 'Dangendorf']
AMV_names = ['HadISSTv2', 'ERSSTv5', 'COBE-SST2']
lowess = sm.nonparametric.lowess



"""
Practical functions
-------------------


"""




def df_smooth(df, window):
    '''
    Function to smooth a dataframe 
    '''
    df_lo = df.copy()
    
    for column in df:
        frac = window/df[column].values.size
        df_lo[column] = lowess(df[column].values, df.index.values, frac, return_sorted=False)
        
        
    return df_lo

def da_smooth(da, window):
    '''
    Function to smooth a dataarray
    
    '''
    years = da.year.values
    frac = frac = window / da.year.size
    
    def lowess_1d(data):
        return lowess(data, years, frac, return_sorted = False)
    
    
    da_low = xr.apply_ufunc(lowess_1d, da, input_core_dims = [['year']], output_core_dims = [['year']], vectorize = True)
    
    return da_low


def detrend_dim(da, dim, deg=1): # deg = 1 for linear fit
        '''
        Function that detrends the data from a dataarray along a single dimension
        '''
    
        p = da.polyfit(dim=dim, deg=deg)
        coord = da.year - da.year.values[0]
        trend = coord*p.polyfit_coefficients.sel(degree=1)
        return da - trend


def xr_select_region(data, coords):
    '''
    Function that selects a given region of data for xarray dataset  
    
    Coords should have shape: [longitude minimum, longitude maximum, latitude minimum, latitude maximum]
    '''
    
    data = data.where(data.lon>=coords[0], drop=True)
    data = data.where(data.lon<=coords[1], drop=True)
    data = data.where(data.lat>=coords[2], drop=True)
    data = data.where(data.lat<=coords[3], drop=True)
    return data


"""
Import observational data
-------------------------


"""

def import_obs_ac_slh_data(smoothed = False, window = 31):
    '''
    Function to import the atmospheric contribution to sea-level time series resulting from the regression between observational data
    
    '''
    
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Regression results/fullperiod/'
    
    
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




def import_obs_sst_data(smoothed = False, window = 31):
    '''
    Function to import the observational sea surface temperature and skin temperature data products
    
    '''
    
    # Declare paths to data 
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/SST/'

    # Import SST and SKT data
    SKT = xr.open_dataset(path + 'skt.mon.mean.nc') 
    SST = xr.open_dataset(path + 'sst.mnmean.nc') 

    # Obtain annual averages 
    SKT = SKT.groupby('time.year').mean('time')
    SST = SST.groupby('time.year').mean('time')

    # Delete 2021
    SKT = SKT.where((SKT.year<2021), drop=True)
    SST = SST.where((SST.year<2021), drop=True)

    # Shift longitudes to -180-180
    SKT.coords['lon'] = (SKT.lon + 180) % 360 - 180
    SST.coords['lon'] = (SST.lon + 180) % 360 - 180

    # Sort dataarrays
    SKT = SKT.sortby(SKT.lon, ascending=True)
    SKT = SKT.sortby(SKT.lat, ascending=True)
    SST = SST.sortby(SST.lon, ascending=True)
    SST = SST.sortby(SST.lat, ascending=True)

    # Select North Atlantic region
    SKT = xr_select_region(SKT, [-100, 10, 0, 88])
    SST = xr_select_region(SST, [-100, 10, 0, 88])

    # Change dataset to dataarray
    SKT = SKT.skt
    SST = SST.sst

    # Select same region for both dataarrays
    SKT = SKT.where(SST.notnull(), drop=True)
    SST = SST.where(SKT.notnull(), drop=True)

    # Remove trend from data
    SKT = detrend_dim(SKT, 'year')
    SST = detrend_dim(SST, 'year')

    # Convert SST from degree Celsius to degree Kelvin
    SST = SST + 273.15
    
    
    if smoothed == True:
        SST = da_smooth(SST, window)
        SKT = da_smooth(SKT, window)
    

    
    return SST, SKT

