"""
File containing the Python functions to import data in order to compare different regression results


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
comparison.ipynb

"""



# Import necessary packages
import pandas as pd
import numpy as np
import xarray as xr

from scipy.signal import detrend


"""
Practical functions
-------------------


"""




def station_names(): 
    """
    Function to obtain tide gauge station names as list
    
    """
    
    return ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']




def station_coords(): 
    """
    Function to obtain the coordinates of the tide gauge stations as a dataframe
    
    """
    
    
    # Necessary declarations to obtain tide gauge station coordinates
    path_locations = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/rlr_annual/filelist.txt'
    loc_num = [20, 22, 23, 24, 25, 32]
    col_names = ['id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality']
    
    # Create dataframe
    df = pd.read_csv(path_locations, sep=';', header=None, names=col_names)
    df = df.set_index('id')
    df = df.loc[loc_num, :]
    df['name'] = stations[:-1]
    df = df.set_index('name')
    df = df.drop(['coastline_code', 'station_code', 'quality'], axis=1)
    
    return df


def nodal_tides_potential(lat, time_years):
    """
    Function to obtain the nodal cycle
    
    """
    h2 = 0.6032
    k2 = 0.298

    #nodal cycle correction
    A = 0.44*(1+k2-h2)*20*(3*np.sin(lat*np.pi/180.)**2-1)/10  # mm to cm
    nodcyc = A*np.cos((2*np.pi*(time_years-1922.7))/18.61 + np.pi)
    
    return nodcyc.values




# Declare global variables
stations = station_names()







"""
OBSERVATIONS
------------

"""

def import_obs_slh_data():
    """
    Function that imports the tide gauge sea level height data as a pandas.dataframe
    
    """
    
    
    # Define paths to data
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/SLH/tg_data.csv'
    
    
    # Open data file
    tg_data_df = pd.read_csv(path)
    
    
    # Set time as index of dataframe
    tg_data_df = tg_data_df.set_index('time')
    
    # Remove nodal cycle
    dfs = []
    
    # Obtain coordinates of the tide gauge stations
    coord_df = station_coords()
    coord_df = coord_df.T
    coord_df['Average'] = coord_df.mean(axis=1)
    coord_df = coord_df.T
    
    for stat in stations:
        dt = tg_data_df[stat]

        #Remove nodal cycle
        nodal_correction = nodal_tides_potential(coord_df['lat'].loc[stat], dt.index)

        dt = dt - nodal_correction
        data_df = pd.DataFrame(data={'time' : dt.index, stat : dt})
        data_df = data_df.set_index('time')
        dfs.append(data_df)

    tg_data_df = pd.concat(dfs, axis=1)
    
    return tg_data_df




def import_cmip6_slh_data(data_type = 'historical'):
    """
    Function that imports cmip6 sea level data
    
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245' 'ssp370', 'ssp585']
    
    """
    
    
    # Define paths to data
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/SLH/slh_annual_{data_type}.nc'
    
    
    # Open data file
    data_xr = xr.open_dataset(path) 
    
    
    # I do not remove the nodal cycle

    
    return data_xr








"""
COMPARISON
----------

"""




def import_reg_results(output, data_type):
    """
    Function to import the dataframes containing the regression results for observational data
    
    For output choose ['results', 'timeseries']
    
    For data_type choose ['era5', '20cr', 'historical']
    
    """
    
    if output == 'results':
        index_col = 'station'
    elif output == 'timeseries':
        index_col = 'time'
    
    
    if data_type == 'era5' or data_type == '20cr':
        
        
        # Define path
        path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Regression results/'
        
        if output == 'results':
            
            
            # Import the files
            np = pd.read_csv(path+f'{output}_NearestPoint_{data_type}.csv', index_col = 'station')
            tim = pd.read_csv(path+f'{output}_Timmerman_{data_type}.csv', index_col = 'station')
            dang = pd.read_csv(path+f'{output}_Dangendorf_{data_type}.csv', index_col = 'station')
            
            tim['u$^2$'] = tim[['Channel u$^2$', 'South u$^2$', 'Mid-West u$^2$', 'Mid-East u$^2$', 'North-West u$^2$', 'North-East u$^2$']].sum(axis=1)
            tim['v$^2$'] = tim[['Channel v$^2$', 'South v$^2$', 'Mid-West v$^2$', 'Mid-East v$^2$', 'North-West v$^2$', 'North-East v$^2$']].sum(axis=1)
            
            
        elif output == 'timeseries':
            
            
            # Import the files
            np = pd.read_csv(path+f'{output}_NearestPoint_{data_type}.csv', header = [0,1,2])
            tim = pd.read_csv(path+f'{output}_Timmerman_{data_type}.csv', header = [0,1,2])
            dang = pd.read_csv(path+f'{output}_Dangendorf_{data_type}.csv', header = [0,1,2])
        
       
            # Set index
            np = np.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
            tim = tim.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
            dang = dang.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
        
            # Set index name
            np.index.names = ['time']
            tim.index.names = ['time']
            dang.index.names = ['time']
            
            
            # Drop extra row
            np = np.droplevel(axis=1, level=2)
            tim = tim.droplevel(axis=1, level=2)
            dang = dang.droplevel(axis=1, level=2)
            
            
    elif data_type == 'historical':
        
        
        # Define path
        path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/'


        # Import the files
        np = xr.open_dataset(path+f'{output}_NearestPoint_{data_type}.nc')
        tim = xr.open_dataset(path+f'{output}_Timmerman_{data_type}.nc')
        dang = xr.open_dataset(path+f'{output}_Dangendorf_{data_type}.nc')

        
    return np, tim, dang







"""
MODEL SELECTION
---------------

"""


def detrend_dim(da, dim, deg=1): 
    """
    Function that detrends the data from a dataarray along a single dimension
    deg=1 for linear fit
    
    """
    
    p = da.polyfit(dim=dim, deg=deg)
    coord = da[dim] - da[dim].values[0]
    trend = coord*p.polyfit_coefficients.sel(degree=1)
    return da - trend







def import_data_model_selection():
    
    
    # Import 20CR observations
    
    
    # Define path
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Regression results/'
    
    # Import the files
    np = pd.read_csv(path+f'timeseries_NearestPoint_20cr.csv', header = [0,1,2])
    tim = pd.read_csv(path+f'timeseries_Timmerman_20cr.csv', header = [0,1,2])
    dang = pd.read_csv(path+f'timeseries_Dangendorf_20cr.csv', header = [0,1,2])
    
    # Set index
    np = np.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
    tim = tim.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
    dang = dang.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
        
    # Set index name
    np.index.names = ['time']
    tim.index.names = ['time']
    dang.index.names = ['time']
            
            
    # Drop extra row
    np = np.droplevel(axis=1, level=2)
    tim = tim.droplevel(axis=1, level=2)
    dang = dang.droplevel(axis=1, level=2)
            
    
    # Create one dataframe only containing 'Average' station and wind contribution to SLH
    # whereof the data is detrended
    df = pd.DataFrame({'time': np.index.values, 
                       'NearestPoint': detrend(np['Average', 'wind total']),
                       'Timmerman': detrend(tim['Average', 'wind total']), 
                       'Dangendorf': detrend(dang['Average', 'wind total'])})


    detrended_timeseries_20cr = df.set_index('time')
    
    
    # Import CMIP
    
    # Define path
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/'


    # Import the files
    np = xr.open_dataset(path+f'timeseries_NearestPoint_historical.nc')
    tim = xr.open_dataset(path+f'timeseries_Timmerman_historical.nc')
    dang = xr.open_dataset(path+f'timeseries_Dangendorf_historical.nc')
        
    # Select data and create dataframe
    np = np.wind_total.sel(station='Average', drop = True)
    tim = tim.wind_total.sel(station='Average', drop = True)
    dang = dang.wind_total.sel(station='Average', drop = True)

    # Detrend data
    np = detrend_dim(np, 'time')
    tim = detrend_dim(tim, 'time')
    dang = detrend_dim(dang, 'time')
    
    # Create dataframe
    np = np.to_pandas().T
    tim = tim.to_pandas().T
    dang = dang.to_pandas().T
    
    detrended_timeseries_cmip6 = pd.concat([np, tim, dang], axis = 1,  keys = ['NearestPoint', 'Timmerman', 'Dangendorf'])

    # Create data of equal time span
    detrended_timeseries_20cr = detrended_timeseries_20cr[
        detrended_timeseries_20cr.index.isin(detrended_timeseries_cmip6.index.values)]
    
    detrended_timeseries_cmip6 = detrended_timeseries_cmip6[
        detrended_timeseries_cmip6.index.isin(detrended_timeseries_20cr.index.values)]
    
    
    return detrended_timeseries_20cr, detrended_timeseries_cmip6