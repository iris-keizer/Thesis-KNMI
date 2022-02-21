"""
File containing the Python functions to load and prepare the data used for the regression between SST or AMV and atmospheric contribution to sea level change. 

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
cmip6_data_preparation.ipynb


"""


# Import necessary packages
import dask
import xarray as xr 


"""
Practical functions
-------------------


"""




def save_nc_data(data, folder, variable, name): 
    """
    Function to save data as NETCDF4 file
    
    For folder choose ['observations', 'cmip6'], for variable choose ['Wind', 'SLH', 'Pressure', 'SST']
    
    """
    data.to_netcdf(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.nc", mode='w')
    


    
     
"""
PREPARE CMIP6 DATA
--------------------------

Functions to import the CMIP6 data and put all models in a dataset.
Also other necessary changes are made to get the data ready for the analysis



"""

   


def prep_tos_cmip6(data_type = 'historical'):
    '''
    Function to prepare the cmip6 tos data.
    '''
    dask.config.set({"array.slicing.split_large_chunks": False})
    
    # Define path to cmip6 data
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/SST/cmip6_tos_{data_type}/'
    
    # Open data files as dataset
    dataset_annual = xr.open_mfdataset(f'{path}cmip6_tos_{data_type}_*.nc')
    
    # Select area
    dataset_annual = dataset_annual.where(dataset_annual.lat > 0, drop=True)
    dataset_annual = dataset_annual.where(dataset_annual.lon > -90, drop=True)
    dataset_annual = dataset_annual.where(dataset_annual.lon < 90, drop=True)
    
    # Change time to integer
    dataset_annual.coords['time'] = dataset_annual.coords['time'].astype(int)
    
    # Change coordinate and variable names
    dataset_annual = dataset_annual.rename({"CorrectedReggrided_tos":"tos"})
    
    # Save annual data as netcdf4           
    save_nc_data(dataset_annual.tos, 'cmip6', 'SST', f'tos_annual_{data_type}')    
    
    return dataset_annual


def prep_amv_cmip6(data, data_type = 'historical'):
    '''
    Function to prepare the cmip6 amv data.
    '''
    
    # Select area
    data = data.where(data.lat < 60, drop=True)
    data = data.where(data.lon > -80, drop=True)
    data = data.where(data.lon < 0, drop=True)
    
    # Detrend SST at each grid point by subtracting ensemble mean
    ensemble_mean = data.mean(dim='model')
    data = data - ensemble_mean
    
    # Average over this area
    data = data.mean(dim=['lat', 'lon'])
    
    # Change name
    data = data.rename({"tos":"amv"})
    
    # Save annual data as netcdf4           
    save_nc_data(data.amv, 'cmip6', 'AMV', f'amv_annual_{data_type}')  
    
    
    return data

def prep_amv_cmip6_simple(data, data_type = 'historical'):
    '''
    Function to prepare the cmip6 amv data.
    '''
    
    # Select area
    data = data.where(data.lat < 60, drop=True)
    data = data.where(data.lon > -80, drop=True)
    data = data.where(data.lon < 0, drop=True)
    
    # Average over this area
    data = data.mean(dim=['lat', 'lon'])
    
    # Obtain anomalies
    data = data - data.mean(dim='time')
    
    # Change name
    data = data.rename({"tos":"amv"})
    
    # Save annual data as netcdf4           
    save_nc_data(data.amv, 'cmip6', 'AMV', f'amv_simple_annual_{data_type}')  
    
    
    return data