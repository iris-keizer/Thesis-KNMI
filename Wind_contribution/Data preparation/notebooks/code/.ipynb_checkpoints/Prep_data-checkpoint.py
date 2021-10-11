"""
File containing the Python functions to load and prepare the data used for the regression between wind stress and sea level height along the Dutch coast. 

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
AnnualAverages.ipynb



"""


# Import necessary packages
import xarray as xr
import numpy as np

# Function to save data
def save_nc_data(data, folder, variable, name): # For folder choose ['observations', 'cmip6'], for variable choose ['Wind', 'SLH', 'Pressure', 'SST']
    data.to_netcdf(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.nc")
    



"""
Observational data
--------------

Obtain annual averages of observational data, obtain wind stress and make changes such that data can be used for analysis
"""

# Function that imports wind data and calculates wind stress and annual averages
def prep_wind_era5():
    # Define the path to code
    path_fp = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Wind/wind_era5_fp.nc' #1950 - 1978
    path_sp = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Wind/wind_era5_sp.nc' #1979 - present

    # Open data file
    data_fp = xr.open_dataset(path_fp) #1950 - 1978
    data_sp = xr.open_dataset(path_sp) #1979 - present


    # Fix time problems
    # Data also contains expver = 5 but this has nan values except for last two months where expver = 1 
    data_sp = data_sp.sel(expver = 1)
    
    
    # Select smaller area of data 
    data_fp = data_fp.where((data_fp.latitude > 40) & (data_fp.latitude < 90), drop=True)
    data_fp = data_fp.where((data_fp.longitude > -40) & (data_fp.longitude < 30), drop=True)
    data_sp = data_sp.where((data_sp.latitude > 40) & (data_sp.latitude < 90), drop=True)
    data_sp = data_sp.where((data_sp.longitude > -40) & (data_sp.longitude < 30), drop=True)


    # Add the two datasets
    dataset = xr.concat([data_fp, data_sp], dim='time')


    # Obtain speed and stress for monthly averaged data

    # Speed
    dataset = dataset.assign(speed = np.hypot(dataset["u10"],dataset["v10"]))

    # Stress
    dataset = dataset.assign(u2 = dataset["u10"]**2*np.sign(dataset["u10"])) #  U
    dataset = dataset.assign(v2 = dataset["v10"]**2*np.sign(dataset["v10"])) # V


    # Calculate annual averages 
    dataset_annual = dataset.groupby('time.year').mean('time')
    
    # Remove 2021 
    dataset_annual = dataset_annual.where(dataset_annual <2021, drop=True)
    
    
    return dataset_annual


















"""
CMIP6 data
--------------

Make datasets and select regions such that data can be used for analysis

"""

# Define path to cmip6 data
data_dir = '/Volumes/Iris 300 GB/CMIP6'