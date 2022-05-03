"""
File containing the Python functions to import and preprocess the data used for the regression between wind stress and sea level height along the Dutch coast. 

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI


These functions are used in the notebooks:
prepare_obs.ipynb
prepare_cmip6.ipynb


"""


# Import necessary packages
import xarray as xr 
import numpy as np
import pandas as pd
import glob as gb
import copy as cp
import os



"""
Practical functions
-------------------


"""


def station_names(): 
    """
    Function to obtain tide gauge station names as a list
    
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
    df['name'] = station_names()[:-1]
    df = df.set_index('name')
    df = df.drop(['coastline_code', 'station_code', 'quality'], axis=1)
    
    return df


def save_nc_data(data, folder, variable, name): 
    """
    Function to save data as NETCDF4 file
    
    For folder choose ['observations', 'cmip6'], for variable choose ['Wind', 'SLH', 'Pressure', 'SST']
    
    """
    
    data.to_netcdf(f"/Users/iriskeizer/Documents/Wind effect/Data/{folder}/{variable}/{name}.nc", mode='w')
    
    
def save_csv_data(data, folder, variable, name): 
    """
    Function to save data as .csv file
    
    For folder choose ['observations', 'cmip6'], for variable choose ['Wind', 'SLH', 'Pressure', 'SST']
    
    """
    
    data.to_csv(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.csv")


    
stations = station_names()




"""
PREPARE OBSERVATIONAL DATA
--------------------------

Functions to obtain annual averages of the data and make changes to the datasets such that they can
be used for the analysis

For wind data:
- obtain wind forcing variables (multiply by absolute value)


"""




def prep_tg_data_obs():
    """
    Function to prepare the observational tide gauge sea level height data for the analysis
    
    
    """
    
    # Define paths to data
    path_tg = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/SLH/rlr_annual'
    path_locations = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/SLH/rlr_annual/filelist.txt'
    
    # Import tide gauge sea level data
    loc_num = [20, 22, 23, 24, 25, 32]
    col_names = ['id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality']
    filelist_df = pd.read_csv(path_locations, sep=';', header=None, names=col_names)
    filelist_df = filelist_df.set_index('id')
    filelist_df = filelist_df.loc[loc_num, :]
    names_col = ('time', 'height', 'interpolated', 'flags')
    
    for i in range(len(loc_num)):
            tg_data = pd.read_csv(path_tg + '/data/' + str(loc_num[i]) + 
                                  '.rlrdata', sep=';', header=None, names=names_col)
            tg_data = tg_data.drop(['interpolated', 'flags'], 1)
            tg_data = tg_data.set_index('time')
            
            # Data before 1890 is incorrect
            tg_data = tg_data[tg_data.index>=1890] 
            
            # Calculate anomalies over whole period
            tg_data.height = tg_data.height - tg_data.height.mean() 

            if i==0:
                tg_data_df = pd.DataFrame(data=dict(time=tg_data.index, col_name=tg_data.height))
                tg_data_df = tg_data_df.set_index('time')
                tg_data_df.columns  = [str(loc_num[i])] 
            else:
                tg_data_df[str(loc_num[i])] = tg_data.height
            

    tg_data_df = tg_data_df.rename(columns={"20": stations[0], 
                              "22": stations[1], "23": stations[2],
                              "24": stations[3], "25": stations[4],
                              "32": stations[5]})

    #tg_data_df = tg_data_df.interpolate(method='slinear') # Interpolate in case of any nan values (but there are not)
    tg_data_df['Average'] = tg_data_df.mean(axis=1) # Add column containing the average of the stations 
    tg_data_df = tg_data_df*0.1 # mm -> cm
    
    
    # Create xarray dataset
    tg_data_xr = xr.Dataset.from_dataframe(tg_data_df)
    
    
    # Save annual data as netcdf4       
    save_nc_data(tg_data_xr, 'observations', 'SLH', 'tg_data')
    
    
    # Save annual data as dataframe
    save_csv_data(tg_data_df, 'observations', 'SLH', 'tg_data')
    
    
    
    return tg_data_df, tg_data_xr
    
        
        



def prep_wind_data_obs(data_type = 'era5'):
    """
    Function to prepare the observational wind data for the analysis
    
    For data_type choose ['era5', '20cr']
    
    
    """
    
    if data_type == 'era5':
        
        
        # Define the paths to code which comes from two different dataproducts
        path_fp = '/Users/iriskeizer/Documents/Wind effect/Data/observations/Wind/wind_era5_fp.nc' #1950 - 1978
        path_sp = '/Users/iriskeizer/Documents/Wind effect/Data/observations/Wind/wind_era5_sp.nc' #1979 - present


        # Open data file
        data_fp = xr.open_dataset(path_fp) #1950 - 1978
        data_sp = xr.open_dataset(path_sp) #1979 - present


        # Add the two datasets
        dataset = xr.concat([data_fp, data_sp], dim='time')


        # Data also contains variable expver = 5 but this has nan values except for last two months (2021) 
        dataset = dataset.drop('expver')
        dataset = dataset.sel(expver=0,drop=True)

        
        # Change coordinate and variable names
        dataset = dataset.rename({'longitude': 'lon','latitude': 'lat', 'u10' : 'u', 'v10' : 'v'})


        # Sort longitudes increasing
        dataset = dataset.sortby('lon')


        
    elif data_type == '20cr':
        
        
        # Define the path to code
        path_u = '/Users/iriskeizer/Documents/Wind effect/Data/observations/Wind/wind_20cr_u.nc' # Path to zonal wind
        path_v = '/Users/iriskeizer/Documents/Wind effect/Data/observations/Wind/wind_20cr_v.nc' # Path to meridional wind


        # Open data file
        u = xr.open_dataset(path_u) 
        v = xr.open_dataset(path_v) 


        # Add the two datasets
        dataset = u.assign(vwnd = v.vwnd)


        # Shift longitudes from 0 - 360 to -180 to 180
        longitudes_list = np.concatenate([np.arange(0, 180), np.arange(-180, 0)])
        dataset = dataset.assign_coords(lon = longitudes_list)


        # Sort longitudes increasing
        dataset = dataset.sortby('lon')


        # Change coordinate and variable names
        dataset = dataset.rename({'uwnd' : 'u', 'vwnd' : 'v'})
    
    
        #Drop 'time_bnds' variables
        dataset = dataset.drop('time_bnds')
        
    
    else: print('Given data_type not correct, choose era5 or 20cr')
    
        
    
    
    # Select smaller area of data 
    dataset = dataset.where((dataset.lat > 40) & (dataset.lat < 90) & (dataset.lon > -40) & (dataset.lon < 30), drop=True)


    # Obtain stress for monthly averaged data 
    dataset = dataset.assign(u2 = dataset.u*(np.sqrt(dataset.u**2+dataset.v**2)))
    dataset = dataset.assign(v2 = dataset.v*(np.sqrt(dataset.u**2+dataset.v**2)))  


    # Calculate annual averages 
    dataset_annual = dataset.groupby('time.year').mean('time')


    # Remove 2021 
    dataset_annual = dataset_annual.where(dataset_annual.year <2021, drop=True)          

    
    # Save annual data as netcdf4       
    save_nc_data(dataset_annual, 'observations', 'Wind', f'wind_annual_{data_type}')
                
                
        
    return dataset_annual
        
    
    
    
    
    
    
def prep_pres_data_obs(data_type = 'era5'):
    """
    Function to prepare the observational pressure data for the analysis
    
    For data_type choose ['era5', '20cr']
    
    
    """
    
    
    
    
    if data_type == 'era5':
        
        
        # Define the path to code
        path_fp = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Pressure/pres_era5_fp.nc' # Path to surface pressure 1950-1978
        path_sp = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Pressure/pres_era5_sp.nc' # Path to surface pressure 1979-present
        

        # Open data file
        dataset_fp = xr.open_dataset(path_fp)
        dataset_sp = xr.open_dataset(path_sp) 
        
        
        # Add the two datasets
        dataset = xr.concat([dataset_fp, dataset_sp], dim='time')


        # Change coordinate and variable names
        dataset = dataset.rename({'longitude': 'lon','latitude': 'lat','msl': 'pressure'})


        # Sort latitudes increasing
        dataset = dataset.sortby('lat')


        
    elif data_type == '20cr':
        
        
        # Define the path to code
        path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Pressure/prmsl.mon.mean.nc' # Path to zonal wind

        
        # Open data file
        dataset = xr.open_dataset(path) 

        
        # Shift longitudes to -180-180 
        dataset.coords['lon'] = (dataset.coords['lon'] + 180) % 360 - 180
        dataset = dataset.sortby(dataset.lon)


        # Change coordinate and variable names
        dataset = dataset.rename({'prmsl': 'pressure'})
    
    
        #Drop 'time_bnds' variables
        dataset = dataset.drop('time_bnds')
        
        
        # Select smaller area of data 
        dataset = dataset.where((dataset.lat >= 0) & (dataset.lat <= 90) & (dataset.lon >= -90) & (dataset.lon <= 90), drop=True)


        
            
    else: print('Given data_type not correct, choose era5 or 20cr')

        
    # Calculate annual averages 
    dataset_annual = dataset.groupby('time.year').mean('time')
   
    # Save annual data as netcdf4           
    save_nc_data(dataset_annual, 'observations', 'Pressure', f'pres_annual_{data_type}')    
    
    
    
    
    return dataset_annual
    
    
    
    
    
    
    

    
    

    
    
    
    
    
"""
PREPARE CMIP6 DATA
--------------------------

Functions to import the CMIP6 data and put all models in a dataset.
Also other necessary changes are made to get the data ready for the analysis



"""




def prep_slh_data_cmip6(data_type = 'historical'):
    """
    Function to prepare the cmip6 sea level height data for the analysis
    
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    """
    
    # Define path to cmip6 data
    path = f'/Volumes/Iris 300 GB/CMIP6/cmip6_zos_{data_type}'
    
    
    
    
    if data_type == 'piControl':
        
        
        
        # Create empty list to save files
        piControl = []

        # Loop over all files in directory
        for file in gb.glob(f'{path}/*'):
            
            
            # Open data file
            data = xr.open_dataset(file)
    
    
            # Select area
            data = data.where((data.lat > 45) & (data.lat < 60) & (data.lon > 0) & (data.lon < 10), drop=True)
    
    
            # Shift time for each model to start at 0 since the values are arbitrary
            time_lst = data.time.values - data.time.values[0]
            data = data.assign_coords(time = time_lst)
    
            
            # Add file to list
            piControl.append(data)
            
        # Merge all files to one dataset 
        dataset_annual = xr.merge(piControl, combine_attrs='override')
        
    
    
    
    else: 
        
        
        # Open data files as dataset
        dataset_annual = xr.open_mfdataset(f'{path}/cmip6_zos_{data_type}_*.nc')
    
    
        # Select area
        dataset_annual = dataset_annual.where((dataset_annual.lat > 45) & (dataset_annual.lat < 60) & (dataset_annual.lon > 0) & (dataset_annual.lon < 10), drop=True)
        
        
        # Change time to integer
        dataset_annual.coords['time'] = dataset_annual.coords['time'].astype(int)
    
    
    # Change coordinate and variable names
    dataset_annual = dataset_annual.rename({"CorrectedReggrided_zos":"zos"})
    
    
    # Obtain coordinates of the tide gauge stations
    coord_df = station_coords()
    coord_df['lon'][1] = 3.5 # Make sure HvH doesn't get nan values

    # Create list of wind dataarrays
    lst = [] 

    # Loop over the tide gauge stations
    for index, row in coord_df.iterrows():
        lst.append(dataset_annual.zos.sel(lon = row.lon, lat = row.lat, method = 'nearest'))
        lst[-1] = lst[-1].drop(['lat', 'lon'])
    
    
    
    # Create a dataarray
    dataset_annual = xr.concat(lst, stations[:-1]).rename({'concat_dim':'station'})
    
    
    # Calculate average station
    average = dataset_annual.mean('station')
    average = average.assign_coords({'station':'Average'})
    
    
    # Concat to original dataarray
    dataset_annual = xr.concat([dataset_annual, average], dim='station')
    
    
    # Save annual data as netcdf4           
    save_nc_data(dataset_annual, 'cmip6', 'SLH', f'slh_annual_{data_type}')    
    
    
    
    
    return dataset_annual






def prep_wind_picontrol(data_type = 'zonal'):
    """
    Function to do some preparation on picontrol data before storing all as one dataset
    
    For data_type choose ['zonal', 'meridional']
    
    """
    if data_type == 'zonal':
        var = 'uas'
    elif data_type == 'meridional':
        var= 'vas'
        
    
    # Define path to cmip6 data
    path = '/Volumes/Iris 300 GB/CMIP6'
    
    
    # Loop over all files in directory
    for file in gb.glob(f'{path}/cmip6_{var}_piControl/*'):
        

        # Open data file
        data = xr.open_dataset(file) # unit: m/s


        # Select area
        data = data.where((data.lat > 40) & (data.lat < 70) & (data.lon > -30) & (data.lon < 30), drop=True)


        # Shift time for each model to start at 0 since the values are arbitrary
        time_lst = data.time.values - data.time.values[0]
        data = data.assign_coords(time = time_lst)


        # Change coordinate and variable names
        data = data.rename({f"CorrectedReggrided_{var}" : f"{var}"})
        
        
        # Obtain wind stress (wind is squared, sign is retained)
        if data_type == 'zonal':
            data = data.assign(u2 = data.uas**2*np.sign(data.uas))
        elif data_type == 'meridional':
            data = data.assign(v2 = data.vas**2*np.sign(data.vas))

        data_vel = data
        # Remove velocity variables
        data = data.drop(var)


        # Save data as netcdf4     
        name = os.path.basename(file)
        data.to_netcdf(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Wind/piControl/{data_type}/{name}", mode='w')

    
    
    
def prep_wind_data_cmip6(data_type = 'historical'):
    """
    Function to prepare the cmip6 wind data for the analysis
    
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    """
    
    
    # Define path to cmip6 data
    path = '/Volumes/Iris 300 GB/CMIP6'
    
    
    
    
    if data_type == 'piControl':
        
        
        # ZONAL
        prep_wind_picontrol()
        
        
        path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Wind/piControl/zonal'   


        # Open data files as dataset
        dataset_u = xr.open_mfdataset(f'{path}/cmip6_uas_piControl_*.nc')


        
        
        # MERIDIONAL
        prep_wind_picontrol(data_type='meridional')
        
        
        path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Wind/piControl/meridional'   


        # Open data files as dataset
        dataset_v = xr.open_mfdataset(f'{path}/cmip6_vas_piControl_*.nc')
    
        
        # Add the two datasets
        dataset_annual = dataset_u.assign(v2 = dataset_v.v2)
    
    
    else: 
        
        
        # Open data files as dataset
        dataset_u = xr.open_mfdataset(f'{path}/cmip6_uas_{data_type}/cmip6_uas_{data_type}_*.nc')
        dataset_v = xr.open_mfdataset(f'{path}/cmip6_vas_{data_type}/cmip6_vas_{data_type}_*.nc')
        
        
        # Select area
        dataset_u = dataset_u.where((dataset_u.lat > 40) & (dataset_u.lat < 90) & (dataset_u.lon > -40) & (dataset_u.lon < 30), drop=True)
        dataset_v = dataset_v.where((dataset_v.lat > 40) & (dataset_v.lat < 90) & (dataset_v.lon > -40) & (dataset_v.lon < 30), drop=True)
    
    
        # Change coordinate and variable names
        dataset_u = dataset_u.rename({"CorrectedReggrided_uas":"uas"})
        dataset_v = dataset_v.rename({"CorrectedReggrided_vas":"vas"})
    

        # Add the two datasets
        dataset_annual = dataset_u.assign(vas = dataset_v.vas)


        # Obtain wind stress (wind is squared, sign is retained)
        dataset_annual = dataset_annual.assign(u2 = dataset_annual.uas**2*np.sign(dataset_annual.uas))
        dataset_annual = dataset_annual.assign(v2 = dataset_annual.vas**2*np.sign(dataset_annual.vas))  


        # Make a dataset without velocity variables
        dataset_annual = dataset_annual.drop(['uas', 'vas'])
        
        
        # Change time to integer
        dataset_annual.coords['time'] = dataset_annual.coords['time'].astype(int)
    
    
    # Save annual data as netcdf4           
    save_nc_data(dataset_annual, 'cmip6', 'Wind', f'wind_annual_{data_type}')    
    
    
    
    
    return dataset_annual








    
def prep_pres_data_cmip6(data_type = 'historical'):
    """
    Function to prepare the cmip6 pressure data for the analysis
    
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    """
    
    
    # Define path to cmip6 data
    path = f'/Volumes/Iris 300 GB/CMIP6/cmip6_ps_{data_type}/'
    
    
    
    
    if data_type == 'piControl':
        
        
        
        # Create empty list to save files
        piControl = []

        # Loop over all files in directory
        for file in gb.glob(f'{path}/*'):
            
            
            # Open data file
            data = xr.open_dataset(file)
    
    
            # Select area
            data = data.where((data.lat > 0) & (data.lat < 90) & (data.lon > -90) & (data.lon < 90), drop=True)
            
            
            # Shift time for each model to start at 0 since the values are arbitrary
            time_lst = data.time.values - data.time.values[0]
            data = data.assign_coords(time = time_lst)
            
            
            # Add file to list
            piControl.append(data)
            
        
        # Merge all files to one dataset 
        dataset_annual = xr.merge(piControl, combine_attrs='override')
        
    
    
    
    else: 
        
        
        # Open data files as dataset
        dataset_annual = xr.open_mfdataset(f'{path}cmip6_ps_{data_type}_*.nc')
    
    
        # Select area
        dataset_annual = dataset_annual.where((dataset_annual.lat > 0) & (dataset_annual.lat < 90) & (dataset_annual.lon > -90) & (dataset_annual.lon < 90), drop=True)
        
        
        # Change time to integer
        dataset_annual.coords['time'] = dataset_annual.coords['time'].astype(int)
    
    
    # Change coordinate and variable names
    dataset_annual = dataset_annual.rename({"CorrectedReggrided_ps":"ps"})
    
    
    # Save annual data as netcdf4           
    save_nc_data(dataset_annual, 'cmip6', 'Pressure', f'pressure_annual_{data_type}')    
    
    
    
    
    return dataset_annual


