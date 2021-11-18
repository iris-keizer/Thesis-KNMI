"""
File containing the Python functions to import the correct data used for the regression between wind stress and sea level height along the Dutch coast. 
Depending on the type of data, used wind model and whether observational or cmip6 data is used, the functions 

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
nearby_wind_regression_obs_era5.ipynb 
nearby_wind_regression_cmip6_historical.ipynb 

"""



# Import necessary packages
import xarray as xr 
import pandas as pd
import numpy as np
import regionmask
import statsmodels.api as sm
import statsmodels as sm
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
    
def timmerman_region_names(): 
    """
    Function to obtain timmerman region names as list
    
    """
    return ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East', 'Average']




def station_coords(): 
    """
    Function to obtain the coordinates of the tide gauge stations as a dataframe
    
    """
    
    
    # Necessary declarations to obtain tide gauge station coordinates
    path_locations = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/rlr_annual/filelist.txt'
    loc_num = [20, 22, 23, 24, 25, 32]
    col_names = ['id', 'lat', 'lon', 'station', 'coastline_code', 'station_code', 'quality']
    
    # Create dataframe
    df = pd.read_csv(path_locations, sep=';', header=None, names=col_names)
    df = df.set_index('id')
    df = df.loc[loc_num, :]
    df['station'] = stations[:-1]
    df = df.set_index('station')
    df = df.drop(['coastline_code', 'station_code', 'quality'], axis=1)
    
    return df


def cmip6_np_coords(): 
    """
    Function to obtain a dataframe containing the coordinates of cmip6 nearest points to tide gauge models
    """
    lat_lst = [51.5, 51.5, 52.5, 53.5, 53.5, 52.5]
    lon_lst = [3.5, 4.5, 4.5, 6.5, 5.5, 4.5]
    df = pd.DataFrame({'station' : stations[:-1], 'lat' : lat_lst, 'lon' : lon_lst})
    df = df.set_index('station')
    
    return df
    

def save_nc_data(data, folder, variable, name): 
    """
    Function to save data as NETCDF4 file
    
    For folder choose ['observations', 'cmip6'], for variable choose ['Wind', 'SLH', 'Pressure', 'SST']
    
    """
    data.to_netcdf(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.nc", mode='w')
    
    
def save_csv_data(data, folder, variable, name): 
    """
    Function to save data as .csv file
    
    For folder choose ['observations', 'cmip6'], for variable choose ['Wind', 'SLH', 'Pressure', 'SST']
    
    """
    data.to_csv(f"/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/{folder}/{variable}/{name}.csv")


def timmerman_regions():
    """
    Function to obtain the timmerman regions 
    
    """
    
    # Declare regions using coordinates
    # As first coordinates take most South-West point and than go anti-clockwise
    Channel = np.array([[-5.1, 48.6], [1.5, 50.1], [1.5, 50.9], [-5.1, 49.9]])
    South = np.array([[0.5, 50.8], [3.2, 51.3], [5.3, 53.1], [1.7, 52.3]])
    Mid_West = np.array([[1.7, 52.3], [5.3, 53.1], [3.7, 55.7], [-1.3, 55.1], [0.5, 53.1], [1.8, 52.7]])
    Mid_East = np.array([[5.3, 53.1], [8.9, 53.9], [7.8, 57.0], [3.7, 55.7]])
    North_West = np.array([[-1.3, 55.1], [3.7, 55.7], [1.1, 59.3], [-3.0, 58.7], [-1.7, 57.5], [-1.5, 55.5]])
    North_East = np.array([[3.7, 55.7], [7.8, 57.0], [7.4, 58.0], [6.1, 58.6], [4.9, 60.3], [1.1, 59.3]])
    
    
    # Declare names, abbreviations and numbers
    region_names = ["Channel", "South", "Mid-West", "Mid-East", "North-West", "North-East"]
    region_abbrevs = ["C", "S", "MW", "ME", "NW", "NE"]
    region_numbers = [1,2,3,4,5,6]
    
    
    # Create regions 
    Timmerman_regions = regionmask.Regions([Channel, South, Mid_West, Mid_East, North_West, North_East], 
                                           numbers = region_numbers, 
                                           names=region_names, abbrevs=region_abbrevs, name="Timmerman")
    
    return Timmerman_regions
    

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
    
    
# Get atmospheric proxies
def get_proxies(pres):
    
    # Select proxy region
    dang_coords_np = np.array([[-8, 55], [23, 55], [23, 70], [-8, 70]])
    dang_coords_pp = np.array([[-15, 28], [7, 28], [7, 43], [-15, 43]])
        
    
    dang_regions = regionmask.Regions([dang_coords_np, dang_coords_pp], numbers = [1,2], 
                                      names=["Negative proxy", "Positive proxy"], abbrevs=["NP", "PP"], 
                                      name="Dangendorf regions")

    
    # Create wind data array per region
    mask = dang_regions.mask(pres)

    regional_data = []  # List containing the dataset per region
    

    for i in range(1,3):
        regional_data.append(pres.where(mask == i, drop=True)) 

    # Obtain spatial averages of proxy region
    slp_mean = [] # List containing the average per region

    for idx, val  in enumerate(regional_data):
        slp_mean.append(val.mean(dim=['lon','lat']))

    
        
    return slp_mean



def new_df_obs_wind_per_var(data, variable  = 'u$^2$', model = 'NearestPoint'):
    """
    Function to create a new dataframe of observed wind data containing only zonal or meridional wind stress data
    
    For variable choose ['u$^2$', 'v$^2$'ß]
    
    For model choose ['NearestPoint', 'Timmerman']
    
    """
    
    if model == 'NearestPoint':
        return pd.DataFrame({stations[0]: data[stations[0],  variable],
                          stations[1]: data[stations[1],  variable],
                          stations[2]: data[stations[2],  variable],
                          stations[3]: data[stations[3],  variable],
                          stations[4]: data[stations[4],  variable],
                          stations[5]: data[stations[5],  variable],}, index = data.index)
    
    
    elif model == 'Timmerman':
        
        return pd.DataFrame({regions[0]: data[regions[0],  variable],
                          regions[1]: data[regions[1],  variable],
                          regions[2]: data[regions[2],  variable],
                          regions[3]: data[regions[3],  variable],
                          regions[4]: data[regions[4],  variable],
                          regions[5]: data[regions[5],  variable],}, index = data.index)
    
    
    else: print('For model choose [NearestPoint, Timmerman]' )

        
def get_frac(window, data, dtype='DataFrame'):
    if dtype == 'DataFrame':
        frac = window / (data.index[-1]-data.index[0])
    elif dtype == 'DataSet':
        frac = window / (data.time.values[-1] - data.time.values[0])
    else: print('Datatype unknown')
    
    return frac






    
# Declare global variables
stations = station_names()
regions = timmerman_region_names()
lowess = sm.nonparametric.smoothers_lowess.lowess


"""
IMPORTING DATA
--------------

Importing the annual data to use it for the (regression) analysis
"""





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
    
    
    # Save tide gauge stations coordinates
    save_csv_data(coord_df, 'observations', 'Coordinates', 'tgstations')
    
    
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








def import_obs_wind_data(model = 'Nearest Point', data_type = 'era5'):
    """
    Function that imports the observed wind data based on the preferred wind data model that is used for regression
    
    For model choose ['Nearest Point', 'Dangendorf', 'Timmerman']
    For data_type choose ['era5', '20cr']
    
    """
    
    
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/'
    
    
    
    
    if model == 'NearestPoint' or model ==  'Timmerman':
        
        dataset_annual = xr.open_dataset(path + f'Wind/wind_annual_{data_type}.nc')  
         
        
        
        
    elif model == 'Dangendorf':
        
        dataset_annual = xr.open_dataset(path + f'Pressure/pres_annual_{data_type}.nc') 
    
    
    
    
    if model == 'NearestPoint':
        
        # Obtain coordinates of the tide gauge stations
        coord_df = station_coords()

        
        # Create list of wind dataframes 
        lst = [] # List containing the created dataframes
        coord_df_np = pd.DataFrame({'station':stations[:-1], 'lat':'', 'lon':''})
        coord_df_np = coord_df_np.set_index('station')
        
        # Loop over the tide gauge stations
        for index, row in coord_df.iterrows():
            lst.append(pd.DataFrame(data={'time': dataset_annual.u2.year, 
                                         'u$^2$' : dataset_annual.u2.sel(lon = row.lon, 
                                                                      lat = row.lat, method = 'nearest').to_series().dropna(), 
                                         'v$^2$' : dataset_annual.v2.sel(lon = row.lon, 
                                                                      lat = row.lat, method = 'nearest').to_series().dropna()}))
            coord_df_np['lat'][index] = dataset_annual.sel(lon = row.lon, lat = row.lat, method = 'nearest').lat.values
            coord_df_np['lon'][index] = dataset_annual.sel(lon = row.lon, lat = row.lat, method = 'nearest').lon.values
            
            lst[-1] = lst[-1].set_index('time')

            annual_df = pd.concat(lst, axis=1, keys = stations[:-1])

        # Save NearestPoint data coordinates
        save_csv_data(coord_df_np, 'observations', 'Coordinates', f'np_{data_type}')
        
        data_u = new_df_obs_wind_per_var(annual_df, variable  = 'u$^2$', model = 'NearestPoint')
        data_v = new_df_obs_wind_per_var(annual_df, variable  = 'v$^2$', model = 'NearestPoint')

        data_u['Average'] = data_u.mean(axis=1)
        data_v['Average'] = data_v.mean(axis=1)

        annual_df = pd.concat([data_u, data_v], keys=['u$^2$','v$^2$'],  axis=1)
        annual_df = annual_df.swaplevel(0,1, axis=1)
            
    elif model == 'Timmerman':
        
         
        # Create wind data array per region
        regions = timmerman_regions()
        mask = regions.mask(dataset_annual)
        regional_data = []  # List containing the dataset per region
        
        for i in range(1,7):
            regional_data.append(dataset_annual.where(mask == i))

        # Create list of wind dataframes containing regional averages of wind stress data
        lst = [] # List containing the created dataframes

        for ds in regional_data:
            ds_avg = ds.mean('lon').mean('lat')
            lst.append(pd.DataFrame(data={'time': ds_avg.year, 'u$^2$' : ds_avg.u2, 'v$^2$' : ds_avg.v2}))
            lst[-1] = lst[-1].set_index('time')

        annual_df = pd.concat(lst, axis=1, keys = regions.names)

    
    
    
    elif model == 'Dangendorf':
        pres_data = get_proxies(dataset_annual) # List containing two xarray.datasets
        
        # Create dataframe
        annual_df = pd.DataFrame(data={'time': pres_data[1].year, 
                                       'Negative corr region' : pres_data[0].pressure.values, 
                                       'Positive corr region' : pres_data[1].pressure.values})
        annual_df =  annual_df.set_index('time')
    
            
    return annual_df






"""
MODEL DATA
----------

"""


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




def cmip6_get_nearest_point(data):
    """
    Function that makes a dataset with nearest wind to each tide gauge station and adds the average
    
    data should be ['dataset_annual.u2', 'dataset_annual.v2']
    """
    
    
    # Obtain coordinates of the tide gauge stations
    coord_df = station_coords()

    # Create list of wind dataarrays
    lst = [] 
    
    # Loop over the tide gauge stations
    for index, row in coord_df.iterrows():
        lst.append(data.sel(lon = row.lon, lat = row.lat, method = 'nearest'))
        lst[-1].attrs['units'] = 'm$^2$/s$^2$'
        lst[-1] = lst[-1].drop(['lat', 'lon'])


    # Create a dataarray
    data = xr.concat(lst, stations[:-1]).rename({'concat_dim':'station'})


    # Calculate average station
    average = data.mean('station')
    average = average.assign_coords({'station':'Average'})


    # Concat to original dataarray
    data = xr.concat([data, average], dim='station')
    
    
    return data




def import_cmip6_wind_data(model = 'NearestPoint', data_type = 'historical'):
    """
    Function that imports the observed wind data based on the preferred wind data model that is used for regression
    
    For model choose ['NearestPoint', 'Dangendorf', 'Timmerman']
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    """
    
    
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/'
    
    
    
    
    if model == 'NearestPoint' or model ==  'Timmerman':
        
        dataset_annual = xr.open_dataset(path + f'Wind/wind_annual_{data_type}.nc') 
         
        
        
        
    elif model == 'Dangendorf':
        
        dataset_annual = xr.open_dataset(path + f'Pressure/pressure_annual_{data_type}.nc') 
    
    
    
    
    if model == 'NearestPoint':
        
        
        # Obtain nearest wind points for zonal and meridional wind stress
        dataset_u = cmip6_get_nearest_point(dataset_annual.u2)
        dataset_v = cmip6_get_nearest_point(dataset_annual.v2)
            
        
        # Add both dataarrays to a dataset
        dataset_annual = xr.Dataset({'u2': dataset_u, 'v2':dataset_v})
            
    
    
            
    elif model == 'Timmerman':
        
         
        # Create wind data array per region
        regions = timmerman_regions()
        mask = regions.mask(dataset_annual)


        regional_data = []  # List containing the dataset per region

        for i in range(1,7):
            regional_data.append(dataset_annual.where(mask == i, drop = True).mean('lon').mean('lat'))
        
        
        
        # Concatenate all datasets
        dataset_annual = xr.concat(regional_data, regions.names).rename({'concat_dim':'tim_region'})
          


        
        
    elif model == 'Dangendorf':
        
        # Create pressure data averaged per positive or negative correlation region
        pres_data = get_proxies(dataset_annual) # List containing two xarray.datasets
        
    
        # Add both dataarrays to a dataset
        dataset_annual = xr.Dataset({'Negative corr region': pres_data[0].ps, 'Positive corr region':pres_data[1].ps})
    
            
    return dataset_annual






