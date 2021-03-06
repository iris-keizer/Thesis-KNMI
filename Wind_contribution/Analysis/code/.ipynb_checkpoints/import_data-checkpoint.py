"""
File containing the Python functions to import the correct data used for the regression between wind/pressure and sea level height along the Dutch coast. 



Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
timmerman_regression_obs.ipynb
timmerman_regression_cmip6.ipynb
nearestpoint_regression_obs.ipynb
nearestpoint_regression_cmip6.ipynb
dangendorf_regression_obs.ipynb
dangendorf_regression_cmip6.ipynb

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
    Function to obtain tide gauge station names as a list
    
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
    path_locations = '/Users/iriskeizer/Documents/Wind effect/Data/rlr_annual/filelist.txt'
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
    lat_lst = [51.5, 52.5, 52.5, 53.5, 53.5, 52.5]
    lon_lst = [3.5, 4.5, 4.5, 6.5, 5.5, 4.5]
    df = pd.DataFrame({'station' : stations[:-1], 'lat' : lat_lst, 'lon' : lon_lst})
    df = df.set_index('station')
    
    return df


    
    
    
    
    
def obs_np_coords(data_type = 'era5'): 
    """
    Function to obtain a dataframe containing the coordinates of observed nearest points above sea to tide gauge models
    """
    if data_type == 'era5':
        lat_lst = [51.5, 52.0, 53.0, 53.5, 53.25, 52.5]
        lon_lst = [3.5, 4.0, 4.75, 7.0, 5.25, 4.5]
    elif data_type == '20cr':
        lat_lst = [52, 52, 53, 54, 53, 52]
        lon_lst = [4, 4, 5, 7, 5, 4]
        
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
    data.to_csv(f"/Users/iriskeizer/Documents/Wind effect/Data/{folder}/{variable}/{name}.csv")


    
    
    
    
    
def timmerman_regions():
    """
    Function to obtain the timmerman regions 
    
    """
    
    # Declare regions using coordinates
    # As first coordinates take most South-West point and than go anti-clockwise
    Channel = np.array([[-5.1, 48.6], [1.5, 50.1], [1.5, 50.9], [-5.1, 50.1]])
    South = np.array([[0.5, 50.8], [3.2, 51.3], [5.3, 53.1], [1.7, 52.3]])
    Mid_West = np.array([[1.7, 52.3], [5.3, 53.1], [3.7, 55.7], [-1.3, 55.1], [0.5, 53.1], [1.8, 52.7]])
    Mid_East = np.array([[5.3, 53.1], [8.9, 53.9], [7.8, 57.0], [3.7, 55.7]])
    North_West = np.array([[-1.3, 55.1], [3.7, 55.7], [1.1, 59.3], [-3.0, 58.7], [-1.7, 57.5], [-1.5, 55.5]])
    North_East = np.array([[3.7, 55.7], [7.8, 57.0], [7.6, 58.1], [6.1, 58.6], [4.9, 60.3], [1.1, 59.3]])
    
    
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
    
    return nodcyc
    
    
    
    
    
    
    
# Get atmospheric proxies
def get_proxies(pres):
    
    # Select proxy region
    dang_coords_np = np.array([[-0.1, 54.9], [40.1, 54.9], [40.1, 75.1], [-0.1, 75.1]])
    dang_coords_pp = np.array([[-20.1, 24.9], [20.1, 24.9], [20.1, 45.1], [-20.1, 45.1]])
        
    
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
    
    For variable choose ['u$^2$', 'v$^2$'??]
    
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


    
    
    
    
    

def detrend_dim(da, dim, deg=1):
    '''
    Function that detrends the data from a dataarray
    deg = 1 for a linear fit
    '''
    
    p = da.polyfit(dim=dim, deg=deg)
    coord = da.year - da.year.values[0]
    trend = coord*p.polyfit_coefficients.sel(degree=1)
    
    return da - trend



    
    
    
    
    
    
# Declare global variables
stations = station_names()
regions = timmerman_region_names()
lowess = sm.nonparametric.smoothers_lowess.lowess


# Only use models occuring in zos, uas, vas and ps datasets for all scenarios
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0',
       'CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1',
       'CanESM5', 'CanESM5-CanOE', 'EC-Earth3', 'EC-Earth3-Veg',
       'EC-Earth3-Veg-LR', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL',
       'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
       'MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
       'NESM3', 'UKESM1-0-LL']

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


"""
IMPORTING DATA
--------------

Importing the annual data to use it for the (regression) analysis
"""





"""
OBSERVATIONS
------------

"""

    
    
    
    
    
def import_obs_slh_data(data_type = 'era5'):
    """
    Function that imports the tide gauge sea level height data as a pandas.dataframe
    
    For data_type choose ['era5', '20cr']
    
    """
    
    
    # Define paths to data
    path = '/Users/iriskeizer/Documents/Wind effect/Data/observations/SLH/tg_data.csv'
    
    
    # Open data file
    tg_data_df = pd.read_csv(path)
    
    # Set time as index of dataframe
    tg_data_df = tg_data_df.set_index('time')
    
    # Obtain coordinates of the tide gauge stations
    coord_df = station_coords()
    
    # Save tide gauge stations coordinates
    save_csv_data(coord_df, 'observations', 'Coordinates', 'tgstations')
    
    
    # Import pressure data
    path_pres = '/Users/iriskeizer/Documents/Wind effect/Data/observations/Pressure/'
    pres = xr.open_dataset(path_pres+f'pres_annual_{data_type}.nc')
    
    # Make sure tg data has same temporal length as pressure
    tg_data_df = tg_data_df[tg_data_df.index.isin(pres.year.values)]
    pres = pres.where((pres.year>=tg_data_df.index[0]) & (pres.year<=tg_data_df.index[-1]), drop=True)
    
    # Create new datafame:
    data_df = pd.DataFrame(data={'time' : tg_data_df.index})
    data_df = data_df.set_index('time')
    
    # Declare variables
    rho = 1030 # Density of ocean water
    g = 9.81 # Acceleration of gravity on Earth's surface
    
    for station in stations[:-1]:

        # Remove nodal cycle
        nodal_correction = nodal_tides_potential(coord_df['lat'].loc[station], tg_data_df.index.values)

        dt_wh_nodal = tg_data_df[station].values - nodal_correction
        
        
        # Remove IBE
        ps = pres.sel(lat = coord_df['lat'][station], lon = coord_df['lon'][station], method='Nearest').pressure.values
        ps = ps - ps.mean() # Calculate anomaly
        ibe = -ps/(rho*g)*100 # Inverse barometer effect in cm
        ibe = ibe - ibe.mean() # Calculate anomaly
        
        dt_wh_nodal_ibe = dt_wh_nodal - ibe
        
        data_df[station] = dt_wh_nodal_ibe
        
    data_df['Average'] = data_df.mean(axis=1)
    
    return data_df




    
    
    
    
    



def import_obs_wind_data(model = 'Nearest Point', data_type = 'era5'):
    """
    Function that imports the observed wind data based on the preferred wind data model that is used for regression
    
    For model choose ['Nearest Point', 'Dangendorf', 'Timmerman']
    For data_type choose ['era5', '20cr']
    
    """
    
    
    path = '/Users/iriskeizer/Documents/Wind effect/Data/observations/'
    
    
    
    
    if model == 'NearestPoint' or model ==  'Timmerman':
        
        dataset_annual = xr.open_dataset(path + f'Wind/wind_annual_{data_type}.nc')  
         
        
        
        
    elif model == 'Dangendorf':
        
        dataset_annual = xr.open_dataset(path + f'Pressure/pres_annual_{data_type}.nc') 
    
    
    
    
    if model == 'NearestPoint':
        
        # Obtain coordinates
        coord_df = obs_np_coords(data_type)

        
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

        annual_df = pd.concat([data_u, data_v], keys=['$u \sqrt{u^2+v^2}$','$v \sqrt{u^2+v^2}$'],  axis=1)
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
            lst.append(pd.DataFrame(data={'time': ds_avg.year, '$u \sqrt{u^2+v^2}$' : ds_avg.u2, '$v \sqrt{u^2+v^2}$' : ds_avg.v2}))
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








    
    
def import_pres_tg_corr_data(data_type = 'era5', year_start = 1950, year_final = 2015):


    # Prepare pressure data

    # Import annual pressure data
    path = '/Users/iriskeizer/Documents/Wind effect/Data/observations/'
    pres = xr.open_dataset(path + f'Pressure/pres_annual_{data_type}.nc') 

    # Linearly detrend
    pres_corr = detrend_dim(pres.pressure, dim='year', deg=1)
    pres_corr = pres_corr.drop('degree')
    
    # Rename year to time
    pres_corr = pres_corr.rename({'year':'time'})
    
    # Select time period
    pres_corr = pres_corr.where(pres_corr.time > year_start - 1, drop = True)
    pres_corr = pres_corr.where(pres_corr.time < year_final + 1, drop = True)
    
    # Prepare tide gauge data

    # Import tide gauge data
    tg = import_obs_slh_data()

    # Linearly detrend
    data = []

    for station in tg.columns:
        data.append(detrend(tg[station]))

    # Create dataarray of tide gauge data
    tg_corr = xr.DataArray(data, dims = ['station', 'time'], coords = dict(time = tg.index, station = tg.columns))

    # Select time period
    tg_corr = tg_corr.where(pres_corr.time > year_start - 1, drop = True)
    tg_corr = tg_corr.where(pres_corr.time < year_final + 1, drop = True)

    return pres_corr, tg_corr

    
    
    
    
    
    
    
    
    
    
    
    


"""
MODEL DATA
----------

"""

    
    
    
    
    

def import_cmip6_slh_data(data_type = 'historical'):
    """
    Function that imports cmip6 sea level data
    
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245' 'ssp370', 'ssp585']
    
    """
    
    
    # Define paths to data\
    path = f'/Users/iriskeizer/Documents/Wind effect/Data/cmip6/SLH/slh_annual_historical.nc'

    # Open data file
    zos = xr.open_dataset(path) 


    # Only use models occuring in both datasets
    zos = zos.where(zos.model.isin(models), drop=True)
    
    
    # For the cmip6 data the nodal and ibe shouldn't be removed

    
    return zos


    
    
    
    
    
    
    
    
    
    


def cmip6_get_nearest_point(data):
    """
    Function that makes a dataset with nearest wind to each tide gauge station and adds the average
    
    data should be ['dataset_annual.u2', 'dataset_annual.v2']
    """
    
    
    # Obtain coordinates of the tide gauge stations
    coord_df = station_coords()

    
    # Create dataframe with cmip6 coords
    coord_cmip6_df = pd.DataFrame({'station':coord_df.index.values, 'lat':'', 'lon':''})
    coord_cmip6_df = coord_cmip6_df.set_index('station' )
    
    
    # Create list of wind dataarrays
    lst = [] 
    
    # Loop over the tide gauge stations
    for index, row in coord_df.iterrows():
        lst.append(data.sel(lon = row.lon, lat = row.lat, method = 'nearest'))
        lst[-1].attrs['units'] = 'm$^2$/s$^2$'
        lst[-1] = lst[-1].drop(['lat', 'lon'])
        
        lat = data.sel(lat = row.lat, lon = row.lon, method='Nearest').lat.values
        lon = data.sel(lat = row.lat, lon = row.lon, method='Nearest').lon.values
        
        coord_cmip6_df['lat'][index] = lat
        coord_cmip6_df['lon'][index] = lon

    # Create a dataarray
    data = xr.concat(lst, stations[:-1]).rename({'concat_dim':'station'})


    # Calculate average station
    average = data.mean('station')
    average = average.assign_coords({'station':'Average'})


    # Concat to original dataarray
    data = xr.concat([data, average], dim='station')
    
    
    # Save cmip6 coordinate dataframe
    save_csv_data(coord_cmip6_df, 'cmip6', 'Coordinates', 'hist')
    
    return data




def import_cmip6_wind_data(model = 'NearestPoint', data_type = 'historical'):
    """
    Function that imports the observed wind data based on the preferred wind data model that is used for regression
    
    For model choose ['NearestPoint', 'Dangendorf', 'Timmerman']
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    """
    
    
    path = '/Users/iriskeizer/Documents/Wind effect/Data/cmip6/'
    
    
    
    
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
    
    
    # Only use models occuring in both datasets
    dataset_annual = dataset_annual.where(dataset_annual.model.isin(models), drop=True)
            
    return dataset_annual






