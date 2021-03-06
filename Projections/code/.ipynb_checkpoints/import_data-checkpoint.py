"""
File containing the Python functions to import the correct data used for the regression between cmip6 model data wind stress and sea level height along the Dutch coast as a preparation for the projections and for the projections themselves


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
pre_projection_regression.ipynb

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
    df = pd.read_csv(path_locations, sep=';', header=None, names=col_names, index_col='id')
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










    
# Declare global variables
stations = station_names()
regions = timmerman_region_names()
lowess = sm.nonparametric.smoothers_lowess.lowess


# Only use models occuring in both datasets
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0','CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2', 
          'CNRM-CM6-1', 'CNRM-ESM2-1','CanESM5', 'CanESM5-CanOE', 'EC-Earth3', 'EC-Earth3-Veg',
          'EC-Earth3-Veg-LR', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL','HadGEM3-GC31-MM', 'INM-CM4-8', 
          'INM-CM5-0', 'IPSL-CM6A-LR','MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 
          'MRI-ESM2-0','NESM3', 'UKESM1-0-LL']

best_models2 = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5', 
               'CanESM5-CanOE', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 
               'EC-Earth3-Veg-LR', 'GFDL-ESM4', 'MIROC-ES2L', 
                'MPI-ESM1-2-HR', 'NESM3']

best_models = ['ACCESS-ESM1-5', 'CAMS-CSM1-0', 'CanESM5-CanOE', 'CNRM-CM6-1', 'EC-Earth3-Veg-LR','GFDL-ESM4']




    


"""
MODEL DATA
----------

"""


def import_cmip6_slh_data(data_type = 'historical', use_models = 'bestmodels'):
    """
    Function that imports cmip6 sea level data
    
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245' 'ssp370', 'ssp585']
    
    For use_models choose ['bestmodels', 'allmodels']
    
    """
    


    
    # Define paths to data
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/SLH/slh_annual_{data_type}.nc'
    path_hist = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/SLH/slh_annual_historical.nc'

    # Open data file
    zos = xr.open_dataset(path) 
    zos_hist = xr.open_dataset(path_hist)
    
    # Only use models as defined
    if use_models == 'bestmodels':
        models = best_models
    
    zos = zos.where(zos.model.isin(models), drop=True)
    zos_hist = zos_hist.where(zos_hist.model.isin(models), drop=True)
    
    
    # For the cmip6 data the nodal and ibe shouldn't be removed

    
    # Only select the average station and create dataframes
    zos = zos.zos.sel(station='Average', drop = True).to_pandas().T
    zos_hist = zos_hist.zos.sel(station='Average', drop = True).to_pandas().T
    
    # Create one dataframe
    zos = pd.concat([zos_hist, zos])#.dropna(axis=1)
    
    return zos





def import_cmip6_wind_contribution_data(wind_model = 'NearestPoint', data_type = 'historical'):
    """
    Function that imports cmip6 wind contribution the sea level rise resulting from the regressions
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    
    """
    
    
    # Define paths to data
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/timeseries_{wind_model}_{data_type}.nc'
   
    # Open data file
    wcontr = xr.open_dataset(path) 

    # Only select the average station
    wcontr = wcontr.sel(station='Average', drop = True)
    
    return wcontr



def import_cmip6_wind_contribution_data_preproj(wind_model = 'NearestPoint', data_type = 'historical'):
    """
    Function that imports cmip6 wind contribution the sea level rise resulting from the regressions
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    
    """
    
    
    # Define paths to data
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/Projections/{wind_model}_wc_timeseries.csv'

    # Open data file
    wcontr = pd.read_csv(path, index_col='time')
    
    return wcontr




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




def import_cmip6_wind_data(model = 'NearestPoint', data_type = 'ssp119', use_models = 'bestmodels'):
    """
    Function that imports the observed wind data based on the preferred wind data model that is used for regression
    
    For model choose ['NearestPoint', 'Dangendorf', 'Timmerman']
    For data_type choose ['historical', 'piControl', 'ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    """
    
    
    scenario = import_cmip6_wind_help(model, data_type)
    hist = import_cmip6_wind_help(model, 'historical')
    
    # Only keep models occuring in both datasets
    scenario = scenario.where(scenario.model.isin(hist.model), drop = True)
    hist = hist.where(hist.model.isin(scenario.model), drop = True)
    
    data = xr.concat([hist, scenario], dim='time') # concatenate data sets
    
    
    # Only use models as defined
    if use_models == 'bestmodels':
        models = best_models
                
    data = data.where(data.model.isin(models), drop=True)
    
                                  
    return data



def import_cmip6_wind_help(model = 'NearestPoint', data_type = 'historical'):
    
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
            
        
        # Only select the average station
        dataset_annual = dataset_annual.sel(station='Average', drop = True)
        
    
            
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
    
    
    
    
    
    
    
    
    


def import_cmip6_regression_results(wind_model = 'NearestPoint', data_type = 'historical'):
    """
    Function that imports cmip6 wind contribution the sea level rise resulting from the regressions
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    
    """
    
    
    # Define paths to data
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/results_{wind_model}_{data_type}.nc'

    # Open data file
    results = xr.open_dataset(path) 

    # Only select the average station
    results = results.sel(station='Average', drop = True)
    
    return results.to_pandas().T




