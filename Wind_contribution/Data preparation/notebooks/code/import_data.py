"""
File containing the Python functions to import the correct data used for the regression between wind stress and sea level height along the Dutch coast. 
Depending on the type of data, used wind model and whether observational or cmip6 data is used, the functions 

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:



"""








"""
IMPORTING DATA
--------------

Importing the annual data and use it for (regression) analysis
"""


# Function that imports observational data

def import_obs_slh_data():
    
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
    station_names = []

    for i in range(len(loc_num)):
            tg_data = pd.read_csv(path_tg + '/data/' + str(loc_num[i]) + 
                                  '.rlrdata', sep=';', header=None, names=names_col)
            tg_data = tg_data.set_index('time')
            tg_data.height = tg_data.height.where(~np.isclose(tg_data.height,-99999))
            tg_data.height = tg_data.height - tg_data.height.mean()

            if i==0:
                tg_data_df = pd.DataFrame(data=dict(time=tg_data.index, col_name=tg_data.height))
                tg_data_df = tg_data_df.set_index('time')
                tg_data_df.columns  = [str(loc_num[i])] 
            else:
                tg_data_df[str(loc_num[i])] = tg_data.height
            station_names.append(filelist_df['name'].loc[loc_num[i]].strip())

    tg_data_df = tg_data_df.rename(columns={"20": station_names[0], 
                              "22": station_names[1], "23": station_names[2],
                              "24": station_names[3], "25": station_names[4],
                              "32": station_names[5]})

    tg_data_df = tg_data_df.interpolate(method='slinear')
    tg_data_df['Average'] = tg_data_df.mean(axis=1) # Add column containing the average of the stations 
    tg_data_df = tg_data_df*0.001 # cm -> m
    
    # Data before 1890 is incorrect
    tg_data_df = tg_data_df[tg_data_df.index>=1890] 
    
    # Select tide gauge data from period_begin onwards and till period_end
    tg_data_df = tg_data_df[tg_data_df.index>=period_begin]
    tg_data_df = tg_data_df[tg_data_df.index<=period_end]
    
    return tg_data_df




def import_obs_wind_data(period_begin=1900, period_end=2000, model = 'NearestPoint', data = 'ERA5'):
    
    # Define paths to data
    path_wind_ERA5 = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/annual_dataset.nc' 
    path_wind_20cr = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/annual_dataset_20cr.nc'
    path_pres_ERA5 = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/pres_ERA5_annual.nc'
    path_pres_20cr = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/pres_annual.nc'
    path_tg = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/rlr_annual'
    path_locations = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/ERA5/Data/rlr_annual/filelist.txt'
    
    
    
    
    
    # Import tide gauge sea level data
    loc_num = [20, 22, 23, 24, 25, 32]
    col_names = ['id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality']
    filelist_df = pd.read_csv(path_locations, sep=';', header=None, names=col_names)
    filelist_df = filelist_df.set_index('id')
    filelist_df = filelist_df.loc[loc_num, :]
    names_col = ('time', 'height', 'interpolated', 'flags')
    station_names = []

    for i in range(len(loc_num)):
            tg_data = pd.read_csv(path_tg + '/data/' + str(loc_num[i]) + 
                                  '.rlrdata', sep=';', header=None, names=names_col)
            tg_data = tg_data.set_index('time')
            tg_data.height = tg_data.height.where(~np.isclose(tg_data.height,-99999))
            tg_data.height = tg_data.height - tg_data.height.mean()

            if i==0:
                tg_data_df = pd.DataFrame(data=dict(time=tg_data.index, col_name=tg_data.height))
                tg_data_df = tg_data_df.set_index('time')
                tg_data_df.columns  = [str(loc_num[i])] 
            else:
                tg_data_df[str(loc_num[i])] = tg_data.height
            station_names.append(filelist_df['name'].loc[loc_num[i]].strip())

    tg_data_df = tg_data_df.rename(columns={"20": station_names[0], 
                              "22": station_names[1], "23": station_names[2],
                              "24": station_names[3], "25": station_names[4],
                              "32": station_names[5]})

    tg_data_df = tg_data_df.interpolate(method='slinear')
    tg_data_df['Average'] = tg_data_df.mean(axis=1) # Add column containing the average of the stations 
    tg_data_df = tg_data_df*0.001 # mm -> m
    
    # Data before 1890 is incorrect
    tg_data_df = tg_data_df[tg_data_df.index>=1890] 
    
    # Select tide gauge data from period_begin onwards and till period_end
    tg_data_df = tg_data_df[tg_data_df.index>=period_begin]
    tg_data_df = tg_data_df[tg_data_df.index<=period_end]
    
    
    
    
    
    # Import wind or pressure data (depending on model)
    if model == 'NearestPoint' or model ==  'Timmerman':
        if data == 'ERA5':
            dataset_annual = xr.open_dataset(path_wind) 
            dataset_annual = dataset_annual.rename({'longitude': 'lon','latitude': 'lat'})

        elif data == '20cr':
            dataset_annual = xr.open_dataset(path_wind_20cr) 
            
        # Select data from period_begin onwards and till period_end
        dataset_annual = dataset_annual.where(dataset_annual.year>=period_begin, drop=True)
        dataset_annual = dataset_annual.where(dataset_annual.year<=period_end, drop=True)
            



    
        if model == 'NearestPoint':

            # Create wind data array per tide gauge station 
            df = [] # List containing the created dataframes

            for idx, i in enumerate(loc_num):
                df.append(pd.DataFrame(data={'time': dataset_annual.u2.year, 
                                                 'u2' : dataset_annual.u2.sel(lon = filelist_df['lon'].loc[i], 
                                                                              lat = filelist_df['lat'].loc[i], 
                                                                              method = 'nearest').to_series().dropna(), 
                                                 'v2' : dataset_annual.v2.sel(lon = filelist_df['lon'].loc[i], 
                                                                              lat = filelist_df['lat'].loc[i], 
                                                                              method = 'nearest').to_series().dropna()}))
                df[-1] = df[-1].set_index('time')

            wind_df = pd.concat([df[0],  df[1],  df[2],  df[3],  df[4],  df[5]], axis=1, keys = station_names)


        elif model ==  'Timmerman':
            # Create wind data array per region
            Timmerman_regions = timmerman_regions()
            mask = Timmerman_regions.mask(dataset_annual)
            regional_data = []  # List containing the dataset per region

            for i in range(1,7):
                regional_data.append(dataset_annual.where(mask == i))

            # Calculate regional averages
            df = [] # List containing the created dataframes

            for ds in regional_data:
                ds_avg = ds.mean('lon').mean('lat')
                df.append(pd.DataFrame(data={'time': ds_avg.year, 'u2' : ds_avg.u2, 'v2' : ds_avg.v2}))
                df[-1] = df[-1].set_index('time')

            wind_df = pd.concat([df[0],  df[1],  df[2],  df[3],  df[4],  df[5]], axis=1, keys = region_names)
    
    
        
        
        
        return (wind_df, dataset_annual, tg_data_df)
    
   

    
    # Import sea level pressure data
    elif model == 'Dangendorf':

        if data == 'ERA5':
            pres = xr.open_dataset(path_pres_ERA5) 
            pres = pres.rename({'longitude': 'lon','latitude': 'lat'})
            
            # Select data from period_begin onwards and till period_end
            pres = pres.where(pres.year>=period_begin, drop=True)
            pres = pres.where(pres.year<=period_end, drop=True)
            
            pres = pres.rename({'msl':'pres'})
            pres = pres.drop('expver')
        elif data == '20cr':
            pres = xr.open_dataset(path_pres_20cr) 
            
            # Select data from period_begin onwards and till period_end
            pres = pres.where(pres.year>=period_begin, drop=True)
            pres = pres.where(pres.year<=period_end, drop=True)
            
            pres = pres.rename({'prmsl':'pres'})

        
        if pres.year.values[0]>period_begin:
            print("The tide gauge data series begins later than the begin year that was given!")
        if pres.year.values[-1]<period_end:
            print("The tide gauge data series ends earlier than the final year that was given!")

        return (pres, tg_data_df)

