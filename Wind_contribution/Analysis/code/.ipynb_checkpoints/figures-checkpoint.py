"""
File containing the Python functions to plot data and results

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
import regionmask 
import matplotlib 

import pandas as pd
import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import linregress



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
    return ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']




def get_savefig_path(data_type = 'observations',  wind_model =  'Nearest Point', wind_data_type = 'era5'):
    
    """
    Function to obtain the correct path to save a figure
    
    For data_type choose ['observations', 'cmip6', 'comparison']
    For wind_model choose ['Nearest Point', 'Dangendorf', 'Timmerman']
    For wind_data_type choose ['era5', '20cr']
    """
    
    path = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/'
    
    if data_type == 'comparison':
        path = path + f'{data_type}/'
    
    else:
        path = path + f'{data_type}/{wind_model}/{wind_data_type}/'
    
    return path



def new_df_obs_wind_per_var(data, variable  = 'u$^2$', model = 'NearestPoint'):
    """
    Function to create a new dataframe of observed wind data containing only zonal or meridional wind stress data
    
    For variable choose ['u$^2$', 'v$^2$']
    
    For model choose ['NearestPoint', 'Timmerman']
    
    """
    
    if model == 'NearestPoint':
        return pd.DataFrame({stations[0]: data[stations[0],  variable],
                          stations[1]: data[stations[1],  variable],
                          stations[2]: data[stations[2],  variable],
                          stations[3]: data[stations[3],  variable],
                          stations[4]: data[stations[4],  variable],
                          stations[5]: data[stations[5],  variable],
                          stations[6]: data[stations[6],  variable],}, index = data.index)
    
    
    elif model == 'Timmerman':
        
        return pd.DataFrame({regions[0]: data[regions[0],  variable],
                          regions[1]: data[regions[1],  variable],
                          regions[2]: data[regions[2],  variable],
                          regions[3]: data[regions[3],  variable],
                          regions[4]: data[regions[4],  variable],
                          regions[5]: data[regions[5],  variable],}, index = data.index)
    
    
    else: print('For model choose [NearestPoint, Timmerman]' )

    
    
    
def get_decadal_trends_stds(data, time_period):
    """
    Function to obtain lists of years, trends and standard errors
    
    """
    y0 = data.index[0] + time_period//2
    yend = data.index[-1] - time_period//2
    years = np.arange(y0, yend)
    starting_years = np.arange(data.index[0], data.index[-1]-time_period)
    trends = []
    stds = []
    
    for yr in starting_years:
        time = np.arange(yr, yr+time_period)
        y = data.loc[yr:yr+time_period-1].values
        trends.append(linregress(time,y).slope)
        stds.append(linregress(time,y).stderr)
        
    return years, trends, stds



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
    
    
    

# Declare global variables
stations = station_names()
regions = timmerman_region_names()
many_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
              'silver', 'lightcoral',  'maroon', 'tomato', 'chocolate', 'peachpuff', 'gold',  'goldenrod', 'yellow', 'yellowgreen', 'lawngreen',
              'palegreen', 'darkgreen', 'mediumseagreen', 'springgreen', 'aquamarine', 'mediumturquoise', 'paleturquoise', 'darkcyan', 'steelblue', 
               'dodgerblue', 'slategray',  'royalblue', 'navy', 'slateblue', 'darkslateblue', 'indigo',  'plum', 'darkmagenta', 'magenta', 'deeppink']

fsize = 12












"""
Creating figures
----------------


"""






"""
OBSERVATIONS
------------

"""

    
def plot_tg_data(data, title = True, period = 'fullperiod'):
    """
    Function to make a lineplot of the tide gauge data for each station
    
    """
    fsize = 13
    
    ax = data[stations[:-1]].plot(figsize=(9,3), fontsize = fsize)
    data['Average'].plot(ax=ax,color = 'k', linewidth = .95, fontsize = fsize)
    plt.ylabel('Sea level change [cm]', fontsize = fsize)
    plt.xlabel('Time [yr]', fontsize = fsize)
    if title == True:
        plt.title('Tide gauge time series', fontsize = fsize)
    plt.legend(bbox_to_anchor=(1, 1), fontsize = 12)
    plt.axhline(color='grey', linestyle='--')
    plt.xticks(fontsize = fsize)
    plt.yticks(fontsize = fsize)
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/tide_gauge.png', bbox_inches = 'tight', dpi = 500)


    
def plot_obs_wind_data(data, model, data_type, title = True, period = 'fullperiod'):
    """
    Function to make lineplots of the observed zonal and meridional wind data for each station 
    
    """
    fsize = 13
    fig = plt.figure(figsize=(9,3))
    u2_df = new_df_obs_wind_per_var(data, model = model)
    
    for station in u2_df.columns:
        if station == 'Average':
            plt.plot(u2_df.index, u2_df[station], color = 'k', label = station, linewidth = .95)
        else:
            plt.plot(u2_df.index, u2_df[station], label = station)
            
    if title == True:
        plt.title(f'Annual zonal wind stress ({data_type})', fontsize = fsize)
        
    plt.axhline(color='grey', linestyle='--')
    
    if data_type == 'era5':
        plt.legend(fontsize = 12)
    
    plt.xlabel('Time [yr]', fontsize = 15)
    plt.ylabel('U|U| [m$^2$/s$^2$]', fontsize = 15)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.xlim(1830, 2026)
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/observations/{model}/{data_type}/u2_all_stations.png', bbox_inches = 'tight', dpi = 500)
    
    
    fig = plt.figure(figsize=(9,3))
    v2_df = new_df_obs_wind_per_var(data, variable  = 'v$^2$', model = model)
    
    for station in v2_df.columns:
        if station == 'Average':
            plt.plot(v2_df.index, v2_df[station], color = 'k', label = station)
        else:
            plt.plot(v2_df.index, v2_df[station], label = station)
            
    if title == True:
        plt.title(f'Annual meridional wind stress ({data_type})', fontsize = fsize)
        
    plt.axhline(color='grey', linestyle='--')
    
    plt.xlabel('Time [yr]', fontsize = 15)
    plt.ylabel('V|V| [m$^2$/s$^2$]', fontsize = 15)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.xlim(1830, 2026)
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/observations/{model}/{data_type}/v2_all_stations.png', bbox_inches = 'tight', dpi = 500)
    
    
    
       
def plot_obs_pres_data(data, model, data_type, title = True, period = 'fullperiod'):
    """
    Function to make a lineplot of the observed pressure proxy for wind data
    
    """
    labels = ['Negative correlation area', 'Positive correlation area']
    cmap = matplotlib.cm.get_cmap('RdBu')    
    
    colors = [cmap(0.9999), cmap(0)]
    fsize = 13
    
    plt.figure(figsize=(9,3))
    for i, column in enumerate(data.columns):
        plt.plot(data.index, data[column]/100, label = labels[i], color = colors[i])
    if data_type == 'era5':
        plt.legend(fontsize = 14)
    plt.xlabel('Time [yr]', fontsize = 15)
    plt.ylabel('Sea level pressure [hPa]', fontsize = 15)
    plt.xlim(1830, 2026)
    plt.ylim(1007, 1019)
    plt.xticks(fontsize = 15)  
    plt.yticks(fontsize = 15)
        
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/obs_pres_data.png', bbox_inches = 'tight', dpi = 500)

            
    
 

    
 
    
    
    
def plot_obs_result_per_station(data, variable, model, data_type, period = 'fullperiod'):
    """
    Function to make a scatter plot of observational regression results per station for a specific variable
    
    For variable choose ['R$^2$', 'RMSE', 'Constant', 'u$^2$', 'v$^2$', 'trend']
    
    """
    
    plt.figure(figsize=(7,3))
    plt.scatter(data.index.values, data[variable].values, marker='x', label=variable)
    plt.tight_layout()
    plt.title(f'{variable} results of regression between slh and wind')
    plt.axhline(color='grey', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/{variable}_per_station.png', bbox_inches='tight')
    
    
def plot_obs_timeseries_one_station_tg_ac(tg_data, timeseries, station, model, data_type, title = True, period = 'fullperiod'):
    """
    Function to make a plot of the tg_data timeseries and atmospheric contribution for one station
    
    """
    fsize = 15
    
    fig= plt.figure(figsize=(9, 3))
    
    plt.plot(tg_data.index.values, tg_data[station].values, color = 'darkgray')
    plt.plot(timeseries.index.values, timeseries[station, 'wind total'].values)
    
    if title == True:
        plt.title('station = '+station, fontsize = fsize)
    plt.xlabel('Time [yr]', fontsize = fsize)
    plt.ylabel('Sea level change [cm]', fontsize = fsize)
    labels = ['Tide gauge data', 'Atmospheric contribution']
    plt.legend(labels = labels, fontsize = fsize)
    plt.axhline(color='grey', linestyle='--')
    if data_type == 'era5':
        plt.ylim(-7, 22)
        plt.xlim(1949,2021)
    if data_type == '20cr':
        plt.ylim(-17, 22)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/timeseries_{station}_tg_ac.png', bbox_inches='tight', dpi = 500)
    
    
    
    
def plot_obs_timeseries_one_station_ac_u_v(tg_data, timeseries, station, model, data_type, title = True, period = 'fullperiod'):
    """
    Function to make a plot of the tg_data timeseries and atmospheric contribution for one station
    
    """
    fsize = 15
    
    fig= plt.figure(figsize=(9, 3))
    
    plt.plot(timeseries.index.values, timeseries[station, 'wind total'].values, color = 'k')
    if model == 'NearestPoint':
        plt.plot(timeseries.index.values, timeseries[station, 'u$^2$'].values)
        plt.plot(timeseries.index.values, timeseries[station, 'v$^2$'].values)
        labels = ['Total wind influence', 'Zonal contribution', 'Meridional contribution']
    elif model == 'Timmerman':
        plt.plot(timeseries.index.values, timeseries[station, 'u$^2$ total'].values)
        plt.plot(timeseries.index.values, timeseries[station, 'v$^2$ total'].values)
        labels = ['Total wind influence', 'Zonal contribution', 'Meridional contribution']
    elif model == 'Dangendorf':
        plt.plot(timeseries.index.values, timeseries[station, 'Negative corr region'].values)
        plt.plot(timeseries.index.values, timeseries[station, 'Positive corr region'].values)
        labels = ['Total wind influence', 'Negative corr contribution', 'Positive corr contribution']
        
        
    if title == True:
        plt.title('station = '+station, fontsize = fsize)
        
    plt.xlabel('Time [yr]', fontsize = fsize)
    plt.ylabel('Sea level change [cm]', fontsize = fsize)
    plt.legend(labels = labels, fontsize = 13, loc = 'upper left', ncol = 2)
    plt.xticks(fontsize = 13)
    plt.xlim(1830, 2026)
    plt.yticks(fontsize = 13)
    plt.axhline(color='grey', linestyle='--')
    if data_type == 'era5':
        plt.ylim(-7, 9)
        plt.xlim(1949,2021)
    if data_type == '20cr':
        plt.ylim(-7, 9)
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/timeseries_{station}_ac_u_v.png', bbox_inches='tight', dpi = 500)
    
    
    
    
    
    
    
    
    
def plot_obs_timeseries_per_station(tg_data, timeseries, var, model, data_type, period = 'fullperiod'):
    """
    Function to make a plot of the tg_data timeseries and regression result for each station
    For var choose a list consisting of ['u$^2$', 'v$^2$', 'trend', 'total', 'wind total']
    
    """
    
    
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))


    for i in range(4):


        ax = axs[i,0]
        ax.plot(tg_data.index.values, tg_data[stations[2*i]].values, color='darkgray')
        for variab in var:
            ax.plot(timeseries.index.values, timeseries[stations[2*i], variab].values)
        ax.set_title(f'station='+stations[2*i])
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('SLH [cm]')
        ax.set_ylim(-20,20)
        plt.tight_layout()


        ax = axs[i,1]
        if i == 3:
            fig.delaxes(axs[3,1])
        else:

            ax.plot(tg_data.index.values, tg_data[stations[2*i+1]].values, color='darkgray')
            for variab in var:
                ax.plot(timeseries.index.values, timeseries[stations[2*i], variab].values)
            ax.set_title(f'station='+stations[2*i])
            ax.set_xlabel('Time [yr]')
            ax.set_ylabel('SLH [cm]')
            ax.set_ylim(-20,20)
            plt.tight_layout()
    
    labels=['tide gauge data']+var
    fig.legend(labels=labels, loc=(0.57, 0.1))
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/timeseries_per_station_{var}.png', bbox_inches='tight', dpi = 500)

    
    
def plot_obs_trends_timeseries_per_station(tg_data, timeseries, var, model, data_type, period = 'fullperiod'):
    """
    Function to make a plot of the trends over the whole timeseries of both 
    tide gauge observations and regression results per station
    
    For var choose a list consisting of ['u$^2$', 'v$^2$', 'trend', 'total', 'wind total']
    
    """
    
    
    plt.figure(figsize = (8.3,4))
    trend_lst = []
    se_lst = []
    for stat in stations:
        trend_lst.append(linregress(tg_data.index, tg_data[stat]).slope)
        se_lst.append(linregress(tg_data.index, tg_data[stat]).stderr)
        
    plt.errorbar(stations, trend_lst, yerr=se_lst, fmt="o", label = 'Tide gauge')

    for variab in var:
        trend_lst = []
        se_lst = []
        for stat in stations:
            trend_lst.append(linregress(timeseries.index, timeseries[stat, variab]).slope)
            se_lst.append(linregress(timeseries.index, timeseries[stat, variab]).stderr)
        
        plt.errorbar(stations, trend_lst, yerr=se_lst, fmt="o", label = variab)

    plt.xlabel('station')
    plt.ylabel('Linear trend $\pm1\sigma$ [cm/yr] ')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1))
    plt.axhline(color='grey', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/timeseries_trends_per_station.png', bbox_inches='tight', dpi = 500)
    
    
    
    
    
    
    
def plot_obs_decadal_trends_timeseries_per_station(tg_data, timeseries, var, time_period, model, data_type, 
                                                   errorbar = True, period = 'fullperiod'):
    """
    Function to make a plot of the trends over a certain decade of both 
    tide gauge observations and regression results per station
    
    time_period specifies the time period over which to calculate the trend 
    
    For var choose a list consisting of ['u$^2$', 'v$^2$', 'trend', 'total', 'wind total']
    
    """
    size = 7
    
    fig, axs = plt.subplots(4, 2, figsize=(14, 9))


    for i in range(4):


        ax = axs[i,0]
        years, trends, stds = get_decadal_trends_stds(tg_data[stations[2*i]], time_period)
        if errorbar == True:
            ax.errorbar(years, trends, yerr=stds, fmt=".", label = 'Tide gauge', s=size)
        else:
            ax.scatter(years, trends, marker='.', label = 'Tide gauge', s=size)
            
            
        for variab in var:
            years, trends, stds = get_decadal_trends_stds(timeseries[stations[2*i], variab], time_period)
            if errorbar == True:
                ax.errorbar(years, trends, yerr=stds, fmt=".", label = variab, s=size)
            else:
                ax.scatter(years, trends, marker='.', label = variab, s=size)
                
                
        ax.set_title(f'station={stations[2*i]} \n linear trends over {time_period} years')
        ax.set_xlabel('Time [yr]')
        if errorbar == True:
            ax.set_ylabel('linear trend [cm/yr]\n $\pm 1\sigma$')
        else:
            ax.set_ylabel('linear trend [cm/yr]')
                
        ax.set_ylim(-0.1,0.4)
        ax.axhline(color='grey', linestyle='--')


        ax = axs[i,1]
        if i == 3:
            fig.delaxes(axs[3,1])
        else:
            years, trends, stds = get_decadal_trends_stds(tg_data[stations[2*i+1]], time_period)
            if errorbar == True:
                ax.errorbar(years, trends, yerr=stds, fmt=".", label = 'Tide gauge', s=size)
            else:
                ax.scatter(years, trends, marker='.', label = 'Tide gauge', s=size)
                
                
            for variab in var:
                years, trends, stds = get_decadal_trends_stds(timeseries[stations[2*i+1], variab], time_period)
                if errorbar == True:
                    ax.errorbar(years, trends, yerr=stds, fmt=".", label = variab, s=size)
                else:
                    ax.scatter(years, trends, marker='.', label = variab, s=size)
                
            ax.set_title(f'station={stations[2*i]} \n linear trends over {time_period} years')
            ax.set_xlabel('Time [yr]')
            ax.set_ylim(-0.1,0.4)
            ax.axhline(color='grey', linestyle='--')
    
    labels = ['Tide gauge']+var
    fig.legend(labels=labels, loc=(0.57, 0.05))
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/{model}/{data_type}/{time_period}_trends_per_station_{model}.png', bbox_inches='tight', dpi = 500)
    

    
    
    
    
    
    
def plot_np_locations(title = True, period = 'fullperiod'):
    '''
    Function that plots a map of the Dutch coast indicating the locations of the tide gauge stations, reanalysis and cmip6 data
    '''
    fsize = 13
    
    tg_coords = station_coords()
    era5_coords = obs_np_coords('era5')
    cr_coords = obs_np_coords('20cr')
    cmip6_coords = cmip6_np_coords()
    
    
    fig = plt.figure(figsize=(8,16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if title == True:
        plt.title('Data locations along the Dutch coast')
    ax.set_extent([3.2, 7.2, 50.8, 54.1], ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    
    gl = ax.gridlines(draw_labels = True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    plt.xlabel('Longitude [°]')
    plt.ylabel('Latitude [°]')
    ax.add_feature(cf.OCEAN)
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.LAKES)
    ax.add_feature(cf.BORDERS)
    
    for i, station in enumerate(tg_coords.index):
        lat = tg_coords['lat'][station]
        lon = tg_coords['lon'][station]
        plt.scatter(lon, lat, s=55, marker='o', color='tab:red', 
                    label = f'{station} ({round(lat,1)}, {round(lon,1)})')
        plt.scatter(era5_coords['lon'][station], era5_coords['lat'][station], s=30, marker='o', color='k')
        plt.scatter(cr_coords['lon'][station], cr_coords['lat'][station], s=30, marker='v', color='k')
        plt.scatter(cmip6_coords['lon'][station], cmip6_coords['lat'][station], s=30, marker='s', color='k')


    # Add station names
    plt.text(tg_coords['lon']['Vlissingen']-0.3, tg_coords['lat']['Vlissingen']-0.35, 'Vlissingen', fontsize = fsize)
    plt.text(tg_coords['lon']['Hoek v. Holland']+0.08, tg_coords['lat']['Hoek v. Holland']-0.06, 'Hoek v. Holland', fontsize = fsize)
    plt.text(tg_coords['lon']['Den Helder']-0.02, tg_coords['lat']['Den Helder']-0.15, 'Den Helder', fontsize = fsize)
    plt.text(tg_coords['lon']['Delfzijl']-0.45, tg_coords['lat']['Delfzijl']-0.04, 'Delfzijl', fontsize = fsize)
    plt.text(tg_coords['lon']['Harlingen']+0.09, tg_coords['lat']['Harlingen']-0.04, 'Harlingen', fontsize = fsize)
    plt.text(tg_coords['lon']['IJmuiden']+0.05, tg_coords['lat']['IJmuiden']-0.05, 'IJmuiden', fontsize = fsize)
    

    plt.legend(labels = ['Tide gauge sea level data', 'ERA5 reanalysis wind data', 
                         '20CRv3 reanalysis wind data', 'CMIP6 data'], loc='upper left', fontsize = 12.5)

    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/np_data_locations.png', bbox_inches='tight', dpi = 500)


def plot_np_locations2(title = True, period = 'fullperiod'):
    '''
    Function that plots a map of the Dutch coast indicating the locations of the tide gauge stations, reanalysis and cmip6 data
    '''
    fsize = 13
    
    tg_coords = station_coords()
    era5_coords = obs_np_coords('era5')
    cr_coords = obs_np_coords('20cr')
    cmip6_coords = cmip6_np_coords()
    
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if title == True:
        plt.title('Data locations along the Dutch coast')
    ax.set_extent([3.0, 7.2, 50.8, 54.2], ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    
    gl = ax.gridlines(draw_labels = True, linestyle='--', alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 13.5}
    gl.ylabel_style = {'size': 13.5}
    ax.add_feature(cf.OCEAN)
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.LAKES)
    ax.add_feature(cf.BORDERS)
    
    for i, station in enumerate(tg_coords.index):
        lat = tg_coords['lat'][station]
        lon = tg_coords['lon'][station]
        plt.scatter(lon, lat, s=80, marker='o', color='tab:red', edgecolor = 'k',
                    label = f'{station} ({round(lat,1)}, {round(lon,1)})')
        plt.scatter(era5_coords['lon'][station], era5_coords['lat'][station], s=100, marker='x', color='k')
        plt.scatter(cr_coords['lon'][station], cr_coords['lat'][station], s=100, marker='o', facecolors='none', edgecolors='k')
        #plt.scatter(cmip6_coords['lon'][station], cmip6_coords['lat'][station], s=100, marker='o', facecolors='none', edgecolors='k')


    # Add station names
    plt.text(tg_coords['lon']['Vlissingen']-0.3, tg_coords['lat']['Vlissingen']-0.35, 'Vlissingen', fontsize = fsize)
    plt.text(tg_coords['lon']['Hoek v. Holland']+0.08, tg_coords['lat']['Hoek v. Holland']-0.06, 'Hoek v. Holland', fontsize = fsize)
    plt.text(tg_coords['lon']['Den Helder']-0.02, tg_coords['lat']['Den Helder']-0.15, 'Den Helder', fontsize = fsize)
    plt.text(tg_coords['lon']['Delfzijl']-0.5, tg_coords['lat']['Delfzijl']-0.04, 'Delfzijl', fontsize = fsize)
    plt.text(tg_coords['lon']['Harlingen']+0.09, tg_coords['lat']['Harlingen']-0.04, 'Harlingen', fontsize = fsize)
    plt.text(tg_coords['lon']['IJmuiden']+0.05, tg_coords['lat']['IJmuiden']-0.05, 'IJmuiden', fontsize = fsize)
    
    
    # Add axes labels
    plt.text(4.5, 50.5, 'Longitude [°]', fontsize = 15)
    plt.text(2.31, 52.2, 'Latitude [°]', fontsize = 15, rotation='vertical')
    
    
    plt.legend(labels = ['Tide gauge sea level data', 'ERA5 wind data', 
                         '20CRv3 wind data'], loc='upper left', fontsize = 15, fancybox=True, frameon = False)

    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/np_data_locations2.png', bbox_inches='tight', dpi = 500)


    
    
    
def plot_np_locations3(title = True, period = 'fullperiod'):
    '''
    Function that plots a map of the Dutch coast indicating the locations of the tide gauge stations, reanalysis and cmip6 data
    '''
    fsize = 13
    
    tg_coords = station_coords()
    era5_coords = obs_np_coords('era5')
    cr_coords = obs_np_coords('20cr')
    cmip6_coords = cmip6_np_coords()
    
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if title == True:
        plt.title('Data locations along the Dutch coast')
    ax.set_extent([3.0, 7.2, 50.8, 54.3], ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    
    gl = ax.gridlines(draw_labels = True, linestyle='--', alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 13.5}
    gl.ylabel_style = {'size': 13.5}
    ax.add_feature(cf.OCEAN)
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.LAKES)
    #ax.add_feature(cf.BORDERS)
    
    for i, station in enumerate(tg_coords.index):
        lat = tg_coords['lat'][station]
        lon = tg_coords['lon'][station]
        plt.scatter(lon, lat, s=80, marker='o', color='tab:red', edgecolor = 'k',
                    label = f'{station} ({round(lat,1)}, {round(lon,1)})')
        plt.scatter(era5_coords['lon'][station], era5_coords['lat'][station], s=100, marker='x', color='k')
        plt.scatter(cr_coords['lon'][station], cr_coords['lat'][station], s=100, marker='o', facecolors='none', edgecolors='navy')
        #plt.scatter(cmip6_coords['lon'][station], cmip6_coords['lat'][station], s=100, marker='o', facecolors='none', edgecolors='navy')


    # Add station names
    plt.text(5.8, 51.0, '1.   Vlissingen \n2.   Hoek v. Holland \n3.   IJmuiden \n4.   Den Helder \n5.   Harlingen \n6.   Delfzijl', fontsize = 14)
    
    # Add numbers to tide gauge stations
    plt.text(tg_coords['lon']['Vlissingen']-0.04, tg_coords['lat']['Vlissingen']-0.22, '1', fontsize = 15)
    plt.text(tg_coords['lon']['Hoek v. Holland']+0.08, tg_coords['lat']['Hoek v. Holland']-0.06, '2', fontsize = 15)
    plt.text(tg_coords['lon']['Den Helder']+0.02, tg_coords['lat']['Den Helder']-0.18, '4', fontsize = 15)
    plt.text(tg_coords['lon']['Delfzijl']-0.15, tg_coords['lat']['Delfzijl']-0.06, '6', fontsize = 15)
    plt.text(tg_coords['lon']['Harlingen']+0.08, tg_coords['lat']['Harlingen']-0.06, '5', fontsize = 15)
    plt.text(tg_coords['lon']['IJmuiden']+0.08, tg_coords['lat']['IJmuiden']-0.06, '3', fontsize = 15)
    
    # Add numbers to ERA5 data
    #plt.text(era5_coords['lon']['Vlissingen']-0.04, era5_coords['lat']['Vlissingen']+0.08, '1', fontsize = 15)
    #plt.text(era5_coords['lon']['Hoek v. Holland']-0.04, era5_coords['lat']['Hoek v. Holland']+0.08, '2', fontsize = 15)
    #plt.text(era5_coords['lon']['Den Helder']-0.04, era5_coords['lat']['Den Helder']+0.08, '4', fontsize = 15)
    #plt.text(era5_coords['lon']['Delfzijl']-0.04, era5_coords['lat']['Delfzijl']+0.08, '6', fontsize = 15)
    #plt.text(era5_coords['lon']['Harlingen']-0.04, era5_coords['lat']['Harlingen']+0.08, '5', fontsize = 15)
    #plt.text(era5_coords['lon']['IJmuiden']-0.04, era5_coords['lat']['IJmuiden']+0.08, '3', fontsize = 15)
    
    # Add numbers to 20CRv3 data
    plt.text(cr_coords['lon']['Vlissingen']-0.12, cr_coords['lat']['Vlissingen']+0.08, '1,2,3', fontsize = 15, color='navy')
    plt.text(cr_coords['lon']['Den Helder']-0.06, cr_coords['lat']['Den Helder']+0.08, '4,5', fontsize = 15, color='navy')
    plt.text(cr_coords['lon']['Delfzijl']-0.04, cr_coords['lat']['Delfzijl']+0.08, '6', fontsize = 15, color='navy')
    
    # Add numbers to CMIP6 data
    #plt.text(cmip6_coords['lon']['Vlissingen']-0.04, cmip6_coords['lat']['Vlissingen']+0.08, '1', fontsize = 15, color='navy')
    #plt.text(cmip6_coords['lon']['Hoek v. Holland']-0.09, cmip6_coords['lat']['Hoek v. Holland']+0.08, '2,3,4', fontsize = 15, color='navy')
    #plt.text(cmip6_coords['lon']['Harlingen']-0.04, cmip6_coords['lat']['Harlingen']+0.08, '5', fontsize = 15, color='navy')
    #plt.text(cmip6_coords['lon']['Delfzijl']-0.04, cmip6_coords['lat']['Delfzijl']+0.08, '6', fontsize = 15, color='navy')
    
    
    # Add axes labels
    plt.text(4.5, 50.5, 'Longitude [°]', fontsize = 15)
    plt.text(2.31, 52.2, 'Latitude [°]', fontsize = 15, rotation='vertical')
    
    
    plt.legend(labels = ['Tide gauge sea level data', 'ERA5 wind data', 
                         '20CRv3 wind data'], loc='upper left', fontsize = 14, fancybox=True, frameon = False)

    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/np_data_locations3.png', bbox_inches='tight', dpi = 500)
    
    
    
    


def timmerman_regions_plot(title = True):
    '''
    Function that plots a map of the North Sea indicating the locations of the Timmerman regions
    '''
    fsize = 13
    fig = plt.figure(figsize=(8,16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    if title == True:
        plt.title('Locations of the Timmerman regions')
    ax.set_extent([-7, 10, 47, 61], ccrs.PlateCarree())

    text_kws = dict(
        bbox=dict(color="none"),
        fontsize=15,
    )


    ax = timmerman_regions().plot(add_ocean=True, resolution="50m", proj=ccrs.Robinson(), label='name', text_kws=text_kws)

    ax.add_feature(cf.BORDERS, linewidth=.7)
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.LAKES)
    ax.coastlines(resolution='50m', linewidth=.7)
    gl = ax.gridlines(draw_labels = True)
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    gl.top_labels = False
    gl.right_labels = False
    plt.xticks(fontsize = fsize)
    plt.yticks(fontsize = fsize)

    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/observations/Timmerman/tim_regions.png', bbox_inches='tight', dpi = 500)



    
    
    
    
    
    
def dangendorf_regions_plot(pres_corr, tg_corr, data_type, title = True, year_start = 1950, year_final = 2015, period = 'fullperiod'):
    '''
    Function that plots a map of the North Atlantic region indicating the locations of the positive and negative correlation region
    '''
    fsize = 13
    
    # Obtain correlation
    corr = xr.corr(pres_corr, tg_corr, dim='time')


    # Select proxy region
    dang_coords_np = np.array([[-0.1, 54.9], [40.1, 54.9], [40.1, 75.1], [-0.1, 75.1]])
    dang_coords_pp = np.array([[-20.1, 24.9], [20.1, 24.9], [20.1, 45.1], [-20.1, 45.1]])

    dang_regions = regionmask.Regions([dang_coords_np, dang_coords_pp], numbers = [1,2], 
                                          names=["Negative correlation\n area", "Positive correlation\n area"], abbrevs=["NP", "PP"], 
                                          name="Dangendorf regions")

    # Plot the map
    fig = plt.figure(figsize=(8,16))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_extent([-35, 45, 20, 80], crs=ccrs.PlateCarree())

    plot = corr.sel(station = 'Average').plot(ax=ax, transform = ccrs.PlateCarree(), vmin = -0.6, vmax = 0.6, 
                                              cmap = 'RdBu_r', add_colorbar=False)
    cb = plt.colorbar(plot, orientation="vertical", shrink = 0.25)                                   
    cb.set_label(label='R$^2$', size=15)
    cb.ax.tick_params(labelsize=15)                                
    ax.set_title('')
    ax.coastlines(resolution = '50m', linewidth=.7, color = 'k')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True, linewidth = .7, color='k', alpha = .7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    gl.ylocator = mticker.FixedLocator([30, 40, 50, 60, 70])
    
    dang_regions.plot(ax = ax, label = 'name', line_kws = {'color':'snow', 'lw':3.1}, text_kws = dict(bbox=dict(color="none"), fontsize = 15, color = 'snow'))

    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/dang_regions_{data_type}_{year_start}_{year_final}.png', bbox_inches='tight', dpi = 500)

    
   
    
    
    
    
    
     
    
def dangendorf_all_stations_corr(pres_corr, tg_corr, data_type, year_start = 1950, year_final = 2015, period = 'fullperiod'):
    
    
    # Obtain correlation
    corr = xr.corr(pres_corr, tg_corr, dim='time')
    
    cols = 4
    rows = 2

    fig, axs = plt.subplots(rows, cols, figsize=(12, 6), subplot_kw = {'projection' : ccrs.Robinson(0)})

    for i in range(rows):

            for j in range (cols):

                if i == (rows-1) and (j == 3):
                    fig.delaxes(axs[i,j])


                else:
                    ax = axs[i,j]

                    fg = corr.sel(station = stations[cols*i+j]).plot(ax=ax, transform = ccrs.PlateCarree(), add_colorbar = False, 
                                                                     vmin = -0.6, vmax = 0.6, cmap = 'RdBu_r')

                    ax.set_extent([-35, 45, 20, 80], ccrs.PlateCarree())

                    if j == 0 and i == 0:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth=.7, color='k', alpha = .7)
                        gl.top_labels = False
                        gl.bottom_labels = False
                        gl.right_labels = False
                        gl.ylocator = mticker.FixedLocator([30, 40, 50, 60, 70])
                    elif j == 0 and i == 1:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth=.7, color='k', alpha = .7)
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.ylocator = mticker.FixedLocator([30, 40, 50, 60, 70])
                    elif i == 1:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth=.7, color='k', alpha = .7)
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.left_labels = False
                    else:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = False, linewidth=.7, color='k', alpha = .7)


                    ax.coastlines()

    
    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(fg, cax = cbar_ax, orientation = 'vertical')
    cbar.set_label('R$^2$ [-]')
    
    #plt.tight_layout()


    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/Dangendorf/{data_type}/dangendorf_corr_all_stations_{year_start}_{year_final}', bbox_inches='tight', dpi = 500)

    
    
    
    
    
    
    
    
    
def nearestpoint_all_stations_corr(corr, data_type, year_start = 1950, year_final = 2015, period = 'fullperiod'):
    
    
    cols = 4
    rows = 2

    fig, axs = plt.subplots(rows, cols, figsize=(12, 6), subplot_kw = {'projection' : ccrs.Robinson(0)})

    for i in range(rows):

            for j in range (cols):

                if i == (rows-1) and (j == 3):
                    fig.delaxes(axs[i,j])


                else:
                    ax = axs[i,j]

                    fg = corr.sel(station = stations[cols*i+j]).plot(ax=ax, transform = ccrs.PlateCarree(), add_colorbar = False, 
                                                                     vmin = -0.6, vmax = 0.6, cmap = 'RdBu_r')

                    ax.set_extent([-35, 30, 40, 80], ccrs.PlateCarree())

                    if j == 0 and i == 0:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth=.7, color='k', alpha = .7)
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.ylocator = mticker.FixedLocator([40, 50, 60, 70, 80])
                    elif j == 0 and i == 1:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth=.7, color='k', alpha = .7)
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.ylocator = mticker.FixedLocator([40, 50, 60, 70, 80])
                    else:
                        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth=.7, color='k', alpha = .7)
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.left_labels = False


                    ax.coastlines()

    
    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(fg, cax = cbar_ax, orientation = 'vertical')
    cbar.set_label('R$^2$ [-]')
    
    #plt.tight_layout()


    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/observations/NearestPoint/{data_type}/nearestpoint_corr_all_stations_{year_start}_{year_final}', bbox_inches='tight', dpi = 500)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

"""
MODEL DATA
------------

"""

def plot_zos_data_per_station(data, data_type, period = 'fullperiod'):
    """
    Function to make a lineplot of all cmip6 zos model data for each station
    
    """
    data = data.zos.assign_attrs(long_name='zos')
    
    data.plot.line(x='time', hue = 'model', col = 'station', col_wrap=3, add_legend=False, figsize = (18, 6), sharex=False)
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/cmip6/zos_per_station.png', bbox_inches='tight')
    

    


def plot_zos_data_per_model(zos, data_type, station='Average', period = 'fullperiod'):
    """
    Function to make a lineplot of all cmip6 zos model data for each model
    For station choose ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']
    
    """
    zos = zos.zos.sel(station=station)
    
    cols = 4
    rows = 7
    
    
    fig, axs = plt.subplots(rows, cols, figsize=(18, 18))
    

    for i in range(rows):

        for j in range (cols):
            
            ax = axs[i,j]
            ax.plot(zos.time.values, zos.sel(model = zos.model.values[4*i+j]))
            ax.axhline(color='grey', linestyle='--')
            ax.set_ylim(-13,13)
            ax.set_title('model = '+zos.model.values[4*i+j])
            ax.set_ylabel('zos [cm]')
            plt.tight_layout()

        

    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/zos_per_model.png', bbox_inches='tight')

    
    
    
    
def plot_zos_data_per_model_allstations(zos, data_type, period = 'fullperiod'):
    """
    Function to make a lineplot of all cmip6 zos model data for each model
    
    """
    cols = 4
    rows = 10
    
    
    fig, axs = plt.subplots(rows, cols, figsize=(18, 18))
    

    for i in range(rows):

        for j in range (cols):
            
            if i == (rows-1) and (j == 1 or j == 2 or j == 3):
                fig.delaxes(axs[i,j])
                
                
            else:
                ax = axs[i,j]

                
                for station in stations:
                    ax.plot(zos.time.values, zos.zos.sel(model = zos.model.values[4*i+j], station = station))
                ax.axhline(color='grey', linestyle='--')
                ax.set_ylim(-13,13)
                ax.set_title('model = '+zos.model.values[4*i+j])
                ax.set_ylabel('zos [cm]')
                plt.tight_layout()


    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/cmip6/{data_type}/zos_per_model_allstations.png', bbox_inches='tight')



    
    
    

def plot_cmip6_wind_data_per_model(data, wind_model, data_type, station = 'Average', tim_region = 'South', period = 'fullperiod'):
    """
    Function to make a lineplot of all cmip6 wind model data for each model
    For station choose ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']
    For region choose ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']
    """
    cols = 4
    rows = 7
    
    
    if wind_model == 'NearestPoint' or wind_model == 'Timmerman':
        ymax = 33
        ymin = -1
    elif wind_model == 'Dangendorf':
        ymax = 100100
        ymin = 97400
        
    
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    

    for i in range(rows):

        for j in range (cols):
            
            ax = axs[i,j]


            if wind_model == 'NearestPoint':
                ax.plot(data.time.values, data['u2'].sel(station=station, model = data.model.values[4*i+j]))
                ax.plot(data.time.values, data['v2'].sel(station=station, model = data.model.values[4*i+j]))
            elif wind_model == 'Timmerman':
                ax.plot(data.time.values, data['u2'].sel(tim_region=tim_region, model = data.model.values[4*i+j]))
                ax.plot(data.time.values, data['v2'].sel(tim_region=tim_region, model = data.model.values[4*i+j]))
            if wind_model == 'Dangendorf':
                ax.plot(data.time.values, data['Negative corr region'].sel(model = data.model.values[4*i+j]))
                ax.plot(data.time.values, data['Positive corr region'].sel(model = data.model.values[4*i+j]))
                    
            if wind_model == 'NearestPoint' or wind_model == 'Timmerman':
                ax.axhline(color='grey', linestyle='--')
            ax.set_ylim(ymin,ymax)
            if wind_model == 'NearestPoint':
                ax.set_title('model = '+data.model.values[4*i+j] + '\n station = '+station)
                labels = ['u$^2$', 'v$^2$']
            elif wind_model == 'Timmerman':
                ax.set_title('model = '+data.model.values[4*i+j] + '\n region = '+tim_region)
                labels = ['u$^2$', 'v$^2$']
            elif wind_model == 'Dangendorf':
                ax.set_title('model = '+data.model.values[4*i+j])
                labels = ['Neg. corr region', 'Pos. corr region']
            
            if i==0 and j==0:
                ax.legend(labels=labels)
            
            ax.set_ylabel('Wind stress [m$^2$/s$^2$]')
            plt.tight_layout()

        

        
    



    
    
    
    
    
    
def plot_cmip6_pres_data(data, variable, model, data_type, period = 'fullperiod'):
    """
    Function to make a lineplot of all cmip6 wind model data for each station
    
    """

        
    data[variable].plot.line(x='time', hue='model', add_legend=False, figsize = (9, 3))
    if variable == 'Negative corr region':
        plt.title(f'Annual cmip6 ({data_type}) negative correlated atmospheric proxy')
    if variable == 'Positive corr region':
        plt.title(f'Annual cmip6 ({data_type}) positive correlated atmospheric proxy')
    plt.ylabel(f'ps [Pa]')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/{period}/cmip6/{model}/{variable}_per_station_{data_type}.png', bbox_inches='tight')
     
    
    
    
    
    
    

    
def plot_cmip6_two_variables(data, var1, var2, data_type, period = 'fullperiod'):
    """
    Function to make a lineplot of all cmip6 wind model data for each station
    
    For var1, var2 choose ['r2', 'rmse', 'constant', 'u2_coef', 'v2_coef', 'trend_coef']
    
    """
    
    xr.plot.scatter(data, var1, var2, hue='model', col='station', col_wrap=3, sharex = False, figsize=(10,8))
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/{var1}_{var2}_per_station_{data_type}.png', bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
def plot_cmip6_result_per_station(data, variable, data_type, period = 'fullperiod'):
    """
    Function to make a scatter plot of all cmip6 wind model data for all models per station for a specific variable
    
    For variable choose ['r2', 'rmse', 'constant', 'u2_coef', 'v2_coef', 'trend_coef']
    
    """
    
    plt.figure(figsize=(8,4))
    plt.grid()
    xr.plot.scatter(data, 'station', variable, hue='model', colors=many_colors[:data.model.size], levels=data.model.size)
    plt.legend(bbox_to_anchor=(1.05, 1), ncol=2)
    
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/{variable}_per_station_{data_type}.png', bbox_inches='tight')
    
    
    
    
    
    
    
    
def plot_cmip6_timeseries_per_station_one_model(zos, timeseries, model, var = ['wind_total'], period = 'fullperiod'):
    """
    Function to make a plot of the zos timeseries and regression result for each station for a specific model
    For model choose timeseries.model.values[i]
    For var choose a list consisting of ['u2', 'v2', 'trend', 'total', 'wind_total']
    
    """
    
    
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))


    for i in range(4):


        ax = axs[i,0]
        ax.plot(zos.time.values, zos.zos.sel(model=model, station = zos.station.values[2*i]), 
                color='darkgray')
        for variab in var:
            ax.plot(timeseries.time.values, timeseries[variab].sel(model=model, station = zos.station.values[2*i]))
        ax.set_title(f'station={zos.station.values[2*i]}, model={model}')
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('SLH [cm]')
        ax.set_ylim(-13,13)
        plt.tight_layout()


        ax = axs[i,1]
        if i == 3:
            fig.delaxes(axs[3,1])
        else:

            ax.plot(zos.time.values, zos.zos.sel(model=model, station = zos.station.values[2*i+1]), 
                    color='darkgray')
            for variab in var:
                ax.plot(timeseries.time.values, timeseries[variab].sel(model=model, station = zos.station.values[2*i+1]))
            ax.set_title(f'station={zos.station.values[2*i+1]}, model={model}')
            ax.set_xlabel('Time [yr]')
            ax.set_ylabel('SLH [cm]')
            ax.set_ylim(-13,13)
            plt.tight_layout()
    
    labels = ['zos']+var
    fig.legend(labels=labels, loc=(0.57, 0.1))
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/timeseries_per_station_{model}.png', bbox_inches='tight')
    

    
    
    
    

    
    





def plot_cmip6_trends_timeseries_per_station_model_averages(zos, timeseries, var, wind_model, data_type, errorbar = True, period = 'fullperiod'):
    """
    Function to make a plot of the trends over the whole timeseries of both 
    tide gauge observations and regression results per station averaged over all models
    
    For var choose a list consisting of ['u$^2$', 'v$^2$', 'trend', 'total', 'wind total']
    
    """
    
    
    plt.figure(figsize = (8.3,4))
    trend_lst = []
    se_lst = []
    for stat in stations:
        trend_lst1 = []
        se_lst1 = []
        for model in timeseries.model.values:
            trend_lst1.append(linregress(zos.time.values, zos.zos.sel(station=stat, model = model).values).slope)
            se_lst1.append(linregress(zos.time.values, zos.zos.sel(station=stat, model = model).values).stderr)
        trend_lst.append(np.mean(trend_lst1))
        se_lst.append(np.mean(se_lst1))
    if errorbar == True:
        plt.errorbar(stations, trend_lst, yerr=se_lst, fmt="o", label = 'Tide gauge')
    else:
        plt.scatter(stations, trend_lst, marker='o', label = 'Tide gauge')
        
    for variab in var:
        trend_lst = []
        se_lst = []
        for stat in stations:
            trend_lst1 = []
            se_lst1 = []
            for model in timeseries.model.values:
                trend_lst1.append(linregress(timeseries.time.values, timeseries[variab].sel(station=stat, model = model)).slope)
                se_lst1.append(linregress(timeseries.time.values, timeseries[variab].sel(station=stat, model = model)).stderr)
            trend_lst.append(np.mean(trend_lst1))
            se_lst.append(np.mean(se_lst1))
        
        if errorbar == True:
            plt.errorbar(stations, trend_lst, yerr=se_lst, fmt=".", label = variab)
        else:
            plt.scatter(stations, trend_lst, marker='.', label = variab)
        

    plt.xlabel('station')
    if errorbar == True:
        plt.ylabel('Linear trend $\pm1\sigma$ [cm/yr] ')
    else:
        plt.ylabel('Linear trend [cm/yr] ')
    plt.tight_layout()
    plt.title('Trend per station averaged over all models')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.axhline(color='k', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/timeseries_trends_per_station_{data_type}.png', bbox_inches='tight')
    
    
    

    
    
    
    
    
    
def plot_zos_regression_result_per_model_one_station(zos, timeseries, labels, wind_reg_model, station = 'Average', period = 'fullperiod'):
    """
    
    Function to make a plot of the zos timeseries and regression results for all cmip6 models, for one station and for all 
    the wind contributions 
    For station choose ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']
    
    Labels should be a list containing a subset of 
    Nearest Point:
    ['u2', 'v2', 'trend', 'total', 'wind_total']
    
    
    Timmerman:
    ['total', 'wind_total', 'channel', 'south', 'midwest', 'mideast', 'northwest',
    'northeast', 'u2_total', 'v2_total', 'trend', 'channel_u2', 'south_u2', 'midwest_u2', 'mideast_u2', 'northwest_u2',
    'northeast_u2', 'channel_v2', 'south_v2', 'midwest_v2', 'mideast_v2', 'northwest_v2','northeast_u2']
    
    
    Dangendorf:
    ['neg_corr_region', 'pos_corr_region', 'trend', 'total', 'wind_total']
    
    """
    models = timeseries.model.values
    
    y_min = 15
    y_max = -15
    
    fig, axs = plt.subplots(7, 4, figsize=(24, 20))


    for i in range(7):

        for j in range(4):
            ax = axs[i,j]


            ax.plot(zos.time.values, zos.zos.sel(station=station, model=models[4*i+j]), color = 'darkgray')

            for label in labels:
                ax.plot(timeseries.time.values, timeseries[label].sel(station = station, model = models[4*i+j]).values)

            ax.set_xlabel('Time [yr]', fontsize = 14)
            ax.set_ylabel('zos [cm]', fontsize = 14)
            ax.axhline(color='k', linestyle='--', linewidth = 1)
            ax.set_title('model = ' + models[4*i+j], fontsize = 14)
            ax.set_ylim(y_min,y_max)
            plt.tight_layout()

        
            if i == 0 and j == 0:
                labels_leg = ['zos'] + labels
                ax.legend(labels = labels_leg, loc = 'upper left', fontsize = 14)
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/zos_timeseries_one_model_{station}_showzos_smoothed', bbox_inches='tight')
     



def plot_zos_wind_effect_1993_2014(zos, timeseries, labels, wind_reg_model, station = 'Average', period = 'fullperiod'):
    """
    
    
    
    """
    models = timeseries.model.values
    
    y_min = -7
    y_max = 15
    
    fig, axs = plt.subplots(7, 4, figsize=(24, 20))

    zos = zos.where(zos.time >= 1993, drop = True)
    timeseries = timeseries.where(zos.time >= 1993, drop = True)
    
    for i in range(7):

        for j in range(4):
            ax = axs[i,j]


            ax.plot(zos.time.values, zos.zos.sel(station=station, model=models[4*i+j]), color = 'darkgray')

            for label in labels:
                ax.plot(timeseries.time.values, timeseries[label].sel(station = station, model = models[4*i+j]).values)

            ax.set_xlabel('Time [yr]', fontsize = 14)
            ax.set_ylabel('zos [cm]', fontsize = 14)
            ax.axhline(color='k', linestyle='--', linewidth = 1)
            ax.set_title('model = ' + models[4*i+j], fontsize = 14)
            ax.set_ylim(y_min,y_max)
            plt.tight_layout()

        
            if i == 0 and j == 0:
                labels_leg = ['zos', 'wind influence']
                ax.legend(labels = labels_leg, loc = 'upper left', fontsize = 14)
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/zos_wind_effect_average_1993_2014', bbox_inches='tight')
     
    


def plot_zos_without_wind_effect_1993_2014(zos, timeseries, station = 'Average', period = 'fullperiod'):
    """
    
    
    
    """
    models = timeseries.model.values
    
    y_min = -7
    y_max = 12
    
    fig, axs = plt.subplots(7, 4, figsize=(24, 20))

    zos = zos.where(zos.time >= 1993, drop = True)
    timeseries = timeseries.where(zos.time >= 1993, drop = True)
    
    for i in range(7):

        for j in range(4):
            ax = axs[i,j]


            ax.plot(zos.time.values, zos.zos.sel(station=station, model=models[4*i+j]), color = 'darkgray')
            ax.plot(timeseries.time.values, timeseries.zos.sel(station = station, model = models[4*i+j]).values)

            ax.set_xlabel('Time [yr]', fontsize = 14)
            ax.set_ylabel('zos [cm]', fontsize = 14)
            ax.axhline(color='k', linestyle='--', linewidth = 1)
            ax.set_title('model = ' + models[4*i+j], fontsize = 14)
            ax.set_ylim(y_min,y_max)
            plt.tight_layout()

        
            if i == 0 and j == 0:
                labels_leg = ['zos', 'zos without wind']
                ax.legend(labels = labels_leg, loc = 'upper left', fontsize = 14)
    
    plt.savefig(f'/Users/iriskeizer/Documents/Wind effect/Figures/Wind contribution/CMIP6/NearestPoint/zos_without_wind_effect_average_1993_2014', bbox_inches='tight')
     
  
