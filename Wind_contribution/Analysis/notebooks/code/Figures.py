"""
File containing the Python functions to plot data and results

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:



"""

# Import necessary packages
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress



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
                          regions[5]: data[regions[5],  variable],
                          regions[6]: data[regions[6],  variable],}, index = data.index)
    
    
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






# Declare global variables
stations = station_names()
regions = timmerman_region_names()
many_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
              'silver', 'lightcoral',  'maroon', 'tomato', 'chocolate', 'peachpuff', 'gold',  'goldenrod', 'yellow', 'yellowgreen', 'lawngreen',
              'palegreen', 'darkgreen', 'mediumseagreen', 'springgreen', 'aquamarine', 'mediumturquoise', 'paleturquoise', 'darkcyan', 'steelblue', 
               'dodgerblue', 'slategray',  'royalblue', 'navy', 'slateblue', 'darkslateblue', 'indigo',  'plum', 'darkmagenta', 'magenta', 'deeppink']




"""
Creating figures
----------------


"""


    
def plot_tg_data(data):
    """
    Function to make a lineplot of the tide gauge data for each station
    
    """
    
    data.plot(figsize=(9,3), title='Tide gauge time series', 
              ylabel = 'Sea level height above NAP [cm]',
             xlabel = 'Time [y]')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/tide_gauge.png')

    
    

    
def plot_obs_wind_data(data, model = 'NearestPoint'):
    """
    Function to make lineplots of the observed zonal and meridional wind data for each station 
    
    """
    
    
    u2_df = new_df_obs_wind_per_var(data, model = model)
    u2_df.plot(figsize=(9,3), title='Annual zonal wind stress', 
              ylabel = 'u$^2$ [m$^2$/s$^2$]',
             xlabel = 'Time [y]')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/NearestPoint/u2_all_stations.png')
    
    
    v2_df = new_df_obs_wind_per_var(data, variable = 'v$^2$', model = model)
    v2_df.plot(figsize=(9,3), title='Annual meridional wind stress', 
              ylabel = 'v$^2$ [m$^2$/s$^2$]',
             xlabel = 'Time [y]')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/NearestPoint/v2_all_stations.png')
    
       
def plot_obs_pres_data(data, model):
    """
    Function to make a lineplot of the observed pressure proxy for wind data
    
    """
    
    data.plot(figsize=(9,3), title='Annual observed atmospheric proxies', 
              ylabel = 'Regional averaged sea level pressure [Pa]',
             xlabel = 'Time [y]')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/tide_gauge.png')

            
    
def plot_zos_data(data, data_type):
    """
    Function to make a lineplot of all cmip6 zos model data for each station
    
    """
    
    
    data.zos.plot.line(x='time', hue='model', col = 'station', col_wrap=2, add_legend=False, figsize = (15, 10), sharex=False)
    
    
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/zos_per_station.png')
    
    
def plot_obs_result_per_station(data, variable, model, data_type):
    """
    Function to make a scatter plot of observational regression results per station for a specific variable
    
    For variable choose ['R$^2$', 'RMSE', 'Constant', 'u$^2$', 'v$^2$', 'trend']
    
    """
    
    plt.figure(figsize=(7,3))
    plt.scatter(data.index.values, data[variable].values, marker='x', label=variable)
    plt.tight_layout()
    plt.title(f'{variable} results of regression between slh and wind')
    plt.axhline(color='grey', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{variable}_per_station_{data_type}.png')
    
    
    
    
def plot_obs_timeseries_per_station(tg_data, timeseries, var, model, data_type):
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
        ax.set_xlabel('time [y]')
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
            ax.set_xlabel('time [y]')
            ax.set_ylabel('SLH [cm]')
            ax.set_ylim(-20,20)
            plt.tight_layout()
    
    labels=['tide gauge data']+var
    fig.legend(labels=labels, loc=(0.57, 0.1))
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/timeseries_per_station_{var}_{data_type}.png')
    
    
    
def plot_cmip6_wind_data(data, variable, model, data_type):
    """
    Function to make a lineplot of all cmip6 wind model data for each station
    
    """
    
    if model == 'NearestPoint':
        col = 'station'
    elif model == 'Timmerman':
        col = 'tim_region'
        
    data[variable].plot.line(x='time', hue='model', col = col, 
                             col_wrap=2, add_legend=False, figsize = (15, 10), sharex=False)
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{model}/{variable}_per_station_{data_type}.png')
    
def plot_cmip6_pres_data(data, variable, model, data_type):
    """
    Function to make a lineplot of all cmip6 wind model data for each station
    
    """

        
    data[variable].plot.line(x='time', hue='model', add_legend=False, figsize = (9, 3))
    if variable == 'Negative corr region':
        plt.title(f'Annual cmip6 ({data_type}) negative correlated atmospheric proxy')
    if variable == 'Positive corr region':
        plt.title(f'Annual cmip6 ({data_type}) positive correlated atmospheric proxy')
    plt.ylabel(f'ps [Pa]')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{model}/{variable}_per_station_{data_type}.png')
     

    
def plot_cmip6_two_variables(data, var1, var2, data_type):
    """
    Function to make a lineplot of all cmip6 wind model data for each station
    
    For var1, var2 choose ['r2', 'rmse', 'constant', 'u2_coef', 'v2_coef', 'trend_coef']
    
    """
    
    xr.plot.scatter(data, var1, var2, hue='model', col='station', col_wrap=3, sharex = False, figsize=(10,8))
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{var1}_{var2}_per_station_{data_type}.png')
    
    
    
    
def plot_cmip6_result_per_station(data, variable, data_type):
    """
    Function to make a scatter plot of all cmip6 wind model data for all models per station for a specific variable
    
    For variable choose ['r2', 'rmse', 'constant', 'u2_coef', 'v2_coef', 'trend_coef']
    
    """
    
    plt.figure(figsize=(8,4))
    plt.grid()
    xr.plot.scatter(data, 'station', variable, hue='model', colors=many_colors[:data.model.size], levels=data.model.size)
    plt.legend(bbox_to_anchor=(1.05, 1), ncol=2)
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{variable}_per_station_{data_type}.png')
    
    
def timeseries_per_station_one_model(zos, timeseries, model, var = ['wind_total']):
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
        ax.set_xlabel('time [y]')
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
            ax.set_xlabel('time [y]')
            ax.set_ylabel('SLH [cm]')
            ax.set_ylim(-13,13)
            plt.tight_layout()
    
    labels = ['zos']+var
    fig.legend(labels=labels, loc=(0.57, 0.1))
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/timeseries_per_station_{model}.png')
    
    

def plot_obs_trends_timeseries_per_station(tg_data, timeseries, var, model, data_type):
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
        
    plt.errorbar(stations, trend_lst, yerr=se_lst, fmt=".", label = 'Tide gauge')

    for variab in var:
        trend_lst = []
        se_lst = []
        for stat in stations:
            trend_lst.append(linregress(timeseries.index, timeseries[stat, variab]).slope)
            se_lst.append(linregress(timeseries.index, timeseries[stat, variab]).stderr)
        
        plt.errorbar(stations, trend_lst, yerr=se_lst, fmt=".", label = variab)

    plt.xlabel('station')
    plt.ylabel('Linear trend $\pm1\sigma$ [cm/y] ')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1))
    plt.axhline(color='grey', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/timeseries_trends_per_station_{data_type}.png')
    
    
    
    
    
    
    
def plot_obs_decadal_trends_timeseries_per_station(tg_data, timeseries, var, time_period, model, data_type, errorbar = True):
    """
    Function to make a plot of the trends over a certain decade of both 
    tide gauge observations and regression results per station
    
    time_period specifies the time period over which to calculate the trend 
    
    For var choose a list consisting of ['u$^2$', 'v$^2$', 'trend', 'total', 'wind total']
    
    """
    
    fig, axs = plt.subplots(4, 2, figsize=(14, 9))


    for i in range(4):


        ax = axs[i,0]
        years, trends, stds = get_decadal_trends_stds(tg_data[stations[2*i]], time_period)
        if errorbar == True:
            ax.errorbar(years, trends, yerr=stds, fmt=".", label = 'Tide gauge')
        else:
            ax.scatter(years, trends, marker='.', label = 'Tide gauge')
            
            
        for variab in var:
            years, trends, stds = get_decadal_trends_stds(timeseries[stations[2*i], variab], time_period)
            if errorbar == True:
                ax.errorbar(years, trends, yerr=stds, fmt=".", label = variab)
            else:
                ax.scatter(years, trends, marker='.', label = variab)
                
                
        ax.set_title(f'station={stations[2*i]} \n linear trends over {time_period} years')
        ax.set_xlabel('time [y]')
        if errorbar == True:
            ax.set_ylabel('linear trend [cm/year]\n $\pm 1\sigma$')
        else:
            ax.set_ylabel('linear trend [cm/year]')
                
        ax.set_ylim(-0.1,0.4)
        ax.axhline(color='grey', linestyle='--')


        ax = axs[i,1]
        if i == 3:
            fig.delaxes(axs[3,1])
        else:
            years, trends, stds = get_decadal_trends_stds(tg_data[stations[2*i+1]], time_period)
            if errorbar == True:
                ax.errorbar(years, trends, yerr=stds, fmt=".", label = 'Tide gauge')
            else:
                ax.scatter(years, trends, marker='.', label = 'Tide gauge')
                
                
            for variab in var:
                years, trends, stds = get_decadal_trends_stds(timeseries[stations[2*i+1], variab], time_period)
                if errorbar == True:
                    ax.errorbar(years, trends, yerr=stds, fmt=".", label = variab)
                else:
                    ax.scatter(years, trends, marker='.', label = variab)
                
            ax.set_title(f'station={stations[2*i]} \n linear trends over {time_period} years')
            ax.set_xlabel('time [y]')
            ax.set_ylim(-0.1,0.4)
            ax.axhline(color='grey', linestyle='--')
    
    labels = ['Tide gauge']+var
    fig.legend(labels=labels, loc=(0.57, 0.05))
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{time_period}_trends_per_station_{model}.png')
   