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
from statistics import *


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



"""
OBSERVATIONS
------------

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
    plt.axhline(color='grey', linestyle='--')
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/tide_gauge.png')

    
    

    
def plot_obs_wind_data(data, model, data_type):
    """
    Function to make lineplots of the observed zonal and meridional wind data for each station 
    
    """
    
    
    u2_df = new_df_obs_wind_per_var(data, model = model)
    u2_df.plot(figsize=(9,3), title=f'Annual zonal wind stress ({data_type})', 
              ylabel = 'u$^2$ [m$^2$/s$^2$]',
             xlabel = 'Time [y]')
    plt.tight_layout()
    plt.axhline(color='grey', linestyle='--')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/u2_all_stations.png')
    
    
    v2_df = new_df_obs_wind_per_var(data, variable = 'v$^2$', model = model)
    v2_df.plot(figsize=(9,3), title=f'Annual meridional wind stress ({data_type})', 
              ylabel = 'v$^2$ [m$^2$/s$^2$]',
             xlabel = 'Time [y]')
    plt.tight_layout()
    plt.axhline(color='grey', linestyle='--')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/v2_all_stations.png')
    
       
def plot_obs_pres_data(data, model, data_type):
    """
    Function to make a lineplot of the observed pressure proxy for wind data
    
    """
    
    data.plot(figsize=(9,3), title='Annual observed atmospheric proxies', 
              ylabel = 'Regional averaged sea level pressure [Pa]',
             xlabel = 'Time [y]')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/obs_pres_data.png')

            
    
 

    
 
    
    
    
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
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/{variable}_per_station.png')
    
    
    
    
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
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/timeseries_per_station_{var}.png')

    
    
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
        
    plt.errorbar(stations, trend_lst, yerr=se_lst, fmt="o", label = 'Tide gauge')

    for variab in var:
        trend_lst = []
        se_lst = []
        for stat in stations:
            trend_lst.append(linregress(timeseries.index, timeseries[stat, variab]).slope)
            se_lst.append(linregress(timeseries.index, timeseries[stat, variab]).stderr)
        
        plt.errorbar(stations, trend_lst, yerr=se_lst, fmt="o", label = variab)

    plt.xlabel('station')
    plt.ylabel('Linear trend $\pm1\sigma$ [cm/y] ')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1))
    plt.axhline(color='grey', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/timeseries_trends_per_station.png')
    
    
    
    
    
    
    
def plot_obs_decadal_trends_timeseries_per_station(tg_data, timeseries, var, time_period, model, data_type, errorbar = True):
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
            ax.set_xlabel('time [y]')
            ax.set_ylim(-0.1,0.4)
            ax.axhline(color='grey', linestyle='--')
    
    labels = ['Tide gauge']+var
    fig.legend(labels=labels, loc=(0.57, 0.05))
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/{model}/{data_type}/{time_period}_trends_per_station_{model}.png')
    

    

"""
MODEL DATA
------------

"""

def plot_zos_data_per_station(data, data_type):
    """
    Function to make a lineplot of all cmip6 zos model data for each station
    
    """
    data = data.zos.assign_attrs(long_name='zos')
    
    data.plot.line(x='time', hue = 'model', col = 'station', col_wrap=3, add_legend=False, figsize = (18, 6), sharex=False)
    
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/zos_per_station.png')
    

    


def plot_zos_data_per_model(zos, data_type, station='Average'):
    """
    Function to make a lineplot of all cmip6 zos model data for each model
    For station choose ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']
    
    """
    zos = zos.zos.sel(station=station)
    
    cols = 4
    rows = 9
    
    
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

        

    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{data_type}/zos_per_model.png')

    
    
    
    
def plot_zos_data_per_model_allstations(zos, data_type):
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


    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{data_type}/zos_per_model_allstations.png')




def plot_cmip6_wind_data_per_model(data, wind_model, data_type, station = 'Average', tim_region = 'South'):
    """
    Function to make a lineplot of all cmip6 wind model data for each model
    For station choose ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']
    For region choose ['Channel', 'South', 'Mid-West', 'Mid-East', 'North-West', 'North-East']
    """
    cols = 4
    rows = 9
    
    
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

        

        
    

    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{wind_model}/wind_per_model_{data_type}.png')


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
    
    
def plot_cmip6_timeseries_per_station_one_model(zos, timeseries, model, var = ['wind_total']):
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
    

    
    
    
    

    
    





def plot_cmip6_trends_timeseries_per_station_model_averages(zos, timeseries, var, wind_model, data_type, errorbar = True):
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
        trend_lst.append(mean(trend_lst1))
        se_lst.append(mean(se_lst1))
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
            trend_lst.append(mean(trend_lst1))
            se_lst.append(mean(se_lst1))
        
        if errorbar == True:
            plt.errorbar(stations, trend_lst, yerr=se_lst, fmt=".", label = variab)
        else:
            plt.scatter(stations, trend_lst, marker='.', label = variab)
        

    plt.xlabel('station')
    if errorbar == True:
        plt.ylabel('Linear trend $\pm1\sigma$ [cm/y] ')
    else:
        plt.ylabel('Linear trend [cm/y] ')
    plt.tight_layout()
    plt.title('Trend per station averaged over all models')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.axhline(color='k', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{wind_model}/timeseries_trends_per_station_{data_type}.png')
    
    
    

def plot_zos_regression_result_per_model_one_station(zos, timeseries, labels, wind_reg_model, station = 'Average'):
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
    
    fig, axs = plt.subplots(9, 4, figsize=(24, 20))


    for i in range(9):

        for j in range(4):
            ax = axs[i,j]


            ax.plot(zos.time.values, zos.zos.sel(station=station, model=models[4*i+j]), color = 'darkgray')

            for label in labels:
                ax.plot(timeseries.time.values, timeseries[label].sel(station = station, model = models[4*i+j]).values)

            ax.set_xlabel('time [y]')
            ax.set_ylabel('zos [cm]')
            ax.axhline(color='k', linestyle='--', linewidth = 1)
            ax.set_title('model = ' + models[4*i+j])
            ax.set_ylim(y_min,y_max)
            plt.tight_layout()

        
            if i == 0 and j == 0:
                labels_leg = ['zos'] + labels
                ax.legend(labels = labels_leg)
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{wind_reg_model}/zos_timeseries_one_model_{station}_showzos_smoothed')
     
    
 

