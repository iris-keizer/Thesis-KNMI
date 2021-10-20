"""
File containing the Python functions to plot data and results

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:



"""

# Import necessary packages
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt




"""
Practical functions
-------------------


"""


def station_names(): 
    """
    Function to obtain tide gauge station names as list
    
    """
    return ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']


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


def new_df_obs_wind_per_var(data,  variable  = 'u$^2$'):
    """
    Function to create a new dataframe of observed wind data containing only zonal or meridional wind stress data
    
    For variable choose ['u$^2$', 'v$^2$']
    
    """
    
    return pd.DataFrame({stations[0]: data[stations[0],  variable],
                      stations[1]: data[stations[1],  variable],
                      stations[2]: data[stations[2],  variable],
                      stations[3]: data[stations[3],  variable],
                      stations[4]: data[stations[4],  variable],
                      stations[5]: data[stations[5],  variable],
                      stations[6]: data[stations[6],  variable],}, index = data.index)




# Declare global variables
stations = station_names()
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
    
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/tide_gauge.png')

    
    

def plot_obs_wind_data(data):
    """
    Function to make lineplots of the observed zonal and meridional wind data for each station 
    
    """
    
    
    u2_df = new_df_obs_wind_per_var(data)
    u2_df.plot(figsize=(9,3), title='Annual zonal wind stress', 
              ylabel = 'u$^2$ [m$^2$/s$^2$]',
             xlabel = 'Time [y]')
    
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/Nearest Point/u2_all_stations.png')
    
    
    v2_df = new_df_obs_wind_per_var(data, variable = 'v$^2$')
    v2_df.plot(figsize=(9,3), title='Annual meridional wind stress', 
              ylabel = 'v$^2$ [m$^2$/s$^2$]',
             xlabel = 'Time [y]')
    
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/Nearest Point/v2_all_stations.png')
    
    
    
    
def plot_zos_data(data, data_type):
    """
    Function to make a lineplot of all cmip6 zos model data for each station
    
    """
    
    
    data.zos.plot.line(x='time', hue='model', col = 'station', col_wrap=2, add_legend=False, figsize = (15, 10), sharex=False)
    
    
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/zos_per_station.png')
    
    
def plot_obs_result_per_station(data, variable, data_type):
    """
    Function to make a scatter plot of observational regression results per station for a specific variable
    
    For variable choose ['R$^2$', 'RMSE', 'Constant', 'u$^2$', 'v$^2$', 'trend']
    
    """
    
    plt.figure(figsize=(7,3))
    plt.scatter(data.index.values, data[variable].values, marker='x', label=variable)
    plt.tight_layout()
    plt.title(f'{variable} results of regression between slh and wind')
    plt.axhline(color='grey', linestyle='--')
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/Nearest Point/{variable}_per_station_{data_type}.png')
    
    
def plot_obs_timeseries_per_station(tg_data, timeseries, var = ['Wind total']):
    """
    Function to make a plot of the tg_data timeseries and regression result for each station
    For var choose a list consisting of ['u$^2$', 'v$^2$', 'trend', 'Total', 'Wind total']
    
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
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/observations/Nearest Point/timeseries_per_station_{var}.png')
    
    
    
def plot_cmip6_wind_data(data, variable, data_type):
    """
    Function to make a lineplot of all cmip6 wind model data for each station
    
    """
    
    
    data[variable].plot.line(x='time', hue='model', col = 'station', col_wrap=2, add_legend=False, figsize = (15, 10), sharex=False)
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/cmip6/{variable}_per_station.png')
    
    

    
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
    
    
