"""
File containing the Python functions to plot the figures for the regression between AMV and the atmospheric contribution to sea level height at the Dutch coast.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
lagged_regression_obs.ipynb

"""

# Import necessary packages
import math

import statsmodels.api as sm
import matplotlib.pyplot as plt

lowess = sm.nonparametric.lowess

AMV_names = ['HadISSTv2', 'ERSSTv5', 'COBE-SST2']
wind_labels = ['NearestPoint', 'Timmerman', 'Dangendorf']



"""
Plot figures
------------


"""




def plot_df_timeseries(data_df, ylabel = 'No label given', title = 'No title given', window = 21):
    '''
    Function to make a simple plot of a dataframe consisting of time series
    
    As an option, a lowess smoothing can be applied
    '''
    
    fsize = 12
    
    plt.figure(figsize = (9,3))
    
    for column in data_df:
        plt.plot(data_df.index, data_df[column].values, label = column)
        plt.title(f'{title}', fontsize = fsize)
        
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('Time [yr]', fontsize = fsize)
    plt.ylabel(f'{ylabel}', fontsize = fsize)
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/SST/timeseries_{title}_{window}')

    
    
def plot_ac_cmip6_timeseries(data_df, ylabel = 'No label given', window = 21, ymin = -6, ymax = 12):
    '''
    Function to make a simple plot of a dataframe consisting of time series for different cmip6 models
    
    As an option, a lowess smoothing can be applied
    '''
    fsize = 12
    
    models = data_df['NearestPoint'].columns
    
    data_df = data_df.swaplevel(0,1,axis=1)
    
    
    n_col = 3
    n_row = math.ceil(len(models) / n_col)
    n_delete = len(models) % n_col
    
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 2.5*n_row))
    
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            ax = axs[i,j]
                
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
            
            
            else:
                
                for wl in wind_labels:
                    ax.plot(data_df.index, data_df[models[n_col*i+j], wl].values)
                ax.set_title(models[n_col*i+j], fontsize = fsize)
            
            ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
            ax.set_ylim(ymin, ymax)
            
            if i == 0 and j == 2:
                ax.legend(labels = wind_labels, loc = 'upper right')
            
            if j == 0:
                ax.set_ylabel(ylabel, fontsize = fsize)
                
            if i == n_row - 1:
                ax.set_xlabel('Time [yr]', fontsize = fsize)
        
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/SST/timeseries_cmip6_ac_{window}')

    
    
    
def plot_era5_20cr_timeseries(data_era5, data_20cr, window = 21):
    '''
    Function to make a plot of both era5 and 20cr atmospheric contribution time series
    '''
    fsize = 12
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    plt.figure(figsize = (9,3))
    
    for i, column in enumerate(data_era5.columns):
        plt.plot(data_20cr.index, data_20cr[column], color = colors[i], label = column + ' - 20CR', alpha = 0.6)
        plt.plot(data_era5.index, data_era5[column], color = colors[i], label = column + ' - ERA5')
            
        plt.title(f'Atmospheric contribution\n to sea-level [cm]', fontsize = fsize)
            
            
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('Time [yr]', fontsize = fsize)
    plt.ylabel('sea-level contribution [cm]', fontsize = fsize)
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/SST/timeseries_era5&20cr_{window}')
    
    
    
def plot_sst_timeseries(sst, skt, locations, name = ''):
    '''
    Function to make plots of the SST and SKT time series for several locations
    
    locations should be a list of [lat, lon] coordinates
    
    '''
    
    fsize = 12
    
    n_col = 4
    n_row = math.ceil(len(locations) / n_col)
    n_delete = len(locations) % n_col
    
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 2.5*n_row))
    
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            ax = axs[i,j]
            
            
            if i == n_row-1 and j in range(n_delete, n_col) and not n_delete == 0:
                fig.delaxes(axs[i,j])

            
            else:
                lat = locations[n_col*i+j][0]
                lon = locations[n_col*i+j][1]
                ax.plot(sst.year.values, sst.sel(lat=lat, lon=lon, method='Nearest').values)
                ax.plot(skt.year.values, skt.sel(lat=lat, lon=lon, method='Nearest').values)
                ax.set_title(f'lat={lat}, lon={lon}', fontsize = fsize)
                #ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
                
            #ax.set_ylim(ymin, ymax)
            
            
            if j == 0:
                ax.set_ylabel('Sea surface\n temperature [K]', fontsize = fsize)
                
            if i == n_row - 1:
                ax.set_xlabel('Time [yr]')
                
            if i == 0 and j == 3:
                ax.legend(labels=['SST', 'SKT'], loc='upper right')
        
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/SST/timeseries_cmip6_sst_{name}')
    
    
    