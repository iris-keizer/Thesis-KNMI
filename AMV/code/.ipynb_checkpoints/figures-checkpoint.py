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
    plt.xlabel('time [y]', fontsize = fsize)
    plt.ylabel(f'{ylabel}', fontsize = fsize)
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_{title}_{window}')

    
    
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
                ax.legend(labels = wind_labels)
            
            if j == 0:
                ax.set_ylabel(ylabel, fontsize = fsize)
                
            if i == n_row - 1:
                ax.set_xlabel('time [y]')
        
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_cmip6_ac_{window}')

    
def plot_amv_cmip6_timeseries(data_df, ylabel = 'No label given', window = 21, ymin = -6, ymax = 12):
    '''
    Function to make a simple plot of a dataframe consisting of time series for different cmip6 models
    
    As an option, a lowess smoothing can be applied
    '''
    fsize = 12
    
    models = data_df.columns
    
    
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
                
                ax.plot(data_df.index, data_df[models[n_col*i+j]].values)
                ax.set_title(models[n_col*i+j], fontsize = fsize)
                ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
                
            ax.set_ylim(ymin, ymax)
            
            if i == 0 and j == 2:
                ax.legend(labels = wind_labels)
            
            if j == 0:
                ax.set_ylabel(ylabel, fontsize = fsize)
                
            if i == n_row - 1:
                ax.set_xlabel('time [y]')
        
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_cmip6_amv_{window}')
    
    
    
    
    
    
    
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
            
        plt.title(f'Atmospheric contribution to mean sea-level', fontsize = fsize)
            
            
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('time [y]', fontsize = fsize)
    plt.ylabel('sea-level contribution [cm]', fontsize = fsize)
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_era5&20cr_{window}')
    
    
    
def plot_result(results_era5, results_20cr, var, ylabel, ymin = -0.01, ymax = 0.15, window = 21):
    '''
    Function to plot a result of the regression between atmospheric contribution to sea-level and the AMV
    
    '''
    fsize = 12
    
    fig, axs = plt.subplots(3, 2, figsize = (15,8))
    
    for i, wm in enumerate(wind_labels):
        
        ax = axs[i,0]
        data_era5 = results_era5.swaplevel(0,1, axis=1)[wm]
        
        for column in AMV_names:
            dataT = data_era5[column].T
            ax.scatter(dataT.index, dataT[var].values)
        
        if i == 2:
            ax.set_xlabel('lag [y]', fontsize=fsize)
            
        ax.set_ylabel(ylabel, fontsize=fsize)
        ax.set_ylim(ymin, ymax)
        ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
        
        if i == 0:
            ax.set_title(f'ERA5 \n {wind_labels[i]}', fontsize=fsize)
        else:
            ax.set_title(wind_labels[i], fontsize=fsize)
        
        ax = axs[i,1]
        data_20cr = results_20cr.swaplevel(0,1, axis=1)[wm]
        
        for column in AMV_names:
            dataT = data_20cr[column].T
            ax.scatter(dataT.index, dataT[var].values)
        ax.set_ylim(ymin, ymax)
        ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
        
        if i == 2:
            ax.set_xlabel('lag [y]', fontsize=fsize)
        
        if i == 0:
            ax.legend(labels = AMV_names)
            ax.set_title(f'20CRv3 \n {wind_labels[i]}')
        else:
            ax.set_title(wind_labels[i])
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/results_{var}_{window}')
 




    
def plot_result_cmip6(results, var, ylabel, ymin = -0.01, ymax = 0.15, window = 21):
    '''
    Function to plot a result of the regression between atmospheric contribution to sea-level and the AMV
    
    '''
    fsize = 12
    
    models = list(results.swaplevel(0,1,axis=1)['NearestPoint'].swaplevel(0,1,axis=1)[0].columns)
    
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
                    data = results[models[n_col*i+j], wl].T[var]
                    ax.scatter(data.index, data.values)
                ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
                ax.set_title(models[n_col*i+j], fontsize = fsize)
                ax.set_ylim(ymin, ymax)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize = fsize)
                
            if i == n_row - 1:
                ax.set_xlabel('lag [y]', fontsize = fsize)
                
            if i == 0 and j == 2:
                ax.legend(labels = wind_labels, loc='upper right')
    
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/results_cmip6_{var}_{window}')
    
   
    
    
    
def plot_timeseries(timeseries, data, lags, data_type, window = 21, ymin = -1.1, ymax = 2.1):
    
    
    n_row = len(lags)
    n_col = 3
    fsize = 14
    
    
    fig, axs = plt.subplots(n_row, n_col, figsize = (18,2.8*n_row))
    
    
    for i in range(n_row):
        
        for j in range(n_col):
            
            
            ax = axs[i,j]
            ax.set_ylim(ymin,ymax)
            ax.plot(data.index, data[wind_labels[j]].values, color = 'gray', label = 'Dependent variable')
            ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
            
            for name in AMV_names:
                ts = timeseries[name, wind_labels[j], lags[i]]
                ax.plot(ts.index, ts.values, label = name)
                
                
            if j == 0:
                ax.set_ylabel('Atmospheric contribution\n to sea-level [cm]', fontsize=fsize)
                ax.legend(loc='lower left')
                
            if i == 0:
                ax.set_title(f'{wind_labels[j]}\nlag = {lags[i]} y', fontsize=fsize) 
            else:
                ax.set_title(f'lag = {lags[i]} y', fontsize=fsize) 
                
            if i == n_row-1:
                ax.set_xlabel('time [y]', fontsize=fsize)
    plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_{data_type}_window{window}')
    
    
    
def plot_timeseries_cmip6_onelag_allmodels(timeseries, lag = 0, window = 21, ymin = -1.1, ymax = 2.1):
    
    models = list(timeseries.swaplevel(0,2,axis=1)[0]['NearestPoint'].columns)
    
    
    fsize = 12
    
    
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
                data = timeseries[models[n_col*i+j]].swaplevel(0,1,axis=1)[lag]
                for wl in wind_labels:
                    ax.plot(data.index, data[wl].values)
                ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
                ax.set_title(models[n_col*i+j], fontsize = fsize)
                ax.set_ylim(ymin, ymax)
                
            if j == 0:
                ax.set_ylabel('AMV influence\n on wind influence\n on sea level change [cm]', fontsize = fsize)
                
            if i == n_row - 1:
                ax.set_xlabel('time [y]', fontsize = fsize)
                
            if i == 0 and j == 2:
                ax.legend(labels = wind_labels, loc='upper right')
    
    plt.tight_layout()
    
                    
                    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_cmip6_allmodels_lag{lag}_window{window}')
    
    