"""
File containing the Python functions to plot the figures for the regression between AMV and the atmospheric contribution to sea level height at the Dutch coast.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
lagged_regression_obs.ipynb

"""

# Import necessary packages

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
    plt.figure(figsize = (9,3))
    
    for column in data_df:
        plt.plot(data_df.index, data_df[column].values, label = column)
        plt.title(f'{title}')
        
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('time [y]')
    plt.ylabel(f'{ylabel}')
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_{title}_{window}')
    
    
def plot_era5_20cr_timeseries(data_era5, data_20cr, window = 21):
    '''
    Function to make a plot of both era5 and 20cr atmospheric contribution time series
    '''
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    plt.figure(figsize = (9,3))
    
    for i, column in enumerate(data_era5.columns):
        plt.plot(data_20cr.index, data_20cr[column], color = colors[i], label = column + ' - 20CR', alpha = 0.6)
        plt.plot(data_era5.index, data_era5[column], color = colors[i], label = column + ' - ERA5')
            
        plt.title(f'Atmospheric contribution to mean sea-level')
            
            
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('time [y]')
    plt.ylabel('sea-level contribution [cm]')
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_era5&20cr_{window}')
    
    
    
def plot_result(results_era5, results_20cr, var, ylabel, ymin = -0.01, ymax = 0.15, window = 21):
    '''
    Function to plot a result of the regression between atmospheric contribution to sea-level and the AMV
    
    '''
    
    
    fig, axs = plt.subplots(3, 2, figsize = (15,8))
    
    for i, wm in enumerate(wind_labels):
        
        ax = axs[i,0]
        data_era5 = results_era5.swaplevel(0,1, axis=1)[wm]
        
        for column in AMV_names:
            dataT = data_era5[column].T
            ax.scatter(dataT.index, dataT[var].values)
        
        if i == 2:
            ax.set_xlabel('lag [-]')
            
        ax.set_ylabel(ylabel)
        ax.set_ylim(ymin, ymax)
        ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
        
        if i == 0:
            ax.set_title(f'ERA5 \n {wind_labels[i]}')
        else:
            ax.set_title(wind_labels[i])
        
        ax = axs[i,1]
        data_20cr = results_20cr.swaplevel(0,1, axis=1)[wm]
        
        for column in AMV_names:
            dataT = data_20cr[column].T
            ax.scatter(dataT.index, dataT[var].values)
        ax.set_ylim(ymin, ymax)
        ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
        
        if i == 2:
            ax.set_xlabel('lag [-]')
        
        if i == 0:
            ax.legend(labels = AMV_names)
            ax.set_title(f'20CRv3 \n {wind_labels[i]}')
        else:
            ax.set_title(wind_labels[i])
    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/results_{var}_{window}')
    
    
    
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
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/AMV/timeseries_{data_type}_{window}')
    
    