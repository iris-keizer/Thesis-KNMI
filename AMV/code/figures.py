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




def plot_df_timeseries(data_df, smoothed = False, window = 21, ylabel = 'No label given', title = 'No title given'):
    '''
    Function to make a simple plot of a dataframe consisting of time series
    
    As an option, a lowess smoothing can be applied
    '''
    plt.figure(figsize = (9,3))
    
    for column in data_df:
        if smoothed == True:
            frac = window/data_df[column].values.size
            ts_lowess = lowess(data_df[column].values, data_df.index.values, frac, return_sorted=False)
            plt.plot(data_df.index, ts_lowess, label = column)
            
            plt.title(f'{title}  \n lowess filtered, window = {window}')
            
        else:
            plt.plot(data_df.index, data_df[column].values, label = column)
            
            plt.title(f'{title}')
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('time [y]')
    plt.ylabel(f'{ylabel}')
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    
    
def plot_era5_20cr_timeseries(data_era5, data_20cr, smoothed = False, window = 21):
    '''
    Function to make a plot of both era5 and 20cr atmospheric contribution time series
    '''
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    plt.figure(figsize = (9,3))
    
    for i, column in enumerate(data_era5.columns):
        if smoothed == True:
            frac = window/data_20cr[column].values.size
            cr20_lowess = lowess(data_20cr[column].values, data_20cr.index.values, frac, return_sorted=False)
            plt.plot(data_20cr.index, cr20_lowess, color = colors[i], label = column + ' - 20CR', alpha = 0.6)
            
            
            frac = window/data_era5[column].values.size
            era5_lowess = lowess(data_era5[column].values, data_era5.index.values, frac, return_sorted=False)
            plt.plot(data_era5.index, era5_lowess, color = colors[i], label = column + ' - ERA5')
            
            plt.title(f'Atmospheric contribution to mean sea-level \n lowess filtered, window = {window}')
        else:
            plt.plot(data_20cr.index, data_20cr[column], color = colors[i], label = column + ' - 20CR', alpha = 0.6)
            plt.plot(data_era5.index, data_era5[column], color = colors[i], label = column + ' - ERA5')
            
            plt.title(f'Atmospheric contribution to mean sea-level')
            
            
    plt.xlim(1835, 2021)
    plt.legend(bbox_to_anchor=[1.04, 0.75])
    plt.xlabel('time [y]')
    plt.ylabel('sea-level contribution [cm]')
    plt.axhline(color='k', linestyle='--', linewidth = 0.9)  
    plt.tight_layout()
    
    
    
    
def plot_result(results_era5, results_20cr, var, ylabel):
    '''
    Function to plot a result of the regression between atmospheric contribution to sea-level and the AMV
    
    '''
    y_min = -0.01
    y_max = 0.15
    
    fix, axs = plt.subplots(3, 2, figsize = (15,8))
    
    for i, wm in enumerate(wind_labels):
        
        ax = axs[i,0]
        data_era5 = results_era5.swaplevel(0,1, axis=1)[wm]
        
        for column in AMV_names:
            dataT = data_era5[column].T
            ax.scatter(dataT.index, dataT[var].values)
        
        if i == 2:
            ax.set_xlabel('lag [-]')
            
        ax.set_ylabel(ylabel)
        #ax.set_ylim(y_min, y_max)
        ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
        
        if i == 0:
            ax.set_title('ERA5')
        
        ax = axs[i,1]
        data_20cr = results_20cr.swaplevel(0,1, axis=1)[wm]
        
        for column in AMV_names:
            dataT = data_20cr[column].T
            ax.scatter(dataT.index, dataT[var].values)
        #ax.set_ylim(y_min, y_max)
        ax.axhline(color='k', linestyle='--', linewidth = 0.9)  
        
        if i == 2:
            ax.set_xlabel('lag [-]')
        
        if i == 0:
            ax.legend(labels = AMV_names)
            ax.set_title('20CRv3')