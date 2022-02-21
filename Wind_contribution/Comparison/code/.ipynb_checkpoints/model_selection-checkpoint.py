"""
File containing the Python functions to be able to compare different regression results
by performing a spectral analysis and kolmogorov smirnov test

The aim is to select a subset of models that perform best compared to the observations


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
best_models_selection.ipynb

"""

# Import necessary packages
from scipy.stats import ks_2samp
from scipy.signal import detrend
from mtspec.multitaper import mtspec
from sklearn.metrics import auc

import math
import numpy as np
import xarray as xr
import pandas as pd
import statsmodels.api as sm
import statsmodels as sm
import matplotlib.pyplot as plt


period_min = 30
period_max = 110
window = 31 # Smoothing window for lowpass filter
lowess = sm.nonparametric.smoothers_lowess.lowess

models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0',
       'CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1',
       'CanESM5', 'CanESM5-CanOE', 'EC-Earth3', 'EC-Earth3-Veg',
       'EC-Earth3-Veg-LR', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL',
       'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
       'MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
       'NESM3', 'UKESM1-0-LL']




labels_windmodel = ['NearestPoint', 'Timmerman', 'Dangendorf']
    

    
    
    

def detrend_dim(da, dim, deg=1): 
    """
    Function that detrends the data from a dataarray along a single dimension
    deg=1 for linear fit
    
    """
    
    p = da.polyfit(dim=dim, deg=deg)
    coord = da[dim] - da[dim].values[0]
    trend = coord*p.polyfit_coefficients.sel(degree=1)
    return da - trend



    
    
def import_data_model_selection():
    
    
    # Import 20CR observations
    
    
    # Define path
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Regression results/fullperiod/'
    
    # Import the files
    np = pd.read_csv(path+f'timeseries_NearestPoint_20cr.csv', header = [0,1,2])
    tim = pd.read_csv(path+f'timeseries_Timmerman_20cr.csv', header = [0,1,2])
    dang = pd.read_csv(path+f'timeseries_Dangendorf_20cr.csv', header = [0,1,2])
    
    # Set index
    np = np.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
    tim = tim.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
    dang = dang.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
        
    # Set index name
    np.index.names = ['time']
    tim.index.names = ['time']
    dang.index.names = ['time']
            
            
    # Drop extra row
    np = np.droplevel(axis=1, level=2)
    tim = tim.droplevel(axis=1, level=2)
    dang = dang.droplevel(axis=1, level=2)
            
    
    # Create one dataframe only containing 'Average' station and wind contribution to SLH
    # whereof the data is detrended
    df = pd.DataFrame({'time': np.index.values, 
                       'NearestPoint': detrend(np['Average', 'wind total']),
                       'Timmerman': detrend(tim['Average', 'wind total']), 
                       'Dangendorf': detrend(dang['Average', 'wind total'])})


    detrended_timeseries_20cr = df.set_index('time')
    
    
    # Import CMIP6
    
    # Define path
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/fullperiod/'


    # Import the files
    np = xr.open_dataset(path+f'timeseries_NearestPoint_historical.nc')
    tim = xr.open_dataset(path+f'timeseries_Timmerman_historical.nc')
    dang = xr.open_dataset(path+f'timeseries_Dangendorf_historical.nc')
        
    # Select data and create dataframe
    np = np.wind_total.sel(station='Average', drop = True)
    tim = tim.wind_total.sel(station='Average', drop = True)
    dang = dang.wind_total.sel(station='Average', drop = True)

    # Detrend data
    np = detrend_dim(np, 'time')
    tim = detrend_dim(tim, 'time')
    dang = detrend_dim(dang, 'time')
    
    # Create dataframe
    np = np.to_pandas().T
    tim = tim.to_pandas().T
    dang = dang.to_pandas().T
    
    detrended_timeseries_cmip6 = pd.concat([np, tim, dang], axis = 1,  keys = ['NearestPoint', 'Timmerman', 'Dangendorf'])

    # Create data of equal time span
    detrended_timeseries_20cr = detrended_timeseries_20cr[
        detrended_timeseries_20cr.index.isin(detrended_timeseries_cmip6.index.values)]
    
    detrended_timeseries_cmip6 = detrended_timeseries_cmip6[
        detrended_timeseries_cmip6.index.isin(detrended_timeseries_20cr.index.values)]
    
    
    return detrended_timeseries_20cr, detrended_timeseries_cmip6



def import_data_model_selection2():
    # No detrending
    
    # Import 20CR observations
    
    
    # Define path
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/observations/Regression results/fullperiod/'
    
    # Import the files
    np = pd.read_csv(path+f'timeseries_NearestPoint_20cr.csv', header = [0,1,2])
    tim = pd.read_csv(path+f'timeseries_Timmerman_20cr.csv', header = [0,1,2])
    dang = pd.read_csv(path+f'timeseries_Dangendorf_20cr.csv', header = [0,1,2])
    
    # Set index
    np = np.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
    tim = tim.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
    dang = dang.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'time'))
        
    # Set index name
    np.index.names = ['time']
    tim.index.names = ['time']
    dang.index.names = ['time']
            
            
    # Drop extra row
    np = np.droplevel(axis=1, level=2)
    tim = tim.droplevel(axis=1, level=2)
    dang = dang.droplevel(axis=1, level=2)
            
    
    # Create one dataframe only containing 'Average' station and wind contribution to SLH
    # whereof the data is detrended
    df = pd.DataFrame({'time': np.index.values, 
                       'NearestPoint': np['Average', 'wind total'],
                       'Timmerman': tim['Average', 'wind total'], 
                       'Dangendorf': dang['Average', 'wind total']})


    timeseries_20cr = df.set_index('time')
    
    
    # Import CMIP6
    
    # Define path
    path = f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/fullperiod/'


    # Import the files
    np = xr.open_dataset(path+f'timeseries_NearestPoint_historical.nc')
    tim = xr.open_dataset(path+f'timeseries_Timmerman_historical.nc')
    dang = xr.open_dataset(path+f'timeseries_Dangendorf_historical.nc')
        
    # Select data and create dataframe
    np = np.wind_total.sel(station='Average', drop = True)
    tim = tim.wind_total.sel(station='Average', drop = True)
    dang = dang.wind_total.sel(station='Average', drop = True)

    
    # Create dataframe
    np = np.to_pandas().T
    tim = tim.to_pandas().T
    dang = dang.to_pandas().T
    
    timeseries_cmip6 = pd.concat([np, tim, dang], axis = 1,  keys = ['NearestPoint', 'Timmerman', 'Dangendorf'])

    # Create data of equal time span
    timeseries_20cr = timeseries_20cr[timeseries_20cr.index.isin(timeseries_cmip6.index.values)]
    
    timeseries_cmip6 = timeseries_cmip6[timeseries_cmip6.index.isin(timeseries_20cr.index.values)]
    
    
    return timeseries_20cr, timeseries_cmip6


    
    
    
def ks_test_df(timeseries_20cr, timeseries_cmip6):
    """
    Function to perform the kolmogorov smirnov test between the observations and each cmip6 model for all wind models
    
    """
 
    df_D = pd.DataFrame({'model':models, 'NearestPoint':'', 'Timmerman':'', 'Dangendorf':''})
    df_D = df_D.set_index('model')
    df_pval = df_D.copy()

    for model in models:

        for wm in labels_windmodel:

            D_ks, p_val = ks_2samp(timeseries_20cr[wm].values, timeseries_cmip6[wm, model].values)

            df_D[wm][model] = D_ks
            df_pval[wm][model] = p_val

    df_D['Average'] = df_D.mean(axis=1)
    df_pval['Average'] = df_pval.mean(axis=1)

    df = pd.concat([df_D, df_pval], axis=1, keys = ['D$_{ks}$', 'p-value'])

    df = df.sort_values(('D$_{ks}$', 'Average'))
    
    return df




def plot_comp_20cr_cmip6(data_20cr, data_cmip6, time_bandwidth = 1.4, show_spec = True, n_cmip6 = 1):
    """
    Function to obtain the variance and partial variance of 20cr and cmip6 data over the overlapping time period.
    
    show_spec should be True when also the three spectra of the 20cr data should be plotted and of the n_cmip6 best performing 
    cmip6 models.
    
    
    """
    
    # Calculate total and partial variance of observations
    df_obs = pd.DataFrame({'variable':['total variance', 'total auc', 'partial auc'], 'NearestPoint':'',
                           'Timmerman':'','Dangendorf':''})
    df_obs = df_obs.set_index('variable')
    
    for label in labels_windmodel:
        df_obs[label]['total variance'] = data_20cr[label].var() # Calculate variance
        
        spec, freq = mtspec(data_20cr[label], 1.0, time_bandwidth) # Obtain spectra
        
        df_obs[label]['total auc'] = auc(freq, spec) # Calculate total area under curve
        
        i_start = next(i for i,v in enumerate(freq) if v>=1/period_max)
        i_end = next(i for i,v in enumerate(freq) if v>1/period_min)
        
        df_obs[label]['partial auc'] = auc(freq[i_start:i_end], spec[i_start:i_end]) # Calculate partial variance

    
    
    # Calculate total and partial variance of models
    
    dfs = []

    for label in labels_windmodel:

        df = pd.DataFrame({'variable':['total variance', 'total auc', 'partial auc']})
        df = df.set_index('variable')

        for model in models:
            df[model] = ''

            df[model]['total variance'] = data_cmip6[label, model].var() # Calculate variance

            spec, freq = mtspec(data_cmip6[label, model], 1.0, time_bandwidth) # Obtain spectra

            df[model]['total auc'] = auc(freq, spec) # Calculate total area under curve

            i_start = next(i for i,v in enumerate(freq) if v>=1/period_max)
            i_end = next(i for i,v in enumerate(freq) if v>1/period_min)

            df[model]['partial auc'] = auc(freq[i_start:i_end], spec[i_start:i_end]) # Calculate partial variance

        dfs.append(df)

    df_cmip6 = pd.concat(dfs, axis=1, keys = labels_windmodel)

    
    # Obtain a dataframe of the best models
    best_models = get_best_models(df_obs, df_cmip6)
    
    
    # Plot the total variance against the partial variance
    markers = ['v', '^', '<', '>']
    
    n_cols = 1
    n_rows = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4, 8) )


    for i in range(n_rows):
        ax = axs[i]
        ax.scatter(df_obs[labels_windmodel[i]]['partial auc'], df_obs[labels_windmodel[i]]['total auc'], marker = 'x', s=70, c='k')
        
        for j, model in enumerate(best_models.index.values):
            ax.scatter(df_cmip6[labels_windmodel[i], model]['partial auc'], df_cmip6[labels_windmodel[i], model]['total auc'],
                  marker = markers[int((3.6*j)/36)], s=25, alpha = .8)
            
        ax.set_ylim(-0.2,13.1)
        ax.set_xlim(-0.2,1.2)
        ax.set_ylabel('Total auc [cm$^2$]')
        ax.set_title(labels_windmodel[i])
        if i == n_rows-1:
            ax.set_xlabel('Partial auc [cm$^2$]')
        plt.tight_layout()
        
    labels = ['20cr'] + list(best_models.index.values)
    plt.legend(labels = labels,ncol=2, bbox_to_anchor=(2.4, 2.8))
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/model selection/tot_part_auc')
    
    
    # Plot the 20cr spectra and for n_cmip6 model spectra
    
    n_cols = 3
    n_rows = 1 + n_cmip6
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3*n_rows))
    
    for i in range(n_rows):

        for j in range(n_cols):
            
            ax = axs[i,j]
            
            if i == 0:
                spec, freq, conf_int, f_stat, n_freedom = mtspec(data_20cr[labels_windmodel[j]], 1.0, time_bandwidth, 
                                                                 statistics = True)

                var = round(df_obs[labels_windmodel[j]]['total variance'],2)
                tot_auc = round(df_obs[labels_windmodel[j]]['total auc'],2)
                part_auc = round(df_obs[labels_windmodel[j]]['partial auc'],2)
                
                ax.set_title(f'20cr - {labels_windmodel[j]} - ({data_20cr.index[0]}-{data_20cr.index[-1]}) \n total var={var} - '+
                             f'total auc={tot_auc} - partial auc={part_auc} \n time_bandwidth = {time_bandwidth}')
            
            else:
                model = best_models.index.values[i-1]
                spec, freq, conf_int, f_stat, n_freedom = mtspec(data_cmip6[labels_windmodel[j], model], 1.0, time_bandwidth, 
                                                                 statistics = True)

                var = round(df_cmip6[labels_windmodel[j], model]['total variance'],2)
                tot_auc = round(df_cmip6[labels_windmodel[j], model]['total auc'],2)
                part_auc = round(df_cmip6[labels_windmodel[j], model]['partial auc'],2)
                
                ax.set_title(f'{model} - {labels_windmodel[j]} - ({data_cmip6.index[0]}-{data_cmip6.index[-1]}) \n total var={var} - '+
                             f'total auc={tot_auc} - partial auc={part_auc} \n time_bandwidth = {time_bandwidth}')
                
                
            ax.plot(1/freq[1:], spec[1:])
            ax.fill_between(1/freq[1:], conf_int[1:, 0], conf_int[1:, 1], color="tab:red", alpha=0.2)
            ax.fill_between(1/freq[i_start:i_end], spec[i_start:i_end], alpha = 0.2)
                
            ax.set_xlim(1,100)
            ax.set_xscale('log')
            ax.set_xticks([1,10,100])
            ax.set_xticklabels(['1','10','100'])
            
            if i == n_rows-1:
                ax.set_xlabel('Period [y/cycle]')


            if j == 0:
                ax.set_ylabel('Power spectral density')
    

    plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/model selection/spectra_20cr_cmip6_{n_cmip6}')
    
    
    return best_models



def get_best_models(df_obs, df_cmip6):
    """
    Function that obtains a dataframe of the euclidian distance between observation and model for each wind 
    model and averaged, sorted by ascending order over the average column
    
    """
    df = pd.DataFrame({'model':models, labels_windmodel[0]:'', labels_windmodel[1]:'', labels_windmodel[2]:''})
    df = df.set_index('model')
    
    for model in models:
        for label in labels_windmodel:
            df[label][model] = np.sqrt((df_cmip6[label, model]['total auc']-df_obs[label]['total auc'])**2 + 
                                       (df_cmip6[label, model]['partial auc']-df_obs[label]['partial auc'])**2) # Calculate Euclidian distance
    
    df['Average'] = df.mean(axis=1)
    
    return df.sort_values('Average')





def select_models(df_spec, df_ks, spec_thres = 2.5, ks_thres = 0.25):
    """
    Function that selects the model meeting the criteria of having smaller distances than spec_thres and ks_thres 
    for all the different wind models
    """
    
    df_ks = df_ks['D$_{ks}$']
    
    # Select models that meet the conditions for the euclidean distance
    select_spec1 = df_spec.loc[df_spec['NearestPoint'] < spec_thres].index
    select_spec2 = df_spec.loc[df_spec['Timmerman'] < spec_thres].index
    select_spec3 = df_spec.loc[df_spec['Dangendorf'] < spec_thres].index
    selection_spec = np.intersect1d(np.intersect1d(select_spec1,select_spec2),select_spec3)

    # Select models that meet the conditions for the KS test
    select_ks1 = df_ks.loc[df_ks['NearestPoint'] < ks_thres].index
    select_ks2 = df_ks.loc[df_ks['Timmerman'] < ks_thres].index
    select_ks3 = df_ks.loc[df_ks['Dangendorf'] < ks_thres].index
    selection_ks = np.intersect1d(np.intersect1d(select_ks1,select_ks2),select_ks3)
    
    best_models = np.intersect1d(selection_spec, selection_ks)
    
    with open('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Comparison results/bestmodels.txt', 'w') as filehandle:
        for listitem in best_models:
            filehandle.write('%s\n' % listitem)
    
    return best_models



def plot_best_models(best_models, timeseries_cmip6):
    """
    Function to plot the smoothed time series of the models selected as best performing
    """
    
    timeseries_cmip6 = timeseries_cmip6.swaplevel(0,1, axis=1)
    timeseries_cmip6 = timeseries_cmip6[list(best_models)]
    
    n_col = 4
    n_row = math.ceil(best_models.size / n_col)
    n_delete = best_models.size % n_col
    
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(20, 10))


    for i in range(n_row):

        for j in range(n_col):
            
            
            ax = axs[i,j]

            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
            
            else:
                for wm in labels_windmodel:
                    data = timeseries_cmip6[best_models[n_row*i+j], wm]
                    lws = lowess(data.values, 
                                  data.index.values,
                                  get_frac(window, data, dtype='DataFrame'), 
                                  return_sorted = False)
                    ax.plot(data.index, lws)
                ax.set_title(best_models[n_row*i+j])
                ax.set_ylim(-1.5,1.5)
                if i == n_row-1:
                    ax.set_xlabel('time [y]')
                if j == 0:
                    ax.set_ylabel('Wind contribution to SLH [cm]')
                plt.tight_layout()
                
            if i == 0 and j == 0:
                ax.legend(labels = labels_windmodel, loc='upper right')
                
    plt.savefig('/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/model selection/ts_best_models')




def get_frac(window, data, dtype='DataFrame'):
    """
    Function to obtain the fraction of data used for lowess smoothing
    """
    if dtype == 'DataFrame':
        frac = window / (data.index[-1]-data.index[0])
    elif dtype == 'DataSet':
        frac = window / (data.time.values[-1] - data.time.values[0])
    else: print('Datatype unknown')
    
    return frac

