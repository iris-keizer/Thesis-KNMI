"""
File containing the Python functions to be able to compare different regression results
by performing a spectral analysis


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
comparison.ipynb

"""

# Import necessary packages
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from mtspec.multitaper import mtspec
from sklearn.metrics import auc




period_min = 20
period_max = 100



models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'BCC-ESM1',
       'CAMS-CSM1-0', 'CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2',
       'CNRM-CM6-1', 'CNRM-ESM2-1', 'CanESM5', 'CanESM5-CanOE',
       'EC-Earth3', 'EC-Earth3-AerChem', 'EC-Earth3-CC', 'EC-Earth3-Veg',
       'EC-Earth3-Veg-LR', 'FGOALS-f3-L', 'GFDL-CM4', 'GFDL-ESM4',
       'GISS-E2-1-G', 'GISS-E2-1-H', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
       'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'MIROC6',
       'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
       'NESM3', 'NorCPM1', 'UKESM1-0-LL']
labels_windmodel = ['NearestPoint', 'Timmerman', 'Dangendorf']
    




def plot_obs_full_spectra(data, time_bandwidth = 1.4,  full_time_range = True, show_data = False):
    """
    Function to plot the multitaper spectra of the observational wind contribution to sea level rise over the whole available time period. 
    
    data should be a list containing the era5 and 20cr average wind contribution 
    
    full_time_range should be True if for era5 and 20cr just the whole time period is used and False when only the overlapping period should be considered.
    
    show_data should be True if the data whereof the spectra is obtained should be plotted either
    """
    labels_data = ['era5', '20cr']

    
    if full_time_range == False:
        
        # Create dataframes of equal time span
        data[0] = data[0][data[0].index.isin(data[1].index)]
        data[1] = data[1][data[1].index.isin(data[0].index)]
    
    
    if show_data == True:
        data[0].plot(figsize = (9,2), xlim=(data[1].index[0],data[0].index[-1]))
        data[1].plot(figsize = (9,2), xlim=(data[1].index[0],data[0].index[-1]), legend=False)
    
    n_cols = 3
    n_rows = 2

        
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3*n_rows))

    dfs = []
    for i in range(n_rows):

        for j in range(n_cols):

            ax = axs[i,j]
            spec, freq, conf_int, f_stat, n_freedom = mtspec(data[i][labels_windmodel[j]], 1.0, time_bandwidth, statistics = True)
            var = round(data[i][labels_windmodel[j]].var(),2)
            total_auc = round(auc(freq, spec),2)

            i_start = next(i for i,v in enumerate(freq) if v>=1/period_max)
            i_end = next(i for i,v in enumerate(freq) if v>1/period_min)

            partial_auc = round(auc(freq[i_start:i_end], spec[i_start:i_end]),2)




            ax.plot(1/freq[1:], spec[1:])
            ax.fill_between(1/freq[1:], conf_int[1:, 0], conf_int[1:, 1], color="tab:red", alpha=0.2)
            ax.fill_between(1/freq[i_start:i_end], spec[i_start:i_end], alpha = 0.2)
            ax.set_xlim(1,100)
            ax.set_xscale('log')
            ax.set_xticks([1,10,100])
            ax.set_xticklabels(['1','10','100'])
            ax.set_title(f'{labels_data[i]} - {labels_windmodel[j]} - ({data[i].index[0]}-{data[i].index[-1]}) \n total var={var} - total auc={total_auc} - partial auc={partial_auc} '+
                         f'\n time_bandwidth = {time_bandwidth}')
            if i == n_rows-1:
                ax.set_xlabel('Period [y/cycle]')


            if j == 0:
                df = pd.DataFrame({'variable':['total variance', 'total auc', 'partial auc']})
                df = df.set_index('variable')
                ax.set_ylabel('Power spectral density')
            df[labels_windmodel[j]] = [var, total_auc, partial_auc]



        dfs.append(df)       



    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/obs_spec_mtspec_{data[i].index[0]}_{data[i].index[-1]}')

    df_spec_obs = pd.concat(dfs, axis=1, keys = ['era5', '20cr']) 
    
    
    
    
    
    
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
    Function that obtains a dataframe of the euclidian distance between observation and model for each wind model and averaged,
    sorted by ascending order over the average column
    
    """
    df = pd.DataFrame({'model':models, labels_windmodel[0]:'', labels_windmodel[1]:'', labels_windmodel[2]:''})
    df = df.set_index('model')
    
    for model in models:
        for label in labels_windmodel:
            df[label][model] = np.sqrt((df_cmip6[label, model]['total auc']-df_obs[label]['total auc'])**2 + 
                                       (df_cmip6[label, model]['partial auc']-df_obs[label]['partial auc'])**2) # Calculate Euclidian distance
    
    df['Average'] = df.mean(axis=1)
    
    return df.sort_values('Average')




