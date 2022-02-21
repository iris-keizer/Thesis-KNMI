"""
File containing the Python functions to be able to compare different regression results
by performing a spectral analysis


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
comparison.ipynb
best_models_selection.ipynb

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
    
    
    
    
    
    
