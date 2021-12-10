"""
File containing the Python functions to make plots of the projections.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
Projection.ipynb

"""

# Import necessary packages
import math

import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt

lowess = sm.nonparametric.lowess

wind_labels = ['NearestPoint', 'Timmerman', 'Dangendorf']






def plot_zos_per_scenario(scenarios, labels, smoothed = False, window = 21,
                         hist_start = 1950):
    '''
    Function to plot the zos data for all models per scenario. 
    
    As an option, a smoothing can be applied.
    
    scenarios should be a list of the datasets for the different scenarios
    
    smoothed should be True if the smoothing should be applied
    
    hist_start defines where the x-axis starts
    
    '''
    
    n_col = 2
    n_row = math.ceil(len(scenarios) / n_col)
    n_delete = len(scenarios) % n_col
    
    alpha = 0.7
    
    # Find scenario with most models
    index = 0
    longest = scenarios[0].columns.size
    for i, scenario in enumerate(scenarios[1:]):
        if scenario.columns.size > longest:
            index = i
    models = scenarios[index].columns.values
    
    colors = ['b', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'g', 'k']
    
    if smoothed == True:
        y_min = -8
        y_max = 35
    else:
        y_min = -15
        y_max = 37
            
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            if i == n_row-1 and j in range(n_delete, n_col):
                ax.legend(bbox_to_anchor=[1.20, 0.8], ncol=2, prop={'size': 12})
            
            
            if n_row > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
            
            
            else:
                scenario = scenarios[n_col*i+j]
                for k, model in enumerate(models):
                    if model in scenario:
                        if smoothed == True:
                            frac = window/scenario[model].values.size
                            scenario_lowess = lowess(scenario[model].values, scenario.index.values, frac, return_sorted=False)
                            ax.plot(scenario.index.values, scenario_lowess, color = colors[k], label = model)


                        else:
                            ax.plot(scenario.index.values, scenario[model].values, color = colors[k], label = model)

                ax.set_title(f'historical and ' + labels[n_col*i+j])
                if j == 0:
                    ax.set_ylabel(f'zos [cm]')
                ax.set_xlim(hist_start, 2101)
                if i == n_row - 1:
                    ax.set_xlabel('time [y]')
                ax.set_ylim(y_min, y_max)
                ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                



    if smoothed == False:         
        plt.savefig(
            f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/zos_per_scenario_{hist_start}')

    else:
        plt.savefig(
            f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/zos_per_scenario_smoothed_{hist_start}')

     
    
    
    
        

def plot_zos_med_percentiles_per_scenarios(scenarios, labels, lower_bound = 0.05, upper_bound = 0.95, ra = 5, hist_start = 1950):
    '''
    Function to make a plot of zos of the median and upper and lower bound of the models for each scenario.
    Define the percentiles by setting the lower_bound and upper_bound and 
    what running average is applied by setting ra
    
    '''
    
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] 
    
    
    n_col = 2
    n_row = math.ceil(len(scenarios) / n_col)
    n_delete = len(scenarios) % n_col
    
    y_min = -10
    y_max = 35
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            if n_row > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
            
              
                
            else:
                scenario = scenarios[n_col*i+j]
                
                med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()

                ax.fill_between(ub.index, ub, lb, color = colors[n_col*i+j], alpha=0.3, 
                                label=f'{int(lower_bound*100)}-{int(upper_bound*100)} percentiles')
                ax.plot(med, color = colors[n_col*i+j], label=f'median')
                
                if i == n_row - 1:
                    ax.set_xlabel('time [y]')
                if j == 0:
                    ax.set_ylabel('zos [cm]')
                ax.set_title(f'historical and {labels[n_col*i+j]}')
                ax.set_xlim(hist_start, 2101)
                ax.set_ylim(y_min, y_max)
                ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                ax.legend(loc='upper left')
    #plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/zos_median_percentiles_per scenario_{hist_start}')

    
    
    
    
    
def plot_wind_per_scenario(scenarios, labels, direction = 'Zonal', smoothed = False, window = 21, 
                           hist_start = 1950, wind_model = 'NearestPoint'):
    '''
    Function to plot the wind data per scenario by calling the functions make_wind_dfs and plot_wind
    '''
    
    if wind_model == 'NearestPoint' or wind_model == 'Timmerman':
        u2, v2 = make_wind_dfs(scenarios, labels, wind_model)
    
        # Plot zonal wind stress
        plot_wind_projections(u2, labels, smoothed = smoothed, window = window, 
                           hist_start = hist_start, wind_model = wind_model)
        
        # Plot meridional wind stress
        plot_wind_projections(v2, labels, direction = 'Meridional', smoothed = smoothed, window = window, 
                           hist_start = hist_start, wind_model = wind_model)
        
    elif wind_model == 'Dangendorf':
        neg, pos = make_wind_dfs(scenarios, labels, wind_model)
        
        # Plot negative proxy
        plot_wind_projections(neg, labels, direction = 'Negative', smoothed = smoothed, window = window, 
                           hist_start = hist_start, wind_model = wind_model)
        
        # Plot positive proxy
        plot_wind_projections(pos, labels, direction = 'Positive', smoothed = smoothed, window = window, 
                           hist_start = hist_start, wind_model = wind_model)
        
        
    
    
    
def make_wind_dfs(scenarios, labels, wind_model):
    
    '''
    Function to pre-process the data (create list of dataframes) such that the wind per scenario over the 
    historical period and with median and percentiles can be plotted 
    
    '''
    if wind_model == 'NearestPoint':
        
        # Create dataframes
        u2 = []
        v2 = []
        for scenario in scenarios:
            u2.append(scenario.u2.to_pandas().T.dropna(axis=1))
            v2.append(scenario.v2.to_pandas().T.dropna(axis=1))
    
        return u2, v2
    
    
    elif wind_model == 'Timmerman':
        
        # Create dataframes
        u2 = []
        v2 = []
        for scenario in scenarios:
            scenario = scenario.mean(dim = 'tim_region')
            u2.append(scenario.u2.to_pandas().T.dropna(axis=1))
            v2.append(scenario.v2.to_pandas().T.dropna(axis=1))
    
        return u2, v2
        
        
    elif wind_model == 'Dangendorf':
        
        # Create dataframes
        neg = []
        pos = []
        for scenario in scenarios:
            neg.append(scenario['Negative corr region'].to_pandas().T.dropna(axis=1))
            pos.append(scenario['Positive corr region'].to_pandas().T.dropna(axis=1))
    
        return neg, pos
        
    
    
    
    
    
    
    
def plot_wind_projections(scenarios, labels, direction = 'Zonal', smoothed = False, window = 21, 
                           hist_start = 1950, wind_model = 'NearestPoint'):
    '''
    Function to plot the wind data for all models per scenario. 
    
    As an option, a smoothing can be applied.
    
    scenarios should be a list of the dataframes for the different scenarios
    
    smoothed should be True if the smoothing should be applied
    
    hist_start defines where the x-axis starts
    
    wind_model defines which wind_model data is given
    
    '''
    
    
    n_col = 2
    n_row = math.ceil(len(scenarios) / n_col)
    n_delete = len(scenarios) % n_col
    
    alpha = 0.7
    
    # Find scenario with most models
    index = 0
    longest = scenarios[0].columns.size
    for i, scenario in enumerate(scenarios[1:]):
        if scenario.columns.size > longest:
            index = i
    models = scenarios[index].columns.values
    
    colors = ['b', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'g', 'k']
    
    if smoothed == True:
        if direction == 'Zonal':
            y_min = -1
            y_max = 30
        elif direction == 'Meridional':
            y_min = -1
            y_max = 8
        elif direction == 'Negative':
            y_min = 99200
            y_max = 100000
        elif direction == 'Positive':
            y_min = 97300
            y_max = 98500
    else:
        if direction == 'Zonal':
            y_min = -1
            y_max = 46
        elif direction == 'Meridional':
            y_min = -1
            y_max = 18
        elif direction == 'Negative':
            y_min = 99000
            y_max = 100200
        elif direction == 'Positive':
            y_min = 97200
            y_max = 98000
            
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            if i == n_row-1 and j in range(n_delete, n_col):
                ax.legend(bbox_to_anchor=[1.20, 0.8], ncol=2, prop={'size': 12})
            
            if n_row > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
                
                
            else:
                scenario = scenarios[n_col*i+j]
                for k, model in enumerate(models):
                    if model in scenario:
                        if smoothed == True:
                            frac = window/scenario[model].values.size
                            scenario_lowess = lowess(scenario[model].values, scenario.index.values, frac, return_sorted=False)
                            ax.plot(scenario.index.values, scenario_lowess, color = colors[k], label = model)
                            
                            
                        else:
                            ax.plot(scenario.index.values, scenario[model].values, color = colors[k], label = model)
                            
                ax.set_title(f'historical and ' + labels[n_col*i+j])
                if not wind_model == 'Dangendorf':
                    ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                if j == 0:
                    if wind_model == 'Dangendorf':
                        ax.set_ylabel(f'{direction} pressure proxy [Pa]')
                    else:
                        ax.set_ylabel(f'{direction} wind stress [m$^2$/s$^2$]')
                ax.set_xlim(hist_start, 2101)
                if i == n_row - 1:
                    ax.set_xlabel('time [y]')
                ax.set_ylim(y_min, y_max)
                plt.tight_layout()
            
            
    
    if smoothed == False:         
        plt.savefig(
            f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_wind_per_scenario_{hist_start}')

    else:
        plt.savefig(
            f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_{direction.lower()}_wind_per_scenario_smoothed_{hist_start}')



        
        
        
        
def plot_wind_med_percentiles_per_scenario(scenarios, labels, direction = 'Zonal', 
                                           lower_bound = 0.05, upper_bound = 0.95, ra = 5, hist_start = 1950,
                                           wind_model = 'NearestPoint'):
    '''
    Function to plot the wind data per scenario by calling the functions make_wind_dfs and plot_wind
    '''
    
    if wind_model == 'NearestPoint' or wind_model == 'Timmerman':
        u2, v2 = make_wind_dfs(scenarios, labels, wind_model)
    
        # Plot zonal wind stress
        plot_wind_med_percentiles(u2, labels, lower_bound = lower_bound, upper_bound = upper_bound, ra = ra, hist_start = hist_start,
                                 wind_model = wind_model)
        
        # Plot meridional wind stress
        plot_wind_med_percentiles(v2, labels, direction = 'Meridional', lower_bound = lower_bound, upper_bound = upper_bound, 
                                  ra = ra, hist_start = hist_start, wind_model = wind_model)
        
    elif wind_model == 'Dangendorf':
        neg, pos = make_wind_dfs(scenarios, labels, wind_model)
        
        # Plot negative proxy
        plot_wind_med_percentiles(neg, labels, direction = 'Negative', lower_bound = lower_bound, upper_bound = upper_bound, 
                                  ra = ra, hist_start = hist_start, wind_model = wind_model)
        
        # Plot positive proxy
        plot_wind_med_percentiles(pos, labels, direction = 'Positive', lower_bound = lower_bound, upper_bound = upper_bound, 
                                  ra = ra, hist_start = hist_start, wind_model = wind_model)
        
        
        
        
        
        


def plot_wind_med_percentiles(scenarios, labels, lower_bound = 0.05, upper_bound = 0.95, ra = 5, hist_start = 1950, 
                              wind_model = 'NearestPoint', direction = 'Zonal'):
    '''
    Function to make a plot of wind stress of the median and upper and lower bound of the models for each scenario.
    Define the percentiles by setting the lower_bound and upper_bound and 
    what running average is applied by setting ra
    
    '''
    
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] 
    
    
    n_col = 2
    n_row = math.ceil(len(scenarios) / n_col)
    n_delete = len(scenarios) % n_col
    
    if direction == 'Zonal':
        y_min = -1
        y_max = 22
    elif direction == 'Meridional':
        y_min = -0.5
        y_max = 8
    
    elif direction == 'Negative':
        y_min = 99100
        y_max = 99900
    elif direction == 'Positive':
        y_min = 97900
        y_max = 97350
    
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            
            if n_row > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
                
            else:
                scenario = scenarios[n_col*i+j]
                
                med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()

                ax.fill_between(ub.index, ub, lb, color = colors[n_col*i+j], alpha=0.3, 
                                label=f'{int(lower_bound*100)}-{int(upper_bound*100)} percentiles')
                ax.plot(med, color = colors[n_col*i+j], label=f'median')
                
                if i == n_row - 1:
                    ax.set_xlabel('time [y]')
                if j == 0:
                    if wind_model == 'Dangendorf':
                        ax.set_ylabel(f'{direction} pressure proxy [Pa]')
                    else:
                        ax.set_ylabel(f'{direction} wind stress [m$^2$/s$^2$]')
                ax.set_title(f'historical and {labels[n_col*i+j]}')
                ax.set_xlim(hist_start, 2101)
                ax.set_ylim(y_min, y_max)
                ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                ax.legend(loc='upper left')
                #plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{direction.lower()}_wind_{wind_model}_median_percentiles_per scenario_{hist_start}')
    
        
        
        
        
        
        
        

def plot_projections_per_scenario(scenarios, labels, smoothed = False, window = 21, 
                                  hist_start = 1950, wind_model = 'NearestPoint', 
                                  ylabel = 'Wind contribution to sea level [cm]'):
    """
    Function to plot all models per scenario. 
    
    As an option, a smoothing can be applied.
    
    scenarios should be a list of the dataframes for the different scenarios
    
    smoothed should be True if the smoothing should be applied
    """
    
    n_col = 2
    n_row = math.ceil(len(scenarios) / n_col)
    n_delete = len(scenarios) % n_col
    
    alpha = 0.7
    
    # Find scenario with most models
    index = 0
    longest = scenarios[0].columns.size
    for i, scenario in enumerate(scenarios[1:]):
        if scenario.columns.size > longest:
            index = i
    models = scenarios[index].columns.values
    
    colors = ['g', 'r', 'b', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    if smoothed == True:
        y_min = -1.5
        y_max = 3
    else:
        y_min = -5
        y_max = 7.5
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            
            if i == n_row-1 and j in range(n_delete, n_col):
                ax.legend(bbox_to_anchor=[1.20, 0.8], ncol=2, prop={'size': 12})
            
             
            if n_row > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])  
                
            else:
                scenario = scenarios[n_col*i+j]
                
                
                for k, model in enumerate(models):
                    if model in scenario:
                        if smoothed == True:
                            frac = window/scenario[model].values.size
                            scenario_lowess = lowess(scenario[model].values, scenario.index.values, frac, return_sorted=False)
                            ax.plot(scenario.index.values, scenario_lowess, color = colors[k], label = model)
                            
                        else:
                            ax.plot(scenario.index.values, scenario[model].values, color = colors[k], label = model)
                            
                ax.set_title('historical and ' + labels[n_col*i+j])
                if j == 0:
                    ax.set_ylabel(ylabel)
                ax.set_xlim(hist_start, 2101)
                if i == n_row - 1:
                    ax.set_xlabel('time [y]')
                if wind_model != 'Dangendorf':
                    ax.set_ylim(y_min, y_max)
                ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                plt.tight_layout()
            
            
    
    if smoothed == False:         
        plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_projection_per_scenario_{hist_start}')

    else:
        plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_projection_per_scenario_smoothed_{hist_start}')


        
        
    
def plot_projections_per_scenario_all_wind_models(scenarios, labels, smoothed = False, window = 21, 
                                  hist_start = 1950):
    """
    Function to plot all models per scenario and for all wind models
    
    As an option, a smoothing can be applied.
    
    scenarios should be a list of the dataframes for the different scenarios
    
    wc_historical should be a list of the historical wind contribution for the different wind models
    
    smoothed should be True if the smoothing should be applied
    """
    
    
    
    n_col = 3
    n_row = len(scenarios[0])
    
    alpha = 0.7
    
    # Find scenario with most models
    index = 0
    longest = scenarios[0][0].columns.size
    for i, scenario in enumerate(scenarios[0][1:]):
        if scenario.columns.size > longest:
            index = i
    models = scenarios[0][index].columns.values
    
    colors = ['b', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'g', 'k']
    
    if smoothed == True:
        y_min = -1
        y_max = 3.2
    else:
        y_min = -5
        y_max = 8
    
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(15, 2.5*n_row))
    
    for i in range(n_row):

        for j in range(n_col):
            
            
            ax = axs[i,j]

            scenario = scenarios[j][i]


            for k, model in enumerate(models):
                if model in scenario:
                    if smoothed == True:
                        frac = window/scenario[model].values.size
                        scenario_lowess = lowess(scenario[model].values, scenario.index.values, frac, return_sorted=False)
                        ax.plot(scenario.index.values, scenario_lowess, label = model)

                    else:
                        ax.plot(scenario.index.values, scenario[model].values, label = model)
            insert = ''
            if smoothed == True:
                insert = f'\n lowess window = {window}'
            ax.set_title(f'{wind_labels[j]} - historical and {labels[i]}{insert}')
            if j == 0:
                ax.set_ylabel('Wind contribution to sea level [cm]')
            ax.set_xlim(hist_start, 2101)
            if i == n_row - 1:
                ax.set_xlabel('time [y]')
            ax.set_ylim(y_min, y_max)
            ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
            ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
            plt.tight_layout()


    fig.legend(labels = models, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=6)
                
    
    if smoothed == False:         
        plt.savefig(
            f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/projection_per_scenario_all_wm_{hist_start}')

    else:
        plt.savefig(
            f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/projection_per_scenario_all_wm_smoothed_{hist_start}')


       
        

def plot_med_percentiles_scenarios(scenarios, labels, lower_bound = 0.05, upper_bound = 0.95, ra = 5, 
                                   hist_start = 1950, wind_model = 'NearestPoint'):
    '''
    Function to make a plot of the median and upper and lower bound of the models for each scenario.
    Define the percentiles by setting the lower_bound and upper_bound and 
    what running average is applied by setting ra
    
    '''
    
    
    plt.figure(figsize = (10,4))
    
    
    for i, scenario in enumerate(scenarios):
        
        med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
        lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
        ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()

        plt.fill_between(ub.index, ub, lb, alpha=0.3, label=f'{labels[i]}, {int(lower_bound*100)}-{int(upper_bound*100)} percentiles')
        plt.plot(med, label=f'{labels[i]} median')
    
    
    plt.xlim(hist_start, 2101)
    plt.xlabel('time [y]')
    plt.ylabel('Wind contribution to sea level [cm]')
    plt.title(f'Compare wind contribution projections from scenarios \n'+
                  f'with running average of {ra} years')
    if wind_model != 'Dangendorf':
        plt.ylim(-2.5,3.5)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 0.7))
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_median_percentiles_{hist_start}')


    
def plot_med_percentiles_per_scenarios(scenarios, labels, lower_bound = 0.05, upper_bound = 0.95, ra = 5
                                       , hist_start = 1950, wind_model = 'NearestPoint'):
    '''
    Function to make a plot of the median and upper and lower bound of the models for each scenario.
    Define the percentiles by setting the lower_bound and upper_bound and 
    what running average is applied by setting ra
    
    '''
    
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] 
    
    
    n_col = 2
    n_row = math.ceil(len(scenarios) / n_col)
    n_delete = len(scenarios) % n_col
    
    y_min = -2.2
    y_max = 3
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(18, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            
            if n_row > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            
            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
                
                
            else:
                scenario = scenarios[n_col*i+j]
                
                
                med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()

                ax.fill_between(ub.index, ub, lb, color = colors[n_col*i+j], alpha=0.3, 
                                label=f'{int(lower_bound*100)}-{int(upper_bound*100)} percentiles')
                ax.plot(med, color = colors[n_col*i+j], label=f'median')
                
                if i == n_row - 1:
                    ax.set_xlabel('time [y]')
                if j == 0:
                    ax.set_ylabel('Wind contribution to sea level [cm]')
                ax.set_title(f'historical and {labels[n_col*i+j]}')
                ax.set_xlim(hist_start, 2101)
                if wind_model != 'Dangendorf':
                    ax.set_ylim(y_min, y_max)
                ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                ax.legend(loc='upper left')
                #plt.tight_layout()
    
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_median_percentiles_per scenario_{hist_start}')





def plot_med_percentiles_per_scenarios_all_wind_models(scenarios, labels, wc_historical, lower_bound = 0.05, upper_bound = 0.95, 
                                                       ra = 5, hist_start = 1950, wind_model = 'NearestPoint'):
    '''
    Function to make a plot of the median and upper and lower bound of the models for each scenario.
    Define the percentiles by setting the lower_bound and upper_bound and 
    what running average is applied by setting ra
    
    '''
    
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] 
    
    # Create dataframe of wind contribution
    wc_historical_np = wc_historical[0].wind_total.to_pandas().T
    wc_historical_tim = wc_historical[1].wind_total.to_pandas().T
    wc_historical_da = wc_historical[2].wind_total.to_pandas().T
    
    # Obtain median and bounds for wind contribution
    med_wc_np = wc_historical_np.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    lb_wc_np = wc_historical_np.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    ub_wc_np = wc_historical_np.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    
    med_wc_tim = wc_historical_tim.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    lb_wc_tim = wc_historical_tim.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    ub_wc_tim = wc_historical_tim.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    
    med_wc_da = wc_historical_da.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    lb_wc_da = wc_historical_da.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    ub_wc_da = wc_historical_da.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
    
    
    n_col = 3
    n_row = math.ceil(len(scenarios[0]) / n_col)
    n_delete = len(scenarios[0]) % n_col
    
    y_min = -5
    y_max = 5
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(20, 8))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            ax = axs[i,j]

            if i == n_row-1 and j in range(n_delete, n_col):
                fig.delaxes(axs[i,j])
                
                
            else:
                
                # NearestPoint
                scenario = scenarios[0][n_col*i+j]
                med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                
                color = colors[0]
                ax.fill_between(ub.index, ub, lb, color = color, alpha=0.3)
                ax.plot(med, color = color, label='NearestPoint')
                
                # Plot historical
                ax.fill_between(ub_wc_np.index, ub_wc_np, lb_wc_np, color = color, alpha=0.3)
                ax.plot(med_wc_np, color = color)

                
                # Timmerman
                scenario = scenarios[1][n_col*i+j]
                med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                
                color = colors[1]
                ax.fill_between(ub.index, ub, lb, color = color, alpha=0.3)
                ax.plot(med, color = color, label='Timmerman')
                
                # Plot historical
                ax.fill_between(ub_wc_tim.index, ub_wc_tim, lb_wc_tim, color = color, alpha=0.3)
                ax.plot(med_wc_tim, color = color)
                
                
                
                
                
                # Dangendorf
                scenario = scenarios[2][n_col*i+j]
                med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
                
                color = colors[2]
                ax.fill_between(ub.index, ub, lb, color = color, alpha=0.3)
                ax.plot(med, color = color, label='Dangendorf')
                
                # Plot historical
                ax.fill_between(ub_wc_da.index, ub_wc_da, lb_wc_da, color = color, alpha=0.3)
                ax.plot(med_wc_da, color = color)
                
                
                
                
                ax.set_xlabel('time [y]')
                ax.set_ylabel('Wind contribution to sea level [cm]')
                ax.set_title(f'historical and  {labels[n_col*i+j]}\n median and {int(lower_bound*100)}-{int(upper_bound*100)} percentiles')
                ax.set_xlim(hist_start, 2101)
                ax.set_ylim(y_min, y_max)
                ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
                ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
                ax.legend(loc='upper left')
                plt.tight_layout()
    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/{wind_model}_median_percentiles_per scenario_all_wind_models_{hist_start}')



def plot_med_percentiles_per_scenarios_all_wind_models(scenarios, labels, lower_bound = 0.05, upper_bound = 0.95, ra = 5
                                       , hist_start = 1950):
    '''
    Function to make a plot of the median and upper and lower bound of the models for each scenario and all three wind models
    Define the percentiles by setting the lower_bound and upper_bound and 
    what running average is applied by setting ra
    
    scenarios should be a list of lists of the wind contribution per scenario for the three wind models
    labels should be a list of the scenario names
    wc_historical should be a list of the three historical wind contributions resulting from the three wind models
    
    '''
    
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] 
    
    
    
    n_col = 3
    n_row = len(scenarios[0])
    
    y_min = -4
    y_max = 4
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(15, 3.5*n_row))
    
    
    for i in range(n_row):

        for j in range(n_col):
            
            ax = axs[i,j]
    
            scenario = scenarios[j][i]
                
                
            med = scenario.quantile(0.5, axis = 1).rolling(ra, center=True, min_periods=1).mean()
            lb = scenario.quantile(lower_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()
            ub = scenario.quantile(upper_bound, axis = 1).rolling(ra, center=True, min_periods=1).mean()

            ax.fill_between(ub.index, ub, lb, color = colors[i], alpha=0.3, 
                                label=f'{int(lower_bound*100)}-{int(upper_bound*100)} percentiles')
            ax.plot(med, color = colors[i], label=f'median')
                
            if i == n_row - 1:
                ax.set_xlabel('time [y]')
            if j == 0:
                ax.set_ylabel('Wind contribution to sea level [cm]')
            ax.set_xlim(hist_start, 2101)
            #ax.set_ylim(y_min, y_max)
            ax.axhline(color='darkgray', linestyle='-', linewidth = 1)  
            ax.axvline(2014.5, color='darkgray', linestyle='-', linewidth = 1)
            if j == 0:
                ax.legend(loc='upper left')
            ax.set_title(f'{wind_labels[j]} - historical and {labels[i]} \n running average = {ra}')
            plt.tight_layout()
            
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Projections/median_percentiles_per scenario_all_wm_{hist_start}')


