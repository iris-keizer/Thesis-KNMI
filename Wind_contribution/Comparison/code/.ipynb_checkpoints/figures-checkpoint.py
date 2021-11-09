"""
File containing the Python functions to plot comparison of different regression results


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
comparison.ipynb

"""




# Import necessary packages
import numpy as np
import xarray as xr
import pandas as pd
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



# Declare global variables
stations = station_names()

many_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
               'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
              'silver', 'lightcoral',  'maroon', 'tomato', 'chocolate', 
               'peachpuff', 'gold',  'goldenrod', 'yellow', 'yellowgreen', 'lawngreen',
              'palegreen', 'darkgreen', 'mediumseagreen', 'springgreen', 'aquamarine', 
               'mediumturquoise', 'paleturquoise', 'darkcyan', 'steelblue', 
               'dodgerblue', 'slategray',  'royalblue', 'navy', 'slateblue', 'darkslateblue', 
               'indigo',  'plum', 'darkmagenta', 'magenta', 'deeppink']





"""
COMPARISON
------------

"""




def plot_obs_tg_wc_one_station(tg_data, ts_lst, labels, station = 'Average', show_tg = True, smoothed = False):
    
    plt.figure(figsize=(10,3))
    
    
    if show_tg:
        plt.plot(tg_data.index.values, tg_data[station], color = 'darkgray')
        
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green']
    alphas = [1,1,1,0.6,0.6,0.6]
    styles = ['--', '--', '--','-', '-', '-']
    
    for i, ts in enumerate(ts_lst):
        
        if smoothed == False:
            plt.plot(ts.index.values, ts[station, 'wind total'], color = colors[i], alpha = alphas[i], linestyle = styles[4])
        elif smoothed == True:
            plt.plot(ts.index.values, ts[station], color = colors[i], alpha = alphas[i], linestyle = styles[4])
        
        
    plt.axhline(color='k', linestyle='--', linewidth = 1)
    plt.xlabel('time [y]')
    plt.ylabel('SLH [cm]')
    plt.title('station = '+station)
    
    if show_tg == False:
        labels = labels[1:]
    plt.legend(labels=labels, bbox_to_anchor=(1,1))
    plt.tight_layout()
    
    if show_tg:
        if smoothed:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_{station}_showtg_smoothed')
        else:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_{station}_showtg')
    else:
        if smoothed:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_{station}_smoothed')
        else:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_{station}')
    
    
    
    
def plot_obs_tg_wc_all_stations(tg_data, ts_lst, labels, show_tg = True, smoothed = False):
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green']
    alphas = [1,1,1,0.4,0.6,0.6]
    styles = ['--', '--', '--','-', '-', '-']
    
    
    fig, axs = plt.subplots(4, 2, figsize=(14, 10))


    for i in range(4):


        ax = axs[i,0]
        
        if show_tg:
            ax.plot(tg_data.index.values, tg_data[stations[2*i]], color='darkgray')
            
            
        for j, ts in enumerate(ts_lst):
        
            if smoothed == False:
                ax.plot(ts.index.values, ts[stations[2*i], 'wind total'], color = colors[j], alpha = alphas[j], linestyle = styles[4])
                
            elif smoothed == True:
                ax.plot(ts.index.values, ts[stations[2*i]], color = colors[j], alpha = alphas[j], linestyle = styles[4])
                
            
        ax.axhline(color='k', linestyle='--', linewidth = 1)   
        ax.set_title('station = '+stations[2*i])
        ax.set_xlabel('time [y]')
        ax.set_ylabel('SLH [cm]')
        if show_tg:
            if smoothed:
                ax.set_ylim(-10,10)
            else:
                ax.set_ylim(-16,16)
        else:
            if smoothed:
                ax.set_ylim(-3,4)
            else:
                ax.set_ylim(-8,9)
        plt.tight_layout()
            
            
            
        ax = axs[i,1]
        if i == 3:
            fig.delaxes(axs[3,1])
        else:

            if show_tg:
                ax.plot(tg_data.index.values, tg_data[stations[2*i+1]], color='darkgray')


            for j, ts in enumerate(ts_lst):
                
                if smoothed == False:
                    ax.plot(ts.index.values, ts[stations[2*i+1], 'wind total'], color = colors[j], alpha = alphas[j], linestyle = styles[4])
                    
                elif smoothed == True:
                    ax.plot(ts.index.values, ts[stations[2*i+1]], color = colors[j], alpha = alphas[j], linestyle = styles[4])
                    

            ax.axhline(color='k', linestyle='--', linewidth = 1)
            ax.set_title('station = '+stations[2*i+1])
            ax.set_xlabel('time [y]')
            ax.set_ylabel('SLH [cm]')
            if show_tg:
                if smoothed:
                    ax.set_ylim(-10,10)
                else:
                    ax.set_ylim(-16,16)
            else:
                if smoothed:
                    ax.set_ylim(-3,4)
                else:
                    ax.set_ylim(-8,9)
            plt.tight_layout()
            
            
    if show_tg == False:
        labels = labels[1:]
    plt.legend(labels = labels, loc=(1.2, -0.15))
    
    
    if show_tg:
        if smoothed:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_allstations_showtg_smoothed')
        else:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_allstations_showtg')
    else:
        if smoothed:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_allstations_smoothed')
        else:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/tg_wc_allstations')
    
    
    
def plot_obs_running_trend_acceleration(data_lst, label_lst, period_length = 40, station = 'Average'):
    df_lst_trend = []
    df_lst_acc = []
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green']
    alphas = [1,1,1,0.6,0.6,0.6]
    
    for data in data_lst:
        starting_idx = np.arange(period_length//2, data.index.size-period_length//2, 1)
        
        time_lst = []
        trend_lst = []
        acc_lst = []
        for i in starting_idx:
            time = data.index[i:i+period_length]
            
            y = data[station][i:i+period_length].values
            
            fit = np.polyfit(time, y, 2)
            time_lst.append(data.index[i+period_length//2])
            trend_lst.append(fit[1])
            acc_lst.append(fit[0])
            
        df_lst_trend.append(pd.DataFrame({'time':time_lst, 'trend':trend_lst}))
        df_lst_trend[-1] = df_lst_trend[-1].set_index('time')
        df_lst_acc.append(pd.DataFrame({'time':time_lst, 'acceleration':acc_lst}))
        df_lst_acc[-1] = df_lst_acc[-1].set_index('time')
        
        
    df_trend = pd.concat(df_lst_trend, axis=1, keys=label_lst)
    df_trend.columns = df_trend.columns.droplevel(1)
    df_acc = pd.concat(df_lst_acc, axis=1, keys=label_lst)
    df_acc.columns = df_acc.columns.droplevel(1)
    
    
    plt.figure(figsize=(9,3))
    for i, label in enumerate(label_lst):
        plt.scatter(df_trend.index, df_trend[label], label = label,
                   marker = 'x', s=3, color = colors[i], alpha = alphas[i])
    plt.xlabel('time [y]')
    plt.ylabel('trend [cm/y]')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.figure(figsize=(9,3))
    for i, label in enumerate(label_lst):
        plt.scatter(df_acc.index, df_acc[label], label = label,
                   marker = 'x', s=3, color = colors[i], alpha = alphas[i])
    plt.xlabel('time [y]')
    plt.ylabel('acceleration [cm/y$^2$]')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    
    
def plot_obs_running_trend(data_lst, label_lst, period_length = 40, station = 'Average'):
    df_lst_trend = []
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green']
    alphas = [1,1,1,0.6,0.6,0.6]
    
    for data in data_lst:
        
        time_lst = []
        trend_lst = []
        for i in range(data.index.size-period_length):
            time = data.index[i:i+period_length]
            
            y = data[station][i:i+period_length].values
            
            fit = np.polyfit(time, y, 1)
            time_lst.append(time[period_length//2])
            trend_lst.append(fit[0])
            
        df_lst_trend.append(pd.DataFrame({'time':time_lst, 'trend':trend_lst}))
        df_lst_trend[-1] = df_lst_trend[-1].set_index('time')
        
        
    df_trend = pd.concat(df_lst_trend, axis=1, keys=label_lst)
    df_trend.columns = df_trend.columns.droplevel(1)
    
    
    plt.figure(figsize=(9,3))
    for i, label in enumerate(label_lst):
        plt.scatter(df_trend.index, df_trend[label]*10, label = label,
                   marker = 'x', s=3, color = colors[i], alpha = alphas[i])
    plt.xlabel('time [y]')
    plt.ylabel('trend [mm/y]')
    plt.ylim(-0.5,1.1)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.axhline(color='k', linestyle='--', linewidth = 1)
    plt.tight_layout()
    

    
   





    
    
    
def plot_zos_wc_per_model_one_station(zos, ts_lst, labels, station = 'Average', show_zos = True, smoothed = False):
    """
    Function to make a plot of the zos timeseries and regression result for all cmip6 models, for one station and for all 
    the wind contributions 
    For station choose ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']
    
    """
    models = ts_lst[0].model.values
    
    
    if show_zos:
        if smoothed:
            y_min = -8
            y_max = 8
        else:
            y_min = -15
            y_max = 15
    else:
        if smoothed:
            y_min = -3
            y_max = 3
        else:
            y_min = -10
            y_max = 10
    
    
    
    
    fig, axs = plt.subplots(9, 4, figsize=(24, 20))


    for i in range(9):


        ax = axs[i,0]
        
        
        if show_zos:
            ax.plot(zos.time.values, zos.sel(station=station, model=models[4*i]), color = 'darkgray')
        
        for ts in ts_lst:
            ax.plot(ts.time.values, ts.sel(station = station, model = models[4*i]).values)
            
        ax.set_xlabel('time [y]')
        ax.set_ylabel('zos [cm]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i])
        ax.set_ylim(y_min,y_max)
        plt.tight_layout()
        
        
        
        ax = axs[i,1]
        
        if show_zos:
            ax.plot(zos.time.values, zos.sel(station=station, model=models[4*i+1]), color = 'darkgray')

        for ts in ts_lst:
            ax.plot(ts.time.values, ts.sel(station = station, model = models[4*i+1]))

        ax.set_xlabel('time [y]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i+1])
        ax.set_ylim(y_min,y_max)
        plt.tight_layout()
    
        
        ax = axs[i,2]
        
        if show_zos:
            ax.plot(zos.time.values, zos.sel(station=station, model=models[4*i+2]), color = 'darkgray')

        for ts in ts_lst:
            ax.plot(ts.time.values, ts.sel(station = station, model = models[4*i+2]))

        ax.set_xlabel('time [y]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i+2])
        ax.set_ylim(y_min,y_max)
        plt.tight_layout()
        
        
        ax = axs[i,3]
        
        if show_zos:
            ax.plot(zos.time.values, zos.sel(station=station, model=models[4*i+3]), color = 'darkgray')

        for ts in ts_lst:
            ax.plot(ts.time.values, ts.sel(station = station, model = models[4*i+3]))

        ax.set_xlabel('time [y]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i+3])
        ax.set_ylim(y_min,y_max)
        plt.tight_layout()
        
        if i == 0:
            if show_zos:
                ax.legend(labels = labels, bbox_to_anchor=(1, 1))
            else:
                ax.legend(labels = labels[1:], bbox_to_anchor=(1, 1))
    
    if show_zos:
        if smoothed:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/zos_wc_per_model_{station}_showzos_smoothed')
        else:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/zos_wc_per_model_{station}_showzos')
    else:
        if smoothed:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/zos_wc_per_model_{station}_smoothed')
        else:
            plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/zos_wc_per_model_{station}')
            
            
       
                    
            
def plot_comp_reg_results_one_station(results_era5, results_20cr, results_hist, station, wind_model):
    markers = ['v', '^', '<', '>']
    
    
    plt.figure()
    
    
    if wind_model == 'NearestPoint' or wind_model == 'Timmerman':
        
        plt.scatter(results_era5['u$^2$'][station], 
                    results_era5['v$^2$'][station], 
                    label = '', marker='x',color='k', s=80)
        
        plt.scatter(results_20cr['u$^2$'][station], 
                    results_20cr['v$^2$'][station], 
                    label = '', marker='+',color='k', s=100)
        
        for idx, model in enumerate(results_hist.model.values):
                
                
            if wind_model == 'NearestPoint':

                plt.scatter(results_hist.u2.sel(station=station, model = model).values, 
                                results_hist.v2.sel(station=station, model = model).values, 
                                marker = markers[int((3.6*idx)/36)], alpha=.8)


            elif wind_model == 'Timmerman':
                 
                
                u2 = (results_hist.channel_u2.sel(station=station, model = model).values + 
                      results_hist.south_u2.sel(station=station, model = model).values + 
                      results_hist.midwest_u2.sel(station=station, model = model).values + 
                      results_hist.mideast_u2.sel(station=station, model = model).values + 
                      results_hist.northwest_u2.sel(station=station, model = model).values + 
                      results_hist.northeast_u2.sel(station=station, model = model).values)

                v2 = (results_hist.channel_v2.sel(station=station, model = model).values + 
                      results_hist.south_v2.sel(station=station, model = model).values + 
                      results_hist.midwest_v2.sel(station=station, model = model).values + 
                      results_hist.mideast_v2.sel(station=station, model = model).values + 
                      results_hist.northwest_v2.sel(station=station, model = model).values + 
                      results_hist.northeast_v2.sel(station=station, model = model).values)

                plt.scatter(u2, v2, 
                            marker = markers[int((3.6*idx)/36)], alpha=.8)

                
        plt.xlabel('u$^2$ reg. coef. [-]')
        plt.ylabel('v$^2$ reg. coef. [-]')  
        plt.xlim(-3.5,3.5)
        plt.ylim(-2,2)
                
                
    elif wind_model == 'Dangendorf':
        
        plt.scatter(results_era5['Negative corr region'][station], 
                    results_era5['Positive corr region'][station], 
                    label = '', marker='x',color='k', s=80)
        
        plt.scatter(results_20cr['Negative corr region'][station], 
                    results_20cr['Positive corr region'][station], 
                    label = '', marker='+',color='k', s=100)

        
        for idx, model in enumerate(results_hist.model.values):
            plt.scatter(results_hist.neg_corr_region.sel(station=station, model = model).values, 
                        results_hist.pos_corr_region.sel(station=station, model = model).values, 
                        marker=markers[int((3.6*idx)/36)], alpha=.8)
        plt.xlabel('negative reg. coef. [-]')
        plt.ylabel('positive reg. coef. [-]')
        plt.ylim(-2.9, 2.9)
        plt.xlim(-2.9, 2.9)
            
    else: print('wind_model does not exist!')
        
    
    plt.title(f'wind regression model = {wind_model}')
    labels = ['era5','20cr'] + list(results_hist.model.values)
    plt.legend(labels=labels, ncol=3, bbox_to_anchor=(1.1, 1))
    plt.axhline(color='k', linestyle='-', linewidth = 0.3)
    plt.axvline(color='k', linestyle='-', linewidth = 0.3)
    plt.grid()

    
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/coefs_{wind_model}')
    
    
    
    
    
def plot_obs_sine_fits(data_lst, fits_dfs, label_lst, labels):
    station = 'Average'
    ymin = -2
    ymax = 2
    xmin = 1836
    xmax = 2020
    
    def test_sine(x, dist, amp, freq, phi):
        return dist + amp * np.sin(freq * x + phi)
    
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 4.5))


    for i in range(2):


        ax = axs[i,0]
        ax.plot(data_lst[3*i].index.values, data_lst[3*i][station].values)
        for fits_df in fits_dfs:
            ax.plot(data_lst[3*i].index.values, test_sine(data_lst[3*i].index.values, 
                                                          fits_df[label_lst[3*i], 'value']['y_distance'], 
                                                          fits_df[label_lst[3*i], 'value']['amplitude'], 
                                                          fits_df[label_lst[3*i], 'value']['frequency'], 
                                                          fits_df[label_lst[3*i], 'value']['phase']))
        
        if i == 0:
            ax.legend(labels=labels)
        if i == 1: ax.set_xlabel('time [y]')
        ax.set_ylabel('SLH [cm]')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_title(f'data = {label_lst[3*i]} \n amp = ' + str(round(fits_df[label_lst[3*i], 'value']['amplitude'], 2))
                     + ' cm, wavelength = ' + str(int(round(2*np.pi/fits_df[label_lst[3*i], 'value']['frequency'],0))) + ' y')
        
        ax = axs[i,1]
        ax.plot(data_lst[3*i+1].index.values, data_lst[3*i+1][station].values)
        for fits_df in fits_dfs:
            ax.plot(data_lst[3*i+1].index.values, test_sine(data_lst[3*i+1].index.values, 
                                                          fits_df[label_lst[3*i+1], 'value']['y_distance'], 
                                                          fits_df[label_lst[3*i+1], 'value']['amplitude'], 
                                                          fits_df[label_lst[3*i+1], 'value']['frequency'], 
                                                          fits_df[label_lst[3*i+1], 'value']['phase']))

        ax.set_title(f'data = {label_lst[3*i+1]} \n amp = ' + str(round(fits_df[label_lst[3*i+1], 'value']['amplitude'], 2))
                     + ' cm, wavelength = ' + str(int(round(2*np.pi/fits_df[label_lst[3*i+1], 'value']['frequency'],0))) + ' y')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)


        ax = axs[i,2]
        ax.plot(data_lst[3*i+2].index.values, data_lst[3*i+2][station].values)
        for fits_df in fits_dfs:
            ax.plot(data_lst[3*i+2].index.values, test_sine(data_lst[3*i+2].index.values, 
                                                          fits_df[label_lst[3*i+2], 'value']['y_distance'], 
                                                          fits_df[label_lst[3*i+2], 'value']['amplitude'], 
                                                          fits_df[label_lst[3*i+2], 'value']['frequency'], 
                                                          fits_df[label_lst[3*i+2], 'value']['phase']))
        
        ax.set_title(f'data = {label_lst[3*i+2]} \n amp = ' + str(round(fits_df[label_lst[3*i+2], 'value']['amplitude'], 2))
                     + ' cm, wavelength = ' + str(int(round(2*np.pi/fits_df[label_lst[3*i+2], 'value']['frequency'], 0))) + ' y')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        
        
        
    plt.tight_layout()

    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/sine_fits_obs')
    
    
    
def plot_comp_cmip6_fit_results(obs_fits_df, cmip6_fits_df):
    wind_models = ['NearestPoint', 'Timmerman', 'Dangendorf']
    
    for wind_model in wind_models:
    
        if wind_model == 'NearestPoint':
            lbl_era5 = 'np_era5'
            lbl_20cr = 'np_20cr'
        elif wind_model == 'Timmerman':
            lbl_era5 = 'tim_era5'
            lbl_20cr = 'tim_20cr'
        elif wind_model == 'Dangendorf':
            lbl_era5 = 'dang_era5'
            lbl_20cr = 'dang_20cr'
        else: print('wind_model does not exist!')


        plt.figure()
        plt.scatter(obs_fits_df[lbl_era5, 'value']['wavelength'], obs_fits_df[lbl_era5, 'value']['amplitude'], 
                    marker='x', color = 'b' ,s=80)
        plt.scatter(obs_fits_df[lbl_20cr, 'value']['wavelength'], obs_fits_df[lbl_20cr, 'value']['amplitude'],
                    marker='x', color = 'r', s=80)

        
        for i, model in enumerate(cmip6_fits_df[wind_model].columns.levels[0].values):
            plt.scatter(cmip6_fits_df[wind_model, model, 'value']['wavelength'], 
                        cmip6_fits_df[wind_model, model, 'value']['amplitude'],
                    marker='x', alpha=.7, color = many_colors[i])

        labels = ['era5','20cr'] + list(cmip6_fits_df[wind_model].columns.levels[0].values)
        plt.legend(labels=labels, ncol=3, bbox_to_anchor=(1.1, 1))
        plt.grid()
        plt.title('wind regression model = '+wind_model)
        plt.xlabel('wavelength [y]')
        plt.ylabel('amplitude [cm]')
        plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/amp_wl_{wind_model}')
        
        
        
def plot_cmip6_fits_per_model(data, fit_df1, fit_df2):
    """
    
    """
    models = data.model.values
    
    y_min = -2
    y_max = 2
    
    
    def test_sine(x, dist, amp, freq, phi):
        return dist + amp * np.sin(freq * x + phi)
    
    
    fig, axs = plt.subplots(9, 4, figsize=(24, 20))


    for i in range(9):


        ax = axs[i,0]
        
        
        ax.plot(data.time.values, data.sel(model = models[4*i]).values)
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df1[models[4*i], 'value']['y_distance'], 
                                                      fit_df1[models[4*i], 'value']['amplitude'], 
                                                      fit_df1[models[4*i], 'value']['frequency'], 
                                                      fit_df1[models[4*i], 'value']['phase'])) 
        
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df2[models[4*i], 'value']['y_distance'], 
                                                      fit_df2[models[4*i], 'value']['amplitude'], 
                                                      fit_df2[models[4*i], 'value']['frequency'], 
                                                      fit_df2[models[4*i], 'value']['phase']),
               linestyle='--') 
        
        ax.set_xlabel('time [y]')
        ax.set_ylabel('zos [cm]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i])
        #ax.set_ylim(y_min,y_max)
        plt.tight_layout()
        
        if i==0:
            labels = ['wind contribution', 'fit $\lambda_0=40$', 'fit $\lambda_0=60$']
            ax.legend(labels=labels)
            
            
        ax = axs[i,1]
        
        
        ax.plot(data.time.values, data.sel(model = models[4*i+1]).values)
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df1[models[4*i+1], 'value']['y_distance'], 
                                                      fit_df1[models[4*i+1], 'value']['amplitude'], 
                                                      fit_df1[models[4*i+1], 'value']['frequency'], 
                                                      fit_df1[models[4*i+1], 'value']['phase'])) 
        
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df2[models[4*i+1], 'value']['y_distance'], 
                                                      fit_df2[models[4*i+1], 'value']['amplitude'], 
                                                      fit_df2[models[4*i+1], 'value']['frequency'], 
                                                      fit_df2[models[4*i+1], 'value']['phase']),
               linestyle='--')    
        
        ax.set_xlabel('time [y]')
        ax.set_ylabel('zos [cm]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i+1])
        #ax.set_ylim(y_min,y_max)
        plt.tight_layout()
    
        
        ax = axs[i,2]
        
        
        ax.plot(data.time.values, data.sel(model = models[4*i+2]).values)
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df1[models[4*i+2], 'value']['y_distance'], 
                                                      fit_df1[models[4*i+2], 'value']['amplitude'], 
                                                      fit_df1[models[4*i+2], 'value']['frequency'], 
                                                      fit_df1[models[4*i+2], 'value']['phase']))   
        
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df2[models[4*i+2], 'value']['y_distance'], 
                                                      fit_df2[models[4*i+2], 'value']['amplitude'], 
                                                      fit_df2[models[4*i+2], 'value']['frequency'], 
                                                      fit_df2[models[4*i+2], 'value']['phase']),
               linestyle='--')     
        
        ax.set_xlabel('time [y]')
        ax.set_ylabel('zos [cm]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i+2])
        #ax.set_ylim(y_min,y_max)
        plt.tight_layout()
        
        
        ax = axs[i,3]
        
        
        ax.plot(data.time.values, data.sel(model = models[4*i+3]).values)
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df1[models[4*i+3], 'value']['y_distance'], 
                                                      fit_df1[models[4*i+3], 'value']['amplitude'], 
                                                      fit_df1[models[4*i+3], 'value']['frequency'], 
                                                      fit_df1[models[4*i+3], 'value']['phase']))   
        
        ax.plot(data.time.values, test_sine(data.time.values, 
                                                      fit_df2[models[4*i+3], 'value']['y_distance'], 
                                                      fit_df2[models[4*i+3], 'value']['amplitude'], 
                                                      fit_df2[models[4*i+3], 'value']['frequency'], 
                                                      fit_df2[models[4*i+3], 'value']['phase']),
               linestyle='--')     
        
        ax.set_xlabel('time [y]')
        ax.set_ylabel('zos [cm]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        ax.set_title('model = ' + models[4*i+3])
        #ax.set_ylim(y_min,y_max)
        plt.tight_layout()
        

        
def plot_cmip6_running_trend(data_lst, label_lst, period_length = 40, station = 'Average'):
    df_lst_trend = []
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green']
    alphas = [1,1,1,0.6,0.6,0.6]
    
    y_min = -1.1
    y_max = 1.1
    
    xr_lst1 = []
    for data in data_lst:
        data = data.sel(station = station)
        
        xr_lst = []
        for model in data.model.values:
            
            time_lst = []
            trend_lst = []
            for i in range(data.time.size-period_length):
                time = data.time[i:i+period_length].values

                y = data.sel(model = model)[i:i+period_length].values

                fit = np.polyfit(time, y, 1)
                time_lst.append(time[period_length//2])
                trend_lst.append(fit[0]*10)
            
            xr_lst.append(xr.DataArray(data = trend_lst, dims = ['time'], coords = dict(time = time_lst), 
                         attrs=dict(units = 'mm/y', description = 'Trend resulting from linear fit', name = 'trend')))
            
        xr_lst1.append(xr.concat(xr_lst, dim=data.model.values).rename({'concat_dim':'model'}))
    
    
    dataset = xr.Dataset(data_vars = {label_lst[0]:xr_lst1[0], label_lst[1]:xr_lst1[1], label_lst[2]:xr_lst1[2]})
    
    
    fig, axs = plt.subplots(9, 4, figsize=(24, 20))
    models = dataset.model.values

    
    for i in range(9):
        
        
        ax = axs[i,0]
        
        for label in label_lst:
            ax.scatter(dataset.time.values, dataset[label].sel(model = models[4*i]).values,
                       marker = 'x', s=3)
        ax.set_ylabel('trend [mm/y]')
        ax.set_title(f'model = {models[4*i]}')
        ax.set_ylim(y_min, y_max)
        if i == 0:
            ax.legend(labels = label_lst)
        if i == 8:
            ax.set_xlabel('time [y]')
        ax.axhline(color='k', linestyle='--', linewidth = 1)
        plt.tight_layout()
        
        
        for j in range(1,4):
            ax = axs[i,j]

            for label in label_lst:
                ax.scatter(dataset.time.values, dataset[label].sel(model = models[4*i+j]).values,
                           marker = 'x', s=3)
            ax.axhline(color='k', linestyle='--', linewidth = 1)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'model = {models[4*i+j]}')
            if i == 8:
                ax.set_xlabel('time [y]')
            plt.tight_layout()
            
    plt.tight_layout()
    plt.savefig(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Figures/Wind contribution/comparison/cmip6_trend_allmodels')
        
        
