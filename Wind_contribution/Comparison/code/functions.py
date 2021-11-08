"""
File containing the Python functions to be able to compare different regression results


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
comparison.ipynb

"""



# Import necessary packages
import statsmodels.api as sm
import statsmodels as sm
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend
from scipy import optimize

"""
Practical functions
-------------------


"""



def station_names(): 
    """
    Function to obtain tide gauge station names as list
    
    """
    return ['Vlissingen', 'Hoek v. Holland', 'Den Helder', 'Delfzijl', 'Harlingen', 'IJmuiden', 'Average']




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



def test_sine(x, dist, amp, freq, phi):
    """
    Define sine function, used for fitting
    """
    
    return dist + amp * np.sin(freq * x + phi)




# Declare global variables
stations = station_names()
lowess = sm.nonparametric.smoothers_lowess.lowess








def obs_lws_smoothed_df(df, window, data_type, detrend_data = False):
    """
    Function to create a new dataframe containing smoothed version of observational data
    
    For data_type choose ['tg', 'reg']
    
    """
    
    if detrend_data == True:
        df = df.apply(detrend)
    
    
    lws_lst = []
    for station in stations:
        if data_type == 'tg':
            column = station
        elif data_type == 'reg':
            column = station, 'wind total'
        
        lws_lst.append(lowess(df[column].values, 
                              df.index.values,
                              get_frac(window, df, dtype='DataFrame'), 
                              return_sorted = False))
    
    
    new_df = pd.DataFrame({'time':df.index.values, 
                           stations[0]:lws_lst[0], 
                           stations[1]:lws_lst[1], 
                           stations[2]:lws_lst[2], 
                           stations[3]:lws_lst[3], 
                           stations[4]:lws_lst[4], 
                           stations[5]:lws_lst[5], 
                           stations[6]:lws_lst[6]
                          })
    new_df = new_df.set_index('time')
    
    return new_df



def wc_cmip6_lws_smoothed_ds(ds, window, detrend_data = False):
    """
    Function to create a new dataset containing smoothed version of observational data

    
    """
    
    if detrend_data == True:
        
        ds = detrend_dim(ds, 'time')
    
    years = ds.time.values
    frac = get_frac(window, ds, dtype='DataSet')
    
    def lowess_1d(data):
        return lowess(data, years, frac, return_sorted = False)

    new_ds = xr.apply_ufunc(lowess_1d, # The 1D function followed by its arguments
                                  ds,
                                  input_core_dims=[['time']],
                                  output_core_dims = [['time']],
                                  vectorize = True,
                           )
    
    return new_ds



def detrend_dim(da, dim, deg=1): 
    """
    Function that detrends the data from a dataarray along a single dimension
    deg=1 for linear fit
    
    """
    
    p = da.polyfit(dim=dim, deg=deg)
    coord = da[dim] - da[dim].values[0]
    trend = coord*p.polyfit_coefficients.sel(degree=1)
    return da - trend





def obs_sine_fit(data, wavelength):
    station = 'Average'
    
    df = pd.DataFrame({'parameter': ['y_distance', 'amplitude', 'frequency', 'phase', 'wavelength']})
    df = df.set_index('parameter')
    
    params, params_covariance = optimize.curve_fit(test_sine, 
                                                   data.index.values, 
                                                   data[station].values, 
                                                   p0=[0, 0.7, 2*np.pi/(wavelength), 0]) # freq = 2pi/wavelength
    
    
    
    df['value'] = list(params) +  [2*np.pi/params[2]]
    df['variance'] = list(np.diag(params_covariance)) + [np.nan]
    
    return df



def obtain_obs_sine_fits(data_lst, label_lst, wavelength):
    
    df_lst = []
    
    for data in data_lst:
        df_lst.append(obs_sine_fit(data, wavelength))
        
    df = pd.concat(df_lst, axis = 1, keys = label_lst)
    
    return df



def obtain_sine_fits_wavelengths(data):
    wavelengths = range(30,71, 10)
    
    df_lst = []
    
    for wl in wavelengths:
        df_lst.append(obs_sine_fit(data, wl))
        
    df = pd.concat(df_lst, axis = 1, keys = wavelengths)
    
    return df



def cmip6_sine_fit(data, wavelength):
    station = 'Average'
    
    
    results_lst = []
    r2_lst = []
    
    for model in data.model.values:
        y = data.sel(station = station, model = model).values
        
        params, params_covariance = optimize.curve_fit(test_sine, 
                                                       data.time.values, 
                                                       y, 
                                                       p0=[0, 0.7, 2*np.pi/(wavelength), 0])
        # freq = 2pi/wavelength
        df = pd.DataFrame({'parameter': ['y_distance', 'amplitude', 'frequency', 'phase', 'wavelength']})
        df = df.set_index('parameter')

        df['value'] = list(params) +  [2*np.pi/params[2]]
        df['variance'] = list(np.diag(params_covariance)) + [np.nan]
        results_lst.append(df)
        
        # Evaluate performance of fit
        y_fit = test_sine(data.time.values, params[0], params[1], params[2], params[3])
        ss_res = np.sum((y - y_fit) ** 2) # residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2) # total sum of squares
        r2 = 1 - (ss_res / ss_tot) # r-squared
        r2_lst.append(r2)
        
    
    df_performance = pd.DataFrame({'model': data.model.values, 'r$^2$': r2_lst})
    df_performance = df_performance.set_index('model')
    
    return pd.concat(results_lst, axis=1, keys = data.model.values), df_performance

def obtain_cmip6_sine_fits(data_lst, label_lst, wavelength):
    
    df_res_lst = []
    df_perf_lst = []
    for data in data_lst:
        df_results, df_performance = cmip6_sine_fit(data, wavelength)
        df_res_lst.append(df_results)
        df_perf_lst.append(df_performance)
        
    df_res = pd.concat(df_res_lst, axis = 1, keys = label_lst)
    df_perf = pd.concat(df_perf_lst, axis = 1, keys = label_lst)
    
    
    return df_res, df_perf