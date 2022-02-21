"""
File containing the Python functions to project the wind contribution to sea level rise into the 21st century.


Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks:
Projection.ipynb

"""

# Import necessary packages
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler


# Only use models occuring in both datasets
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0',
       'CAS-ESM2-0', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1',
       'CanESM5', 'CanESM5-CanOE', 'EC-Earth3', 'EC-Earth3-Veg',
       'EC-Earth3-Veg-LR', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL',
       'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
       'MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
       'NESM3', 'UKESM1-0-LL']

best_models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5', 
               'CanESM5-CanOE', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 
               'EC-Earth3-Veg-LR', 'GFDL-ESM4', 'MIROC-ES2L', 
                'MPI-ESM1-2-HR', 'NESM3']



def wind_contr_proj(wind_sce, wind_model = 'NearestPoint', use_models = 'bestmodels'):
    '''
    Function that creates a dataframe cointaining the projections into the 21st century using the regression
    coefficients of the regression without trend. 
    
    For wind_model choose ['NearestPoint', 'Timmerman', 'Dangendorf']
    For models choose ['bestmodels', 'allmodels']
    
    '''
    
    # Import scalers for standardization
    file = open(f'/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/Projections/scalers_{wind_model}.pkl', "rb")
    scalers = pickle.load(file)
    
    if use_models == 'bestmodels':
        models = best_models
    
    # Import regression coefficients
    path_reg_results = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Regression results/Projections/'
    results = pd.read_csv(path_reg_results + f'{wind_model}_results.csv', index_col = 'result')

    
    # Select models
    results = results[models]
    
    
    df = pd.DataFrame({'time' : wind_sce.time.values})
    df = df.set_index('time')
    
    
    for model in wind_sce.model.values:
        
        constant = results[model]['constant']
        
        if wind_model == 'NearestPoint':
        
            u2_coef = results[model]['u$^2$']
            v2_coef = results[model]['v$^2$']

            df['u2_data'] = wind_sce.u2.sel(model=model, drop=True).values
            df['v2_data'] = wind_sce.v2.sel(model=model, drop=True).values
            
            # Standardize dataframe
            df.iloc[:,:] = scalers[model].fit_transform(df)
            
            wind_contr = constant + (u2_coef * df['u2_data']) + (v2_coef * df['v2_data'])
            
            df = df.drop(['u2_data', 'v2_data'], axis=1)
            
        elif wind_model == 'Timmerman':
            
            wind_contr = constant
            
            for region in wind_sce.tim_region.values:
                u2_coef = results[model][f'{region} u$^2$']
                v2_coef = results[model][f'{region} u$^2$']
                
                df['u2_data'] = wind_sce.u2.sel(model=model, tim_region = region, drop=True).values
                df['v2_data'] = wind_sce.v2.sel(model=model, tim_region = region, drop=True).values
                
                # Standardize dataframe
                df.iloc[:,:] = scalers[model].fit_transform(df)
                
                wind_contr += (u2_coef * df['u2_data']) + (v2_coef * df['v2_data'])
                
                df = df.drop(['u2_data', 'v2_data'], axis=1)
                
        elif wind_model == 'Dangendorf':
        
            neg_coef = results[model]['Negative corr region']
            pos_coef = results[model]['Positive corr region']

            df['neg_data'] = wind_sce['Negative corr region'].sel(model=model, drop=True).values
            df['pos_data'] = wind_sce['Positive corr region'].sel(model=model, drop=True).values
            
            # Standardize dataframe
            df.iloc[:,:] = scalers[model].fit_transform(df)
            
            wind_contr = constant + (neg_coef * df['neg_data']) + (pos_coef * df['pos_data'])
            
            df = df.drop(['neg_data', 'pos_data'], axis=1)
            
        df[model] = wind_contr
        
    return df, results



