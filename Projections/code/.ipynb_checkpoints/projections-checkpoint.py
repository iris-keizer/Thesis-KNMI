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
        # Import best models
        path_best_models = '/Users/iriskeizer/Projects/ClimatePhysics/Thesis/Data/cmip6/Comparison results/'
        models = []

        # Source: https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/
        # open file and read the content in a list
        with open(path_best_models+'bestmodels.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                models.append(currentPlace)
    else:
        models = wind_hist.model.values
    
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



