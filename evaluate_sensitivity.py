"""
Read csvs containing predicted reference values for NSW BioNet plots
Each csv corresponds to a set of parameters
For each csv / parameter combo:
Calculate departure of EI indicators
Generate RF model of VI, tot_native_richness, FCS, mVI
Evaluate OOB or CV error
"""


import numpy as np
import pandas as pd
import copy
import os
import rasterio as rio
import glob
from shapely.ops import transform
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
from dask.distributed import Client, progress
from dask import delayed
from dask.distributed import as_completed
import dask
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score
import sys

#%%

#nci
wdir='/g/data/xc0/project/natint/'
#obs_path=wdir+'output/v1/sample_rasters/pts_updated_250m_v1_1km_25volexp_90th_sampled.csv'
obs_dir=wdir+'output/v1/predict_BRT/sensitivity/'
cluster_fn=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'

#%%

cluster_key = cluster_fn.split('/')[-1].split('.tif')[0]
cat_features = [cluster_key]

#%%

obs=glob.glob(obs_dir+'*sampled_predictions.csv')
print(str(len(obs))+' files found')

vars=['VI', 'tot_native_richness', 'mVI']

#%%

@delayed
def process_ob(ob, vars, cluster_key):
    ob_fn = ob.split('/')[-1]
    print(ob_fn)
    sys.stdout.flush() 

    ob_pts=ob.replace('predict_BRT/sensitivity/', 'sample_rasters/').replace('_predictions','')
    ob_df=pd.read_csv(ob_pts)
    ob_len=len(ob_df)

    rem = re.search(r'(\d+)km_(\d+)volexp_(\d+)distp_(\d+)cov_(\d+)pca', ob_fn)
    dist = int(rem.group(1))  
    sfactor = int(rem.group(2))    
    cov = int(rem.group(4))     
    pca_no = int(rem.group(5))  

    df = pd.read_csv(ob)
    
    resps = [col.removesuffix('_predicted') for col in df.keys() if col.endswith('_predicted')]

    preds = [cluster_key]
    
    for resp in resps:
        print(resp)
        df[resp + '_diff'] = df[resp + '_predicted'] - df[resp]
        epsilon = 0.01
        df[resp + '_pdiff'] = (df[resp + '_predicted'] - df[resp]) / (df[resp + '_predicted'] + epsilon) * 100
        preds.append(resp + '_diff')
        preds.append(resp + '_pdiff')

    perf_rows = []

    for v in vars:
        preds2 = preds.copy()
        preds2.append(v)
        df_all = df[preds2].dropna()
        
        if df_all.empty:
            continue
        
        # Prepare data for training
        X = df_all[preds]
        y = df_all[v]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f'Training model for {v}')
        sys.stdout.flush() 
        rf = RandomForestRegressor(n_estimators=250, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")
        sys.stdout.flush() 

        perf_rows.append([dist, sfactor, cov, pca_no, v, r2, mse, ob_len])

    return perf_rows


client = Client(n_workers=16) 
print(client)

#ob=obs[0]
results = [process_ob(ob, vars, cluster_key) for ob in obs]
print(results)
compiled_results = dask.compute(*results)
print('Finished')

flattened_results = [item for sublist in compiled_results for item in sublist]
perf_df = pd.DataFrame(flattened_results, columns=['min_dist', 'sfactor', 'cov', 'no_pca', 'var', 'R2', 'MSE', 'No_obs'])

perf_df.to_csv(f'{obs_dir}sensitivity_performance_v2.csv', index=False)


#%%

