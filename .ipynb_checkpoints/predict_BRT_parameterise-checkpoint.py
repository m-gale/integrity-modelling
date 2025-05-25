# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:24:39 2024

@author: mattg

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
from rasterio.enums import Resampling
import gc
import joblib
import time
from dask import delayed, compute
import re



#%%

#nci
wdir='/g/data/xc0/project/natint/'
#obs_path=wdir+'output/v1/sample_rasters/pts_updated_250m_v1_1km_25volexp_90th_sampled.csv'
obs_dir=wdir+'output/v1/sample_rasters/'
pred_csv_path=wdir+'input/predictors_described_v6.csv'
resp_csv_path=wdir+'input/responses_described_v3.csv'
outdir=wdir+'output/v1/predict_BRT/sensitivity'
data_path='/scratch/xc0/mg5402/raster_subset_v3'
cluster_fn=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'
outdir_temp=wdir+'scrap'
vi_dir=obs_dir+'VI_GBRT_filtered_input_v2.csv'

if os.path.exists(outdir)==False:
    os.mkdir(outdir)
    
#%%

"""
Read
"""

dfs=glob.glob(obs_dir+'*NSW-only_sampled.csv')
print(str(int(len(dfs)))+' files found')
df1=pd.read_csv(dfs[0])

vi=pd.read_csv(vi_dir)

preds=pd.read_csv(pred_csv_path)
print(preds)
resps=pd.read_csv(resp_csv_path)
print(resps)

preds2=list(preds['Predictor'][0:])

fn1=os.listdir(data_path)
fns=[s for s in fn1 if '.tif' in s]
fn_dirs=glob.glob(data_path+'/*.tif')

print(str(len(fn_dirs))+' rasters for prediction')

#df=df.drop(['cluster_raster46_s_simplified'], axis=1)

cluster_key = cluster_fn.split('/')[-1].split('.tif')[0]
outfn1 = data_path + '/' + cluster_key + '.tif'
cat_features = [cluster_key]


#%%

"""
Fivefold cross validation function, to run for each response
Export csv with obs. vs. pred. for later graphics
"""

def fivefold(common_params, x_train, y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    obs_vs_pred_df = pd.DataFrame(columns=['Observed', 'Predicted', 'Fold'])
    
    for counter, (train_index, val_index) in enumerate(kf.split(x_train), start=1):
        X_train_fold, X_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]  # x_train is a DataFrame
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # y_train is a NumPy array
        
        gbr = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **common_params)
        print(f'Training CV model {counter}...')
        gbr.fit(X_train_fold, y_train_fold)
        y_pred = gbr.predict(X_val_fold)

        fold_df = pd.DataFrame({'Observed': y_val_fold, 'Predicted': y_pred, 'Fold': counter})
        obs_vs_pred_df = pd.concat([obs_vs_pred_df, fold_df], ignore_index=True)

    return obs_vs_pred_df

#%%


gtiff_list=[]
pred_list=[]
for a in df1.keys()[4:]:
    if os.path.isfile(data_path+'//'+a+'.tif'):
        print(a)
        if a in preds2:
            gtiff_list.append(data_path+'//'+a+'.tif')
            pred_list.append(a)
    else:           
            print('Error for '+a)
    #cluster fn is not in predictors csv so add manually
    if 'cluster' in a:
            gtiff_list.append(outfn1)
            pred_list.append(a)
        
#%%

"""
Some corrections to input predictor/response vars and rasters as required, for nans etc. 

"""

def fix_cols(df):
    target_cols = [
        "AusEFlux_GPP_longterm_mean_NSW", "AusEFlux_GPP_month_of_max_NSW", "BS_PC_50", "Forest_height_2019_AUS",
        "pcf_0-5", "pcf_5-10", "pcf_10-30", "h_peak_foliage_density", "NPV_PC_50", "OzWALD.LAI.AnnualMeans_NSW_clip",
        "Veg_AVHRR_FPAR_StdDev", "Veg_NDVI_mean_Q1", "Veg_NDVI_mean_Q2", "Veg_NDVI_mean_Q3", "Veg_NDVI_mean_Q4",
        "SOC_005_015_EV_N_P_AU_NAT_N_20220727", "PV_PC_50", "Veg_Persistant_green_Veg", "Veg_AVHRR_FPAR_Mean",
        "sdev_2013-2024_mean_australia", "bcdev_2013-2024_mean_australia", "edev_2013-2024_mean_australia",
        "agb_australia_90m", "bgb_australia_90m", "annual_range_eta_2013-2024", "mean_eta_autumn_2013-2024",
        "mean_eta_summer_2013-2024", "month_of_max_eta_2013-2024", "month_of_min_eta_2013-2024",
        "SOC_060_100_EV_N_P_AU_NAT_N_20220727", "CEC_005_015_EV_N_P_AU_TRN_N_20220826", "CEC_060_100_EV_N_P_AU_TRN_N_20220826"
    ]
    
    
    # Replace 9999 and 99999 with NaN if column exists
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].replace([9999, 99999], np.nan)
    
    # Handle Forest_height_2019_AUS separately
    if 'Forest_height_2019_AUS' in df.columns:
        df.loc[df['Forest_height_2019_AUS'].isna(), 'Forest_height_2019_AUS'] = 0
        df.loc[df['Forest_height_2019_AUS'] > 90, 'Forest_height_2019_AUS'] = np.nan
    
    df[cluster_key] = df[cluster_key].astype("category")
    print(str(len(df))+' points unfiltered')

    return df


#%%

"""
Drop these predictors because they show anthropogenic influence.
They also don't make response candidates because unlikely to be strongly related to integrity.
"""

cols_to_drop=['Phosphorus_oxide_prediction_median',
                       'SND_005_015_EV_N_P_AU_TRN_N_20210902',
                       'SND_060_100_EV_N_P_AU_TRN_N_20210902',
                       'SLT_005_015_EV_N_P_AU_TRN_N_20210902',
                       'SLT_060_100_EV_N_P_AU_TRN_N_20210902',
                       'PM_radmap_v4_2019_filtered_pctk_GAPFilled',
                       'PM_radmap_v4_2019_filtered_ppmt_GAPFilled',
                       'PM_radmap_v4_2019_filtered_ppmu_GAPFilled',
                       'PM_radmap_v4_2019_ratio_tk_GAPFilled',
                       'PM_radmap_v4_2019_ratio_u2t_GAPFilled',
                       'PM_radmap_v4_2019_ratio_uk_GAPFilled',
                       'Soil_Illite',
                       'CLY_005_015_EV_N_P_AU_TRN_N_20210902', 
                       'CLY_060_100_EV_N_P_AU_TRN_N_20210902',
                       'NTO_005_015_EV_N_P_AU_NAT_C_20231101',
                       'Phosphorus_oxide_prediction_median',
                       'Iron_oxide_prediction_median', 
                       'Magnesium_oxide_prediction_median', 
                       'Manganese_oxide_prediction_median',
                       'Silica_oxide_prediction_median'
                       'Soil_Kaolinite', 
                       'Soil_Smectite',
                       'PHW_005_015_EV_N_P_AU_TRN_N_20220520',
                       'PHW_060_100_95_N_P_AU_TRN_N_20220520',
                       'Soil_Kaolinite',
                       'Silica_oxide_prediction_median',
                       'Potassium_oxide_prediction_median',
                       'AWC_005_015_EV_N_P_AU_TRN_N_20210614',
                       'Calcium_oxide_prediction_median',
                       'AVP_060_100_EV_N_P_AU_TRN_N_20220826',
                       'cluster_raster1_s_simplified.1'
                       ]
        
pred_list = [col for col in pred_list if col not in cols_to_drop]     

#remove from gtiff list
filtered_gtiff_list = [
    path for path in gtiff_list
    if not any(col in path for col in cols_to_drop)
]
    

#%%


# resp='Forest_height_2019_AUS'
# resp='agb_australia_90m'
# resp='AusEFlux_GPP_longterm_mean_NSW'
#resps
resp='agb_australia_90m'
resp_list = [
    'agb_australia_90m', 
    'edev_2013-2024_mean_australia', 
    'Forest_height_2019_AUS', 
    'wcf_wagb_90m_v2', 
    'wcf_vegh_90m_v2', 
    'bcdev_2013-2024_mean_australia', 
    'bs_pc_50_2013-2024_mean_australia', 
    'npv_pc_50_2013-2024_mean_australia', 
    'Veg_NDVI_mean_Q1', 
    'Veg_NDVI_mean_Q3',
    'SOC_005_015_EV_N_P_AU_NAT_N_20220727', 
    'Veg_Persistant_green_Veg' 
]

client = Client(n_workers=16)  # Creates a local Dask cluster
print(client)

df_fn=dfs[0]

#%%
@delayed
def process_df_resp(df_fn):
    counter=0
    
    output_path = outdir + '/' + df_fn.split('/')[-1].replace('.csv', '_predictions.csv')
    #if os.path.isfile(output_path):
    #    print('Already exists')
    #else:
    for resp in resp_list:
        df = pd.read_csv(df_fn)
        df=fix_cols(df)
        #list(df.keys())
        if resp not in df.columns:
            print(f'{resp} not present in df')
        else:
                
            # Prepare data
            df4 = df.copy()
            pred_list2 = pred_list.copy()
            pred_list2.append(resp)
            cv_all = df4[pred_list2]
        
            invalid_counts = df.isna().sum() + (df == -999).sum() + (df == -9999).sum()
            invalid_percentage = (invalid_counts / len(df)) * 100
            valid_columns = invalid_percentage[invalid_percentage <= 10].index.intersection(cv_all.columns)
            cv_all = cv_all[valid_columns].dropna()
        
            if resp not in cv_all.columns:
                print('{resp} not present in training data')
            else:
            
                # Separate features and target
                x_train = cv_all.drop(resp, axis=1)
                y_train = np.ravel(cv_all[resp].values.reshape(-1, 1))
                x_train_cols = x_train.columns
            
                # Convert categorical features to 'category' dtype
                for cat in cat_features:
                    if cat in x_train.columns:
                        x_train[cat] = x_train[cat].astype('category')
            
                # Model selection
                if 'month' in resp:
                    gbr = lgb.LGBMClassifier(**common_params)
                else:
                    gbr = lgb.LGBMRegressor(objective="regression", alpha=0.5, **common_params)
            
                # Training model
                print(f'Training model for {resp} in {df_fn}...')
                model = gbr.fit(x_train, y_train, categorical_feature=cat_features)
            
                # Feature importance
                imp = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': x_train.columns, 
                    'Importance': imp
                }).sort_values(by="Importance", ascending=False)
            
                print(f'Most important predictors for {resp}:')
                print(importance_df.head())
            
                # Cross-validation (Fivefold)
                #print(f'Conducting fivefold CV for {resp} in {df_fn}...')
                #obs_vs_pred = fivefold(common_params, x_train, y_train)
                #output_cv_path = outdir + '/' + df_fn.split('/')[-1].replace('.csv', f'_{resp}_CV.csv')
                #obs_vs_pred.to_csv(output_cv_path)
                #r2 = r2_score(obs_vs_pred['Observed'], obs_vs_pred['Predicted'])
                #print(f"Overall R²: {r2:.4f} for {resp} in {df_fn}")
            
                # Predict on validation points
                rstack_new = pd.DataFrame(vi, columns=pred_list)
                rstack_new = rstack_new[list(x_train_cols)]
                rstack_new = rstack_new.replace(9999, np.nan)
                all_nan_mask = rstack_new.isna().any(axis=1)
            
                for cat in cat_features:
                    if cat in rstack_new.columns:
                        rstack_new[cat] = rstack_new[cat].astype('category')
            
                pred_q = model.booster_.predict(rstack_new, num_threads=-1)
                pred_q[all_nan_mask] = np.nan
                if counter==0:
                    df_new=vi.copy()
                    counter=1
                df_new[resp + '_predicted'] = pred_q
    
    # Save predictions
    df_new.to_csv(output_path)
    return output_path

#%%

common_params = dict(
    learning_rate=0.06,
    n_estimators=500,
    max_depth=40,
    num_leaves=256,
    min_child_samples=5,
    min_split_gain=0.1,
    max_bin=512,
    reg_alpha=0.1,
    reg_lambda=0.5,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbosity=-1,
    n_jobs=-1
)

#%%

tasks = []
for df_fn in dfs:
    tasks.append(process_df_resp(df_fn))

print(f'{len(tasks)} tasks scheduled')

results = compute(*tasks, scheduler='threads')  # or use 'processes'
print("Completed files:", results)


#%%
