# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:24:39 2024

@author: mattg



Check/improve:

* -999s being used for prediction?

* Only raster stack the most important predictors / those that have a minimum usefulness. Could reduce
memory requirements. 

* 500 km tiled prediction for loop could be implemented in parallel with enough memory. Can manage 
max. 2 simultaneously with PC.

* Further nan and unlikely value corrections for responses? E.g., forest height >100 m

* Hyperparameters set after some fiddling. Could incorporate a sensitivity analysis for learning_rate,
n_estimators, max_depth, num_leaves. This would add significant run time. 

* Add 0.9 and 0.1 quantiles for estimating reasonable range of departure to be considered degraded.
Assess whether useful. 

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

#%%

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'

#most recent reference site csv
obs_path=wdir+'scrap\\pts_updated_500m_v11_1kmspacing_20volexp_sampled_v1.csv'

#predictor and response lists with pathnames
pred_csv_path=wdir+'data\\predictors_described_v4.csv'
resp_csv_path=wdir+'data\\responses_described_v2.csv'

outdir = 'C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'
tile_outdir = 'F:\\veg2_postdoc\\raster_subset_v3\\'

#path with predictor and response rasters resampled to standard res, extent, CRS
data_path='F:\\veg2_postdoc\\raster_subset_v3'


#%%

"""
Read
"""

df=pd.read_csv(obs_path)
print(df)

preds=pd.read_csv(pred_csv_path)
print(preds)
resps=pd.read_csv(resp_csv_path)
print(resps)

preds2=list(preds['Predictor'][0:])

fn1=os.listdir(data_path)
fns=[s for s in fn1 if '.tif' in s]
fn_dirs=glob.glob(data_path+'\\*.tif')

#%%

"""
Check whether any predictors in dataframe are not present as geotiff in folder
Add preds to gtiff list

"""

gtiff_list=[]
pred_list=[]
for a in df.keys()[4:]:
    if os.path.isfile(data_path+'//'+a+'.tif'):
        print(a)
        if a in preds2:
            gtiff_list.append(data_path+'//'+a+'.tif')
            pred_list.append(a)
    else:
        print('Error for '+a)

#%%

"""
Some corrections to input predictor/response vars and rasters as required, for nans etc. 

"""

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

df[target_cols] = df[target_cols].replace(9999, np.nan)
df[target_cols] = df[target_cols].replace(99999, np.nan)

df['Forest_height_2019_AUS'].loc[pd.isna(df['Forest_height_2019_AUS'])]=0
df['Forest_height_2019_AUS'].loc[df['Forest_height_2019_AUS']>90]=np.nan

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
                       'AVP_060_100_EV_N_P_AU_TRN_N_20220826'
                       ]
        
pred_list = [col for col in pred_list if col not in cols_to_drop]     
print(str(len(df))+' points unfiltered')

#remove from gtiff list
filtered_gtiff_list = [
    path for path in gtiff_list
    if not any(col in path for col in cols_to_drop)
]

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

"""
Initialise 500 km tiles so that i can run predict without loading all predictors into memory

"""

tile_size = 500000  # m

tile_dir=tile_outdir+'tiled_'+str(int(tile_size/1000))+'km_v2'
if os.path.exists(tile_dir)==False:
    os.mkdir(tile_dir)

#open a raster to get its dimensions
with rio.open(gtiff_list[0]) as src:
    width, height = src.width, src.height
    transform = src.transform
    
    pixel_size_x = abs(transform.a)  # X resolution (meters per pixel)
    pixel_size_y = abs(transform.e)  # Y resolution (meters per pixel)

    # Convert tile size to pixels
    tile_size_x = int(tile_size / pixel_size_x)
    tile_size_y = int(tile_size / pixel_size_y)

    # Determine number of tiles in each direction
    num_tiles_x = (width // tile_size_x) + 1
    num_tiles_y = (height // tile_size_y) + 1

    # Create list of windows
    tiles = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_off = i * tile_size_x
            y_off = j * tile_size_y
            
            window = Window(x_off, y_off, tile_size_x, tile_size_y)
            tiles.append(window)

print(f"Generated {len(tiles)} tiles, each {tile_size}m x {tile_size}m")

#%%

# resp='Forest_height_2019_AUS'
# resp='agb_australia_90m'
# resp='Veg_NDVI_mean_Q1'
# resp='mean_eta_summer_2013-2024'
# resp='AusEFlux_GPP_longterm_mean_NSW'
# resp='PV_PC_50'

for resp in resps['Response']:
    
    print('')
    print(resp)
    if resp in df.keys():    
        
        df4=copy.deepcopy(df)
        pred_list2=copy.deepcopy(pred_list)
        
        pred_list2.append(resp)
        cv_all=df4[pred_list2]
        
        #check nan in cols
        invalid_counts = df.isna().sum() + (df == -999).sum() + (df == -9999).sum()
        invalid_percentage = (invalid_counts / len(df)) * 100
        valid_columns = invalid_percentage[invalid_percentage <= 10].index.intersection(cv_all.columns)
        cv_all = cv_all[valid_columns]
                
        cv_all=cv_all.dropna(axis=0)
        print('NaN filtered - '+str(len(cv_all)))
        
        if resp in cv_all.keys():
        
            x_train=cv_all.drop(resp, axis=1)
            y_train=np.ravel(cv_all[resp].values.reshape(-1, 1))
            
            print('Predictors: ')
            print(list(x_train.keys()))
            
            common_params = dict(
                learning_rate=0.06,
                n_estimators=2000,
                max_depth=30,
                num_leaves=256,
                min_child_samples=10,
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
                        
            if 'month' in resp:
                gbr = lgb.LGBMClassifier(**common_params)
            else:
                #using quantile for future upper and lower confidence estimation
                gbr = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **common_params)
                #gbr = lgb.LGBMRegressor(objective="regeression", **common_params)
        
            print('Training model...')
            all_models = gbr.fit(x_train, y_train)
                        
            imp = gbr.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': x_train.columns, 
                'Importance': imp
            }).sort_values(by="Importance", ascending=False)
            
            #cross-validate
            print('Conducting fivefold CV...')
            obs_vs_pred=fivefold(common_params, x_train, y_train)
            obs_vs_pred.to_csv(outdir+resp+'_BRT_q'+'_'+obs_path.split('\\')[-1].replace('.csv', '_v11.csv'))
            r2 = r2_score(obs_vs_pred['Observed'], obs_vs_pred['Predicted'])
            print(f"Overall R²: {r2:.4f}")
            
            for idx, window in enumerate(tiles):
                print(f"Processing tile {idx + 1}/{len(tiles)}")
                resampled_layers = []
                
                with rio.open(filtered_gtiff_list[0]) as src:
                    tile_data=src.read(window=window)
                    
                if (np.isnan(tile_data).all()) | ((tile_data==9999).all()):
                    print('No data in tile')
                    pred_q_reshape=np.full_like(tile_data, np.nan, dtype='float32')[0]
                
                else:
                    print('Loading rasters for tile...')
                    for f in filtered_gtiff_list:
                        #print(f)      
                        with rio.open(f) as src:
                            tile_data = src.read(window=window)
                        resampled_layers.append(tile_data)
                    
                    rstack = np.stack(resampled_layers, axis=-1)
                    n, n_rows, n_cols, n_bands = rstack.shape
                    rstack_new = pd.DataFrame(rstack.reshape(n_rows * n_cols, n_bands), columns=pred_list)
                    del rstack
                    rstack_new = rstack_new[list(x_train.columns)]
                    rstack_new=rstack_new.replace(9999, np.nan)
                    all_nan_mask = rstack_new.isna().any(axis=1)
                    
                    print('Predicting...')
                    pred_q = all_models.predict(rstack_new)                    
                    pred_q[all_nan_mask] = np.nan
                    pred_q_reshape = pred_q.reshape((n_rows, n_cols))
                    
                #export
                with rio.open(filtered_gtiff_list[0]) as src:
                    profile = src.profile
                tile_transform = src.window_transform(window)
                tile_profile = profile.copy()
                tile_profile.update({
                    "height": pred_q_reshape.shape[0],
                    "width": pred_q_reshape.shape[1],
                    "transform": tile_transform,
                    "count": 1,
                    "dtype": pred_q_reshape.dtype,
                    "compress": "lzw"
                })
                tile_out_path=tile_dir+'\\'+resp+'_tile_'+str(idx)+'.tif'
                with rio.open(tile_out_path, "w", **tile_profile) as dst:
                    dst.write(pred_q_reshape.astype(tile_profile["dtype"]), 1)
                print('Exported '+tile_out_path)
                print('')
          
        else:
            print('Not enough good data for response: '+resp)      
    else:
        print('Not in dataframe')
  

    
#%%