# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:24:39 2024

@author: mattg

Check/improve:

* -999s being used for prediction?

* Only raster stack the most important predictors / those that have a minimum usefulness. Could reduce
memory requirements. 

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
from shapely.geometry import box
import geopandas as gpd
from dask.distributed import Client, progress
from dask import delayed
from dask.distributed import as_completed  
from rasterio.enums import Resampling
import gc
import joblib
import time
import itertools



#%%

#local
wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scripts\\env1'
#most recent reference site csv
obs_path=wdir+'\\pts_updated_250m_v2_2km_175volexp_50distp_95cov_4pca_95out_residuals_v2_sampled.csv'
#predictor and response lists with pathnames
pred_csv_path=wdir+'\\predictors_described_v6.csv'
resp_csv_path=wdir+'\\responses_described_v3.csv'
outdir = wdir+'\\predict_BRT'
tile_outdir = outdir+'\\out_tiles'
data_path='F:\\veg2_postdoc\\raster_subset_v3'
#cluster fn
cluster_fn=wdir+'cluster_raster1_s_simplified.tif'
outdir_temp=outdir+'/models'

#nci
wdir='/g/data/xc0/project/natint/'
#obs_path=wdir+'output/v1/sample_rasters/pts_updated_250m_v1_1km_25volexp_90th_sampled.csv'
obs_path=wdir+'output/v2/sample_rasters/pts_updated_250m_v2_2km_175volexp_50distp_95cov_4pca_95out_residuals_v2_sampled.csv'
pred_csv_path=wdir+'input/predictors_described_v6.csv'
resp_csv_path=wdir+'input/responses_described_v3.csv'
outdir=wdir+'output/v2/predict_BRT'
data_path='/scratch/xc0/mg5402/raster_subset_v3'
tile_outdir=outdir+'/out_tiles'
cluster_fn=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'
outdir_temp=outdir+'/models'

if os.path.exists(outdir)==False:
    os.mkdir(outdir)
    
if os.path.exists(tile_outdir)==False:
    os.mkdir(tile_outdir)

if os.path.exists(outdir_temp)==False:
    os.mkdir(outdir_temp)

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
fn_dirs=glob.glob(data_path+'/*.tif')

print(str(len(fn_dirs))+' rasters for prediction')

#df=df.drop(['cluster_raster46_s_simplified'], axis=1)

#%%

"""
Resample cluster raster to match other rasters
"""

def standardise_ras(fn, fn_dir, resamp, outbnds, outdir_temp, outdir, res, r_mask, mask_meta, mask_ibra):
    with rio.open(fn_dir) as src:
        if src.crs is None:
            src_srs = 'EPSG:4326'  
        else:
            src_srs = str(src.crs)

        if resamp == 'BL':  # Bilinear resampling for continuous data
            resampling_method = Resampling.bilinear
        else:  # Nearest Neighbour for categorical data
            resampling_method = Resampling.nearest

        outfn1 = outdir_temp + '/' +fn + '.tif'

        with rio.open(outfn1, 'w', driver='GTiff', 
                           count=1, 
                           dtype='float32', 
                           crs=src.crs, 
                           transform=src.transform, 
                           width=int((outbnds[2] - outbnds[0]) / res), 
                           height=int((outbnds[3] - outbnds[1]) / res)) as dst:

            data = src.read(1, resampling=resampling_method)
            dst.write(data, 1)

    if mask_ibra:
        # Mask the raster based on the water mask and IBRA mask
        with rio.open(outfn1) as src:
            meta = src.meta
            ras = src.read(1)

            ras[(r_mask == 1) & (ras == meta['nodata'])] = 0
            ras[r_mask == mask_meta['nodata']] = meta['nodata']
            ras[r_mask == 0] = meta['nodata']

            outfn2 = outdir + '/' + fn + '.tif'
            with rio.open(outfn2, 'w', **meta) as dest:
                dest.write(ras, 1)

cluster_key = cluster_fn.split('/')[-1].split('.tif')[0]
#cluster_key='cluster_raster1_s_simplified'
outfn1 = data_path + '/' + cluster_key + '.tif'
#outfn1 = wdir + '/'+cluster_key + '.tif'

if not os.path.isfile(outfn1):
    with rio.open(fn_dirs[0]) as src:
        outbnds = src.bounds
        res=int(src.res[0])
    fn = cluster_fn   
    resamp = 'NN'
    r_masko = rio.open(data_path + '/water_mask_250m.tif')
    r_mask = r_masko.read(1)
    with rio.open(data_path + '/water_mask_250m.tif') as src:
        mask_meta = src.meta
    standardise_ras(cluster_key, fn, resamp, outbnds, outdir_temp, data_path, res, r_mask, mask_meta, mask_ibra=True)

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

def process_tile(idx, window, filtered_gtiff_list, model_path, x_train_cols, cat_features, tile_dir, resp, pred_list):
#for idx, window in enumerate(tiles[1000:1100]):
    #print(f"Processing tile {idx + 1}/{len(tiles)}")
    resampled_layers = []
    
    # Read the first raster for the window
    with rio.open(filtered_gtiff_list[0]) as src:
        tile_data = src.read(window=window)

    if np.isnan(tile_data).all() or (tile_data == 9999).all():
        #print('No data in tile')
        pred_q_reshape = np.full_like(tile_data, np.nan, dtype='float32')[0]
    else:
        all_models, cat_levels = joblib.load(model_path)
        #print('Loading rasters for tile...')
        for f in filtered_gtiff_list:
            with rio.open(f) as src:
                tile_data = src.read(window=window)
            resampled_layers.append(tile_data)

        rstack = np.stack(resampled_layers, axis=-1)
        n, n_rows, n_cols, n_bands = rstack.shape
        #print(f"pred_list length: {len(pred_list)}")
        rstack_flat = rstack.reshape(n_rows * n_cols, n_bands)

        rstack_new = pd.DataFrame(rstack_flat, columns=pred_list)

        del rstack 
        del resampled_layers
        gc.collect()

        rstack_new = rstack_new[list(x_train_cols)]
        rstack_new = rstack_new.replace(9999, np.nan)
        all_nan_mask = rstack_new.isna().any(axis=1)

        # Convert categorical features to category type
        #for cat in cat_features:
        #    if cat in rstack_new.columns:
        #        rstack_new[cat] = rstack_new[cat].astype('category')

        for cat in cat_features:
            if cat in rstack_new.columns:
            # Use saved levels for consistency
                categories = cat_levels.get(cat)
                if categories:
                    rstack_new[cat] = pd.Categorical(rstack_new[cat], categories=categories)


        #print('Predicting...')
        pred_q = all_models.booster_.predict(rstack_new, num_threads=-1)
        #pred_q = all_models.predict(rstack_new)
        pred_q[all_nan_mask] = np.nan
        pred_q_reshape = pred_q.reshape((n_rows, n_cols))

    # Write the prediction to a GeoTIFF
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

    tile_out_path = f"{tile_dir}/tile_{idx}_{resp}.tif"
    with rio.open(tile_out_path, "w", **tile_profile) as dst:
        dst.write(pred_q_reshape.astype(tile_profile["dtype"]), 1)
    
    #print(f'Exported {tile_out_path}')
    return tile_out_path

#%%
"""
Generates tile windows and a GeoDataFrame tile index for a raster.

Parameters:
    gtiff_list (list): List of raster file paths (only the first is used to get dimensions).
    tile_outdir (str): Output directory to save tile data.
    tile_size (int): Size of the tile in meters (default is 50000).

Returns:
    tiles (list): List of rasterio.windows.Window objects.
    gdf_tiles (GeoDataFrame): GeoDataFrame containing tile geometry and metadata.
"""

def generate_tile_index(gtiff_list, tile_outdir, tile_size):
    
    tile_dir = os.path.join(tile_outdir, f'tiled_{int(tile_size/1000)}km')
    os.makedirs(tile_dir, exist_ok=True)

    tile_polys = []
    tile_ids = []
    tile_col_offs = []
    tile_row_offs = []

    with rio.open(gtiff_list[0]) as src:
        width, height = src.width, src.height
        transform = src.transform

        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)

        tile_size_x = int(tile_size / pixel_size_x)
        tile_size_y = int(tile_size / pixel_size_y)

        num_tiles_x = (width // tile_size_x) + 1
        num_tiles_y = (height // tile_size_y) + 1

        tiles = []
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                x_off = i * tile_size_x
                y_off = j * tile_size_y

                window = Window(x_off, y_off, tile_size_x, tile_size_y)
                tiles.append(window)

                bounds = rio.windows.bounds(window, transform=transform)
                geom = box(*bounds)

                tile_polys.append(geom)
                tile_ids.append(len(tiles) - 1)
                tile_col_offs.append(x_off)
                tile_row_offs.append(y_off)

        gdf_tiles = gpd.GeoDataFrame({
            'tile_id': tile_ids,
            'col_off': tile_col_offs,
            'row_off': tile_row_offs,
            'geometry': tile_polys
        }, crs=src.crs)

    print(f"Generated {len(gdf_tiles)} tiles for prediction")
    return tile_dir, tiles, gdf_tiles

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
    #cluster fn is not in predictors csv so add manually
    if 'cluster' in a:
            gtiff_list.append(outfn1)
            pred_list.append(a)
        
#%%

"""
Some corrections to input predictor/response vars and rasters as required, for nans etc. 

"""

# Replace 9999 and 99999 with NaN if column exists
for col in df.columns:
    df[col] = df[col].replace([9999, 99999], np.nan)

# Handle Forest_height_2019_AUS separately
if 'Forest_height_2019_AUS' in df.columns:
    df.loc[df['Forest_height_2019_AUS'] > 90, 'Forest_height_2019_AUS'] = np.nan

if 'FCOV30_total_2001-2023' in df.columns:
    df.loc[df['FCOV30_total_2001-2023']==-9, 'FCOV30_total_2001-2023'] = np.nan
    df.loc[df['FCOV30_woody_2001-2023']==-9, 'FCOV30_woody_2001-2023'] = np.nan
    df.loc[df['FCOV30_grass_2001-2023']==-9, 'FCOV30_grass_2001-2023'] = np.nan

if 'bs_pc_50_2013-2024_mean_australia' in df.columns:
        df.loc[df['bs_pc_50_2013-2024_mean_australia']==255, 'bs_pc_50_2013-2024_mean_australia'] = np.nan
        df.loc[df['npv_pc_50_2013-2024_mean_australia']==255, 'npv_pc_50_2013-2024_mean_australia'] = np.nan
        df.loc[df['pv_pc_50_2013-2024_mean_australia']==255, 'pv_pc_50_2013-2024_mean_australia'] = np.nan

df[cluster_key] = df[cluster_key].astype("category")
df['PM_Lithol_Raster_from_Geol_SoilAtlas'] = df['PM_Lithol_Raster_from_Geol_SoilAtlas'].astype("category")
cat_features = [cluster_key, 'PM_Lithol_Raster_from_Geol_SoilAtlas']

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
                       'PM_radmap_v4_2019_filtered_pctk_GAPFilled', #artefacts
                       'PM_radmap_v4_2019_filtered_ppmt_GAPFilled', #artefacts
                       'PM_radmap_v4_2019_filtered_ppmu_GAPFilled', #artefacts
                       'PM_radmap_v4_2019_ratio_tk_GAPFilled', #artefacts
                       'PM_radmap_v4_2019_ratio_u2t_GAPFilled', #artefacts
                       'PM_radmap_v4_2019_ratio_uk_GAPFilled', #artefacts
                       'Soil_Illite',
                       'CLY_005_015_EV_N_P_AU_TRN_N_20210902', 
                       'CLY_060_100_EV_N_P_AU_TRN_N_20210902',
                       'NTO_005_015_EV_N_P_AU_NAT_C_20231101',
                       'NTO_060_100_EV_N_P_AU_NAT_C_20231101',
                       'Phosphorus_oxide_prediction_median',
                       'Iron_oxide_prediction_median', 
                       'Magnesium_oxide_prediction_median', 
                       'Manganese_oxide_prediction_median',
                       'Silica_oxide_prediction_median',
                       'Soil_Kaolinite', 
                       'Soil_Smectite',
                       'PHW_005_015_EV_N_P_AU_TRN_N_20220520',
                       'PHW_060_100_95_N_P_AU_TRN_N_20220520',
                       'Silica_oxide_prediction_median',
                       'Potassium_oxide_prediction_median',
                       'AWC_005_015_EV_N_P_AU_TRN_N_20210614',
                       'Calcium_oxide_prediction_median',
                       'AVP_060_100_EV_N_P_AU_TRN_N_20220826',
                       'Titanium_oxide_prediction_median',
                       'Aluminium_oxide_prediction_median',
                       'Sodium_oxide_prediction_median'
                       ]
        
pred_list = [col for col in pred_list if col not in cols_to_drop]     
print(str(len(df))+' points unfiltered')
len(pred_list)

#remove from gtiff list
filtered_gtiff_list = [
    path for path in gtiff_list
    if not any(col in path for col in cols_to_drop)
]

len(filtered_gtiff_list)
#sorted(filtered_gtiff_list)

#%%

"""
Initialise 500 km tiles so that i can run predict without loading all predictors into memory

"""

tile_dir, tiles, gdf_tiles = generate_tile_index(gtiff_list, tile_outdir, tile_size=50000)
#gdf_tiles.to_file(outdir+'tile_index_'+str(int(tile_size/1000))+'_km_v2.shp')

#%%

# resp='Forest_height_2019_AUS'
# resp='AusEFlux_GPP_longterm_mean_NSW'
resp='wcf_wagb_90m_v2'

client = Client(n_workers=8)  # Creates a local Dask cluster
print(client)

#%%

for resp in resps['Response']:
    
    print('')
    print(resp)
    if resp in df.keys():    
        
        df4=df.copy()
        pred_list2=pred_list.copy()
        
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
            x_train_cols=x_train.columns
            
            print('Predictors: ')
            print(list(x_train.keys()))
            
            #to overfit a little
            common_params = dict(
                learning_rate=0.06,
                n_estimators=1000,
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
                        
            if 'month' in resp:
                gbr = lgb.LGBMClassifier(**common_params)
            else:
                #using quantile for future upper and lower confidence estimation
                gbr = lgb.LGBMRegressor(objective="regression", alpha=0.5, **common_params)
                #gbr = lgb.LGBMRegressor(objective="regeression", **common_params)

            model_path=outdir_temp+'/'+resp+'_brt-model.joblib'
            if os.path.isfile(model_path) == False:
                print('Training model...')
                all_models = gbr.fit(x_train, y_train, categorical_feature=cat_features)
                #all_models = gbr.fit(x_train, y_train)
                
                imp = gbr.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': x_train.columns, 
                    'Importance': imp
                    }).sort_values(by="Importance", ascending=False)
                cat_levels = {}
                for col in cat_features:
                    if col in x_train.columns:
                        if not pd.api.types.is_categorical_dtype(x_train[col]):
                            x_train[col] = pd.Categorical(x_train[col])
                        cat_levels[col] = list(x_train[col].cat.categories)
                joblib.dump((all_models, cat_levels), model_path)
                print(f'Model saved to {model_path}')                
                print('Most important predictors:')
                print(importance_df[0:5])        
                del all_models
            
           
            #cross-validate
            # print('Conducting fivefold CV...')
            # obs_vs_pred=fivefold(common_params, x_train, y_train)
            # obs_vs_pred.to_csv(outdir+'/'+resp+'_BRT_q'+'_'+obs_path.split('/')[-1].replace('.csv', '_v1.csv'))
            # r2 = r2_score(obs_vs_pred['Observed'], obs_vs_pred['Predicted'])
            #print(f"Overall R²: {r2:.4f}")

            #tiles=tiles[1000:1500]            
            batch_size = 100
            counter=1
            total_batches = int(len(tiles) / batch_size)
            start_time = time.time()
            
            for i in range(0, len(tiles), batch_size):
                batch_futures = []
                batch_start = time.time()
                for idx, window in enumerate(tiles[i:i+batch_size], start=i):
                    future = client.submit(
                        process_tile,
                        idx,
                        window,
                        filtered_gtiff_list,
                        model_path,
                        x_train_cols,
                        cat_features,
                        tile_dir,
                        resp,
                        pred_list
                    )
                    batch_futures.append(future)

                print(f"Processing batch {counter} | {total_batches}")
                for future in as_completed(batch_futures):
                    result = future.result()
                    #print("Finished:", result)
               
                batch_end = time.time()
                elapsed = batch_end - start_time
                avg_per_batch = elapsed / counter
                remaining_batches = total_batches - counter
                est_remaining = avg_per_batch * remaining_batches
                elapsed_h, elapsed_m = divmod(int(elapsed), 3600)
                elapsed_m, elapsed_s = divmod(elapsed_m, 60)
                rem_h, rem_m = divmod(int(est_remaining), 3600)
                rem_m, rem_s = divmod(rem_m, 60)
            
                print(f"Elapsed: {elapsed_h}h {elapsed_m}m {elapsed_s}s "
                      f"| Est. Remaining: {rem_h}h {rem_m}m {rem_s}s")
                print('')
                counter+=1

            print('Completed '+resp)
            
        else:
            print('Not enough good data for response: '+resp)      
    else:
        print('Not in dataframe')
  

#%%


for resp in resps['Response']:
    
    print('')
    print(resp)
    if resp in df.keys():    
        
        df4=df.copy()
        pred_list2=pred_list.copy()
        
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
            x_train_cols=x_train.columns
            
            print('Predictors: ')
            print(list(x_train.keys()))
            
            #to overfit a little
            common_params = dict(
                learning_rate=0.06,
                n_estimators=50,
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
                        
            if 'month' in resp:
                gbr = lgb.LGBMClassifier(**common_params)
            else:
                #using quantile for future upper and lower confidence estimation
                gbr = lgb.LGBMRegressor(objective="regression", alpha=0.5, **common_params)
                #gbr = lgb.LGBMRegressor(objective="regeression", **common_params)

            # model_path=outdir_temp+'/'+resp+'_brt-model.joblib'
            # if os.path.isfile(model_path) == False:
            #     print('Training model...')
            #     all_models = gbr.fit(x_train, y_train, categorical_feature=cat_features)
            #     #all_models = gbr.fit(x_train, y_train)
                
            #     imp = gbr.feature_importances_
            #     importance_df = pd.DataFrame({
            #         'Feature': x_train.columns, 
            #         'Importance': imp
            #         }).sort_values(by="Importance", ascending=False)
            #     cat_levels = {}
            #     for col in cat_features:
            #         if col in x_train.columns:
            #             if not pd.api.types.is_categorical_dtype(x_train[col]):
            #                 x_train[col] = pd.Categorical(x_train[col])
            #             cat_levels[col] = list(x_train[col].cat.categories)
            #     joblib.dump((all_models, cat_levels), model_path)
            #     print(f'Model saved to {model_path}')                
            #     print('Most important predictors:')
            #     print(importance_df[0:5])        
            #     del all_models
            
            lr_vals   = [0.09, 0.12, 0.15]
            md_vals   = [40]
            nl_vals   = [512, 1024]
            
            cv_results = []            
            for lr, md, nl in itertools.product(lr_vals, md_vals, nl_vals):
                params = common_params.copy()
                params.update({
                    'learning_rate': lr,
                    'max_depth': md,
                    'num_leaves': nl
                })
                
                print(f"Running CV with lr={lr}, max_depth={md}, num_leaves={nl}...")
                obs_vs_pred = fivefold(params, x_train, y_train)

                r2 = r2_score(obs_vs_pred['Observed'], obs_vs_pred['Predicted'])
                print(f" → R² = {r2:.4f}\n")
                
                # 3d. Store results
                cv_results.append({
                    'learning_rate': lr,
                    'max_depth': md,
                    'num_leaves': nl,
                    'r2': r2
                })
            
            # 4. Convert to DataFrame and save
            df_cv = pd.DataFrame(cv_results)
            df_cv.to_csv(outdir + '/'+resp+'_cv_grid_search_results.csv', index=False)
            
            print("Grid search completed. Results written to cv_grid_search_results.csv")

