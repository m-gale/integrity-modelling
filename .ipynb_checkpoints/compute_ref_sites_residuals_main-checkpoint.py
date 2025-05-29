# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:28:58 2024

@author: mattg


Requires associated functions in other script


To do:

* See if you can increase the sampled random subset of points when maximising distance

* Re-run with smaller min dist for classes that don't reach threshold after full hierarchy?

"""

#%%

import os
import glob
import time
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import rowcol
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree, ConvexHull, Delaunay, distance
from scipy.spatial.distance import cdist, mahalanobis
from scipy.linalg import inv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyproj import Transformer
import copy
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#%%


#nci
wdir='/g/data/xc0/project/natint/'
out_diri=wdir+'output/v2'

if os.path.exists(out_diri)==False:
    os.mkdir(out_diri)
    
out_dir=wdir+'output/v2/ref_sites/'
ref_dir=wdir+'input/compute_ref_sites/'
pts_dir=out_diri+'/ref_sites/pts_updated_250m_v2_2km_175volexp_50distp_95cov_4pca_95out.shp'

if os.path.exists(out_dir)==False:
    os.mkdir(out_dir)
    
#integrity hierarchy layers
ref_dirs=glob.glob(ref_dir+'*v2.tif')

#residuals analysis layers
res_dir=wdir+'/output/v2/predict_BRT/out_tiles/tiled_50km_mosaic/'
res_fns=glob.glob(res_dir+'*residuals.tif')
ext_dir='/scratch/xc0/mg5402/raster_subset_v3/'
ext_fns=glob.glob(ext_dir+'*.tif')

#%%

"""
Load classes, pts

"""

#ecotype classification
cluster_dir=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'
with rio.open(cluster_dir) as src:
    cluster_raster=src.read(1).astype('int16')

    
#%%

"""
Load reference hierarchy
Each int (e.g., 1-5) refers to a reference level
Resample and re-readif required

"""

ref_fn=ref_dirs[0]

with rio.open(ref_fn) as src:
    ref_ras = src.read(1)    
    ref_trans=src.transform
    ref_crs=src.crs

if ref_ras.shape!=cluster_raster.shape:
    if os.path.isfile(ref_fn.replace('.tif', '_resampled.tif')):
         ref_fn=ref_fn.replace('.tif', '_resampled.tif')
    else:        
        resample_raster(ref_fn, cluster_dir, ref_fn.replace('.tif', '_resampled.tif'))
        ref_fn=ref_fn.replace('.tif', '_resampled.tif')
    with rio.open(ref_fn) as src:
        ref_ras = src.read(1)    
        ref_trans=src.transform
        ref_crs=src.crs

#%%


"""
Load residuals
"""

selected_vars=['wcf_wagb_90m_v2', 'Forest_height_2019_AUS',  'h_peak_foliage_density']

out_path = res_dir+'residuals_pc1_component.tif'
if os.path.isfile(out_path)==False:
    
    res_layers = []
    for fn in selected_vars: 
        print(fn)
        bench_dir = [fn1 for fn1 in res_fns if fn in os.path.basename(fn1)][0]
        ext_dir = [fn1 for fn1 in ext_fns if fn in os.path.basename(fn1)][0]
        with rio.open(bench_dir) as src:
            bench=src.read(1)
            bench_meta=src.meta
        with rio.open(ext_dir) as src:
            ext=src.read(1)
            ext_meta=src.meta
        res=abs(bench-ext)
        res_layers.append(res)
    del src
    
    res_stack = np.stack(res_layers, axis=-1) 
    H, W, B = res_stack.shape
    flat = res_stack.reshape(-1, res_stack.shape[-1])
    valid_mask = ~np.any(np.isnan(flat), axis=1)
    valid_data = flat[valid_mask]
    del res_stack
    
    # Sample 1% of valid data
    n_samples = int(0.01 * valid_data.shape[0])
    sample_idx = np.random.choice(valid_data.shape[0], size=n_samples, replace=False)
    sample_data = valid_data[sample_idx]
    del sample_idx
    scaler = StandardScaler()
    sample_data_std = scaler.fit_transform(sample_data)
    
    pca = PCA(n_components=1)
    pca.fit(sample_data_std)
    del sample_data_std
    print('Explained variance: '+str(pca.explained_variance_ratio_))
    
    valid_std = scaler.transform(valid_data)
    pc1_vals = pca.transform(valid_std)[:, 0]
    del valid_std
    del valid_data
    
    pc1_flat = np.full(flat.shape[0], np.nan, dtype='float32')
    pc1_flat[valid_mask] = pc1_vals
    pc1_raster = pc1_flat.reshape(H, W)
    del pc1_flat
    
    out_meta = bench_meta.copy()  
    out_meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan
    })
    
    with rio.open(out_path, 'w', **out_meta) as dst:
        dst.write(pc1_raster, 1)
    print(f"PC1 raster written to: {out_path}")

else:
    with rio.open(out_path) as src:
        pc1_raster=src.read(1)

#%%

"""
Initial pts

"""

pts=gpd.read_file(pts_dir)

#%%

"""
Run parameters

"""

#min distance between new points. will control no. sites
min_distance = 1000  

#simplifier controls batch size for max min distance function, larger is more simple/faster
#do not set above 0.05. very small numbers will revert to removing points one at a time
simplifier = 0.001 

#determines how many potential new sites are run in maximise min distance function
#more points take longer to run
#can be adjusted in turn with simplifier
maxkd=20000

#number of new sites based on residuals analysis per ecotype class
num_new_pts=100

#%%

"""
Initialise

"""

coords = np.array([(pt.x, pt.y) for pt in pts.geometry])

#initialise distance tree
if len(coords) > 0:
    global_tree = KDTree(coords)
else:
    global_tree = KDTree(np.empty((0, 2))) 

pts_updated=copy.deepcopy(pts)

cuq=np.unique(cluster_raster)

#%%

"""
Do the thing

"""

# Process through ecosystem integrity levels
#pass_no=2
#i=2

#exclude the lowest condition classes
#cant trust the anomalies wont be disturbance
pass_nos = range(1, 4)  
for pass_no in pass_nos:
    print(f"\nProcessing reference site area level: {pass_no}")
        
    #i=1
    #cov_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)
    pass_ref=(ref_ras == pass_no) 
    pc1_pass=(~np.isnan(pc1_raster))
    for i in cuq[1:]:
        if ~np.isnan(i):
            print('')
            print('Pass '+str(pass_no))
            print(f"Processing class {i}")            
            existing_samples = sum(pts_updated['class_kmea'] == i)

            if sum((pts_updated['class_kmea']==i) & (pts_updated['source'].str.contains('residuals_')))>=100:
                print('Already reached residuals target number')
            else:
                cluster_pass=(cluster_raster == i)
                #cluster_pass.dtype
                #cluster_raster.dtype
                
                ref_locs = (cluster_pass) & (pass_ref)
                init_locs = (ref_locs) & (pc1_pass)
                if np.nansum(init_locs) >0:
                    #print(poo)
                    reses=pc1_raster[init_locs]
                    p90=np.nanpercentile(reses, 95)
                    p10=np.nanpercentile(reses, 5)
                    #p_locs = np.zeros_like(pc1_raster, dtype=bool)
                    p_locs=(ref_locs) & ((pc1_raster>p90) | (pc1_raster<p10))
                    ref_rows, ref_cols = np.where(p_locs)
                    ref_indices = np.column_stack((ref_rows, ref_cols))
                    print(f"Pixels available for sampling: {ref_rows.size}")
                    if len(ref_rows)>0:
                        new_sites_geo = [ref_trans * (col, row) for row, col in ref_indices]
                        transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
                        new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]
                        #candidate points outside min distance with existing sites
                        global_distances, _ = global_tree.query(new_sites_pro)
                        new_sites=np.array(new_sites_pro)[global_distances>min_distance]
                        if len(new_sites)>0:
                            #this is to avoid kdtree calculations that are unnecessarily large
                            if len(new_sites)>maxkd:
                                new_sites = new_sites[np.random.choice(new_sites.shape[0], size=maxkd, replace=False)]
                            #if len(new_sites)>num_new_pts:
                            #initial distance filter to try to maximise distance between points
                            mmd_batch=max(int(len(new_sites)*simplifier), 1)
                            max_it=int((len(new_sites)/mmd_batch)*1.1)
                            new_sites_filt=maximize_min_distance(new_sites, min_distance, mmd_batch, max_iterations=max_it)
                            if len(new_sites_filt)>100:
                                #still too many
                                new_sites_filt=new_sites_filt[np.random.choice(new_sites_filt.shape[0], size=100, replace=False)]
                            new_sites=new_sites_filt
                            print(f"Points meeting distance criteria: {new_sites.shape[0]}")
    
                            #add them
                            new_df = pd.DataFrame(new_sites, columns=['longitude', 'latitude'])
                            new_df['source'] = 'residuals_'+str(pass_no)
                            new_df['class_kmea'] = i
                            new_gdf = gpd.GeoDataFrame(new_df, 
                                                   geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                   crs='EPSG:3577') 
                            
                            print('Adding '+str(len(new_gdf))+' points')
                            pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_kmea', 'geometry']]], ignore_index=True)
                            
                            #update global tree
                            global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry])    


#%%

"""

Export
"""

print('Number of reference points: '+str(len(pts_updated)))
pts_updated.to_file(pts_dir.replace('.shp', '_residuals.shp'))

#pts_updated[pts_updated['source'].str.contains('residuals')].to_file(scrap_dir+'pts_updated_500m_v11_1kmspacing_20volexp_residuals_v1.shp')



#%%

"""
Average heirarchy per class

"""

pts=gpd.read_file(scrap_dir+'pts_updated_500m_v11_1kmspacing_20volexp.shp')
cluster_new=np.empty(cluster_raster.shape)
i=1
for i in np.unique(cluster_raster):
    sources=pts.loc[pts['class_pca']==i, 'source']
    pass_numbers = sources.str.extract(r'pass(\d+)')[0].astype(int).tolist()
    cluster_new[cluster_raster==i]=np.nanmean(pass_numbers)

meta=src.meta
with rio.open(wdir+'scrap\\av_level.tif', "w", **meta) as dst:
    dst.write(cluster_new, 1)
    

#%%












gdm_dir=wdir+'input/classify_gdm_kmeans_input/GDM_250m/'
gdm_fns=glob.glob(gdm_dir+'*.tif')
len(gdm_fns)

#gdm_fns = [fn for fn in gdm_fns if 'GeoDist' not in fn]

gdm_layers = []
for fn in gdm_fns[:10]:  # take the first 10
    print(fn)
    if "WaterCoverage" in fn:
        with rio.open(fn) as src:
            water_mask=src.read(1)
            gdm_layers.append(water_mask)
    else:
        with rio.open(fn) as src:
            gdm_layers.append(src.read(1))
        
# Convert list to 3D NumPy array (shape: height x width x 10)
gdm_raster = np.stack(gdm_layers, axis=-1)  
gdm_raster[water_mask>0.5]=-9999
del gdm_layers
del water_mask

#mask so that nan values are not selected as rows, cols
gdm_raster[(gdm_raster == -9999)] = np.nan
gdm_raster_mask=~np.isnan(gdm_raster).any(axis=2)



#%%

"""
Fit PCA to GDM
So the convex hull is calculated in five dimensions instead of 10

"""

if os.path.isfile(out_dir+'pca_fit_4comp.joblib'):
    pca=joblib.load(out_dir+'pca_fit_4comp.joblib')
else:   
    # Reshape raster: (rows * cols, bands)
    print('Fitting PCA...')
    n_rows, n_cols, n_bands = gdm_raster.shape
    gdm_2d = gdm_raster.reshape(-1, n_bands)
    
    # Remove NaNs and invalid values (-999)
    mask = (~np.isnan(gdm_2d).any(axis=1)) & ((gdm_2d != -999).all(axis=1))
    gdm_valid = gdm_2d[mask]
    
    # Take a random 1% sample of valid pixels
    sample_size = int(0.01 * gdm_valid.shape[0])
    sample_indices = np.random.choice(gdm_valid.shape[0], sample_size, replace=False)
    gdm_sample = gdm_valid[sample_indices]
    
    # Fit PCA on the sampled data
    pca = PCA(n_components=min(n_bands, 4))
    pca.fit(gdm_sample)  # Learn PCA components from sample
        
    gdm_pcs_valid = pca.transform(gdm_valid)
    n_pixels = gdm_2d.shape[0]
    pc_full = np.full((n_pixels, 4), np.nan)
    pc_full[mask, :] = gdm_pcs_valid
    
    del gdm_valid, gdm_sample, mask, gdm_2d
    del gdm_pcs_valid

    pc_stack = pc_full.reshape((n_rows, n_cols, 4)).transpose((2, 0, 1))  # shape: (3, rows, cols)
    
    with rio.open(
        out_dir+"gdm_pca_4comp.tif", "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=4,
        dtype=src.meta['dtype'],
        crs=src.meta['crs'],  # Replace with actual CRS
        transform=src.meta['transform'],
        nodata=src.meta['nodata']  # Or set to a specific value like -9999 if desired
    ) as dst:
        dst.write(pc_stack)
        
    joblib.dump(pca, out_dir+'pca_fit_4comp.joblib')
    del pc_stack




