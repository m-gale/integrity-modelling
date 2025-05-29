# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:28:58 2024

@author: mattg


Requires associated functions in other script


To do:

* remove more outliers from convex hull of ecotype calculation

* to increase pts: lower expansion, higher threshold, stricter on outliers, remove 50th distance requirement

* Sensitivity analysis:

    PCs = [2, 3, 4]   (complexity of environmental space)
    Outliers = [90, 95, 99]   (effect of outliers on 'sampled' environmental space)
    Hull expansion = [10, 25, 50]   (hull boundary uncertainty and environmental novelty of new pts)
    Min distance = [1000, 2000]   (spatial autocorrelation and preferential selection of distant points) 

    = 54 combinations!
    
    1. Clip ecotypes to those intersecting NSW ...DONE
    2. Compute ref sites x16 (for now)
    3. Train models for ~36 RS vars
    4. Sample ~60,000 NSW dataframe for environmental predictors
    5. Predict on NSW dataframe
    6. Model VI vs. 36 RS vars across parameters
    7. Compare R2 across parameter combinations 

    ... Use optimal parameters for national predictions.

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


#%%

#local
wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
scrap_dir=wdir+'scrap\\'
ref_dir='F:\\veg2_postdoc\\data\\reference\\National\\V1\\'


#nci
wdir='/g/data/xc0/project/natint/'
out_diri=wdir+'output/v2'

if os.path.exists(out_diri)==False:
    os.mkdir(out_diri)
    
out_dir=wdir+'output/v2/ref_sites/'
ref_dir=wdir+'input/compute_ref_sites/'

if os.path.exists(out_dir)==False:
    os.mkdir(out_dir)
    
#integrity hierarchy layers
ref_dirs=glob.glob(ref_dir+'*v2.tif')

#%%

"""
Load classes, pts

"""

#ecotype classification
cluster_dir=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'
with rio.open(cluster_dir) as src:
    cluster_raster=src.read(1)


#%%

"""
Load GDM layers

"""

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
Run parameters

"""

#site of site starter seed, where no sites exist in a class
#will control no. of sites to some degree.
start_seed=30

#min distance between new points. will control no. sites
min_distance = 2000  

#max convex hull volume overage proportion
#will determine total no. sites and how well each class is environmentally represented
cov_threshold=0.95

#simplifier controls batch size for max min distance function, larger is more simple/faster
#do not set above 0.05. very small numbers will revert to removing points one at a time
simplifier = 0.001 

#determines how much the existing sites hull is expanded to ensure that new sites are truly outside the existing hull
#i.e., 1.1 expands by 10% overall volume, with equal expansion in all dimensions
sfactor=1.75

#determines how many potential new sites are run in maximise min distance function
#more points take longer to run
#can be adjusted in turn with simplifier
maxkd=20000

#take a percentile of candidate points of highest distance from existing points.
#e.g., consider only the top half furthest points from existing points
distpa=50

batch_size = 50


#%%

"""
Initialise

"""

#start points from scratch
pts = gpd.GeoDataFrame(columns=["id", "class_kmeans", "source"], geometry=[], crs="EPSG:3577")
coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]

#initialise distance tree
if len(coords) > 0:
    global_tree = KDTree(coords)
else:
    global_tree = KDTree(np.empty((0, 2))) 

pts_updated=copy.deepcopy(pts)

cuq=np.unique(cluster_raster)

error_log=[]

print('Ready')

#%%

"""
Do the thing

"""

# Process through ecosystem integrity levels
#pass_no=1
#i=6
pass_nos = range(1, 6)  
for pass_no in pass_nos:
    print(f"\nProcessing reference site area level: {pass_no}")
        
    #i=1
    cov_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)
    for i in np.unique(cluster_raster):
        try:
            if ~np.isnan(i):
                print('')
                print('Pass '+str(pass_no))
                print(f"Processing class {i}")
                
                existing_samples = sum(pts_updated['class_kmeans'] == i)
                target_locs = (cluster_raster == i) & (gdm_raster_mask)
                ref_locs = (cluster_raster == i) & (ref_ras == pass_no) & (gdm_raster_mask)
                rows, cols = np.where(target_locs)
                ref_rows, ref_cols = np.where(ref_locs)
                ref_indices = np.column_stack((ref_rows, ref_cols))
    
                if len(ref_rows) > 0:
                    print(f"Pixels available for sampling: {ref_rows.size}")
                    # Extract point coordinates
                    pts_coords = [(geom.x, geom.y) for geom in pts_updated[pts_updated['class_kmeans']==i].geometry]
                    rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                    existing_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in rowcols])
                    candidate_sites_gdm = gdm_raster[target_locs, :]
                    ref_sites = gdm_raster[ref_locs, :]
                    #~15 sites min to generate hull after outlier removal
                    if (existing_sites_gdm.shape[0] > 15) & (len(ref_sites)>1):
                        existing_sites_pca = pca.transform(existing_sites_gdm)  
                        candidate_sites_pca = pca.transform(candidate_sites_gdm)  
                        ref_sites_pca = pca.transform(ref_sites)
                        if len(candidate_sites_pca)>500000:
                            candidate_sites_pca = candidate_sites_pca[np.random.choice(candidate_sites_pca.shape[0], size=500000, replace=False)]
                        existing_hull_volume = convex_hull_volume(existing_sites_pca, 99)
                        candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
                        #expand the hull of current sites by 10% volume to capture more diverse potential new sites
                        d = existing_sites_pca.shape[1]  # number of dimensions
                        centroid = np.mean(existing_sites_pca, axis=0)                                 
                        scale_factor = (sfactor) ** (1 / d)
                        expanded_existing_sites_pca = centroid + (existing_sites_pca - centroid) * scale_factor
                        hull = ConvexHull(remove_outliers(expanded_existing_sites_pca, 95))
                        outside_mask = points_outside_hull(hull, ref_sites_pca)
                        #initialise the combined hull, which gets updated in batch for loop
                        #combined hull serves volume comparisons
                        #expanded hull makes sure that truly outside-hull points are added
                        combined_hull=hull  
                        expanded_combined_hull=hull
                        if candidate_hull_volume > 0:
                            coverage_ratio = existing_hull_volume / candidate_hull_volume
                            print(f"Coverage ratio: {coverage_ratio:.2%}")
                            cov_raster[cluster_raster == i] = coverage_ratio  
                            outside_points = ref_sites_pca[outside_mask]
                            outside_indices = ref_indices[outside_mask]
    
                            print(f"Candidate sites: {outside_mask.shape[0]}")
                            print(f"Points outside convex hull: {outside_points.shape[0]}")
                            
                            if outside_points.shape[0] > 0:
                                new_sites_geo = [ref_trans * (col, row) for row, col in outside_indices]
                                #candidate points outside min distance with existing sites
                                global_distances, _ = global_tree.query(new_sites_geo)
                                new_sites=np.array(new_sites_geo)[global_distances>min_distance]
                                if len(new_sites)>0:
                                    #this is to avoid kdtree calculations that are unnecessarily large
                                    if len(new_sites)>maxkd:
                                        new_sites = new_sites[np.random.choice(new_sites.shape[0], size=maxkd, replace=False)]
                                    #initial distance filter to try to maximise distance between points
                                    mmd_batch=max(int(len(new_sites)*simplifier), 1)
                                    max_it=int((len(new_sites)/mmd_batch)*1.1)
                                    new_sites_filt=maximize_min_distance(new_sites, min_distance, mmd_batch, max_iterations=max_it)
                                    print(f"Points meeting distance criteria: {new_sites_filt.shape[0]}")
                                    if len(new_sites_filt)>0:
                                        #process batches, building the kdtree
                                        #re-evaluate the convex hull
                                        #len(new_sites_filt_geo)
                                        batches = [new_sites_filt[i:i + batch_size] for i in range(0, len(new_sites_filt), batch_size)]
                                        pts_new=[]
                                        new_coverage_ratio=coverage_ratio
                                        counter=0
                                        for batch in batches:
                                            if (new_coverage_ratio < cov_threshold):
                                                counter=counter+1
                                                #add if still outside hull
                                                batch_rowcols = [rowcol(ref_trans, x, y) for x, y in batch]
                                                batch_gdm = np.array([gdm_raster[row, col, :] for row, col in batch_rowcols])
                                                # Create a list of valid row-col indices where the data is not NaN
                                                nan_mask = np.array([~np.any(np.isnan(gdm_raster[row, col, :])) for row, col in batch_rowcols])
                                                batch_gdm=batch_gdm[nan_mask]
                                                batch_filt_pca = pca.transform(batch_gdm)
                                                #outside_mask = points_outside_hull(combined_hull, batch_filt_pca)
                                                outside_mask, min_distances = points_and_distance_outside_hull(expanded_combined_hull, batch_filt_pca)
                                                outside_mask=outside_mask & (min_distances>np.nanpercentile(min_distances[outside_mask], distpa)) 
                                                outside_points = batch_filt_pca[outside_mask]
                                                outside_indices = np.array(batch)[nan_mask][outside_mask]
                                                batch_new=[tuple(coord) for coord in outside_indices]
                                                print('Batch '+str(counter)+' - '+str(len(batch_new))+' still outside hull')
                                                #add pts in batch outside of convex hull
                                                pts_coords.extend(batch_new)
                                                pts_new.extend(batch_new)
                                                #calculate new %volume
                                                new_sites_rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                                                new_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in new_sites_rowcols])
                                                new_sites_filt_pca = pca.transform(new_sites_gdm)
                                                if len(remove_outliers(new_sites_filt_pca, 99))>6:
                                                    combined_hull = ConvexHull(remove_outliers(new_sites_filt_pca, 99))
                                                    centroid = np.mean(new_sites_filt_pca, axis=0)                                 
                                                    expanded_new_sites_filt_pca = centroid + (new_sites_filt_pca - centroid) * scale_factor
                                                    expanded_combined_hull=ConvexHull(remove_outliers(expanded_new_sites_filt_pca, 95))
                                                combined_hull_volume = combined_hull.volume
                                                new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                                            else:
                                                print('Already reached target coverage')
                                        new_df = pd.DataFrame(pts_new, columns=['longitude', 'latitude'])
                                        new_df['source'] = pass_no
                                        new_df['class_kmeans'] = i
                                        new_gdf = gpd.GeoDataFrame(new_df, 
                                                               geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                               crs='EPSG:3577') 
                                        print('Adding '+str(len(new_gdf))+' points')
                                        print('New coverage: '+str(new_coverage_ratio*100)[0:4]+'%')
                                        pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_kmeans', 'geometry']]], ignore_index=True)
                                        #update global tree
                                        global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry.to_crs("EPSG:3577")])    
                                else:
                                    print('No sites matching distance criteria')
                            else:
                                print('No sites outside of existing hull')
                        else:
                            print('No reference sites to choose from')
                    else:
                        print('Not enough existing sites to generate hull')
                        #not enough existing points to calculate a hull
                        #add a few points to get it going
                        new_sites_geo = [ref_trans * (col, row) for row, col in ref_indices]
                        if len(new_sites_geo)>5000:
                            new_sites_geo = np.array(new_sites_geo)[np.random.choice(len(new_sites_geo), size=5000, replace=False)]
                        #candidate points outside min distance with existing sites
                        global_distances, _ = global_tree.query(new_sites_geo)
                        new_sites=np.array(new_sites_geo)[global_distances>min_distance]
                        if len(new_sites)>0:
                            mmd_batch=max(int(len(new_sites)*simplifier), 1)
                            max_it=int((len(new_sites)/mmd_batch)*1.1)
                            new_sites_filt=maximize_min_distance(new_sites, min_distance, mmd_batch, max_iterations=max_it)
                            print(f"Points meeting distance criteria: {new_sites_filt.shape[0]}")
                            if len(new_sites_filt)>0:
                                new_sites_filt_geo = new_sites_filt[np.random.choice(len(new_sites_filt), size=min(len(new_sites_filt), start_seed), replace=False)]
                                #check nan in new sites
                                rowcols_check = [rowcol(ref_trans, x, y) for x, y in new_sites_filt_geo]
                                nan_mask = np.array([~np.any(np.isnan(gdm_raster[row, col, :])) for row, col in rowcols_check])
                                new_sites_filt_geo = new_sites_filt_geo[nan_mask]
                                new_df = pd.DataFrame(new_sites_filt_geo, columns=['longitude', 'latitude'])
                                new_df['source'] = pass_no
                                new_df['class_kmeans'] = i
                                new_gdf = gpd.GeoDataFrame(new_df, 
                                                       geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                       crs='EPSG:3577') 
                                print('Adding '+str(len(new_gdf))+' points')
                                pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_kmeans', 'geometry']]], ignore_index=True)
                                #update global tree
                                global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry.to_crs("EPSG:3577")])
                                
                                #run again
                                if len(new_sites_filt)>15:
                                    pts_coords = [(geom.x, geom.y) for geom in pts_updated[pts_updated['class_kmeans']==i].geometry]
                                    rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                                    existing_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in rowcols])
                                    candidate_sites_gdm = gdm_raster[target_locs, :]
                                    ref_sites = gdm_raster[ref_locs, :]
                                    existing_sites_pca = pca.transform(existing_sites_gdm)  
                                    candidate_sites_pca = pca.transform(candidate_sites_gdm)  
                                    ref_sites_pca = pca.transform(ref_sites)
                                    ref_indices = np.column_stack((ref_rows, ref_cols))
                                    if len(candidate_sites_pca)>500000:
                                        candidate_sites_pca = candidate_sites_pca[np.random.choice(candidate_sites_pca.shape[0], size=500000, replace=False)]
                                    existing_hull_volume = convex_hull_volume(existing_sites_pca, 99)
                                    candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
                                    #expand the hull of current sites by 10% volume to capture more diverse potential new sites
                                    d = existing_sites_pca.shape[1]  # number of dimensions
                                    centroid = np.mean(existing_sites_pca, axis=0)                                 
                                    scale_factor = (sfactor) ** (1 / d)
                                    expanded_existing_sites_pca = centroid + (existing_sites_pca - centroid) * scale_factor
                                    hull = ConvexHull(remove_outliers(expanded_existing_sites_pca, 95))
                                    outside_mask = points_outside_hull(hull, ref_sites_pca)
                                    #initialise the combined hull, which gets updated in batch for loop
                                    #combined hull serves volume comparisons
                                    #expanded hull makes sure that truly outside-hull points are added
                                    combined_hull=hull  
                                    expanded_combined_hull=hull                
                                    if candidate_hull_volume > 0:
                                        coverage_ratio = existing_hull_volume / candidate_hull_volume
                                        print(f"Coverage ratio: {coverage_ratio:.2%}")
                                        cov_raster[cluster_raster == i] = coverage_ratio  
                                        outside_points = ref_sites_pca[outside_mask]
                                        outside_indices = ref_indices[outside_mask]
        
                                        print(f"Candidate sites: {outside_mask.shape[0]}")
                                        print(f"Points outside convex hull: {outside_points.shape[0]}")
                                        
                                        if outside_points.shape[0] > 0:
                                            new_sites_geo = [ref_trans * (col, row) for row, col in outside_indices]
                                            #candidate points outside min distance with existing sites
                                            global_distances, _ = global_tree.query(new_sites_geo)
                                            new_sites=np.array(new_sites_geo)[global_distances>min_distance]
                                            if len(new_sites)>0:
                                                #this is to avoid kdtree calculations that are unnecessarily large
                                                if len(new_sites)>maxkd:
                                                    new_sites = new_sites[np.random.choice(new_sites.shape[0], size=maxkd, replace=False)]
                                                #initial distance filter to try to maximise distance between points
                                                mmd_batch=max(int(len(new_sites)*simplifier), 1)
                                                max_it=int((len(new_sites)/mmd_batch)*1.1)
                                                new_sites_filt=maximize_min_distance(new_sites, min_distance, mmd_batch, max_iterations=max_it)
                                                print(f"Points meeting distance criteria: {new_sites_filt.shape[0]}")
                                                if len(new_sites_filt)>0:
                                                    #process batches, building the kdtree
                                                    #re-evaluate the convex hull
                                                    batches = [new_sites_filt[i:i + batch_size] for i in range(0, len(new_sites_filt), batch_size)]
                                                    pts_new=[]
                                                    new_coverage_ratio=coverage_ratio
                                                    counter=0
                                                    #batch=batches[0]
                                                    for batch in batches:
                                                        if (new_coverage_ratio < cov_threshold):
                                                            counter=counter+1
                                                            #add if still outside hull
                                                            batch_rowcols = [rowcol(ref_trans, x, y) for x, y in batch]
                                                            batch_gdm = np.array([gdm_raster[row, col, :] for row, col in batch_rowcols])
                                                            # Create a list of valid row-col indices where the data is not NaN
                                                            nan_mask = np.array([~np.any(np.isnan(gdm_raster[row, col, :])) for row, col in batch_rowcols])
                                                            batch_gdm=batch_gdm[nan_mask]
                                                            batch_filt_pca = pca.transform(batch_gdm)
                                                            #outside_mask = points_outside_hull(combined_hull, batch_filt_pca)
                                                            outside_mask, min_distances = points_and_distance_outside_hull(expanded_combined_hull, batch_filt_pca)
                                                            outside_mask=outside_mask & (min_distances>np.nanpercentile(min_distances[outside_mask], distpa)) 
                                                            outside_points = batch_filt_pca[outside_mask]
                                                            outside_indices = np.array(batch)[nan_mask][outside_mask]
                                                            batch_new=[tuple(coord) for coord in outside_indices]
                                                            print('Batch '+str(counter)+' - '+str(len(batch_new))+' still outside hull')
                                                            #add pts in batch outside of convex hull
                                                            pts_coords.extend(batch_new)
                                                            pts_new.extend(batch_new)
                                                            #calculate new %volume
                                                            new_sites_rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                                                            new_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in new_sites_rowcols])
                                                            new_sites_filt_pca = pca.transform(new_sites_gdm)
                                                            if len(remove_outliers(new_sites_filt_pca, 99))>6:
                                                                combined_hull = ConvexHull(remove_outliers(new_sites_filt_pca, 99))
                                                                centroid = np.mean(new_sites_filt_pca, axis=0)                                 
                                                                expanded_new_sites_filt_pca = centroid + (new_sites_filt_pca - centroid) * scale_factor
                                                                expanded_combined_hull=ConvexHull(remove_outliers(expanded_new_sites_filt_pca, 95))
                                                            combined_hull_volume = combined_hull.volume
                                                            new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                                                        else:
                                                            print('Already reached target coverage')
                                                    new_df = pd.DataFrame(pts_new, columns=['longitude', 'latitude'])
                                                    new_df['source'] = pass_no
                                                    new_df['class_kmeans'] = i
                                                    new_gdf = gpd.GeoDataFrame(new_df, 
                                                                           geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                                           crs='EPSG:3577') 
                                                    print('Adding '+str(len(new_gdf))+' points')
                                                    print('New coverage: '+str(new_coverage_ratio*100)[0:4]+'%')
                                                    pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_kmeans', 'geometry']]], ignore_index=True)
                                                    #update global tree
                                                    global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry.to_crs("EPSG:3577")])    
                                            else:
                                                print('No sites matching distance criteria')
                                        else:
                                            print('No sites outside of existing hull')
                                else:
                                    print('No reference sites to choose from')
                else:
                    print('No reference sites available at level '+str(pass_no))
        except:
            print('Error')
            error_log.append([pass_no, i])

#%%

"""

Export
"""

vno='v2'
sfac=str(int(sfactor*100))
cthr=str(int(cov_threshold*100))
mindi=str(int(min_distance/1000))
diststr=str(distpa)

print('Number of reference points: '+str(len(pts_updated)))
outfn=out_dir+'pts_updated_250m_'+vno+'_'+mindi+'km_'+sfac+'volexp_'+diststr+'distp_'+cthr+'cov_4pca_95out.shp'

pts_updated.to_file(outfn)


#%%

"""
Average heirarchy per class

"""

pts=gpd.read_file(outfn)
                 
cluster_new=np.empty(cluster_raster.shape)
#i=1
for i in np.unique(cluster_raster):
    print(str(i))
    sources=pts.loc[pts['class_kmea']==i, 'source']
    sources = pd.to_numeric(sources, errors='coerce')
    average = sources.mean(skipna=True)
    cluster_new[cluster_raster==i]=average

meta=src.meta
with rio.open(outfn.replace('.shp', '_av_heirarchy.tif'), "w", **meta) as dst:
    dst.write(cluster_new, 1)
    

#%%

"""
Final coverage and count per ecotype

"""

cov_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)
count_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)

#i=368
pts_updated=pts
pass_no=1
for i in np.unique(cluster_raster):
    if ~np.isnan(i):
        print('')
        print(f"Processing class {i}")
        
        existing_samples = sum(pts_updated['class_kmea'] == i)
        count_raster[cluster_raster == i] = existing_samples
        target_locs = (cluster_raster == i) & (gdm_raster_mask)
        ref_locs = (cluster_raster == i) & (ref_ras == pass_no) & (gdm_raster_mask)
        rows, cols = np.where(target_locs)
        ref_rows, ref_cols = np.where(ref_locs)
        ref_indices = np.column_stack((ref_rows, ref_cols))

        #if len(ref_rows) > 0:
        print(f"Pixels available for sampling: {ref_rows.size}")
        # Extract point coordinates
        pts_coords = [(geom.x, geom.y) for geom in pts_updated[pts_updated['class_kmea']==i].geometry]
        rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
        existing_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in rowcols])
        candidate_sites_gdm = gdm_raster[target_locs, :]
        ref_sites = gdm_raster[ref_locs, :]
        #~15 sites min to generate hull after outlier removal
        if (existing_sites_gdm.shape[0] > 15) & (len(ref_sites)>1):
            existing_sites_pca = pca.transform(existing_sites_gdm)  
            candidate_sites_pca = pca.transform(candidate_sites_gdm)  
            ref_sites_pca = pca.transform(ref_sites)
            if len(candidate_sites_pca)>500000:
                candidate_sites_pca = candidate_sites_pca[np.random.choice(candidate_sites_pca.shape[0], size=500000, replace=False)]
            existing_hull_volume = convex_hull_volume(existing_sites_pca, 99)
            candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
            if candidate_hull_volume > 0:
                coverage_ratio = existing_hull_volume / candidate_hull_volume
                print(f"Coverage ratio: {coverage_ratio:.2%}")
                cov_raster[cluster_raster == i] = coverage_ratio  

        else:
            print('Not enough existing sites to generate hull')
            cov_raster[cluster_raster == i] = 0 

meta=src.meta
with rio.open(outfn.replace('.shp', '_coverage.tif'), "w", **meta) as dst:
    dst.write(cov_raster, 1)

with rio.open(outfn.replace('.shp', '_count.tif'), "w", **meta) as dst:
    dst.write(count_raster, 1)

#%%




