# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:28:58 2024

@author: mattg

1. 
2. 
3. 
4. 
5. Repeat 3 and 4 through the heirarchy


To do:

See if you can increase the sampled points when maximising distance
Re-run with smaller min dist for harder classes?
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


#%%

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
scrap_dir=wdir+'scrap\\'
ref_dir='F:\\veg2_postdoc\\data\\reference\\National\\V1\\'


#%%

"""
Load reference areas

"""

ref_dirs=glob.glob(ref_dir+'*v1.tif')


#%%

"""
Load classes, pts

"""

#cluster_dir=scrap_dir+'cluster_raster24_10240000_s_90m.tif'
cluster_dir=scrap_dir+'cluster_raster46_s_simplified_500m.tif'
with rio.open(cluster_dir) as src:
    cluster_raster=src.read(1)

obs_path=wdir+'data\\HCAS_ref_sites\\HCAS_2.3\\data\\0.Inferred_Reference_Sites\\HCAS23_RC_BenchmarkSample_NSW_clipped.shp'
#obs_path='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\new_pts_test13.shp'
#obs_path='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\pts_updated_test19_250m.shp'
pts = gpd.read_file(obs_path)
pts=pts.to_crs(src.crs)
coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]

with rio.open(cluster_dir) as cluster_src:
    class_values = list(cluster_src.sample(coords))
class_values = [val[0] if val or val == 0 else np.nan for val in class_values]
pts['class_pca']=class_values


#%%

"""
Load GDM layers

"""

gdm_dir='F:\\veg2_postdoc\\data\\reference\\National\\mokany_etal_GDM_layers\\data\\250m\\'
gdm_fns=glob.glob(gdm_dir+'*9s.tif')
len(gdm_fns)

#if required
#for fn in gdm_fns[0:10]:
#    resample_raster(fn, cluster_dir, fn.replace('.tif', 'resampled_500m.tif'))

#gdm_fns=glob.glob(gdm_dir+'*90m.tif')
gdm_fns=glob.glob(gdm_dir+'*resampled_500m.tif')

gdm_layers = []
for fn in gdm_fns[:10]:  # Ensure we only take the first 10
    print(fn)
    with rio.open(fn) as src:
        gdm_layers.append(src.read(1))  # Read first band

# Convert list to 3D NumPy array (shape: height x width x 10)
gdm_raster = np.stack(gdm_layers, axis=-1)  
del gdm_layers

#mask so that nan values are not selected as rows, cols
gdm_raster[gdm_raster == -9999] = np.nan
gdm_raster_mask=~np.isnan(gdm_raster).any(axis=2)

#%%

"""
Fit PCA

"""

# Reshape raster: (rows * cols, bands)
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
pca = PCA(n_components=min(n_bands, 5))
pca.fit(gdm_sample)  # Learn PCA components from sample

del gdm_valid, gdm_sample, mask, gdm_2d

#%%
"""
Do the thing

"""

#reset pts
#HCAS isn't useful
pts=pts.iloc[0:0]

sample_coords = [geom.coords[0] for geom in pts.to_crs("EPSG:3577").geometry]

#site of site starter seed, where no sites exist in a class
#will control no. of sites to some degree.
start_seed=30

#min distance between new points. will control no. sites
min_distance = 1000  

#max convex hull volume overage proportion
#will determine total no. sites and how well each class is environmentally represented
cov_threshold=0.9

#simplifier controls batch size for max min distance function, larger is more simple/faster
#do not set above 0.05. very small numbers will revert to removing points one at a time
simplifier = 0.001 

#determines how much the existing sites hull is expanded to ensure that new sites are truly outside the existing hull
#i.e., 1.1 expands by 10% overall volume, with equal expansion in all dimensions
sfactor=1.2

#determines how many potential new sites are run in maximise min distance function
#more points take longer to run
#can be adjusted in turn with simplifier
maxkd=20000

#initialise distance tree
if len(sample_coords) > 0:
    global_tree = KDTree(sample_coords)
else:
    global_tree = KDTree(np.empty((0, 2))) 

pts_updated=copy.deepcopy(pts)

cuq=np.unique(cluster_raster)

error_log=[]

# Process through ecosystem integrity levels
#pass_no=0
#if starting from scratch we need to plant the seed at pass 0 but repeat it to fill hull
#pass_nos=[0, 1, 2, 3]
#i=6
pass_nos = range(0, 5)  
for pass_no in pass_nos:
    print(f"\nProcessing reference site area level: {pass_no}")
    ref_fn = ref_dirs[pass_no]
    print(ref_fn)
    
    #read ref raster and resample to cluster res if required
    if os.path.isfile(ref_fn.replace('.tif', '_resampled_500m.tif'))==False:
        print('Resampling '+ref_fn)
        resample_raster(ref_fn, cluster_dir, ref_fn.replace('.tif', '_resampled_500m.tif'))
    with rio.open(ref_fn.replace('.tif', '_resampled_500m.tif')) as resamp_src:
        ref_ras = resamp_src.read(1)    
        ref_trans=resamp_src.transform
        ref_crs=resamp_src.crs
        
    #i=1
    cov_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)
    for i in cuq:
        try:
            if ~np.isnan(i):
                print('')
                print('Pass '+str(pass_no))
                print(f"Processing class {i}")
                
                existing_samples = sum(pts_updated['class_pca'] == i)
                target_locs = (cluster_raster == i) & (gdm_raster_mask)
                ref_locs = (cluster_raster == i) & (ref_ras == 1) & (gdm_raster_mask)
                rows, cols = np.where(target_locs)
                ref_rows, ref_cols = np.where(ref_locs)
                ref_indices = np.column_stack((ref_rows, ref_cols))
    
                if len(ref_rows) > 0:
                    print(f"Pixels available for sampling: {ref_rows.size}")
                    # Extract point coordinates
                    pts_coords = [(geom.x, geom.y) for geom in pts_updated[pts_updated['class_pca']==i].geometry]
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
                        existing_hull_volume = convex_hull_volume(existing_sites_pca, 100)
                        candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
                        #expand the hull of current sites by 10% volume to capture more diverse potential new sites
                        d = existing_sites_pca.shape[1]  # number of dimensions
                        centroid = np.mean(existing_sites_pca, axis=0)                                 
                        scale_factor = (sfactor) ** (1 / d)
                        expanded_existing_sites_pca = centroid + (existing_sites_pca - centroid) * scale_factor
                        hull = ConvexHull(remove_outliers(expanded_existing_sites_pca, 100))
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
                                transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
                                new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]
                                #candidate points outside min distance with existing sites
                                global_distances, _ = global_tree.query(new_sites_pro)
                                new_sites=np.array(new_sites_pro)[global_distances>min_distance]
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
                                        transformer = Transformer.from_crs("EPSG:3577", pts.crs, always_xy=True)
                                        new_sites_filt_geo=[transformer.transform(lon, lat) for lon, lat in new_sites_filt]
                                        #len(new_sites_filt_geo)
                                        batch_size = 50
                                        batches = [new_sites_filt_geo[i:i + batch_size] for i in range(0, len(new_sites_filt_geo), batch_size)]
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
                                                outside_mask=outside_mask & (min_distances>np.nanpercentile(min_distances[outside_mask], 50)) 
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
                                                    expanded_combined_hull=ConvexHull(remove_outliers(expanded_new_sites_filt_pca, 99))
                                                combined_hull_volume = combined_hull.volume
                                                new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                                            else:
                                                print('Already reached target coverage')
                                        new_df = pd.DataFrame(pts_new, columns=['longitude', 'latitude'])
                                        new_df['source'] = ref_fn.split('\\')[-1]
                                        new_df['class_pca'] = i
                                        new_gdf = gpd.GeoDataFrame(new_df, 
                                                               geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                               crs='EPSG:4326') 
                                        print('Adding '+str(len(new_gdf))+' points')
                                        print('New coverage: '+str(new_coverage_ratio*100)[0:4]+'%')
                                        pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_pca', 'geometry']]], ignore_index=True)
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
                        transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
                        new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]
                        #candidate points outside min distance with existing sites
                        global_distances, _ = global_tree.query(new_sites_pro)
                        new_sites=np.array(new_sites_pro)[global_distances>min_distance]
                        if len(new_sites)>0:
                            mmd_batch=max(int(len(new_sites)*simplifier), 1)
                            max_it=int((len(new_sites)/mmd_batch)*1.1)
                            new_sites_filt=maximize_min_distance(new_sites, min_distance, mmd_batch, max_iterations=max_it)
                            print(f"Points meeting distance criteria: {new_sites_filt.shape[0]}")
                            if len(new_sites_filt)>0:
                                transformer = Transformer.from_crs("EPSG:3577", pts.crs, always_xy=True)
                                new_sites_filt_geo=np.array([transformer.transform(lon, lat) for lon, lat in new_sites_filt])
                                new_sites_filt_geo = new_sites_filt_geo[np.random.choice(len(new_sites_filt_geo), size=min(len(new_sites_filt_geo), start_seed), replace=False)]
                                #check nan in new sites
                                rowcols_check = [rowcol(ref_trans, x, y) for x, y in new_sites_filt_geo]
                                nan_mask = np.array([~np.any(np.isnan(gdm_raster[row, col, :])) for row, col in rowcols_check])
                                new_sites_filt_geo = new_sites_filt_geo[nan_mask]
                                new_df = pd.DataFrame(new_sites_filt_geo, columns=['longitude', 'latitude'])
                                new_df['source'] = ref_fn.split('\\')[-1]
                                new_df['class_pca'] = i
                                new_gdf = gpd.GeoDataFrame(new_df, 
                                                       geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                       crs='EPSG:4326') 
                                print('Adding '+str(len(new_gdf))+' points')
                                pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_pca', 'geometry']]], ignore_index=True)
                                #update global tree
                                global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry.to_crs("EPSG:3577")])
                                
                                #run again
                                if len(new_sites_filt)>15:
                                    pts_coords = [(geom.x, geom.y) for geom in pts_updated[pts_updated['class_pca']==i].geometry]
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
                                    existing_hull_volume = convex_hull_volume(existing_sites_pca, 100)
                                    candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
                                    #expand the hull of current sites by 10% volume to capture more diverse potential new sites
                                    d = existing_sites_pca.shape[1]  # number of dimensions
                                    centroid = np.mean(existing_sites_pca, axis=0)                                 
                                    scale_factor = (sfactor) ** (1 / d)
                                    expanded_existing_sites_pca = centroid + (existing_sites_pca - centroid) * scale_factor
                                    hull = ConvexHull(remove_outliers(expanded_existing_sites_pca, 100))
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
                                            transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
                                            new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]
                                            #candidate points outside min distance with existing sites
                                            global_distances, _ = global_tree.query(new_sites_pro)
                                            new_sites=np.array(new_sites_pro)[global_distances>min_distance]
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
                                                    transformer = Transformer.from_crs("EPSG:3577", pts.crs, always_xy=True)
                                                    new_sites_filt_geo=[transformer.transform(lon, lat) for lon, lat in new_sites_filt]
                                                    #len(new_sites_filt_geo)
                                                    batch_size = 50
                                                    batches = [new_sites_filt_geo[i:i + batch_size] for i in range(0, len(new_sites_filt_geo), batch_size)]
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
                                                            outside_mask=outside_mask & (min_distances>np.nanpercentile(min_distances[outside_mask], 50)) 
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
                                                                expanded_combined_hull=ConvexHull(remove_outliers(expanded_new_sites_filt_pca, 99))
                                                            combined_hull_volume = combined_hull.volume
                                                            new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                                                        else:
                                                            print('Already reached target coverage')
                                                    new_df = pd.DataFrame(pts_new, columns=['longitude', 'latitude'])
                                                    new_df['source'] = ref_fn.split('\\')[-1]
                                                    new_df['class_pca'] = i
                                                    new_gdf = gpd.GeoDataFrame(new_df, 
                                                                           geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                                           crs='EPSG:4326') 
                                                    print('Adding '+str(len(new_gdf))+' points')
                                                    print('New coverage: '+str(new_coverage_ratio*100)[0:4]+'%')
                                                    pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_pca', 'geometry']]], ignore_index=True)
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
#export
print('Number of reference points: '+str(len(pts_updated)))
pts_updated.to_file(scrap_dir+'pts_updated_500m_v11_1kmspacing_20volexp.shp')

#%%

No reference sites available at level 0

Pass 0
Processing class 460.0
Pixels available for sampling: 180
Not enough existing sites to generate hull
59 / 89 max. iterations
Points meeting distance criteria: 22
Adding 21 points
Coverage ratio: 12.30%
Candidate sites: 180
Points outside convex hull: 128
1 / 2 max. iterations
Points meeting distance criteria: 1
Traceback (most recent call last):

  File "C:\Users\mattg\AppData\Local\Temp/ipykernel_31332/1205628890.py", line 271, in <module>
    batch_filt_pca = pca.transform(batch_gdm)

  File "C:\Users\mattg\anaconda3\envs\fire_severity\lib\site-packages\sklearn\decomposition\_base.py", line 117, in transform
    X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)

  File "C:\Users\mattg\anaconda3\envs\fire_severity\lib\site-packages\sklearn\base.py", line 561, in _validate_data
    X = check_array(X, **check_params)

  File "C:\Users\mattg\anaconda3\envs\fire_severity\lib\site-packages\sklearn\utils\validation.py", line 797, in check_array
    raise ValueError(

ValueError: Found array with 0 sample(s) (shape=(0, 10)) while a minimum of 1 is required.


"""
Finish the cover raster

"""

#pts_updated=pts

pass_no=3
ref_fn = ref_dirs[pass_no]
print(ref_fn)
if os.path.isfile(ref_fn.replace('.tif', '_resampled_90m.tif'))==False:
    print('Resampling '+ref_fn)
    resample_raster(ref_fn, cluster_dir, ref_fn.replace('.tif', '_resampled_90m.tif'))
with rio.open(ref_fn.replace('.tif', '_resampled_90m.tif')) as resamp_src:
    ref_ras = resamp_src.read(1)    
    ref_trans=resamp_src.transform
    ref_crs=resamp_src.crs
cov_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)
for i in np.unique(cluster_raster):
    if ~np.isnan(i):
        print('')
        print(f"Processing class {i}")

        existing_samples = sum(pts_updated['class_pca'] == i)
        target_locs = (cluster_raster == i)
        ref_locs = (cluster_raster == i) & (ref_ras == 1)
        rows, cols = np.where(target_locs)
        ref_rows, ref_cols = np.where(ref_locs)

        if len(rows) > 0:
            print(f"Pixels available for sampling: {rows.size}")
            # Extract point coordinates
            pts_coords = [(geom.x, geom.y) for geom in pts_updated[pts_updated['class_pca']==i].geometry]
            rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
            existing_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in rowcols])
            candidate_sites_gdm = gdm_raster[target_locs, :]
            ref_sites = gdm_raster[ref_locs, :]
            #~15 sites min to generate hull after outlier removal
            if (existing_sites_gdm.shape[0] > 15) & (len(ref_sites)>1):
                existing_sites_pca = pca.transform(existing_sites_gdm)  
                candidate_sites_pca = pca.transform(candidate_sites_gdm)  
                ref_sites_pca = pca.transform(ref_sites)
                ref_indices = np.column_stack((ref_rows, ref_cols))
                if len(candidate_sites_pca)>500000:
                    candidate_sites_pca = candidate_sites_pca[np.random.choice(candidate_sites_pca.shape[0], size=500000, replace=False)]
                existing_hull_volume = convex_hull_volume(existing_sites_pca, 99)
                candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
                hull = ConvexHull(remove_outliers(existing_sites_pca, 99))
                outside_mask  = points_outside_hull(combined_hull, batch_filt_pca)
                #initialise the combined hull, which gets updated in batch for loop
                combined_hull=hull                        
                if candidate_hull_volume > 0:
                    coverage_ratio = existing_hull_volume / candidate_hull_volume
                    print(f"Coverage ratio: {coverage_ratio:.2%}")
                    cov_raster[cluster_raster == i] = coverage_ratio  
                    
with rio.open(scrap_dir+'cov_raster_21_90m_50thp_1km.tif', 'w', driver='GTiff',
                   count=1, dtype='float32', crs=cluster_src.crs,
                   transform=src.transform, width=cluster_src.width, height=src.height) as dst:
    dst.write(cov_raster, 1)


#%%

"""
plot animation of convex hull expansion

"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import numpy as np

def points_outside_hull(hull, points):
    points = points[:, :3]  # Use only first 3 PCA dimensions
    del_hull = Delaunay(hull.points[:, :3])
    return del_hull.find_simplex(points) < 0




for i in np.unique(cluster_raster):
    if ~np.isnan(i):
        print(f"Processing class {i}")

        existing_samples = sum(pts['class_pca'] == i)
        target_locs = (cluster_raster == i)
        ref_locs = (cluster_raster == i) & (ref_ras == 1)
        rows, cols = np.where(target_locs)
        ref_rows, ref_cols = np.where(ref_locs)

        if len(rows) > 0:
            print(f"Pixels available for sampling: {rows.size}")
            # Extract point coordinates
            pts_coords = [(geom.x, geom.y) for geom in pts[pts['class_pca']==i].geometry]
            rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
            existing_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in rowcols])
            candidate_sites_gdm = gdm_raster[target_locs, :]
            ref_sites = gdm_raster[ref_locs, :]
            if (existing_sites_gdm.shape[0] > 1) & (len(ref_sites)>1):
                existing_sites_pca = pca.transform(existing_sites_gdm)  
                candidate_sites_pca = pca.transform(candidate_sites_gdm)  
                ref_sites_pca = pca.transform(ref_sites)
                ref_indices = np.column_stack((ref_rows, ref_cols))
                if len(candidate_sites_pca)>500000:
                    candidate_sites_pca = candidate_sites_pca[np.random.choice(candidate_sites_pca.shape[0], size=500000, replace=False)]
                existing_hull_volume = convex_hull_volume(existing_sites_pca, 99)
                candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
                if existing_hull_volume>0:
                    hull = ConvexHull(existing_sites_pca)
                    outside_mask = points_outside_hull(hull, ref_sites_pca)
                    #initialise the combined hull, which gets updated in batch for loop
                    combined_hull=hull
                else:
                    outside_mask=np.zeros(len(ref_sites_pca))==0
                    
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
                        transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
                        new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]
                        #candidate points outside min distance with existing sites
                        global_distances, _ = global_tree.query(new_sites_pro)
                        new_sites=np.array(new_sites_pro)[global_distances>min_distance]
                        if len(new_sites)>0:
                            tree = KDTree(new_sites)
                            init_distances, indices = tree.query(new_sites, k=2)
                            init_distances=init_distances[:,1]
                            valid_indices = np.where((init_distances > np.nanpercentile(init_distances, 50)) | (
                                (init_distances > min_distance)))[0]
                            distances, _ = tree.query(np.array(new_sites)[valid_indices], k=2)
                            distances=distances[:,1]
                            new_sites_filt=new_sites[valid_indices][distances>min_distance]
                            if len(new_sites_filt)>0:
                                #process batches, building the kdtree
                                #re-evaluate the convex hull
                                transformer = Transformer.from_crs("EPSG:3577", pts.crs, always_xy=True)
                                new_sites_filt_geo=[transformer.transform(lon, lat) for lon, lat in new_sites_filt]
                                len(new_sites_filt_geo)
                                batch_size = 50
                                batches = [new_sites_filt_geo[i:i + batch_size] for i in range(0, len(new_sites_filt_geo), batch_size)]
                                pts_new=[]
                                new_coverage_ratio=coverage_ratio
                                counter=0
                                def points_outside_hull(hull, points):
                                   # Ensure both hull and points are in 3D space
                                   points = points[:, :3]  # Use only the first 3 PCA dimensions
                                   del_hull = Delaunay(hull.points[:, :3])  # Same here for the convex hull
                                   return del_hull.find_simplex(points) < 0

                                # Start the batch loop
                                for batch in batches:
                                    counter += 1
                                    
                                    # Add if still outside hull
                                    batch_rowcols = [rowcol(ref_trans, x, y) for x, y in batch]
                                    batch_gdm = np.array([gdm_raster[row, col, :] for row, col in batch_rowcols])
                                    batch_filt_pca = pca.transform(batch_gdm)
                                    outside_mask = points_outside_hull(combined_hull, batch_filt_pca)
                                    outside_points = batch_filt_pca[outside_mask]
                                    outside_indices = np.array(batch)[outside_mask]
                                    batch = [tuple(coord) for coord in outside_indices]
                                    
                                    print(f'Batch {counter} - {len(batch)} still outside hull')
                                    
                                    # Add points to list
                                    pts_coords.extend(batch)
                                    pts_new.extend(batch)
                                    
                                    # Calculate new % volume
                                    new_sites_rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                                    new_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in new_sites_rowcols])
                                    new_sites_filt_pca = pca.transform(new_sites_gdm)
                                    combined_hull = ConvexHull(new_sites_filt_pca[:, :3])  # Use only first 3 PCA dimensions
                                    combined_hull_volume = combined_hull.volume
                                    new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                                    
                                    # Visualization
                                    fig = plt.figure(figsize=(10, 8))  # Increased figure size
                                    ax = fig.add_subplot(111, projection='3d')
                                    ax.set_xlabel('PC1')
                                    ax.set_ylabel('PC2')
                                    ax.set_zlabel('PC3')
                                
                                    # Set axis limits to be consistent across batches
                                    ax.set_xlim(new_sites_filt_pca[:, 0].min(), new_sites_filt_pca[:, 0].max())
                                    ax.set_ylim(new_sites_filt_pca[:, 1].min(), new_sites_filt_pca[:, 1].max())
                                    ax.set_zlim(new_sites_filt_pca[:, 2].min(), new_sites_filt_pca[:, 2].max())
                                
                                    for simplex in combined_hull.simplices:
                                        ax.plot_trisurf(new_sites_filt_pca[simplex, 0],
                                                        new_sites_filt_pca[simplex, 1],
                                                        new_sites_filt_pca[simplex, 2],
                                                        color='orange', alpha=0.3)
                                    
                                    plt.show()  # Display each figure separately
#%%

plt.show()  # Final static plot

                                 for batch in batches:
                                     if (new_coverage_ratio < cov_threshold):
                                         counter=counter+1
                                         #add if still outside hull
                                         batch_rowcols = [rowcol(ref_trans, x, y) for x, y in batch]
                                         batch_gdm = np.array([gdm_raster[row, col, :] for row, col in batch_rowcols])
                                         batch_filt_pca = pca.transform(batch_gdm)
                                         outside_mask = points_outside_hull(combined_hull, batch_filt_pca)
                                         outside_points = batch_filt_pca[outside_mask]
                                         outside_indices = np.array(batch)[outside_mask]
                                         batch=[tuple(coord) for coord in outside_indices]
                                         print('Batch '+str(counter)+' - '+str(len(batch))+' still outside hull')
                                         #add pts in batch outside of convex hull
                                         pts_coords.extend(batch)
                                         pts_new.extend(batch)
                                         #calculate new %volume
                                         new_sites_rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                                         new_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in new_sites_rowcols])
                                         new_sites_filt_pca = pca.transform(new_sites_gdm)
                                         combined_hull = ConvexHull(new_sites_filt_pca)
                                         combined_hull_volume = combined_hull.volume
                                         new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                                     else:
                                         print('Already reached target coverage')
                                 new_df = pd.DataFrame(pts_new, columns=['longitude', 'latitude'])
                                 new_df['source'] = ref_fn.split('\\')[-1]
                                 new_df['class_pca'] = i
                                 new_gdf = gpd.GeoDataFrame(new_df, 
                                                        geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                                        crs='EPSG:4326') 
                                 print('Adding '+str(len(new_gdf))+' points')
                                 pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_pca', 'geometry']]], ignore_index=True)
                                 #update global tree
                                 global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry.to_crs("EPSG:3577")])    
                         else:
                             print('No sites matching distance criteria')
                     else:
                         print('No sites outside of existing hull')



                            #the problem is that each new batch of points needs to be reevaluated as lying outside of the new hull
                                    
                                
                            num_new_sites = 100
                            new_sites_indices = np.random.choice(outside_points.shape[0], size=num_new_sites, replace=False)
                            new_sites_rowcols = outside_indices[new_sites_indices]
                            new_sites_geo = [ref_trans * (col, row) for row, col in new_sites_rowcols]
                            transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
                            new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]

                           
                            candidate_coords_t = candidate_coords_t[valid_indices]
                            print(f"Adding {num_new_sites} new sites for class {i}")
                            new_coords_batch = []
                            counter = 0
                            for idx in new_sites_geo:
                                coord=tuple(idx)
                                if len(sample_coords) > 1:
                                    # Check if the new point satisfies the minimum distance condition
                                    if len(tree.query_ball_point(coord, min_distance)) == 0:
                                        if all(np.linalg.norm(np.array(coord) - np.array(existing)) > min_distance for existing in new_coords_batch):
                                            new_coords_batch.append(coord)  # Add only if it passes both distance checks
                                            counter += 1
                                else:
                                    # For the first point, there is no KDTree to check against
                                    new_coords_batch.append(coord)
                                    counter += 1
                                # Process the batch when it reaches a certain size or if strat_size is met
                                if len(new_coords_batch) >= 100 or counter >= strat_size/2:
                                    sample_coords.extend(new_coords_batch)  # Add all valid points in the batch
                                    tree = KDTree(sample_coords)  # Rebuild KDTree with the updated points
                                    new_coords_batch = []  # Reset the batch
                                # Stop adding points if we've reached the target count
                                if counter >= strat_size/2:
                                    break
                            sample_coords.extend(new_coords_batch)  # Add all valid points in the batch
                            tree = KDTree(sample_coords)  # Rebuild KDTree with the updated points
                            sample_coords_new=sample_coords[len(pts2):]
                            if len(sample_coords_new)>1:
                                new_pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in sample_coords_new], crs="EPSG:3577")
                                print('Appending '+str(len(new_pts))+' points to geodataframe...')
                                new_pts['source']=ref_fn.split('\\')[-1]
                                new_pts['class_pca']=i
                                pts2 = pd.concat([pts2, new_pts], ignore_index=True)
                            
                            
                            
                            
                    # Update cov_raster (or another output) with new coverage info
                    cov_raster[cluster_raster == i] = len(new_sites_rowcols) / target_size


                    # Convert raster row/col to geographic coordinates
                    new_sites_geo = [ref_trans * (col, row) for row, col in new_sites_rowcols]

                    print(f"Adding {num_new_sites} new sites for class {i}")
                    # Update cov_raster (or another output) with new coverage info
                    cov_raster[cluster_raster == i] = len(new_sites_rowcols) / target_size

                            
                            new_sites_indices = [np.argmin(np.linalg.norm(candidate_sites_pca - p, axis=1)) for p in new_sites]
                            new_sites_rowcols = [(rows[idx], cols[idx]) for idx in new_sites_indices]
                            new_sites_geo = [ref_trans * (col, row) for row, col in new_sites_rowcols]

                            
                        print('')
                    else:
                        print("Candidate hull volume is zero — cannot compute coverage.")
                else:
                    explained_variance = 0  # If no samples yet, start from 0

                print(f"Existing convex hull coverage (after 95th percentile removal): {coverage:.2f}")

with rio.open(scrap_dir+'cov_raster_25.tif', 'w', driver='GTiff',
                   count=1, dtype='float32', crs=cluster_src.crs,
                   transform=src.transform, width=cluster_src.width, height=src.height) as dst:
    dst.write(cov_raster, 1)
    

                if explained_variance >= 0.95:
                    print(f"Skipping class {i}, variance already covered.")
                    continue  # Skip to next class

                # Select new sites based on distance and weighting
                distances, _ = tree.query(candidate_coords_t)
                valid_indices = np.where(distances > min_distance)[0]
                candidate_coords_t = candidate_coords_t[valid_indices]

                # Weight selection towards points further from existing samples
                weights = 1 / (distances[valid_indices] + 1e-6)
                weights /= weights.sum()
                sorted_indices = np.argsort(-weights)  # Sort descending

                candidate_coords_t = candidate_coords_t[sorted_indices]
                weights = weights[sorted_indices]

                # Add points iteratively until 95% variance is reached
                new_coords_batch = []
                counter = 0
                for idx in candidate_coords_t:
                    coord = tuple(idx)

                    # Check if the new point maintains min_distance from others
                    if len(tree.query_ball_point(coord, min_distance)) == 0:
                        if all(np.linalg.norm(np.array(coord) - np.array(existing)) > min_distance for existing in new_coords_batch):
                            new_coords_batch.append(coord)
                            counter += 1

                    # Check variance coverage after adding each batch
                    if len(new_coords_batch) >= 50 or counter >= target_size / 2:
                        sample_coords.extend(new_coords_batch)
                        tree = KDTree(sample_coords)
                        new_coords_batch = []

                        # Update variance coverage
                        selected_sites_gdm = gdm_raster[np.isin(sample_coords, pts2.geometry), :]
                        explained_variance = pca_variance_coverage(selected_sites_gdm, candidate_sites_gdm)

                        print(f"Updated variance coverage: {explained_variance:.2f}")

                        if explained_variance >= 0.95:
                            break  # Stop adding points if variance coverage is met

                sample_coords.extend(new_coords_batch)
                tree = KDTree(sample_coords)

                # Add new points to geodataframe
                sample_coords_new = sample_coords[len(pts2):]
                if len(sample_coords_new) > 1:
                    new_pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in sample_coords_new], crs="EPSG:3577")
                    print(f"Appending {len(new_pts)} points to geodataframe...")
                    new_pts['source'] = ref_fn.split('\\')[-1]
                    new_pts['class_pca'] = i
                    pts2 = pd.concat([pts2, new_pts], ignore_index=True)

            else:
                print(f"No pixels available for class {i}. Moving to next class.")

print("Final site selection complete.")


#%%

pts2=pts.to_crs("EPSG:3577")    
counts = pts2['class_pca'].value_counts()
print(counts)
#target_size=int(np.nanpercentile(counts, 90))
#target_size=2000
sample_coords = [geom.coords[0] for geom in pts2.geometry]
min_distance=1000

if len(sample_coords) > 0:
    tree = KDTree(sample_coords)
else:
    tree = KDTree(np.empty((0, 2))) 

pass_nos=range(0, 4)
#pass_no=1
for pass_no in pass_nos:
    print('')
    print('Processing reference site area level: '+str(pass_no))
    ref_fn=ref_dirs[pass_no]
    print(ref_fn)
    with rio.open(ref_fn) as ref_src:
        ref_ras = ref_src.read(1)    
    #i=22
    for i in np.unique(cluster_raster):
        if ~np.isnan(i):
            print(i)
            target_size=weighted_samples[weighted_samples['Class']==i]['Weighted_Samples'].values[0]
            strat_size=(target_size-sum(pts2['class_pca']==i))*2
            print('Maximum samples required: '+str(strat_size/2))
            if strat_size>0:
                target_locs=(cluster_resamp==i) & (ref_ras==1)
                #np.sum(target_locs)
                rows, cols = np.where(target_locs)
                if (len(rows)>0):
                    print('Pixels for random sampling: '+str(rows.size))
                    random.seed(None)
                    if len(rows)>strat_size*1000:
                        sample_indices = random.sample(range(len(rows)), strat_size*1000)
                    else:
                        if len(rows)>strat_size*10:
                            sample_indices = random.sample(range(len(rows)), strat_size*10)
                        else:
                            if len(rows)>strat_size:
                                sample_indices = random.sample(range(len(rows)), strat_size)
                            else:
                                sample_indices = random.sample(range(len(rows)), len(rows))
                    counter=0
                    new_coords_batch = []
                    #inverse distance weighting priority for further points from existing points
                    candidate_coords = [(cols[idx], rows[idx]) for idx in sample_indices]
                    candidate_coords_t = np.array([resamp_trans * (c, r) for c, r in candidate_coords])
                    distances, _ = tree.query(candidate_coords_t)
                    candidate_coords_t=candidate_coords_t[np.where(distances > min_distance)[0]]
                    weights=1/(distances[np.where(distances > min_distance)[0]] +1e-6)
                    weights /= weights.sum()
                    sorted_indices = np.argsort(-weights)  # Negative sign to sort descending
                    candidate_coords_t = candidate_coords_t[sorted_indices]
                    weights = weights[sorted_indices] 
                    for idx in candidate_coords_t:
                        coord=tuple(idx)
                        if len(sample_coords) > 1:
                            # Check if the new point satisfies the minimum distance condition
                            if len(tree.query_ball_point(coord, min_distance)) == 0:
                                if all(np.linalg.norm(np.array(coord) - np.array(existing)) > min_distance for existing in new_coords_batch):
                                    new_coords_batch.append(coord)  # Add only if it passes both distance checks
                                    counter += 1
                        else:
                            # For the first point, there is no KDTree to check against
                            new_coords_batch.append(coord)
                            counter += 1
                        # Process the batch when it reaches a certain size or if strat_size is met
                        if len(new_coords_batch) >= 100 or counter >= strat_size/2:
                            sample_coords.extend(new_coords_batch)  # Add all valid points in the batch
                            tree = KDTree(sample_coords)  # Rebuild KDTree with the updated points
                            new_coords_batch = []  # Reset the batch
                        # Stop adding points if we've reached the target count
                        if counter >= strat_size/2:
                            break
                    sample_coords.extend(new_coords_batch)  # Add all valid points in the batch
                    tree = KDTree(sample_coords)  # Rebuild KDTree with the updated points
                    sample_coords_new=sample_coords[len(pts2):]
                    if len(sample_coords_new)>1:
                        new_pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in sample_coords_new], crs="EPSG:3577")
                        print('Appending '+str(len(new_pts))+' points to geodataframe...')
                        new_pts['source']=ref_fn.split('\\')[-1]
                        new_pts['class_pca']=i
                        pts2 = pd.concat([pts2, new_pts], ignore_index=True)
                else:
                    print('No pixels for sampling')

            
pts2.to_file(scrap_dir+'\\new_pts_test13.shp')   

#%%

class_count(cluster_raster, pts2, scrap_dir+'count_raster_25.tif')

unique_classes, class_counts = np.unique(cluster_raster, return_counts=True)

# Step 2: Create a mapping of class -> count
class_count_map = dict(zip(unique_classes, class_counts))

# Step 3: Replace each pixel in the cluster_raster with the count of its class
count_raster = np.vectorize(class_count_map.get)(cluster_raster)


#%%

    #sample KDE values for samples points
        pts_kde_sampled = kde_sample_pts(clim_kde_path, pts_sampled, 'clim_kde')
        pts_kde_sampled = kde_sample_pts(soil_kde_path, pts_kde_sampled, 'soil_kde')
        #convert to 3577 before adding new coords
        pts_kde_sampled=pts_kde_sampled.to_crs("EPSG:3577")
        
        #Climate:
        #resample KDE to match higher-resolution sample-able area raster
        clim_resampled_kde, uint_fac=resample_kde(clim_kde_path, pass_path)
        #generate int8 high-res raster representing quantiles of clim PCA, to stratify additional samples
        clim_quant_resampled=resample_quant(clim_quant_path, pass_path)
        #pre-determine how many new samples to make
        additional_samples_needed, bin_edges=required_samples(clim_kde_raster, p_threshold, 
                                                              n_bins, frac_samples_clim, pts_kde_sampled)
        bin_edges=bin_edges*uint_fac
    
        #add new samples to geodataframe
        if pass_no==1:
            pts_kde_sampled['source']='HCAS'
        pts=add_samples(additional_samples_needed, pts_kde_sampled, clim_resampled_kde, 
                        n_bins, bin_edges, th1, th2, frac_samples_clim, pass_path, clim_quant) 
        del clim_resampled_kde
        
        #Soil:  
        #resample KDE to match higher-resolution sample-able area raster
        soil_resampled_kde, uint_fac=resample_kde(soil_kde_path, pass_path)
        #generate int8 high-res raster representing quantiles of clim PCA, to stratify additional samples
        soil_quant_resampled=resample_quant(soil_quant_path, pass_path)
        #pre-determine how many new samples to make
        additional_samples_needed, bin_edges=required_samples(soil_kde_raster, p_threshold, 
                                                              n_bins, frac_samples_soil, pts_kde_sampled)
        bin_edges=bin_edges*uint_fac
        #add new samples to geodataframe
        pts=add_samples(additional_samples_needed, pts, soil_resampled_kde, 
                        n_bins, bin_edges, th1, th2, frac_samples_soil, pass_path, soil_quant) 
        del soil_resampled_kde



#%%

pts.to_file(wdir+'\\scrap\\ref_test_'+str(frac_samples_clim)+'_'+str(frac_samples_soil)+'.shp')


#%%

"""
Average heirarchy per class

"""

pts=pd.read_csv(wdir+'scrap\\pts_updated_test21_90m_50thp_1km_sampled_v2.csv')
pts['class_pca']
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




