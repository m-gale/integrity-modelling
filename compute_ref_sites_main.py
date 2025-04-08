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


#%%

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
scrap_dir=wdir+'scrap\\'

#integrity hierarchy layers
ref_dir='F:\\veg2_postdoc\\data\\reference\\National\\V1\\'
ref_dirs=glob.glob(ref_dir+'*v1.tif')


#%%

"""
Load classes, pts

"""

#ecotype classification
cluster_dir=scrap_dir+'cluster_raster46_s_simplified_500m.tif'
with rio.open(cluster_dir) as src:
    cluster_raster=src.read(1)

#%%
"""
Starter seed for points

"""

#build on HCAS points...
obs_path=wdir+'data\\HCAS_ref_sites\\HCAS_2.3\\data\\0.Inferred_Reference_Sites\\HCAS23_RC_BenchmarkSample_NSW_clipped.shp'
pts = gpd.read_file(obs_path)
pts=pts.to_crs(src.crs)
coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]
with rio.open(cluster_dir) as cluster_src:
    class_values = list(cluster_src.sample(coords))
class_values = [val[0] if val or val == 0 else np.nan for val in class_values]
pts['class_pca']=class_values

#...or start from scratch
pts=pts.iloc[0:0]
coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]


#%%

"""
Load GDM layers

"""

gdm_dir='F:\\veg2_postdoc\\data\\reference\\National\\mokany_etal_GDM_layers\\data\\250m\\'
gdm_fns=glob.glob(gdm_dir+'*9s.tif')
len(gdm_fns)
gdm_fns=glob.glob(gdm_dir+'*resampled_500m.tif')

gdm_layers = []
for fn in gdm_fns[:10]:  # take the first 10
    print(fn)
    with rio.open(fn) as src:
        gdm_layers.append(src.read(1))

# Convert list to 3D NumPy array (shape: height x width x 10)
gdm_raster = np.stack(gdm_layers, axis=-1)  
del gdm_layers

#mask so that nan values are not selected as rows, cols
gdm_raster[gdm_raster == -9999] = np.nan
gdm_raster_mask=~np.isnan(gdm_raster).any(axis=2)

#%%

"""
Fit PCA to GDM
So the convex hull is calculated in five dimensions instead of 10

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
Run parameters

"""

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

#%%

"""
Initialise

"""

#initialise distance tree
if len(sample_coords) > 0:
    global_tree = KDTree(sample_coords)
else:
    global_tree = KDTree(np.empty((0, 2))) 

pts_updated=copy.deepcopy(pts)

cuq=np.unique(cluster_raster)

error_log=[]

#%%

"""
Do the thing

"""

# Process through ecosystem integrity levels
#pass_no=0
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

"""

Export
"""

print('Number of reference points: '+str(len(pts_updated)))
pts_updated.to_file(scrap_dir+'pts_updated_500m_v11_1kmspacing_20volexp.shp')


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




