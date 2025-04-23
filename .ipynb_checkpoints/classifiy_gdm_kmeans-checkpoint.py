# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:53:44 2025

@author: mattg




To do:

* Put the low-variation class combination into a function and implement it twice. 

"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import geopandas as gpd
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
from glob import glob
from scipy.ndimage import median_filter
from heapq import heappop, heappush
import os


#%%

#location of GDM transformed vars of Mokany et al. 2022
#local
#wdir='F:\\veg2_postdoc\\data\\reference\\National\\mokany_etal_GDM_layers\\data\\250m\\'
#scrap_dir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'

#nci
wdir='/g/data/xc0/project/natint/input/classify_gdm_kmeans_input/GDM_250m/'
out_dir='/g/data/xc0/project/natint/output/v1/gdm_kmeans'
if os.path.exists(out_dir)==False:
    os.mkdir(out_dir)

gdm_fns=glob(wdir+'*.tif')
print(str(len(gdm_fns))+' files found')

#%%


"""
Read resampled GDM layers
"""

with rasterio.open(gdm_fns[0]) as pc1_src:
    gdm1 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[1]) as pc1_src:
    gdm2 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[2]) as pc1_src:
    gdm3 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[3]) as pc1_src:
    gdm4 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[4]) as pc1_src:
    gdm5 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[5]) as pc1_src:
    gdm6 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[6]) as pc1_src:
    gdm7 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[7]) as pc1_src:
    gdm8 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[8]) as pc1_src:
    gdm9 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[9]) as pc1_src:
    gdm10 = pc1_src.read(1)  
    nodata_value=pc1_src.nodata


#%%

"""
Pixels to dataframe

"""

#weight the the soil some more
gdm8=gdm8*3
gdm9=gdm9*3

pixels = np.vstack((gdm1.flatten(), gdm2.flatten(),gdm3.flatten(),gdm4.flatten(),gdm5.flatten(),
                    gdm6.flatten(),gdm7.flatten(),gdm8.flatten(),gdm9.flatten(),gdm10.flatten(),)).T

del gdm2, gdm3, gdm4, gdm5, gdm6, gdm7, gdm8, gdm9, gdm10 

valid_mask = (~np.isnan(pixels).any(axis=1)) & (~np.isinf(pixels).any(axis=1)) & (~(pixels == nodata_value).any(axis=1))
pixels = pixels[valid_mask]  # Keep only valid pixels

#convert to int for computational reasons
pixels = (pixels * 1000).astype('int16')
#np.nanmin(pixels)
#np.nanmax(pixels)

#%%

"""
k-means clustering on the valid pixels

"""

#version no.
vno='1'

#no. clusters
k = 1000  

#small bach size gives faster processing via some approximation.
#over 10000 seems pretty stable
batch_size=10240

#fit
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
kmeans.fit(pixels)

#export centroids, if required
centroids = kmeans.cluster_centers_  # Centroids of the 30 clusters from k-means
centroids_df = pd.DataFrame(centroids, columns=[f'Feature_{i+1}' for i in range(centroids.shape[1])])
centroids_df['Cluster'] = np.arange(1, len(centroids) + 1)
centroids_df.to_csv(out_dir+"/centroids"+vno+".csv", index=False)


#%%

"""
Export initial class raster
"""

labels = kmeans.labels_

# Create the clusters array with the same shape as the resampled climate PCA raster
clusters = np.full(gdm1.shape, np.nan)

# Assign the cluster labels to the valid positions using the valid mask
# Note: valid_mask only contains valid indices where NaN pixels were removed
clusters_flat = clusters.flatten()
clusters_flat[valid_mask] = labels

# Reshape the cluster array back to the 2D raster shape
cluster_raster = clusters_flat.reshape(gdm1.shape)

output_path = out_dir + '/cluster_raster'+vno+'_s_original.tif'
# Retrieve the transform and metadata from the resampled climate PCA raster
with rasterio.open(gdm_fns[9]) as src:
    transform = src.transform
    crs = src.crs

metadata = {
    'driver': 'GTiff',
    'count': 1,  
    'dtype': 'float32', 
    'crs': crs, 
    'transform': transform,  
    'width': cluster_raster.shape[1], 
    'height': cluster_raster.shape[0], 
    'nodata': np.nan  
}

# Apply a 3x3 median filter
smoothed_raster = median_filter(cluster_raster, size=3)

# Write the cluster labels to the new raster file
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(smoothed_raster, 1)  


#%%

"""
Simplify the original by combining classes with close centroids
Until we end up with a desired number of classes
"""

#target number of clusters
final_k = k/2 

#extract the elevation roughness centroids
elevation_centroids = centroids[:, 0] 
#extract the plant growth index centroids
pg_centroids = centroids[:, 5]

#only simplify classes where plant growth index and elevation range are low
elevation_threshold = np.percentile(elevation_centroids, 80)
pg_threshold = np.percentile(pg_centroids, 60)

#initial pairwise distances
distances = cdist(centroids, centroids, metric='euclidean')
np.fill_diagonal(distances, np.inf)

merge_candidates = np.where((elevation_centroids <= elevation_threshold) & 
                            (pg_centroids <= pg_threshold))[0]
merge_candidates_set = set(merge_candidates)

# Priority queue for merging (min heap)
heap = []
for i in merge_candidates:
    for j in merge_candidates:
        if i < j: 
            heappush(heap, (distances[i, j], i, j))

#keep track of active clusters
cluster_map = {i: i for i in range(k)} 

#merge classes until we reach k/2 clusters
while len(set(cluster_map.values())) > final_k:
    if not heap:
        break

    # Find the closest pair within mergeable clusters
    _, c1, c2 = heappop(heap)

    # Check if clusters are still active
    if cluster_map[c1] != c1 or cluster_map[c2] != c2:
        continue  # Skip outdated merges

    # Merge c2 into c1 (average the centroids)
    new_centroid = (centroids[c1] + centroids[c2]) / 2
    centroids[c1] = new_centroid
    cluster_map[c2] = c1  # Mark c2 as merged into c1

    # Recalculate distances for the merged centroid within eligible clusters
    for c3 in merge_candidates:
        if c3 != c1 and cluster_map[c3] == c3:  # Only active clusters
            new_dist = np.linalg.norm(centroids[c1] - centroids[c3])
            heappush(heap, (new_dist, c1, c3))

#assign final cluster IDs
final_cluster_ids = np.array([cluster_map[label] for label in kmeans.labels_])

print('Classes simplified from '+str(k)+' to '+str(len(np.unique(final_cluster_ids))))

# Update the raster with the new cluster IDs
clusters = np.full(gdm1.shape, np.nan)
clusters_flat = clusters.flatten()
#valid_mask = (~np.isnan(pixels).any(axis=1)) & (~np.isinf(pixels).any(axis=1))
clusters_flat[valid_mask] = final_cluster_ids
clusters = clusters_flat.reshape(gdm1.shape)

# Apply a 3x3 median filter
smoothed_raster = median_filter(clusters, size=3)

#export
output_path = out_dir + '/cluster_raster'+vno+'_s_simplified.tif'
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(smoothed_raster, 1)

#%%