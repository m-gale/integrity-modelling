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


#%%

#location of GDM transformed vars of Mokany et al. 2022
wdir='F:\\veg2_postdoc\\data\\reference\\National\\mokany_etal_GDM_layers\\data\\250m\\'
scrap_dir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'

gdm_fns=glob(wdir+'*.tif')
#len(gdm_fns)

#%%

"""
Resample the GDM layers to a resolution, extent, CRS, and water mask indicated by a national mask layer
"""

def resample_raster(input_raster_path, reference_raster_path, output_raster_path, mask_dir, mask,):
    with rasterio.open(reference_raster_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_dtype = "float32"

    with rasterio.open(input_raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, ref_crs, ref_width, ref_height, *src.bounds, dst_transform=ref_transform
        )
        profile = src.profile.copy()
        profile.update(
            {
                "crs": ref_crs,
                "transform": ref_transform,
                "width": ref_width,
                "height": ref_height,
                "dtype": ref_dtype,
            }
        )

        with rasterio.open(output_raster_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest,
                )
                
    if mask:
        with rasterio.open(output_raster_path, "r+") as dst:
            # Read the output raster
            data = dst.read(1)

            # Read the mask raster
            with rasterio.open(mask_dir) as mask_src:
                mask_data = mask_src.read(1, out_shape=(dst.height, dst.width), resampling=Resampling.nearest)

                # Apply mask: where mask_data is NaN, set output raster to NaN
                data[mask_data == mask_src.nodata] = float("nan")  # Handle nodata in mask

            # Write the masked data back to the output raster
            dst.write(data, 1)
            

resample_raster(gdm_fns[0], scrap_dir+'national_mask_500m.tif', gdm_fns[0].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[1], scrap_dir+'national_mask_500m.tif', gdm_fns[1].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[2], scrap_dir+'national_mask_500m.tif', gdm_fns[2].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[3], scrap_dir+'national_mask_500m.tif', gdm_fns[3].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[4], scrap_dir+'national_mask_500m.tif', gdm_fns[4].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[5], scrap_dir+'national_mask_500m.tif', gdm_fns[5].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[6], scrap_dir+'national_mask_500m.tif', gdm_fns[6].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[7], scrap_dir+'national_mask_500m.tif', gdm_fns[7].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[8], scrap_dir+'national_mask_500m.tif', gdm_fns[8].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[9], scrap_dir+'national_mask_500m.tif', gdm_fns[9].replace('.tif', '_resampled500.tif'),scrap_dir+'pca_mask4.tif', mask=False)

#%%

"""
Read resampled GDM layers
"""

with rasterio.open(gdm_fns[0].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm1 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[1].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm2 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[2].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm3 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[3].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm4 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[4].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm5 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[5].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm6 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[6].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm7 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[7].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm8 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[8].replace('.tif', '_resampled500.tif')) as pc1_src:
    gdm9 = pc1_src.read(1)  # Read the first band
    #climate_pca_resampled, trans = resample_raster(clim_src, target_shape)
with rasterio.open(gdm_fns[9].replace('.tif', '_resampled500.tif')) as pc1_src:
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
#centroids_df.to_csv(scrap_dir+"centroids33.csv", index=False)


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

output_path = scrap_dir + 'cluster_raster48_s_original.tif'
# Retrieve the transform and metadata from the resampled climate PCA raster
with rasterio.open(gdm_fns[9].replace('.tif', '_resampled500.tif')) as src:
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
output_path = scrap_dir + 'cluster_raster48_s_simplified.tif'
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(smoothed_raster, 1)

#%%