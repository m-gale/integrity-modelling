# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:53:44 2025

@author: mattg

To do:

How can i ensure that each class covers a similar total cariation rather than land area.
I.e, there are too many desert classes. Remove lat/lon inputs or weight them?

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
import numpy as np
from heapq import heappop, heappush
from collections import defaultdict


#%%

#wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\data\\reference_selection\\pca_rasters\\nsw\\'
wdir='F:\\veg2_postdoc\\data\\reference\\National\\mokany_etal_GDM_layers\\data\\250m\\'
scrap_dir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'

gdm_fns=glob(wdir+'*.tif')
len(gdm_fns)

obs_path='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\data\\HCAS_ref_sites\\HCAS_2.3\\data\\0.Inferred_Reference_Sites\\HCAS23_RC_BenchmarkSample_NSW_clipped.shp'

pts = gpd.read_file(obs_path)

#%%


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

pixels = np.vstack((gdm1.flatten(), gdm2.flatten(),gdm3.flatten(),gdm4.flatten(),gdm5.flatten(),
                    gdm6.flatten(),gdm7.flatten(),gdm8.flatten(),gdm9.flatten(),gdm10.flatten(),)).T
#PCA nan
del gdm2, gdm3, gdm4, gdm5, gdm6, gdm7, gdm8, gdm9, gdm10 
valid_mask = (~np.isnan(pixels).any(axis=1)) & (~np.isinf(pixels).any(axis=1)) & (~(pixels == nodata_value).any(axis=1))
pixels = pixels[valid_mask]  # Keep only valid pixels

scaler = StandardScaler()
pixels = scaler.fit_transform(pixels)
pixels = (pixels * 1000).astype('int16')
#np.nanmin(pixels)

#%%

# k-means clustering on the valid pixels

k = 500  # Number of clusters
#batch_size=2048*10
#batch_size=10240000
batch_size=10240
#kmeans = KMeans(n_clusters=k, random_state=42)
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
kmeans.fit(pixels)

centroids = kmeans.cluster_centers_  # Centroids of the 30 clusters from k-means
centroids_df = pd.DataFrame(centroids, columns=[f'Feature_{i+1}' for i in range(centroids.shape[1])])
centroids_df['Cluster'] = np.arange(1, len(centroids) + 1)
#centroids_df.to_csv(scrap_dir+"centroids33.csv", index=False)



#%%

#export original class raster

labels = kmeans.labels_
np.nanmax(labels)

# Create the clusters array with the same shape as the resampled climate PCA raster
clusters = np.full(gdm1.shape, np.nan)

# Assign the cluster labels to the valid positions using the valid mask
# Note: valid_mask only contains valid indices where NaN pixels were removed
clusters_flat = clusters.flatten()
clusters_flat[valid_mask] = labels

# Reshape the cluster array back to the 2D raster shape
cluster_raster = clusters_flat.reshape(gdm1.shape)

output_path = scrap_dir + 'cluster_raster46_s_original.tif'
# Retrieve the transform and metadata from the resampled climate PCA raster
#with rasterio.open(pc1_dir.replace('.tif', '_resampled3.tif')) as src:
with rasterio.open(gdm_fns[9].replace('.tif', '_resampled500.tif')) as src:
    # Use the transform from the resampled raster (already applied to the resampled data)
    transform = src.transform
    crs = src.crs

# Create metadata for the output raster
metadata = {
    'driver': 'GTiff',
    'count': 1,  # Only one band (cluster labels)
    'dtype': 'float32',  # Use float32 for the raster
    'crs': crs,  # Coordinate reference system
    'transform': transform,  # The affine transformation (already in resampled space)
    'width': cluster_raster.shape[1],  # Width of the resampled raster
    'height': cluster_raster.shape[0],  # Height of the resampled raster
    'nodata': np.nan  # Define no data value as NaN
}

# Apply a 3x3 median filter
smoothed_raster = median_filter(cluster_raster, size=3)

# Write the cluster labels to the new raster file
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(smoothed_raster, 1)  # Write the clusters to the first band


#%%

#simplify
#combine close clusters until we end up with 500 classes

final_k = k/2 #target number of clusters

elevation_centroids = centroids[:, 0]  # Extracts the first column
pg_centroids = centroids[:, 5]  # Extracts the first column

#only simplify classes where plant growth index and elevation range are low
elevation_threshold = np.percentile(elevation_centroids, 80)
pg_threshold = np.percentile(pg_centroids, 60)

# Compute initial pairwise distances
distances = cdist(centroids, centroids, metric='euclidean')
np.fill_diagonal(distances, np.inf)

merge_candidates = np.where((elevation_centroids <= elevation_threshold) & 
                            (pg_centroids <= pg_threshold))[0]
merge_candidates_set = set(merge_candidates)

# Compute pairwise distances
distances = cdist(centroids, centroids, metric='euclidean')
np.fill_diagonal(distances, np.inf)  # Avoid self-merging

# Priority queue for merging (min heap)
heap = []
for i in merge_candidates:
    for j in merge_candidates:
        if i < j:  # Avoid duplicate pairs
            heappush(heap, (distances[i, j], i, j))

# Keep track of active clusters
cluster_map = {i: i for i in range(k)}  # Maps old cluster index to new cluster index

# Merging loop until we reach 250 clusters
while len(set(cluster_map.values())) > final_k:
    if not heap:
        break  # No more mergeable clusters

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

# Assign final cluster IDs
final_cluster_ids = np.array([cluster_map[label] for label in kmeans.labels_])

print('Classes simplified from '+str(k)+' to '+str(len(np.unique(final_cluster_ids))))

# Update the raster with the new cluster IDs
clusters = np.full(gdm1.shape, np.nan)
clusters_flat = clusters.flatten()
#valid_mask = (~np.isnan(pixels).any(axis=1)) & (~np.isinf(pixels).any(axis=1))
clusters_flat[valid_mask] = final_cluster_ids
clusters = clusters_flat.reshape(gdm1.shape)

output_path = scrap_dir + 'cluster_raster46_s_simplified.tif'

# Apply a 3x3 median filter
smoothed_raster = median_filter(clusters, size=3)

# Write the cluster labels to the new raster file
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(smoothed_raster, 1)  # Write the clusters to the first band

#%%

min_distances = np.min(distances, axis=1)

labels = kmeans.labels_
unique_clusters = np.unique(labels)  # Get unique cluster labels
cluster_distances = np.array([closest_centroid_distances[c] for c in unique_clusters])
closest_centroid_distances = {i: min_distances[i] for i in range(len(centroids))}
centroid_er = {i: elevation_centroids[i] for i in range(len(centroids))}

distance_map = np.zeros(clusters.shape)  # Ensure indexing is correct
er_map = np.zeros(clusters.shape)  # Ensure indexing is correct

for cluster, distance in closest_centroid_distances.items():
    print(str(cluster))
    distance_map[clusters==cluster] = distance  # Assign distances

for cluster, distance in centroid_er.items():
    print(str(cluster))
    er_map[clusters==cluster] = distance  # Assign distances


#%%


# Get the cluster labels
labels = kmeans.labels_
np.nanmax(labels)

# Create the clusters array with the same shape as the resampled climate PCA raster
clusters = np.full(gdm1.shape, np.nan)

# Assign the cluster labels to the valid positions using the valid mask
# Note: valid_mask only contains valid indices where NaN pixels were removed
clusters_flat = clusters.flatten()
clusters_flat[valid_mask] = labels

# Reshape the cluster array back to the 2D raster shape
cluster_raster = clusters_flat.reshape(gdm1.shape)

#cluster_raster=clusters
#cluster_raster=er_map
# Visualize the clustering result
plt.figure(figsize=(10, 8))
plt.imshow(cluster_raster, cmap='tab20', interpolation='nearest')
plt.title('K-Means Clustering of Climate-Soil Interaction', fontsize=16)
plt.colorbar(label='Cluster ID')
plt.show()

output_path = scrap_dir + 'cluster_raster38_s_m2.tif'
# Retrieve the transform and metadata from the resampled climate PCA raster
#with rasterio.open(pc1_dir.replace('.tif', '_resampled3.tif')) as src:
with rasterio.open(gdm_fns[9].replace('.tif', '_resampled500.tif')) as src:
    # Use the transform from the resampled raster (already applied to the resampled data)
    transform = src.transform
    crs = src.crs

# Create metadata for the output raster
metadata = {
    'driver': 'GTiff',
    'count': 1,  # Only one band (cluster labels)
    'dtype': 'float32',  # Use float32 for the raster
    'crs': crs,  # Coordinate reference system
    'transform': transform,  # The affine transformation (already in resampled space)
    'width': cluster_raster.shape[1],  # Width of the resampled raster
    'height': cluster_raster.shape[0],  # Height of the resampled raster
    'nodata': np.nan  # Define no data value as NaN
}

# Apply a 3x3 median filter
smoothed_raster = median_filter(cluster_raster, size=3)

# Write the cluster labels to the new raster file
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(smoothed_raster, 1)  # Write the clusters to the first band
    
#%%
 
#map the count of sample points across classes   
    
pts = pts.to_crs(pca1.crs)
coords = [(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]

with rasterio.open(output_path) as clim_src:
    # Use rasterio.sample to extract values for each point's coordinates
    class_values = list(clim_src.sample(coords))
class_values = [val[0] if val or val == 0 else np.nan for val in class_values]
pts['class_pca']=class_values

count_raster = np.full_like(pca1, np.nan, dtype=np.float32)
for cluster_id in pts['class_pca'].unique():
    count_raster[cluster_raster == cluster_id] = np.nansum(pts['class_pca']==cluster_id)    

np.nanmax(count_raster)
np.nanmin(count_raster)

output_path = scrap_dir + 'count_raster12.tif'
with rasterio.open(output_path, 'w', driver='GTiff',
                   count=1, dtype='float32', crs=clim_src.crs,
                   transform=clim_src.transform, width=clim_src.width, height=clim_src.height) as dst:
    dst.write(count_raster, 1)
    
#%%

pts = pts.dropna(subset=['pca1', 'pca2', 'pca3'])
centroids = kmeans.cluster_centers_  # Centroids of the 30 clusters from k-means
distances = cdist(pts[['pca1', 'pca2', 'pca3']], centroids)  # Shape: (num_plots, num_clusters)
weights = 1 / (distances + 1e-6)  # Add small value to avoid division by zero
weights_normalized = weights / weights.sum(axis=1, keepdims=True)
representativeness = weights_normalized.sum(axis=0)

representativeness_df = pd.DataFrame({
    'Cluster ID': np.arange(len(representativeness)),
    'Representativeness': representativeness
})
representativeness_df_sorted = representativeness_df.sort_values(by='Representativeness', ascending=False)
print(representativeness_df_sorted)

representativeness_raster = np.full_like(pca1, np.nan, dtype=np.float32)

for cluster_id in range(len(representativeness)):
    representativeness_raster[cluster_raster == cluster_id] = representativeness[cluster_id]
    
np.nanmax(count_raster)
np.nanmin(count_raster)

output_path = scrap_dir + 'rep_euc_raster11.tif'
with rasterio.open(output_path, 'w', driver='GTiff',
                   count=1, dtype='float32', crs=clim_src.crs,
                   transform=clim_src.transform, width=clim_src.width, height=clim_src.height) as dst:
    dst.write(representativeness_raster, 1)

#%%

# Compute Mahalanobis distance between each sample plot and each cluster centroid
cov_matrix = np.cov(pixels.T)  # Compute covariance matrix of the PCA features
inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse of the covariance matrix

# Compute Mahalanobis distances
distances_mahalanobis = np.array([distance.mahalanobis(x, centroid, inv_cov_matrix)
                                  for x, centroid in zip(pixels, centroids)])

distances_mahalanobis = []
# Loop over each sample plot (pixel)
for x in pts[['pca1', 'pca2', 'pca3']].values:
    # Create a list to store distances from this sample plot to all centroids
    distances_to_centroids = []
    
    # Loop over each centroid
    for centroid in centroids:
        # Calculate the Mahalanobis distance from the sample plot to the centroid
        dist = distance.mahalanobis(x, centroid, inv_cov_matrix)
        distances_to_centroids.append(dist)
    
    # Append the list of distances for this sample plot to the main list
    distances_mahalanobis.append(distances_to_centroids)
    
distances_mahalanobis = np.vstack(distances_mahalanobis)

# Assign each sample plot to the nearest cluster
cluster_assignments = pts['class_pca']

# Cluster-specific scaling: normalize distances within each cluster
normalized_distances = distances_mahalanobis.copy()
for cluster_id in range(len(centroids)):
    cluster_mask = cluster_assignments == cluster_id  # Points assigned to the cluster
    print(str(sum(cluster_mask)))
    cluster_distances = distances_mahalanobis[cluster_mask, cluster_id]  # Distances to the cluster centroid

    # Compute mean and standard deviation of distances for the cluster
    cluster_mean = np.mean(cluster_distances)
    cluster_std = np.std(cluster_distances)

    # Normalize distances for the cluster
    if cluster_std > 0:  # Avoid division by zero
        normalized_distances[cluster_mask, cluster_id] = (
            (cluster_distances - cluster_mean) / cluster_std
        )

  
#weights_m = 1 / (distances_mahalanobis + 1e-6)  # Add small value to avoid division by zero
weights_m = 1 / (normalized_distances + 1e-6)  # Add small value to avoid division by zero
weights_normalized_m = weights_m / weights_m.sum(axis=1, keepdims=True)
representativeness_m = weights_normalized_m.sum(axis=0)

representativeness_df = pd.DataFrame({
    'Cluster ID': np.arange(len(representativeness)),
    'Representativeness_m': representativeness_m
})

# Sort the DataFrame by representativeness, high to low
representativeness_df_sorted = representativeness_df.sort_values(by='Representativeness_m', ascending=False)

# Display the sorted list
print(representativeness_df_sorted)

# Create an empty array for the representativeness raster
representativeness_m_raster = np.full_like(pca1, np.nan, dtype=np.float32)

for cluster_id in range(len(representativeness)):
    representativeness_m_raster[cluster_raster == cluster_id] = representativeness_m[cluster_id]
    
np.nanmax(representativeness_m_raster)
np.nanmin(representativeness_m_raster)

output_path = scrap_dir + 'rep_mahn_raster11.tif'
with rasterio.open(output_path, 'w', driver='GTiff',
                   count=1, dtype='float32', crs=clim_src.crs,
                   transform=clim_src.transform, width=clim_src.width, height=clim_src.height) as dst:
    dst.write(representativeness_m_raster, 1)

#%%


gdm_fns=glob(wdir+'*resampled3.tif')
len(gdm_fns)

gdm_layers = []
for gdm_fn in gdm_fns:
    with rasterio.open(gdm_fn) as src:
        gdm_layer = src.read(1) 
        gdm_layer[gdm_layer==src.nodata]=np.nan
        gdm_layers.append(gdm_layer)

gdm_stack = np.stack(gdm_layers, axis=0)

output_path = scrap_dir + 'cluster_raster24_10240000_s.tif'
with rasterio.open(output_path) as cluster_src:
    cluster_raster = cluster_src.read(1)
    cluster_nodata = cluster_src.nodata

unique_classes = np.unique(cluster_raster[cluster_raster != cluster_nodata])
class_variances = {}
class_areas = {}
for cls in unique_classes:
    # Get the mask for the current class
    class_mask = cluster_raster == cls
    
    # Extract the GDM values for this class across all layers
    class_values = gdm_stack[:, class_mask]  # Shape: (10, num_pixels_in_class)

    # Calculate variance across the 10 layers
    class_variance = np.nanvar(class_values, axis=1)  # Variance per layer
    class_variances[cls] = class_variance
    class_areas[cls] = np.nansum(class_mask)

class_variance_mean={}
for cls, var in class_variances.items():
    print(f"Class {cls}: Variance per layer: {var}")
    class_variance_mean[cls] = np.nanmean(var)
    
valid_class_variances = {cls: var for cls, var in class_variance_mean.items() if not np.isnan(var)}
valid_class_areas = {cls: var for cls, var in class_areas.items() if not (np.isnan(var) | var==0)}

variances = np.array(list(valid_class_variances.values()))
areas=np.array(list(valid_class_areas.values()))

va=(variances*3)*(areas*0.5)

va_df = pd.DataFrame(va.astype(int), index=valid_class_areas.keys(), columns=["Weighted_Samples"])
va_df['Class']=va_df.index

va_df.to_csv(scrap_dir+"weighted_samples_24_2.csv", index=False)


#%%

sorted_class_variance_mean = dict(sorted(class_variance_mean.items(), key=lambda item: item[1], reverse=True))


# Normalize variances between 0 and 1
normalized_variances = (variances - variances.min()) / (variances.max() - variances.min())

# Apply square root scaling to smooth the distribution
scaled_variances = np.sqrt(normalized_variances)

# Scale to range [500, 1000]
minsamp=1000

weighted_samples = {
    cls: int(1000 + (2000 - 1000) * scale)
    for cls, scale in zip(class_variance_mean.keys(), scaled_variances)
}


sample_values = np.array(list(weighted_samples.values()))

# Generate the histogram
plt.figure(figsize=(8, 6))
plt.hist(sample_values, bins=np.arange(900, 2000, 20), color='skyblue', edgecolor='black')  # Adjust bins as needed
#plt.hist(sample_values, bins=np.arange(0, 0.02, 0.0002), color='skyblue', edgecolor='black')  # Adjust bins as needed


# Add labels and title
plt.title('Histogram of Weighted Samples', fontsize=16)
plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#export
weighted_samples_df = pd.DataFrame(
    list(weighted_samples.items()), columns=["Class", "Weighted_Samples"]
)
# Export to CSV
weighted_samples_df.to_csv(scrap_dir+"weighted_samples_24.csv", index=False)




#%%

# Define the minimum, median, and maximum samples
min_samples = 1000
median_samples = 750
max_samples = 1000

# Filter out classes with NaN variances
valid_class_variances = {cls: var for cls, var in class_variance_mean.items() if not np.isnan(var)}

# Extract the minimum, maximum, and median variance values
min_variance = min(valid_class_variances.values())
max_variance = max(valid_class_variances.values())
median_variance = np.median(list(valid_class_variances.values()))

# Calculate the weighted number of samples
weighted_samples = {}
for cls, mean_var in valid_class_variances.items():
    # Normalize the variance to 0–1
    if mean_var <= median_variance:
        # Scale between min and median
        scaled_samples = ((mean_var - min_variance) / (median_variance - min_variance)) * (median_samples - min_samples) + min_samples
    else:
        # Scale between median and max
        scaled_samples = ((mean_var - median_variance) / (max_variance - median_variance)) * (max_samples - median_samples) + median_samples
    
    weighted_samples[cls] = int(round(scaled_samples))  # Round and convert to integer

# Print the results
for cls, samples in weighted_samples.items():
    print(f"Class {cls}: Weighted samples: {samples}")
    
sample_values = np.array(list(valid_class_variances.values()))

# Example: Variance values from your calculation
variances = np.array(list(class_variance_mean.values()))

# Normalize variances between 0 and 1
normalized_variances = (variances - variances.min()) / (variances.max() - variances.min())

# Apply square root scaling to smooth the distribution
scaled_variances = np.sqrt(normalized_variances)

# Scale to range [500, 1000]
weighted_samples = {
    cls: int(1000 + (1050 - 1000) * scale)
    for cls, scale in zip(class_variance_mean.keys(), scaled_variances)
}


sample_values = np.array(list(weighted_samples.values()))

# Generate the histogram
plt.figure(figsize=(8, 6))
plt.hist(sample_values, bins=np.arange(500, 1000, 20), color='skyblue', edgecolor='black')  # Adjust bins as needed
#plt.hist(sample_values, bins=np.arange(0, 0.02, 0.0002), color='skyblue', edgecolor='black')  # Adjust bins as needed


# Add labels and title
plt.title('Histogram of Weighted Samples', fontsize=16)
plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#export
weighted_samples_df = pd.DataFrame(
    list(weighted_samples.items()), columns=["Class", "Weighted_Samples"]
)
# Export to CSV
weighted_samples_df.to_csv(scrap_dir+"weighted_samples_24.csv", index=False)

#%%