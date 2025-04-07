# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:28:48 2025

@author: mattg
"""

#%%

def resample_raster(input_raster_path, reference_raster_path, output_raster_path):
    with rio.open(reference_raster_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_dtype = "float32"

    with rio.open(input_raster_path) as src:
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

        with rio.open(output_raster_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest,
                )

#%%
"""
Class count representativeness
"""

def class_count(cluster_raster, pts, output_path):
    count_raster = np.full_like(cluster_raster, np.nan, dtype=np.float32)
    for cluster_id in pts['class_pca'].unique():
        count_raster[cluster_raster == cluster_id] = np.nansum(pts['class_pca']==cluster_id)    
    #output_path = scrap_dir + 'count_raster11.tif'
    with rio.open(output_path, 'w', driver='GTiff',
                       count=1, dtype='float32', crs=cluster_src.crs,
                       transform=src.transform, width=cluster_src.width, height=src.height) as dst:
        dst.write(count_raster, 1)
    
    return count_raster

#%%


#data=existing_sites_gdm
#data=candidate_sites_gdm
def remove_outliers(data, percentile):
    """Remove points beyond the specified percentile in each dimension."""
    lower_bound = np.percentile(data, 100 - percentile, axis=0)
    upper_bound = np.percentile(data, percentile, axis=0)
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    filtered=data[mask]
    return filtered

#clean_data=filtered
def convex_hull_volume(data, percentile):
    """Calculate the volume of the convex hull, with outlier removal."""
    if data.shape[0] <= data.shape[1]:  # Not enough points for a convex hull
        return 0
    clean_data = remove_outliers(data, percentile)
    # Check variance along each dimension
    #variances = np.var(clean_data, axis=0)
    #print("Variances:", variances)  
    #clean_data = clean_data[:, variances > 1e-10]
    #clean_data = np.unique(clean_data, axis=0)
    if clean_data.shape[0] <= clean_data.shape[1]:  # Still not enough points
        return 0
    hull = ConvexHull(clean_data, qhull_options='QJ')
    return hull.volume

def points_and_distance_outside_hull(hull, points):
    """Returns a boolean mask of points outside the convex hull and their minimum distance to the hull."""
    del_hull = Delaunay(hull.points[hull.vertices])  # Delaunay triangulation
    outside_mask = del_hull.find_simplex(points) < 0  # Points outside the hull

    # Compute minimum distance to the hull for each point
    min_distances = np.full(points.shape[0], np.nan)  # Initialize array
    for i, point in enumerate(points):
        # Calculate distance from point to each hull facet
        distances = np.abs(hull.equations[:, :-1] @ point + hull.equations[:, -1]) / np.linalg.norm(hull.equations[:, :-1], axis=1)
        min_distances[i] = np.min(distances)  # Get minimum distance

    return outside_mask, min_distances

def points_outside_hull(hull, points):
    """Returns a boolean mask of points outside the convex hull."""
    del_hull = Delaunay(hull.points[hull.vertices])
    return ~del_hull.find_simplex(points) >= 0



#%%

#points=new_sites
#mmd_batch=100
#max_iterations=max(int(len(new_sites)/simplifier), 10)
#mmd_batch=int(len(new_sites)/simplifier)
#max_iterations=int((len(new_sites)/mmd_batch)*1.1)

def maximize_min_distance(points, min_distance, mmd_batch, max_iterations=1000):
    points = np.array(points)
    tree = KDTree(points)
    
    counter=0
    for _ in range(max_iterations):
        # Find nearest neighbor distances
        distances, indices = tree.query(points, k=2)
        nearest_distances = distances[:, 1]
        
        # Check the current min distance
        current_min_dist = nearest_distances.min()
        if current_min_dist >= min_distance:
            break
        
        # Find the point with the smallest nearest neighbor distance
        idx_to_remove = np.argsort(nearest_distances)[:mmd_batch]
        points = np.delete(points, idx_to_remove, axis=0)
        
        # Rebuild the tree after removal
        tree = KDTree(points)

        counter=counter+1
    
    print(str(counter)+' / '+str(max_iterations)+' max. iterations')
              
    return points

#%%


def calculate_convex_hull_coverage(existing_sites_gdm, candidate_sites_gdm, ref_sites, ref_rows, ref_cols, 
                                    pca, gdm_raster, ref_trans, pts, min_distance, cov_threshold, ref_fn, 
                                    cluster_id, pts_updated, global_tree):
    """
    Evaluates coverage of reference sites based on convex hull volume and selects new candidate sites 
    to maximize coverage while maintaining a minimum distance.
    """
    existing_sites_pca = pca.transform(existing_sites_gdm)
    candidate_sites_pca = pca.transform(candidate_sites_gdm)
    ref_sites_pca = pca.transform(ref_sites)
    ref_indices = np.column_stack((ref_rows, ref_cols))
    
    if len(candidate_sites_pca) > 500000:
        candidate_sites_pca = candidate_sites_pca[np.random.choice(candidate_sites_pca.shape[0], size=500000, replace=False)]
    
    existing_hull_volume = convex_hull_volume(existing_sites_pca, 99)
    candidate_hull_volume = convex_hull_volume(candidate_sites_pca, 99)
    hull = ConvexHull(remove_outliers(existing_sites_pca, 99))
    outside_mask = points_outside_hull(hull, ref_sites_pca)
    combined_hull = hull  # Initialize combined hull
    
    if candidate_hull_volume > 0:
        coverage_ratio = existing_hull_volume / candidate_hull_volume
        print(f"Coverage ratio: {coverage_ratio:.2%}")
        outside_points = ref_sites_pca[outside_mask]
        outside_indices = ref_indices[outside_mask]
        
        if outside_points.shape[0] > 0:
            new_sites_geo = [ref_trans * (col, row) for row, col in outside_indices]
            transformer = Transformer.from_crs(pts.crs, "EPSG:3577", always_xy=True)
            new_sites_pro = [transformer.transform(lon, lat) for lon, lat in new_sites_geo]
            
            global_distances, _ = global_tree.query(new_sites_pro)
            new_sites = np.array(new_sites_pro)[global_distances > min_distance]
            
            if len(new_sites) > 0:
                if len(new_sites) > 10000:
                    new_sites = new_sites[np.random.choice(new_sites.shape[0], size=10000, replace=False)]
                
                new_sites_filt = maximize_min_distance(new_sites, min_distance, max_iterations=len(new_sites))
                print(f"Points meeting distance criteria: {new_sites_filt.shape[0]}")
                
                if len(new_sites_filt) > 0:
                    transformer = Transformer.from_crs("EPSG:3577", pts.crs, always_xy=True)
                    new_sites_filt_geo = [transformer.transform(lon, lat) for lon, lat in new_sites_filt]
                    
                    batch_size = 50
                    batches = [new_sites_filt_geo[i:i + batch_size] for i in range(0, len(new_sites_filt_geo), batch_size)]
                    pts_new = []
                    new_coverage_ratio = coverage_ratio
                    counter = 0
                    
                    for batch in batches:
                        if new_coverage_ratio < cov_threshold:
                            counter += 1
                            batch_rowcols = [rowcol(ref_trans, x, y) for x, y in batch]
                            batch_gdm = np.array([gdm_raster[row, col, :] for row, col in batch_rowcols])
                            batch_filt_pca = pca.transform(batch_gdm)
                            outside_mask = points_outside_hull(combined_hull, batch_filt_pca)
                            outside_points = batch_filt_pca[outside_mask]
                            outside_indices = np.array(batch)[outside_mask]
                            batch = [tuple(coord) for coord in outside_indices]
                            
                            print(f'Batch {counter} - {len(batch)} still outside hull')
                            pts_coords.extend(batch)
                            pts_new.extend(batch)
                            
                            new_sites_rowcols = [rowcol(ref_trans, x, y) for x, y in pts_coords]
                            new_sites_gdm = np.array([gdm_raster[row, col, :] for row, col in new_sites_rowcols])
                            new_sites_filt_pca = pca.transform(new_sites_gdm)
                            
                            if len(remove_outliers(new_sites_filt_pca, 99)) > 6:
                                combined_hull = ConvexHull(remove_outliers(new_sites_filt_pca, 99))
                            combined_hull_volume = combined_hull.volume
                            new_coverage_ratio = combined_hull_volume / candidate_hull_volume
                        else:
                            print('Already reached target coverage')
                    
                    new_df = pd.DataFrame(pts_new, columns=['longitude', 'latitude'])
                    new_df['source'] = ref_fn.split('\\')[-1]
                    new_df['class_pca'] = cluster_id
                    new_gdf = gpd.GeoDataFrame(new_df, 
                                               geometry=gpd.points_from_xy(new_df['longitude'], new_df['latitude']),
                                               crs='EPSG:4326')
                    print(f'Adding {len(new_gdf)} points')
                    pts_updated = pd.concat([pts_updated, new_gdf[['source', 'class_pca', 'geometry']]], ignore_index=True)
                    global_tree = KDTree([geom.coords[0] for geom in pts_updated.geometry.to_crs("EPSG:3577")])
            else:
                print('No sites matching distance criteria')
        else:
            print('No sites outside of existing hull')
    else:
        print('No reference sites to choose from')
    
    return pts_updated, global_tree


#%%



