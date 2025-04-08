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
Removes outliers from convex hull

"""

#data=existing_sites_gdm
#data=candidate_sites_gdm
def remove_outliers(data, percentile):
    """Remove points beyond the specified percentile in each dimension."""
    lower_bound = np.percentile(data, 100 - percentile, axis=0)
    upper_bound = np.percentile(data, percentile, axis=0)
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    filtered=data[mask]
    return filtered

#%%

"""
Calculates the volume of the convex hull, with outlier removal.

"""

#clean_data=filtered
def convex_hull_volume(data, percentile):
    
    if data.shape[0] <= data.shape[1]:  # Not enough points for a convex hull
        return 0
    clean_data = remove_outliers(data, percentile)

    if clean_data.shape[0] <= clean_data.shape[1]:  # Still not enough points
        return 0
    hull = ConvexHull(clean_data, qhull_options='QJ')
    return hull.volume

#%%

"""
Returns a boolean mask of points outside the convex hull and their minimum distance to the hull.
"""

def points_and_distance_outside_hull(hull, points):
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

"""
Returns candidate reference sites that meet the min distance requirement and maximises distance from one another

""" 

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



