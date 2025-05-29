# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:49:29 2025

@author: mattg
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
wdir='F:\\veg2_postdoc\\data\\reference\\National\\mokany_etal_GDM_layers\\data\\90m\\'
scrap_dir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'
mask_dir='F:\\veg2_postdoc\\data\\misc\\national_mask_250m.tif'

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
            

resample_raster(gdm_fns[0], mask_dir, gdm_fns[0].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[1], mask_dir, gdm_fns[1].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[2], mask_dir, gdm_fns[2].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[3], mask_dir, gdm_fns[3].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[4], mask_dir, gdm_fns[4].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[5], mask_dir, gdm_fns[5].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[6], mask_dir, gdm_fns[6].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[7], mask_dir, gdm_fns[7].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[8], mask_dir, gdm_fns[8].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)
resample_raster(gdm_fns[9], mask_dir, gdm_fns[9].replace('.tif', '_resampled250.tif'),scrap_dir+'pca_mask4.tif', mask=False)

#%%