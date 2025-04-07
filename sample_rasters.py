# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:24:39 2024

@author: mattg

"""

import pandas as pd
import os
import rasterio as rio
import geopandas as gpd
import glob
from pyproj import Transformer
from shapely.ops import transform
import copy


#%%

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
obs_path=wdir+'scrap\\pts_updated_500m_v11_1kmspacing_20volexp.shp'
#obs_path=wdir+'data\\HCAS_ref_sites\\HCAS_2.3\\data\\0.Inferred_Reference_Sites\\HCAS23_RC_BenchmarkSample_NSW_clipped.shp'

pred_csv_path=wdir+'data\\predictors_described_v4.csv'
resp_csv_path=wdir+'data\\responses_described_v2.csv'
outdir = 'C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'

#%%
#
df=gpd.read_file(obs_path)
print(df)
#
preds=pd.read_csv(pred_csv_path)
print(preds)

data_path='F:\\veg2_postdoc\\raster_subset_v3'
fn1=os.listdir(data_path)
fns=[s for s in fn1 if '.tif' in s]
fn_dirs=glob.glob(data_path+'\\*.tif')


#%%

"""
Sample predictor and response values for reference sites

"""

points = [(geom.x, geom.y) for geom in df.geometry]

# Loop through each raster and extract values at point locations
for raster_path in fn_dirs:
    print(raster_path)
    with rio.open(raster_path) as src:
        if df.crs != src.crs:
            transformer = Transformer.from_crs(df.crs, src.crs, always_xy=True)
            df['geometry_repro'] = df['geometry'].apply(lambda geom: transform(transformer.transform, geom))
        else:
            df['geometry_repro'] = df['geometry']
        # Convert reprojected geometries to a list of (x, y) tuples
        points = [(geom.x, geom.y) for geom in df['geometry_repro']]
        print('Sampling values...')
        values = [v[0] for v in src.sample(points)]  # Always take the first band's value
        raster_name = raster_path.split("\\")[-1].replace(".tif", "")  # Extract filename as column name
        df[raster_name] = values  # Add values as new column


#%%

df2=copy.copy(df)
df3=pd.DataFrame(df2)
df3.to_csv(obs_path.replace('.shp', '_sampled_v1.csv'))

    
#%%