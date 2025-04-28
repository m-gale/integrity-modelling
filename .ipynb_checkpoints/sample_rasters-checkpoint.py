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
from shapely import wkt
import dask
from dask import delayed, compute
import dask.threaded  



#%%

#local
wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
obs_path=wdir+'scrap\\pts_updated_500m_v11_1kmspacing_20volexp_residuals.shp'
#obs_path=wdir+'scrap\\pts_updated_500m_v11_1kmspacing_20volexp_sampled_v1.csv'
pred_csv_path=wdir+'data\\predictors_described_v4.csv'
resp_csv_path=wdir+'data\\responses_described_v2.csv'
outdir = 'C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'
data_path='F:\\veg2_postdoc\\raster_subset_v3'

#nci
wdir='/g/data/xc0/project/natint/'
obs_path=wdir+'output/v1/ref_sites/pts_updated_250m_v1_1km_25volexp_90th.shp'
pred_csv_path=wdir+'input/predictors_described_v4.csv'
resp_csv_path=wdir+'inpit/responses_described_v2.csv'
outdir=wdir+'output/v1/sample_rasters'
data_path=wdir+'input/raster_subset_v4'


if os.path.exists(outdir)==False:
    os.mkdir(outdir)
    
#%%

df=gpd.read_file(obs_path)
print(df)

#if from already sampled csv
# df=pd.read_csv(obs_path)
# print(df)
# df['geometry'] = df['geometry'].apply(wkt.loads)
# df = gpd.GeoDataFrame(df, geometry='geometry')
# df.set_crs(epsg=3577, inplace=True)

preds=pd.read_csv(pred_csv_path)
print(preds)

fn1=os.listdir(data_path)
fns=[s for s in fn1 if '.tif' in s]
#fn_dirs=glob.glob(data_path+'\\*.tif')
fn_dirs=glob.glob(data_path+'/*.tif')

#%%

"""
Sample predictor and response values for reference sites

"""


def sample_raster(raster_path, df, original_crs):
    with rio.open(raster_path) as src:
        if original_crs != src.crs:
            transformer = Transformer.from_crs(original_crs, src.crs, always_xy=True)
            geometries_repro = df['geometry'].apply(lambda geom: transform(transformer.transform, geom))
        else:
            geometries_repro = df['geometry']
        
        points = [(geom.x, geom.y) for geom in geometries_repro]
        
        print(f"Sampling {raster_path}...")
        values = [v[0] for v in src.sample(points)]
        
        raster_name = raster_path.split("\\")[-1].replace(".tif", "")
        raster_name = raster_path.split("/")[-1].replace(".tif", "")

        return pd.DataFrame({raster_name: values})

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

"""
Export
"""

df2=copy.copy(df)
df3=pd.DataFrame(df2)
df3.to_csv(obs_path.replace('.shp', '_sampled_v1.csv'))

    
#%%