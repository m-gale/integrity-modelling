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
from dask.distributed import Client, progress
from dask import delayed



#%%

#local
wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
obs_path=wdir+'scrap\\pts_updated_500m_v11_1kmspacing_20volexp_residuals_v2.shp'
#obs_path=wdir+'scrap\\pts_updated_500m_v11_1kmspacing_20volexp_sampled_v1.csv'
pred_csv_path=wdir+'data\\predictors_described_v6.csv'
resp_csv_path=wdir+'data\\responses_described_v3.csv'
outdir = 'C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'
data_path='F:\\veg2_postdoc\\raster_subset_v3'

#nci
wdir='/g/data/xc0/project/natint/'
obs_path=wdir+'output/v2/ref_sites/pts_updated_250m_v2_2km_175volexp_50distp_95cov_4pca_95out_residuals_v2.shp'
pred_csv_path=wdir+'input/predictors_described_v6.csv'
resp_csv_path=wdir+'input/responses_described_v3.csv'
outdir=wdir+'output/v2/sample_rasters'
data_path='/scratch/xc0/mg5402/raster_subset_v3'
cluster_fn=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'

if os.path.exists(outdir)==False:
    os.mkdir(outdir)
    
#%%

"""
Process if doesn't exist
Clip pts to NSW
Sample in loop
"""


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
fn_dirs.append(cluster_fn)

print(str(len(fn_dirs))+' rasters for sampling')

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


original_df = df.copy()

client = Client(n_workers=16)

tasks = [delayed(sample_raster)(raster_path, original_df, original_df.crs) for raster_path in fn_dirs]
print(str(len(tasks))+' tasks')

futures = client.compute(tasks)
progress(futures)

results = client.gather(futures)

for partial_df in results:
    original_df = pd.concat([original_df.reset_index(drop=True), partial_df.reset_index(drop=True)], axis=1)

df = original_df  

#%%

"""
Export
"""

df2=copy.copy(df)
df3=pd.DataFrame(df2)
df3.to_csv(outdir+'/'+obs_path.split('/')[-1].replace('.shp', '_sampled.csv'))

    
#%%

#debug
for raster_path in fn_dirs:
    print(raster_path)
    with rio.open(raster_path) as src:
        ras=src.read(1)

