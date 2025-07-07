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

#nci
wdir='/g/data/xc0/project/natint/'
#obs_path=wdir+'output/v1/ref_sites/pts_updated_250m_v1_1km_25volexp_90th.shp'
pred_csv_path=wdir+'input/predictors_described_v6.csv'
resp_csv_path=wdir+'input/responses_described_v3.csv'
outdir=wdir+'output/v1/sample_rasters'
data_path='/scratch/xc0/mg5402/raster_subset_v3'
cluster_fn=wdir+'output/v1/gdm_kmeans/cluster_raster1_s_simplified.tif'
nsw_dir=wdir+'input/misc/NSW_polygon.shp'
vi_dir=wdir+'input/misc/VI_GBRT_filtered_input.csv'

if os.path.exists(outdir)==False:
    os.mkdir(outdir)

obs_paths=glob.glob(wdir+'output/v1/ref_sites/*NSW-only.shp')
print(obs_paths)
len(obs_paths)

#%%

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

nsw_poly=gpd.read_file(nsw_dir)

#%%

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
        

#%%

client = Client(n_workers=16)
print(client)

for obs_path in obs_paths:
    outpath=outdir+'/'+obs_path.split('/')[-1].replace('.shp', '_sampled.csv')
    if os.path.exists(outpath)==False:
        print(obs_path)
        df=gpd.read_file(obs_path)
        #list(df.keys())
        clipped = gpd.sjoin(df, nsw_poly, predicate='intersects', how='inner')
        
        tasks = [delayed(sample_raster)(raster_path, clipped, clipped.crs) for raster_path in fn_dirs]
        
        futures = client.compute(tasks)
        progress(futures)
        
        results = client.gather(futures)
        
        for partial_df in results:
            clipped = pd.concat([clipped.reset_index(drop=True), partial_df.reset_index(drop=True)], axis=1)
        
        clipped.to_csv(outdir+'/'+obs_path.split('/')[-1].replace('.shp', '_sampled.csv'))
        print('Exported: '+outdir+'/'+obs_path.split('/')[-1].replace('.shp', '_sampled.csv'))
        print('')

        client.cancel(futures)
        del futures, results, tasks
        import gc; gc.collect()
        del df, clipped, partial_df

print('Finished')


#%%

"""
Sample predictor and response values for NSW validation sites

"""

val_csv=pd.read_csv(vi_dir)
val_csv=val_csv[['geometry', 'VI', 'mVI', 'CCS', 'SCS', 'tot_native_richness']]
val_csv['geometry'] = val_csv['geometry'].apply(wkt.loads)
val=gpd.GeoDataFrame(val_csv, geometry=val_csv['geometry'])
val.crs='EPSG:3577'

tasks = [delayed(sample_raster)(raster_path, val, val.crs) for raster_path in fn_dirs]
#len(tasks)
futures = client.compute(tasks)
progress(futures)
results = client.gather(futures)

for partial_df in results:
    val = pd.concat([val.reset_index(drop=True), partial_df.reset_index(drop=True)], axis=1)

val.to_csv(outdir+'/'+vi_dir.split('/')[-1].replace('.csv', '_sampled_v2.csv'))
print('Exported: '+outdir+'/'+vi_dir.split('/')[-1].replace('.shp', '_sampled_v2.csv'))
print('')


    
#%%