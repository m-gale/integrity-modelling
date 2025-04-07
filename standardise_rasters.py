# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:16:59 2023

@author: mattg

R predict function needs rasters to have equal extent and resolution

"""

#%%

import rasterio as rio
import geopandas as gpd
import pandas as pd
from osgeo import gdal
import glob
import rioxarray as rx
import numpy as np
from shapely.geometry import Polygon
import copy
import fiona
import os

#%%

#directory for rasters and sample points

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\data'
fn_lu=pd.read_csv(wdir+'\\predictors_described_v4.csv')
resp_lu=pd.read_csv(wdir+'\\responses_described_v2.csv')


#%%

#shapefile mask for minimum extent of Australia not including territories

mask_dir=wdir+'\\misc\\IBRA7_regions_states\\aus_mask_non-water.shp'
msk=gpd.read_file(mask_dir)
msk=msk.to_crs('EPSG:3577')

#%%

outdir_temp='F:\\veg2_postdoc\\raster_subset_v3\\TEMP\\'
outdir='F:\\veg2_postdoc\\raster_subset_v3\\'
if os.path.exists(outdir)==False:
    os.mkdir(outdir)
if os.path.exists(outdir_temp)==False:
    os.mkdir(outdir_temp)

minx=-2000000
maxx=2120000
miny=msk.bounds['miny'][0]-1000
maxy=msk.bounds['maxy'][0]+1000

outbnds=(minx, miny, maxx, maxy)
res=250

#%%

#the GDM water coverage layer will make a good water mask

#this was made in qgis
water_mask_dir='F:\\veg2_postdoc\\raster_subset_v3\\TEMP\\water_mask_250m_temp.tif'

#src = rio.open(water_mask_dir)
#meta=src.meta
#ro=gdal.Open(water_mask_dir)

#ras=np.array(ro.GetRasterBand(1).ReadAsArray())
#ras[(ras>0.5) | (np.isnan(ras))] = meta['nodata']
#ras[ras!=meta['nodata']]=1
#ras[ras!=1]=0

#meta['dtype']='uint8'

outfn2=water_mask_dir.replace('_temp', '')
#with rio.open(outfn2, "w", **meta) as dest:
#        dest.write(ras, 1)

#%%

outfn2=water_mask_dir.replace('_temp', '')
r_masko=gdal.Open(outfn2)
r_mask=np.array(r_masko.GetRasterBand(1).ReadAsArray())

with rio.open(outfn2) as src:
    mask_meta=src.meta


#%%

"""
For predictors

"""

counter=1
#i=2
for i in range(0, len(fn_lu)):
    print(fn_lu['Predictor'][i]  +' - '+str(counter)+'  /  '+str(len(fn_lu)))
    fn=fn_lu['Predictor'][i]    
    resamp=fn_lu['Resample_method'][i]
    if len(glob.glob(fn_lu['Directory'][i]+'\\'+fn+'.'+fn_lu['Format'][i]))>0:
        fn_dir=glob.glob(fn_lu['Directory'][i]+'\\'+fn+'.'+fn_lu['Format'][i])[0]
        #fn_dir=glob.glob('F:\\veg2_postdoc\\raster_subset_v1\\'+fn+'.tif')[0]
        outfn1=outdir_temp+fn+'.tif'
        if os.path.isfile(outdir+fn+'.tif'):
            print('Already exists')
        else:
            standardise_ras(fn, fn_dir, resamp, outbnds, outdir_temp, outdir, res, r_mask, mask_meta, mask_ibra=True)
            counter=counter+1        
    else:
        print(poo)
        print('File not found: '+str(i))
        print('Error: '+fn)
        print(' ')
            
#%%

"""
For responses
"""

counter=1
#i=3
for i in range(0, len(resp_lu)):
    print('')
#for i in [2]:
    try:
        print(resp_lu['Response'][i]  +' - '+str(counter)+'  /  '+str(len(resp_lu)))
        fn=resp_lu['Response'][i]    
        resamp=resp_lu['Resample_method'][i]
        fn_dir=glob.glob(resp_lu['Directory'][i]+'\\'+fn+'.'+resp_lu['Format'][i])[0]
        #fn_dir=glob.glob('F:\\veg2_postdoc\\raster_subset_v1\\'+fn+'.tif')[0]
        #outfn1=outdir_temp+fn+'.tif'
        if os.path.isfile(outdir+fn+'.tif'):
            print('Already exists')
        else:
            standardise_ras(fn, fn_dir, resamp, outbnds, outdir_temp, outdir, res, r_mask, mask_meta, mask_ibra=True)
            counter=counter+1        
    except:
        print('Error: '+str(i))
        print('Error: '+fn)

#%%

"""
Function to standardise predictor rasters
"""

def standardise_ras(fn, fn_dir, resamp, outbnds, outdir_temp, outdir, res, r_mask, mask_meta, mask_ibra):
    src = rio.open(fn_dir)
    if src.crs is None:
        #assume epsg 4326
        src_srs='EPSG:4326'
    else:
        src_srs=str(src.crs)
    
    outfn1=outdir_temp+fn+'.tif'
    if resamp=='BL':
        options = gdal.WarpOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_Bilinear, outputBounds=outbnds, 
                                   outputBoundsSRS='EPSG:3577', srcSRS=src_srs, dstSRS='EPSG:3577', dstNodata=9999)
    else:
        options = gdal.WarpOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_NearestNeighbour, outputBounds=outbnds, 
                                   outputBoundsSRS='EPSG:3577', srcSRS=src_srs, dstSRS='EPSG:3577', dstNodata=9999)
    ds = gdal.Warp(outfn1, fn_dir, options=options)
    
    if mask_ibra == True:
        
        #mask to ibra
        src=rio.open(outfn1)   
        meta=src.meta
        
        #where the ibra mask suggests that there should be data, but there isn't data, 
        #i.e., raster nan value rather than -1
        #set to the median value of the raster
        ro=gdal.Open(outfn1)
        ras=np.array(ro.GetRasterBand(1).ReadAsArray())
        ras[(r_mask==1) & (ras==meta['nodata'])]=np.nanmedian(ras[ras!=meta['nodata']])
        ras[r_mask==mask_meta['nodata']]=meta['nodata']
        ras[r_mask==0]=meta['nodata']

        
        #msk_ras=rio.mask.mask(src, msk['geometry'], nodata=-999)
            
        outfn2=outdir+fn+'.tif'
        with rio.open(outfn2, "w", **meta) as dest:
                dest.write(ras, 1)
                
                
#%%


def standardise_ras_for_mask(fn, fn_dir, fn_lu, outbnds, outdir_temp, outdir, res, mask_ibra):
    src = rio.open(fn_dir)
    if src.crs is None:
        #assume epsg 4326
        src_srs='EPSG:4326'
    else:
        src_srs=str(src.crs)
    
    outfn1=outdir_temp+fn+'.tif'
    options = gdal.WarpOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_Bilinear, outputBounds=outbnds, 
                               outputBoundsSRS='EPSG:3577', srcSRS=src_srs, dstSRS='EPSG:3577', dstNodata=-999)
    ds = gdal.Warp(outfn1, fn_dir, options=options)
    
    if mask_ibra == True:
        
        #mask to ibra
        src=rio.open(outfn1)   
        meta=src.meta
        
        #where the ibra mask suggests that there should be data, but there isn't data, 
        #i.e., raster nan value rather than -1
        #set to the median value of the raster
        ro=gdal.Open(outfn1)
        ras=np.array(ro.GetRasterBand(1).ReadAsArray())
        #ras[(r_mask==1) & (ras==meta['nodata'])]=np.nanmedian(ras[ras!=meta['nodata']])
        #ras[r_mask==meta['nodata']]=meta['nodata']
        
        #msk_ras=rio.mask.mask(src, msk['geometry'], nodata=-999)
            
        outfn2=outdir+fn+'.tif'
        with rio.open(outfn2, "w", **meta) as dest:
                dest.write(ras, 1)

#%%

"""
Variables generated at sampling workflow point
E.g., distance to intensive land use
Time since clearing, time since fire

"""


fns=['D:\\SLATS\\merged_1988-2020\\slats_merged_days_since_1-1-2021.tif',
    'D:\\fire_history\\ts_fire_nsw.tif',
    'D:\\landuse\\BCT_ts2024.tif',
    'D:\\landuse\\NPWS_Estate_ts2024.tif',
    'D:\\biodiversity_nvace_v1\\patch_size.tif']

res=90
for fn_dir in fns:
    print(fn_dir)
    
    src = rio.open(fn_dir)
    if src.crs is None:
        #assume epsg 4326
        src_srs='EPSG:4326'
    else:
        src_srs=str(src.crs)
    
    fn=fn_dir.split('.tif')[0].split('\\')[-1]
    if 'ts_fire' in fn:    
        fn='ts_fire'
    if 'slats' in fn:
        fn='ts_clear'
    if 'BCT_ts' in fn:
        fn='ts_bct_gazette'
    if 'NPWS_Estate' in fn:
        fn='ts_npws_gazette'
    if 'patch_size' in fn:
        fn='patch_area'
        
    outfn1=outdir_temp+fn+'.tif'
    #uses nearest neighbour interpolation
    options = gdal.WarpOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_NearestNeighbour, outputBounds=outbnds, 
                               outputBoundsSRS='EPSG:3308', srcSRS=src_srs, dstSRS='EPSG:3308', dstNodata=-999)
    ds = gdal.Warp(outfn1, fn_dir, options=options)
    
    #mask to ibra
    src=rio.open(outfn1)   
    meta=src.meta
    
    #where the ibra mask suggests that there should be data, but there isn't data, 
    #i.e., raster nan value rather than -1
    #set to the median value of the raster
    ro=gdal.Open(outfn1)
    ras=np.array(ro.GetRasterBand(1).ReadAsArray())
    if ('ts_fire' in fn) | ('ts_clear' in fn):
        ras[(r_mask==1) & (ras==meta['nodata'])]=np.nanmax(ras[ras!=meta['nodata']])
    else:
        if ('gazette' in fn) | ('patch_size' in fn):
            ras[(ras==meta['nodata'])]=0
        else:
            ras[(r_mask==1) & (ras==meta['nodata'])]=np.nanmedian(ras[ras!=meta['nodata']])
    #water mask
    ras[r_mask==meta['nodata']]=meta['nodata']
    
    outfn2=outdir+fn+'.tif'
    with rio.open(outfn2, "w", **meta) as dest:
            dest.write(ras, 1)
    
            
#%%
"""
To generate predictor surface for species score method
Take the median
Currently 2, though should be revised with new study area
Assumes above section has been run

"""
med=1
ras[ras!=meta['nodata']]=med
ras[ras==meta['nodata']]=-999

meta['nodata']=-999
meta['dtype']='int16'

outfn2=outdir+'Species_sc.tif'
with rio.open(outfn2, "w", **meta) as dest:
    dest.write(ras, 1)
        

#%%

#scrap

from matplotlib import pyplot as plt

hist, bin_edges = np.histogram(ras, bins=50)
a=plt.hist(x=hist, bins=bin_edges, color='#0504aa',
           alpha=0.7, rwidth=0.85)

infn='C:\\Users\\mattg\\Documents\\ANU_HD\\veg_postdoc\\data\\TERN\\Vegetation\\NSW_clip\\OzWALD.BS.AnnualMeans_NSW_clip.nc'

outdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg_postdoc\\scrap\\resample_test11.nc'

fo=fiona.open(template_dir)
template=Polygon(fo[0]['geometry']['coordinates'][0])
minx=148
maxx=149.3
miny=-36
maxy=-35

minx=9400000
maxx=9500000
miny=4180000
maxy=4300000

options = gdal.WarpOptions(xRes=30, yRes=30, resampleAlg=gdal.GRA_Bilinear, outputBounds=(minx, miny, maxx, maxy), 
                           outputBoundsSRS='EPSG:4283', srcSRS='EPSG:4283', dstSRS='EPSG:3308')
ds = gdal.Warp(outfn, infn, options=options)


crop=pd.Series([{'type':'Polygon', 'coordinates': mapping(template)['coordinates']}], name='geometry')
  

#two clips for lanscape and patch calculations
  r_patch=r.rio.clip(crop)[0]
  r_ls=np.array(r_patch, dtype='float32')

template_src=rio.open(template_dir)

src.meta


fns=['D:\\SLATS\\merged_1988-2020\\slats_merged_days_since_1-1-2021.tif',
    'D:\\fire_history\\ts_fire_nsw.tif']

for fn_dir in fns:
    print(fn_dir)
    fn=fn_dir.split('\\')[-1]
    so=gdal.Open(fn_dir)
    sarr=np.array(so.GetRasterBand(1).ReadAsArray())
    sarr[sarr==0]=np.nanmax(sarr)        
    src=rio.open(fn_dir)
    meta=src.meta
    outfn=fn_dir.split('.tif')[0]+'_nanfilled.tif'
    with rio.open(outfn, "w", **meta) as dest:
            dest.write(sarr, 1)


