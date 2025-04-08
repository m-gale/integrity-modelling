# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:16:59 2023

@author: mattg


Purpose: R predict function needs rasters to have equal extent and resolution

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

#adjust the x manually
minx=-2000000
maxx=2120000
miny=msk.bounds['miny'][0]-1000
maxy=msk.bounds['maxy'][0]+1000

outbnds=(minx, miny, maxx, maxy)

#rest resample resolution
res=250

#%%

#set outdir

outdir_temp='F:\\veg2_postdoc\\raster_subset_v3\\TEMP\\'
outdir='F:\\veg2_postdoc\\raster_subset_v3\\'
if os.path.exists(outdir)==False:
    os.mkdir(outdir)
if os.path.exists(outdir_temp)==False:
    os.mkdir(outdir_temp)


#%%

"""
Make a water mask by classifying the seasonal water layer
"""
water_mask_dir='F:\\veg2_postdoc\\raster_subset_v3\\TEMP\\water_mask_250m_temp.tif'

outfn2=water_mask_dir.replace('_temp', '')
if os.path.isfile(outfn2)==False:
    src = rio.open(water_mask_dir)
    meta=src.meta
    ro=gdal.Open(water_mask_dir)
    ras=np.array(ro.GetRasterBand(1).ReadAsArray())
    ras[(ras>0.5) | (np.isnan(ras))] = meta['nodata']
    ras[ras!=meta['nodata']]=1
    ras[ras!=1]=0
    meta['dtype']='uint8'
    outfn2=water_mask_dir.replace('_temp', '')
    with rio.open(outfn2, "w", **meta) as dest:
            dest.write(ras, 1)

#read it
r_masko=gdal.Open(outfn2)
r_mask=np.array(r_masko.GetRasterBand(1).ReadAsArray())

with rio.open(outfn2) as src:
    mask_meta=src.meta
    
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
    
    #bilinear ig continuous
    #NN if categorical
    if resamp=='BL':
        options = gdal.WarpOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_Bilinear, outputBounds=outbnds, 
                                   outputBoundsSRS='EPSG:3577', srcSRS=src_srs, dstSRS='EPSG:3577', dstNodata=9999)
    else:
        options = gdal.WarpOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_NearestNeighbour, outputBounds=outbnds, 
                                   outputBoundsSRS='EPSG:3577', srcSRS=src_srs, dstSRS='EPSG:3577', dstNodata=9999)
    #resample
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

"""
Do it for predictors

"""

counter=1
#i=2
for i in range(0, len(fn_lu)):
    print(fn_lu['Predictor'][i]  +' - '+str(counter)+'  /  '+str(len(fn_lu)))
    fn=fn_lu['Predictor'][i]    
    resamp=fn_lu['Resample_method'][i]
    if len(glob.glob(fn_lu['Directory'][i]+'\\'+fn+'.'+fn_lu['Format'][i]))>0:
        fn_dir=glob.glob(fn_lu['Directory'][i]+'\\'+fn+'.'+fn_lu['Format'][i])[0]
        outfn1=outdir_temp+fn+'.tif'
        if os.path.isfile(outdir+fn+'.tif'):
            print('Already exists')
        else:
            standardise_ras(fn, fn_dir, resamp, outbnds, outdir_temp, outdir, res, r_mask, mask_meta, mask_ibra=True)
            counter=counter+1        
    else:
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

