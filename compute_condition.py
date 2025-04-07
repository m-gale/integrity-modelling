# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:44:56 2025

@author: mattg
"""

import glob
import os
import rasterio as rio
from osgeo import gdal
import numpy as np
import pandas as pd

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
bench_dir=wdir+'scrap\\'
ext_dir='F:\\veg2_postdoc\\raster_subset_v3\\'
resp_csv_path=wdir+'data\\responses_described_v2.csv'
outdir = 'C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'


#%%
#

resps=pd.read_csv(resp_csv_path)
print(resps)

#resp='Forest_height_2019_AUS'
resp='agb_australia_90m'
bench_fn=bench_dir+resp+'_250m_national.tif'

for resp in resps['Response']:
    print(resp)
    if len(glob.glob(bench_dir+resp+'*pts_updated_test21_90m_50thp_1km_sampled_v2_v10.tif'))>0:
        #bench_fn=glob.glob(bench_dir+resp+'*pts_updated_test21_90m_50thp_1km_sampled_v2_v10.tif')[0]
        ext_fn=ext_dir+resp+'.tif'
        
        with rio.open(bench_fn) as bench_src:
            bench=bench_src.read(1)
            meta=bench_src.meta
            target_resolution=bench_src.transform[0]
        
        out_filename = bench_fn.replace('.tif', '_ext.tif')
        if os.path.isfile(out_filename)==True:
            
            with rio.open(ext_fn) as src:
                src_crs = src.crs
                #target_resolution = src.transform[0]
                src_dtype = src.dtypes[0] 
                data = src.read(1, masked=True)  
                valid_data = data.compressed()  
                sample = np.random.choice(valid_data, 10000, replace=False)
                if np.all(np.isclose(sample, np.round(sample))):
                    is_int=True
                else:
                    is_int=False
                del data
                del valid_data
                
            dtype_map = {
                "uint8": gdal.GDT_Byte,
                "uint16": gdal.GDT_UInt16,
                "int16": gdal.GDT_Int16,
                "uint32": gdal.GDT_UInt32,
                "int32": gdal.GDT_Int32,
                "float32": gdal.GDT_Float32,
                "float64": gdal.GDT_Float64
            }
        
            gdal_dtype = dtype_map.get(src_dtype, gdal.GDT_Float32)
            
            if is_int==False:
                options = gdal.WarpOptions(
                    xRes=target_resolution,
                    yRes=target_resolution,
                    resampleAlg=gdal.GRA_Bilinear,
                    srcSRS=src_crs,
                    dstNodata=99999,
                    outputType=gdal_dtype        
                )
            else:
                options = gdal.WarpOptions(
                    xRes=target_resolution,
                    yRes=target_resolution,
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    srcSRS=src_crs,
                    dstNodata=255,
                    outputType=gdal_dtype        
                )
            
            ds = gdal.Warp(out_filename, ext_fn, options=options)
            
            with rio.open(out_filename) as ext_src:
                ext=ext_src.read(1)
                
            #implementation depends on type of product
            if 'month' in resp:
                loss1 = (bench - ext) % 12
                loss2 = (ext - bench) % 12
                loss = np.where(loss1 <= loss2, loss1, -loss2)
                ploss= np.empty(loss.shape)
            else:
                loss=bench-ext
                ploss=loss/bench*100
                #ploss[ploss>1000]=np.nan
                #ploss[ploss<-1000]=np.nan
                #loss[ploss>1000]=np.nan
                #loss[ploss<-1000]=np.nan
                
            mask=(bench==bench_src.nodata) | (ext==ext_src.nodata)
            
            if is_int==False:
                loss[mask]=np.nan
                ploss[mask]=np.nan
                meta['nodata']=np.nan
            else:
                loss[mask]=255
                ploss[mask]=255
                meta['nodata']=255
    
            loss_out=bench_fn.replace('.tif', '_loss_v2.tif')
            with rio.open(loss_out, "w", **meta) as dest:
                    dest.write(loss, 1)
            
            ploss_out=bench_fn.replace('.tif', '_ploss_v2.tif')
            with rio.open(ploss_out, "w", **meta) as dest:
                    dest.write(ploss, 1)
    else:
        print('No input file exists')

    
    
   
    


