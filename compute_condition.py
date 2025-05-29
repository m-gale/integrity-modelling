# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:44:56 2025

@author: mattg
"""

import glob
import os
import rasterio as rio
#from osgeo import gdal
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

#local
wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\'
bench_dir=wdir+'scrap\\'
ext_dir='F:\\veg2_postdoc\\raster_subset_v3\\'
resp_csv_path=wdir+'data\\responses_described_v2.csv'
outdir = 'C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scrap\\'

#nci
wdir='/g/data/xc0/project/natint/'
bench_dir=wdir+'output/v2/predict_BRT/out_tiles/tiled_50km_mosaic'
ext_dir='/scratch/xc0/mg5402/raster_subset_v3/'
resp_csv_path=wdir+'input/responses_described_v3.csv'
outdir=wdir+'output/v2/compute_condition'

#%%
#

if os.path.exists(outdir)==False:
    os.mkdir(outdir)
    
resps=pd.read_csv(resp_csv_path)
print(resps)

print('Response mosaics:')
print(os.listdir(bench_dir))

#%%

#resp='Forest_height_2019_AUS'
resp='agb_australia_90m'
resp='Forest_height_2019_AUS'
resp='wcf_wagb_90m_v2'
#bench_fn=bench_dir+resp+'_250m_national_v2.tif'

for resp in resps['Response']:
    print(resp)
    bench_fn=bench_dir+'/'+resp+'_mosaic.tif'
    ext_fn=ext_dir+resp+'.tif'
    if os.path.isfile(bench_fn):
        with rio.open(bench_fn) as bench_src:
            bench=bench_src.read(1)
            meta=bench_src.meta
            target_resolution=bench_src.transform[0]
            
        out_filename = outdir+'/'+resp
        if os.path.isfile(out_filename+'_loss.tif'):
            print('Out files already exist')
        else:            
            with rio.open(ext_fn) as src:
                ext_transform=src.transform
    
            with rio.open(bench_fn) as src:
                bench_transform=src.transform
    
            if ext_transform != bench_transform:
                raise ValueError('Error: unequal benchmark and extant raster dimensions')
            else:
                with rio.open(ext_fn) as ext_src:
                    ext=ext_src.read(1)
                    data = ext_src.read(1, masked=True)  
                    valid_data = data.compressed()  
                    #check whether int or fload
                    sample = np.random.choice(valid_data, 10000, replace=False)
                    if np.all(np.isclose(sample, np.round(sample))):
                        is_int=True
                    else:
                        is_int=False
                del data
     
                with rio.open(bench_fn) as bench_src:
                    bench=bench_src.read(1)
            
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
        
                loss_out=outdir+'/'+resp+'_loss.tif'
                with rio.open(loss_out, "w", **meta) as dest:
                        dest.write(loss, 1)
                print('Exported '+loss_out)
                
                ploss_out=outdir+'/'+resp+'_ploss.tif'
                with rio.open(ploss_out, "w", **meta) as dest:
                        dest.write(ploss, 1)
                print('Exported '+ploss_out)



    
#%%
    
   
    


