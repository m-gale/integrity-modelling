# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:00:13 2024

@author: mattg
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from scipy.ndimage import binary_erosion
import copy


#%%

wdir='F:\\veg2_postdoc\\data\\reference\\'
capad_dir=wdir+'National\\capad\\capad\\'

#%%

def read_raster_as_array(path):
    with rio.open(path) as src:
        array = src.read(1)  # Assuming single-band raster
    return array

#30 m 
structure = np.ones((5, 5), dtype=bool) #60 m

#90 m 
structure = np.ones((3, 3), dtype=bool) #90 m

#250 m 
structure = np.ones((3, 3), dtype=bool) #250 m


#%%
"""
National POC
"""

lu_path=capad_dir+'NLUM_250m_3577.tif'
capad_path=capad_dir+'capad_250m_IUCN_cat_3577.tif'
hcas_path=capad_dir+'HCAS_inferred_250m_3577.tif'
lc_path=capad_dir+'DEA_LC_250m_3577.tif'

lu_array = read_raster_as_array(lu_path)
capad_array = read_raster_as_array(capad_path)
hcas_array = read_raster_as_array(hcas_path)
lc_array = read_raster_as_array(lc_path)


#%%

#lu == 110-134 is native veg
#capad >= 1 is protected, as follows:
# shapefile.loc[shapefile['IUCN']=='Ia', 'iucn_code']=1
# shapefile.loc[shapefile['IUCN']=='Ib', 'iucn_code']=12
# shapefile.loc[shapefile['IUCN']=='II', 'iucn_code']=2
# shapefile.loc[shapefile['IUCN']=='III', 'iucn_code']=3
# shapefile.loc[shapefile['IUCN']=='IV', 'iucn_code']=4
# shapefile.loc[shapefile['IUCN']=='V', 'iucn_code']=5
# shapefile.loc[shapefile['IUCN']=='VI', 'iucn_code']=6
# shapefile.loc[shapefile['IUCN']=='NAS', 'iucn_code']=7
# shapefile.loc[shapefile['IUCN']=='NA', 'iucn_code']=8
# shapefile.loc[shapefile['IUCN']=='NR', 'iucn_code']=9

#stats, as required
#tot=np.nansum(capad_array>=1)
#(np.nansum(capad_array==1)+np.nansum(capad_array==12))/tot*100
#(np.nansum(capad_array==2)+np.nansum(capad_array==3))/tot*100
#(np.nansum(capad_array>=4))/tot*100

"""
DEA land cover:
    
111		Cultivated Terrestrial Vegetation (CTV)
112		(Semi-)Natural Terrestrial Vegetation (NTV)
124		Natural Aquatic Vegetation (NAV)
215		Artificial Surface (AS)
216		Natural Bare Surface (NS)
220		Water

"""

#%%

#first pass, only hcas
print('First pass...')
p1=np.ones(lu_array.shape, dtype=np.uint8)
#land cover
p1[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#hcas
p1[((hcas_array==0))]=0

shrunken_array = binary_erosion(p1 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'National\\V1\\national_pass1_v1.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p1

#%%

#second pass, only iucn 1+2 protected areas, no hcas limiting requirement
print('Second pass...')
p2=np.ones(lu_array.shape, dtype=np.uint8)
#land use
p2[~((lu_array>=110) & (lu_array<=134))]=0
#protected areas
p2[~((capad_array==1) | (capad_array==12) | (capad_array==2))]=0
#land cover
p2[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#hcas areas still ok
p2[((hcas_array==1))]=1

shrunken_array = binary_erosion(p2 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'National\\V1\\national_pass2_v1.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p2

#%%

#third pass, only iucn 1-4 protected areas, no hcas limiting requirement
print('Third pass...')
p3=np.ones(lu_array.shape, dtype=np.uint8)
#land use
p3[~((lu_array>=110) & (lu_array<=134))]=0
#protected areas
p3[~((capad_array==1) | (capad_array==12) | (capad_array==2) | (capad_array==3) | (capad_array==4))]=0
#land cover
p3[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#hcas areas still ok
p3[((hcas_array==1))]=1

shrunken_array = binary_erosion(p3 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'National\\V1\\national_pass3_v1.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p3

#%%

#fourth pass, all IUCN protected areas, no hcas limiting requirement
print('Fourth pass...')
p4=np.ones(lu_array.shape, dtype=np.uint8)
#land use
p4[~((lu_array>=110) & (lu_array<=134))]=0
#protected areas
p4[~((capad_array>0))]=0
#land cover
p4[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#hcas areas still ok
p4[((hcas_array==1))]=1

shrunken_array = binary_erosion(p4 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'National\\V1\\national_pass4_v1.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p4

#%%

#fifth pass, all IUCN protected areas, no hcas limiting requirement, include relatively natural land uses
print('Fifth pass...')
p5=np.ones(lu_array.shape, dtype=np.uint8)
#land use
p5[~((lu_array>=110) & (lu_array<=222))]=0
#land cover
p5[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#hcas areas still ok
p5[((hcas_array==1))]=1

shrunken_array = binary_erosion(p5 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'National\\V1\\national_pass5_v1.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p5

#%%




#%%

#sixth pass, all IUCN protected areas, no hcas limiting requirement, include relatively natural land uses
print('Fifth pass...')
p5=np.ones(lu_array.shape, dtype=np.uint8)
#land use
p5[~((lu_array>=110) & (lu_array<=222))]=0
#land cover
p5[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#hcas areas still ok
p5[((hcas_array==1))]=1

shrunken_array = binary_erosion(p5 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'National\\V1\\national_pass5_v1.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p5

#%%

#second pass, include iucn 2 and 3 protected areas
print('Second pass...')
p2=np.ones(lu_array.shape, dtype=np.uint8)
p2[~((lu_array>=110) & (lu_array<=134))]=0
#np.sum(p2==1)
p2[mvs_array!=1]=0
p2[nvace_array!=1]=0
p2[~((capad_array==1) | (capad_array==12) | (capad_array==2) | (capad_array==3))]=0
#shrink the edges
structure = np.ones((5, 5), dtype=bool) #60 m
shrunken_array = binary_erosion(p2 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
shrunken_array_br=shrunken_array*br_array
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'NSW\\nsw_pass2_v3.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array_br, 1)

del p2


#third pass, include iucn 4+ protected areas
print('Third pass...')
p3=np.ones(lu_array.shape, dtype=np.uint8)
p3[~((lu_array>=110) & (lu_array<=134))]=0
#np.sum(p3==1)
p3[mvs_array!=1]=0
p3[nvace_array!=1]=0
p3[~capad_array>=1]=0
#shrink the edges
structure = np.ones((5, 5), dtype=bool) #60 m
shrunken_array = binary_erosion(p3 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
shrunken_array_br=shrunken_array*br_array
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'NSW\\nsw_pass3_v3.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array_br, 1)

del p3


#fourth pass, remove protected areas requirement, relatively natural environments included
print('Fourth pass...')
p4=np.ones(lu_array.shape, dtype=np.uint8)
p4[~((lu_array>=110) & (lu_array<=222))]=0
#np.sum(p3==1)
p4[mvs_array!=1]=0
p4[nvace_array!=1]=0
#shrink the edges
structure = np.ones((5, 5), dtype=bool) #60 m
shrunken_array = binary_erosion(p4 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array<=0))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
shrunken_array_br=shrunken_array*br_array
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
outdir=wdir+'NSW\\nsw_pass4_v3.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array_br, 1)

del p4
del lu_array, mvs_array, nvace_array


#%%


