# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:00:13 2024

@author: mattg

* Adapt as required to new hierarchies
* Save as new branch?


WA other minimal use 1.3.0?

"""

import numpy as np
import rasterio as rio
from scipy.ndimage import binary_erosion


#%%

wdir='F:\\veg2_postdoc\\data\\reference\\'

#directory containing iucn classes, dea land cover, and national land use 
capad_dir=wdir+'National\\capad\\capad\\'

#%%

def read_raster_as_array(path):
    with rio.open(path) as src:
        array = src.read(1)  # Assuming single-band raster
    return array

#%%

"""
Shrinking window to reduce edge effects
Adjust to suit resolution of hierarchy layers
"""

#90 m 
#structure = np.ones((3, 3), dtype=bool) #90 m

#250 m 
structure = np.ones((3, 3), dtype=bool) #250 m


#%%
"""
Read rasters
National proof of concept layers
"""

#land use
lu_path=wdir+'National\\clum_national_50m\\clum_50m_2023_v2\\clum_50m_2023_v2_250m_3577.tif'

#protected areas
capad_path=capad_dir+'capad_250m_IUCN_cat_3577.tif'

#hcas inferred high-integrity areas
hcas_path=capad_dir+'HCAS_inferred_250m_3577.tif'

#DEA land cover
lc_path=capad_dir+'DEA_LC_250m_3577.tif'

#euc distance road dir
road_path=wdir+'national\\roads\\2024_12_NationalRoads_eucdist_250m_3577.tif'

#victoria native vegetation, to fill inconsistency in land use categories
vic_path=wdir+'VIC\\NVR2017_EXTENT_250m_3577.tif'

#nvis major vegetation groups, to discriminate mapped non-native vegetation
nvis_path=wdir+'National\\GRID_NVIS6_0_AUST_EXT_MVG\\GRID_NVIS6_0_AUST_EXT_MVG\\NVIS_MVG_E_250m_3577.tif'

#state classified raster for differential landuse treatment
st_path='F:\\veg2_postdoc\\data\\misc\\state_raster\\aus_states_250m.tif'

lu_array = read_raster_as_array(lu_path)
capad_array = read_raster_as_array(capad_path)
hcas_array = read_raster_as_array(hcas_path)
lc_array = read_raster_as_array(lc_path)
road_array = read_raster_as_array(road_path)
vic_array = read_raster_as_array(vic_path)
nvis_array = read_raster_as_array(nvis_path)
st_array = read_raster_as_array(st_path)


#%%

"""
Legend/coding:
    

lu == 110-134 is native veg

capad >= 1 is protected, as follows:
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

DEA land cover:
    
111		Cultivated Terrestrial Vegetation (CTV)
112		(Semi-)Natural Terrestrial Vegetation (NTV)
124		Natural Aquatic Vegetation (NAV)
215		Artificial Surface (AS)
216		Natural Bare Surface (NS)
220		Water

NVIS MVG:

25 Cleared, non-native vegetation, buildings 
29 Regrowth, modified native vegetation 
99 Unknown/no data 

Victoria NVR:
    
1 Native - woody cover (including heaths and woody wetlands)
Don't trust the native grassland and non-woody wetland classes

State classified raster:
    #ACT
1
    #NSW
2
    #TAS
3
    #VIC
4
    #SA
5
    #NT
6
    #QLD
7
    #WA
8

"""

#%%

#first pass, only iucn 1+2
print('First pass...')
p1=np.ones(lu_array.shape, dtype=np.uint8)

#controls
#land cover
p1[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#NVIS
p1[(nvis_array==25) | (nvis_array==29) | (nvis_array==99)]=0
#roads 500 m
p1[(road_array<500)]=0

#hierarchy
p1[~((capad_array==1) | (capad_array==12) | (capad_array==2))]=0

#shrink edges and export
shrunken_array = binary_erosion(p1 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array>=6000))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
meta['nodata']=255
outdir=wdir+'National\\V2\\national_pass1_v2.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p1

#%%

#second pass, only iucn 1+2+3+4 protected areas
print('Second pass...')
p2=np.ones(lu_array.shape, dtype=np.uint8)

#controls
#land cover
p2[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#NVIS
p2[(nvis_array==25) | (nvis_array==29) | (nvis_array==99)]=0
#roads 250 m
p2[(road_array<250)]=0

#hierarchy
p2[~((capad_array==1) | (capad_array==12) | (capad_array==2) | (capad_array==3) | (capad_array==4))]=0

shrunken_array = binary_erosion(p2 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array>=6000))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
meta['nodata']=255
outdir=wdir+'National\\V2\\national_pass2_v2.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p2

#%%

#third pass, all IUCN protected areas

print('Third pass...')
p3=np.ones(lu_array.shape, dtype=np.uint8)

#controls
#land cover
p3[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#NVIS
p3[(nvis_array==25) | (nvis_array==29) | (nvis_array==99)]=0
#roads 250 m
p3[(road_array<100)]=0

#hierarchy
p3[~((capad_array>0))]=0

shrunken_array = binary_erosion(p3 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array>=6000))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
meta['nodata']=255
outdir=wdir+'National\\V2\\national_pass3_v2.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p3

#%%

#fourth pass, all IUCN protected areas, include relativelty natural land uses
print('Fourth pass...')
p4=np.zeros(lu_array.shape, dtype=np.uint8)

#hierarchy
#capad
p4[((capad_array>0))]=1
#vic nvr extent
p4[(vic_array==1)]=1
#parks, in case missing in CAPAD
p4[(lu_array<=116)]=1
#managed resource protection. in vic this is the same as production native forest, so dont include yet
p4[(lu_array==120)]=1
p4[(lu_array==120) & (st_array==4)]=0
#other conserved area
p4[(lu_array==117)]=1
#remnant native cover
p4[(lu_array==133)]=1
#stock route
p4[(lu_array==132)]=1
#traditional indigenous uses
p4[(lu_array==125)]=1

#controls
#land cover
p4[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#NVIS
p4[(nvis_array==25) | (nvis_array==29) | (nvis_array==99)]=0
#roads 100 m
p4[(road_array<1)]=0

shrunken_array = binary_erosion(p4 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array>=6000))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
meta['nodata']=255
outdir=wdir+'National\\V2\\national_pass4_v2.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p4


#%%

#fifth pass, all IUCN protected areas, include relativelty natural land uses + production native
#no roads buffer
print('Fifth pass...')
p5=np.zeros(lu_array.shape, dtype=np.uint8)

#hierarchy
#capad
p5[((capad_array>0))]=1
#vic nvr extent
p5[(vic_array==1)]=1
#parks, in case missing in CAPAD
p5[(lu_array<=116)]=1
#managed resource protection. in vic this is the same as production native forest
p5[(lu_array==120)]=1
#other conserved area
p5[(lu_array==117)]=1
#remnant native cover
p5[(lu_array==133)]=1
#stock route
p5[(lu_array==132)]=1
#traditional indigenous uses
p5[(lu_array==125)]=1
#production native forest
p5[(lu_array==220)]=1
#grazing native vegetation
p5[(lu_array==210)]=1

#controls
#dea land cover
p5[~((lc_array==112) | (lc_array==124) | (lc_array==216))]=0
#NVIS
p5[(nvis_array==25) | (nvis_array==29) | (nvis_array==99)]=0
#no roads buffer

shrunken_array = binary_erosion(p5 == 1, structure=structure).astype(np.uint8)
ptot=(np.sum(shrunken_array==1)/np.sum(lu_array>=6000))*100
print('Total sample-able area: '+str(ptot)[0:4]+'% of study area')
with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
meta['nodata']=255
outdir=wdir+'National\\V2\\national_pass5_v2.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(shrunken_array, 1)

del p5


#%%

#make a combined layer

pc=np.zeros(lu_array.shape, dtype=np.uint8)

outdir=wdir+'National\\V2\\national_pass5_v2.tif'
p5=read_raster_as_array(outdir)
pc[p5==1]=5
del p5

outdir=wdir+'National\\V2\\national_pass4_v2.tif'
p4=read_raster_as_array(outdir)
pc[p4==1]=4
del p4

outdir=wdir+'National\\V2\\national_pass3_v2.tif'
p3=read_raster_as_array(outdir)
pc[p3==1]=3
del p3

outdir=wdir+'National\\V2\\national_pass2_v2.tif'
p2=read_raster_as_array(outdir)
pc[p2==1]=2
del p2

outdir=wdir+'National\\V2\\national_pass1_v2.tif'
p1=read_raster_as_array(outdir)
pc[p1==1]=1
del p1

with rio.open(lu_path) as lu_src:
    meta=lu_src.meta
meta['dtype']='uint8'
meta['nodata']=255
outdir=wdir+'National\\V2\\national_passes_combined_v2.tif'
with rio.open(outdir, 'w', **meta) as dest:
    dest.write(pc, 1)

#%%



