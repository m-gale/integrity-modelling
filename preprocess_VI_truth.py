# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:40:52 2025

@author: mattg
"""

import pandas as pd
import glob

wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg_postdoc\\'
gz_dir=wdir+'analysis\\nsw_v2_filtered\\filtered_df_fullmodel.csv'
filt_dir1='E:\\Outputs_V2\\*\\all_data\\*\\*filtered_input.csv'
filt_dir2='E:\\Outputs\\v2\\*\\*\\*filtered_input.csv'


gz=pd.read_csv(gz_dir)
#list(gz.keys())

gz=gz[['source',
    'Census Key',
    'Census DB ID',
    'SurveyName',
    'SiteNo',
    'ReplicateNo',
    'Replicate_preferred',
    'SurveyName_SiteNo_RepPref',
    'SubplotID',
    'CensusID',
    'SurveyID',
    'DateFirst',
    'Day',
    'Month',
    'Year',
    'Description',
    'Latitude_GDA94',
    'Longitude_GDA94',
    'plot_area_m2',
    'SCS',
    'CCS',
    'FCS',
    'mVI',
    'VI',
    'tot_richness',
    'tot_native_richness',
    'tot_cover',
    'tot_native_cover',
    'vegetationFormation',
    'vegetationClass',
    'PCT Assignment Category',
    'PCTID',
    'PCT Name',
    'rich_tree_TG',
    'rich_shrub_SG',
    'rich_grass_GG',
    'rich_forb_FG',
    'rich_fern_EG',
    'rich_other_OG',
    'rich_exotic',
    'rich_unidentified',
    'cover_tree_TG',
    'cover_shrub_SG',
    'cover_grass_GG',
    'cover_forb_FG',
    'cover_fern_EG',
    'cover_other_OG',
    'cover_exotic',
    'cover_unidentified',
    'BM_rich_tree_TG',
    'BM_rich_shrub_SG',
    'BM_rich_grass_GG',
    'BM_rich_forb_FG',
    'BM_rich_fern_EG',
    'BM_rich_other_OG',
    'BM_cover_tree_TG',
    'BM_cover_shrub_SG',
    'BM_cover_grass_GG',
    'BM_cover_forb_FG',
    'BM_cover_fern_EG',
    'BM_cover_other_OG',
    'UCS_cover_tree_TG',
    'UCS_cover_shrub_SG',
    'UCS_cover_grass_GG',
    'UCS_cover_forb_FG',
    'UCS_cover_fern_EG',
    'UCS_cover_other_OG',
    'UCS_rich_tree_TG',
    'UCS_rich_shrub_SG',
    'UCS_rich_grass_GG',
    'UCS_rich_forb_FG',
    'UCS_rich_fern_EG',
    'UCS_rich_other_OG',]]

filts1=glob.glob(filt_dir1)
filts2=glob.glob(filt_dir2)

for filt in filts1:
    fn=filt.split('\\')[-1].split('_GBRT')[0]
    if (fn in gz.keys()) & (fn+'_include' not in gz.keys()):
        print(fn)
        gz[fn+'_include']=0
        filt_csv=pd.read_csv(filt)
        gz.loc[gz['Census Key'].isin(filt_csv['Census.Key']), fn+'_include']=1
        print(str(sum(gz[fn+'_include']==1))+' points included')
        
for filt in filts2:
    fn=filt.split('\\')[-1].split('_GBRT')[0]
    if (fn in gz.keys()) & (fn+'_include' not in gz.keys()):
        print(fn)
        gz[fn+'_include']=0
        filt_csv=pd.read_csv(filt)
        gz.loc[gz['Census Key'].isin(filt_csv['Census.Key']), fn+'_include']=1
        print(str(sum(gz[fn+'_include']==1))+' points included')

gz.to_csv('C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\data\\bionet_truthing\\all_data_checkinclude.csv')
        
        