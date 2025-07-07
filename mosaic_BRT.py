
import rasterio
from rasterio.merge import merge
import glob
import os
import re
import datetime


#nci
wdir='/g/data/xc0/project/natint/'
outdir=wdir+'output/v2/predict_BRT'

#local
wdir='C:\\Users\\mattg\\Documents\\ANU_HD\\veg2_postdoc\\scripts\\env1'
outdir="E:\\veg2_postdoc\\output\\predict_BRT"

#initial mosaics for residual analysis?
residuals=True
residuals=False

tile_size=250
#tile_dir=outdir+'/out_tiles/tiled_'+str(tile_size)+'km'
tile_dir=wdir+'/predict_BRT//out_tiles/tiled_'+str(tile_size)+'km'

cutoff_dt = datetime.datetime(2025, 6, 10, 9)
cutoff_ts = cutoff_dt.timestamp()

fns_tot = [
    fn for fn in os.listdir(tile_dir)
    if os.path.getmtime(os.path.join(tile_dir, fn)) > cutoff_ts
]

tile_outdir=outdir+'/out_tiles/tiled_'+str(tile_size)+'km_mosaic'
#len(fns_tot)

if os.path.exists(tile_outdir)==False:
    os.mkdir(tile_outdir)

resps = set()
for f in fns_tot:
    match = re.match(r'tile_\d+_(.+)\.tif', f)
    if match:
        resps.add(match.group(1))
print(resps)
len(resps)

#resp='bs_pc_50_2013-2024_mean_australia'
#resp='agb_australia_90m'
resp='Forest_height_2019_AUS'



#%%

def merge_tiles(fns, resp, tile_outdir, residuals):
    """
    Merges raster tiles into a single raster using rasterio.

    Parameters:
        tile_outdir (str): Directory containing raster tiles.
        resp (str): Substring to match the tile files for a specific response variable.
        out_fn (str): Output filename for the merged raster.

    Returns:
        output_path (str): Path to the merged raster file.
    """
    print(resp)
    if len(fns) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {search_pattern}")
    
    if residuals == True:
        output_fn=os.path.join(tile_outdir, resp+'_mosaic_residuals.tif')
    else:
        output_fn=os.path.join(tile_outdir, resp+'_mosaic.tif')

    #if os.path.isfile(output_fn)==False: 
    
    print('Sourcing fns...')
    src_files_to_mosaic = [rasterio.open(fp) for fp in fns]
    print('Merging...')
    mosaic, out_transform = merge(src_files_to_mosaic)

    print('Exporting...')
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "driver": "GTiff"
    })

    output_path = os.path.join(tile_outdir, 'mosaic')
    if os.path.exists(output_path)==False:
        os.mkdir(output_path)

    with rasterio.open(output_fn, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster written to: {output_fn}")

#%%

#output_path=tile_outdir
for resp in resps:
    try:
        fns=glob.glob(tile_dir+'/*'+resp+'*.tif')
        print(f"{len(fns)} files found")
        if os.path.isfile(tile_outdir+'\\'+resp+'_mosaic.tif')==False:
            merge_tiles(fns, resp, tile_outdir, residuals=False)
    except:
        print('Error')

#%%