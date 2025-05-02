
import rasterio
from rasterio.merge import merge
import glob
import os

wdir='/g/data/xc0/project/natint/'
outdir=wdir+'output/v1/predict_BRT'

resp='agb_australia_90m'
tile_size=50
tile_outdir=outdir+'/out_tiles/tiled_'+str(tile_size)+'km'

fns=glob.glob(tile_outdir+'/*'+resp+'*.tif')
print(f"{len(fns)} files found")

#%%

def merge_tiles(fns, resp, tile_outdir):
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

    output_fn=os.path.join(output_path, resp+'_mosaic.tif')
    with rasterio.open(output_fn, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster written to: {output_fn}")

#%%

merge_tiles(fns, resp, tile_outdir)

#%%