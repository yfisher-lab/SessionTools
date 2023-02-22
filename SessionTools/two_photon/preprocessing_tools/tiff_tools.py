"""Library for reading in TIFFs ripped from Bruker raw data."""
import os
import logging
import pathlib

import numpy as np
import dask
from dask import diagnostics
import dask.array as da

from skimage.io import imread
from SessionTools import utilities

logger = logging.getLogger(__name__)

class TiffToolsError(Exception):
    """Error while processing tiff files"""

def read(base, size, layout, first_chan=1):
    """Read 2p dataset of TIFF files into a dask.Array."""
    # shape_yx = (size['y_px'], size['x_px'])
    # dtype = read_file(base, 0, channel, 0).dtype

    num_cycles, num_z_planes, num_ch = layout['sequences'], size['z_planes'], size['channels']
    
    def skimread(file):
        """suppress juliandate reading errors"""
        with utilities.suppress_output(suppress_stderr=True):
            return imread(file)
        
    filenames = []
    # 
    for cycle in range(1,num_cycles+1):
        _filenames = []
        for frame in range(1,num_z_planes+1):
            _frame = []
            if num_ch == 1:
                _frame.append(str(base) + f'_Cycle{cycle:05d}_Ch{first_chan}_{frame:06d}.ome.tif')
            else:
                for ch in range(1,num_ch+1):
                    _frame.append(str(base) + f'_Cycle{cycle:05d}_Ch{ch}_{frame:06d}.ome.tif')
            _filenames.append(_frame)
        # filenames[cycle].append(str(basename_input) + f'_Cycle{cycle+1:05d}_Ch{channel}_{frame+1:06d}.ome.tif')
        filenames.append(_filenames)
    # replace first tiff to avoid sizing errors
    # print(filenames[0])
    filenames[0][0][0] = filenames[0][1][0]    
        
    
    logger.info('Found tiff files (channels: %i, frames: %i, z_planes: %i' % (num_cycles, num_z_planes, num_ch))
    
    def read_one_image(block_id, filenames=filenames):
        # print(block_id)
        path = filenames[block_id[1]][block_id[2]][block_id[0]]
        image = skimread(path)
        return np.expand_dims(image, axis=(0,1,2))
    
    logger.info('Mapping dask array...')
    sample = skimread(filenames[0][0][0])
    data = da.map_blocks(
        read_one_image,
        dtype=sample.dtype,
        chunks=((1,)*num_ch,
                (1,)*num_cycles, 
                (1,)*num_z_planes, 
                *sample.shape)
        )
    logger.info('Completed mapping dask array')
    
    return data


def convert_to_hdf5(data, hdf5_outname, hdf5_key = '/data', overwrite=False):
    """_summary_

    Args:
        data (_type_): _description_
        hdf5_outname (_type_): _description_
        hdf5_key (str, optional): _description_. Defaults to '/data'.
    """
    if isinstance(hdf5_outname,str):
        hdf5_outname = pathlib.Path(hdf5_outname)
    
    if os.path.exists(hdf5_outname):
        if overwrite:
            logger.info(f'{hdf5_outname} exists. Overwriting')
            os.remove(hdf5_outname)
        else:
            raise TiffToolsError(f'{hdf5_outname} exists. To overwrite set overwrite=True')
        
    
    with diagnostics.ProgressBar():
        
        logger.info('Writing data to %s', hdf5_outname)
        # ensure parent directory exists
        os.makedirs(hdf5_outname.parent, exist_ok=True)
        data.to_hdf5(hdf5_outname, hdf5_key)
        
