"""Library for reading in TIFFs ripped from Bruker raw data."""
import os
import logging
import warnings
import h5py

import dask
from dask import diagnostics
import dask.array as da

from skimage.io import imread

logger = logging.getLogger(__name__)


def read(base, size, layout, channel):
    """Read 2p dataset of TIFF files into a dask.Array."""
    shape_yx = (size['y_px'], size['x_px'])
    dtype = read_file(base, 0, channel, 0).dtype

    num_cycles = layout['sequences']
    frames_are_z = num_cycles == 1
    
    lazy_imread = dask.delayed(read_file, pure=True)

    data_cycles = []
    for cycle in range(num_cycles):
        data_frames = []
        for frame in range(layout['frames_per_sequence']):

            # Reading the first OME TIFF file is slow, so we substitute the following frame/cycle:
            # - use the next frame if a single-cycle where frames are z-planes
            # - use the next cycle if multi-cycle
            if frames_are_z:
                if frame == 0:
                    frame = 1
            else:
                if cycle == 0:
                    cycle = 1

            lazy_image = lazy_imread(base,cycle,channel,frame)
            # data_frames.append(lazy_image)
            # lazy_image = dask.delayed(read_file)(base, cycle, channel, frame)
            # data_frames.append(lazy_image)
            data_frames.append(da.from_delayed(lazy_image, shape_yx, dtype=dtype))
        data_cycles.append(da.stack(data_frames))

    data = da.stack(data_cycles)
    if frames_are_z:
        data = data.swapaxes(0, 1)

    logger.info('Found data with shape(frames, z_planes, y_pixels, x_pixels): %s', data.shape)
    return data


def read_file(base, cycle, channel, frame):
    """Read in one TIFF file."""
    fname = str(base) + f'_Cycle{cycle+1:05d}_Ch{channel}_{frame+1:06d}.ome.tif'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in true_divide", RuntimeWarning)
        return imread(fname)
    
    
def unlink(fname):
    """Helper script to delete a file."""
    try:
        os.remove(fname)
    except OSError:
        pass


def convert_to_hdf5(data, hdf5_outname, hdf5_key = '/data'):
    """_summary_

    Args:
        data (_type_): _description_
        hdf5_outname (_type_): _description_
        hdf5_key (str, optional): _description_. Defaults to '/data'.
    """
    
    
    with diagnostics.ProgressBar():
        
        logger.info('Writing data to %s', hdf5_outname)
        unlink(hdf5_outname)
        os.makedirs(hdf5_outname.parent, exist_ok=True)
        data.to_hdf5(hdf5_outname, hdf5_key)
        
        
# def convert(data, fname_data, df_artefacts=None, fname_uncorrected=None):
#     """Convert TIFF files from 2p dataset in HDF5.  Optionally create artefact-removed dataset."""
#     # Important: code expects no chunking in z, y, z -- need to have -1 for these dimensions.
#     data = data.rechunk((64, -1, -1, -1))  # 64 frames will be processed together for artefact removal.

#     with diagnostics.ProgressBar():
#         if df_artefacts is None:
#             logger.info('Writing data to %s', fname_data)
#             unlink(fname_data)
#             os.makedirs(fname_data.parent, exist_ok=True)
#             data.to_hdf5(fname_data, HDF5_KEY)
#         else:
#             # This writes 2 hdf5 files, where the 2nd one depends on the same data being
#             # written to the first.  Ideally, both would be written simultaneously, but
#             # that cannot be done using dask.  Instead, the 1st file is written and then
#             # read back to write the 2nd one.
#             logger.info('Writing uncorrected data to %s', fname_uncorrected)
#             unlink(fname_uncorrected)
#             os.makedirs(fname_uncorrected.parent, exist_ok=True)
#             data.to_hdf5(fname_uncorrected, HDF5_KEY)

#             logger.info('Writing corrected data to %s', fname_data)
#             with h5py.File(fname_uncorrected, 'r') as hfile:
#                 arr = da.from_array(hfile[HDF5_KEY])
#                 # Depth of 1 in the first coordinate means to bring in the frames before and after
#                 # the chunk -- needed for doing diffs.
#                 depth = (1, 0, 0, 0)
#                 data_corrected = arr.map_overlap(remove_artefacts,
#                                                  depth=depth,
#                                                  dtype=data.dtype,
#                                                  df=df_artefacts,
#                                                  mydepth=depth)
#                 unlink(fname_data)
#                 os.makedirs(fname_data.parent, exist_ok=True)
#                 data_corrected.to_hdf5(fname_data, HDF5_KEY)

# # # # D-LAB stim artefact removal
# def remove_artefacts(chunk, df, mydepth, block_info):
#     """Remove artefacts from a chunk representing a set of frames."""
#     chunk = chunk.copy()
#     frame_min, frame_max = block_info[0]['array-location'][0]

#     # The array-location is not the frame number -- it is offset by depth when using map_overlap.
#     frame_chunk = block_info[0]['chunk-location'][0]
#     frame_offset = mydepth[0] * (1 + 2 * frame_chunk)
#     frame_min -= frame_offset
#     frame_max -= frame_offset

#     for index, frame in enumerate(range(frame_min, frame_max)):
#         # Skip first/last frames, which are just the edge frames pulled in to allow
#         # computation using before/after.
#         if index in (0, chunk.shape[0] - 1):
#             continue

#         # Skip if the frame does not have an artefact.
#         if frame not in df.index:
#             continue

#         # Use `frame:frame` so the following slice always returns a frame.  Using just `frame`
#         # would lead to a series being returned if there was only one present.
#         for row in df.loc[frame:frame].itertuples():
#             y_slice = slice(int(row.y_min), int(row.y_max) + 1)
#             before = chunk[index - 1, row.z_plane, y_slice]
#             after = chunk[index + 1, row.z_plane, y_slice]
#             chunk[index, row.z_plane, y_slice] = (before + after) / 2
#     return chunk