
import os

import numpy as np
from scipy.ndimage import gaussian_filter

import SessionTools as st




ref_channel = 0

ref_intervals = np.arange(0,data.shape[1],round(data.shape[1]/10)).tolist()
ref_frames = []
for frame in ref_intervals:
    for i in range(20):
        ref_frames.append(frame+i)
ref_img = skimage.img_as_float(data[ref_channel, ref_frames,:,:,:].mean(axis=0))


f = h5py.File(h5name)
data = f['/data']

from skimage.registration import phase_cross_correlation
from joblib import Parallel, delayed
from scipy.ndimage import shift as spshift

n_frames = metadata['size']['frames']
n_zplanes = metadata['size']['z_planes']
n_ch = metadata['size']['channels']
shift = np.zeros((2, n_frames, n_zplanes))
error = np.zeros((n_frames, n_zplanes))
diffphase = np.zeros((n_frames, n_zplanes))

chunk_size = int(np.minimum(4000,n_frames))
data_chunk = data[:,:chunk_size,:,:,:]
def compute_shift(f,z):
    frame = skimage.img_as_float(data_chunk[ref_channel, f, z, :, :])
    with st.utilities.suppress_output(suppress_stderr=True):
        shift[:, f,z], error[f,z], diffphase[f,z] = phase_cross_correlation(ref_img[z, :, :], frame, upsample_factor = 10, space= 'fourier')
    shifts = [spshift(data_chunk[ch,f,z,:,:], shift[:,f,z], order = 1, mode='reflect') for ch in range(n_ch)]
    return (f,z, shifts)
    
# align_z_stack(0,0)
results = Parallel(n_jobs = -1)(delayed(compute_shift)(f,z) for f in range(chunk_size) for z in range(n_zplanes))

for (f,z,shifts) in results:
    data_chunk[0,f,z,:,:] = shifts[0]
    data_chunk[1,f,z,:,:] = shifts[1]