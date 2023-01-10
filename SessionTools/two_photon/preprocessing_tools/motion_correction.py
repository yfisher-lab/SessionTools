
import os

from joblib import Parallel, delayed

import skimage
from skimage.registration import phase_cross_correlation

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as spshift

import SessionTools as st


def make_ref_img(data, ref_channel):
    
    ref_intervals = np.arange(0,data.shape[1],round(data.shape[1]/10)).tolist()
    ref_frames = []
    for frame in ref_intervals:
        for i in range(20):
            ref_frames.append(frame+i)
        
    ref_stack = data[:, ref_frames,:,:,:]
    ref_img = skimage.img_as_float(ref_stack[ref_channel,:,:,:,:].mean(axis=0))
    ref_stack, _, _, _ = align_data_chunk(ref_stack, ref_img, 
                                        ref_channel=ref_channel, 
                                        in_place=True)
    return ref_stack.mean(axis=1)
    

def align_data_chunk(data_chunk, ref_img, ref_channel=0, in_place=True):
    
    n_ch = data_chunk.shape[0]
    chunk_size, n_zplanes = data_chunk.shape[1], data_chunk.shape[2]
    
    # shift = np.nan*np.zeros((2, chunk_size, n_zplanes))
    # error = np.nan*np.zeros((chunk_size, n_zplanes))
    # diffphase = np.nan*np.zeros((chunk_size, n_zplanes))
    
    def compute_shift(f,z):
        frame = skimage.img_as_float(data_chunk[ref_channel, f, z, :, :])
        with st.utilities.suppress_output(suppress_stderr=True):
            shift, error, diffphase = phase_cross_correlation(ref_img[z, :, :], frame, upsample_factor = 10, space='fourier')
        shifted_data = [spshift(data_chunk[ch,f,z,:,:], shift, order = 1, mode='reflect') for ch in range(n_ch)]
        return (f,z, shifted_data, shift, error, diffphase)
    
    
    shift_results = Parallel(n_jobs=-1)(delayed(compute_shift)(f,z) for f in range(chunk_size) for z in range(n_zplanes))
    
    if in_place:
        data_corr = data_chunk
    else:
        data_corr = 0*data_chunk
    
    data_corr, shifts, errors, diffphases = apply_shifts(data_corr, shift_results)
    return data_corr, shifts, errors, diffphases

def apply_shifts(data_corr, shift_results):
    n_ch = data_corr.shape[0]
    shifts = np.nan*np.zeros((2,*data_corr.shape[1:3]))
    errors = np.nan*np.zeros((data_corr.shape[1], data_corr.shape[2]))
    diffphases = errors = np.nan*np.zeros((data_corr.shape[1], data_corr.shape[2]))
    
    for (f, z, shifted_data, shift, error, diffphase) in shift_results:
        shifts[:,f,z] = shift
        errors[f,z] = error
        diffphases[f,z] = diffphase
        for ch in range(n_ch):
            data_corr[ch, f, z, :, :] = shifted_data[ch]
    return data_corr, np.array(shifts), np.array(errors), np.array(diffphases)
    