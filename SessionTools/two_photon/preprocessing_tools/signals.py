import os
import itertools

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d as sp_interp1d

from SessionTools.utilities import pol2cart, cart2pol

def binarize(signals, thresh = 3):
    
    ndims = len(signals.shape)
    if ndims==2:
        n_signals = signals.shape[1]
    else:
        n_signals = 1
        
        
    if n_signals>1:
        if isinstance(thresh,list) or isinstance(thresh, tuple):
            thresh = np.array(thresh)[np.newaxis,:]
            
        elif isinstance(thresh,np.array):
            if len(thresh.shape)==1:
                thresh = thresh[np.newaxis,:]
                
        else:
            pass
    
    return 1*(signals>thresh)
    
    
def rising_edges(binary_signals, axis=0, prepend=0, append=None):
    if append is None and prepend is None:
        return 1*(np.diff(binary_signals, axis=axis)>0)
    elif append is not None and prepend is None:
        return 1*(np.diff(binary_signals, axis=axis, append=append)>0)
    elif prepend is not None and append is None:
        return 1*(np.diff(binary_signals, axis=axis, prepend=prepend)>0)
    else:
        return 1*(np.diff(binary_signals, axis=axis, prepend=prepend, append=append)>0)

def falling_edges(binary_signals, axis=0, prepend=0, append=None):
    if append is None and prepend is None:
        return 1*(np.diff(binary_signals, axis=axis)<0)
    elif append is not None and prepend is None:
        return 1*(np.diff(binary_signals, axis=axis, append=append)<0)
    elif prepend is not None and append is None:
        return 1*(np.diff(binary_signals, axis=axis, prepend=prepend)<0)
    else:
        return 1*(np.diff(binary_signals, axis=axis, prepend=prepend, append=append)<0)

 
def change_default_column_names(df):   
    df.rename(columns = { ' Input 0': ' Start Trigger',
                          ' Input 1': ' Opto Trigger',
                          ' Input 2': ' FicTrac Cam Exp.',
                          ' Input 3': ' Fictrac Frame Proc.',
                          ' Input 4': ' Heading',
                          ' Input 5': ' Y/Index',
                          ' Input 6': ' Arena DAC1',
                          ' Input 7': ' Arena DAC2'},
              inplace=True)
        
    
def align_vr_2p(vr_df,frame_times):
    
    if ' Input 7' in vr_df.columns:
        change_default_column_names(vr_df)
        
    binary_columns = [' Start Trigger', 
                  ' Opto Trigger', 
                  ' FicTrac Cam Exp.',
                  ' FicTrac Frame Proc.']
    
    periodic_columns = [' Heading', 
                        ' Y/Index', 
                        ' Arena DAC1', 
                        ' Arena DAC2']
    max_voltage = 10
    
    # binarize trigger columns and find rising edges
    vr_df[binary_columns] = rising_edges(binarize(vr_df[binary_columns], thresh = (3,3,2,3)))
    
    vr_times = vr_df['Time(ms)'].to_numpy().ravel()
    
    # allocate downsampled dataframe
    ds_vr_df = pd.DataFrame(columns = vr_df.columns, index=np.arange(frame_times.shape[0]))
    ds_vr_df['Time(ms)'] = frame_times
    
    #interpolate binary columns
    # take cummulative sum, take nearest value, then take difference
    interp_nearest = sp_interp1d(vr_times, np.cumsum(vr_df[binary_columns],axis=0), axis=0, kind='nearest')
    ds_vr_df[binary_columns] = np.diff(interp_nearest(frame_times),axis=0, prepend=0)
    
    # interpolate periodic columns
    # convert to cartesian coordinates to be able to take the average
    cartesian_columns = []    
    for col in periodic_columns:
        phi_vec = np.maximum(0, 2*np.pi*vr_df[col].to_numpy().ravel()/max_voltage)
        rho_vec = np.ones((vr_df.shape[0],))
        x,y = pol2cart(rho_vec,phi_vec)
        vr_df[f'{col}_cartx'] = x
        vr_df[f'{col}_carty'] = y
        cartesian_columns.append(f'{col}_cartx')
        cartesian_columns.append(f'{col}_carty')
    
    interp_mean = sp_interp1d(vr_times, vr_df[cartesian_columns], axis=0, kind='linear')
    ds_vr_df[cartesian_columns] = interp_mean(frame_times)
    
    # convert back to polar coordinates
    for col in periodic_columns:
        _, ds_vr_df[col] = cart2pol(ds_vr_df[f'{col}_cartx'].to_numpy().ravel(),
                                    ds_vr_df[f'{col}_carty'].to_numpy().ravel())
    
    return ds_vr_df
    

def stim_artifact_frames(aligned_voltage_recording):
    pass

def extract_2p_timeseries(data, masks, n_rois):
    n_ch = data.shape[0]
    n_timepoints = data.shape[1]
    
    F = np.zeros((n_ch, n_rois, n_timepoints))
    
    for r in range(n_rois):
        mask = masks==r+1
        for ch, fr in itertools.product(range(n_ch),range(n_timepoints)):
            frame = data[ch, fr, :, :, :]
            F[ch, r, fr] = frame[mask].mean()
    return F


def dff(func_data, baseline_data=None, axis=1):
    
    if baseline_data is None:
        baseline_data = np.copy(func_data)
        
    
