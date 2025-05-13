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
                          ' Input 2': ' Pump Trigger',
                          ' Input 3': ' Fictrac Frame Proc.',
                          ' Input 4': ' Heading',
                          ' Input 5': ' Y/Index',
                          ' Input 6': ' Arena DAC1',
                          ' Input 7': ' Arena DAC2'},
              inplace=True)
        
    
def align_vr_2p(vr_df,frame_times):
    
    # backwards compatibility
    if ' Input 7' in vr_df.columns:
        change_default_column_names(vr_df)
        
    binary_columns = [' Start Trigger', 
                  ' Opto Trigger', 
                  ' Pump Trigger',
                  ' FicTrac Frame Proc.']
    binary_columns = [col for col in binary_columns if col in vr_df.columns]
    
    periodic_columns = [' Heading', 
                        # ' Y/Index', 
                        ' Arena DAC1']#, 
                        # ' Arena DAC2']
    
    orig_columns = [' Y/Index', ' Arena DAC2']
    max_voltage = 10
    
    # binarize trigger columns and find rising edges
    # vr_df[binary_columns] = vr_df[binary_columns]
    
    vr_times = vr_df['Time(ms)'].to_numpy().ravel()
    
    #ToDo: make sure this isn't going to screw up a bunch of stuff
    #  add warning if difference is more than 1 frame
    max_time = np.max([vr_times[-1],frame_times[-1]])
    vr_times[-1] = max_time
    
    # allocate downsampled dataframe
    ds_vr_df = pd.DataFrame(columns = vr_df.columns, index=np.arange(frame_times.shape[0]))
    ds_vr_df['Time(ms)'] = frame_times
    
    #interpolate binary columns
    # take cummulative sum, take nearest value, then take difference
    interp_nearest = sp_interp1d(vr_times, np.cumsum(vr_df[binary_columns],axis=0), axis=0, kind='nearest')
    ds_vr_df[binary_columns] = np.diff(interp_nearest(frame_times),axis=0, prepend=0)
    
    interp_nearest = sp_interp1d(vr_times, vr_df[orig_columns], axis=0, kind='nearest')
    ds_vr_df[orig_columns] = interp_nearest(frame_times)
    
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
    raise NotImplementedError

def extract_2p_timeseries(data, masks, n_rois, bckgnd_mask=None,  max_proj = True):
    n_ch = data.shape[0]
    n_timepoints = data.shape[1]
    
    F = np.zeros((n_ch, n_rois, n_timepoints))
    
    for r in range(n_rois):
        print(r)
        mask = masks==r+1
        # mask = np.ma.masked_equal(masks,r+1)
        for ch, fr in itertools.product(range(n_ch),range(n_timepoints)):
            frame = data[ch, fr, :, :, :]
            if max_proj:
                frame = np.ma.masked_where(masks==r+1,frame)
                F[ch,r,fr] = np.amax(frame,axis=0).ravel().mean()
            else:
                F[ch, r, fr] = frame[mask].mean()
                
    
    notF = np.zeros((n_ch, n_timepoints))
    if bckgnd_mask is None:
        bckgnd_mask = masks<1
    else:
        bckgnd_mask = bckgnd_mask>0
    
    for ch, fr in itertools.product(range(n_ch),range(n_timepoints)):
            frame = data[ch, fr, :, :, :]
            # if max_proj:
            #     frame = np.ma.masked_where(masks==r+1,frame)
            #     notF[ch,fr] = np.amax(frame,axis=0).ravel().mean()
            # else:
            notF[ch, fr] = frame[bckgnd_mask].mean()
    return F, notF


def dff(func_data, baseline_data=None, axis=1):
    raise NotImplementedError
    # if baseline_data is None:
    #     baseline_data = np.copy(func_data)
        
def read_fictrac_dat(filename):
    names = ( 'col',
         'd rot x (cam)', 'd rot y (cam)', 'd rot z (cam)', 
         'd rot err (cam)', 
         'd rot x (lab)', 'd rot y (lab)', 'd rot z (lab)', 
         'abs rot x (cam)', 'abs rot y (cam)', 'abs rot z (cam)',
         'abs rot x (lab)', 'abs rot y (lab)', 'abs rot z (lab)',
         'integ x (lab)', 'integ y (lab)', 
         'integ heading (lab)', 
         'movement dir (lab)',
         'movement speed',
         'integ forward',
         'integ side',
         'timestamp',
         'seq counter',
         'd timestamp',
         'alt. timestamp')
    return pd.read_csv(filename, names = names, encoding_errors='ignore', low_memory=False)

def extract_fictrac_data(fictrac_df, vr_df, scan_pkl_dict):
    start_ind = fictrac_df.loc[fictrac_df['col']==scan_pkl_dict['start'][0]].index[0]
    
    #binarized fictrac frame processes time
    frame_pin = 1*(vr_df[' FicTrac Frame Proc.']>3)
    frame_pin_fall_edge = np.ediff1d(frame_pin, to_begin=0)<0
    
    n_frames = frame_pin_fall_edge.sum()
    fictrac_df_scan = fictrac_df[start_ind:start_ind+n_frames]
    vr_df_ft_frames = vr_df.loc[frame_pin_fall_edge,:]
    
    fictrac_df_scan.insert(1, 'Time(ms)', vr_df_ft_frames.loc[:,'Time(ms)'].to_list())
    return fictrac_df_scan

def align_fictrac_2p(ft_df, frame_times):
    '''
    
    '''
    
    
    
    periodic_columns = ['d rot x (cam)', 'd rot y (cam)', 'd rot z (cam)', 
         'd rot err (cam)', 
         'd rot x (lab)', 'd rot y (lab)', 'd rot z (lab)', 
         'abs rot x (cam)', 'abs rot y (cam)', 'abs rot z (cam)',
         'abs rot x (lab)', 'abs rot y (lab)', 'abs rot z (lab)',
         'integ heading (lab)', 'movement dir (lab)',]
    
    orig_columns = ['col',
                    'seq counter']
    
    cartesian_columns = ['integ x (lab)', 'integ y (lab)', 
                         'movement speed',
                        'integ forward',
                        'integ side']
    
    
    ft_times = ft_df['Time(ms)'].to_numpy().ravel()
    
    
    max_time = np.max([ft_times[-1],frame_times[-1]])
    ft_times[-1] = max_time
    
    # # allocate downsampled dataframe
    ds_ft_df = pd.DataFrame(columns = ft_df.columns, index=np.arange(frame_times.shape[0]))
    ds_ft_df['Time(ms)'] = frame_times
    
    
    
    interp_nearest = sp_interp1d(ft_times, ft_df[orig_columns], axis=0, kind='nearest')
    ds_ft_df[orig_columns] = interp_nearest(frame_times)
    
    # interpolate periodic columns
    # convert to cartesian coordinates to be able to take the average
  
    for col in periodic_columns:
        phi_vec = ft_df[col].to_numpy(dtype=float).ravel()
        rho_vec = np.ones((ft_df.shape[0],))
        x,y = pol2cart(rho_vec,phi_vec)
        ft_df[f'{col}_cartx'] = x
        ft_df[f'{col}_carty'] = y
        cartesian_columns.append(f'{col}_cartx')
        cartesian_columns.append(f'{col}_carty')
    
    interp_mean = sp_interp1d(ft_times, ft_df[cartesian_columns], axis=0, kind='linear')
    ds_ft_df[cartesian_columns] = interp_mean(frame_times)
    
    # convert back to polar coordinates
    for col in periodic_columns:
        _, ds_ft_df[col] = cart2pol(ds_ft_df[f'{col}_cartx'].to_numpy().ravel(),
                                    ds_ft_df[f'{col}_carty'].to_numpy().ravel())
    
    return ds_ft_df
    

