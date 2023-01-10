import os
import numpy as np


def align_vr_2p(voltage_recording,twop_frametimes,n_zplanes):
    pass

def stim_artifact_frames(aligned_voltage_recording):
    pass

def extract_2p_timeseries(masks, data):
    pass


def dff(func_data, baseline_data=None, axis=1):
    
    if baseline_data is None:
        baseline_data = np.copy(func_data)
        
    
