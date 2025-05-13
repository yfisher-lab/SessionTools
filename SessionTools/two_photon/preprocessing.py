from abc import ABC, abstractmethod
import pathlib
from glob import glob
import itertools

import dask.dataframe as dd
import h5py
import numpy as np
import scipy as sp
import pandas as pd
import napari
import cloudpickle
from sklearn.linear_model import LinearRegression as LinReg

from .preprocessing_tools import *
from .params import *
from .. import utilities as u

class Preprocess(ABC):
    
    def __init__(self,session_info=None, 
                 bruker_base_dir=BRUKER_BASE_DIR, bruker_dir = None,
                 fictrac_base_dir=FICTRAC_BASE_DIR, fictrac_dir=None,
                 fictrac_scan_num = None, fictrac_dat_file = None,
                 output_base_dir=OUTPUT_BASE_DIR, 
                 fictrac_pkl_path=None, **kwargs):
        
        """_summary_
        """
        # make sure minimal keys are in session_info and assign them
        min_keys = ('date', 'genotype_dir', 'fly', 
                    'session', 'full_genotype', 'ecl date')
        u.validate_dict_keys(session_info, min_keys)
        self.session_info = {k: v for k, v in session_info.items()} 
        
        
        gd = self.session_info['genotype_dir']
        d = self.session_info['date']
        f = self.session_info['fly']
        s = self.session_info['session']
        
        # set bruker directory 
        if bruker_dir is None:
            self.bruker_dir = bruker_base_dir.joinpath(f'{gd}/{d}/{f}/{s}/{s}')
        else:
            self.bruker_dir = pathlib.PurePath(bruker_dir)
            
        # set fictrac directories
        if fictrac_dir is None:
            self.fictrac_dir = fictrac_base_dir.joinpath(f'{gd}/{d}')
        else:
            self.fictrac_dir = pathlib.PurePath(fictrac_dir)
            
        if fictrac_dat_file is None:
            self.fictrac_path = self.fictrac_dir.joinpath(f'/{f}')
        else:
            self.fictrac_path = self.fictrac_dir.joinpath(fictrac_dat_file)
            
        if fictrac_pkl_path is None:
               
            if fictrac_scan_num is None:
                fictrac_scan_num = int(s[-3:])
            self.fictrac_pkl_path = self.fictrac_dir.joinpath(f'{f}_scan{fictrac_scan_num}.pkl')
        else:
            self.fictrac_pkl_path = fictrac_pkl_path
        
        
        # output directory
        output_base_dir = pathlib.PurePath(output_base_dir)
        self.output_dir = output_base_dir.joinpath(f'{gd}/{d}/{f}/{s}')
        
        self.metadata = None
        
        self.h5path_raw = None
        self.h5path_motcorr = None
        
        self.ref_img = None
        
        self.voltage_recording_path = None
        self.voltage_recording_aligned = None
        
        self.fictrac_aligned = None
     
        
        
    def extract_metadata(self, overwrite=False):
        """_summary_

        Args:
            overwrite (bool, optional): _description_. Defaults to False.
        """
        # extract metadata
        if (self.metadata is None) or overwrite:
            self.metadata = bruker_metadata.read(self.bruker_dir)
       
    def bruker_to_h5(self, first_chan=1, overwrite=False):
        """_summary_

        Args:
            outdir (_type_, optional): _description_. Defaults to pathlib.Path('/media/mplitt/SSD_storage/2P_scratch').
            first_chan (int, optional): _description_. Defaults to 1.
            overwrite (bool, optional): _description_. Defaults to False.
        """
        
        
        gd = self.session_info['genotype_dir']
        d = self.session_info['date']
        f = self.session_info['fly']
        s = self.session_info['session']
        
        self.h5path_raw = pathlib.Path(self.output_dir.joinpath(f'data.h5'))
        
        
        if (not self.h5path_raw.is_file()) or overwrite:
            tiff_data = tiff_tools.read( self.bruker_dir, 
                                        self.metadata['size'],
                                        self.metadata['layout'],
                                        first_chan=first_chan)
            
            tiff_tools.convert_to_hdf5(tiff_data, self.h5path_raw, overwrite=True)
        
           
    @property
    def data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # add that it looks for motion corrected h5 file
        with h5py.File(self.h5path_raw, 'r') as f:
            return f['/data'][:]
        
    def motion_correct(self, data, ref_channel=0, write_to_h5=True):
        """_summary_

        Args:
            data (_type_): _description_
            ref_channel (int, optional): _description_. Defaults to 0.
            write_to_h5 (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        self.ref_img = motion_correction.make_ref_img(data, ref_channel)
        data_corr, shifts, error, diffphase = motion_correction.align_data_chunk(data, self.ref_img[ref_channel,:,:,:])

        if write_to_h5:
            
            self.h5path_motcorr = pathlib.Path(self.h5path_raw.with_suffix('').as_posix() + '_motcorr.h5')
            with h5py.File(self.h5path_motcorr, 'w') as f:
                f.create_dataset('data', data=data_corr)
                f.create_dataset('shifts', data=shifts)
                f.create_dataset('error', data=error)
                f.create_dataset('diffphase', data=diffphase)
        
        return data_corr, shifts, error, diffphase  
        
        
    @property
    def data_corr(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with h5py.File(self.h5path_motcorr, 'r') as f:
            return f['/data'][:]
        
    
    def align_voltage_recording(self):
        """_summary_
        """
        csv_files = glob(f'{self.bruker_dir.parent.as_posix()}/*.csv')
        self.voltage_recording_path = pathlib.Path(csv_files[0])
        df = dd.read_csv(self.voltage_recording_path).compute()  
        
        frame_times = np.array(self.metadata['frame_times']).mean(axis=-1)*1000
        self.voltage_recording_aligned = signals.align_vr_2p(df,frame_times)
    
    def align_fictrac(self):
        """
        """        
        
        with open(self.fictrac_pkl_path, 'rb') as file:
            ft_scan_info = cloudpickle.load(file)
        
        
        ft_df = signals.extract_fictrac_data(signals.read_fictrac_dat(self.fictrac_path),
                                            dd.read_csv(self.voltage_recording_path).compute(),
                                            ft_scan_info)
        
        frame_times = np.array(self.metadata['frame_times']).mean(axis=-1)*1000
        self.fictrac_aligned = signals.align_fictrac_2p(ft_df, frame_times)
        
    
    @abstractmethod
    def save(self):
        pass
        
    
    @classmethod
    @abstractmethod
    def from_file(filename):
        pass
    
    
    @abstractmethod
    def open_napari(self, check_for_existing=True):
        pass
    
    @abstractmethod
    def extract_timeseries(self):
        pass
    

class EBImagingSession(Preprocess):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.napari_labels_layers = None
        self.napari_layers = None
        self.timeseries = None
        
        
        
    def save(self):
        
        with open(self.output_dir.joinpath('preprocess.pkl'), 'wb') as file:
            cloudpickle.dump(self, file)
            # file.write(serialized_data)
        
    @classmethod
    def from_file(cls, filename):
        
        with open(filename, 'rb') as file:
            inst = pd.read_pickle(file)
            # inst = cloudpickle.load(file) 
        # loaded_dict = json.loads(loaded_data)
        # inst = cls(**loaded_dict)
        
        # for k, v in loaded_dict.items():
        #     if not hasattr(inst, k):                
        #         setattr(inst, k, v)
        
        return inst
        
    
     
    def open_napari(self, check_for_existing=False, path=None):
        if check_for_existing:
            if path is None:
                raise ValueError("Checking for existing napari session, but path is None")
            elif pathlib.Path.exists(path):
                return napari_tools.EBSession().open_existing_session(path)
            else:
                return napari_tools.EBSession().new_session(self.ref_img)
        else:
            return napari_tools.EBSession().new_session(self.ref_img)
    
    def save_napari(self, nap, overwrite=True):
        nap.save_layers(self.napari_output_path)

    def napari_pickle_noviewer(self, filename):
        napari_tools.EBSession().topickle_noviewer(self.ref_img, filename)
    
    def get_layers(self, nap):
        self.napari_labels_layers = {}
        self.napari_layers = {}
        for layer in nap.viewer.layers: 
            if isinstance(layer, napari.layers.Labels):
                self.napari_labels_layers[layer.name] = layer.data
            else:
                self.napari_layers[layer.name] = layer.data
        

    def extract_timeseries(self, data=None, max_proj=False):
        self.timeseries = {}
        if data is None:
            data = self.data_corr
            
        n_ch = data.shape[0]
        n_timepoints = data.shape[1]
        
        def _extract_ts(masks, _max_proj=False):
            n_rois = int(np.amax(masks.ravel()))
            F = np.zeros((n_ch, n_rois, n_timepoints))
            for r in range(n_rois):
                mask = masks==r+1
                
                for ch, fr in itertools.product(range(n_ch), range(n_timepoints)):
                    
                    frame = data[ch, fr, :, :, :]
                    # print(mask.shape, frame.shape)
                    if _max_proj:
                        _frame = np.amax(frame,axis=0)
                        _mask = np.amax(mask,axis=0)
                        # print(mask.shape, frame.shape)
                        F[ch,r,fr] = _frame[_mask].mean()
                        # F[ch,r,fr] = np.amax(frame[mask].mean(axis=-1).mean(axis=-1))
                    else:
                        F[ch,r,fr] = frame[mask].mean()
            return F
        
        
        for name, mask_arr in self.napari_labels_layers.items():
            # if name == 'background':
                
            #     self.timeseries[name]=_extract_ts(mask_arr, _max_proj=False)
            # else:
            self.timeseries[name]=_extract_ts(mask_arr, _max_proj=max_proj)
            
            
    def calculate_zscored_F(self, ts_key,
                      exp_detrend = True, 
                      background_ts='background',
                      other_ts_2_subtract = None,
                      channel = None,
                      add_to_timeseries_dict = True,
                      new_ts_name = None,
                      zscore = True,
                      reg_other_channel=False):
        
        if channel is not None:
            F = self.timeseries[ts_key][channel:channel+1, :, :]
        else: 
            F = self.timeseries[ts_key]
        
        # make sure background is the same length
        if background_ts is not None:
            if isinstance(background_ts,str):
                if background_ts not in self.timeseries.keys():
                    raise ValueError(f"key {background_ts} not in self.timeseries ")
            elif isinstance(background_ts, np.ndarray):
                assert np.broadcast(F, background_ts), f"background_ts must be broadcastable to {ts_key}"
            else:
                raise ValueError("background_ts must be a string or np array")
        
        # make sure other ts is correct shape
        if other_ts_2_subtract is not None:
            if isinstance(other_ts_2_subtract, str):
                if not other_ts_2_subtract in self.timeseries.keys():
                    raise ValueError(f"key {other_ts_2_subtract} not in self.timeseries")
            elif isinstance(other_ts_2_subtract, np.ndarray):
                assert np.broadcast(F, other_ts_2_subtract), f"other_ts_2_subtract must be broadcastable to {ts_key}"
        
        # convert keys to arrays
        if background_ts is not None:
            if isinstance(background_ts, str):
                background_ts = self.timeseries[background_ts]
            if exp_detrend:
                background_ts = np.log(background_ts+1E-3)
                
            background_ts = background_ts*np.ones(F.shape)
            
                
        if other_ts_2_subtract is not None:
            if isinstance(other_ts_2_subtract, str):
                other_ts_2_subtract = self.timeseries[other_ts_2_subtract]
            if exp_detrend:
                other_ts_2_subtract = np.log(other_ts_2_subtract)
                
            other_ts_2_subtract = other_ts_2_subtract*np.ones(F.shape)
        
        if new_ts_name is None:
            new_ts_name = ts_key+'_z'
            
        if (other_ts_2_subtract is None) and (background_ts is None) and not exp_detrend:
            ts_z = sp.stats.zscore(F,axis=-1)
            if add_to_timeseries_dict:
                self.timeseries[new_ts_name] = ts_z
            return ts_z
        
        
        def reg_single_chan(X,y):
            # lr = LinReg(positive=True).fit(X,y)
            ynans = np.isnan(y)
            if ynans.any():
                Xtmp = X[~ynans,:]
                ytmp = y[~ynans]
                
                # print(Xtmp.shape, ytmp.shape)
                lr = LinReg(positive=False).fit(Xtmp,ytmp)
                
                
                if reg_other_channel:
                    raise NotImplementedError
                else:
                    ypred = np.zeros_like(y)
                    ypred[~ynans] = lr.predict(Xtmp)
                    ypred[ynans] = np.nan
                    # print(np.isnan(lr.predict(Xtmp)).any())
                    return ypred
            else:
                lr = LinReg(positive=False).fit(X,y)
                
                
                    
                if reg_other_channel:
                    # print(lr.coef_.shape, lr.intercept_)
                    coef = lr.coef_
                    coef[-1] = 1.*coef[-1]
                    # print(coef)
                    
                    return (X*coef[np.newaxis,:]).sum(axis=-1) + lr.intercept_
                    
                else:
                    return lr.predict(X)
        
        ts_z = np.zeros(F.shape)
        baseline = np.zeros(F.shape)
        for ch in range(F.shape[0]):
            for roi in range(F.shape[1]):
                y = F[ch, roi, :]
                if exp_detrend:
                    y = np.log(y +1E-3)
                    
                X = []
                if background_ts is not None:
                    X.append(background_ts[ch,roi,:])
                    # y -= .5*background_ts[ch,roi,:]
                
                if other_ts_2_subtract is not None:
                    X.append(other_ts_2_subtract[ch,roi,:])
                    
                if exp_detrend:
                    X.append(-1*np.array(self.metadata['frame_times']).mean(axis=-1))

                if reg_other_channel:
                    if ch<3:
                        if exp_detrend:
                            X.append(np.log(F[ch-1,roi,:]+1E-3))
                        else:
                            X.append(np.copy(F[ch-1,roi,:]))
                    
            
                X = np.column_stack(X)    
                
                
                baseline[ch,roi,:] = reg_single_chan(X,y)
                ts_z[ch,roi,:] = y
        
        
        if exp_detrend:
            ts_z = np.exp(ts_z)
            baseline = np.exp(baseline)
            
        if zscore:
            ts_z = sp.stats.zscore(ts_z-baseline,axis=-1, nan_policy='omit')
        else:
            ts_z = ts_z/baseline 
        if add_to_timeseries_dict:
            self.timeseries[new_ts_name] = ts_z
        return ts_z
        
        
    
    
class ColumnarRingImagingSession(EBImagingSession):
    
   
    def open_napari(self, check_for_existing=True, path=None):
        if check_for_existing:
            if path is None:
                return napari_tools.EB_R4d_Session().new_session(self.ref_img)
            elif pathlib.Path.exists(path):
                return napari_tools.EB_R4d_Session().open_existing_session(path)
            else:
                return napari_tools.EB_R4d_Session().new_session(self.ref_img)
        else:
            return napari_tools.EB_R4d_Session().new_session(self.ref_img)
        
    
    

class ExVivoRingImagingSession(EBImagingSession):
    
    
    def open_napari(self, check_for_existing=True):
        raise NotImplementedError
    
    
    