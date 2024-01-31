from abc import ABC, abstractmethod
import pathlib
from glob import glob
import itertools

import dask.dataframe as dd
import h5py
import numpy as np
import napari

from .preprocessing_tools import *

class Preprocess(ABC):
    
    def __init__(self):
        
        self.bruker_folder = None
        self.session_info = {'date': None, 
                             'genotype_dirname': None, 
                             'session': None,
                             'full_genotype': None}
        
        
        self.metadata = None
        
        self.h5path_raw = None
        self.h5path_motcorr = None
        
    def extract_metadata(self,session_info, bruker_folder, overwrite=False):
        
        # make sure minimal keys are in session_info and assign them
        for k in self.session_info.keys():
            try:
                self.session_info[k] = session_info[k]
            except KeyError:
                print(f'missing key in session_info: {k}')
        
        # assign any extra keys
        for k, v in session_info.items():
            self.session_info[k]=v
                
        # extract metadata
        if (self.metadata is None) or overwrite:
            self.bruker_folder = pathlib.Path(bruker_folder)
            self.metadata = bruker_metadata.read(self.bruker_folder)
            self._metadata_extracted=True
       
    def bruker_to_h5(self, 
                     outdir=pathlib.Path('/media/mplitt/SSD_storage/2P_scratch'),
                     first_chan=1,
                     overwrite=False):
        
        
        gd = self.session_info['genotype_dir']
        d = self.session_info['date']
        f = self.session_info['fly']
        s = self.session_info['session']
        
        self.h5path_raw = pathlib.Path(outdir.joinpath(f'{gd}/{d}/{f}/{s}/data.h5'))
        
        
        if (not self.h5path_raw.is_file()) or overwrite:
            tiff_data = tiff_tools.read( None, 
                                        self.metadata['size'],
                                        self.metadata['layout'],
                                        first_chan=first_chan)
            
            tiff_tools.convert_to_hdf5(tiff_data, self.h5path_raw, overwrite=True)
        
           
    @property
    def data(self):
        # add that it looks for motion corrected h5 file
        with h5py.File(self.h5path_raw, 'r') as f:
            return f['/data'][:]
        
        
    
    def motion_correct(self, data, ref_channel=0, write_to_h5=True):
        self.ref_img = motion_correction.make_ref_img(data, ref_channel)
        data_corr, shifts, error, diffphase = motion_correction.align_data_chunk(data, self.ref_img[ref_channel,:,:,:])
        
        if write_to_h5:
            
            self.h5path_motcorr = self.h5path_raw.stem.join('_motcorr.h5')
            with h5py.File(self.h5path_motcorr, 'w') as f:
                f.create_dataset('data', data=data_corr)
                f.create_dataset('shifts', data=shifts)
                f.create_dataset('error', data=error)
                f.create_dataset('diffphase', data=diffphase)
        
        return data_corr, shifts, error, diffphase  
        
        
    @property
    def data_corr(self):
        with h5py.File(self.h5path_motcorr, 'r') as f:
            return f['/data'][:]
        
    
    def align_voltage_recording(self):
        
        csv_files = glob(f'{self.bruker_folder.as_posix}/*.csv')
        self.voltage_recording_path = pathlib.Path(csv_files[0])
        df = dd.read_csv(self.voltage_recording_path).compute()  
        
        frame_times = np.array(self.metadata['frame_times']).mean(axis=-1)*1000
        self.voltage_recording_aligned = signals.align_vr_2p(df,frame_times)
            
    
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
    
    
    def save(self):
        raise NotImplementedError
        
    @classmethod
    def from_file(cls, filename):
        raise NotImplementedError
    
    
    def open_napari(self, check_for_existing=True, path=None):
        if check_for_existing:
            if path is None:
                pass
            if pathlib.Path.exists():
                return napari_tools.EBSession().open_existing_session(path)
            else:
                return napari_tools.EBSession().new_session(self.ref_img)
        else:
            return napari_tools.EBSession().new_session(self.ref_img)
    
    def save_napari(self, nap, overwrite=True):
        nap.save_layers(self.napari_output_path)
    
    def get_labels_layers(self, nap):
        self.napari_layers = {}
        for layer in nap.viewer.layers: 
            if isinstance(layer, napari.layers.Labels):
                self.nap_layers[layer.name] = layer.data
        

    def extract_timeseries(self, data_shape):
        self.timeseries = {}
        n_ch = data_shape[0]
        n_timepoints = data_shape[1]
        
        def _extract_ts(masks):
            n_rois = int(np.amax(masks.ravel()))
            F = np.zeros((n_ch, n_rois, n_timepoints))
            for r in range(n_rois):
                mask = masks==r+1
                
                for ch, fr in itertools.product(range(n_ch), range(n_timepoints)):
                    
                    frame = data[ch, fr, :, :, :]
                    F[ch,r,fr] = frame[mask].mean()
            return F
        
        for name, data in self.napari_layers.items():
            self.timeseries[name]=_extract_ts(data)
            
            
        
        
    
    
class ColumnarRingImagingSession(EBImagingSession):
    
   
    def open_napari(self, check_for_existing=True, path=None):
        if check_for_existing:
            if path is None:
                pass
            if pathlib.Path.exists():
                return napari_tools.EB_R4d_Session().open_existing_session(path)
            else:
                return napari_tools.EB_R4d_Session().new_session(self.ref_img)
        else:
            return napari_tools.EB_R4d_Session().new_session(self.ref_img)
        
    

class ExVivoRingImagingSession(EBImagingSession):
    
    
    def open_napari(self, check_for_existing=True):
        raise NotImplementedError
    
    
    