from abc import ABC, abstractmethod
import pathlib
import h5py

from .preprocessing_tools import *

class Preprocess(ABC):
    
    def __init__(self):
        
        self.bruker_folder = None
        self.session_info = {'date': None, 
                             'genotype_dirname': None, 
                             'session': None,
                             'full_genotype': None}
        
        
        self.metadata = None
        
        self.h5path = None
        
        self._motion_corrected = False
        self._rois_created = False
        self._rois_extracted = False
        self._voltage_recording_aligned = False
        self._dff_calculated = False 
        
        
        
    
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
        
        self.h5path = pathlib.Path(outdir.joinpath(f'{gd}/{d}/{f}/{s}'))
        
        
        if (not self.h5path.is_file()) or overwrite:
            tiff_data = tiff_tools.read( None, 
                                        self.metadata['size'],
                                        self.metadata['layout'],
                                        first_chan=first_chan)
            
            tiff_tools.convert_to_hdf5(tiff_data, self.h5path, overwrite=True)
            
    @property
    def data(self):
        f = h5py.File(self.h5path)
        return f['/data'][:]
    
    def motion_correct(self, data, ref_channel=0):
        self.ref_img = motion_correction.make_ref_img(data, ref_channel)
        
        
        pass
    
    
    def align_voltage_recording(self):    
        pass
    
    def save(self):
        pass
    
    @classmethod
    def from_file(filename):
        pass
    
    
    @abstractmethod
    def run_napari(self):
        pass
    
    @abstractmethod
    def extract_timeseries(self):
        pass
    
    @abstractmethod
    def calculate_dff(self):
        pass
    

