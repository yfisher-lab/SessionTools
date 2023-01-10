from .preprocessing_tools import *

class Preprocess:
    
    def __init__(self):
        
        self._metadata_extracted = False
        self._motion_corrected = False
        self._rois_created = False
        self._rois_extracted = False
        self._voltage_recording_aligned = False
        self._dff_calculated = False 
        
        pass
    
    def extract_metadata(self,session_info, bruker_folder):
        pass 
    
    def motion_correct(self):
        pass
    
    def run_napari(self):
        pass
    
    def extract_timeseries(self):
        pass
    
    def calculate_dff(self):
        pass
    
    def align_voltage_recording(self):    
        pass
    
    def save(self):
        pass
    
    @classmethod
    def from_file(filename):
        pass
    
    

