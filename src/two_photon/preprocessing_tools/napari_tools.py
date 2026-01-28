import itertools

import numpy as np
import scipy as sp
import napari
import cloudpickle

# add abstract base class for napari wrappers


class EBSession:
    
    def __init__(self) -> None:
        
        
        self.ref_img = None
        self.n_ch = None
        self.n_zplanes = None
        self.img_size = None
        
        self.viewer = None
        self.inner_ring = None
        self.outer_ring = None
        
        self._phase_mat = None
        self._com = None
        self._phase_donut = None
        self._phase_bin_edges = None
        self.phase_bin_centers = None
        self.masks = None
        
    
    
    def open_existing_session(self, filename):
        
        with open(filename, 'rb') as file:
            napari_layers = cloudpickle.load(file)    
        
        self.ref_img = napari_layers['ref_img']
        self.n_ch = self.ref_img.shape[0]
        self.n_zplanes = self.ref_img.shape[1]
        self.img_size = self.ref_img.shape[-2:]
        
        self.viewer = napari.view_image(napari_layers['ref_ch1'], name = 'ref_ch1')
        self.viewer.add_image(napari_layers['ref_ch1'].max(axis=0), name = 'ref_ch1_maxp')
        for ch in range(1,napari_layers['n_ch']):
            _ch = ch+1
            self.viewer.add_image(napari_layers[f'ref_ch{_ch}'], name = f'ref_ch{_ch}')
            self.viewer.add_image(napari_layers[f'ref_ch{_ch}'].max(axis=0), name = f'ref_ch{_ch}_maxp')
            
        
        try:
        
            self._add_rings(inner_ring_data=napari_layers['inner_ring'],
                            outer_ring_data=napari_layers['outer_ring'],
                            background=napari_layers['background'])
        except:
            self._add_rings(inner_ring_data=napari_layers['inner_ring'],
                            outer_ring_data=napari_layers['outer_ring'],
                            background=None)
        
        self.make_phase_masks()
        return self
        
        
        

    def new_session(self, ref_img):
        self.ref_img = ref_img
        self.n_ch = ref_img.shape[0]
        self.n_zplanes = ref_img.shape[1]
        self.img_size = ref_img.shape[-2:]
        
        self.viewer = napari.view_image(np.squeeze(ref_img[0,:,:,:]), name = 'ref_ch1' )
        self.viewer.add_image(ref_img[0,:,:,:].max(axis=0), name = 'ref_ch1_maxp')
        if self.n_ch>1:
            for ch in range(1,self.n_ch):
                _ch = ch+1
                self.viewer.add_image(np.squeeze(ref_img[ch,:,:,:]), name = f'ref_ch{_ch}')
                self.viewer.add_image(np.squeeze(ref_img[ch,:,:,:]).max(axis=0,keepdims=True), name = f'ref_ch{_ch}_maxp')
        self._add_rings()
        return self
        
        
    def _add_rings(self, inner_ring_data = None, outer_ring_data = None, background = None):
        
        if inner_ring_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='inner_ring')
        else:
            self.viewer.add_labels((inner_ring_data).astype(int), name='inner_ring')
            
        if outer_ring_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='outer_ring')
        else:
            self.viewer.add_labels((outer_ring_data).astype(int), name='outer_ring')
            
        if background is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='background')
        else:
            self.viewer.add_labels((background).astype(int), name='background')
            
            
        self.inner_ring = self.viewer.layers['inner_ring']
        self.outer_ring = self.viewer.layers['outer_ring']
        self.background = self.viewer.layers['background']
        
    
    def _get_inner_ring_com(self):
        self._com = np.nan*np.zeros((self.inner_ring.data.shape[0],2))
        for z in range(self.n_zplanes):
            plane = self.inner_ring.data[z,:,:]
            if plane.ravel().sum() > 0 :
                self._com[z,:] = sp.ndimage.center_of_mass(plane)
        
        
    def make_phase_masks(self, n_rois=16):
        
        self._phase_mat = np.nan*np.zeros(self.inner_ring.data.shape)
        self._get_inner_ring_com()
        # print(self._phase_mat.shape)
        
        for z in range(self.n_zplanes):
            for row, col in itertools.product(range(self.img_size[0]), range(self.img_size[1])):
                if self._com[z,0] != np.nan:
                    self._phase_mat[z,row,col] = np.arctan2(col-self._com[z,1], row - self._com[z,0]) + np.pi 
        
        self._phase_donut = np.nan*np.zeros(self._phase_mat.shape)
        inds = (self.outer_ring.data - self.inner_ring.data)>0
        
        self._phase_donut[inds] = self._phase_mat[inds]
        
        self.masks = np.zeros(self.ref_img.shape[1:])
        self._phase_bin_edges = np.linspace(-1E-3,2*np.pi+1E-3, num=n_rois+1)
        for mask_i, (ledge, redge) in enumerate(zip(self._phase_bin_edges[:-1].tolist(),
                                                    self._phase_bin_edges[1:].tolist())):
            bin_inds = (self._phase_donut>=ledge) & (self._phase_donut<redge)
            self.masks[bin_inds] = mask_i +1
    
        self.viewer.add_labels(self.masks.astype(int), name='rois')
        self.rois = self.viewer.layers['rois']
        self.phase_bin_centers = self._phase_bin_edges[:-1] + self._phase_bin_edges[1:]
        self.phase_bin_centers /= 2.
        
    def save_layers(self, filename, return_layers=False):
        napari_layers = {layer.name: layer.data for layer in self.viewer.layers}
        napari_layers['n_ch'] = self.n_ch
        napari_layers['ref_img'] = self.ref_img
        with open(filename, 'wb') as file:
            cloudpickle.dump(napari_layers,file)
            
        if return_layers:
            return napari_layers

    def topickle_noviewer(self, ref_img, filename):

        if ref_img.shape[0] == 1:
            napari_layers = {'ref_ch1': ref_img[0,:,:,:],
                            'ref_ch1_maxp': ref_img[0,:,:,:].max(axis=0),
                            'n_ch': ref_img.shape[0],
                            'ref_img': ref_img,
                            'inner_ring': 0*ref_img[0,:,:,:].astype(int),
                            'outer_ring': 0*ref_img[0,:,:,:].astype(int),
                            'background': 0*ref_img[0,:,:,:].astype(int),
                            }
        else:
            napari_layers = {'ref_ch1': ref_img[0,:,:,:],
                            'ref_ch1_maxp': ref_img[0,:,:,:].max(axis=0),
                            'ref_ch2': ref_img[1,:,:,:],
                            'ref_ch2_maxp': ref_img[1,:,:,:].max(axis=0),
                            'n_ch': ref_img.shape[0],
                            'ref_img': ref_img,
                            'inner_ring': 0*ref_img[0,:,:,:].astype(int),
                            'outer_ring': 0*ref_img[0,:,:,:].astype(int),
                            'background': 0*ref_img[0,:,:,:].astype(int),
                            }
        with open(filename, 'wb') as file:
            cloudpickle.dump(napari_layers,file)
        
    
class EB_R4d_Session:
    
    def __init__(self) -> None:
        
        
        self.ref_img = None
        self.n_ch = None
        self.n_zplanes = None
        self.img_size = None
        
        self.viewer = None
        self.inner_ring = None
        self.outer_ring_EB = None
        self.outer_ring_R4d = None
        
        self._phase_mat = None
        
        self._com_EB = None
        self._phase_donut_EB = None
        self.masks_EB = None
        
        # self._phase_mat_R4d = None
        self._com_R4d = None
        self._phase_donut_R4d = None
        self.masks_R4d = None
        
        
        self._phase_bin_edges = None
        self.phase_bin_centers = None
        
    
    
    def open_existing_session(self, filename):
        
        with open(filename, 'rb') as file:
            napari_layers = cloudpickle.load(file)    
        
        self.ref_img = napari_layers['ref_img']
        self.n_ch = self.ref_img.shape[0]
        self.n_zplanes = self.ref_img.shape[1]
        self.img_size = self.ref_img.shape[-2:]
        
        self.viewer = napari.view_image(napari_layers['ref_ch1'], name = 'ref_ch1')
        self.viewer.add_image(napari_layers['ref_ch1'].max(axis=0), name = 'ref_ch1_maxp')
        for ch in range(1,napari_layers['n_ch']):
            _ch = ch+1
            self.viewer.add_image(napari_layers[f'ref_ch{_ch}'], name = f'ref_ch{_ch}')
            self.viewer.add_image(napari_layers[f'ref_ch{_ch}'].max(axis=0), name = f'ref_ch{_ch}_maxp')
            
        
        try:
        
            self._add_rings(inner_ring_data=napari_layers['inner_ring'],
                            outer_ring_EB_data=napari_layers['outer_ring_EB'],
                            outer_ring_R4d_data=napari_layers['outer_ring_R4d'],
                            background=napari_layers['background'])
        except:
            self._add_rings(inner_ring_data=napari_layers['inner_ring'],
                            outer_ring_EB_data=napari_layers['outer_ring_EB'],
                            outer_ring_R4d_data=napari_layers['outer_ring_R4d'],
                            background=None)
        
        self.make_phase_masks()
        
        
        

    def new_session(self, ref_img):
        self.ref_img = ref_img
        self.n_ch = ref_img.shape[0]
        self.n_zplanes = ref_img.shape[1]
        self.img_size = ref_img.shape[-2:]
        
        self.viewer = napari.view_image(np.squeeze(ref_img[0,:,:,:]), name = 'ref_ch1' )
        self.viewer.add_image(ref_img[0,:,:,:].max(axis=0), name = 'ref_ch1_maxp')
        if self.n_ch>1:
            for ch in range(1,self.n_ch):
                _ch = ch+1
                self.viewer.add_image(np.squeeze(ref_img[ch,:,:,:]), name = f'ref_ch{_ch}')
                self.viewer.add_image(np.squeeze(ref_img[ch,:,:,:]).max(axis=0), name = f'ref_ch{_ch}_maxp')
        self._add_rings()
        return self
        
        
    def _add_rings(self, inner_ring_data = None, outer_ring_EB_data = None, outer_ring_R4d_data = None, background = None):
        
        if inner_ring_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='inner_ring')
        else:
            self.viewer.add_labels((inner_ring_data).astype(int), name='inner_ring')
            
        if outer_ring_EB_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='outer_ring_EB')
        else:
            self.viewer.add_labels((outer_ring_EB_data).astype(int), name='outer_ring_EB')
            
        if outer_ring_R4d_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='outer_ring_R4d')
        else:
            self.viewer.add_labels((outer_ring_R4d_data).astype(int), name='outer_ring_R4d')
            
        if background is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='background')
        else:
            self.viewer.add_labels((background).astype(int), name='background')
            
            
        self.inner_ring = self.viewer.layers['inner_ring']
        self.outer_ring_EB = self.viewer.layers['outer_ring_EB']
        self.outer_ring_R4d = self.viewer.layers['outer_ring_R4d']
        self.background = self.viewer.layers['background']
        
    
    def _get_inner_ring_com(self):
        self._com = np.nan*np.zeros((self.inner_ring.data.shape[0],2))
        for z in range(self.n_zplanes):
            plane = self.inner_ring.data[z,:,:]
            if plane.ravel().sum() > 0 :
                self._com[z,:] = sp.ndimage.center_of_mass(plane)
        
        
    def make_phase_masks(self, n_rois=16):
        
        self._phase_mat = np.nan*np.zeros(self.inner_ring.data.shape)
        self._get_inner_ring_com()
        # print(self._phase_mat.shape)
        
        for z in range(self.n_zplanes):
            for row, col in itertools.product(range(self.img_size[0]), range(self.img_size[1])):
                if self._com[z,0] != np.nan:
                    self._phase_mat[z,row,col] = np.arctan2(col-self._com[z,1], row - self._com[z,0]) + np.pi 
        
        self._phase_donut_EB = np.nan*np.zeros(self._phase_mat.shape)
        self._phase_donut_R4d = np.nan*np.zeros(self._phase_mat.shape)
        
        inds_EB = (self.outer_ring_EB.data - self.inner_ring.data)>0
        self._phase_donut_EB[inds_EB] = self._phase_mat[inds_EB]
        
        inds_R4d = (self.outer_ring_R4d.data - self.inner_ring.data)>0
        self._phase_donut_R4d[inds_R4d] = self._phase_mat[inds_R4d]
        
        self.masks_EB = np.zeros(self.ref_img.shape[1:])
        self.masks_R4d = np.zeros(self.ref_img.shape[1:])
        self._phase_bin_edges = np.linspace(-1E-3,2*np.pi+1E-3, num=n_rois+1)
        for mask_i, (ledge, redge) in enumerate(zip(self._phase_bin_edges[:-1].tolist(),
                                                    self._phase_bin_edges[1:].tolist())):
            
            bin_inds_EB = (self._phase_donut_EB>=ledge) & (self._phase_donut_EB<redge)
            self.masks_EB[bin_inds_EB] = mask_i +1
            
            bin_inds_R4d = (self._phase_donut_R4d>=ledge) & (self._phase_donut_R4d<redge)
            self.masks_R4d[bin_inds_R4d] = mask_i +1
    
        self.viewer.add_labels(self.masks_EB.astype(int), name='rois_EB')
        self.rois_EB = self.viewer.layers['rois_EB']
        
        self.viewer.add_labels(self.masks_R4d.astype(int), name='rois_R4d')
        self.rois_R4d = self.viewer.layers['rois_R4d']
        
        self.phase_bin_centers = self._phase_bin_edges[:-1] + self._phase_bin_edges[1:]
        self.phase_bin_centers /= 2.
        
    def save_layers(self, filename):
        napari_layers = {layer.name: layer.data for layer in self.viewer.layers}
        napari_layers['n_ch'] = self.n_ch
        napari_layers['ref_img'] = self.ref_img
        with open(filename, 'wb') as file:
            cloudpickle.dump(napari_layers,file)


    