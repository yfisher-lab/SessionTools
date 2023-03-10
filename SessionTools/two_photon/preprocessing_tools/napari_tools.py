import itertools

import numpy as np
import scipy as sp
import napari
import cloudpickle


class EllipsoidBodySession:
    
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
            
        
        
        self._add_rings(inner_ring_data=napari_layers['inner_ring'],
                        outer_ring_data=napari_layers['outer_ring'])
        
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
        
        
    def _add_rings(self, inner_ring_data = None, outer_ring_data = None):
        
        if inner_ring_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='inner_ring')
        else:
            self.viewer.add_labels((inner_ring_data).astype(int), name='inner_ring')
            
        if outer_ring_data is None:
            self.viewer.add_labels((0*self.ref_img[0,:,:,:]).astype(int), name='outer_ring')
        else:
            self.viewer.add_labels((outer_ring_data).astype(int), name='outer_ring')
            
        self.inner_ring = self.viewer.layers['inner_ring']
        self.outer_ring = self.viewer.layers['outer_ring']
        
    
    def _get_inner_ring_com(self):
        self._com = np.nan*np.zeros((self.inner_ring.data.shape[0],2))
        for z in range(self.n_zplanes):
            plane = self.inner_ring.data[z,:,:]
            if plane.ravel().sum() > 0 :
                self._com[z,:] = sp.ndimage.center_of_mass(plane)
        
        
    def make_phase_masks(self):
        
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
        self._phase_bin_edges = np.linspace(-1E-3,2*np.pi+1E-3, num=17)
        for mask_i, (ledge, redge) in enumerate(zip(self._phase_bin_edges[:-1].tolist(),
                                                    self._phase_bin_edges[1:].tolist())):
            bin_inds = (self._phase_donut>=ledge) & (self._phase_donut<redge)
            self.masks[bin_inds] = mask_i +1
    
        self.viewer.add_labels(self.masks.astype(int), name='rois')
        self.rois = self.viewer.layers['rois']
        self.phase_bin_centers = self._phase_bin_edges[:-1] + self._phase_bin_edges[1:]
        self.phase_bin_centers /= 2.
        
    def save_layers(self, filename):
        napari_layers = {layer.name: layer.data for layer in self.viewer.layers}
        napari_layers['n_ch'] = self.n_ch
        napari_layers['ref_img'] = self.ref_img
        with open(filename, 'wb') as file:
            cloudpickle.dump(napari_layers,file)
        
        
        
    
    
    