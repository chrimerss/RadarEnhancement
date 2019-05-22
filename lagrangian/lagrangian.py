#!/usr/bin/python
'''
Semi-lagrangian algorithm
'''
import numpy as np
from Image import PIL
import skimage
from glob import glob
import time
import scipy.ndimage.interpolation as sni

class SemiLagrangian(object):
    def __init__(self):
        '''define image path'''
        self.radar_path= '../radarimages'
        self.bench_mark= '0000_20181102071000.tif'

    def wind_field_uniform(self):
        v= np.array([i for i in np.arange(0,20,1000)]*1000).reshape(1000,1000)
        u= v.copy()

        return u,v

    def forecast(self, frame, wind_field, lead_steps, **kwargs):
        '''
        Args:
        -------------
        frame: numpy ndarray; single radar frame
        wind_field: wind velocity vector (2,m,n)
        lead_steps: int, specify the total lead time
        method: str; now only 'semi-lagrangian' is allowable.
        n_iter: int, number of iteration needed for semi-lagrangian, default 3
        inverse: bool, if True, trajectory is computed backwards, default True
        verbose: bool, enable verboseness

        Returns:
        -------------
        pred_frames: ndarray; (lead_steps, m, n)

        '''
        n_iter = kwargs.get("n_iter", 3)
        inverse= kwargs.get("inverse", True)
        verbose= kwargs.get("verbose", False)

        m,n= frame.shape

        if verbose:
            print('---------computing with Semi-Lagrangian scheme -------')
            tic= time.time()
        coef= 1.0 if not verbose else -1.0
        x_grid, y_grid= np.meshgrid(np.arange(frame.shape[0]), np.arange(frame.shape[1]))
        coords= np.stack([x_grid, y_grid])

        #initialize D displacement vector
        D= np.zeros((2,m,n))

        out= []

        for it in range(lead_steps):
            V_inc= np.zeros(D.shape)

            for k in range(n_iter):
                dis_coords= coords+D-V_inc/2.
                dis_coords= [dis_coords[1,:,:], dis_coords[0,:,:]]

                u_field= sni.map_coordinates(wind_field[0,:,:], dis_coords, mode='nearest',
                                            order=0, prefilter=False)
                v_field= sni.map_coordinates(wind_field[1,:,:], dis_coords, mode='nearest',
                                            order=0, prefilter=False)

                V_inc[0,:,:]= u_field/n_iter
                V_inc[1,:,:]= v_field/n_iter

                D+= coef*V_inc

            dis_coords= coords+ D
            dis_coords= dis_coords[1,:,:], dis_coords[0,:,:]]

            precip_map= sni.map_coordinates(frame, XYW, mode="constant", cval=np.nanmin(frame), order=0,
                                prefilter=False)

            out.append(precip_map.reshape(frame.shape))

        pred_frames= np.stack(out)

        return pred_frames

    def visualize(self):
        pass
