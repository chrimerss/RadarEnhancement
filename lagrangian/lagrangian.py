#!/usr/bin/python
'''
Semi-lagrangian algorithm
'''
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage
from glob import glob
import time
import scipy.ndimage.interpolation as sni
import os
import datetime

class SemiLagrangian(object):
    def __init__(self):
        '''define image path'''
        self.radar_path= '../radarimages'
        self.tif_file= '0000_20181212150300.tif'
        self.bench_mark= np.array(Image.open(os.path.join(self.radar_path,self.tif_file)))
        self.time= datetime.datetime.strptime(self.tif_file.split('.')[0].split('_')[-1],'%Y%m%d%H%M%S')

    def wind_field_uniform(self):
        '''simple uniform wind vector'''

        v= np.array([i for i in np.linspace(0,20,1000)]*1000).reshape(1000,1000)
        u= v.copy()

        return u,v

    def wind_field_circular(self):
        '''circular wind field'''

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
            dis_coords= [dis_coords[1,:,:], dis_coords[0,:,:]]

            precip_map= sni.map_coordinates(frame, dis_coords, mode="constant", cval=np.nanmin(frame), order=0,
                                prefilter=False)

            out.append(precip_map.reshape(frame.shape))

        pred_frames= np.stack(out)
        
        if verbose:
            toc= time.time()
            print('------------finished with %.2f seconds! ------------'%(toc-tic))

        return pred_frames

    def process(self, wind=None):
        if wind is None:
            wind= np.stack(self.wind_field_uniform())

        pred_frames=  self.forecast(self.bench_mark, wind, lead_steps=10, verbose=True)

        return pred_frames

    def visualize(self, frames, wind_field, save_path, **kwargs):
        '''
        matplotlib back-end plot to produce gif file

        Args:
        -----------------------
        frames: numpy.nadarray; forecasted array (tsize, m, n)
        wind_field: numpy.ndarray; (2,m,n)
        save_path: str; specify where to save the file
        gif_writer: str; default 'imagemagick'
        fps: int; frame per second; default 1
        bitrate: int; int per frame rate (the quality of gif); default 50
        interval: int; interval between frames; default 0
        fig_size: tuple; determine the figure size; default (8,10)

        Return:
        -----------------------
        None

        '''
        gif_writer= kwargs.get('gif_writer', 'imagemagick')
        fps= kwargs.get('fps', 1)
        bitrate= kwargs.get('bitrate',50)
        interval= kwargs.get('interval',0)
        fig_size= kwargs.get('fig_size',(8,10))
        fig, ax= plt.subplots(1,1, figsize=fig_size)

        def _frame(frames, wind_field):
            for i, frame in enumerate(frames):
                predtime= self.time+ datetime.timedelta(minutes=(i+1)*2)
                yield frame, wind_field, predtime

        def _update(args):
            frame= args[0]
            wind= args[1]
            title= args[2]
            ax.imshow(frame)
            ax.quiver(np.arange(0, wind.shape[1],100), np.arange(0, wind.shape[2],100),
                     wind[0,0:1000:100,0:1000:100], -wind[1,0:1000:100,0:1000:100],color='white')
            ax.set_title(title)


        ani= animation.FuncAnimation(fig, _update, _frame(frames, wind_field), interval=0)
        ani.save(save_path, writer= gif_writer, fps=fps, bitrate= bitrate)

if __name__=='__main__':
    semi_model= SemiLagrangian()
    wind= np.stack(semi_model.wind_field_uniform())
    pred_frames= semi_model.process()
    semi_model.visualize(pred_frames, wind,'demo_uniform.gif')