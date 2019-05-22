'''
In this test, we use two sucessive radar images to predict the onset images,
and compare the POD/FAR with original one.
we select the two events from last year's radar sequences.
'''

import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import transform as tf
import dateutil
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.ndimage import map_coordinates
import skimage.transform as sktf

# Define some hyperparameters for optical flow
# Shi-Tomasi coner detection
FEATURE_PARAMS = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

LK_PARAMS= dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                   10, 0.03))
# ====== model configurations ========
IMAGE_PATH= '../radarimages'

MODEL= LinearRegression()

TRANSFORMATION= sktf.AffineTransform()

def prep_radar_data(path=IMAGE_PATH):
    global IMAGE_PATH
    tiffs= sorted(os.listdir(IMAGE_PATH))[:10]
    ind=1
    while True:
        if ind>=len(tiffs):
            break
        prev= os.path.join(IMAGE_PATH,tiffs[ind-1])
        now_img= os.path.join(IMAGE_PATH,tiffs[ind])
        prev= np.array(Image.open(prev))
        prev= (prev/prev.max()*255.).astype(np.uint8)
        now_img= np.array(Image.open(now_img))
        now_img= (now_img/now_img.max()*255).astype(np.uint8)
        curr_time= tiffs[ind].split('.')[0].split('_')[-1]
        ind+=1
        print('processing: ', curr_time)
        yield prev, now_img, dateutil.parser.parse(curr_time)

def optical_flow(src, dst, frame,time, lead_steps=15, show_points=False):
    global TRANSFORMATION
    last_frame= frame.copy()
    for it in range(lead_steps+1):
        if it==0:
            yield frame,time
        else:
            time+= datetime.timedelta(minutes=2)
            b= TRANSFORMATION.estimate(src, dst[it-1])
            if not b:
                raise ValueError('transformation failed...')

            #nowcast
            nowcast_frame = sktf.warp(last_frame/255., TRANSFORMATION.inverse)

            nowcast_frame = (nowcast_frame*255.).astype('uint8')

            print('predict: ', time)
            yield nowcast_frame, time

def _trf(path=IMAGE_PATH, lead_steps=15):
    global IMAGE_PATH, MODEL
    first= True
    x= []
    y= []
    for i, (old, new, time) in enumerate(prep_radar_data(path)):
        if old.shape!=(1000,1000):
            assert ValueError('Input data is invalid, expected (1000,1000) but %s received.'%(str(old.shape)))
        #coner feature track applicable only for the first
        if first:
            p0 = cv2.goodFeaturesToTrack(old, mask = None, **FEATURE_PARAMS)
            first=False
        else:
            p0= p1.copy()
        #LK algorithm
        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **LK_PARAMS)
        #select some good points
        good_old= p0[st==1]
        good_new= p1[st==1]

        # mark those good points
        for ii, (old_pt, good_pt) in enumerate(zip(good_old, good_new)):
            a,b= old_pt.ravel()
            c,d= good_pt.ravel()
            x.append(c) ### edit here
            y.append(d)
            frame= cv2.line(new, (a,b),(c,d),(255,0,0),2)
            frame= cv2.circle(new, (c,d),5, (0,0,255),-1)

    # filter out those paths that are not consistent
    x= np.array(x).reshape(i+1,ii+1); y= np.array(y).reshape(i+1,ii+1);
    # full_path= [np.sum(np.isnan(x[:,i])) for i in range(x.shape[1])]
    # print('find full path: ', full_path)
    # x= x[:,full_path].copy()
    # print(x)
    # y= y[:,full_path].copy()
    x_new= np.full((lead_steps,x.shape[1]), np.nan)
    y_new= np.full((lead_steps,y.shape[1]), np.nan)

    for p in range(x.shape[1]):

        x_train= x[:,p]
        y_train= y[:,p]

        X= np.arange(x.shape[0]+lead_steps)
        # we use polynomial extrapolation
        polyfeatures= PolynomialFeatures(degree=2)
        X= polyfeatures.fit_transform(X.reshape(-1,1))
        X_train= X[:x.shape[0],:]
        X_pred= X[x.shape[0]:,:]

        x_pred= MODEL.fit(X_train, x_train).predict(X_pred)
        y_pred= MODEL.fit(X_train, y_train).predict(X_pred)

        x_new[:,p]= x_pred
        y_new[:,p]= y_pred

    # stack x and y
    # print(x[-1,:].reshape(-1,1))
    src= np.hstack([x[-1,:].reshape(-1,1), y[-1,:].reshape(-1,1)])
    # stack prediction
    dst= np.array([np.hstack([x_new[i,:].reshape(-1,1), y_new[i,:].reshape(-1,1)]) for i in range(x_new.shape[0])])
    # print(src, dst)
    return src, dst, new, time


def pnt_warp(src, dst,method='affine'):

    return tf.estimate_transform(method, src, dst)


def update(args):
    frame= args[0]
    title= args[1]
    name= title.strftime('0000_%Y%m%d%H%M%S.tif')
    gt= np.array(Image.open('../radarimages/%s'%name))
    ax[0].imshow(((gt/gt.max())*255.).astype(np.uint8))
    ax[0].set_title(str(title)+'_ground truth')
    ax[1].imshow(frame)
    ax[1].set_title(str(title)+'_prediction')


if __name__=='__main__':
    src, dst, frame, time= _trf()
    fig, ax= plt.subplots(1,2, figsize=(10,8))
    ani = animation.FuncAnimation(fig,update,optical_flow(src,dst,frame,time), interval=0)
    ani.save('optical_flow.gif', writer='imagemagick', fps=1,bitrate=50)
    # print(next(optical_flow()))


