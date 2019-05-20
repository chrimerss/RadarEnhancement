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

IMAGE_PATH= '../radarimages'

def prep_radar_data(path=IMAGE_PATH):
    global IMAGE_PATH
    tiffs= sorted(os.listdir(IMAGE_PATH))
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
        yield prev, now_img, curr_time

def optical_flow(path=IMAGE_PATH, show_points=False):
    first= True
    for old, new, time in prep_radar_data(path):
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
        for i, (old_pt, good_pt) in enumerate(zip(good_old, good_new)):
            a,b= old_pt.ravel()
            c,d= good_pt.ravel()
            frame= cv2.line(new, (a,b),(c,d),(255,0,0),2)
            frame= cv2.circle(new, (c,d),5, (0,0,255),-1)

        yield frame, time

def frame_transform(path= IMAGE_PATH, pred_steps=15):
    # load first and second image
    tiffs= os.listdir(IMAGE_PATH)
    old_frm= Image.open(os.path.join(path,tiffs[0]))
    new_frm= Image.open(os.path.join(path,tiffs[1]))
    curr_time= tiffs[1].split('.')[0].split('_')[-1]
    curr_time= dateutil.parser.parse(curr_time)
    # find corresponding points
    p0 = cv2.goodFeaturesToTrack(old, mask = None, **FEATURE_PARAMS)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **LK_PARAMS)
    # store only good points
    old_points= p0[st==1]
    new_points= p1[st==1]
    for i in range(pred_steps):
        t_delta= 2 #minutes
        curr_time+= datetime.time_delta(minutes=t_delta)
        pass


def pnt_warp(src, dst,method='affine'):

    return tf.estimate_transform(method, src, dst)



def update(args):
    frame= args[0]
    title= args[1]
    ax.imshow(frame)
    ax.set_title(title)





if __name__=='__main__':
    fig, ax= plt.subplots(1,1, figsize=(10,8))
    ani = animation.FuncAnimation(fig,update,optical_flow, interval=0)
    ani.save('optical_flow.gif', writer='imagemagick', fps=1,bitrate=50)



