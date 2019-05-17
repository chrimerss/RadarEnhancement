'''
In this test, we use two sucessive radar images to predict the onset images,
and compare the POD/FAR with original one.
we select the two events from last year's radar sequences.
'''

import cv2
import numpy as np
from PIL import Image
import os

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

def prep_radar_data(path):
	global IMAGE_PATH
	tiffs= os.listdir(IMAGE_PATH)
	ind=1
	while True:
		prev= tiff[ind-1]
		now= tiff[ind]
		if ind>len(tiffs):
			break
		yield np.array(Image.open(prev)), np.array(Image.open(now))

def optical_flow(path):
	for old, new in prep_radar_data(path):
		if old.shape!=(1000,1000):
			assert ValueError('Input data is invalid, expected (1000,1000) but %s received.'%(str(old.shape)))
		#coner feature track
		p0 = cv2.goodFeaturesToTrack(old, mask = None, **FEATURE_PARAMS)
		#LK algorithm
		p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **LK_PARAMS)
		



