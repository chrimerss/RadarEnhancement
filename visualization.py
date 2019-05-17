import matplotlib.animation as animation
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import dateutil

IMAGE_PATH= './radarimages'
SEQUENCES= os.listdir(IMAGE_PATH)
print(SEQUENCES)

def frame():
	global SEQUENCES, IMAGE_PATH
	ind=-1
	while True:
		ind+=1
		if ind>=len(SEQUENCES):
			break
		name= SEQUENCES[ind]
		tif= np.array(Image.open(os.path.join(IMAGE_PATH,name)))

		tif[tif<5]=0
		time= dateutil.parser.parse(name.split('.')[0].split('_')[-1])
		yield tif, time

def update(args):
	img= args[0]
	time= args[1]
	ax.imshow(img)
	ax.set_title(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':
	fig, ax= plt.subplots(1,1,figsize=(10,8))
	ani = animation.FuncAnimation(fig,update,frame, interval=0)
	ani.save('demo.gif', writer='imagemagick', fps=1,bitrate=50)

