import os
import h5py
from PIL import Image
import numpy as np
import torch.utils.data as udata
import random
import torch

target_path= 'D:\\Radar Projects\\lizhi\\for LiZhi\\ModelForecast\\RadarRaw'
num_tifs=len(os.listdir(target_path))

def prepare_data(range_to_process):
	global target_path
	files= os.listdir(target_path)
	tsize= 10
	start= range_to_process[0]
	end= range_to_process[1]

	target_h5f = h5py.File('dataset/target_1', 'w')
	input_h5f = h5py.File('dataset/input_1', 'w')

	for x in range(start, end-tsize-1):
		
		base_imgs= []
		pred_imgs= []
		first=True

		for it in range(tsize):

			base_img= np.array(Image.open(os.path.join(target_path,files[x+it])))
			base_img= base_img/base_img.max().astype(np.float16)
			pred_img= np.array(Image.open(os.path.join(target_path,files[x+it+1])))
			pred_img= pred_img/pred_img.max().astype(np.float16)

			if first:
				base_img_file_name= files[x+it].split(os.sep)[-1].split('.')[0]
				pred_img_file_name= files[x+it+1].split(os.sep)[-1].split('.')[0]
				first=False

			base_imgs.append(base_img[400:600,400:600])
			pred_imgs.append(pred_img[400:600,400:600])

		input_h5f.create_dataset(base_img_file_name, data=np.array(base_imgs))
		target_h5f.create_dataset(pred_img_file_name, data=np.array(pred_imgs))
		print(x, 'processing ...')

	input_h5f.close()
	target_h5f.close()

class DataSet(udata.Dataset):
	def __init__(self, datapath=None):
		super(DataSet, self).__init__()

		self.datapath= datapath

		input_path= os.path.join(self.datapath, 'input_1')
		target_path= os.path.join(self.datapath, 'target_1')

		target_h5f= h5py.File(target_path,'r')
		input_h5f= h5py.File(input_path, 'r')

		self.target_keys= list(target_h5f.keys())
		self.input_keys= list(input_h5f.keys())

		target_h5f.close()
		input_h5f.close()

	def __len__(self):
		return len(self.input_keys)

	def __getitem__(self, index):
		target_path= os.path.join(self.datapath, 'target_1')
		input_path= os.path.join(self.datapath, 'input_1')
		target_h5f= h5py.File(target_path,'r')       #(15,1000,1000)
		input_h5f= h5py.File(input_path, 'r')

		target_key= self.target_keys[index]
		input_key= self.input_keys[index]
		target= np.array(target_h5f[target_key])[:,np.newaxis,:,:] #(15,1,1000,1000)
		input= np.array(input_h5f[input_key])[:,np.newaxis,:,:]
		target_h5f.close()
		input_h5f.close()

		return torch.Tensor(input), torch.Tensor(target)

if __name__=='__main__':
	prepare_data([100,300])



