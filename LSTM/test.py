import h5py
from PIL import Image
import numpy as np
import unittest
import os


class TestClass(unittest.TestCase):

	def test_dataset(self):
		# randomly test each data
		datapath= 'dataset'
		input_file= os.path.join(datapath, 'input_1')
		target_file= os.path.join(datapath, 'target_1')
		base_imgs= h5py.File(input_file, 'r')
		pred_imgs= h5py.File(target_file, 'r')
		n= np.random.randint(1,100)
		base_keys= base_imgs.keys()
		pred_keys= pred_imgs.keys()
		base_key= list(base_keys)[n+1]
		pred_key= list(pred_keys)[n]
		print(base_key, pred_key)
		base= np.array(base_imgs.get(base_key))
		target= np.array(pred_imgs.get(pred_key))
		np.testing.assert_array_equal(base, target)




if __name__=='__main__':
	unittest.main()

