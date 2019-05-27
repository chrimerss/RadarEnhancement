import torch.nn as nn
import torch


class Generator(nn.Module):
	def __init__(self,use_gpu=True):
		pass

	def forward(self, x):
		#input data shape (bsize, tsize, channel, m, n)
		pass


class Descrimitor(nn.Module):
	def __init__(self, use_gpu=True):
		pass

	def forward(self,x):
		#input data shape (bsize, tsize, channel, m, n)