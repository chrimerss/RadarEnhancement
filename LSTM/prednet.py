'''
Pytorch implementation of PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning
'''
import torch.nn as nn
import torch.Tensor as Tensor
from torch.autograd import Variable

class BaseNet(nn.Module):
	
	def __init__(self, input_channel, hidden_layer, padding, stride):
		super(BaseNet, self).__init__()
		self.hidden_layer= hidden_layer
		self.padding= padding
		self.stride= stride
		self.input_channel= input_channel

	def forward(self,x,h,c):
		bsize,tsize,_,h,w= x.size()
		self.Wxi= nn.conv2d(1, self.hidden_layer, self.padding, self.stride)


	def hidden_init(self):
		# initialize h, c
		return Variable(Tensor())


class PredNet(nn.Module):
	def __init__(self):
		super(PredNet, self).__init__()
