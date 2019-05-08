'''
In this folder, we build LSTM model for radar nowcast.
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseNet(nn.Module):

	def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, use_gpu=False):
		super(BaseNet,self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels
		self.kernel_size= kernel_size
		self.bias= bias
		self.use_gpu= use_gpu

		assert self.hidden_channels%2==0

		self.num_features=4
		self.padding = int((kernel_size-1)/2) #'same'

		# assign weights
		self.Wxi= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Whi= nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Wxf= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Whf= nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Wxc= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Whc= nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Wxo= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)
		self.Who= nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,1,self.padding,bias=True)

		self.Wci= None
		self.Wcf= None

	def forward(self, x, h, c):

		ci= torch.sigmoid(self.Wxi(x)+ self.Whi(h)+ c* self.Wci)
		cf= torch.sigmoid(self.Wxf(x)+ self.Whf(h)+ c* self.Wcf)
		cc= cf* c+ ci* torch.tanh(self.Wxc(x)+ self.Whc(h))
		co= torch.sigmoid(self.Wxo(x)+ self.Who(h)+ cc*self.Wco)
		ch= co* torch.tanh(cc)

		return ch, cc

	def init_hidden(self, batch_size,hidden, shape):
		if self.use_gpu:
			self.Wci= Variable(torch.zeros(1,hidden,shape[0],shape[1])).cuda()
			self.Wcf= Variable(torch.zeros(1,hidden,shape[0],shape[1])).cuda()
			self.Wco= Variable(torch.zeros(1,hidden,shape[0],shape[1])).cuda()

			return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
					Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda()))

		else: #gpu is not available
			self.Wci= Variable(torch.zeros(1,hidden,shape[0],shape[1]))
			self.Wcf= Variable(torch.zeros(1,hidden,shape[0],shape[1]))
			self.Wco= Variable(torch.zeros(1,hidden,shape[0],shape[1]))

			return (Variable(torch.zeros(batch_size,hidden, shape[0], shape[1])),
					Variable(torch.zeros(batch_size,hidden, shape[0], shape[1])))


class RadarNet(nn.Module):
	# Encoder-Predictor with BaseNet
	def __init__(self, input_channels=1, hidden_channels=16, kernel_size=3,
				bias=True, use_gpu=False):

		super(RadarNet, self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels
		self.kernel_size= kernel_size
		self.bias= bias
		self.use_gpu= use_gpu

		self.encoder= BaseNet(self.input_channels, self.hidden_channels, self.kernel_size, self.bias, self.use_gpu)
		
		self.predictor= BaseNet(self.input_channels, self.hidden_channels, self.kernel_size, self.bias, self.use_gpu)

		self.padding= int((self.kernel_size-1)/2)
		self.lastconv= nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=True)

	def forward(self, input):
		x= input

		bsize, tsize, channels, height, width= x.size()

		(he, ce)= self.encoder.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
		(hp, cp)= self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
		#encoding
		for it in range(tsize):
			(he, ce)= self.encoder(x[:,it,:,:,:], he, ce)

		hp= he
		cp= ce

		#predictor
		xzero= Variable(torch.zeros(bsize,channels, height, width)).cuda() if self.use_gpu else Variable(torch.zeros(bsize, channels,height, width))
		xout= Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda() if self.use_gpu else Variable(torch.zeros(bsize, tsize, channels,height, width))

		for it in range(tsize):
			(hp, cp)= self.predictor(xzero, hp, cp)
			xout[:,it,:,:,:]= self.lastconv(hp)

		return xout