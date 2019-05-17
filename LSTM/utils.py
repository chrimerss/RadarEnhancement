import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from numba import jit
from math import exp

def cal_stat(train, real):
	train[train*255.<0.5]=0
	real[real*255.<0.5]=0
	@jit(nopython=True)
	def loop(train, real):
		bsize, tsize, _, h, w= train.shape
		a=0; b=0; c=0; d=0
		for ib in range(bsize):
			for it in range(tsize):
				for ih in range(h):
					for iw in range(w):
						if train[ib,it,0,ih,iw]>0 and real[ib,it,0,ih,iw]>0: a+=1
						if train[ib,it,0,ih,iw]==0 and real[ib,it,0,ih,iw]>0: b+=1
						if train[ib,it,0,ih,iw]>0 and real[ib,it,0,ih,iw]==0: c+=1
						if train[ib,it,0,ih,iw]==0 and real[ib,it,0,ih,iw]==0: d+=1
		return a,b,c,d
	a,b,c,d= loop(train.cpu().detach().numpy(), real.cpu().detach().numpy())
	return a,b,c,d
 
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=True)

    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    # print(img1.size(),img2.size(), window.size())
    mu1 = F.conv2d(img1, window,stride=1, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window,stride=1, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class ComLoss(torch.nn.Module):
	def __init__(self, window_size = 11, size_average = True):
		super(ComLoss, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		ssim=0
		(_,tsize, channel, _, _) = img1.size() 
		for it in range(tsize):
			_img1= img1[:,it,:,:,:].detach()
			_img2= img2[:,it,:,:,:].detach()
			window= self.window.cuda()
			window = self.window.type_as(_img1)
			ssim-=_ssim(_img1, _img2, window, self.window_size, channel, self.size_average)

		return ssim

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)