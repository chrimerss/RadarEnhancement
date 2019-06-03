import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self,input_channels, hidden_channels, kernel_size, bias=True, use_gpu=True,device=None):
        super(GRUCell,self).__init__()
        self.use_gpu= use_gpu
        self.input_channels= input_channels
        self.hidden_channels= hidden_channels
        self.kernel_size= kernel_size
        self.bias= bias
        self.use_gpu= use_gpu
        self.device= device
        self.padding= int((self.kernel_size-1)/2) #'same'

        self.W_z= nn.Conv2d(self.input_channels,self.hidden_channels,self.kernel_size,1,self.padding, bias=self.bias) #update gate
        self.U_z= nn.Conv2d(self.hidden_channels,self.hidden_channels,self.kernel_size,1,self.padding, bias=self.bias)

        self.W_r= nn.Conv2d(self.input_channels,self.hidden_channels,self.kernel_size,1,self.padding, bias=self.bias)
        self.U_r= nn.Conv2d(self.hidden_channels,self.hidden_channels,self.kernel_size,1,self.padding, bias=self.bias)

        self.W_c= nn.Conv2d(self.input_channels,self.hidden_channels,self.kernel_size,1,self.padding, bias=self.bias)
        self.U_c= nn.Conv2d(self.hidden_channels,self.hidden_channels,self.kernel_size,1,self.padding, bias=self.bias)


    def forward(self,x,h):
        #x size (bsize, tsize, channels, height, width)

        # update gate
        z_t= F.sigmoid(self.W_z(x)+self.U_z(h))
        #reset gate
        r_t= F.sigmoid(self.W_r(x)+self.U_r(h))
        #current memory content
        c_t= F.tanh(self.W_c(x)+ r_t*self.U_c(h))
        #final memory
        h_t= z_t*h +(1-z_t)*c_t

        return h_t

    def init_hidden(self, batch_size, hidden_layers,shape):

        if self.use_gpu:
            return Variable(torch.zeros(batch_size,hidden_layers,shape[0],shape[1])).to(device=self.device)    #initialize variable h_0
        else:
            return Variable(torch.zeros(batch_size,hidden_layers,shape[0],shape[1]))


class RadarNet(nn.Module):
    def __init__(self, input_channels=1, hidden_layers=32, kernel_size=3, bias=True,
                    use_gpu=True, device=None):
        super(RadarNet, self).__init__()
        self.input_channels=input_channels
        self.hidden_layers= hidden_layers
        self.kernel_size= kernel_size
        self.bias= bias
        self.use_gpu= use_gpu
        self.device= device
        self.padding= int((self.kernel_size-1)/2)

        self.encoder= GRUCell(self.input_channels, self.hidden_layers, self.kernel_size, self.bias, self.use_gpu, self.device)
        self.predictor= GRUCell(self.input_channels,self.hidden_layers, self.kernel_size, self.bias, self.use_gpu, self.device)

        self.lastlayer= nn.Conv2d(self.hidden_layers, 1, self.kernel_size,1,self.padding,bias=self.bias)

    def forward(self, x):
        bsize, tsize, channel, height, width= x.size()
        # print(x.size())

        h_e= self.encoder.init_hidden(bsize, self.hidden_layers, (height, width))
        h_p= self.predictor.init_hidden(bsize, self.hidden_layers, (height, width))

        for it in range(tsize):
            h_e= self.encoder(x[:,it,:,:,:], h_e)

        h_p= h_e
        # print('hp', h_p.size())
        x_in= Variable(torch.zeros(bsize,channel, height, width)).to(device=self.device) if self.use_gpu else Variable(torch.zeros(bsize, channel,height, width))
        x_out= Variable(torch.zeros(bsize,tsize,channel, height, width)).to(device=self.device) if self.use_gpu else Variable(torch.zeros(bsize, tsize, channel,height, width))

        for it in range(tsize):
            h_p= self.predictor(x_in, h_p)
            x_out[:,it,:,:,:]= self.lastlayer(h_p)

        return x_out

