from model import RadarNet
from utils import DataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
from torchsummary import summary
from utils import ComLoss
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print ('Available devices ', torch.cuda.device_count())

use_gpu=True
device=torch.device('cuda:0')
torch.cuda.empty_cache()

def main():
	global use_gpu,device
	# set up some parameters
	batch_size=1
	lr= 1e-3
	logging_path= 'logging/'
	num_epoches= 500
	epoch_to_save= 10
	tsize=10

	data_path= '../dataset'
	print('loading data sets ...')
	dataset_train= DataSet(datapath= data_path)
	loader_train= DataLoader(dataset= dataset_train, num_workers=8, batch_size=batch_size, shuffle=True)

	print("# of training samples: %d\n" %int(len(dataset_train)))

	model= RadarNet(use_gpu=use_gpu,device=device)
	print(model)

	criterion= ComLoss()

	# model.load_state_dict(torch.load('../logging/newest-5_8.pth'))

	if use_gpu:
		model= model.to(device)
		criterion.to(device)

	#optimizer
	optimizer= torch.optim.Adam(model.parameters(), lr=lr)
	scheduler= MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)

	#record
	writer= SummaryWriter(logging_path)

	#start training
	step= 0
	for epoch in range(num_epoches):
		scheduler.step(epoch)

		for param_group in optimizer.param_groups:
			print('learning rate %f' %param_group['lr'])

		for i, (input_train, target_train) in enumerate(loader_train, 0):
			# input size: (4,10,1,200,200)
			# target size: (4,10,1,200,200)
			model.train()
			model.zero_grad()
			optimizer.zero_grad()

			input_train, target_train= Variable(input_train), Variable(target_train)
			if use_gpu:
				input_train, target_train= input_train.to(device), target_train.to(device)

			out_train= model(input_train)
			loss= -criterion(target_train, out_train)

			loss.backward()
			optimizer.step()

			# training track
			model.eval()
			out_train= model(input_train)
			output_train= torch.clamp(out_train, 0, 1)
			print("[epoch %d/%d][%d/%d]  obj: %.4f "%(epoch+1,num_epoches, i+1,len(loader_train),-loss.item()/tsize))

			if step% 10 ==0:
				writer.add_scalar('loss', loss.item())

			step+=1
			# print('memory allocated: ', torch.cuda.memory_allocated(device=device))
			# print('max memory allocated: ',torch.cuda.max_memory_allocated(device=device))
		#save model
		# if epoch % epoch_to_save==0:
		# 	torch.save(model.state_dict(), os.path.join(logging_path,'net_epoch%d.pth'%(epoch+1)))

	torch.save(model.state_dict(), os.path.join(logging_path,'newest.pth'))


if __name__=='__main__':
	main()
