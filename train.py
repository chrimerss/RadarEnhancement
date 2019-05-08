from radarnet import RadarNet
from dataprep import DataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
from torchsummary import summary

use_gpu=True

def main():
	global use_gpu
	# set up some parameters
	batch_size=4
	lr= 1e-3
	logging_path= 'logging/'
	num_epoches= 500
	epoch_to_save= 10
	tsize=10

	data_path= 'dataset'
	print('loading data sets ...')
	dataset_train= DataSet(datapath= 'dataset')
	loader_train= DataLoader(dataset= dataset_train, num_workers=8, batch_size=batch_size, shuffle=True)

	print("# of training samples: %d\n" %int(len(dataset_train)))

	model= RadarNet(use_gpu=use_gpu)
	print(model)

	#criterion
	criterion= torch.nn.MSELoss()

	if use_gpu:
		model= model.cuda()
		criterion.cuda()

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
			# input size: (4,15,1,1000,1000)
			# target size: (4,15,1,1000,1000)
			model.train()
			model.zero_grad()
			optimizer.zero_grad()

			input_train, target_train= Variable(input_train), Variable(target_train)
			if use_gpu:
				input_train, target_train= input_train.cuda(), target_train.cuda()

			out_train= model(input_train)
			loss= criterion(target_train, out_train)*1000

			loss.backward()
			optimizer.step()

			# training track
			model.eval()
			out_train= model(input_train)
			output_train= torch.clamp(out_train, 0, 1)
			print("[epoch %d/%d][%d/%d]  loss: %.4f "%(epoch+1,num_epoches, i+1,len(loader_train),loss.item()))

			if step% 10 ==0:
				writer.add_scalar('loss', loss.item())

			step+=1

		#save model
		if epoch % epoch_to_save==0:
			torch.save(model.state_dict(), os.path.join(logging_path,'net_epoch%d.pth'%(epoch+1)))

	torch.save(model.state_dict(), os.path.join(logging_path,'newest.pth'))

if __name__=='__main__':
	main()
