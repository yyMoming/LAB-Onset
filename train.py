# -*- coding: utf-8 -*-
# @Time    : 2019/11/1 11:27
# @Author  : moMing.yang
# @File    : train.py
# @Software: PyCharm
import os
import time
import torch
import argparse
import numpy as np
import config as cfg
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataset import onset_dataset
from model import factory_net
from losses import focalloss
from evaluate import evaluate
import multiprocessing
parser = argparse.ArgumentParser()
'''
	some constants
'''
pad_length = 4
spec_style = 'cqt'
net_style = 'onset'
dual = False
batch_size = 256
num_worker = 12
alpha = 5.0
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
lr = 0.001#学习速率
momentum = 0.9
weight_decay = 0.0005
display = 100
parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', help="if begin to train", action="store_true")
parser.add_argument('--epoches', type=int, default=40, help="decide the epoches for the train")
args = parser.parse_args()
'''
	model save path
'''
model_path = os.path.join('./model','{}_{}_{}_ywmtest_mono'.format(
        spec_style, net_style, pad_length))
if not os.path.exists(model_path):
    os.makedirs(model_path)
'''
	get train set and test set
'''
def get_solf_wav_anno_files(root_path):
	train_root = os.path.join(root_path,'train')
	test_root = os.path.join(root_path,'test')

	train_wav_list = [os.path.join(train_root,file) for file in os.listdir(train_root)\
	                  if file.endswith('.wav') or file.endswith('.mp3')]
	test_wav_list = [os.path.join(test_root,file) for file in os.listdir(test_root)\
	                  if file.endswith('.wav') or file.endswith('.mp3')]
	train_anno_list = [(wav_file[:-4]+'.txt') for wav_file in train_wav_list]
	test_anno_list = [(wav_file[:-4]+'.txt') for wav_file in test_wav_list]
	train_wav_list = np.array(train_wav_list)
	train_anno_list = np.array(train_anno_list)
	test_wav_list = np.array(test_wav_list)
	test_anno_list = np.array(test_anno_list)

	return train_wav_list, train_anno_list, test_wav_list, test_anno_list
'''
	files path list
'''
path = './shuffle_data'
train_wav_list, train_anno_list, \
test_wav_list, test_anno_list = get_solf_wav_anno_files(path)
# test_dataset = onset_dataset(test_wav_list,
# 			                 test_anno_list,
# 			                 pad_length=pad_length,
# 			                 spec_style="cqt",
# 			                 dual_channel=False,
# 			                 is_filter=False,
# 			                 is_training=False)

'''
	get test DataLoader, net, model save_path.... 
	linux num_workers can be used normally
'''

# test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,
#                              num_workers=num_worker)
net = factory_net(net_style=net_style,pad_length=pad_length,spec_style=spec_style,
                  dual_channel=dual)

'''
	use_cuda, choose criterion and optimizer
'''
if use_cuda:
    net = net.cuda()#将所有的模型参数和缓存赋值GPU
    torch.cuda.manual_seed(1)
criterion = focalloss(alpha)#修正LOSS函数
# criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(),#构造网络优化器
                            lr=lr,#学习速率
                            momentum=momentum,
                            weight_decay=weight_decay)#default=0.0005
torch.backends.cudnn.benchmark = True
'''
	train function
'''
split_num = 30
def train(epoch):
	net.train()
	total_loss = 0.0
	total_precise = 0.0
	total_recall= 0.0
	total_Fscore = 0.0
	steps = 0
	evaluator = evaluate()
	last_display_time = time.time()
	f_record = open('./log/record.txt','w')
	precise = 0.0
	recall = 0.0
	Fscore = 0.0
	for index in range(split_num):
		wav_files, anno_files = get_file(index)

		train_dataset = onset_dataset(wav_files,
		                             anno_files,
		                             pad_length=pad_length,
		                             spec_style="cqt",
		                             dual_channel=False,
		                             is_filter=False,
		                             is_training=False)
		train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
		                              num_workers=num_worker)
		for idx,(inputs,labels) in enumerate(train_dataloader):
			# inputs = Variable(inputs)
			# labels = Variable(labels)
			if use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			optimizer.zero_grad()
			output = net(inputs)
			loss = criterion(output,labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.data
			eval = evaluator(output.data,labels.data)
			steps += 1
			precise += eval['P']
			recall += eval['R']
			Fscore += eval['F']
			total_times = time.time() - last_display_time
			if steps % display == 0:  # display=100
				P = precise / steps
				R = recall / steps
				F = Fscore / steps
				t = total_times / steps
				ls = total_loss / steps
				per_display_t = time.time() - last_display_time
				last_display_time = time.time()
				print('epoch:%d || losses:%.6f || precise:%.6f || recall:%.6f|| F_score:%.6f || batch time:%.6f'
				      % (epoch, ls, P, R, F, per_display_t))
			f_record.write(str(loss.data)+'\n')
			f_record.flush()
	f_record.close()

'''
	validation function
'''
def val(epoch):
	best_Fscore = 0.0
	net.eval()
	precise, recall, Fscore = 0.0, 0.0, 0.0
	total_losses = 0.0
	steps = 0
	evaluator = evaluate()
	for batch_idx,(inputs,labels) in enumerate(test_dataloader):
		output = net(inputs)
		optimizer.zero_grad()
		loss = criterion(inputs,labels)
		loss.backward()
		optimizer.step()
		steps += 1
		total_losses += loss.data
		eval = evaluator(output.data, labels.data)
		precise += eval['P']
		recall += eval['R']
		Fscore += eval['F']

	loss_avg = total_losses / steps
	precise = precise / steps
	recall = recall / steps
	Fscore = Fscore / steps
	print('test log:epoch%d || losses:%.6f precise:%.6f || recall:%.6f || F_score:%.6f'
	      % (epoch, loss_avg, precise, recall, Fscore))
	model_name = 'epoch_{}.pth'.format(epoch + 1)
	model_name = os.path.join(model_path, model_name)
	torch.save(net.state_dict(), model_name)
	if Fscore > best_Fscore:
		model_name = 'best.pth'
		model_name = os.path.join(model_path,model_name)
		torch.save(net.state_dict(),model_name)

'''
	previously shuffle the files for training  each time
'''
def shuffle_idx():
	global train_anno_list, train_wav_list
	per_arr = np.random.permutation(len(train_wav_list))
	train_wav_list = train_wav_list[per_arr]
	train_anno_list = train_anno_list[per_arr]

'''
	adjust the momentum
'''
def adjust_momentum(epoch):
	global lr
	lr = lr * pow(0.3, epoch)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

'''
	seperate train data
'''
def get_file(index):
    start = index * len(train_wav_list) // split_num #整除
    end = (index + 1) * len(train_wav_list) // split_num
    wav_files = train_wav_list[start:end]
    anno_files = train_anno_list[start:end]
    return wav_files, anno_files

def main(epoch):
	shuffle_idx()
	train(epoch)
	val(epoch)
	adjust_momentum(epoch)

if __name__ == '__main__':
	multiprocessing.freeze_support()
	test_dataset = onset_dataset(test_wav_list,
	                             test_anno_list,
	                             pad_length=pad_length,
	                             spec_style="cqt",
	                             dual_channel=False,
	                             is_filter=False,
	                             is_training=False)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
	                             num_workers=num_worker)
	# path = os.path.dirname(__file__)
	# data_path = os.path.join(path,"shuffle_data")
	if args.train:
		for epoch in range(args.epoches):
			main(epoch)