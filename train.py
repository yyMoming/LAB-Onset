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
rain_wav_list, train_anno_list, test_wav_list, test_anno_list = get_solf_wav_anno_files(path)
test_dataset = onset_dataset(test_wav_list,
			                 test_anno_list,
			                 pad_length=pad_length,
			                 spec_style="cqt",
			                 dual_channel=False,
			                 is_filter=False,
			                 is_training=True)

'''
	get test DataLoader, net, model save_path....
'''

test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,
                             num_workers=num_worker)
net = factory_net(net_style=net_style,pad_length=pad_length,spec_style=spec_style,
                  dual_channel=dual)
model_path = os.path.join('./model','{}_{}_{}_ywmtest_mono'.format(
        spec_style, net_style, pad_length))
if not os.path.exists(model_path):
    os.makedirs(model_path)
if use_cuda:
    net = net.cuda()#将所有的模型参数和缓存赋值GPU
    torch.cuda.manual_seed(1)
criterion = focalloss(alpha)#修正LOSS函数
# criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(),#构造网络优化器
                            lr=args.lr,#学习速率
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)#default=0.0005

if __name__ == '__main__':
	path = os.path.dirname(__file__)
	data_path = os.path.join(path,"shuffle_data")
	a,b,c,d =get_solf_wav_anno_files(data_path)
	print('hi')