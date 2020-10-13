# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 16:27
# @Author  : moMing.yang
# @File    : senet.py
# @Software: PyCharm

import torch.nn as nn
import torch
import math
import collections
from model import onset_net_cfg
from torchsummary import summary
import torch.nn.functional as F
from collections import OrderedDict
'''
	尝试在原网络上加入S-E
	调换一下BN和ReLU的位置
'''
class SeLayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SeLayer, self).__init__()
		self.avg = nn.AdaptiveAvgPool2d(1)
		self.features = nn.Sequential(
			nn.Linear(channel, math.ceil(channel / reduction), bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(math.ceil(channel / reduction), channel, bias=False),
			nn.Sigmoid()
		)
	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg(x).view(b, c)
		y = self.features(y).view(b, c, 1, 1)
		return x * y.expand_as(x)

'''
	CBAM module
'''
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class SE_onsetnet(nn.Module):
	def __init__(self, pad_length=4, spec_style='cqt', dual_channel=False):
		super(SE_onsetnet, self).__init__()
		nchannel = 2 if dual_channel else 1  # 是否双通道
		self.config = onset_net_cfg['{}_pad_{}'.format(spec_style, pad_length)]  # 选择卷积网络

		self.features = nn.Sequential(
			nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1']),
            SeLayer(21),
			nn.BatchNorm2d(21),  # 归一化权重
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']),
			nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            SeLayer(42),
			nn.BatchNorm2d(42),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2'])
		)
		self.fc1 = nn.Linear(self.config['fc1'], 256)
		self.fc2 = nn.Linear(256, 1)
		self.drop = nn.Dropout()

	def forward(self, x):
		x = self.features(x)
		# print(x.size())

		x = x.view(-1, self.config['fc1'])
		x = self.drop(x)

		x = F.relu(self.fc1(x))

		x = self.drop(x)
		x = self.fc2(x)
		return torch.sigmoid(x)

class SE_onsetnet2(nn.Module):
    def __init__(self, pad_length=4, spec_style='cqt', dual_channel=False):
        super(SE_onsetnet2, self).__init__()
        n_channels = 1 if not dual_channel else 2
        self.config = onset_net_cfg['{0}_pad_{1}'.format(spec_style, pad_length)]
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 21, kernel_size=self.config['conv1']),
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            # SeLayer(21),
            nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']),

            nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            nn.BatchNorm2d(42),
            nn.ReLU(inplace=True),
            # SeLayer(42),
            nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2']),
        )
        self.fc1 = nn.Linear(in_features=self.config['fc1'], out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.drop = nn.Dropout()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        out_1 = self.features(x)
        out_1 = out_1.view(-1, self.config['fc1'])
        out_1 = self.drop(out_1)
        out_1 = self.relu(self.fc1(out_1))
        out_1 = self.drop(out_1)
        out_1 = self.fc2(out_1)
        return self.sig(out_1)

if __name__ == '__main__':
	if torch.cuda.is_available():
	    senet = SE_onsetnet2().cuda()
	    summary(senet, ( 1, 267, 9))

	# senet = SE_onsetnet().cuda()
	# input = torch.randn((256, 1, 267, 9)).cuda()
	# senet(input)
