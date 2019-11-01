# -*- coding: utf-8 -*-
# @Time    : 2019/11/1 16:03
# @Author  : moMing.yang
# @File    : dataloader_test.py
# @Software: PyCharm
#
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

x_np = np.random.randn(99, 1, 267, 9)
y = np.random.randn(99)


class myData(Dataset):
	def __init__(self):
		self.x = x_np
		self.y = y

	def __getitem__(self, index):
		return torch.FloatTensor(self.x[index]), torch.FloatTensor([self.y[index]])

	def __len__(self):
		return self.y.size


myset = myData()
dataloader = DataLoader(dataset=myset, shuffle=True, batch_size=15,num_workers=4)
if __name__ == '__main__':

	for i,(x,y) in enumerate(dataloader):
		print(i,"--","x.size:",x.size(),"y.size:",y.size())