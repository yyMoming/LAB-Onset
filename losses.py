# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 9:33
# @Author  : moMing.yang
# @File    : losses.py
# @Software: PyCharm
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index,classes):
	size = index.size()+(classes,)
	view = index.size()+(1,)
	mask = torch.Tensor(*size).fill_(0)
	index = index.view(*view).long()
	ones = 1.0

	if isinstance(index,Variable):
		ones = Variable(torch.Tensor(index.size()).fill_(1)).cuda()
		mask = Variable(mask,volatile=index.volatile).cuda()

	return mask.scatter_(1,index,ones)

class focalloss(nn.Module):
	"""docstring for F"""
	def __init__(self,alpha=3.0,eps=1e-7,size_average=True):
		super(focalloss, self).__init__()
		self.eps = eps
		self.size_average = size_average
		self.alpha = alpha

	def forward(self,inputs,target):
		#y = one_hot(target,inputs.size(-1))
		inputs = inputs.clamp(self.eps,1-self.eps)
		loss = -(self.alpha*target*torch.log(inputs)+(1-target)*torch.log(1-inputs))
		#loss = -(self.alpha*y*torch.log(logit)+(1-self.alpha)*(1-y)*torch.log(1-logit))
		#loss = loss*(1-inputs)**self.gamma

		if self.size_average:return loss.mean()
		else:return loss.sum()

if __name__ == '__main__':
    x = torch.randn(100,1,9)
    y = torch.randn(100,1,9)
    fo = focalloss()
    loss = fo(x,y)
    print(loss)