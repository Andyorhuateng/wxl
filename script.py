#coding:utf8

import torch
import numpy as np

#验证gpu是否可用
# print(torch.cuda.is_available())

#numpy基本操作
# x = np.array([[1,2,3],
#               [4,5,6]])
#
# print(x.ndim)
# print(x.shape)
# print(x.size)
# print(np.sum(x))
# print(np.sum(x, axis=0))
# print(np.sum(x, axis=1))
# print(np.reshape(x, (3,2)))
# print(np.sqrt(x))
# print(np.exp(x))
# print(x.transpose())
# print(x.flatten())

#
# print(np.zeros((3,4,5)))
# print(np.random.rand(3,4,5))
#
# x = np.random.rand(3,4,5)
# x = torch.FloatTensor(x)
# print(x.shape)
# print(torch.exp(x))
# print(torch.sum(x, dim=0))
# print(torch.sum(x, dim=1))
# print(x.transpose(1, 0))
# print(x.flatten())

bn = torch.nn.BatchNorm1d(10)
torch.nn.BatchNorm2d()
torch.nn.BatchNorm3d()
print(bn.state_dict())
x = torch.randn(2, 10)
print(bn(x))
