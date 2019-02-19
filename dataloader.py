from torch.utils import data
import torch
import os
import random
import h5py
import numpy as np

class MNIST(data.Dataset):

	def __init__(self, file_dir):
		self.file = file_dir
		self.data = []
		self.labels = []
		self.preprocess()
		self.num_images = len(self.labels)

	def preprocess(self):
		file = h5py.File(self.file, 'r')
		self.data = np.array(file['data'])
		self.labels = np.array(file['labels'])
		# num_points = x.shape[1]
		# self.data = x
		# x_min = np.min(x, axis=1)
		# print(np.repeat(x_min, num_points, axis=1).reshape(x.shape))
		# x_max = np.max(x, axis=1)
		# print(x_max)
		# x = (x-x_min)/(x_max-x_min)
		# print(x.shape)
		# print(x_mean.shape)
		# self.labels = y
		# print(y.shape)

	def __getitem__(self, index):
		return torch.FloatTensor(self.data[index]), torch.LongTensor([self.labels[index]])

	def __len__(self):
		return self.num_images


def getLoader(image_path, batch_size, mode):
	return data.DataLoader(dataset=MNIST(image_path), batch_size=batch_size, shuffle=(mode=='train'), num_workers=1, drop_last=True)


# file = h5py.File("./mnistPC/train.hdf5", "r")
# print(file['labels'])
# print(list(file.keys()))