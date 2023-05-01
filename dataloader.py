import os
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
from torch.utils.data import dataset

random.seed(1143)


def populate_list(images_path):

	image_list = glob.glob(images_path +"/**/"+"*.jpg",recursive=True)+glob.glob(images_path +"/**/"+"*.png",recursive=True)

	random.shuffle(image_list)

	return image_list

	

class unpaired_data_loader(data.Dataset):

	def __init__(self, images_path):

		self.data_list = populate_list(images_path)
		self.size = 256
		print("Total examples:", len(self.data_list))


		

	def __getitem__(self, index):

		data_path = self.data_list[index]
		
		data = Image.open(data_path)
		
		data= data.resize((self.size,self.size), Image.ANTIALIAS)

		data = (np.asarray(data)/255.0)
		data= torch.from_numpy(data).float()

		return data.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)


class paired_data_loader(data.Dataset):

	def __init__(self, images_path):
		self.data_list = populate_list(os.path.join(images_path,"low"))
		self.label_list = [path.replace("low","high") for path in self.data_list]
		self.size = 256
		print("Total data:"+str(len(self.data_list))+"Total label:"+str(len(self.label_list)))

	def __getitem__(self, index):
		data_path = self.data_list[index]
		label_path = self.label_list[index]
		data = Image.open(data_path)
		label = Image.open(label_path)
		data = data.resize((self.size, self.size), Image.ANTIALIAS)
		label =  label.resize((self.size, self.size), Image.ANTIALIAS)
		data = (np.asarray(data) / 255.0)
		label = (np.asarray(label) / 255.0)
		data = torch.from_numpy(data).float()
		label = torch.from_numpy(label).float()

		return data.permute(2, 0, 1), label.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)