#Copyright (C) 2021 Alessandro Saviolo, 
#FlexSight SRL, Padova, Italy

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import glob
import matplotlib.pyplot as plt
import torch
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import logging

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True

WIDTH = 640
HEIGHT = 480

def save(image, path):
  w, h, *_ = image.shape
  fig = plt.figure(figsize=(h/100, w/100))
  fig.add_axes([0, 0, 1, 1])
  plt.axis('off')
  plt.imshow(image, cmap='gray')
  plt.savefig(path)
  plt.close()

def run(data_path):
  clahe_2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  clahe_4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
  hed_network = HEDnetwork().cuda().eval()
  images_path = glob.glob(data_path+'/imgs/*.png')
  save_path = data_path+'/imgs_preprocessed'
  num_images = len(images_path)
  for i, image_path in enumerate(images_path):
    logging.info(f'Pre-processing image {i+1}/{num_images}')
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    save(image, image_path)
    image_clahe_2 = clahe_2.apply(image)
    image_clahe_4 = clahe_4.apply(image)
    image_hed = numpy.array(detect(hed_network, image_path))
    merged_image = cv2.merge((image_hed, image_clahe_2, image_clahe_4))
    save(merged_image, save_path+'/'+image_path.split('/')[-1])

def detect(network, image_path):
	input_tensor = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(image_path).convert("RGB"))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0/255.0)))
	output_tensor = network(input_tensor.cuda().view(1, 3, input_tensor.shape[1], input_tensor.shape[2]))[0, :, :, :].cpu()
	return PIL.Image.fromarray((output_tensor.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8))

class HEDnetwork(torch.nn.Module):
	def __init__(self):
		super(HEDnetwork, self).__init__()
		self.netVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)
		self.netVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)
		self.netVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)
		self.netVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)
		self.netVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)
		self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)
		url = 'http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch'
		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url=url, file_name='hed-bsds500').items() })

	def forward(self, tenInput):
		tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 127
		tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 127
		tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 127
		tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)
		tenVggOne = self.netVggOne(tenInput)
		tenVggTwo = self.netVggTwo(tenVggOne)
		tenVggThr = self.netVggThr(tenVggTwo)
		tenVggFou = self.netVggFou(tenVggThr)
		tenVggFiv = self.netVggFiv(tenVggFou)
		tenScoreOne = self.netScoreOne(tenVggOne)
		tenScoreTwo = self.netScoreTwo(tenVggTwo)
		tenScoreThr = self.netScoreThr(tenVggThr)
		tenScoreFou = self.netScoreFou(tenVggFou)
		tenScoreFiv = self.netScoreFiv(tenVggFiv)
		tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))

