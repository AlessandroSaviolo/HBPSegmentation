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

import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import os
import cv2
import math
import glob
import logging
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader

WIDTH = 640
HEIGHT = 480

class Dataset(BaseDataset):
  def __init__(self, images_dir, masks_dir, image_scale=0.5, augmentation=None, preprocessing=None):
    self.ids = os.listdir(images_dir)
    self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
    self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
    self.image_scale = image_scale
    self.augmentation = augmentation
    self.preprocessing = preprocessing
    
  def __getitem__(self, i):
    image = cv2.imread(self.images_fps[i])
    image = cv2.resize(image, (int(self.image_scale * WIDTH), int(self.image_scale * HEIGHT)))
    mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    if self.augmentation:
      sample = self.augmentation(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    image, mask = normalize(image, mask)
    # (Height, Width, Channels) to (Channels, Height, Width)
    if self.preprocessing:
      sample = self.preprocessing(image=image, mask=mask)
      image, mask = sample['image'], sample['mask']
    else:
      image, mask = toTensor(image), toTensor(mask)
    return image, mask

  def getWithoutProcessing(self, i):
    image = cv2.imread(self.images_fps[i])
    mask = cv2.imread(self.masks_fps[i], 0)
    return image, mask

  def __len__(self):
    return len(self.ids)

def normalize(image, mask):
  cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
  cv2.normalize(mask, mask, 0, 1, cv2.NORM_MINMAX)
  return image, mask

def getDataLoader(path, batch_size, image_scale, encoder, encoder_weights, use_preprocessing_module=False, use_augmentation=None, shuffle_data=False):
  augmentation = getAugmentation() if use_augmentation else None
  preprocessing = getPreprocessing(smp.encoders.get_preprocessing_fn(encoder, encoder_weights)) if encoder_weights else None
  if use_preprocessing_module:
    dataset = Dataset(path+'/imgs_preprocessed', path+'/masks', image_scale=image_scale, augmentation=augmentation, preprocessing=preprocessing)
  else:
    dataset = Dataset(path+'/imgs', path+'/masks', image_scale=image_scale, augmentation=augmentation, preprocessing=preprocessing)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=8)
  return dataset, loader

def getAugmentation():
  train_transform = [
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomContrast(p=0.2),
    albu.IAAAdditiveGaussianNoise(p=0.2),
    albu.OneOf([albu.RandomBrightness(p=1), albu.RandomGamma(p=1)], p=0.9),
    albu.OneOf([albu.IAASharpen(p=1), albu.Blur(blur_limit=3, p=1), albu.MotionBlur(blur_limit=3, p=1)], p=0.9)
  ]
  return albu.Compose(train_transform)

def toTensor(x, **kwargs):
  if len(x.shape) == 2:
    x = np.expand_dims(x, axis=2)
  return x.transpose(2, 0, 1).astype('float32')

def getPreprocessing(preprocessing_fn):
  _transform = [albu.Lambda(image=preprocessing_fn), albu.Lambda(image=toTensor, mask=toTensor)]
  return albu.Compose(_transform)

def visualize(**images):
  n = len(images)
  plt.figure(figsize=(16, 5))
  for i, (name, image) in enumerate(images.items()):
    plt.subplot(1, n, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(' '.join(name.split('_')).title())
    plt.imshow(image)
  plt.show()

def parseHistory():
  history = pd.read_csv(max(glob.glob('./history/*.csv'), key=os.path.getctime))
  history.rename(columns={history.columns[0]: "epoch"}, inplace=True)
  return history

