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

import os
import glob
import logging
import argparse
import torch
import cv2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import warnings
import matplotlib.pyplot as plt

import utils
import preprocessing_module

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")


def predict(net, device, data, image_scale, use_preprocessing_module, visualize):
  testing_path = './data/'+data
  logging.info(f'Predicting {testing_path} testing data.')

  assert os.path.exists(testing_path), 'The data folder specified does not exist! '+\
                                       'Please, make sure that the specified folder is inside data directory.'
  assert len(glob.glob(testing_path+'/imgs/*')), 'No images have been found! '+\
                                                 'Please, make sure that the '+testing_path+'/imgs folder contains images.'
  if use_preprocessing_module:
    if not os.path.exists(testing_path+'/imgs_preprocessed'):
      logging.info(f'Creating imgs_preprocessed folder.')
      os.makedirs(testing_path+'/imgs_preprocessed')
    if not len(glob.glob(testing_path+'/imgs_preprocessed/*')):
      logging.info(f'Pre-processing module is enabled.')
      preprocessing_module.run(testing_path)
  if not os.path.exists(testing_path+'/masks'):
    logging.info(f'Creating masks folder.')
    os.makedirs(testing_path+'/masks')
  test_dataset, test_loader = utils.getDataLoader(
    testing_path,
    batch_size=1,
    image_scale=image_scale,
    encoder='se_resnet50',
    encoder_weights='imagenet',
    use_preprocessing_module=use_preprocessing_module,
    use_augmentation=False,
    shuffle_data=False
  )
  loss = smp.utils.losses.DiceLoss()
  metrics = [smp.utils.metrics.IoU()]
  test_epoch = smp.utils.train.ValidEpoch(model=net, loss=loss, metrics=metrics, device=device)
  test_logs = test_epoch.run(test_loader)
  num_predictions = len(glob.glob(testing_path+'/imgs/*'))
  for i in range(num_predictions):
    logging.info(f'Testing image {i+1}/{num_predictions}.')
    image_vis = test_dataset.getWithoutProcessing(i)[0]
    image, _ = test_dataset[i]
    predicted_mask = (net.predict(torch.from_numpy(image).to(device).unsqueeze(0)).squeeze().cpu().numpy().round()).astype('uint8')
    save_path = testing_path+'/masks/'+test_dataset.images_fps[i].split('/')[-1]
    save(image_vis, predicted_mask, save_path)
    if visualize:
      utils.visualize(image=image_vis, predicted_mask=predicted_mask)

def save(image, predicted_mask, save_path):
  w, h = predicted_mask.shape
  fig = plt.figure(figsize=(25.92, 19.44))
  fig.add_axes([0, 0, 1, 1])
  plt.axis('off')
  plt.imshow(predicted_mask, cmap='gray')
  plt.savefig(save_path)
  plt.close()

def getArgs():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', dest='model', type=str, default=max(glob.glob('./checkpoint/*.pth'), key=os.path.getctime),
                      help="Specify the file in which the model is stored (default takes last model saved in checkpoint folder)")
  parser.add_argument('-s', '--scale', dest='image_scale', type=float, default=1.0,
                      help="Scale factor for the input images (default size is 640x480)")
  parser.add_argument('-d', '--data', dest='data', type=str, default='high',
                      help='Data folder (default is high)')
  parser.add_argument('-p', '--use_preprocessing_module', dest='use_preprocessing_module', type=bool, default=True,
                      help='Use pre-processing module (default is True)')
  parser.add_argument('-v', '--visualize', dest='visualize', type=bool, default=False,
                      help='Visualize predicted masks (default is False)')
  return parser.parse_args()

def main(args):
  net = smp.UnetPlusPlus(encoder_name='se_resnet50', encoder_weights='imagenet', classes=1, activation='sigmoid')
  try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(args.model, map_location=device))
  except RuntimeError:
    device = 'cpu'
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info(f'Using {device}!')
  logging.info(f'Model loaded from {args.model}.')
  net.to(device=device)
  predict(net, device, args.data, args.image_scale, args.use_preprocessing_module, args.visualize)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = getArgs()
  main(args)

