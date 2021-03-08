# Human Body Part Segmentation

This repository contains the code associated to our paper: *Learning to Segment Human Body Parts with Synthetically Trained Deep Convolutional Networks*.
 
<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/HBPSegmentation/blob/main/paper/framework.png" width="800">
 </p>
 
**Abstract**. This paper presents a new framework for human body part segmentation based on Deep Convolutional Neural Networks trained using only synthetic data. The proposed approach achieves cutting-edge results without the need of training the models with real annotated data of human body parts. Our contributions include a data generation pipeline, that exploits a game engine for the creation of the synthetic data used for training the network, and a novel pre-processing module, that combines edge response map and adaptive histogram equalization to guide the network to learn the shape of the human body parts ensuring robustness to changes in the illumination conditions. For selecting the best candidate architecture, we performed exhaustive tests on manually-annotated images of real human body limbs. We further present an ablation study to validate our pre-processing module. The results show that our method outperforms several state-of-the-art semantic segmentation networks by a large margin.

If you use this code in an academic context, please cite our [paper](https://arxiv.org/abs/2102.01460):
```
@ARTICLE{Saviolo2021HBPSegmentation
  title={Learning to Segment Human Body Parts with Synthetically Trained Deep Convolutional Networks},
  author={Saviolo, Alessandro and Bonotto, Matteo and Evangelista, Daniele and Imperoli, Marco and Menegatti, Emanuele and Pretto, Alberto},
  journal={arXiv:2102.01460},
  year={2021}
}
```

## Installation

### Requirements

The code was tested with Ubuntu 18.04 and Anaconda 4.9.2.

If you do not have Anaconda installed, follow the instructions given at this [link](https://docs.anaconda.com/anaconda/install/linux/).

### Setup project
```
git clone https://github.com/AlessandroSaviolo/HBPSegmentation
cd HBPSegmentation
conda env update --name hbpsegmentation_env --file conda_environment.yml
conda activate hbpsegmentation_env
```

## Extract your first limbs from real images

1. Download provided test datasets:
```
cd ~/HBPSegmentation
gdown --id 1XbCc7Ukog_W6A6xlOt6a8BanRirGLNLD
unzip data.zip
rm data.zip
```

2. Download provided pre-trained segmentation model:
```
cd ~/HBPSegmentation
mkdir checkpoint
cd checkpoint
gdown --id 1p6Gjqzyji53OLcVFNwXTnfs6QlYKoivO
cd ..
```

3. Make predictions on High test set:
```
python predict.py -d high
```

**Note**:
- If you want to visualize results when predicting use the command ```python predict.py -d high -v True```
- Predicted masks are saved in ```~/HBPSegmentation/data/high/masks```
- The first time you run this project, when executing ```predict.py``` it will automatically download the encoder SE-ResNet-50

## Output example

<p align="center"> 
   <img src="https://github.com/AlessandroSaviolo/HBPSegmentation/blob/main/paper/results.gif" width="800">
</p>

(from left to right: input image, ground truth mask, predicted mask)

## Extract limbs from your custom images

**Important**: The segmentation network was trained over images of size 640x480 and the predict.py script automatically resizes the input images to that size. Make sure that your images roughly match that aspect ratio.

1. Create custom folder path:
```
cd ~/HBPSegmentation/data
mkdir -p custom_folder/imgs
```

2. Move custom images to the new custom folder ```custom_folder/imgs```.

3. Download provided pre-trained segmentation model:
```
cd ~/HBPSegmentation
mkdir checkpoint
cd checkpoint
gdown --id 1p6Gjqzyji53OLcVFNwXTnfs6QlYKoivO
cd ..
```
Note: skip this step if you already have downloaded the provided pre-trained model.

4. Make predictions on your custom images:
```
python predict.py -d custom_folder
```

**Note**:
- If you want to visualize results when predicting use the command ```python predict.py -d custom_folder -v True```
- Predicted masks are saved in ```~/HBPSegmentation/data/custom_folder/masks```
- The first time you run this project, when executing ```predict.py``` it will automatically download the encoder SE-ResNet-50

## Credits

The implementation of the segmentation networks used in this work are taken from [qubvel](https://github.com/qubvel/segmentation_models.pytorch).

We took inspiration from the PyTorch implementation of [sniklaus](https://github.com/sniklaus/pytorch-hed) for the Holistically-Nested Edge Detection algorithm.

## License

Copyright (C) 2021 Alessandro Saviolo, [FlexSight SRL](http://www.flexsight.eu/), Padova, Italy
```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
