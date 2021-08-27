# Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation


This repository implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). This repository aims to be the code base for researches based on SSD.


## Installation
### Requirements

1. Python3
1. PyTorch 1.0 or higher
1. yacs
1. [Vizer](https://github.com/lufficc/Vizer)
1. GCC >= 4.9
1. OpenCV


### Step-by-step installation

```bash
git clone https://github.com/pkr076/AIML_ExamProject.git
cd SSD
# Required packages: torch torchvision yacs tqdm opencv-python vizer
pip install -r requirements.txt

# Done! That's ALL! No BUILD! No bothering SETUP!

# It's recommended to install the latest release of torch and torchvision.
```


## Train

### Setting Up Datasets
#### Pascal VOC

Make the folder structure like this:
```
datasets
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ clipart1k
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
|__ ...
```




### Baseline SSD training

For training the SSD, the parameters can be set inside file present at config/vgg_ssd300_voc0712.yaml. By default, the model is trained with lr = 1e-3 for 60k iterations and then with lr = 1e-4 for next 20k iterations, total of 80k iterations. Use the following command to start the training.
```bash
python train.py --config-file configs/vgg_ssd300_voc0712.yaml --skip-test --eval_step -1
```
### Finetune the Baseline SSD model
To finetune the baseline SSD model, follow the following steps:

1) Replace the VOC2007 and VOC2012 images (present inside the JPEGImages folder) with their domain transferred versions (obtained through CycleGAN or AdaIN).
2) Make the necessary changes in the config file `vgg_ssd300_voc0712.yaml`. For example, if the baseline training was done for MAX_ITER = 80000 and if you want to finetune this for 10000 more iterations then make the MAX_ITER=90000.


## Evaluate

To test the model against the clipart dataset, use the following command

```bash
# for example, evaluate SSD300:
python test.py --config-file configs/vgg_ssd300_voc0712.yaml
```



## Demo

Predicting image in a folder is simple: 
```bash
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt ./path/of/your/model_final.pth
```
If you want to use the pretrained models, use the following commands -
```bash
# For baseline pretrained model:
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://drive.google.com/file/d/1MqlQE0dp1eSHuKSqRk1VR6Muep1Lkclh/view?usp=sharing
```
```bash
# For DT1 pretrained model: obtained by finetuning the baseline model for 10k iterations with domain transferred images obtained through CycleGAN
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://drive.google.com/file/d/1E7li2fWtchOxs8WIcYBl9AEwrvskymLQ/view?usp=sharing
```
```bash
# For DT2 pretrained model: obtained by finetuning the baseline model for 10k iterations with domain transferred images obtained through AdaIN
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://drive.google.com/file/d/1cQuk_6uElFONrCgzxubf0YH0KHYLkFFa/view?usp=sharing
```

Then it will download and cache the model automatically and predicted images with boxes, scores and label names will saved to `demo/result` folder by default.




## Develop Guide

If you want to add your custom components, please see [DEVELOP_GUIDE.md](DEVELOP_GUIDE.md) for more details.


## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel free to open a new issue.
