# Cross-Domain Object Detection through Domain Adaptation

Deep learning neural networks perform well in object detection when applied on natural images but, these neural networks still struggle to identify objects in art images such as paintings and drawings. To address this issue, this project aims to study the object detection task, and how we can transfer the learned knowledge from a natural image to a Clipart Images. We could increase the accuracy of the model by 14 percent in terms of mAP in comparison to SSD model. For details, you can refer the project report present in the current repository.

This repository implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). This repository contains the code (in pyTorch) and I have used the code available at https://github.com/lufficc/SSD for training the SSD model for object detection.

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

Make the folder named 'datasets' in the project directory and make the folder structure like this:
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
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://github.com/pkr076/AIML_ExamProject/releases/download/v0.1/model_final_bl.pth
```
```bash
# For DT1 pretrained model: obtained by finetuning the baseline model for 10k iterations with domain transferred images obtained through CycleGAN
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://github.com/pkr076/AIML_ExamProject/releases/download/v0.2/model_final_DT1.pth
```
```bash
# For DT2 pretrained model: obtained by finetuning the baseline model for 10k iterations with domain transferred images obtained through AdaIN
python demo.py --config-file configs/vgg_ssd300_voc0712.yaml --images_dir demo --ckpt https://github.com/pkr076/AIML_ExamProject/releases/download/v0.3/model_final.pth
```

Then it will download and cache the model automatically and predicted images with boxes, scores and label names will saved to `demo/result` folder by default.

