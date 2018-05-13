#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 4/24/18 3:50 PM
# @Author  : Xiaoyu Chai
# @FileName: vgg16_example.py
# @Software: PyCharm

'''
Funtion: input one image, this program can implement classification and
feature extraction by vgg16.

'''

import os
import numpy as np

import torch
import torch.nn
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

import VGGNet

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_to_tensor = transforms.ToTensor()

# model & image path initializatioin
model_path = 'D:\Workspace\\remoteworkspace\\test_projects\\vgg16\\vgg16-397923af.pth'
img_path = 'husky_dog.jpg'

# model
def load_model(model_path):
    print('loading model ... \n')
    resmodel = VGGNet.vgg16(model_path)
    resmodel.cuda()
    return resmodel

# clasification
def inference(resmodel, imgpath):
    print('clasification: \n')
    resmodel.eval()  # evaluation mode

    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    tensor = tensor.cuda()  # 数据加载至gpu

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()  #
    max_index = np.argmax(result_npy[0]) #

    return max_index


# feature extraction
def extract_feature(resmodel, imgpath):
    print('feature extraction: \n')
    resmodel.fc = torch.nn.LeakyReLU(0.1)
    resmodel.eval() # 将模型设置为评估模式

    img = Image.open(imgpath) # 打开图片
    img = img.resize((224, 224)) # 调整图片尺寸
    tensor = img_to_tensor(img) # 将图片转化为张量tensor

    tensor = tensor.resize_(1, 3, 224, 224) # 张量中调整大小
    tensor = tensor.cuda() # 数据加载至gpu

    result = resmodel(Variable(tensor)) # 将变量送入模型
    result_npy = result.data.cpu().numpy() # 得到结果后将数据返回至cpu并以numpy的形式保存

    return result_npy[0]


if __name__ == "__main__":
    # load model
    model = load_model(model_path)
    # 显示图片
    print('test image:\n')
    test = mpimg.imread(img_path)
    plt.imshow(test)
    plt.show()
    # classification
    print(inference(model, img_path))

    # feature extraction
    # print(extract_feature(model, img_path))
    feature = extract_feature(model, img_path)
    print(feature)
