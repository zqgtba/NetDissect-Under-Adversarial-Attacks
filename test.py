from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import cv2
from PIL import Image
from torch.autograd import Variable
#获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
#图像加载以及预处理
image_path=r"D:\Downloads\BaiduNetdiskDownload\broden1_227\images\opensurfaces\7.jpg"
orig = cv2.imread(image_path)[..., ::-1]
# orig = cv2.resize(orig, (227, 227))
img = orig.copy().astype(np.float32)
 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)
 
img=np.expand_dims(img, axis=0)
 
img = Variable(torch.from_numpy(img).to(device).float())
print(img.shape)
 
#使用预测模式 主要影响droupout和BN层的行为，用的是resnet18模型，现成的
model = models.resnet18(pretrained=True).to(device).eval()
#取真实标签
label=np.argmax(model(img).data.cpu().numpy())#这里为什么要加cup（）？因为np无法直接转为cuda使用，要先转cpu
print("label={}".format(label))
 
epoch = 1#训练轮次
target = Variable(torch.Tensor([float(label)]).to(device).long())#转换数据类型
print(target)


#导入cleverhans中的FGSM函数
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
def FGSM(model):
  for i in range(epoch):
    adver_example = fast_gradient_method(model, img.data, 0.01, np.inf)
    adver_target = torch.max(model(adver_example),1)[1]
    if adver_target != target:
        print("FGSM attack 成功")
    print("epoch={} adver_target={}".format(epoch,adver_target))
    print(adver_example.shape)
  return adver_example
adver_tensor = FGSM(model).reshape(3, 227, 227)
adver_tensor = adver_tensor.cpu().detach().numpy().transpose(1, 2, 0)
adver_tensor = adver_tensor*std + mean
adver_tensor *= 255
adver_image = Image.fromarray(np.uint8(adver_tensor)).convert('RGB')
savepath = r"C:\Users\ThinkPad\Desktop\adv_image.jpg"
adver_image.save(savepath)