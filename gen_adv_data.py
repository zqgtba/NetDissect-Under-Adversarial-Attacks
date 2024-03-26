from __future__ import print_function
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.autograd import Variable
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

#获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    # 使用 list(filter(lambda())) 筛选出 jpg 后缀的文件
    img_names = list(filter(lambda x: x.endswith(format), file_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    return img_names

def FGSM(model, img, epoch=1):
  for i in range(epoch):
    adver_example = fast_gradient_method(model, img.data, 0.05, np.inf) # 0.01
    adver_target = np.argmax(model(adver_example).data.cpu().numpy())
  return adver_example, adver_target

if __name__ == "__main__":

    img_dir = "/content/NetDissect-Lite/dataset/broden1_224/images/dtd"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_names = get_img_name(img_dir)
    img_num = len(img_names)
    print(img_num)

    model = models.vgg11(pretrained=True).to(device).eval()

    for idx, img_name in enumerate(img_names):

        print(f"--------------------{idx} / {img_num}---------------------")
        img_path = os.path.join(img_dir, img_name)
        orig = cv2.imread(img_path)[..., ::-1]
        # print(orig)
        img = orig.copy().astype(np.float32)
        img /= 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img=np.expand_dims(img, axis=0)
        img = Variable(torch.from_numpy(img).to(device).float())
        # label=np.argmax(model(img).data.cpu().numpy())
        # print(label)

        adv_img, adv_label = FGSM(model, img)
        # print(adv_label)
        adv_img = adv_img.reshape(3, 224, 224)
        adv_img = adv_img.cpu().detach().numpy().transpose(1, 2, 0)
        adv_img = adv_img*std + mean
        adv_img *= 255
        # print(adv_img)
        adv_img = adv_img[..., ::-1]
        # adv_img = Image.fromarray(np.uint8(adv_img)).convert('RGB')
        # adv_img.save(img_path)
        cv2.imwrite(img_path, adv_img)

        # final = cv2.imread(img_path)[..., ::-1]
        # print(final)
        # img = final.copy().astype(np.float32)
        # img /= 255.0
        # img = (img - mean) / std
        # img = img.transpose(2, 0, 1)
        # img=np.expand_dims(img, axis=0)
        # img = Variable(torch.from_numpy(img).to(device).float())
        # new_label=np.argmax(model(img).data.cpu().numpy())
        # print(new_label)
        # print(abs(orig - final))