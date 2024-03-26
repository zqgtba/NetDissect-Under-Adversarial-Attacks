from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.autograd import Variable
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

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

final_dir = r"C:\Users\ThinkPad\Desktop\test"
orig_dir = r"C:\Users\ThinkPad\Desktop\test - 副本"
img_names = get_img_name(orig_dir)
for idx, img_name in enumerate(img_names):
    orig_path = os.path.join(orig_dir, img_name)
    final_path = os.path.join(final_dir, img_name)
    orig = cv2.imread(orig_path)[..., ::-1]
    final = cv2.imread(final_path)[..., ::-1]
    img = orig.copy().astype(np.float32)
    adv_img = final.copy().astype(np.float32)
    print(np.linalg.norm(abs(img - adv_img)))
    print(abs(img - adv_img))