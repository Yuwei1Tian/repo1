import random
import torch
import torch.utils.data as data
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from PIL import Image
import imageio
import skimage.transform
import os

random.seed(1143)

def populate_train_list(images_path):
    image_list = glob.glob(images_path + "*.*")

    train_list = image_list

    random.shuffle(train_list)

    return train_list


class My_Dataset(data.Dataset):
    def __init__(self, root1, root2):

        self.img_list1_ = os.listdir(root1)
        self.imgs_list1 = populate_train_list(root1)
        self.imgs_list2 = populate_train_list(root2)

        print("Total training examples:", len(self.imgs_list1))

    def __getitem__(self, index):
        img_path1 = self.imgs_list1[index]
        pil_img1 = Image.open(img_path1)

        img_path2 = self.imgs_list2[index]
        pil_img2 = Image.open(img_path2)


        pil_img1 = np.asarray(pil_img1)  # 由于用pil打开，需要转换数据类型
        pil_img2 = np.asarray(pil_img2)

        pil_img1 = skimage.transform.resize(pil_img1, (160, 160, 3))

        pil_img1 = (pil_img1.astype(np.float32)) / 255.0
        pil_img2 = (pil_img2.astype(np.float32)) / 255.0

        if pil_img1.shape.__len__() == 2:
            pil_img1 = np.expand_dims(pil_img1, 2)

        pil_img1 = np.transpose(pil_img1, (2, 0, 1))
        data1 = torch.from_numpy(pil_img1)

        if pil_img2.shape.__len__() == 2:
            pil_img2 = np.expand_dims(pil_img2, 2)
        pil_img2 = np.transpose(pil_img2, (2, 0, 1))
        data2 = torch.from_numpy(pil_img2)
        return data1, data2

    def __len__(self):
        return len(self.imgs_list1)

