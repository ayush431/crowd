import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F


class listDataset(Dataset):
   
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        random.shuffle(root)
        print("iam at dataset.py under listdataser method")
        print("value of self",self)
        self.nSamples = len(root)
        self.lines = root
        print("self lines",self.lines)
        self.transform = transform
        print("Transforming an images",self.transform)
        self.train = train
        print("train  ",self.train)
        self.shape = shape
        print("shaping an images",self.shape)
        self.seen = seen
        print("seen  ",self.seen)
        self.batch_size = batch_size
        print("batch size ",self.batch_size)
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
        print("iam at len dataset.py")
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        print("index",index)
        img_path = self.lines[index]
        print("img_path",img_path)
        
        img,target = load_data(img_path,self.train)
        print("image loaded")
        
        if self.transform is not None:
            img = self.transform(img)
            print("inside if condition of dataset.py transform")
        return img,target
