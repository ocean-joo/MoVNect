import os
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, index):
        img = cv2.imread("../data/dummy.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.size, self.size))

        annot = None
        return img, annot

    def __len__(self):
        return 1000
    

class MPIIDataset(Dataset) :
    def __init__(self) :
        pass
    
    def __getitem__(self, index) :
        pass
    
    def __len__(self) :
        pass
    
    
class Human36MDataset(Dataset) :
    def __init__(self) :
        pass
    
    def __getitem__(self, index) :
        pass
    
    def __len__(self) :
        pass
        