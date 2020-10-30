import torch 
import torchvision
from model.student import MoVNect

def inference(model, img) : 
    loc, x, y, z = model(img)


def main() :
    model = MovNect()
