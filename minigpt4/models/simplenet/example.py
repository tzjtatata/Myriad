from simplenet_interface import SimpleNet
import yaml
import cv2
from torchvision import transforms
import os
import sys
from PIL import Image
import numpy as np
import torch

if __name__ == "__main__":
    ckpt_rt="./SimpleNet/results/MVTecAD_Results/simplenet_mvtec/run/models/0"
    yaml_path="params.yaml" # if not set, using params in package
    img_path="./mvtec/bottle/test/good/000.png"
    img=Image.open(img_path)
    net=SimpleNet(ckpt_rt)
    s,m=net.predict(img,"bottle",use_torch_transform=True)
    print(s)
    cv2.imwrite("test.png",(m[0][...,np.newaxis])*255)
