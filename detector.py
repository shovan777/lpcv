from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utils import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """Parse arguments for NN arch and input info.

    Returns:
        object -- has arguments as properties with their values.
    """
    parser = argparse.ArgumentParser(description='YOLO v3\
         Detection Module')
    parser.add_argument("--images", dest="images",
                        help="Image / Directory containing images to perform detection",
                        default="imgs", type=str)
    parser.add_argument("--det", dest="det",
                        help="Image / Directory to store detections",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs",
                        help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object confidence threshold",
                        default=0.5)
    parser.add_argument("--nms_threshold", dest="nms_threshold",
                        help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest="cfgfile",
                        help="Model Configuration file")
    parser.add_argument("--weights", dest="weightfile",
                        help="Pretrained weight file",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest="reso",
                        help="Input resolution of the network. \
                            Increase to increase accuracy. \
                            Decrease to increase speed",
                        default="416", type=str)
    # parser
    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_threshold)
start = 0

# check if gpu is available
cuda = torch.cuda.is_available()

# load the object classes
def load_classes(names_file):
    with open(names_file, 'r') as file:
        # here :-1 is sliced to remove empty line at the end
        names = file.read().split('\n')[:-1]

class_path = 'data/coco.names'
num_classes = 80
classes = load_classes(class_path)

# Initialize the network and load the weights

# setup the NN
print("Loading network................")
model = Darknet(args.cfgfile)
model.load_weights(args.weightfile)
print("Network load sucessfully")

model.net_info['height'] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# if GPU available, put model in GPU
if cuda:
    model.cuda()

# set the model in evaluation mode
model.eval()

