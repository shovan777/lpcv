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
CUDA = torch.cuda.is_available()

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

# read the input images
read_dir_timer = time.time()

try:
    img_list = [osp.join(osp.realpath('.'), images,
                         img) for img in os.listdir((images))]
except NotADirectoryError:
    img_list = []
    img_list.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No such file or dir named {}".format(images))
    exit()

# create a dir to save the detections
if not os.path.exists(args.det):
    os.makedirs(args.det)

# load the images with opencv
load_batch_timer = time.time()
loaded_imgs = [cv2.imread(x) for x in img_list]

# pytorch variable with batch of images
img_batches = list(map(prep_image, loaded_imgs,
                       [inp_dim for x in range(len(img_list))]))
# keep a list containing the dimensions of original image
img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]
img_dim_list = torch.FloatTensor(img_dim_list).repeat(1, 2)
# @TODO: why are u repeat() here

if CUDA:
    img_dim_list = img_dim_list.cuda()


# create the batches
leftover = 0
if (len(img_dim_list) % batch_size):
    leftover = 1
# here 1 extra batch is added if there is some remainder

if batch_size != 1:
    num_batches = len(img_list) // batch_size + leftover
    img_batches = [torch.cat((img_batches[i*batch_size: min((i + 1)*batch_size,
                                                            len(img_batches))]))
                   for i in range(num_batches)]


write = 0
start_det_loop = time.time()
for i, batch in enumerate(img_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    # prediction = model(Variable(batch, volatile=True), CUDA)
    # but the Variable is deprecated as normal tensor
    # supports gradient calc. so we may use tensor also
    # also require_grad is false at inference as we don't need
    # grad or graph formation during inference
    # but we may need to do that for every layer instead
    # if we do volatile = True in input then, no graph is made
    # for any later operations
    # prediction = model(torch.Tensor(batch, volatile=True), CUDA)
    # since pytorch v0.4, it does not support volatile instead
    with torch.no_grad():
        prediction = model(batch, CUDA)
    prediction = write_results(prediction, confidence,
                               num_classes, nms_conf=nms_thresh)
    end = time.time()

    if type(prediction) == int:
        for img_num, image in enumerate(img_list[i*batch_size:
                                                 min((i + 1)*batch_size,
                                                     len(img_list))]):
            img_id = i*batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    # transform the atribute from index in batch
    # to index in imlist
    prediction[:, 0] += i*batch_size

    # if output in not initialized
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for img_num, image in enumerate(
        img_list[i*batch_size: min((i + 1)*batch_size,
                                   len(img_list))]):
        im_id = i*batch_size + img_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(
            image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

    # The line torch.cuda.synchronize makes sure that
    # CUDA kernel is synchronized with the CPU.
    # Otherwise, CUDA kernel returns the control to CPU
    # as soon as the GPU job is queued and well before
    # the GPU job is completed (Asynchronous calling).
    # This might lead to a misleading time if end = time.time()
    # gets printed before the GPU job is actually over.
