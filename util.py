from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, in_dim, anchors, num_classes, CUDA=True):
    """Transform detected feature map into 2d tensor.

    Arguments:
        prediction {4D tensor} -- output of detection layer(B*C*H*W)
        in_dim {shape} -- input box size.
        anchors {array} -- default boxes that are transformed to match prediction.
        num_classes {int} -- number of predicted classes.

    Keyword Arguments:
        CUDA {bool} -- flag to decide if GPU is used. (default: {True})

    Returns:
        3D tensor -- predictions reshaped with each box in consecutive rows.
        (B * grid_size * box_attrs(i.e 5))
    """
    batch_size = prediction.size(0)
    stride = in_dim // prediction.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # attributes describe the dimensions of the input image,
    # which is larger (by a factor of stride) than the detection map.
    # thus, we must divide the anchors by the stride of the detection
    # feature map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the centre_X, centre_Y. and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2)
    prediction[:, :, :2] += x_y_offset

    # Apply anchors to the bounding box dims

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid*grid_size, 1)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]*anchors)

    # apply sigmoid activation to the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((
        prediction[:, :, 5: 5 + num_classes]
    ))

    # resize the detection map to the size of the input image
    prediction[:, :, :4] *= stride
    return prediction
