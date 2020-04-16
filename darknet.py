# external modules
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# self made modules
from utils import *


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_config(cfg_file):
    """Generate a list of nn blocks.

    Each block describes the block in the nn arch to be built.
    Block is represented as a dictionary in the list.

    Arguments:
        cfg_file {file} -- Provide the description of NN layers

    Returns:
        list -- list of blocks
    """

    # read the file and split each line
    with open(cfg_file, 'r') as file:
        lines = file.read().splitlines()

    # clean up
    lines = [line for line in lines if len(line) > 0]  # get rid of empty lines
    lines = [line for line in lines if line[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip()
             for x in lines]  # get rid of fringe whitespaces

    blocks = []
    block = {}

    for line in lines:
        if line[0] == '[':
            if block:
                blocks.append(block)
                block = {}
            layer_name = line.strip('[]').rstrip().lstrip()
            # get the name of the layer
            block['name'] = layer_name
        else:
            key, value = line.split('=')
            key = key.rstrip().lstrip()
            value = value.rstrip().lstrip()
            values = value.split(',  ')
            # check if it is a 2d attribute
            if len(values) > 1:
                values = [value.split(',') for value in values]
            else:
                values = value.split(',')
            # check if it is a single element
            if len(values) == 1:
                values = values[0]

            block[key] = values
    blocks.append(block)
    return blocks


def create_modules(blocks):
    """Generate NN layers and list them.

    Arguments:
        blocks {dict} -- Parsed dict from config.
    Return:
        net_info {dict} -- info about the input layer.
        module_list {list} -- Array of NN layers listed sequentially but not connected.

    """
    # create the actual nn using the blocks
    # Captures the information about the input and pre-processing
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

# iterate throught the blocks and create module for each block
    for index, layer in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check the type of block
        # create a new moduel for the block
        # append to module_list
        if (layer['name'] == 'convolutional'):
            try:
                batch_norm = int(layer['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True
            filters = int(layer['filters'])
            kernel_size = int(layer['size'])
            stride = int(layer['stride'])
            padding = int(layer['pad'])

            activation = layer['activation']

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv_layer = nn.Conv2d(
                in_channels=prev_filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=bias
            )
            module.add_module('Conv_{}'.format(index), conv_layer)

            # Add if there is batch norm
            if batch_norm:
                batchN_layer = nn.BatchNorm2d(
                    num_features=filters
                )
                module.add_module('BatchN_{}'.format(index), batchN_layer)

            # Check the type of activation and add
            if activation == 'leaky':
                # here the negative slope is 0.01 which is default
                # but the tutorial uses 0.1 why?????
                l_relu_layer = nn.LeakyReLU(
                    inplace=True,
                    negative_slope=0.01
                )
                module.add_module('LRelu_{}'.format(index), l_relu_layer)

        elif layer['name'] == 'upsample':
            stride = int(layer['stride'])
            upsample_layer = nn.Upsample(
                scale_factor=stride,
                mode='bilinear',
                align_corners=True
            )
            module.add_module('Upsample_{}'.format(index), upsample_layer)

        elif layer['name'] == 'route':
            prev_layers = layer['layers']
            # get the prev layers output
            # concatenate them and send as output
            try:
                # route layers may have two / one layers
                first_layer = int(prev_layers[0])
                second_layer = int(prev_layers[1])
            except:
                first_layer = int(prev_layers)
                second_layer = 0

            # calculate the number of outputs
            if not second_layer:  # no second layer
                filters = output_filters[index + first_layer]
            else:
                filters = output_filters[index + first_layer] +\
                    output_filters[second_layer]
            route_layer = EmptyLayer()
            module.add_module('Route_{}'.format(index), route_layer)
        elif layer['name'] == 'shortcut':
            # shortcut = Emptylayer()
            prev_layers = layer['from']
            activation = layer['activation']

            # filters = output_filters[prev_layers]
            shorcut_layer = EmptyLayer()
            module.add_module('Shorcut_{}'.format(index), shorcut_layer)
        elif layer['name'] == 'yolo':
            masks = layer['mask']
            masks = [int(mask) for mask in masks]

            anchors = layer['anchors']
            # throw any unmasked anchors
            anchors = [(int(anchors[i][0]), int(anchors[i][1])) for i in masks]
            detection_layer = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index),
                              detection_layer)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)


class Darknet(nn.Module):
    """Construct the darknet NN model."""

    def __init__(self, cfgfile):
        """Initialize NN.

        Arguments:
            cfgfile {file} -- cfg file that defines the NN architecture.
        """
        super(Darknet, self).__init__()
        self.blocks = parse_config(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def load_weights(self, weightfile):
        # open the weights file
        with open(weightfile, 'rb') as file:
            # The first 5 values are header information
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number
            # 4,5. Images seen by the network (during training)
            header = np.fromfile(file, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(file, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['name']

            # If module_type is convolutional load weights
            # Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                if (batch_normalize):

                    b_norm = model[1]

                    # get the number of weights of batch norm layer
                    num_bN_biases = b_norm.bias.numel()

                    # Load the weights
                    bN_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bN_biases])
                    ptr += num_bN_biases

                    bN_weights = torch.from_numpy(
                        weights[ptr:ptr + num_bN_biases])
                    ptr += num_bN_biases

                    bN_running_mean = torch.from_numpy(
                        weights[ptr:ptr + num_bN_biases])
                    ptr += num_bN_biases

                    bN_running_var = torch.from_numpy(
                        weights[ptr:ptr + num_bN_biases])
                    ptr += num_bN_biases

                    # cast  the loaded weights into dims of model weights
                    bN_biases = bN_biases.view_as(b_norm.bias.data)
                    bN_weights = bN_weights.view_as(b_norm.weight.data)
                    bN_running_mean = bN_running_mean.view_as(
                        b_norm.running_mean)
                    bN_running_var = bN_running_var.view_as(b_norm.running_var)

                    # copy the data to the model
                    b_norm.bias.data.copy_(bN_biases)
                    b_norm.weight.data.copy_(bN_weights)
                    b_norm.running_mean.copy_(bN_running_mean)
                    b_norm.running_var.copy_(bN_running_var)
                else:
                    # number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    # reshape the loaded weights as per dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # load the weights of the conv layers
                num_weights = conv.weight.numel()

                # load the weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def forward(self, x, CUDA):
        """Forward propagation function for darknet NN.

        Arguments:
            x {tensor} -- output
            CUDA {bool} -- flags if to used dedicated GPUdatetime A combination of a date and a time. Attributes: ()

        Returns:
            tensor -- detection with box attr in each row.
        """
        modules = self.blocks[1:]
        outputs = {}  # cache the outputs for route layer

        write = 0  # this flag indicates the encounter
        # of our first detection

        for i, module in enumerate(modules):
            module_name = (module['name'])

            if module_name == 'convolutional' or module_name == 'upsample':
                x = self.module_list[i](x)
            elif module_name == 'route':
                prev_layers = module['layers']
                try:
                    first_layer = int(prev_layers[0])
                    second_layer = int(prev_layers[1])
                except:
                    first_layer = int(prev_layers)
                    second_layer = 0

                if not second_layer:
                    x = outputs[i + first_layer]
                else:
                    x = torch.cat((outputs[i + first_layer],
                                   outputs[second_layer]), 1)
            elif module_name == 'shortcut':
                prev_layer = int(module['from'])
                activation = module['activation']
                x = outputs[i - 1] + outputs[i + prev_layer]
            elif module_name == 'yolo':
                anchors = self.module_list[i][0].anchors
                in_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                # transform the detection
                # print('*********')
                # print(x.shape)
                x = x.data
                x = predict_transform(x, in_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


# model = Darknet('cfg/yolov3.cfg')
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# # print(pred)
# # print(pred.shape)

# # blocks = parse_config("cfg/yolov3.cfg")
# # print(len(create_modules(blocks)[1]))

# # model = Darknet("cfg/yolov3.cfg")
# model.load_weights("yolov3.weights")
