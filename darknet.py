from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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

            activation = int(layer['activation'])

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
                mode='bilinear'
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
        super(Darknet, self).__init__()
        self.blocks = parse_config(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # cache the outputs for route layer

        for i, module in enumerate(modules):
            module_name = (module['name'])

            if module_name == 'convolutional' or module_name == 'upsampling':
                x = self.module_list[i](x)
            if module_name == 'route':
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
            if module_name == 'shortcut':
                prev_layer = module['from']
                activation = module['activation']
                x = outputs[i - 1] + outputs[i + prev_layer]
            if module_name == 'yolo':
