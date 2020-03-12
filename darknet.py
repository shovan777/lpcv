from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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
    # create the actual nn using the blocks
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
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
                module.add_module('batch_{}'.format(index), batchN_layer)
        


