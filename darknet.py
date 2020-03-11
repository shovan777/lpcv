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
    return blocks
