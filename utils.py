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
    # grid_size = in_dim // stride
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    # print('*******')
    # print(num_anchors)
    # print(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # attributes describe the dimensions of the input image,
    # which is larger (by a factor of stride) than the detection map.
    # thus, we must divide the anchors by the stride of the detection
    # feature map
    anchors = [(int(a[0])/stride, int(a[1])/stride) for a in anchors]

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
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # apply sigmoid activation to the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((
        prediction[:, :, 5: 5 + num_classes]
    ))

    # resize the detection map to the size of the input image
    prediction[:, :, :4] *= stride
    return prediction


def unique(tensor):
    """Find the object classes detected.

    Arguments:
        tensor {pytorch tensor} -- list of all detected object classes.

    Returns:
        pytorch tensor -- sorted set of detected object classes.
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res


def bbox_iou(box1, box2):
    """Calculate IoU between two boxes.

    Arguments:
        box1 {array} -- current box coordinates
        box2 {array list} -- list of array of other box coordinates

    Returns:
        float -- similarity between two boxes
    """
    # get the bounding box diagonal coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = (box1[:, 0], box1[:, 1],
                                  box1[:, 2], box1[:, 3])
    b2_x1, b2_y1, b2_x2, b2_y2 = (box2[:, 0], box2[:, 1],
                                  box2[:, 2], box2[:, 3])

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    # intersection area
    inter_area = (torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,
                              min=0) *
                  torch.clamp(inter_rect_y2 - inter_rect_y1,
                              min=0))

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou_score = inter_area / (b1_area + b2_area - inter_area)

    return iou_score


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # convert center box coordinate
    # to diangonal coordinates
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2]/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3]/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2]/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3]/2
    prediction[:, :, 4] = box_corner[:, :, 4]

    # Perform NMS over each image in a batch
    batch_size = prediction.size(0)
    write = False
    # write flag is used to indicate that we haven't intialized
    # output tensor that collects all true detections across
    # entire batch

    for batch_num in range(batch_size):
        image_pred = prediction[batch_num]  # image tensor
        max_conf, max_conf_object = torch.max(
            image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_object = max_conf_object.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_object)
        image_pred = torch.cat(seq, 1)

        non_zero_index = torch.nonzero(image_pred[:, 4])

        # The try-except block is there to handle situations
        # where we get no detections.
        # In that case, we use continue to
        # skip the rest of the loop body for this image.
        try:
            image_pred_ = image_pred[non_zero_index.squeeze(),
                                     :].view(-1, 7)
        except:
            continue
        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        # get the various classes detected in the image
        obj_classes = unique(image_pred_[:, -1])  # -1 holds the class index

        # perform NMS classwise
        for obj in obj_classes:
            # get detections with one particular class
            image_pred_1obj = image_pred_[image_pred_[:, -1] == obj]

            # sort in descending order by objectness score
            sort_ind = image_pred_1obj[:, -2].sort(descending=True)[1]
            image_pred_1obj_sorted = image_pred_1obj[sort_ind]
            num_det = image_pred_1obj_sorted.size(0)  # numbet of detections

            for i in range(num_det):
                # get the IOUs with all the boxes that come after the current box
                try:
                    iou_s = bbox_iou(
                        image_pred_1obj_sorted[i].unsqueeze(0),
                        image_pred_1obj_sorted[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # zero out all the detections that have IOU > threshold
                # remove the entries with IOU > threshold
                image_pred_1obj_pruned = image_pred_1obj_sorted[i+1:][
                    iou_s < nms_conf
                ]
            batch_ind = image_pred_1obj_pruned.new(image_pred_1obj_pruned.size(0),
                                                   1).fill_(batch_num)
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_1obj_pruned

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    # check if output has been initialized
    # if not initialized it means there isn't any detection
    try:
        return output
    except:
        return 0
