import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet import Darknet

from median_pool import MedianPool2d
from torch.autograd import Variable


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """
    def __init__(self, cls_id, target_id,  num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.target = target_id

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 3)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)

        # transform the output tensor from [batch, 425, 13, 13] to [batch, 80, 845]
        output = YOLOoutput.view(batch, 3, 5 + self.num_cls , h * w)  # [batch, 5, 85, 169]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 169]
        output = output.view(batch, 5 + self.num_cls , 3 * h * w)  # [batch, 85, 845]
        output_objectness = torch.nn.Sigmoid()(output[:, 4, :])  # [batch, 845]
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 845]
         
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.softmax(output, dim=1)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = confs_for_class * output_objectness
        # find the max probability for car
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        max_target = normal_confs[:, self.target, max_conf_idx]

        return max_conf, max_conf_idx, max_target


class BBoxLoss(nn.Module):
    """BBoxLoss: obtains the iou value according the max class probability for class from YOLO output.

    Module providing the functionality necessary to obtain the iou value according the max class probability for one class from YOLO output.

    """
    def __init__(self, num_cls, width):
        super(BBoxLoss, self).__init__()
        self.num_cls = num_cls
        self.width = width


    def forward(self, YOLOoutput, max_conf_idx, groundTruth, anchors, img_width, img_height, scale, padding_x, padding_y):
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        # print("anchors: ", anchors)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 3)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        
        output = YOLOoutput.view(batch, 3, 5 + self.num_cls , h * w)  # [batch, 5, 85, 169]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 169]
        output = output.view(batch, 5 + self.num_cls , 3 * h * w)  # [batch, 85, 845]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 845]
        output_bbox = output[:, :4, :] #[batch, 4, 845]      

        bbox_pred = []
        # anchor_num = max_conf_idx / (w * h)
        anchor_num = max_conf_idx // (w * h)
        # anchor_num = torch.floor_divide(max_conf_idx, w * h)
        # print('BBoxLoss anchor_num is ', anchor_num)
         
        cell = max_conf_idx - anchor_num * w * h
        cell_x = cell % w
        cell_y = cell // w
        
        stride = self.width / w   
        
        bbox = output_bbox[:, :, int(max_conf_idx[0].item())]
 
        bx = torch.sigmoid(bbox[:, 0]) + cell_x.type(torch.FloatTensor).cuda()
        by = torch.sigmoid(bbox[:, 1]) + cell_y.type(torch.FloatTensor).cuda()

        bx = bx * stride
        by = by * stride        

        bw = anchors[2 * anchor_num[0]] * torch.exp(bbox[:, 2]) 
        bh = anchors[2 * anchor_num[0] + 1] * torch.exp(bbox[:, 3])

        boxes1 = torch.cat((bx - bw/2.0, by - bh/2.0, bx + bw/2.0, by + bh/2.0)).unsqueeze(0) 

        boxes1[:, 0] = (boxes1[:, 0] - padding_x) / scale           
        boxes1[:, 1] = (boxes1[:, 1] - padding_y) / scale           
        boxes1[:, 2] = (boxes1[:, 2] - padding_x) / scale           
        boxes1[:, 3] = (boxes1[:, 3] - padding_y) / scale   

        boxes2 = groundTruth

        # calculating the IOU  
        mx = torch.min(boxes1[:, 0], boxes2[:, 0])
        Mx = torch.max(boxes1[:, 2], boxes2[:, 2])
        my = torch.min(boxes1[:, 1], boxes2[:, 1])
        My = torch.max(boxes1[:, 3], boxes2[:, 3])
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        uw = Mx - mx
        uh = My - my
   
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh

        mask = ((cw <= 0) + (ch <= 0) > 0)
        
        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        carea[mask] = 0
        uarea = area1 + area2 - carea
        return carea/uarea


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side, patch_width):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side, patch_width),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side, width):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, width), red))
            printability_imgs.append(np.full((side, width), green))
            printability_imgs.append(np.full((side, width), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PersonDataset(Dataset):
    '''
    Dataloader
    '''

    def __init__(self, img_dir, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(img_dir), '*.txt'))
        
        self.len = n_images
        self.img_dir = img_dir


        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:

            # lab_path = os.path.join(self.img_dir, img_name + '.txt')
            lab_path = os.path.join(self.img_dir, img_name.replace('.jpg', '.txt'))
            self.lab_paths.append(lab_path)
            # print("PersonDataset self.img_paths: ", self.img_paths)
            # print("PersonDataset self.lab_paths: ", self.lab_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = self.img_names[idx].replace('.jpg', '.txt')   
        image = Image.open(img_path).convert("RGB")
        size = image.size
        image = transforms.ToTensor()(image)
        # print("PersonDataset image.size, lab_path: ", image.size(), lab_path)
        return image, lab_path
