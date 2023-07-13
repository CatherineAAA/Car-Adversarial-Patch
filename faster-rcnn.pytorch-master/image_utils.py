import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from median_pool import MedianPool2d
import patch_config  # ori is patch_config
import math
import random
import os

config = patch_config.patch_configs['base']()    # ori is patch_config

def padding_resize(img, width, height):
    """
    Padding the width and height of the input image to the specified size of the model.
    :img: the input image
    :width: the input width required by the model
    :height: the input height required by the model
    :return: sized image, scale factor, padding size of width, padding size of height
    """
    scale = width / max(img.size()) 
    print("scale is ", scale)
    w0 = int(scale * img.size()[3])
    h0 = int(scale * img.size()[2])
    print("w0, h0 is ", w0, h0) 
    img_resized = F.interpolate(img, (h0, w0))
    # img_resized = F.upsample(img, (h0, w0))
    # img_resized = img.resize(img.size()[0], img.size()[1], h0, w0)
    sized = torch.ones((img.size()[0], img.size()[1], height, width)) * 0.5
    
    x = int((width - w0) / 2.)
    y = int((height - h0) / 2.)
    sized[:,:,y:y+h0,x:x+w0] = img_resized
    sized = sized.cuda()
    return sized, scale, x, y


def patch_transform(path, patch_bboxes, size):
    """
    Transform the adversarial patch, including resize, rotation, blur etc.
    :path: the adversarial patch
    :patch_bboxes: the position to paste the patch
    :size: the size of the original input image. (batch, channel, height, width)
    :return: transformed patch
    """

    batch_size = config.batch_size
    adv_patch = MedianPool2d(7,same=True)(path.unsqueeze(0)).squeeze()

    brightness = torch.cuda.FloatTensor(batch_size).uniform_(config.min_brightness, config.max_brightness).cuda()

    # *****************random contrast******************
    contrast = torch.cuda.FloatTensor(batch_size).uniform_(config.min_contrast, config.max_contrast).cuda()
    
    adv_patch = adv_patch * contrast + brightness

    width = ((patch_bboxes[:, 2] - patch_bboxes[:, 0]) / 2.0).cpu().numpy()
    height = ((patch_bboxes[:, 3] - patch_bboxes[:, 1]) / 2.0).cpu().numpy()

    bboxes = np.array([np.array(config.patch_width / 2.0 - width), np.array(config.patch_height / 2.0 - height),
                       np.array(config.patch_width / 2.0 + width), np.array(config.patch_height / 2.0 + height)]).astype(np.int32)
    bboxes = bboxes.transpose(1, 0)

    # ***********paste the resized patch on the car****************
    if size[2] > size[3]:
        delta_px = int((size[2] - size[3]) / 2.)
        delta_py = 0
        patch_real = torch.zeros(size[0], size[1], size[2], size[2]).cuda()
    else:
        delta_py = int((size[3] - size[2]) / 2.)
        delta_px = 0
        patch_real = torch.zeros(size[0], size[1], size[3], size[3]).cuda()

    # ***************resize generated patch by using affine transformation ***************
    sx = config.patch_width / (patch_bboxes[:, 2] - patch_bboxes[:, 0])
    sy = config.patch_height / (patch_bboxes[:, 3] - patch_bboxes[:, 1])
    theta = torch.Tensor([[sx[0].item(), 0, 0], [0, sy[0].item(), 0]]).cuda()

    box_height = bboxes[0][3] - bboxes[0][1]
    box_wid = bboxes[0][2] - bboxes[0][0]

    y0 = int(patch_bboxes[0][1].item())
    y1 = int(patch_bboxes[0][1].item() + box_height)
    x0 = int(patch_bboxes[0][0].item())
    x1 = int(patch_bboxes[0][0].item() + box_wid)

    # boundary value discussion
    if x1 >= size[3]:
        delta_x = (x1 - size[3])
        x1 -= delta_x
        x0 -= delta_x

    if x0 <= 0:
        delta_x = (0 - x0)
        x1 += delta_x
        x0 += delta_x

    if y1 >= size[2]:
        delta_y = y1 - size[2]
        y1 -= delta_y
        y0 -= delta_y

    if y0 <= 0:
        delta_y = (0 - y0)
        y1 += delta_y
        y0 += delta_y
    # print("y0, y1, x0, x1 is ", y0, y1, x0, x1)
    # print("delta_x, delta_y is ", delta_x, delta_y)
    # print("box_width, box_height is ", box_wid, box_height)

    # scale the patch
    if sx >= 1:
        # zoom out
        grid = F.affine_grid(theta.unsqueeze(0), adv_patch.unsqueeze(0).size()).cuda()                        
        applier = F.grid_sample(adv_patch.unsqueeze(0), grid)
        patch_x0 = x0 + delta_px
        patch_x1 = x0 + delta_px + box_wid
        patch_y0 = y0 + delta_py
        patch_y1 = y0 + delta_py + box_height
        
        patch_real[0, :, patch_y0:patch_y1, patch_x0:patch_x1] = applier[0, :, int(bboxes[0][1]):int(bboxes[0][3]), int(bboxes[0][0]):int(bboxes[0][2])]                        

    else:
        # zoom in
        padx0 = int(((patch_bboxes[:, 2] - patch_bboxes[:, 0]) - config.patch_width) / 2.)
        padx1 = int(patch_bboxes[:, 2] - patch_bboxes[:, 0] - config.patch_width - padx0)
        pady0 = int(((patch_bboxes[:, 3] - patch_bboxes[:, 1]) - config.patch_height) / 2.)
        pady1 = int(patch_bboxes[:, 3] - patch_bboxes[:, 1] - config.patch_height - pady0)
        adv_patch_expand = torch.nn.ConstantPad2d((padx0, padx1, pady0, pady1), 0)(adv_patch).cuda()
        grid = F.affine_grid(theta.unsqueeze(0), adv_patch_expand.unsqueeze(0).size()).cuda()
        applier = F.grid_sample(adv_patch_expand.unsqueeze(0), grid)
        patch_x0 = x0 + delta_px
        patch_x1 = x1 + delta_px
        patch_y0 = y0 + delta_py
        patch_y1 = y1 + delta_py
     
        patch_real[0, :, patch_y0:patch_y1, patch_x0:patch_x1] = applier[0,:,0:(y1-y0),0:(x1-x0)]


    # rotation the patch_real
    angle =math.pi / 180 * random.randint(-5, 5)
    theta1 = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle) ,0]], dtype=torch.float)
   
    grid1 = F.affine_grid(theta1.unsqueeze(0), patch_real.size()).cuda()
    out_patch = F.grid_sample(patch_real, grid1).cuda()

    out_patch = out_patch[:, :, delta_py:delta_py + size[2], delta_px:delta_px + size[3]]
    return out_patch

def get_car_label(labdirs):
        gt = []
        adv_bboxes = []

        for i in range(len(labdirs)):
            label_dirs = os.path.join(config.label_dir, labdirs[i])
            bonnet_dirs = os.path.join(config.bonnet_dir, labdirs[i])
            gt = []
            fr = open(label_dirs, 'r')
            for lines in fr:
                gt.append([float(x) for x in lines.strip().split(' ')[1:]])

            bonnet = np.loadtxt(bonnet_dirs)
            bonnet = bonnet.reshape((-1, 4))
            Center_X = (bonnet[:, 2] + bonnet[:, 0]) / 2
            Center_Y = (bonnet[:, 3] + bonnet[:, 1]) / 2
            bonnet_height = bonnet[:, 3] - bonnet[:, 1]
            bonnet_width = bonnet[:, 2] - bonnet[:, 0]
            adv_height = (8 / 22.) * bonnet_height
            adv_width = (8 / 22.) * bonnet_width

            adv_bboxes = np.array([Center_X-adv_width, Center_Y-adv_height, Center_X+adv_width, Center_Y+adv_height])
            adv_bboxes = adv_bboxes.transpose((1, 0))

        gt = torch.Tensor(np.array(gt))
        adv_bboxes = torch.Tensor(adv_bboxes).cuda()

        return adv_bboxes, gt
