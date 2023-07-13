import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from median_pool import MedianPool2d
import patch_config_multiobjective  # ori is patch_config
import math
import random
import os


config = patch_config_multiobjective.patch_configs['base']()    # ori is patch_config

def padding_resize(img, width, height):
    """
    Padding the width and height of the input image to the specified size of the model.
    :img: the input image
    :width: the input width required by the model
    :height: the input height required by the model
    :return: sized image, scale factor, padding size of width, padding size of height
    """
    scale = width / max(img.size()) 
    w0 = int(scale * img.size()[3])
    h0 = int(scale * img.size()[2]) 
    img_resized = F.interpolate(img, (h0, w0))  # 上下采样

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
    angle =math.pi / 180 * random.randint(-10, 10)
    theta1 = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle) ,0]], dtype=torch.float)
   
    grid1 = F.affine_grid(theta1.unsqueeze(0), patch_real.size()).cuda()
    out_patch = F.grid_sample(patch_real, grid1).cuda()

    out_patch = out_patch[:, :, delta_py:delta_py + size[2], delta_px:delta_px + size[3]]
    return out_patch
    

def get_label(labdirs, size_y):
    '''
    Get the ground truth of the person in images, and the position to paste the patch
    :labdirs: the path of the label(bounding box of persons)
    :size_y: the height of the original input image
    :return: pasting positon, ground truth
    '''
        
    gt = [] 
    adv_bboxes = []
    transform = transforms.ToTensor()
   
           
    for i in range(len(labdirs)):
        # change config from patch_config to patch_config_multiobjective
        label_dirs = os.path.join(config.label_dir, labdirs[i])
        gt = []
        label_fr = open(label_dirs, 'r')
        for lines in label_fr:
            gt.append([(float(x)) for x in lines.strip().split(' ')])
        gt = np.array(gt)
        width = gt[0][2] - gt[0][0]
        rd = random.uniform(1/3., 2/3)
        adv_width = rd * width / 2.
        adv_height = adv_width * 210 / 297.
        center_x = random.randint(int(gt[0][0] + adv_width), int(gt[0][2] - adv_width))

        top = gt[0][1] + width / 2.
        bottom = gt[0][1] + width * 3 / 2.
        center_y = random.randint(int(top + adv_height), int(min(bottom, size_y) - adv_height))

        adv_bboxes = np.array([[center_x - adv_width, center_y - adv_height, center_x + adv_width, center_y + adv_height]])            
         
        gt = torch.Tensor(gt)
       
        adv_bboxes = torch.Tensor(adv_bboxes).cuda()
        
    return adv_bboxes, gt
    
"""
Original implement of function get_car_label(labdirs) 
"""
def get_car_label(labdirs):
    gt = []
    adv_bboxes = []
    transform = transforms.ToTensor()

    for i in range(len(labdirs)):
        label_dirs = os.path.join(config.label_dir, labdirs[i])
        bonnet_dirs = os.path.join(config.bonnet_dir, labdirs[i])
        # gt_box = np.loadtxt(label_dirs)
        gt = []
        fr = open(label_dirs, 'r')
        for lines in fr:
            gt.append([float(x) for x in lines.strip().split(' ')[1:]])

        bonnet = np.loadtxt(bonnet_dirs)
        bonnet = bonnet.reshape((-1,4))
        Center_X = (bonnet[:, 2] + bonnet[:, 0]) / 2
        Center_Y = (bonnet[:, 3] + bonnet[:, 1]) / 2
        bonnet_height = bonnet[:, 3] - bonnet[:, 1]
        bonnet_width = bonnet[:, 2] - bonnet[:, 0]
        adv_height = (8 / 22.) * bonnet_height
        adv_width = (8 / 22.) * bonnet_width
        # rd = random.uniform(2., 3.)
        # adv_width = rd * adv_height
        adv_bboxes = np.array([Center_X - adv_width, Center_Y - adv_height, Center_X + adv_width, Center_Y + adv_height])

        adv_bboxes = adv_bboxes.transpose((1, 0))

    gt = torch.Tensor(np.array(gt))
    adv_bboxes = torch.Tensor(adv_bboxes).cuda()

    return adv_bboxes, gt

"""
load label info of multi-objective
"""
def get_multi_label(labdirs, size_y):
    '''
        Get the ground truth of the multi-objective in images, and the position to paste the patch
        :labdirs: the path of the label(bounding box of multi-objective)
        :size_y: the height of the original input image
        :return: pasting positon, ground truth
        '''

    gt = []
    adv_bboxes = []
    transform = transforms.ToTensor()

    for i in range(len(labdirs)):
        # change config from patch_config to patch_config_multiobjective
        label_dirs = os.path.join(config.label_dir, labdirs[i])
        # print("label_dirs: ", label_dirs)
        gt = []
        gt_label = []
        label_fr = open(label_dirs, 'r').readlines()
        
        for lines in label_fr[:1]:
            gt.append([(float(x)) for x in lines.strip().split(' ')[1:]])
            gt_label.append(lines.strip().split(' ')[0])
        class_name = gt_label[0]
        # print("gt_bbox and class_name is ", gt[0], class_name)


        """
        the latest version
        """
        gt = np.array(gt)
        width = gt[0][2] - gt[0][0]
        height = gt[0][3] - gt[0][1]
        rd = height / width
        if width < height:
            rd = width / height
        # print("ratio", rd)
        adv_width = width * rd * 0.9 / 2.
        adv_height = adv_width * 624 / 1040.

        wid_range = int(gt[0][2] - adv_width) - int(gt[0][0] + adv_width)
        center_x = random.randint(int(gt[0][0] + adv_width + wid_range * 0.2),
                                  int(gt[0][2] - adv_width - wid_range * 0.2))

        height_range = int(gt[0][3] - adv_height) - int(gt[0][1] + adv_height)
        center_y = random.randint(int(gt[0][1] + adv_height + height_range * 0.2),
                                  int(gt[0][3] - adv_height - height_range * 0.2))

        """
        the previous version
        """
        # gt = np.array(gt)
        # width = gt[0][2] - gt[0][0]
        # rd = random.uniform(1 / 3., 2 / 3)
        # adv_width = rd * width / 2.
        # adv_height = adv_width * 210 / 297.
        # # print("adv_width, adv_height is ", adv_width, adv_height)
        # left_range = int(gt[0][0] + adv_width)
        # right_range = int(gt[0][2] - adv_width)
        # if left_range < right_range:
        #     center_x = random.randint(left_range, right_range)
        # else:
        #     center_x = random.randint(right_range, left_range)
        # # top = gt[0][1] + width / 2.
        # # bottom = gt[0][1] + width * 3 / 2.
        # # center_y = random.randint(int(top + adv_height), int(min(bottom, size_y) - adv_height))
        # center_y = random.randint(int(gt[0][1] + adv_height/10.), int(gt[0][3] - adv_height))
        # # print("center_x, center_y is ", center_x, center_y)

        adv_bboxes = np.array(
            [[center_x - adv_width, center_y - adv_height, center_x + adv_width, center_y + adv_height]])

        gt = torch.Tensor(gt)

        adv_bboxes = torch.Tensor(adv_bboxes).cuda()

    return adv_bboxes, gt, class_name
