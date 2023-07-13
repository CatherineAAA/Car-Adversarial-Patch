"""
created by yipengao
physical attack for faster rcnn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import imutils
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import scipy
import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
from tqdm import tqdm
import patch_config
from load_data import *
from patch_warp import PatchWarp
from image_utils import padding_resize, patch_transform, get_car_label

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='./cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="./data/pretrained_model")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="./images")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save images for test',
                      default="./detect_res")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=7, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args

pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  # print("------------------func: _get_image_blob-------------------")
  im_orig = im.astype(np.float32, copy=True)
  # scipy.misc.imsave("/data/yipengao/code/faster-rcnn.pytorch-master/adv_examples/cls_iou/im_orig_0.png", im_orig)
  im_orig -= cfg.PIXEL_MEANS
  # scipy.misc.imsave("/data/yipengao/code/faster-rcnn.pytorch-master/adv_examples/cls_iou/im_orig_sub_means_0.png", im_orig)

  im_shape = im_orig.shape
  # print("im_shape is ", im_shape)
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  # print("im_size_min and im_size_max is ", im_size_min, im_size_max)
  processed_ims = []
  im_scale_factors = []
  # print("cfg.TEST.SCALES is ", cfg.TEST.SCALES)
  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    # scipy.misc.imsave(f"/data/yipengao/code/faster-rcnn.pytorch-master/adv_examples/cls_iou/im_resize_{target_size}.png", im)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  # print("----------------func: _get_image_blob over-----------------")
  return blob, np.array(im_scale_factors)

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.patch_height = self.config.patch_height
        self.patch_width = self.config.patch_width
        # model
        self.variations = self.load_model()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_height, self.config.patch_width).cuda()
        self.total_variation = TotalVariation().cuda()
        self.ph = 30
        self.param = 0.0001
        self.scale = 1.0
        self.x = 0.1
        self.y = -15
        self.deformation = 50
        self.warp_patch = PatchWarp(self.ph, self.param, self.scale, self.x, self.y)

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        cfgfile = self.config.cfgfile
        weightfile = self.config.weightfile

        if cfgfile is not None:  # 配置文件
            cfg_from_file(cfgfile)
        if args.set_cfgs is not None:  # 设置配置
            cfg_from_list(args.set_cfgs)

        cfg.USE_GPU_NMS = args.cuda

        print('Using config:')
        pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)  # 设置随机数种子，每次运行代码时设置相同的seed，则每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样

        if args.net == 'vgg16':
            fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            # 到了pdb.set_trace()就会定下来，就可以看到调试的提示符（Pdb）了
            pdb.set_trace()

        fasterRCNN.create_architecture()  # 初始化faster rcnn模型，初始化权重

        print("load checkpoint %s" % (weightfile))
        if args.cuda > 0:
            checkpoint = torch.load(weightfile)
        else:
            # 在cpu上加载预先训练好的GPU模型，强制所有的GPU张量在CPU中的方式
            checkpoint = torch.load(weightfile, map_location=(lambda storage, loc: storage))
        fasterRCNN.load_state_dict(checkpoint['model'])  # 恢复模型
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')

        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        batch_size = self.config.batch_size
        n_epochs = 1000
        time_str = time.strftime("%Y%m%d-%H%M%S")
        im_data, im_info, num_boxes, gt_boxes = self.variations[0], self.variations[1], self.variations[2], self.variations[3]
        # Generate starting point
        adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("")
        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(PersonDataset(self.config.img_dir,
                                                                 shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        # print("force_cpu is ", cfg.USE_GPU_NMS)       

        for epoch in range(n_epochs):
            ep_loss = 0
            length = 500    # self.epoch_length
            for i_batch, (images, datadirs) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                    total=self.epoch_length):
                if i_batch < length:
                    with autograd.detect_anomaly():
                        images_batch = images.cuda()
                        size = images.size()

                        patch_bboxes, gt = get_car_label(datadirs)
                        gt = gt.cuda()

                        # crop the bbox of the car
                        mask_w = (gt[:, 2] - gt[:, 0]).cpu().numpy()
                        mask_h = (gt[:, 3] - gt[:, 1]).cpu().numpy()

                        mask = [gt[:, 0].cpu().numpy() - 1 / 10. * mask_w, gt[:, 1].cpu().numpy() - 1 / 10. * mask_h,
                                gt[:, 2].cpu().numpy() + 1 / 10. * mask_w, gt[:, 3].cpu().numpy() + 1 / 10. * mask_h]
                        mask = torch.Tensor(np.array(mask).transpose((1, 0)))
                        image_mask = torch.ones_like(images).cuda() * 0.5
                        for gt_idx in range(len(mask)):
                            image_mask[:, :, int(mask[gt_idx][1].item()):int(mask[gt_idx][3].item()),
                            int(mask[gt_idx][0].item()):int(mask[gt_idx][2].item())] = 1

                        images_batch = torch.where((image_mask == 1), images_batch, image_mask)
                        # print("----------crop the bbox of the car----------")

                        patch_bboxes = patch_bboxes.cuda()
                        adv_patch = adv_patch_cpu.cuda()

                        out_patch = patch_transform(adv_patch, patch_bboxes, size)

                        p_img_batch = torch.where((out_patch == 0), images_batch, out_patch)  # (1, 3, 1920, 1080)
                        # print("---------get adv examples-----------")

                        # p_img_batch, scale, padding_x, padding_y = padding_resize(p_img_batch, 560, 315)  # width, height

                        # save image
                        img = p_img_batch[0, :, :, ]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        img.save(
                            f"/data/yipengao/code/faster-rcnn.pytorch-master/adv_examples/cls_iou/adv_examples_{epoch}.png")
                        # print("p_img_batch size is ", p_img_batch.size())

                        # transform to faster rcnn input
                        im = p_img_batch.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                        im *= 255.0
                        im = im.astype(np.uint8)
                        im = im[:, :, ::-1]  # rgb-bgr
                        # print("----------get im-----------")                     

                        blobs, im_scales = _get_image_blob(im)
                        # print("blobs is ", blobs)
                        # print("im_scales is ", im_scales)                     

                        assert len(im_scales) == 1, "Only single-image batch implemented"
                        im_blob = blobs

                        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
                        im_data_pt = torch.from_numpy(im_blob)
                        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                        im_info_pt = torch.from_numpy(im_info_np)

                        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
                        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
                        gt_boxes.data.resize_(1, 1, 5).zero_()
                        num_boxes.data.resize_(1).zero_()
                       
                        # im_data = im_data.cuda()
                        # im_info = im_info.cuda()
                        # gt_boxes = gt_boxes.cuda()
                        # num_boxes = num_boxes.cuda()
                        # print("im_data is ", im_data)
                        # print("im_info is ", im_info)
                        # print("gt_boxes is ", gt_boxes)
                        # print("num_boxes is ", num_boxes)


                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                        # print("rois is ", rois)
                        # print("cls_prob ", cls_prob)
                        # print("bbox_pred ", bbox_pred)
                        # scores = cls_prob.data
                        # boxes = rois.data[:, :, 1:5]
                        scores = cls_prob
                        boxes = rois[:, :, 1:5]
                        # print("scores is ", scores)
                        # print("boxes is ", boxes)
                      
                        # bounding-box regression deltas
                        if cfg.TEST.BBOX_REG:
                            # box_deltas = bbox_pred.data
                            box_deltas = bbox_pred
                            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                                # Optionally normalize targets by a precomputed mean and stdev
                                if args.class_agnostic:
                                    if args.cuda > 0:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                    else:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                            cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                                    box_deltas = box_deltas.view(1, -1, 4)
                                else:
                                    if args.cuda > 0:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                    else:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                            cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                        else:
                            _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                            pred_boxes = _.cuda() if args.cuda > 0 else _

                        pred_boxes /= im_scales[0]
                        scores = scores.squeeze()  # [300, 21]
                        pred_boxes = pred_boxes.squeeze()  # [300, 84]
                        print("pred_boxes is ", pred_boxes)
                        print("scores is ", scores)

                        # train the patch
                        predict_loss = 0
                        bbox_loss = 0
                        # try:
                        max_probs, max_bboxes = self.prob_extractor(scores=scores, pred_boxes=pred_boxes,
                                                                        cls_id=7)  # cls_id = 7

                        max_bboxes = max_bboxes.cuda()
                        max_probs = max_probs.cuda()
                        # print(type(max_bboxes))
                        print("max_probs and max_bboxes is ", max_probs, max_bboxes)
                        if max_bboxes is None:
                                length -= 1
                                continue
                        iouloss = self.bboxloss(gt=gt, bboxes=max_bboxes)
                        predict_loss = max(torch.mean(max_probs), predict_loss)
                        bbox_loss = max(torch.max(iouloss), bbox_loss)
                        # except:
                        img.save(
                                f"/data/yipengao/code/faster-rcnn.pytorch-master/adv_examples/cls_iou/adv_examples_{i_batch}.png")
                        print("Error i_batch is ", i_batch)
                        length -= 1
                            # continue

                    predict_loss = predict_loss.cuda()
                    # predict_loss = (predict_loss + 1 - target_loss).cuda()
                    bbox_loss = bbox_loss.cuda()
                    print("predict_loss and bbox_loss is ", predict_loss, bbox_loss)

                    nps = self.nps_calculator(adv_patch).cuda()
                    tv = self.total_variation(adv_patch).cuda()
                    print("nps and tv loss is ", nps, tv)

                    loss = predict_loss + bbox_loss + 0.03 * nps + torch.max(3 * tv, torch.tensor(0.1).cuda())
                    ep_loss += loss
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    adv_patch_cpu.data.clamp_(0, 1)

                    if i_batch + 1 >= length:
                        print('\n')
                    else:
                        del scores, pred_boxes, max_probs, max_bboxes, p_img_batch, loss
                        torch.cuda.empty_cache()
            print("ep_loss and length is ", ep_loss, length)
            ep_loss = ep_loss / length
            # print("ep_loss is ", ep_loss)
            # save patch
            # print("adv_patch_cpu is ", adv_patch_cpu.data)
            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # im.save(f'/data/yipengao/code/faster-rcnn.pytorch-master/adv_patch/cls_iou/adv_patch_{epoch}.png')
            adv_img = adv_patch_cpu.detach().numpy().transpose(1, 2, 0) * 255
            adv_img = adv_img.astype(np.uint8)
            # print("adv_img is ", adv_img)
            cv2.imwrite("/data/yipengao/code/faster-rcnn.pytorch-master/adv_patch/cls_iou/adv_patch_%d.png" % epoch, adv_img)
            scheduler.step(ep_loss)
            if True:
                    print('  EPOCH NR: ', epoch),
                    print('EPOCH LOSS: ', ep_loss)
                    torch.cuda.empty_cache()


    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.
        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch
        :return: adv_patch
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_height, self.config.patch_width), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_height, self.config.patch_width))
        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch.
        :param path: Path to the image to be read.
        :return: Returns he transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def prob_extractor(self, scores, pred_boxes, cls_id, thresh=0.05):
        max_probs = torch.Tensor([0.])
        max_prob_idx = torch.Tensor(0)
        bboxes = torch.Tensor([[0., 0., 0., 0.]])
        # print("scores is ", scores)
        inds = torch.nonzero(scores[:, cls_id]>thresh).view(-1) # thresh = 0.05
        # print("inds is ", inds)
        if inds.numel() > 0:
            cls_scores = scores[:,cls_id][inds] # [36]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, cls_id*4:(cls_id+1)*4]  # [36, 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)   # [36, 5] examples：tensor([ 176.8578,  275.3472,  236.0581,  366.3774,    0.9357])
            cls_dets = cls_dets[order]

            # model.nms.nms_wrapper
            keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            max_probs = cls_dets[:, -1]
            bboxes = cls_dets[:, :4]
            # print("max_probs and bboxes is ", max_probs, bboxes)
            return max_probs, bboxes

        return max_probs, bboxes

    def bboxloss(self, gt, bboxes):
        bboxes1 = bboxes
        bboxes2 = gt
        
        # print("bboxes2 is ", bboxes2)
        # print("bboxes2 size is ", bboxes2.shape)
        # print("bboxes2[:, 0] is ", bboxes2[:, 0])

        # calculating the IOU
        mx = torch.min(bboxes1[:, 0], bboxes2[:, 0])
        Mx = torch.max(bboxes1[:, 2], bboxes2[:, 2])
        my = torch.min(bboxes1[:, 1], bboxes2[:, 1])
        My = torch.max(bboxes1[:, 3], bboxes2[:, 3])

        w1 = bboxes1[:, 2] - bboxes1[:, 0]
        h1 = bboxes1[:, 3] - bboxes1[:, 1]
        w2 = bboxes2[:, 2] - bboxes2[:, 0]
        h2 = bboxes2[:, 3] - bboxes2[:, 1]

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

    def load_model(self):
        
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

        return (im_data, im_info, num_boxes, gt_boxes)


def main():

    # if len(sys.argv) != 2:
    #     print('You need to supply (only) a confohiration mode.')
    #     print('Possible modes are: ')
    #     print(patch_config.patch_configs)

    trainer = PatchTrainer('base') # base
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    main()
