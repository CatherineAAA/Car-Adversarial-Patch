# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
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
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.utils.blob import im_list_to_blob
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
import pdb

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
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="data/pretrained_model")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save images for test',
                      default="detect_res")
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
                      default=16, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None: # 配置文件
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None: # 设置配置
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)  # 设置随机数种子，每次运行代码时设置相同的seed，则每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样


  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  # input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  input_dir = args.load_dir # 模型文件位置
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # array和asarray都可以将数据结构转换为ndarray,但是主要区别在于当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会
  pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])

  # initilize the network here.
  # class_agnostic方式只回归2类bounding box,即前景和背景
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

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    # 在cpu上加载预先训练好的GPU模型，强制所有的GPU张量在CPU中的方式
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])   # 恢复模型
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  # 新建一些 一维Tensor
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

  # make variable
  # variable的volatile属性默认为False，如果某一个variable的volatile属性被设为True，
  # 那么所有依赖它的节点volatile属性都为True。
  # volatile属性为True的节点不会求导，volatile的优先级比requires_grad高。
  # volatile 用来防止优化
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()
  # 对于dropout和batch normalization的操作在训练和测试的时候是不一样的
  # pytorh 会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :  # 是否使用电脑录视频
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  else:
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))


  while (num_images >= 0):
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():  # 摄像头开启失败
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read() # ret为True或False，代表有没有读取到图片
        im_in = np.array(frame) #截取到一帧的图像，存储为numpy
      # Load the demo image
      else:
        im_file = os.path.join(args.image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
          im_in = im_in[:,:,np.newaxis]
          im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        # rgb -> bgr
        im_in = im_in[:,:,::-1]
      im = im_in

      blobs, im_scales = _get_image_blob(im)    # 图片变换，转成网络输入
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      # [128, 128, 3]
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)    # 从 numpy 变成Tensor
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)   # [1, 3, 128, 128]
      im_info_pt = torch.from_numpy(im_info_np)     # 图像信息也变为tensor

      # 将tensor的大小调整为指定的大小
      # 如果元素个数比当前的内存大，就将底层存储大小调整为与新元素数目一致的大小
      im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.data.resize_(1, 1, 5).zero_()
      num_boxes.data.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      # rois: 包含R个感兴趣的区域，每个区域为包含5个元素的元组(n, x1, y1, x2, y2), 索引和包围框
      # cls_prob: softmax 得到的概率值
      # bbox_pred: 偏移量
      # rpn_loss_cls: 分类损失，计算softmax的损失，输入labels和cls layer的18个输出(中间reshape了一下)，输出损失函数的具体值
      # rpn_loss_box: 计算的框回归损失函数具体的值

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data    # 分类概率值
      boxes = rois.data[:, :, 1:5]  # 包围框的坐标

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          # model.rpn.bbox_transform 根据anchor和偏移量计算proposals
          # 最后返回的是左上和右下顶点的坐标[x1, y1, x2, y2]
          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          # model.rpn.bbox_transform
          # 将改变坐标信息后超过图像边界的框的边框裁剪一下，使之在图像边界之内
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          # tile()函数，将原矩阵横向、纵向地复制，这里是横向
          _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
          pred_boxes = _.cuda() if args.cuda > 0 else _

      pred_boxes /= im_scales[0]

      # squeeze函数，从数组的形状中删除单维度条目，即把shape中为1的维度去掉
      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im2show = np.copy(im)

      txt_file = os.path.join(args.save_dir, imglist[num_images][:-4] + ".txt")
      print('txt_file: ', txt_file)
      fw = open(txt_file, 'a', encoding='utf-8')
      for j in range(1, len(pascal_classes)):
          # torch.nonzero
          # 返回一个包含输入input中非零元素索引的张量，输出张量中的每行包含输入中非零元素的索引
          # 若输入input有n维，则输出的索引张量output形状为z*n，这里z是输入张量input中所有非零元素的个数
          inds = torch.nonzero(scores[:,j]>thresh).view(-1) # thresh=0.05
          # if there is det
          if inds.numel() > 0:  # torch.numel()返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]


            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # model.nms.nms_wrapper
            keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]

            for i in range(np.minimum(10, cls_dets.shape[0])):
                bbox = tuple(int(np.round(x)) for x in cls_dets[i, :4])
                score = cls_dets[i, -1]
                if score > 0.5:
                    print("class, score, pred bbox: ", pascal_classes[j], score, bbox[0], bbox[1], bbox[2], bbox[3])
                    fw.writelines("%s %s %s %s %s %s\n".format(pascal_classes[j], score, bbox[0], bbox[1], bbox[2], bbox[3]))

            if vis:
              im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5, fw)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          result_path = txt_file.replace(".txt", ".jpg")
          cv2.imwrite(result_path, im2show)
      else:
          cv2.imshow("frame", im2show)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()
