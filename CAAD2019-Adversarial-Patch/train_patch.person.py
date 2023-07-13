"""
Training code for Adversarial patch training

"""

import PIL
import load_data
from tqdm import tqdm
import random
from torch.nn import functional as F
from utils import nms
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
import math
import copy
from torch.autograd import Variable
import patch_config
from image_utils import padding_resize, patch_transform, get_label
from median_pool import MedianPool2d
import sys
import time

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.patch_size = self.config.patch_size
        self.patch_width = self.config.patch_width
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        self.prob_extractor = MaxProbExtractor(0, 17, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size, self.config.patch_width).cuda()
        self.total_variation = TotalVariation().cuda()
        self.bboxloss = BBoxLoss(80, self.darknet_model.width).cuda()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 10000        

        time_str = time.strftime("%Y%m%d-%H%M%S")
        
        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("./adversarial_patch/finetune/cls_bbox_rotation_15/20191021-195928_base_139.png")

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
        
        for epoch in range(n_epochs):
            
            ep_loss = 0
            
            for i_batch, (images, datadirs) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                       total=self.epoch_length):
                
               with autograd.detect_anomaly():
                                        
                    images_batch = images.cuda()
                    size = images.size()
                   
                    # *********** patch_bboxes -> the random position of the patch on a person************    
                    patch_bboxes, gt = get_label(datadirs, size[2])                   
                    gt = gt.cuda()                    
 
                    patch_bboxes = patch_bboxes.cuda()
                    adv_patch = adv_patch_cpu.cuda()

                    out_patch = patch_transform(adv_patch, patch_bboxes, size)

                    p_img_batch = torch.where((out_patch == 0), images_batch, out_patch)              
                    p_img_batch, scale, padding_x, padding_y = padding_resize(p_img_batch, self.darknet_model.width, self.darknet_model.height)
                    
                    # **************visualization of the person with adversarial patch*************   
                    img = p_img_batch[0, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())                    
                    img.save('./adversarial_patch/finetune/geekpwn.jpg')  

                    # **************train the patch**************
                    _, output = self.darknet_model(p_img_batch)
                    predict_loss = 0
                    bbox_loss = 0 
                    target_loss = 0

                    # *********calculate the best score************
                    for iters in range(len(output)):
                        # *****************************************************
                        # max_prob --> the highest confidece of detecting car
                        # max_idx --> the index of grids corresponding ti the highest confidence 
                        # max_target --> the confidence of classifying the detected ojected as the target object
                        # ****************************************************
                        max_prob, max_idx, max_target = self.prob_extractor(output[iters])
                        # ********the iou of the detected objected and ground truth***********
                        anchors = self.darknet_model.anchors[(2 - iters) * 6:(2 - iters + 1) * 6]
                        iouloss = self.bboxloss(output[iters], max_idx, gt, anchors, size[3], size[2], scale, padding_x, padding_y)                     
                        predict_loss = max(torch.mean(max_prob), predict_loss)
                        target_loss = max(torch.mean(max_target), target_loss)
                        bbox_loss = max(torch.max(iouloss), bbox_loss)
                    
                    predict_loss = predict_loss.cuda()
                    # predict_loss = (predict_loss + 1 - target_loss).cuda()
                    bbox_loss = bbox_loss.cuda()                    

                    nps = self.nps_calculator(adv_patch).cuda()
                    tv = self.total_variation(adv_patch).cuda()

                    # loss = predict_loss + bbox_loss + 0.01 * nps + torch.max(2.5*tv, torch.tensor(0.1).cuda())
                    # loss = bbox_loss +  0.01 * nps + torch.max(2.5*tv, torch.tensor(0.1).cuda())
                    # loss = predict_loss +  0.01 * nps + torch.max(2.5*tv, torch.tensor(0.1).cuda())
                    loss = 0.5 * predict_loss + 0.5 * bbox_loss +  0.01 * nps + torch.max(2.5 * tv, torch.tensor(0.1).cuda())  

                    ep_loss += loss
 

                    loss.backward()
                   
                    optimizer.step()
                    optimizer.zero_grad()

                    adv_patch_cpu.data.clamp_(0,1)


                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del output, max_prob, p_img_batch,  loss
                        torch.cuda.empty_cache()

            ep_loss = ep_loss/len(train_loader)

            # **********save the patch*************
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)

            im.save(f'adversarial_patch/finetune/geekpwn/{time_str}_{self.config.patch_name}_{epoch}.png')
            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                
                del output, max_prob, p_img_batch, loss
                torch.cuda.empty_cache()


    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_width), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_width))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    
    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


