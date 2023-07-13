"""
@author:yipengao
@description: the training code for adversarial patch of multi-objective.
"""

from tqdm import tqdm
import random
from torch import autograd
from  torchvision import transforms
import patch_config_multiobjective
from load_data import *
from image_utils import padding_resize, patch_transform, get_car_label, get_label, get_multi_label
import sys
import time
from patch_warp import PatchWarp

class_index = {}
file_open = open('./data/coco.names', 'r')
classes = file_open.readlines()
for i in range(len(classes)):
    class_index[classes[i].strip()] = i

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config_multiobjective.patch_configs[mode]()
        self.patch_size = self.config.patch_height
        self.cls_idx = 0    # add for adjust the training class
        self.patch_width = self.config.patch_width
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        self.prob_extractor = MaxProbExtractor(self.cls_idx, 17, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_height, self.config.patch_width).cuda()
        self.total_variation = TotalVariation().cuda()
        self.bboxloss = BBoxLoss(80, self.darknet_model.width).cuda()
        self.ph = 30
        self.param = 0.0001
        self.scale = 1.0
        self.x = 0.1
        self.y = -15
        self.deformation = 50
        self.wrap_patch = PatchWarp(self.ph, self.param, self.scale, self.x, self.y)

    def train(self):
        # img_size = self.darknet_model.height
        batch_size = self.config.batch_size  # 1    # up
        n_epochs = 1000

        adv_patch_cpu = self.generate_patch('gray')
        # adv_patch_cpu = self.read_image(path) # load generated patch
        adv_patch_cpu.requires_grad_(True)
        # print("self.config.img_dir", self.config.img_dir)
        train_loader = torch.utils.data.DataLoader(PersonDataset(self.config.img_dir, shuffle=True),
                                                   batch_size = batch_size,
                                                   shuffle = True,
                                                   num_workers = 10)

        self.epoch_length = len(train_loader)
        # print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)   # 0.03
        scheduler = self.config.scheduler_factory(optimizer)

        for epoch in range(n_epochs):
            ep_loss = 0
            for i_batch, (images, datadirs) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                    total=self.epoch_length):
                with autograd.detect_anomaly():
                    images_batch = images.cuda()
                    size = images.size()
                    # print("images.size() is ", size)
                    # print("--------------datadirs: %s---------------" % (datadirs))
                    # select a random position of the patch on a foreobject
                    # patch_bboxes, gt = get_car_label(datadirs)  # adv_patch bboxes
                    try:                        
                        patch_bboxes, gt, class_name = get_multi_label(datadirs, size[2])
                    # print("class_name is ", class_name)
                    # add map of class_name and class_idx
                    
                        
                        cls_idx = class_index[class_name]
                        self.cls_idx = cls_idx
                        
                    except:
                        print("\n this class does not be converted: ", class_name)
                        continue
                    # print("class_name and self.cls_idx is ", self.cls_idx)
                    #print("patch_bboxes and gt are ", patch_bboxes, gt)
                    gt = gt.cuda()

                    # crop the bbox of the car
                    mask_w = (gt[:, 2] - gt[:, 0]).cpu().numpy()
                    mask_h = (gt[:, 3] - gt[:, 1]).cpu().numpy()
                    # mask = [gt[:, 0].cpu().numpy() - 1/10. * mask_w, gt[:, 1].cpu().numpy() - 1/10. * mask_h,
                    #         gt[:, 2].cpu().numpy() + 1/10. * mask_w, gt[:, 3].cpu().numpy() + 1/10. * mask_h]
                    mask = [max(np.array([0]).astype(np.float32), gt[:, 0].cpu().numpy() - 1 / 10. * mask_w),
                            max(np.array([0]).astype(np.float32), gt[:, 1].cpu().numpy() - 1 / 10. * mask_h),
                            min(np.array([size[3] - 1]).astype(np.float32), gt[:, 2].cpu().numpy() + 1 / 10. * mask_w),
                            min(np.array([size[2] - 1]).astype(np.float32), gt[:, 3].cpu().numpy() + 1 / 10. * mask_h)]

                    mask = torch.Tensor(np.array(mask).transpose((1, 0)))
                    image_mask = torch.ones_like(images).cuda() * 0.5
                    for gt_idx in range(len(mask)):
                        image_mask[:,:,int(mask[gt_idx][1].item()):int(mask[gt_idx][3].item()),
                        int(mask[gt_idx][0].item()):int(mask[gt_idx][2].item())] = 1
                    images_batch = torch.where((image_mask == 1), images_batch, image_mask)

                    patch_bboxes = patch_bboxes.cuda()
                    adv_patch = adv_patch_cpu.cuda()

                    out_patch = patch_transform(adv_patch, patch_bboxes, size)

                    p_img_batch = torch.where((out_patch == 0), images_batch, out_patch)
                    p_img_batch, scale, padding_x, padding_y = padding_resize(p_img_batch, self.darknet_model.width, self.darknet_model.height)

                    img = p_img_batch[0, :, :]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.save(f'./adv_examples_0527/multi_objective_examples_{epoch}.png')

                    # training
                    _, output = self.darknet_model(p_img_batch)
                    print("len(output) is ", len(output))
                    predict_loss = 0
                    bbox_loss = 0
                    target_loss = 0

                    for iters in range(len(output)):
                        # max_prob, max_idx, max_target = self.prob_extractor(output[iters])  # 参数修改
                        max_prob, max_idx, max_target = self.prob_extractor(output[iters])
                                                 
                        print('output', output[iters])
                        print("max_prob, max_idx, max_target", max_prob, max_idx, max_target)
                        anchors = self.darknet_model.anchors[(2-iters) * 6:(2-iters+1)*6]
                        iouloss = self.bboxloss(output[iters], max_idx, gt, anchors, size[3], size[2], scale, padding_x, padding_y)
                        predict_loss = max(torch.mean(max_prob), predict_loss)
                        target_loss = max(torch.mean(max_target), target_loss)
                        bbox_loss = max(torch.max(iouloss), bbox_loss)

                    predict_loss = predict_loss.cuda()
                    bbox_loss = bbox_loss.cuda()
                    nps = self.nps_calculator(adv_patch).cuda()
                    tv = self.total_variation(adv_patch).cuda()

                    loss = predict_loss + bbox_loss + 0.1 * nps + torch.max(2.5 * tv, torch.tensor(0.1).cuda())

                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0.1)

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del output, max_prob, p_img_batch, loss
                        torch.cuda.empty_cache()

            ep_loss = ep_loss / len(train_loader)
            print("ep_loss is ", ep_loss)
            # print("tmp_error is ", tmp_error)
            # save patch
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            im.save(f'./adv_patch_0527/epoch_{epoch}.png')
            scheduler.step(ep_loss)
            if True:
                print('  Epoch NR: ', epoch)
                print('Epoch Loss: ', ep_loss)

                del output, max_prob, p_img_batch, loss
                torch.cuda.empty_cache()

    def generate_patch(self, type):
        """
        Generate a random patch for starting optimization
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_height, self.config.patch_width), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_height, self.config.patch_width))
        return adv_patch_cpu

    def read_image(self, path):
        patch_img = Image.open(path).convert('RGB')
        patch_tf = transforms.ToTensor()
        adv_patch_cpu = patch_tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()
