import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from image_utils import patch_transform


image_path = '/data5/gepei/adversarial_example/car_patch/pred_nowarp/20191218-141013_base_20.png'

patch = Image.open(image_path)
patch = transforms.ToTensor()(patch).cuda()



test_path = '/data5/gepei/adversarial_example/outdoors/original_images'
testdirs = os.listdir(test_path)
for test_img in testdirs:
    img = Image.open(os.path.join(test_path, test_img))
    img = transforms.ToTensor()(img).cuda()
    img = img.unsqueeze(0)
    size = img.size()
    test_txt = os.path.join(test_path.replace('images', 'hood/hood'), test_img.replace('.jpg', '.txt'))
    bonnet = np.loadtxt(test_txt)
    bonnet = bonnet.reshape((-1,4))
    Center_X = (bonnet[:, 2] + bonnet[:, 0]) / 2
    Center_Y = (bonnet[:, 3] + bonnet[:, 1]) / 2
    bonnet_height = bonnet[:, 3] - bonnet[:, 1]
    bonnet_width = bonnet[:, 2] - bonnet[:, 0]
    adv_height = (8 / 22.) * bonnet_height
    adv_width = (8 / 22.) * bonnet_width
    adv_bboxes = np.array([Center_X - adv_width, Center_Y - adv_height, Center_X + adv_width, Center_Y + adv_height])
    adv_bboxes = adv_bboxes.transpose((1, 0))
    adv_bboxes = torch.Tensor(adv_bboxes).cuda()
    # print(adv_bboxes.size()) 
    for i in range(adv_bboxes.size()[0]):
        adv_box = adv_bboxes[i,:].unsqueeze(0)
        out_patch = patch_transform(patch, adv_box, size)
        img = torch.where((out_patch == 0), img, out_patch)
    image = img[0,:,:,:]

    save_img = transforms.ToPILImage()(image.detach().cpu())
    
    save_img.save(os.path.join('/data5/gepei/adversarial_example/outdoors/generated_images/pred', test_img))
