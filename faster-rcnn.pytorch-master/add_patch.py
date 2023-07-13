import cv2
import os
from image_utils import patch_transform
import torch
import scipy
from scipy.misc import imread
from load_data import *

save_path = "./images_car"
img_path = '/data/yipengao/dataset/car_data/training_data/images'
label_path = '/data/yipengao/dataset/car_data/training_data/labels'

# images = os.listdir(img_path)[:10]
# images = ['indoor_2880.jpg']
# for img in images:
#     print("img", img)
#     lab = open(os.path.join(label_path, img.replace('jpg', 'txt')), 'r').readlines()
#     lab = list(map(float, lab[0].strip().split(' ')))
#     # print("lab", lab)
#     lab = [int(x) for x in lab]
#     print("lab is ", lab)
#     img_cv = cv2.imread(os.path.join(img_path, img))
#     print("img_cv.shape is ", img_cv.shape)
#     # img_cv[lab[1]:lab[3], lab[0]:lab[2], :] = 128
#     img_cv[440:543, 727:1138, :] = 128
#     cv2.imwrite(os.path.join(save_path, img), img_cv)

def get_car_label(labdirs):
    gt = []
    adv_bboxes = []

    for i in range(len(labdirs)):
        label_dirs = os.path.join(os.path.join(label_path, 'bbox'), labdirs[i])
        bonnet_dirs = os.path.join(os.path.join(label_path, 'hood'), labdirs[i])
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

        adv_bboxes = np.array(
            [Center_X - adv_width, Center_Y - adv_height, Center_X + adv_width, Center_Y + adv_height])
        adv_bboxes = adv_bboxes.transpose((1, 0))

    gt = torch.Tensor(np.array(gt))
    adv_bboxes = torch.Tensor(adv_bboxes).cuda()

    return adv_bboxes, gt


def generate_patch(type):
    if type == 'gray':
        adv_patch_cpu = torch.full((3, 624, 1040), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, 624, 1040))
    return adv_patch_cpu


if __name__ == '__main__':
    paths = os.listdir(img_path)[:5]
    adv_patch_cpu = generate_patch("gray")

    for path in paths:

        idx = path.split('_')[1].split('.')[0]
        print("path and idx is ", path, idx)

        img = os.path.join(img_path, path)
        print("img path is ", img)
        images = Image.open(img).convert('RGB')
        images = transforms.ToTensor()(images)
        images = images.unsqueeze(0)
        images_batch = images.cuda()
        size = images.size()
        print("size is ", size)
        patch_bboxes, gt = get_car_label([path.replace('jpg', 'txt')])
        print("patch_bboxes and gt is ", patch_bboxes, gt)
        
        save1 = images_batch[0, :, :, ]
        save1 = transforms.ToPILImage()(save1.detach().cpu())
        save1.save(f"/data/yipengao/code/faster-rcnn.pytorch-master/images_car/images_batch_{idx}.png")
        print("images_batch size is ", images_batch.size())

        gt = gt.cuda()

        # crop the bbox of the car
        mask_w = (gt[:, 2] - gt[:, 0]).cpu().numpy()
        mask_h = (gt[:, 3] - gt[:, 1]).cpu().numpy()
        mask = [gt[:, 0].cpu().numpy() - 1 / 10. * mask_w, gt[:, 1].cpu().numpy() - 1 / 10. * mask_h,
                gt[:, 2].cpu().numpy() + 1 / 10. * mask_w, gt[:, 3].cpu().numpy() + 1 / 10. * mask_h]

        mask = torch.Tensor(np.array(mask).transpose((1, 0)))
        image_mask = torch.ones_like(images).cuda() * 0.5
        print("image_mask size is ", image_mask.size())
        for gt_idx in range(len(mask)):
            image_mask[:, :, int(mask[gt_idx][1].item()):int(mask[gt_idx][3].item()),
            int(mask[gt_idx][0].item()):int(mask[gt_idx][2].item())] = 1

        images_batch = torch.where((image_mask == 1), images_batch, image_mask)
        # save1 = images_batch[0, :, :, ]
        # save1 = transforms.ToPILImage()(save1.detach().cpu())
        # save1.save(f"/data/yipengao/code/faster-rcnn.pytorch-master/images_car/images_batch_{idx}.png")
        # print("images_batch size is ", images_batch.size())

        patch_bboxes = patch_bboxes.cuda()
        adv_patch = adv_patch_cpu.cuda()

        out_patch = patch_transform(adv_patch, patch_bboxes, size)
        # save2 = out_patch[0, :, :, ]
        # save2 = transforms.ToPILImage()(save2.detach().cpu())
        # save2.save(f"/data/yipengao/code/faster-rcnn.pytorch-master/adv_examples/cls_iou/out_patch_{idx}.png")
        # print("out_patch size is ", out_patch.size())

        p_img_batch = torch.where((out_patch == 0), images_batch, out_patch)  # (1, 3, 1920, 1080)
        # save3 = p_img_batch[0, :, :, ]
        # save3 = transforms.ToPILImage()(save3.detach().cpu())
        # save3.save(f"/data/yipengao/code/faster-rcnn.pytorch-master/images_car/adv_examples_{idx}.png")
        # print("p_img_batch size is ", p_img_batch.size())

        im = p_img_batch.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        scipy.misc.imsave(f"/data/yipengao/code/faster-rcnn.pytorch-master/images_car/adv_examples_{idx}.png",im)
        print("im is ", im)
        # im = im[:, :, ::-1]


