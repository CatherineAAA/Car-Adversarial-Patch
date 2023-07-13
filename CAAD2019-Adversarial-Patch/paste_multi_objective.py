import cv2
import fnmatch
from torch.nn import functional as F
import os
import torch
from PIL import Image, ImageDraw
import numpy as np
import random
from torchvision import transforms
from median_pool import MedianPool2d
from image_utils import padding_resize
import math

patch_path = './adv_patch_0527/epoch_360.png'
img_path = './datasets/openimage/testing_data/images'
label_path = './datasets/openimage/testing_data/labels'
imgs = fnmatch.filter(os.listdir(img_path), '*.jpg')
labels = fnmatch.filter(os.listdir(label_path), '*.txt')

def get_label(path):
    """
    get the ground truth
    """
    gt = []
    gt_label = []
    lines = open(path, 'r').readlines()
    for line in lines[:1]:
        gt.append([(float(x)) for x in line.strip().split(' ')[1:]])
        gt_label.append(line.strip().split(' ')[0])

    gt = np.array(gt)
    width = gt[0][2] - gt[0][0]
    height = gt[0][3] - gt[0][1]
    rd = height / width
    if width < height:
        rd = width / height
    adv_width = width * rd * 0.9 / 2.
    adv_height = adv_width * 624 / 1040.

    wid_range = int(gt[0][2]- adv_width) - int(gt[0][0] + adv_width)
    center_x = random.randint(int(gt[0][0] + adv_width + wid_range * 0.2), int(gt[0][2]- adv_width - wid_range * 0.2))
    height_range = int(gt[0][3] - adv_height) - int(gt[0][1] + adv_height)
    center_y = random.randint(int(gt[0][1] + adv_height + height_range * 0.2), int(gt[0][3] - adv_height - height_range * 0.2))

    adv_bboxes = np.array(
        [[center_x - adv_width, center_y - adv_height, center_x + adv_width, center_y + adv_height]])

    gt = torch.Tensor(gt)
    adv_bboxes = torch.Tensor(adv_bboxes).cuda()
    return adv_bboxes, gt, gt_label[0]

def patch_transform(path, patch_bboxes, size):
    """
    Transform the adversarial patch, including resize, rotation, blur etc.
    :path: the adversarial patch
    :patch_bboxes: the position to paste the patch
    :size: the size of the original input image. (batch, channel, height, width)
    :return: transformed patch
    """

    batch_size = 1
    adv_patch = MedianPool2d(7, same=True)(path.unsqueeze(0)).squeeze()

    brightness = torch.cuda.FloatTensor(batch_size).uniform_(-0.1, 0.1).cuda()

    # *****************random contrast******************
    contrast = torch.cuda.FloatTensor(batch_size).uniform_(0.8, 1.2).cuda()

    adv_patch = adv_patch * contrast + brightness
    width = ((patch_bboxes[:, 2] - patch_bboxes[:, 0]) / 2.0).cpu().numpy()
    height = ((patch_bboxes[:, 3] - patch_bboxes[:, 1]) / 2.0).cpu().numpy()
    bboxes = np.array([np.array(1040 / 2.0 - width), np.array(624 / 2.0 - height),
                       np.array(1040 / 2.0 + width), np.array(624 / 2.0 + height)]).astype(
        np.int32)
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
    sx = 1040 / (patch_bboxes[:, 2] - patch_bboxes[:, 0])
    sy = 624 / (patch_bboxes[:, 3] - patch_bboxes[:, 1])
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

    # scale the patch
    if sx >= 1:
        # zoom out
        grid = F.affine_grid(theta.unsqueeze(0), adv_patch.unsqueeze(0).size()).cuda()
        applier = F.grid_sample(adv_patch.unsqueeze(0), grid)
        patch_x0 = x0 + delta_px
        patch_x1 = x0 + delta_px + box_wid
        patch_y0 = y0 + delta_py
        patch_y1 = y0 + delta_py + box_height
        patch_real[0, :, patch_y0:patch_y1, patch_x0:patch_x1] = applier[0, :, int(bboxes[0][1]):int(bboxes[0][3]),
                                                                 int(bboxes[0][0]):int(bboxes[0][2])]

    else:
        # zoom in
        padx0 = int(((patch_bboxes[:, 2] - patch_bboxes[:, 0]) - 1040) / 2.)
        padx1 = int(patch_bboxes[:, 2] - patch_bboxes[:, 0] - 1040 - padx0)
        pady0 = int(((patch_bboxes[:, 3] - patch_bboxes[:, 1]) - 624) / 2.)
        pady1 = int(patch_bboxes[:, 3] - patch_bboxes[:, 1] - 624 - pady0)
        adv_patch_expand = torch.nn.ConstantPad2d((padx0, padx1, pady0, pady1), 0)(adv_patch).cuda()
        grid = F.affine_grid(theta.unsqueeze(0), adv_patch_expand.unsqueeze(0).size()).cuda()
        applier = F.grid_sample(adv_patch_expand.unsqueeze(0), grid)
        patch_x0 = x0 + delta_px
        patch_x1 = x1 + delta_px
        patch_y0 = y0 + delta_py
        patch_y1 = y1 + delta_py
        patch_real[0, :, patch_y0:patch_y1, patch_x0:patch_x1] = applier[0, :, 0:(y1 - y0), 0:(x1 - x0)]

    # rotation the patch_real
    angle = math.pi / 180 * random.randint(-10, 10)
    theta1 = torch.tensor([[math.cos(angle), math.sin(-angle), 0], [math.sin(angle), math.cos(angle), 0]],
                          dtype=torch.float)

    grid1 = F.affine_grid(theta1.unsqueeze(0), patch_real.size()).cuda()
    out_patch = F.grid_sample(patch_real, grid1).cuda()

    out_patch = out_patch[:, :, delta_py:delta_py + size[2], delta_px:delta_px + size[3]]
    return out_patch

def generate_patch(type):
        """
        Generate a random patch for starting optimization
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, 624, 1040), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, 624, 1040))
        return adv_patch_cpu

def read_image(path):
        patch_img = Image.open(path).convert('RGB')
        patch_tf = transforms.ToTensor()
        adv_patch_cpu = patch_tf(patch_img)
        return adv_patch_cpu
if __name__ == '__main__':
   
    # adv_patch_cpu = generate_patch('gray')
    adv_patch_cpu = read_image(patch_path)

    for i in range(len(imgs)):
        img = imgs[i]
        print("-------------img name %s--------------" % (img))
        img_dir = os.path.join(img_path, img)
        label_dir = os.path.join(label_path, img.replace('.jpg', '.txt'))
        image = Image.open(img_dir).convert("RGB")

        patch_bboxes, gt, class_name = get_label(label_dir)
        print("patch_bboxes", patch_bboxes)
        print("gt", gt)
        print("class_name", class_name)

        # image.show()
        # image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # draw the rectangle
        # draw_gt = cv2.rectangle(image_cv, (int(gt[0][0]), int(gt[0][1])), (int(gt[0][2]), int(gt[0][3])), (0, 255, 0), 5)
        # cv2.imwrite(os.path.join('./training_data/image_patch', img), draw_gt)

        # draw_patch = cv2.rectangle(image_cv, (int(patch_bboxes[0][0]), int(patch_bboxes[0][1])), (int(patch_bboxes[0][2]), int(patch_bboxes[0][3])), (255, 0, 0), 5)
        # cv2.imwrite(os.path.join('./training_data/image_patch', img), draw_patch)

        gt = gt.cuda()
        images = transforms.ToTensor()(image)
        images = torch.unsqueeze(images, 0)
        size = images.size()
        images_batch = images.cuda()
        # crop the bbox of the car
        # mask_w = (gt[:, 2] - gt[:, 0]).cpu().numpy()
        # mask_h = (gt[:, 3] - gt[:, 1]).cpu().numpy()
        # mask = [max(np.array([0]).astype(np.float32), gt[:, 0].cpu().numpy() - 1/10. * mask_w), max(np.array([0]).astype(np.float32), gt[:, 1].cpu().numpy() - 1/10. * mask_h),
        #         min(np.array([size[3]-1]).astype(np.float32), gt[:, 2].cpu().numpy() + 1/10. * mask_w), min(np.array([size[2]-1]).astype(np.float32), gt[:, 3].cpu().numpy() + 1/10. * mask_h)]
    #
        # mask = torch.Tensor(np.array(mask).transpose(1, 0))
        # image_mask = torch.ones_like(images).cuda() * 0.5
        # for gt_idx in range(len(mask)):
        #     image_mask[:, :, int(mask[gt_idx][1].item()):int(mask[gt_idx][3].item()),
        #     int(mask[gt_idx][0].item()):int(mask[gt_idx][2].item())] = 1
        # images_batch = torch.where((image_mask == 1), images_batch, image_mask)

        # save_1 = images_batch[0, :, :]
        # save_1 = transforms.ToPILImage()(save_1.detach().cpu())
        # save_1.save(f'./training_data/image_patch/images_batch_{img}')

        patch_bboxes = patch_bboxes.cuda()
        adv_patch = adv_patch_cpu.cuda()
        out_patch = patch_transform(adv_patch, patch_bboxes, size)

        # save_2 = out_patch[0, :, :]
        # save_2 = transforms.ToPILImage()(save_2.detach().cpu())
        # save_2.save(f'./training_data/image_patch/out_patch_{img}')

        p_img_batch = torch.where((out_patch == 0), images_batch, out_patch)
        # p_img_batch, scale, padding_x, padding_y = padding_resize(p_img_batch, 608, 608)

        save_3 = p_img_batch[0, :, :]
        save_3 = transforms.ToPILImage()(save_3.detach().cpu())
        save_3.save(f'./add_patch/{img}')

        del out_patch
