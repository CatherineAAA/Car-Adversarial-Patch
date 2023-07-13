from utils import *
from darknet import Darknet
import os
import time
from PIL import Image


def detect(cfgfile, weightfile, imgfiles, labfiles, patchfiles):
    m = Darknet(cfgfile)
    s1 = time.time()
    # m.print_network()
    m.load_weights(weightfile)
    # print("anchors: ", m.anchors)
    print('Loading weights from %s... Done!' % (weightfile))
    
    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

   
    imgdirs = os.listdir(imgfiles)
    idx = 0
    for imgfile in imgdirs:
        # print("imgfile", imgfile)
        s2 = time.time()
        img = Image.open(os.path.join(imgfiles, imgfile)).convert('RGB')
        # sized = img.resize((m.width, m.height))
        sized = img
        label_dir = os.path.join(labfiles, imgfile.replace('.jpg', '.txt'))
        boxes, scale, x, y, save_img = do_detect_multi(m, sized, label_dir, patchfiles, 0.5, 0.4, use_cuda)
        
        # boxes, scale, x, y = do_detect(m, sized, 0.5, 0.4, use_cuda)
        # print(boxes)

        # # write the boxes
        # width = img.width
        # height = img.height
        # max_area = 0
        # save_box = []
        # for i in range(len(boxes)):
        #     box = boxes[i]
        #     if box[6].item() == 0:
        #         x1 = (box[0].item() - box[2].item()/2.0) * width
        #         y1 = (box[1].item() - box[3].item()/2.0) * height
        #         x2 = (box[0].item() + box[2].item()/2.0) * width
        #         y2 = (box[1].item() + box[3].item()/2.0) * height
        #         area = (y2 - y1) * (x2 - x1)
        #         if area > max_area:
        #             save_box = [box]
        #             max_area = area
        # count = 0
        class_names = load_class_names(namesfile)
        print("save path is ", os.path.join('./OpenImage/labels_test', imgfile))
        plot_boxes(save_img, scale, x, y, boxes, os.path.join('./OpenImage/labels_test', imgfile), class_names)


if __name__ == '__main__':

    cfgfile = './cfg/yolov3.cfg'
    weightfile = './weights/yolov3.weights'
    imgfile = './datasets/openimage/testing_data/images'
    labfile = './datasets/openimage/testing_data/labels'
    patchfile = './adv_patch_0527/epoch_400.png'
    
    start = time.time()
    detect(cfgfile=cfgfile, weightfile=weightfile, imgfiles=imgfile, labfiles=labfile, patchfiles=patchfile)
    end = time.time()
    print("time: ", end - start)
  
