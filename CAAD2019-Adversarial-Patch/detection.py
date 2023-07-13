import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import os


def detect(cfgfile, weightfile, imgfiles):
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
        print("imgfile", imgfile)
        s2 = time.time()
        img = Image.open(os.path.join(imgfiles, imgfile)).convert('RGB')
        # sized = img.resize((m.width, m.height))
        sized = img
        boxes, scale, x, y = do_detect(m, sized, 0.5, 0.4, use_cuda)
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
        plot_boxes(img, scale, x, y, boxes, os.path.join('./OpenImage/labels_test', imgfile), class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)




if __name__ == '__main__':

    cfgfile = './cfg/yolov3.cfg'
    weightfile = './weights/yolov3.weights'
    imgfile = './add_patch'
    start = time.time()
    detect(cfgfile=cfgfile, weightfile=weightfile, imgfiles=imgfile)
    end = time.time()
    print("time: ", end - start)
    
    """
        if len(sys.argv) == 4:
        cfgfile = 'cfg/yolov3.cfg'    # sys.argv[1]
        weightfile = 'weights/yolov3.weights' # sys.argv[2]
        imgfile = 'images'    # sys.argv[3]
        # start = time.time()
        detect(cfgfile, weightfile, imgfile)
        # end = time.time()
        # print("time: ", end - start)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
    """

