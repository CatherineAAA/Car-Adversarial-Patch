# encoding: utf-8
from torch import optim

class BaseConfig(object):
    """
    Default parameters for all config files
    """

    def __init__(self):
        """
        Set the defaults
        """

        self.img_dir = "/data/yipengao/dataset/car_data/training_data/images/"
        self.label_dir = "/data/yipengao/dataset/car_data/training_data/labels/bbox/"
        self.bonnet_dir = "/data/yipengao/dataset/car_data/training_data/labels/hood/"
        self.patch_height = 624
        self.patch_width = 1040

        self.cfgfile = "cfgs/res101.yml"
        self.weightfile = "data/pretrained_model/faster_rcnn_1_7_10021.pth"
        self.printfile = "non_printability/30values.txt"

        self.start_learning_rate = 0.03
        self.patch_name = 'base'

        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.min_contrast = 0.8
        self.max_contrast = 1.2

        # 降低网络学习率，mode='min'表示当监控量停止下降的时候，学习率减小，'max'表示监控量停止上升；
        # patience，容忍网络的性能不提升的次数，高于这个次数就降低学习率
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)

        self.max_tv = 0
        self.batch_size = 1

        self.loss_target = lambda obj, cls: obj * cls


patch_configs = {
    "base": BaseConfig
}
