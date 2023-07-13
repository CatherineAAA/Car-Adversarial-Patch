from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """    
        
        # person
        # self.img_dir = "train_data/select/geek-image1/"
        # self.label_dir = "train_data/select/geek-txt1/"
        # self.patch_size = 210 * 2 # height
        # self.patch_width = 297 * 2
        
        # car
        self.img_dir = "/data2/gepei/package/adversarial/adversarial_car/train_data/images/train/"
        self.label_dir = "/data2/gepei/package/adversarial/adversarial_car/train_data/label/train/bbox/"
        self.bonnet_dir = "/data2/gepei/package/adversarial/adversarial_car/train_data/label/train/hood/"
        self.patch_size = 624 # height
        self.patch_width = 1040
        
        
        self.cfgfile = "cfg/yolov3.cfg"
        self.weightfile = "weights/yolov3.weights"
        self.printfile = "non_printability/30values.txt"

        self.start_learning_rate = 0.03

        self.patch_name = 'base'
        
        self.min_brightness = -0.1 
        self.max_brightness = 0.1

        self.min_contrast = 0.8
        self.max_contrast = 1.2

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0
        self.batch_size = 1 

        self.loss_target = lambda obj, cls: obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly
}
