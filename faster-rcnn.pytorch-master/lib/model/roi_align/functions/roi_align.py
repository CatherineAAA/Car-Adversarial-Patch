import torch
from torch.autograd import Function
from .._ext import roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        # print("RoIAlign forward output before : ", output)
        # print("features.is_cuda: ", features.is_cuda)  # True
        # print("self.aligned_height: ", self.aligned_height)
        # print("self.aligned_width: ", self.aligned_width)
        # print("self.spatial_scale: ", self.spatial_scale)
        # print("features: ", features)
        # print("rois: ", rois)
        
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.spatial_scale, features,
                                             rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height,
                                        self.aligned_width,
                                        self.spatial_scale, features,
                                        rois, output)
#            raise NotImplementedError
        # print("RoIAlign forward output after : ", output)

        return output

    def backward(self, grad_output):
        # print("self.feature_size is ", self.feature_size)
        # print("grad_output.is_cuda is ", grad_output.is_cuda)
        
        grad_output = grad_output.cuda()
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        # grad_input = self.rois.new(batch_size, num_channels, data_height,
          #                         data_width).zero_()
        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        # print("grad_input type is ", type(grad_input))
        # self.rois = self.rois.cuda()
        # print("grad_output is ", grad_output)
        # print("grad_input is ", grad_input)
        # print("self.rois is ", self.rois)
        roi_align.roi_align_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.spatial_scale, grad_output.cuda(),
                                          self.rois.cuda(), grad_input.cuda())
        
        # print grad_input
        
        return grad_input, None
