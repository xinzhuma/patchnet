import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.backbones.plainnet import PlainNet
from lib.backbones.senet import senet18_patchnet as senet
from lib.backbones.resnet import resnet18_patchnet as resnet
from lib.backbones.resnext import resnext_patchnet_1 as resnext
from lib.helpers.fpointnet_helper import parse_outputs
from lib.extensions.mask_global_pooling import mask_global_max_pooling_2d
from lib.extensions.mask_global_pooling import mask_global_avg_pooling_2d
from lib.helpers.misc_helper import init_weights


class PatchNet(nn.Module):
    def __init__(self, cfg, num_heading_bin, num_size_cluster, mean_size_arr):
        super().__init__()
        self.cfg = cfg
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr

        # center estimation module
        self.center_reg_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256], kernal_size=1)
        self.center_reg_head = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                             nn.Linear(128, 3))
        # box estiamtion module
        assert cfg['backbone'] in ['plainnet', 'resnet', 'resnext', 'senet']
        if cfg['backbone'] == 'plainnet':
            self.box_est_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256, 512], kernal_size=3, padding=1)
        if cfg['backbone'] == 'resnet':
            self.box_est_backbone = resnet()
        if cfg['backbone'] == 'senet':
            self.box_est_backbone = senet()
        if cfg['backbone'] == 'resnext':
            self.box_est_backbone = resnext()

        self.box_est_head1 = nn.Sequential(nn.Linear(515, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        self.box_est_head2 = nn.Sequential(nn.Linear(515, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        self.box_est_head3 = nn.Sequential(nn.Linear(515, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))


        init_weights(self, self.cfg['init'])


    def forward(self, patch, one_hot_vec):
        output_dict = {}

        # get [h, w] of input patch  [for global pooling]
        _, _, h, w = patch.shape

        # mask generation
        depth_map = patch[:, 2, :, :]
        threshold = depth_map.mean(-1).mean(-1) + self.cfg['threshold_offset']
        threshold = threshold.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
        zeros, ones = torch.zeros_like(depth_map), torch.ones_like(depth_map)
        mask = torch.where(depth_map < threshold, ones, zeros)
        mask_xyz_mean = mask_global_avg_pooling_2d(patch, mask)
        patch = patch - mask_xyz_mean
        mask_xyz_mean = mask_xyz_mean.squeeze(-1).squeeze(-1)

        # center regressor
        center_reg_features = mask_global_max_pooling_2d(self.center_reg_backbone(patch), mask)
        center_reg_features = torch.cat([center_reg_features.view(-1, 256), one_hot_vec], -1)  # add one hot vec
        center_tnet = self.center_reg_head(center_reg_features)
        stage1_center = center_tnet + mask_xyz_mean  # Bx3
        output_dict['stage1_center'] = stage1_center

        # get patch in object coordinate
        patch = patch - center_tnet.unsqueeze(-1).unsqueeze(-1)

        # 3d box regressor
        box_est_features = self.box_est_backbone(patch)
        box_est_features = mask_global_max_pooling_2d(box_est_features, mask)  # global max pooling
        box_est_features = torch.cat([box_est_features.view(-1, 512), one_hot_vec], -1)   # add one hot vec
        box1 = self.box_est_head1(box_est_features)
        box2 = self.box_est_head2(box_est_features)
        box3 = self.box_est_head3(box_est_features)
        box  = result_selection_by_distance(stage1_center, box1, box2, box3)

        output_dict = parse_outputs(box, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, output_dict)
        output_dict['center'] = output_dict['center_boxnet'] + stage1_center  # Bx3
        return output_dict


def result_selection_by_distance(center, box1, box2, box3):
    disntance = torch.zeros(center.shape[0], 1).cuda()
    disntance[:, 0] = center[:, 2] # select batch dim, make mask shape (B, 1)
    box = box1
    box = torch.where(disntance < 30, box, box2)
    box = torch.where(disntance < 50, box, box3)
    return box


if __name__ == '__main__':
    import yaml
    from lib.helpers.kitti_helper import Kitti_Config
    dataset_config = Kitti_Config()
    cfg = {'name': 'patchnet', 'init': 'xavier', 'threshold_offset': 0.5,
           'patch_size': [32, 32], 'num_heading_bin': 12, 'num_size_cluster': 8,
           'backbone': 'plainnet'}

    input = torch.rand(2, 3, 64, 64)
    one_hot = torch.Tensor(2, 3)

    model = PatchNet(cfg,
                     dataset_config.num_heading_bin,
                     dataset_config.num_size_cluster,
                     dataset_config.mean_size_arr)
    output_dict = model(input, one_hot)
    print (output_dict.keys())

