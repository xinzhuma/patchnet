'''
F-PointNet, http://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Frustum_PointNets_for_CVPR_2018_paper.html
'''

import torch
import torch.nn as nn

from lib.backbones.pointnet import PointNet
from lib.backbones.pointnet import PointNet_SEG
from lib.helpers.fpointnet_helper import point_cloud_masking
from lib.helpers.fpointnet_helper import parse_outputs
from lib.helpers.misc_helper import init_weights


class FPointNet(nn.Module):
    def __init__(self, cfg, num_heading_bin, num_size_cluster, mean_size_arr):
        super().__init__()
        self.cfg = cfg
        self.mean_size_arr = mean_size_arr  # fixed anchor
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.point_seg_backbone = PointNet_SEG(num_points=self.cfg['num_frustum_point'],
                                               input_channels=cfg['input_channel'])
        self.point_seg_head = nn.Sequential(nn.Dropout(0.5),
                                            nn.Conv1d(128, 2, 1))
        self.center_reg_backbone = PointNet(num_points=self.cfg['num_object_points'],
                                            input_channels=3,
                                            layer_cfg=[128, 128, 256],
                                            batch_norm=True)
        self.center_reg_head = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                             nn.Linear(128, 3))
        self.box_est_backbone = PointNet(num_points=self.cfg['num_object_points'],
                                         input_channels=3,
                                         layer_cfg=[128, 128, 256, 512],
                                         batch_norm=True)
        self.box_est_head = nn.Sequential(nn.Linear(515, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))

        init_weights(self, self.cfg['init'])



    def forward(self, point_cloud, one_hot_vec):
        output_dict = {}

        # background points segmentation
        seg_features = self.point_seg_backbone(point_cloud, one_hot_vec)
        seg_logits = self.point_seg_head(seg_features).transpose(1, 2).contiguous()    # B*C*N -> B*N*C
        output_dict['mask_logits'] = seg_logits

        # background points masking
        # select masked points and translate to masked points' centroid
        point_cloud = point_cloud.transpose(1, 2).contiguous()  # B*C*N -> B*N*C, meet the shape requirements of funcs
        object_point_cloud_xyz, mask_xyz_mean, output_dict = \
            point_cloud_masking(point_cloud, seg_logits, self.cfg['num_object_points'], output_dict)

        # T-Net and coordinate translation
        object_point_cloud_xyz = object_point_cloud_xyz.transpose(1, 2).contiguous()  # B*N*C -> B*C*N
        center_features = self.center_reg_backbone(object_point_cloud_xyz).view(-1, 256)  #  B*C*1 -> B*C
        center_features = torch.cat([center_features, one_hot_vec], 1)
        center_tnet = self.center_reg_head(center_features)
        stage1_center = center_tnet + mask_xyz_mean  # Bx3
        output_dict['stage1_center'] = stage1_center

        # Get object point cloud in object coordinate
        object_point_cloud_xyz_new = object_point_cloud_xyz - center_tnet.view(-1, 3, 1)

        # Amodel Box Estimation PointNet
        box_features = self.box_est_backbone(object_point_cloud_xyz_new).view(-1, 512)  # B*C*1 -> B*C
        box_features = torch.cat([box_features, one_hot_vec], 1)
        box_results = self.box_est_head(box_features)
        output_dict = parse_outputs(box_results, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, output_dict)
        output_dict['center'] = output_dict['center_boxnet'] + stage1_center  # Bx3

        return output_dict




if __name__ == '__main__':
    # only for debug
    import yaml
    import numpy as np
    cfg_file = '../../experiments/fpointnet/config_fpointnet.yaml'
    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)['model']
    fpointnet = FPointNet(cfg, 12, 8, np.zeros((8, 3), dtype=np.float32))

    points = torch.Tensor(2, 4, 1024)
    one_hot_vec = torch.Tensor(2, 3)

    outputs = fpointnet(points, one_hot_vec)
    for key in outputs:
        print((key, outputs[key].shape))
    print(len(outputs.keys()))









