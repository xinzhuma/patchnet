'''
loss function for patchnet
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.kitti.kitti_utils import boxes3d_to_corners3d_torch



def get_loss(center_label,
             heading_class_label,
             heading_residual_label,
             size_class_label,
             size_residual_label,
             num_heading_bin,
             num_size_cluster,
             mean_size_arr,
             output_dict,
             corner_loss_weight=10.0,
             box_loss_weight=1.0):

    center_loss = F.l1_loss(output_dict['center'], center_label)
    stage1_center_loss = F.l1_loss(output_dict['stage1_center'], center_label)

    # Heading loss
    heading_class_label = heading_class_label.long()  # label of cross entroy loss shuold be long datatype
    heading_class_loss = F.cross_entropy(output_dict['heading_scores'], heading_class_label)

    hcls_onehot = torch.zeros(heading_class_label.shape[0], num_heading_bin).cuda().scatter_(
                              dim=1, index=heading_class_label.view(-1, 1), value=1)
    heading_residual_label = heading_residual_label.float()
    heading_residual_normalized_label = heading_residual_label / (np.pi/ num_heading_bin)

    heading_residual_normalized = torch.sum(output_dict['heading_residuals_normalized']*hcls_onehot, 1)
    heading_residual_normalized_loss = F.l1_loss(heading_residual_normalized, heading_residual_normalized_label)

    # Size loss
    size_class_loss = F.cross_entropy(output_dict['size_scores'], size_class_label.long())
    scls_onehot = torch.zeros(size_class_label.shape[0], num_size_cluster).cuda().scatter_(
                              dim=1, index=size_class_label.long().view(-1, 1), value=1)
    scls_onehot = scls_onehot.view(size_class_label.shape[0],  num_size_cluster, 1).repeat(1, 1, 3)
    size_residual_normalized = torch.sum(output_dict['size_residuals_normalized'] * scls_onehot, 1)

    mean_size_label = torch.sum(torch.from_numpy(mean_size_arr).cuda() * scls_onehot, 1)
    size_residual_label = size_residual_label.float()
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_residual_normalized_loss = F.l1_loss(size_residual_label_normalized, size_residual_normalized)


    # Corner loss
    size_pred = output_dict['size_residuals'] + torch.from_numpy(mean_size_arr).cuda().view(1, -1, 3)
    size_pred = torch.sum(size_pred * scls_onehot, 1)
    # true pred heading
    heading_bin_centers = torch.from_numpy(np.arange(0, 2 * np.pi, 2 * np.pi / num_heading_bin)).cuda().float()
    heading_pred = output_dict['heading_residuals'] + heading_bin_centers.view(1, -1)
    heading_pred = torch.sum(heading_pred * hcls_onehot, 1)

    box3d_pred = torch.cat([output_dict['center'], size_pred, heading_pred.view(-1, 1)], 1)
    corners_3d_pred = boxes3d_to_corners3d_torch(box3d_pred)

    # heading true label
    heading_bin_centers = torch.from_numpy(np.arange(0,2*np.pi,2*np.pi/num_heading_bin)).cuda().float()
    heading_label = heading_residual_label.view(-1, 1) + heading_bin_centers.view(1, -1)
    heading_label = torch.sum(hcls_onehot*heading_label, -1).float()

    # size true label
    size_label = torch.sum(torch.from_numpy(mean_size_arr).cuda() * scls_onehot, 1) + size_residual_label
    size_label = size_label.float()

    # corners_3d label
    box3d = torch.cat([center_label, size_label, heading_label.view(-1, 1)], 1)

    # true 3d corners
    corners_3d_gt = boxes3d_to_corners3d_torch(box3d)
    corners_3d_gt_flip = boxes3d_to_corners3d_torch(box3d, flip=True)
    corners_loss = torch.min(F.l1_loss(corners_3d_pred, corners_3d_gt),
                             F.l1_loss(corners_3d_pred, corners_3d_gt_flip))

    # # Weighted sum of all losses
    total_loss =  box_loss_weight * (center_loss + \
        heading_class_loss + size_class_loss + \
        heading_residual_normalized_loss*20 + \
        size_residual_normalized_loss*20 + \
        stage1_center_loss + \
        corner_loss_weight*corners_loss)

    return total_loss



if __name__ == '__main__':
    pass
