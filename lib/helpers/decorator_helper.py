'''
decorator, used to compute loss and collect statistics
'''
import torch
import numpy as np

from lib.losses.fpointnet_loss import get_loss as fpointnet_loss
from lib.losses.patchnet_loss import get_loss as patchnet_loss
from lib.utils.fpointnet_utils import compute_box3d_iou


def decorator(model, data, type):
    '''
    :param model:
    :param data:
    :return: loss, disp_dict
    '''
    if type == 'fpointnet':
        points, seg, center, angle_class, angle_residual, size_class, size_residual,\
                                                            rot_angle, one_hot_vec = data
        output_dict = model(points, one_hot_vec)
        loss = fpointnet_loss(seg, center, angle_class, angle_residual, size_class, size_residual,
                        model.num_heading_bin, model.num_size_cluster, model.mean_size_arr, output_dict)

    elif type == 'patchnet':
        patch, center, angle_class, angle_residual, size_class, size_residual, rot_angle, one_hot_vec = data
        output_dict = model(patch, one_hot_vec)
        loss = patchnet_loss(center, angle_class, angle_residual, size_class, size_residual,
                             model.num_heading_bin, model.num_size_cluster, model.mean_size_arr, output_dict)

    # collect statistics
    stat_dict = {}
    batch_size = center.shape[0]

    # segmentation accuracy
    if type == 'fpointnet':
        batch_size, num_point, _ = points.shape
        total_seen = batch_size * num_point
        preds_val = torch.argmax(output_dict['mask_logits'], 2)
        correct = torch.sum(preds_val == seg.long()).item()
        stat_dict.update({'seg_acc': correct / total_seen})


    # TODO: translate following codes from numpy to torch
    # 3d/bev avg iou
    iou2ds, iou3ds = compute_box3d_iou(output_dict['center'].cpu().detach().numpy(),
                                       output_dict['heading_scores'].cpu().detach().numpy(),
                                       output_dict['heading_residuals'].cpu().detach().numpy(),
                                       output_dict['size_scores'].cpu().detach().numpy(),
                                       output_dict['size_residuals'].cpu().detach().numpy(),
                                       center.cpu().detach().numpy(),
                                       angle_class.cpu().detach().numpy(),
                                       angle_residual.cpu().detach().numpy(),
                                       size_class.cpu().detach().numpy(),
                                       size_residual.cpu().detach().numpy())

    iou3d_correct = np.sum(iou3ds >= 0.7)
    stat_dict.update({'box3d_iou': np.sum(iou3ds) / batch_size})
    stat_dict.update({'box2d_iou': np.sum(iou2ds) / batch_size})
    stat_dict.update({'box_acc': iou3d_correct / batch_size})
    stat_dict.update({'loss': loss.item()})
    return loss, stat_dict