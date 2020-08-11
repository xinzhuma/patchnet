import torch
import numpy as np



def parse_outputs(output, num_heading_bin, num_size_cluster, mean_size_arr, output_dict):
    ''' Parse batch output to separate tensors (added to end_points)
    Input:
        output: tensor in shape (B, 3+2*num_heading_bin+4*num_size_cluster)
        end_points: dict
    Output:
        output_dict: dict (updated)
    '''
    batch_size = output.shape[0]
    center = output[:, 0:3]
    output_dict['center_boxnet'] = center

    heading_scores = output[:, 3:num_heading_bin+3]
    output_dict['heading_scores'] = heading_scores  # B x num_heading_bin
    heading_residuals_normalized = output[:, 3+num_heading_bin:3+2*num_heading_bin]
    output_dict['heading_residuals_normalized'] = heading_residuals_normalized
    output_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin)

    size_scores = output[:, 3+2*num_heading_bin:\
                            3+2*num_heading_bin+num_size_cluster]
    output_dict['size_scores'] = size_scores
    size_residuals_normalized = output[:, 3+2*num_heading_bin+num_size_cluster: \
                                          3+2*num_heading_bin+4*num_size_cluster]
    size_residuals_normalized = size_residuals_normalized.view(batch_size, num_size_cluster, 3) # B x num_size_cluster x 3
    output_dict['size_residuals_normalized'] = size_residuals_normalized
    mean_size = torch.from_numpy(mean_size_arr).cuda().view(1, num_size_cluster, 3).repeat(batch_size, 1, 1)
    output_dict['size_residuals'] = size_residuals_normalized * mean_size
    return output_dict


def point_cloud_masking(point_cloud, logits, keep_points, output_dict, xyz_only=True):
    ''' Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.

    Input:
        point_cloud: tensor in shape (B,N,C)
        logits: tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: tensor in shape (B,3)
    '''
    batch_size, num_point, num_channel = point_cloud.shape
    mask = logits[:, :, 0:1] < logits[:, :, 1:2]
    mask = mask.to(torch.float32)  # bool -> float32
    output_dict['mask'] = mask
    point_cloud_xyz = point_cloud[:, :, 0:3]
    mask_xyz_mean = torch.sum(point_cloud_xyz * mask.repeat(1, 1, 3), 1).view(batch_size, 1, -1)
    mask_count = torch.sum(mask, 1).view(batch_size, 1, 1).repeat(1,1,3)
    # set minimum value in mask_count to 1
    mask_count = mask_count.clamp(min=1)
    mask_xyz_mean = mask_xyz_mean / mask_count

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean.repeat(1, num_point, 1)

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = point_cloud[:, :, 3:]
        point_cloud_stage1 = torch.cat([point_cloud_xyz_stage1, point_cloud_features], -1)

    object_point_cloud = gather_object_points(point_cloud_stage1, mask, keep_points)

    return object_point_cloud, mask_xyz_mean.squeeze(), output_dict


def gather_object_points(point_cloud, mask, npoints):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: tensor in shape (B,N,C)
        mask: tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep
    Output:
        object_pc: tensor in shape (B,npoint,C)
    '''

    def mask_to_indices(mask):
        # NOTE: this is a numpy function
        # input: mask, numpy array (B*N*1)
        # ouput: indices, numpy array (B*N), can be used as indices in tensors
        indices = np.zeros((mask.shape[0], npoints), dtype=np.int32)
        for i in range(mask.shape[0]): # batch
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices), npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices), npoints - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :] = pos_indices[choice]
        return indices

    indices = mask_to_indices(mask.cpu().numpy())

    batch_size, _, channel = point_cloud.shape
    object_point_cloud = torch.zeros((batch_size, npoints, channel)).cuda()
    for i in range(batch_size):
        object_point_cloud[i, :] = point_cloud[i][indices[i]]

    object_point_cloud = object_point_cloud
    return object_point_cloud




