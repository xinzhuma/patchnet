import torch
import torch.nn.functional as F

def mask_global_max_pooling_2d(features, mask):
    assert features.shape[2:4] == mask.shape[1:3], 'shape mismatch between features and mask'
    b, c, h, w = features.shape
    mask = mask.unsqueeze(1).repeat(1, c, 1, 1)   # B*H*W -> B*1*H*W -> B*C*H*W
    block_value = features.min()  # min values of whole tensor [B*C*H*W]
    features = torch.where(mask < 1, block_value, features)
    return F.max_pool2d(features, [h, w])



def mask_global_avg_pooling_2d(features, mask):
    '''
    features: B * C * H * W
    mask: B * H * W
    '''
    assert features.shape[2:4] == mask.shape[1:3], 'shape mismatch between features and mask'
    b, c, h, w = features.shape
    mask = mask.unsqueeze(1).repeat(1, c, 1, 1)  # B*H*W -> B*1*H*W
    features = features * mask
    pooled_features =  F.avg_pool2d(features, [h, w])
    scale = (h * w) / mask.sum(-1).sum(-1).clamp(min=1.0)
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    return pooled_features * scale



if __name__ == '__main__':

    b, c, h, w = 1, 3, 4, 4
    features = torch.rand(b, c, h, w)
    mask = torch.zeros(b, h, w)
    mask[:, 1:3, 1:3] = 1
    print ('features', features)
    # print ('mask', mask)

    a, b = F.max_pool2d_with_indices(features, [h, w])
    print(a)
    print(b)



    # max = mask_global_max_pooling_2d(features, mask)
    # avg = mask_global_avg_pooling_2d(features, mask)
    # print(max)
    # print(avg)