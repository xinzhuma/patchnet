'''
PointNet, https://arxiv.org/abs/1612.00593
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    '''
    input:
        points, (B, C, N)
    output:
        global features, (B, C', 1)
    '''
    def __init__(self,
                 num_points = 4096,
                 input_channels = 3,
                 layer_cfg = [64, 64, 64, 128, 1024],
                 batch_norm = True):
        super().__init__()
        self.num_points = num_points
        self.input_channels = input_channels
        self.layer_cfg = layer_cfg
        self.batch_norm = batch_norm
        self.features = self.make_layers()


    def forward(self, points):
        point_features = self.features(points)
        global_features = F.max_pool1d(point_features, self.num_points)
        return global_features


    def make_layers(self):
        layers = []
        input_channels = self.input_channels

        for output_channels in self.layer_cfg:
            layers += [nn.Conv1d(input_channels, output_channels, kernel_size=1)]
            if self.batch_norm:
                layers += [nn.BatchNorm1d(output_channels)]
            layers += [nn.ReLU(inplace=True)]
            input_channels = output_channels

        return nn.Sequential(*layers)



class PointNet_SEG(nn.Module):
    '''
    pointnet segmentation backbone for Frustum-PointNet
    input:
        points, (B, C, N)
    output:
        point-wise features, (B, C', N)
    '''
    def __init__(self, num_points = 4096, input_channels = 4):
        super().__init__()
        self.num_points = num_points
        self.point_channels = input_channels
        self.conv1 = nn.Conv1d(self.point_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.conv6 = nn.Conv1d(1091, 512,  1)
        self.conv7 = nn.Conv1d(512,  256,  1)
        self.conv8 = nn.Conv1d(256,  128,  1)
        self.conv9 = nn.Conv1d(128,  128,  1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)


    def forward(self, x, one_hot_vec):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        point_features = x

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_features = F.max_pool1d(x, self.num_points)
        global_features = global_features.repeat(1, 1, self.num_points)
        one_hot_vec = one_hot_vec.view(-1, 3, 1).repeat(1, 1, self.num_points)
        x = torch.cat([point_features, global_features, one_hot_vec], 1)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        return x


if __name__ == '__main__':
    net = PointNet()
    print(net)
    x = torch.randn(1, 3, 512)
    y = net(x)
    print(y.size())

