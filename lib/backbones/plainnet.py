'''
2D equivalent implementation of PointNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainNet(nn.Module):
    def __init__(self,
                 input_channels = 3,
                 layer_cfg = [128, 128, 256],
                 kernal_size = 1,
                 padding = 0,
                 batch_norm = True):
        super().__init__()
        self.input_channels = input_channels
        self.layer_cfg = layer_cfg
        self.kernal_size = kernal_size
        self.padding = padding
        self.batch_norm = batch_norm
        self.features = self.make_layers()


    def forward(self, patch):
        return self.features(patch)


    def make_layers(self):
        layers = []
        input_channels = self.input_channels

        for output_channels in self.layer_cfg:
            layers += [nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                 kernel_size=self.kernal_size, padding=self.padding)]
            if self.batch_norm:
                layers += [nn.BatchNorm2d(output_channels)]
            layers += [nn.ReLU(inplace=True)]
            input_channels = output_channels

        return nn.Sequential(*layers)



class PlainNet_SEG(nn.Module):
    def __init__(self, input_channels = 3):
        super().__init__()
        self.channels = input_channels
        self.conv1 = nn.Conv2d(self.channels, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.conv5 = nn.Conv2d(128, 1024, 1)
        self.conv6 = nn.Conv2d(1091, 512, 1)
        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 128, 1)
        self.conv9 = nn.Conv2d(128, 128, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)


    def forward(self, x, one_hot_vec):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        local_features = x # bchw

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        _, _, h, w = x.shape
        global_features = F.max_pool2d(x, (h, w))
        global_features = global_features.repeat(1, 1, h, w)
        one_hot_vec = one_hot_vec.view(-1, 3, 1, 1).repeat(1, 1, h, w)
        x = torch.cat([local_features, global_features, one_hot_vec], 1)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        return x


if __name__ == '__main__':
    patch = torch.Tensor(2, 3, 32, 32)
    one_hot = torch.Tensor(2, 3)
    net = PlainNet_SEG(input_channels=3)
    print (net)
    output = net(patch, one_hot)
    print(output.shape)

