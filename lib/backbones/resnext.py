import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNeXt', 'resnext18_2x64d', 'resnext18_32x4d', 'resnext34_2x64d','resnext34_32x4d',
           'resnext50_2x64d', 'resnext50_32x4d', 'resnext_patchnet_1', 'resnext_patchnet_2',
           'resnext_patchnet_3', 'resnext_patchnet_4']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, cardinality=32, bottleneck_width=4, stride=1):
        super().__init__()
        channels = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, cardinality=32, bottleneck_width=4, stride=1):
        super().__init__()
        channels = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, self.expansion*channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, bottleneck_width, strides):
        super().__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = 64

        # temporally for small patches
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(block, num_blocks[0], strides[0])
        self.layer2 = self.make_layer(block, num_blocks[1], strides[1])
        self.layer3 = self.make_layer(block, num_blocks[2], strides[2])
        self.layer4 = self.make_layer(block, num_blocks[3], strides[3])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, self.cardinality, self.bottleneck_width, stride))
            self.in_channels = block.expansion * self.cardinality * self.bottleneck_width
        # increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)



class ResNeXt_with_features(nn.Module):
    def __init__(self, block, num_blocks, cardinality, bottleneck_width, strides):
        super().__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = 64

        # temporally for small patches
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(block, num_blocks[0], strides[0])
        self.layer2 = self.make_layer(block, num_blocks[1], strides[1])
        self.layer3 = self.make_layer(block, num_blocks[2], strides[2])
        self.layer4 = self.make_layer(block, num_blocks[3], strides[3])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        feature1 = self.layer1(out)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        return feature2, feature3, feature4

    def make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, self.cardinality, self.bottleneck_width, stride))
            self.in_channels = block.expansion * self.cardinality * self.bottleneck_width
        # increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)


def resnext18_2x64d():
    return ResNeXt(BasicBlock, num_blocks=[2,2,2,2], cardinality=2, bottleneck_width=64, strides=[1,2,2,2])

def resnext18_32x4d():
    return ResNeXt(BasicBlock, num_blocks=[2,2,2,2], cardinality=32, bottleneck_width=4, strides=[1,2,2,2])

def resnext34_2x64d():
    return ResNeXt(BasicBlock, num_blocks=[3,4,6,3], cardinality=2, bottleneck_width=64, strides=[1,2,2,2])

def resnext34_32x4d():
    return ResNeXt(BasicBlock, num_blocks=[3,4,6,3], cardinality=32, bottleneck_width=4, strides=[1,2,2,2])

def resnext50_2x64d():
    return ResNeXt(Bottleneck, num_blocks=[3,4,6,3], cardinality=2, bottleneck_width=64, strides=[1,2,2,2])

def resnext50_32x4d():
    return ResNeXt(Bottleneck, num_blocks=[3,4,6,3], cardinality=32, bottleneck_width=4, strides=[1,2,2,2])

# resnext_10_32x2
def resnext_patchnet_0():
    return ResNeXt(BasicBlock, num_blocks=[1,1,1,1], cardinality=32, bottleneck_width=2, strides=[1,1,1,1])

# resnext_18_32x2
def resnext_patchnet_1():
    return ResNeXt(BasicBlock, num_blocks=[2,2,2,2], cardinality=32, bottleneck_width=2, strides=[1,1,1,1])

# resnext_34_32x2
def resnext_patchnet_2():
    return ResNeXt(BasicBlock, num_blocks=[3,4,6,3], cardinality=32, bottleneck_width=2, strides=[1,1,1,1])

# resnext_34_4x16
def resnext_patchnet_3():
    return ResNeXt(BasicBlock, num_blocks=[3,4,6,3], cardinality=4, bottleneck_width=16, strides=[1,1,1,1])

# resnext_50_32x2
def resnext_patchnet_4():
    return ResNeXt(Bottleneck, num_blocks=[3,4,6,3], cardinality=32, bottleneck_width=2, strides=[1,1,1,1])

# resnext_50_32x2
def resnext_patchnet_with_features():
    return ResNeXt_with_features(BasicBlock, num_blocks=[3,4,6,3], cardinality=32, bottleneck_width=2, strides=[1,1,1,1])


if __name__ == '__main__':
    import torch
    net = resnext_patchnet_with_features()
    print(net)
    x = torch.randn(1,3,32,32)
    y1, y2, y3 = net(x)
    print(y1.size(), y2.size(), y3.size())