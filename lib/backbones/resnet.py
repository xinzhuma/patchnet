import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet18_patchnet']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
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

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion)

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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, strides): # strides for the first layer of four layer blocks
        super().__init__()
        self.in_channels = 64

        # temporally for small patches
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self.make_layer(block,  64, num_blocks[0], strides[0])
        self.layer2 = self.make_layer(block, 128, num_blocks[1], strides[1])
        self.layer3 = self.make_layer(block, 256, num_blocks[2], strides[2])
        self.layer4 = self.make_layer(block, 512, num_blocks[3], strides[3])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # strides for the layers in each layer blocks
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)



def resnet18():
    return ResNet(BasicBlock, [2,2,2,2], [1,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3], [1,2,2,2])

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3], [1,2,2,2])

def resnet101():
    return ResNet(Bottleneck, [3,4,23,3], [1,2,2,2])

def resnet152():
    return ResNet(Bottleneck, [3,8,36,3], [1,2,2,2])

def resnet18_patchnet():
    return ResNet(BasicBlock, [2,2,2,2], [1,1,1,1])

# def resnet18_(pretrained=False, **kwargs):
#     model = ResNet(BasicBlock, [2, 2, 2, 2], [1,1,1,1])
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
#     return model


if __name__ == '__main__':
    import torch
    net = resnet18_patchnet(pretrained=True)
    #net = ResNet(BasicBlock, [2, 2, 2, 2], [1, 1, 1, 1])
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())