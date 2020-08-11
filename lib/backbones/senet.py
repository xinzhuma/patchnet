import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SENet', 'senet18', 'senet34', 'senet18_patchnet']

class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels)
            )

        # SE layers
        self.fc1 = nn.Conv2d(channels, channels//16, kernel_size=1) # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(channels//16, channels, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.shape[2:4])   # out.shape : [b, c, h, w]
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, strides):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block,   64, num_blocks[0], strides[0])
        self.layer2 = self.make_layer(block,  128, num_blocks[1], strides[1])
        self.layer3 = self.make_layer(block,  256, num_blocks[2], strides[2])
        self.layer4 = self.make_layer(block,  512, num_blocks[3], strides[3])

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out



def senet18():
    return SENet(BasicBlock, [2,2,2,2], [1,2,2,2])

def senet34():
    return SENet(BasicBlock, [3,4,6,3], [1,2,2,2])

def senet18_patchnet():
    return SENet(BasicBlock, [2,2,2,2], [1,1,1,1])


if __name__ == '__main__':
    net = senet18()
    #print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())
    print(net)
