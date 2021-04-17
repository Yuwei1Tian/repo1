import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class A_Net(nn.Module):
    def __init__(self):
        super(A_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.convmp1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )
        self.convmp2 = nn.Sequential(
            nn.Conv2d(32, 16, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        out = self.conv1(x)  # 32,43,43
        out = self.convmp1(out) #32,13,13
        out = self.convmp2(out) #16,3,3
        out = self.conv2(out) #3,1,1
        return out


class T_Net(nn.Module):
    def __init__(self):
        super(T_Net, self).__init__()
        self.ConvRelu1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2)
        self.ConvRelu2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU()
        )
        self.ConvRelu3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU()
        )
        ## 上
        self.Block1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.ReLU(),
            Upsample(scale_factor=2, mode='bilinear')
        )
        ## 中
        self.Block2 = nn.Sequential(
            nn.Conv2d(17, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.ReLU(),
            Upsample(scale_factor=2, mode='bilinear')
        )
        ## 下
        self.Block3 = nn.Sequential(
            nn.Conv2d(17, 16, 3, 1, 1),
            nn.ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU()
        )
        self.ConvRelu4 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.ConvRelu1(x)
        out1 = out
        out = self.maxpool(out)
        out = self.ConvRelu2(out)
        out = self.ConvRelu3(out)
        out2 = self.Block1(out)
        out5 = self.maxpool(out)
        out3 = torch.cat((out5, out2), 1)
        out3 = self.Block2(out3)
        out4 = torch.cat((out, out3), 1)
        out4 = self.Block3(out4)
        out = torch.cat((out1, out4), 1)
        out = self.ConvRelu4(out)
        return out2, out3, out    #依次为上、中、结果


class A_New(nn.Module):
    def __init__(self):
        super(A_New, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 7, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.convmp1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )
        self.convmp2 = nn.Sequential(
            nn.Conv2d(32, 16, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        out = torch.cat((x, y), 1)
        out = self.conv1(out)  # 32,43,43
        out = self.convmp1(out)  # 32,13,13
        out = self.convmp2(out)  # 16,3,3
        out = self.conv2(out)  # 3,1,1
        return out


class L_Net(nn.Module):
    def __init__(self):
        super(L_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 7, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.convmp1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )
        self.convmp2 = nn.Sequential(
            nn.Conv2d(32, 16, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        out = torch.cat((x, y), 1)
        out = self.conv1(out)  # 32,43,43
        out = self.convmp1(out)  # 32,13,13
        out = self.convmp2(out)  # 16,3,3
        out = self.conv2(out)  # 1,1,1
        return out


# net = L_Net()
# x = torch.randn(1, 3, 49, 49)
# y = torch.randn(1, 1, 49, 49)
# ret = net(x, y)
# print(ret.shape)
# net = T_Net()
# ret = net(torch.randn(1, 3, 160, 160))
# print(ret.shape)