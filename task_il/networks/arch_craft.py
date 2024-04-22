import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
        
class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class BasicBlockRes(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockRes, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if in_planes != planes * self.expansion or stride !=1:
            self.shortcut = DownsampleB(in_planes, planes * self.expansion, stride)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, taskcla, depth, width, pool_code, double_code):
        super().__init__()
        self.taskcla = taskcla
        self.layers = nn.ModuleList()
        self.pool_code = pool_code
        self.first_conv = BasicBlock(3, width, 1)
        
        s=32
        for p in pool_code:
            if p < depth:
                s=s//2
        for i in range(depth):
            last_width = width
            while i in double_code:
                double_code.remove(i)
                width *= 2
            
            self.layers.append(BasicBlockRes(last_width, width, 1))
        
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(width*s*s, n))
        

    def forward(self, x):
        h=self.first_conv(x)
        p_code = copy.deepcopy(self.pool_code)
        for i, layer in enumerate(self.layers):
            while i in p_code:
                p_code.remove(i)
                h = F.max_pool2d(h, 2)
            h = layer(h)
        out = h
        out = out.view(out.size(0), -1)
        
        y = []
        for t, n in self.taskcla:
            y.append(self.last[t](out))
        return y

