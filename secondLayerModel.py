from __future__ import print_function
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
show=ToPILImage()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(64*5*5,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096,10)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,64*5*5)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x
    
model = AlexNet()
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch.pth')

variables = checkpoint['net']
tmp = variables.copy()
tmp['conv5.weight'] = tmp['conv5.weight'][192:]
tmp['conv5.bias'] = tmp['conv5.bias'][192:]
tmp['dense1.weight'] = tmp['dense1.weight'][:, 192 * 25:]
model.load_state_dict(tmp)
state = {'net':model.state_dict()}
torch.save(state, 'C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_secondLayer_0.pth')

class AlexNet2(nn.Module):
    def __init__(self):
        super(AlexNet2,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=72, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(72*5*5,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096,10)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,72*5*5)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x

model = AlexNet2()
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch.pth')

variables = checkpoint['net']
tmp = variables.copy()
tmp['conv5.weight'] = torch.cat((tmp['conv5.weight'][172: 188], tmp['conv5.weight'][200:]), 0)
tmp['conv5.bias'] = torch.cat((tmp['conv5.bias'][172: 188], tmp['conv5.bias'][200:]), 0)
tmp['dense1.weight'] = torch.cat((tmp['dense1.weight'][:, 172*25: 25 * 188], tmp['dense1.weight'][:, 25 * 200:]), 1)
model.load_state_dict(tmp)
state = {'net':model.state_dict()}
torch.save(state, 'C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_secondLayer_1.pth')

class AlexNet3(nn.Module):
    def __init__(self):
        super(AlexNet3,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=72, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(72*5*5,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096,10)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,72*5*5)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x

model = AlexNet3()
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch.pth')

variables = checkpoint['net']
tmp = variables.copy()
tmp['conv5.weight'] = torch.cat((torch.cat((tmp['conv5.weight'][164: 192], tmp['conv5.weight'][204:216]), 0), tmp['conv5.weight'][224:]), 0)
tmp['conv5.bias'] = torch.cat((torch.cat((tmp['conv5.bias'][164: 192], tmp['conv5.bias'][204:216]), 0), tmp['conv5.bias'][224:]), 0)
tmp['dense1.weight'] = torch.cat((torch.cat((tmp['dense1.weight'][:, 164 * 25: 25 * 192], tmp['dense1.weight'][:, 25 * 204: 25 * 216]), 1), tmp['dense1.weight'][:, 25 * 224:]), 1)
model.load_state_dict(tmp)
state = {'net':model.state_dict()}
torch.save(state, 'C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_secondLayer_2.pth')