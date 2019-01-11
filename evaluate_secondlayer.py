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

batchSize=16
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

t = 0
c = 0

class AlexNet0(nn.Module):
    def __init__(self):
        super(AlexNet0,self).__init__()
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

net=AlexNet0().cuda()
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_secondLayer_0.pth')
net.load_state_dict(checkpoint['net'])

net.eval()
correct=0
total=0
for data in testloader:
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(len(predicted)):
            total = total + 1
            t = t + 1
            if labels[i] == predicted[i]:
                correct = correct + 1
                c = c + 1
print(str(correct) + "/" + str(total))


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
    
net=AlexNet2().cuda()
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_secondLayer_1.pth')
net.load_state_dict(checkpoint['net'])

net.eval()
correct=0
total=0
for data in testloader:
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(len(predicted)):
            total = total + 1
            t = t + 1
            if labels[i] == predicted[i]:
                correct = correct + 1
                c = c + 1
print(str(correct) + "/" + str(total))

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

net=AlexNet3().cuda()
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_secondLayer_2.pth')
net.load_state_dict(checkpoint['net'])

net.eval()
correct=0
total=0
for data in testloader:
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(len(predicted)):
            total = total + 1
            t = t + 1
            if labels[i] == predicted[i]:
                correct = correct + 1
                c = c + 1
print(str(correct) + "/" + str(total))
print(str(c) + "/" + str(t))