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

subtest0 = []
batchSize=16
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

    

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(256*5*5,1024)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(1024,3)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,256*5*5)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x
    
net=AlexNet().cuda()

subtest0 = []
subtest1 = []
subtest2 = []

checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_firstLayer.pth')
net.load_state_dict(checkpoint['net'])

net.eval()
for data in testloader:
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(len(predicted)):
        if int(predicted[i]) == 0:
            subtest0.append((images[i], labels[i]))
        elif int(predicted[i]) == 1:
            subtest1.append((images[i], labels[i]))
        else:
            subtest2.append((images[i], labels[i]))

print("First layer finished.")

count = 0
total = 0

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
a = 0
b = 0
r = 0
#print(len(subtest0))

while a < len(subtest0):
    images = subtest0[a][0]
    images = images.cpu()
    images = images.numpy()
    images = images[np.newaxis,:,:,:]
    images = torch.from_numpy(images)
    labels = subtest0[a][1]
    labels = labels.cpu()
    labels = labels.numpy()
    labels = labels[np.newaxis]
    labels = torch.from_numpy(labels)
    a = a + 1
    b = 1
    while a < len(subtest0) and b < 16:
        tmpImage = subtest0[a][0]
        tmpImage = tmpImage.cpu()
        tmpImage = tmpImage.numpy()
        tmpImage = tmpImage[np.newaxis,:,:,:]
        tmpImage = torch.from_numpy(tmpImage)
        images = torch.cat((images[:], tmpImage[:]), 0)
        tmpLabel = subtest0[a][1]
        tmpLabel = tmpLabel.cpu()
        tmpLabel = tmpLabel.numpy()
        tmpLabel = tmpLabel[np.newaxis]
        tmpLabel = torch.from_numpy(tmpLabel)
        labels = torch.cat((labels[:], tmpLabel[:]), 0)
        a = a + 1
        b = b + 1
    tmp = b
    while b < 16:
        images = torch.cat((images, images[-1:, :,:,:]), 0)
        labels = torch.cat((labels, labels[-1:]), 0)
        b = b + 1
    images=images.cuda()
    labels=labels.cuda()
    
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(tmp):
        total = total + 1
        if predicted[i] == labels[i]:
            r = r + 1
            count = count + 1
print("Second Layer 0 finished.")
print("Accuracy: " + str(r) + "/" + str(len(subtest0)) + ", " + str(float(r) / len(subtest0)))

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
a = 0
b = 0
r = 0
#print(len(subtest1))

while a < len(subtest1):
    images = subtest1[a][0]
    images = images.cpu()
    images = images.numpy()
    images = images[np.newaxis,:,:,:]
    images = torch.from_numpy(images)
    labels = subtest1[a][1]
    labels = labels.cpu()
    labels = labels.numpy()
    labels = labels[np.newaxis]
    labels = torch.from_numpy(labels)
    a = a + 1
    b = 1
    while a < len(subtest1) and b < 16:
        tmpImage = subtest1[a][0]
        tmpImage = tmpImage.cpu()
        tmpImage = tmpImage.numpy()
        tmpImage = tmpImage[np.newaxis,:,:,:]
        tmpImage = torch.from_numpy(tmpImage)
        images = torch.cat((images[:], tmpImage[:]), 0)
        tmpLabel = subtest1[a][1]
        tmpLabel = tmpLabel.cpu()
        tmpLabel = tmpLabel.numpy()
        tmpLabel = tmpLabel[np.newaxis]
        tmpLabel = torch.from_numpy(tmpLabel)
        labels = torch.cat((labels[:], tmpLabel[:]), 0)
        a = a + 1
        b = b + 1
    tmp = b
    while b < 16:
        images = torch.cat((images, images[-1:, :,:,:]), 0)
        labels = torch.cat((labels, labels[-1:]), 0)
        b = b + 1
    images=images.cuda()
    labels=labels.cuda()
    
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(tmp):
        total = total + 1
        if predicted[i] == labels[i]:
            count = count + 1
            r = r + 1
print("Second Layer 1 finished.")
print("Accuracy: " + str(r) + "/" + str(len(subtest1)) + ", " + str(float(r) / len(subtest1)))

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
a = 0
b = 0
r = 0
#print(len(subtest2))

while a < len(subtest2):
    images = subtest2[a][0]
    images = images.cpu()
    images = images.numpy()
    images = images[np.newaxis,:,:,:]
    images = torch.from_numpy(images)
    labels = subtest2[a][1]
    labels = labels.cpu()
    labels = labels.numpy()
    labels = labels[np.newaxis]
    labels = torch.from_numpy(labels)
    a = a + 1
    b = 1
    while a < len(subtest2) and b < 16:
        tmpImage = subtest2[a][0]
        tmpImage = tmpImage.cpu()
        tmpImage = tmpImage.numpy()
        tmpImage = tmpImage[np.newaxis,:,:,:]
        tmpImage = torch.from_numpy(tmpImage)
        images = torch.cat((images[:], tmpImage[:]), 0)
        tmpLabel = subtest2[a][1]
        tmpLabel = tmpLabel.cpu()
        tmpLabel = tmpLabel.numpy()
        tmpLabel = tmpLabel[np.newaxis]
        tmpLabel = torch.from_numpy(tmpLabel)
        labels = torch.cat((labels[:], tmpLabel[:]), 0)
        a = a + 1
        b = b + 1
    tmp = b
    while b < 16:
        images = torch.cat((images, images[-1:, :,:,:]), 0)
        labels = torch.cat((labels, labels[-1:]), 0)
        b = b + 1
    images=images.cuda()
    labels=labels.cuda()
    
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs,1)
    for i in range(tmp):
        total = total + 1
        if predicted[i] == labels[i]:
            count = count + 1
            r = r + 1
print("Second Layer 2 finished.")
print("Accuracy: " + str(r) + "/" + str(len(subtest2)) + ", " + str(float(r) / len(subtest2)))
print("Result:" + str(count) + "/" + str(total))