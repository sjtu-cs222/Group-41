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
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=252, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(252*5*5,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096,4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096,10)

    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x=x.view(-1,252*5*5)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x


checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch.pth')

variables = checkpoint['net']
'''
for k, l in variables.items():
    print(k)
print(variables['conv5.weight'].shape)
variables['conv5.weight'] = variables['conv5.weight'][128:]
print(variables['conv5.bias'].shape)
variables['conv5.bias'] = variables['conv5.bias'][128:]
#print(test.shape)
print(variables['dense1.weight'].shape)
variables['dense1.weight'] = variables['dense1.weight'][:,25* 128:]
'''

#print(model)

transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
batchSize=16
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

correct=[]
for i in range(10):
    t = []
    correct.append(t)

baseline=[0.9597, 0.9968, 0.9665, 0.9788, 0.9569, 0.9918, 0.9236, 0.9872, 0.9942, 0.9915]
for i in range(63,64):
    model=AlexNet().cuda()
    tmp = variables.copy()
    tmp['conv5.weight'] = torch.cat((tmp['conv5.weight'][0: 4 * i], tmp['conv5.weight'][4*i + 4:]), 0)
    tmp['conv5.bias'] = torch.cat((tmp['conv5.bias'][0: 4 * i], tmp['conv5.bias'][4*i + 4:]), 0)
    tmp['dense1.weight'] = torch.cat((tmp['dense1.weight'][:, 0: 25 * i], tmp['dense1.weight'][:, 25 * i + 100:]), 1)
    model.load_state_dict(tmp)
    model.eval()
    total = 0
    current = [0.0] * 10
    for data in testloader:
        images,labels=data
        images=images.cuda()
        labels=labels.cuda()
        outputs=model(Variable(images))
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        #correct+=(predicted==labels).sum()
        for m in range(10):
            for j in range(batchSize):
                if (predicted[j] == m and labels[j] == m) or (predicted[j] != m and labels[j] != m):
                    current[m] = current[m] + 1
        del images
        del labels
    for m in range(10):
        correct[m].append(current[m] / total * 100 - baseline[m] * 100)
        print('Accuracy of the network on %d in evaluation %d/63: %f %%.' % (m, i + 1, current[m] / total * 100))
    del model
correct = np.array(correct, ndmin = 2)
correct = np.transpose(correct)
pre_save = pd.DataFrame(columns = ['0', '1','2','3','4','5','6','7','8','9'], data = correct)
pre_save.to_csv("Evaluation.csv")