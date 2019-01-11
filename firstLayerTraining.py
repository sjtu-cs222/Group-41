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

#
batchSize=16

##load data
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

####network
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
print (net)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

converting_dict = {1:2, 
                   2:1, 
                   3:0, 
                   4:1, 
                   5:2, 
                   6:2, 
                   7:0, 
                   8:1, 
                   9:0, 
                   0:0}
#train

print ("training begin")
for epoch in range(3):
    start = time.time()
    running_loss=0
    for i,data in enumerate(trainloader,0):
        # print (inputs,labels)
        image,label=data
        for k in range(len(label)):
            label[k] = converting_dict[int(label[k])]
        image = image.cuda()
        label = label.cuda()
        image=Variable(image)
        label=Variable(label)

        # imshow(torchvision.utils.make_grid(image))
        # plt.show()
        # print (label)
        optimizer.zero_grad()

        # print (image.shape)
        outputs=net(image)
        # print (outputs)
        loss=criterion(outputs,label)

        loss.backward()
        optimizer.step()

        running_loss+=loss.data

        if i%100==99:
            end=time.time()
            print ('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s'%(epoch+1,(i+1)*16,running_loss/100,(end-start)))
            start=time.time()
            running_loss=0
print ("finish training")

state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
torch.save(state, 'C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_firstLayer.pth')

#test
checkpoint = torch.load('C:\\Users\\user\\Desktop\\Algorithm Project\\model\\alexNet_3epoch_firstLayer.pth')
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
    total+=labels.size(0)
    for k in range(len(labels)):
        labels[k] = converting_dict[int(labels[k])]
    correct+=(predicted==labels).sum()
print('Accuracy of the network on the %d test images: %f %%' % (total , 100.0 * correct / total))
'''