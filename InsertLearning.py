""" 1. import pacakge"""
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from data_process_YW import load_data
""" 2. set up hyper-parameters"""
Batch_size = 1
numWorkers = 0

""" 3. set up device"""
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

""" 4. load data"""
trainloader, testloader = load_data(Batch_size, numWorkers)
# training set 1000, test set 500
# shape of single image: torch.Size([1, 3, 256, 256])
# shape of single label: torch.Size([1])

""" 5. define model, loss function, optimizer"""
"""5.1 agent model"""
import torch.nn as nn
import torch.nn.functional as F

class LocalNet(nn.Module):
    def __init__(self):
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3 * 32 * 32, 120)
        self.fc2 = nn.Linear(3 * 256 * 256, 1)

    def forward(self, x):
        x = x.view(-1, 3 * 256 * 256)
        #print(x.shape)
        x = self.fc2(x)
        return x
# mobile_net = LocalNet()
mobile_net = LocalNet().to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.SGD(mobile_net.parameters(), lr=0.001, momentum=0.9)
"""5.1.2 train agent model"""
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # print(inputs.shape)
        # print(labels)

        # labels=labels*2-1
        labels = torch.reshape(labels, (Batch_size, 1))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mobile_net(inputs)
        # outputs = (outputs+1)/2

        # print(labels)
        # print(outputs)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
PATH = './localnet.pth'
torch.save(mobile_net.state_dict(), PATH)
"""5.1.3 test agent model"""
mobile_net = LocalNet().to(device)
mobile_net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        #images, labels = data
        images, labels = data[0].to(device),data[1].to(device)
        outputs = mobile_net(images)
        #_, predicted = torch.max(outputs.data, 1)
        predicted = [1 if x > 0 else 0 for x in outputs]
        predicted=torch.tensor(predicted).to(device)
        #print(type(predicted))
        #print(type(labels))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 500 test images: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
correct_1=0
correct_0=0
with torch.no_grad():
    for data in testloader:
        #images, labels = data
        images, labels = data[0].to(device),data[1].to(device)
        outputs = mobile_net(images)
        #_, predicted = torch.max(outputs.data, 1)
        predicted = [1 if x > 0 else 0 for x in outputs]
        predicted=torch.tensor(predicted).to(device)
        #print(type(predicted))
        #print(type(labels))
        total += labels.size(0)
        correct_1 += (predicted == labels and labels==1).sum().item()
        correct_0+=(predicted == labels and labels==0).sum().item()
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 500 test images: %d %%' % (
    100 * correct / total))
print('Total sample size: %d' %total)
print('correct 1: %d'%correct_1)
print('correct 0: %d'%correct_0)

"""5.2 consultant model"""



