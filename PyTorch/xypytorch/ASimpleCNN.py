import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
# import的问题

# Prepare Dataset
def PrepareDataset():
    batch_size=64
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
        ])

    train_dataset = datasets.MNIST(root='../dataset/mnist',train=True,transform=transform,download=True)
    test_dataset = datasets.MNIST(root='../dataset/mnist',train=False,transform=transform,download=True)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    # print(train_dataset)
    # print(test_dataset)
    return (train_loader,test_loader)
    # number of datapoints

# Design Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()  # 784??batch_size
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)  # 24*24
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)  # 8*8
        self.pooling = torch.nn.MaxPool2d(2)  # kernel_size=2 ??  10*10       
        self.linear1 = torch.nn.Linear(320,10)  # C*W*H=20*4*4=320

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))  # 12*12
        x = F.relu(self.pooling(self.conv2(x)))  # 4*4 
        x = x.view(batch_size,-1)  # flatten
        x = self.linear1(x)
        return x

model1 = Net()

# construct loss and optim
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(),lr = 0.01,momentum = 0.5)

# train cycle
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,1):
        inputs,target = data
        # print(batch_idx)
        # print(data)
        # break
        optimizer.zero_grad()

        # forward + backward+ update
        outputs = model1(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            #print('{0}'.format(running_loss/300))
            running_loss = 0.0
            

# test
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs,target = data
            outputs = model1(inputs)
            _,predicted = torch.max(outputs.data,dim=1)
            print(_)
            print(predicted)
            total += target.size(0)
            correct += (predicted == target).sum().item()  # ???
    # print('Accuracy on test set is :{0}'.format(100*correct/total))




if __name__ == '__main__':
    train_loader,test_loader = PrepareDataset()  #train_loader,test_loader
    for epoch in range(1):
        train(epoch)
        test()

# predicted _ data  ???
# 图片存储？  tensor??
