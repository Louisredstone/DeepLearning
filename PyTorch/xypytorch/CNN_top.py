import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
# import的问题

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
    return (train_loader,test_loader)
    # print(train_dataset)
    # print(test_dataset)
    # number of datapoints

class Net(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # 784??batch_size
        self.linear1=torch.nn.Linear(784,512)
    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

if __name__ == '__main__':
    PrepareDataset()
# PrepareDataset()