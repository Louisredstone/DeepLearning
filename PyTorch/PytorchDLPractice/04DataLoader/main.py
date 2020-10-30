################################
#       prepare data
################################

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath='diabetes.csv.gz'):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,-1])
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2)

################################
#       define model
################################
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.Sigmoid()
    def forward(self,x):
        result=self.activate(
            self.linear3(
                self.activate(
                    self.linear2(
                        self.activate(
                            self.linear1(
                                x
                            )
                        )
                    )
                )
            )
        )
        return result


if __name__ == '__main__':
    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            # Prepare Data
            inputs,labels = data
            # Forward
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,i,loss.item())
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()
    print("w1 = ",model.linear1.weight.detach().numpy())
    print("b1 = ",model.linear1.bias.detach().numpy())
    print("w2 = ",model.linear2.weight.detach().numpy())
    print("b2 = ",model.linear2.bias.detach().numpy())
    print("w3 = ",model.linear3.weight.detach().numpy())
    print("b3 = ",model.linear3.bias.detach().numpy())