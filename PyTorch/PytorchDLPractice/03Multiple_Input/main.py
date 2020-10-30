################################
#       prepare data
################################

import numpy as np
import torch

xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,-1])

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

model = Model()



criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=0.1)


for epoch in range(10000):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()

print("w1 = ",model.linear1.weight.numpy())
print("b1 = ",model.linear1.bias.numpy())
print("w2 = ",model.linear2.weight.numpy())
print("b2 = ",model.linear2.bias.numpy())
print("w3 = ",model.linear3.weight.numpy())
print("b3 = ",model.linear3.bias.numpy())