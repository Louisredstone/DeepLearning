import torch

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

criterion = torch.nn.BCELoss()
model = LogisticRegressionModel()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w = ",model.linear.weight.item())
print("b = ",model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print("y_test = ",y_test.data)






import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,200)
x_t = torch.Tensor(x).view((200,1))# view：维度转换
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()