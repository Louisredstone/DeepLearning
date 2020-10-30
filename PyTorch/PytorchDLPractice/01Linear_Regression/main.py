import torch
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

opdict={
    'SGD':torch.optim.SGD(model.parameters(),lr=0.01),
    'Adagrad':None,
    'Adam':None,
    'Adamax':None,
    'ASGD':None,
    'LBFGS':None,
    'RMSprop':None,
    'Rprop':None
}

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

for opname,optimizer in opdict.items():
    if optimizer == None:
        continue
    model = LinearModel()
    criterion = torch.nn.MSELoss()
    for epoch in range(1000):
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