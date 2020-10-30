import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_dataset = datasets.MNIST(
    root='./MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = datasets.MNIST(
    root='./MNIST',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False
)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(784,256)
        self.linear2 = torch.nn.Linear(256,32)
        self.linear3 = torch.nn.Linear(32,10)
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

def toOnehot(t):
    return torch.nn.functional.one_hot(t.to(torch.int64),10)

if __name__ == '__main__':
    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    for epoch in range(1):
        for batch_idx,(inputs,labels) in enumerate(train_loader,0):
            # prepare data
            labels=toOnehot(labels).to(torch.float)
            inputs=inputs.view(-1,784)
            # forward
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            print(epoch,batch_idx,loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()