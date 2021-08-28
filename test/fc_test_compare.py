import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchviz import make_dot

class Model(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_out)
    
    def forward(self, x):
        y = self.linear1(x)
        y = self.linear2(y)
        return y

x = Variable(torch.randn(10))
model = Model(10,50,10)
y_label = x*2
# optimizer = torch.optim.BGD(model.parameters(), lr = 0.01)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

EPOCHS = 20
for i in range(EPOCHS):
    y_pred = model(x)
    loss = torch.sum((y_pred-y_label)**2)

    loss.backward()
    optimizer.step()  #perform weights/bias updates based on calculated gradients
    print(f'current status: {i}, {loss}')
    optimizer.zero_grad()

print(f'current prediction and label are: \n')
print(model(x))
print(y_label)


