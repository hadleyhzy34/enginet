import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w,x)
b = torch.add(w,1)
y = torch.mul(a,b)

import ipdb;ipdb.set_trace()

y.backward()
print(w.grad)
