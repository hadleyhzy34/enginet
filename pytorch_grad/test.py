import numpy as np

x = np.array([[1,2,3]])
y = np.array([[2,4,6]])

epochs = 20
lr = 0.01
w = np.zeros((3,3))

for epoch in range(epochs):
    # import ipdb;ipdb.set_trace()
    y_pred = np.matmul(x,w)
    # loss = y_pred - y
    # dw = -2*(y_pred - y)*x
    # dw = (x.T).dot(-2*(y-y_pred))
    dw = (x.T).dot(-2*y+2*y_pred)
    # dw = (x.T).dot(-y+y_pred)
    w = w - lr*dw
    print(w, y_pred)
