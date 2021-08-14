import numpy as np

x = np.array([[1.,2.,3.]])
y = np.array([[3.,6.]])
# y = np.array([[2,4,6]])

epochs = 20
lr = 0.01
# w1 = np.zeros((3,3))
# w2 = np.zeros((3,2))
w1 = np.random.randn(3,3)
b1 = np.random.randn(1,3)
w2 = np.random.randn(3,2)
b2 = np.random.randn(1,2)

for epoch in range(epochs):
    # import ipdb;ipdb.set_trace()
    y_1 = np.matmul(x,w1)+b1
    y_pred = np.matmul(y_1,w2)+b2
    # loss = y_pred - y
    # dw = -2*(y_pred - y)*x
    dw2 = (y_1.T).dot(-2*(y-y_pred))
    db2 = -2*(y-y_pred)
    dw1 = (x.T).dot(-2*(y-y_pred).dot(w2.T))
    db1 = -2*(y-y_pred).dot(w2.T)
    w1 = w1 - lr*dw1
    w2 = w2 - lr*dw2
    b2 = b2 - lr*db2
    b1 = b1 - lr*db1
    print(f'current w1:{w1}, \nw2: {w2}, \n b1:{b1}, \nb2:{b2},\ny_pred: {y_pred}\n')
