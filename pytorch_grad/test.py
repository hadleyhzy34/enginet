import numpy as np

x = np.array([[1,2,3]])
y = np.array([[2,4,6]])

epochs = 20
lr = 0.01
# w = np.zeros((3,3))
w = np.array([[0.68037,-0.21123,0.5661985],[0.5968801,0.8232948,-0.6048973],[-0.3295545,0.5364592,-0.4444506]]);
print(w)

for epoch in range(epochs):
    print(f'-------------episode: {epoch}--------------------')
    # import ipdb;ipdb.set_trace()
    y_pred = np.matmul(x,w)
    # loss = y_pred - y
    # dw = -2*(y_pred - y)*x
    # dw = (x.T).dot(-2*(y-y_pred))
    dw = (x.T).dot(-2*y+2*y_pred)
    # dw = (x.T).dot(-y+y_pred)
    w = w - lr*dw
    print(f'current w is: {w}\n')
    print(f'current prediction is: {y_pred}\n')
