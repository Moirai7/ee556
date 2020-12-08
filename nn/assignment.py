import numpy as np
hidden_dim = 5
input_dim = 2
output_dim = 1
w1 = np.array(np.random.rand(input_dim,hidden_dim) * .01)
c1 = np.zeros((1, hidden_dim))

w2 = np.array(np.random.rand(hidden_dim, output_dim) * .01)
c2 = np.zeros((1, output_dim))

x = np.array([[0,0],[1,0],[0,1],[1,1]])
#y = np.array([[1, 0],[0, 1],[0, 1],[1, 0]])
y = np.array([[0],[1],[1],[0]])
batch = 1000
idx = np.random.choice(np.arange(len(x)), batch, replace=True)
x = x[idx]
y = y[idx]
print(y.shape)
epoch = 1000
lr = 0.01
def forward(x):
    z1 = np.dot(x, w1) + c1
    a1 = 1 / (1 + np.exp(-z1)) #sigmoid
    #a1 = np.maximum(z1, 0)#relu
    z2 = np.dot(a1, w2) + c2
    #a2 = np.maximum(z2, 0)#relu
    a2 = 1 / (1 + np.exp(-z2)) #sigmoid
    #a2 = np.exp(z2)/np.exp(z2).sum(axis=1, keepdims=True) #softmax
    return z1, a1, z2, a2

def backward(z1, a1, z2, a2):
    global w2, w1, c1, c2
    da2 = 2. * (a2 - y)
    dz2 = da2 * a2 * (1. - a2)
    dw2 = np.dot(a1.T, dz2)
    dc2 = np.sum(dz2, axis = 0, keepdims = True)

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * a1 * (1. - a1)
    dw1 = np.dot(x.T, dz1)
    dc1 = np.sum(dz1, axis = 0, keepdims = True)
    w1 = w1 - lr * dw1
    c1 = c1 - lr * dc1
    w2 = w2 - lr * dw2
    c2 = c2 - lr * dc2

costs = []
def cost(a):
    delta = 1e-7
    cost = np.square(a-y).sum()
    costs.append(cost)
    print(cost, a[:10], y[:10])
    
def train():
    for i in range(epoch):
        z1, a1, z2, a2 = forward(x)
        cost(a2)
        backward(z1, a1, z2, a2)


train()
import matplotlib.pyplot as plt 
def plot():
    plt.plot(costs)
    plt.show() 

plot()
