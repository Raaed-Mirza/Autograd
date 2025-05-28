import numpy as np
import matplotlib.pyplot as plt
from engine import Value
from neural_network import Neuron, Layer, MLP
import math
# f.grad = 4.0
# d.grad = -2.0
# L.grad = 1.0
    
@staticmethod
def testing():

    h = 0.001

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10)
    d = (a * b) + c
    f = Value(-2.0)
    L = d * f
    L1 = L.data

    a = Value(2.0)
    a.data += h
    b = Value(-3.0)
    b.data += h
    c = Value(10)
    c.data += h 
    d = (a * b) + c
    #d.data += h
    f = Value(-2.0)
    f.data += h
    L = d * f
    L2 = L.data

    # return (L2 - L1) / h
    return L.data

#print((a.__mul__(b)).__add__(c))
# print("d =", d)
# print("d.prev =", d.prev, "d._op = ", d._op)
# print("L =", L) 

def test_neuron():
    #inputs
    x1 = Value(2.0)
    x2 = Value(0.0)
    #weights
    w1 = Value(-3.0)
    w2 = Value(1.0)
    #bias
    b = Value(6.881)
    x1w1 = x1 * w1
    x2w2 = x2 * w2

    x1w1x2w2 = x1w1 + x2w2
    # x1*w1 + x2*w2 + b
    n = x1w1x2w2 + b

    e = (2*n).exp()

    o = (e - 1) / (e + 1)

    

    o.grad = 1.0
    o.backward()
    return {
        'o': o.data,
        'n': n.data,
        'x1.grad': x1.grad,
        'x2.grad': x2.grad,
        'w1.grad': w1.grad,
        'w2.grad': w2.grad,
        'b.grad': b.grad,
        'n.grad': n.grad,
        'o.grad': o.grad
    }
    #return 1-o.data**2, o.grad
    # n.grad = o.grad * (1-o.data**2)
    # return n.grad
    # return o


def testing_single_neuron():
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    n = MLP(3, [4, 4, 1])
    out = n(x)
    #return out
    xs = [[2.0, 3.0, -1.0],
          [3.0, -1.0, 0.5],
          [0.5, 1.0, 1.0],
          [1.0, 1.0, 1.0]]

    ys = [1.0, -1.0, -1.0, 1.0]
    ypred = [n([Value(i) for i in x]) for x in xs]
    #return ypred 

    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    loss.backward()

    print("x0 contributions to first neuron output:")
    for x in xs:
        print(f"{x[0]} * {n.layers[0].neurons[0].w[0].data}")

    for i, w in enumerate(n.layers[0].neurons[0].w):
        print(f"w[{i}] = {w.data}, grad = {w.grad}")

    return n.layers[0].neurons[0].w[0].grad

if __name__ == "__main__":
    #print(testing())
    print(testing_single_neuron())
    #print(test_neuron())
# plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2))) , plt.grid();
# plt.show()