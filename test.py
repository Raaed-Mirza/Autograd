import numpy as np
import matplotlib.pyplot as plt
from operations import Operations as Value
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
    b = Value(6.7)
    x1w1 = x1 * w1
    x2w2 = x2 * w2

    x1w1x2w2 = x1w1 + x2w2
    # x1*w1 + x2*w2 + b
    n = x1w1x2w2 + b

    o = n.tanh()

if __name__ == "__main__":
    print(testing())

# plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2))) , plt.grid();
# plt.show()