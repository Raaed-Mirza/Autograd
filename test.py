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
    b = Value(-3.0)
    c = Value(10)
    c.data += h 
    d = (a * b) + c
    #d.data += h
    f = Value(-2.0)
    L = d * f
    L2 = L.data

    return (L2 - L1) / h

#print((a.__mul__(b)).__add__(c))
# print("d =", d)
# print("d.prev =", d.prev, "d._op = ", d._op)
# print("L =", L) 

if __name__ == "__main__":
    print("Finite difference result:", testing())