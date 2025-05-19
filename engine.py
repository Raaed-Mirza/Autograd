class Value:

    def __init__(self, data, _children=(), _op=''): #_childern is a tuple used to store the inputs (or parents) of the current node
        self.data = data
        self.grad = 0.0 #grad is used to store the gradient of the current node
        self.prev = set(_children) #prev is a set used to store the inputs (or parents) of the current node
        self._op = _op #op is used to store the operation that produced this node

    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    L.grad = 1.0
    
    @staticmethod
    def lol():

        h = 0.0001

        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10)
        d = (a * b) + c
        f = Value(-2.0)
        L = d * f
        L1 = L.data

        a = Value(2.0 + h)
        b = Value(-3.0)
        c = Value(10)
        d = (a * b) + c
        f = Value(-2.0)
        L = d * f
        L2 = L.data

        return (L2 - L1) / h

#print((a.__mul__(b)).__add__(c))
# print("d =", d)
# print("d.prev =", d.prev, "d._op = ", d._op)
# print("L =", L) 

print(Value.lol())