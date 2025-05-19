class Value:

    def __init__(self, data, _children=(), _op=''): #_childern is a tuple used to store the inputs (or parents) of the current node
        self.data = data
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
    
a = Value(2.0)
# print(a)
b = Value(-3.0)
c = Value(10)
d = (a * b) + c

#print((a.__mul__(b)).__add__(c))
print("d =", d)
print("d.prev =", d.prev, "d._op = ", d._op)