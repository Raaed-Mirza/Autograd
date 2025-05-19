class Value:

    def __init__(self, data, _children=()): #_childern is a tuple
        self.data = data
        self.prev = set(_children)

    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out
    
a = Value(2.0)
# print(a)
b = Value(-3.0)
c = Value(10)

print((a.__mul__(b)).__add__(c))
print((a * b) + c)