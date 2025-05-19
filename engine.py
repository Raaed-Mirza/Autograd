class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
    
a = Value(2.0)
# print(a)
b = Value(3.0)

print(a.__add__(b))
