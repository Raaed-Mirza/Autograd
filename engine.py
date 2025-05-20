import math
class Value:

    def __init__(self, data, _children=(), _op=''): #_childern is a tuple used to store the inputs (or parents) of the current node
        self.data = data
        self.grad = 0.0 #grad is used to store the gradient of the current node
        self._backward = lambda: None
        self.prev = set(_children) #prev is a set used to store the inputs (or parents) of the current node
        self._op = _op #op is used to store the operation that produced this node

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        out =  Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out


    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-')
        return out

    def __truediv__(self, other):   
        out = Value(self.data / other.data, (self, other), '/')
        return out

    def __pow__(self, other):
        out = Value(self.data ** other.data, (self, other), '^')
        return out

    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += 1.0 * (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

