class Value:

    def __init__(self, data, _children=(), _op=''): #_childern is a tuple used to store the inputs (or parents) of the current node
        self.data = data
        self.grad = 0.0 #grad is used to store the gradient of the current node
        self.prev = set(_children) #prev is a set used to store the inputs (or parents) of the current node
        self._op = _op #op is used to store the operation that produced this node

    def __repr__(self):
        return f"Value({self.data})"