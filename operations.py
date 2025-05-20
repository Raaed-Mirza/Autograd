from engine import Value
class Operations(Value):

    def __add__(self, other):
        return Operations(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Operations(self.data * other.data, (self, other), '*')

    def __sub__(self, other):
        return Operations(self.data - other.data, (self, other), '-')   

    def __truediv__(self, other):   
        return Operations(self.data / other.data, (self, other), '/')

    def __pow__(self, other):
        return Operations(self.data ** other.data, (self, other), '^')

    def __neg__(self):
        return Operations(-self.data, (self,), 'neg')
        