from engine import Value
import math
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

    def tanh(self):
        n = self.data
        return Operations((math.exp(2*n)-1)/(math.exp(2*n)+1), (self, ), 'tanh')
        