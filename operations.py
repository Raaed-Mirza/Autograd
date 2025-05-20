from engine import Value
class Operations(Value):

    def __add__(self, other):
        return Operations(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Operations(self.data * other.data, (self, other), '*')