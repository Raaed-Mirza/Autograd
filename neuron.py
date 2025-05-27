import random
from engine import Value
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # initialize weights for each input
        self.b = Value(random.uniform(-1,1))