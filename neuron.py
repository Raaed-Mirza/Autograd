import random
from engine import Value
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # initialize weights for each input
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        #(w * x) + b
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b # zip the weights and inputs together and compute the activation
        out = act.tanh() # apply the tanh activation function
        return out

    