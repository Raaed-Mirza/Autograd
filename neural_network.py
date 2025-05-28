import random
from engine import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # initialize weights for each input
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        #(w * x) + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # zip the weights and inputs together and compute the activation
        print("activation before tanh:", act.data)
        out = act.tanh() # apply the tanh activation function
        return out

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        #(w * x) + b
        out = [neuron(x) for neuron in self.neurons] # call each neuron in the layer with the input x and collect the outputs
        return out if len(out) > 1 else out[0] # return a list of outputs if there are multiple neurons, otherwise return the single output value

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)] # create a list of layers with the specified sizes

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x # return the output of the last layer, which is a single value if there's only one output, or a list of outputs otherwise