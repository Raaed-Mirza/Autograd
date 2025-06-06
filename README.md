# 🧠 A Minimal Autograd Engine

TinyFlow is a simple, educational autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd).  
It builds a computation graph from basic operations (`+`, `*`, `tanh`, etc.) and performs backpropagation to compute gradients — just like PyTorch under the hood.

> 📌 Built from scratch in pure Python to help you understand how autograd works!

---

## 🚀 Features

- Scalar-based automatic differentiation  
- Forward and backward pass using computation graph  
- Custom `Value` class with:
  - Operator overloading (`+`, `-`, `*`, `/`, `**`)
  - `tanh()`, `sigmoid()`, `exp()` activations
- Topological sorting for proper gradient flow

---
## 📁 Project Structure
```
.
├── engine.py         # Core autograd engine with the Value class
├── neuron.py         # Neuron, Layer, and MLP implementations
├── test.py           # Demo script that simulates a simple neuron
├── README.md         # Project documentation
```
---

## 📦 Installation
```bash
git clone https://github.com/Raaed-Mirza/Autograd.git
```
```bash
cd Autograd
```
```bash
pip install -r requirements.txt
```

## 🧪 Example

```python
from engine import Value

# Inputs
x1 = Value(2.0)
x2 = Value(0.0)

# Weights and bias
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.881)

# Simple neuron: n = x1*w1 + x2*w2 + b
n = (x1 * w1) + (x2 * w2) + b
o = n.tanh()

# Backpropagation
o.backward()

# Gradients
print(x1.grad, w1.grad, b.grad)
```
## 🎓 Training Example

```python
from engine import Value
from neuron import MLP

# Sample inputs and target labels
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, 1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# Initialize a multi-layer perceptron
n = MLP(3, [4, 4, 1])

# Training loop
for epoch in range(1000):
    # Forward pass
    ypred = [n([Value(i) for i in x]) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Zero gradients
    for p in n.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Update weights
    for p in n.parameters():
        p.data += -0.01 * p.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

```

### Running the demo
```bash
python test.py
```
