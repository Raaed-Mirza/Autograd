# 🧠 TinyFlow - A Minimal Autograd Engine

TinyFlow is a simple, educational autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd).  
It builds a computation graph from basic operations (`+`, `*`, `tanh`, etc.) and performs backpropagation to compute gradients — just like PyTorch under the hood.

> 📌 Built from scratch in pure Python to help you understand how autograd works!

---

## 🚀 Features

- Scalar-based automatic differentiation  
- Forward and backward pass using computation graph  
- Custom `Value` class with:
  - Operator overloading (`+`, `-`, `*`, `/`, `**`)
  - `tanh()` activation
- Topological sorting for proper gradient flow  
- Easily extensible with new ops like `ReLU`, `sigmoid`, etc.

---
## 📁 Project Structure
```
.
├── engine.py         # Core autograd engine with the Value class
├── test.py           # Demo script that simulates a simple neuron
├── README.md         # Project documentation
```
---

## 📦 Installation
```bash
git clone https://github.com/yourusername/micrograd-python.git
```
```bash
cd micrograd-python
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
