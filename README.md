![Andrej Karpathy](assets/Pasted%20image.png)

# SmolGrad(Inspired by sir Karpathy)

A tiny scalar autograd engine with neural network capabilities.

This is a small autograd engine inspired by micrograd. It allows you to define mathematical expressions and automatically compute gradients of any node with respect to any of its children. The engine now includes neural network components for building and training MLPs.

## Implemented Features

### Core Autograd Engine
The `Value` object supports the following operations:

- Addition (`+`)
- Multiplication (`*`)
- Subtraction (`-`)
- Division (`/`)
- Power (`**`)
- `tanh` activation function
- `exp` activation function
- **Automatic backpropagation** via `backward()` method

### Neural Network Components
- **Neuron**: Single neuron with weights, bias, and tanh activation
- **Layer**: Collection of neurons forming a layer
- **MLP**: Multi-layer perceptron with configurable architecture

## How it works

The engine builds a computation graph as you define your expressions. You can then perform automatic backward pass to compute the gradients using the `backward()` method.

### Basic Usage
```python
import engine as e

# Create some values
x1 = e.Value(2.0, _label='x1')
w1 = e.Value(-3.0, _label='w1')
b = e.Value(6.8813735870, _label='b')

n = x1*w1 + b
o = n.tanh()

o.backward()
print(f"Gradient of x1: {x1._grad}")
```

### Neural Network Training
```python
from engine import MLP

n = MLP(3, [16, 8, 4, 1])

# Training loop
for epoch in range(500):
    # Forward pass
    ypred = [n(x) for x in x_train]
    loss = sum((yout - yt)**2 for yt, yout in zip(y_train, ypred))
    
    # Backward pass
    for p in n.params():
        p._grad = 0.0
    loss.backward()
    
    # Update parameters
    for p in n.params():
        p._data += -0.01 * p._grad
```

## Example

See `test.py` for a complete example of training an MLP on a simple dataset with loss visualization.
![A cute panda](assets/panda.jpg) 
