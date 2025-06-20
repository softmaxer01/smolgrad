![Andrej Karpathy](assets/Pasted%20image.png)

# SmolGrad(Inspired by sir Karpathy)

A tiny scalar autograd engine.

This is a small autograd engine inspired by micrograd. It allows you to define mathematical expressions and automatically compute gradients of any node with respect to any of its children.

## Implemented Features

Currently, the `Value` object supports the following operations:

- Addition (`+`)
- Multiplication (`*`)
- Subtraction (`-`)
- Division (`/`)
- Power (`**`)
- `tanh` activation function
- `exp` activation function

## How it works

The engine builds a computation graph as you define your expressions. You can then perform a backward pass to compute the gradients.

**Note:** The backward pass is not fully automated yet. You need to manually build a topological sort of the graph and then call the `_backward()` function on each node in reverse order. See `test.py` for an example.

## Usage

Here is a simple example of how to use the engine:

```python
import engine as e

# Create some values
x1 = e.Value(2.0, _label='x1')
w1 = e.Value(-3.0, _label='w1')
b = e.Value(6.8813735870, _label='b')

# Define an expression
n = x1*w1 + b
o = n.tanh()

# Manually perform backpropagation
# (requires topological sort of the graph)
# ... see test.py ...
```

## Say hi to our mascot!

Please save the image of the panda you provided as `panda.jpg` inside the `assets` directory.

![A cute panda](assets/panda.jpg) 
