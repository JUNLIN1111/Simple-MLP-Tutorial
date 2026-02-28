# Simple MLP Actor – Neural Network from Scratch

A clean, easy-to-understand **Multi-Layer Perceptron (MLP)** implemented with pure NumPy — no PyTorch, TensorFlow, or autograd needed!

Perfect for:
- Students learning how neural networks work under the hood
- Reinforcement Learning beginners (this is an "actor" network)
- Anyone who wants to see backpropagation written manually

## Features

- 3-layer MLP (input → hidden → hidden → output)
- tanh activation everywhere → outputs in [-1, 1]
- Manual backpropagation + gradient descent
- Batch training support
- Very few dependencies (just NumPy)

## Installation

You only need Python and NumPy:

```bash
pip install numpy
# Optional (for plotting learning curves):
pip install matplotlib
