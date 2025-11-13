# BayesNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

'bayesnn' provides a minimal set of PyTorch layers, functionals, and loss functions that make it easy to experiment with fully Bayesian neural networks. The package powers the demo notebooks/scripts in this repository and is now installable so you can reuse the same building blocks in your own projects.

## Features
- `BayesianLinear` layer with reparameterized weight sampling.
- Utility functionals for KL divergence and Monte Carlo log-likelihood estimates (Gaussian and categorical).
- Variational Bayes (`BNN_VBLoss`) and α-divergence (`BNN_AlphaDivergenceLoss`) objectives with MC sampling.

## Installation

```bash
pip install bayesnn
```

To install from source instead:

```bash
git clone https://github.com/PandaThr/BayesNN.git
cd BayesianNN
pip install .
```

For local development, use editable mode so changes are picked up immediately:

```bash
pip install -e .
```

## Quickstart

```python
import torch
from torch import nn

from bayesnn import BayesianLinear, BNN_VBLoss


class BayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim),
            nn.ReLU(),
            BayesianLinear(hidden_dim, output_dim, output_noise=True),
        )
        # Fixed observation noise variance 
        self.register_buffer("noise_var", torch.tensor(1e-2))

    def forward(self, x):
        return self.net(x)


model = BayesianMLP(input_dim=2, hidden_dim=64, output_dim=1)
criterion = BNN_VBLoss(model, N=1000, mc=50, beta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x_batch = torch.randn(128, 2)
y_batch = torch.sin(x_batch[:, :1])

loss = criterion(x_batch, y_batch)
loss.backward()
optimizer.step()

```

## Project structure
- `bayesnn/`: Source package exposed when you install the library.
